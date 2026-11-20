"""Physical kernel -> flow matching -> 10-day ensemble and its entropy (docs/design_ko.md).

    # 0. ERA5 5.625 deg daily means in the KAI layout (once)
    python third_party/pgfm/scripts/wb2_to_kai.py --out data/era5_5p625 --daily-mean --no-climode --years 2014 2015 2016 2017
    # 1. the tendency core with the physical terms G(x) (PG-CFM's recipe, vendored script)
    python scripts/run_fpk.py core --iters 3000
    # 2. the physical kernel K(x, y) on the frozen core
    python scripts/run_fpk.py kernel --iters 1500
    # 3. flow matching from the kernel (and the Gaussian-source control)
    python scripts/run_fpk.py flow --source kernel --iters 3000
    python scripts/run_fpk.py flow --source gauss --iters 3000 --tag flow_gauss
    # 4. 10-day ensembles on the test year, scores and entropy
    python scripts/run_fpk.py score --arms core kernel flow flow_gauss
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fpk import data as D  # noqa: E402
from fpk.env import DYN, TISR, contract, pgfm_args, rkt  # noqa: E402
from fpk.ensemble import Scorer, crps_torch, rollout  # noqa: E402
from fpk.flow import FlowConfig, KernelFlow  # noqa: E402
from fpk.devflow import DevFlowConfig, DeviationFlow  # noqa: E402
from fpk.devkernel import DevConfig, EntropyDeviationKernel  # noqa: E402
from fpk.kernel import KernelConfig, PhysicsKernel  # noqa: E402
from pgcfm.kai_tend import core_step  # noqa: E402


def common(p):
    p.add_argument("--data-root", default="data/era5_5p625")
    p.add_argument("--out", default="runs/fpk")
    p.add_argument("--train-years", type=int, nargs="+", default=[2014, 2015])
    p.add_argument("--val-year", type=int, default=2016)
    p.add_argument("--test-year", type=int, default=2017)
    p.add_argument("--core-tag", default="core_gng")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=0)


def pargs(a, extra=None):
    return pgfm_args(a.data_root, a.out, a.train_years, a.val_year, a.test_year, a.device, a.core_tag, extra)


def load_core(a):
    pa = pargs(a)
    cl = D.climate(pa)
    core = rkt().build_core(pa, cl.mean, cl.std, cl.inc, cl.lat, tag=a.core_tag)
    if getattr(core, "entv", False) or (getattr(core, "hist", False) and not getattr(core, "mem", 0)):
        raise SystemExit("the kernel is built on a plain or a memory core (no history-only / entropy-state core)")
    if getattr(core, "mem", 0) and (getattr(core, "acc", ()) or getattr(core, "kern", 0) or getattr(core, "pacc", ())):
        raise SystemExit("memory core: the long-memory terms (acc / kern / pacc) are not supported by the kernel")
    if getattr(core, "stride", 1) != 1:
        raise SystemExit("the kernel pipeline is daily: a core with stride > 1 (multi-day step) is not supported")
    return pa, cl, core


def log(d):
    print(json.dumps(d), flush=True)


# ---------------------------------------------------------------------------------------------------------- core
def cmd_core(a, rest):
    """Train the tendency core with PG-CFM's own script: plain core, GroupNorm, physical terms G(x)."""
    argv = ["--arm", "core", "--tag", a.core_tag, "--core-norm", "gn", "--core-gfeat", "--log-every", "250",
            "--iters", str(a.iters), "--seed", str(a.seed)] + list(rest)
    pa = pargs(a, argv)
    os.makedirs(pa.out, exist_ok=True)
    R = rkt()
    R.train(pa, "core")


# -------------------------------------------------------------------------------------------------------- kernel
def cmd_kernel(a, rest):
    torch.manual_seed(a.seed)
    pa, cl, core = load_core(a)
    dev = torch.device(a.device)
    years = D.load(pa, a.train_years)
    va = D.load(pa, [a.val_year])
    W = int(getattr(core, "mem", 0))
    tr, vr = D.Pairs(years, a.seed, hist=W), D.Pairs(va, a.seed + 1, hist=W)
    w = D.area_weights(cl.lat).to(dev)
    K = PhysicsKernel(core, KernelConfig(n_centres=a.centres, hidden=a.hidden)).to(dev)
    opt = torch.optim.AdamW([p for p in K.parameters() if p.requires_grad], lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
    vx = [vr(a.batch) for _ in range(4)]
    dyn = torch.tensor(DYN)
    hdyn = lambda hi: None if hi is None else hi[:, :, dyn].to(dev)  # noqa: E731

    def val():
        K.eval()
        tot = 0.0
        with torch.no_grad():
            for x, y, doy, tn, hi in vx:
                K.start(x[:, dyn].to(dev), doy.to(dev), hdyn(hi))
                c, h, _, _ = K(x[:, dyn].to(dev), doy.to(dev), tn.to(dev))
                tot += float(K.nll(c, h, y[:, dyn].to(dev), w))
        K.train()
        return tot / len(vx)

    t0 = time.time()
    log(dict(kernel_val_nll_start=round(val(), 4)))
    for it in range(1, a.iters + 1):
        x, y, doy, tn, hi = tr(a.batch)
        K.start(x[:, dyn].to(dev), doy.to(dev), hdyn(hi))
        c, h, _, _ = K(x[:, dyn].to(dev), doy.to(dev), tn.to(dev))
        loss = K.nll(c, h, y[:, dyn].to(dev), w)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(K.parameters(), 1.0)
        opt.step()
        sch.step()
        if it % a.log_every == 0 or it == a.iters:
            e = dict(eps_m=round(float(K.eps_m().mean()), 4)) if K.mem else {}
            log(dict(it=it, nll=round(loss.item(), 4), val_nll=round(val(), 4), eps=round(float(K.eps().mean()), 4),
                     h=round(float(h.mean()), 4), **e, sec=round(time.time() - t0)))
    state = {k: v for k, v in K.state_dict().items() if not k.startswith("core.")}
    torch.save(dict(state=state, cfg=K.cfg.__dict__, core_tag=a.core_tag), os.path.join(a.out, f"{a.tag}.pt"))


def load_kernel(a, core, tag):
    ck = torch.load(os.path.join(a.out, f"{tag}.pt"), map_location="cpu", weights_only=False)
    if ck.get("core_tag") not in (None, a.core_tag):
        raise SystemExit(f"kernel {tag} was fitted on core {ck['core_tag']}, not --core-tag {a.core_tag}")
    cfg = ck["cfg"]
    cfg["blocks"] = tuple(cfg["blocks"])
    K = PhysicsKernel(core, KernelConfig(**cfg))
    miss = K.load_state_dict(ck["state"], strict=False)
    assert not miss.unexpected_keys and all(k.startswith("core.") for k in miss.missing_keys), miss
    return K.to(a.device).eval()


# ---------------------------------------------------------------------------------------------------------- flow
def cmd_flow(a, rest):
    torch.manual_seed(a.seed)
    pa, cl, core = load_core(a)
    dev = torch.device(a.device)
    K = load_kernel(a, core, a.kernel_tag)
    Fm = KernelFlow(K, FlowConfig(hidden=a.hidden, source=a.source, steps=a.steps)).to(dev)
    years = D.load(pa, a.train_years)
    va = D.load(pa, [a.val_year])
    W = int(getattr(core, "mem", 0))
    tr, vr = D.Pairs(years, a.seed, hist=W), D.Pairs(va, a.seed + 1, hist=W)
    w = D.area_weights(cl.lat).to(dev)
    params = [p for p in Fm.net.parameters()]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.iters, pct_start=0.05)
    vx = [vr(a.batch) for _ in range(4)]
    dyn = torch.tensor(DYN)
    hdyn = lambda hi: None if hi is None else hi[:, :, dyn].to(dev)  # noqa: E731
    gv = torch.Generator().manual_seed(123)

    def val():
        Fm.eval()
        gv.manual_seed(123)
        tot, err = 0.0, 0.0
        with torch.no_grad():
            for x, y, doy, tn, hi in vx:
                xd, yd = x[:, dyn].to(dev), y[:, dyn].to(dev)
                K.start(xd, doy.to(dev), hdyn(hi))
                tot += float(Fm.loss(xd, yd, doy.to(dev), tn.to(dev), w, gv))
        Fm.train()
        return tot / len(vx)

    t0 = time.time()
    log(dict(flow_val_start=round(val(), 4), source=a.source))
    for it in range(1, a.iters + 1):
        x, y, doy, tn, hi = tr(a.batch)
        K.start(x[:, dyn].to(dev), doy.to(dev), hdyn(hi))
        loss = Fm.loss(x[:, dyn].to(dev), y[:, dyn].to(dev), doy.to(dev), tn.to(dev), w)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sch.step()
        if it % a.log_every == 0 or it == a.iters:
            log(dict(it=it, loss=round(loss.item(), 4), val=round(val(), 4), sec=round(time.time() - t0)))
    torch.save(dict(state=Fm.net.state_dict(), cfg=Fm.cfg.__dict__, kernel_tag=a.kernel_tag),
               os.path.join(a.out, f"{a.tag}.pt"))


def load_flow(a, K, tag):
    ck = torch.load(os.path.join(a.out, f"{tag}.pt"), map_location="cpu", weights_only=False)
    if ck.get("kernel_tag") not in (None, a.kernel_tag):
        raise SystemExit(f"flow {tag} was trained on kernel {ck['kernel_tag']}, not --kernel-tag {a.kernel_tag}")
    cfg = ck["cfg"]
    cfg["blocks"] = tuple(cfg["blocks"])
    Fm = KernelFlow(K, FlowConfig(**cfg))
    Fm.net.load_state_dict(ck["state"])
    return Fm.to(a.device).eval()


# ---------------------------------------------------------------------------------------------------------- devk
def cmd_devk(a, rest):
    """The entropy deviation kernel (fpk.devkernel): the mean runs the core's own trajectory for D days, the
    deviations are carried by the equation, and the entropy residual r (and the size h of the new part) is fitted
    by the Gaussian log score of the data around the mean on every day (calibrated entropy: E[IGN] = H)."""
    torch.manual_seed(a.seed)
    pa, cl, core = load_core(a)
    dev = torch.device(a.device)
    Dk = EntropyDeviationKernel(core, DevConfig(pairs=a.pairs, hidden=a.hidden, eps0=a.eps0, smooth=a.smooth)).to(dev)
    W, Dmax = int(getattr(core, "mem", 0)), max(a.days)
    years = D.load(pa, a.train_years)
    va = D.load(pa, [a.val_year])
    tr, vr = D.Pairs(years, a.seed, hist=W, ahead=Dmax), D.Pairs(va, a.seed + 1, hist=W, ahead=Dmax)
    w = D.area_weights(cl.lat).to(dev)
    params = [p_ for p_ in Dk.parameters() if p_.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
    dyn = torch.tensor(DYN)
    vx = [vr(a.batch) for _ in range(a.val_batches)]

    def run(batch, days, gen=None):
        x, ys, doy, tns, hi = batch
        if ys.dim() == 4:
            ys, tns = ys[:, None], tns[:, None]
        xd, yd, doy, tns = x[:, dyn].to(dev), ys[:, :, dyn].to(dev), doy.to(dev), tns.to(dev)
        e = Dk.start(xd, doy, None if hi is None else hi[:, :, dyn].to(dev), gen)
        xb, ign, ent, prod = xd, [], [], []
        for k in range(days):
            xb, e, H, (st, r) = Dk.day(xb, e, doy + k, tns[:, k], k, gen)
            e = e.detach()
            i, h_ = Dk.ign(H, xb, yd[:, k], w)
            ign.append(i)
            ent.append(float(h_))
            m = Dk.keep * w
            prod.append((float((st * m).sum() / m.expand_as(st).sum()), float((r * m).sum() / m.expand_as(r).sum())))
        return ign, ent, prod

    def val():
        Dk.eval()
        gv = torch.Generator().manual_seed(123)
        with torch.no_grad():
            r = [run(b, Dmax, gv) for b in vx]
        Dk.train()
        ign = np.mean([[float(v) for v in q[0]] for q in r], 0)
        ent = np.mean([q[1] for q in r], 0)
        # calibrated: IGN - H = 1/2 (E[err^2] = variance); > 1/2 = too sure, < 1/2 = too wide
        return [round(float(v), 3) for v in ign], [round(float(v), 3) for v in ign - ent]

    t0 = time.time()
    i0, g0 = val()
    log(dict(devk_val_start=dict(ign=i0, ign_minus_H=g0)))
    for it in range(1, a.iters + 1):
        days = a.days[min(len(a.days) - 1, (it - 1) * len(a.days) // a.iters)]
        ign, ent, prod = run(tr(a.batch), days)
        loss = sum(ign) / days
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sch.step()
        if it % a.log_every == 0 or it == a.iters:
            vi, vg = val()
            log(dict(it=it, days=days, ign=round(float(loss), 4), val_ign=vi, val_ign_minus_H=vg,
                     stretch=[round(p_[0], 3) for p_ in prod], r=[round(p_[1], 3) for p_ in prod],
                     h=round(float(Dk.h().mean()), 4), sec=round(time.time() - t0)))
    state = {k: v for k, v in Dk.state_dict().items() if not k.startswith("core.")}
    torch.save(dict(state=state, cfg=Dk.cfg.__dict__, core_tag=a.core_tag), os.path.join(a.out, f"{a.tag}.pt"))


def load_devk(a, core, tag):
    ck = torch.load(os.path.join(a.out, f"{tag}.pt"), map_location="cpu", weights_only=False)
    if ck.get("core_tag") not in (None, a.core_tag):
        raise SystemExit(f"deviation kernel {tag} was fitted on core {ck['core_tag']}, not --core-tag {a.core_tag}")
    cfg = ck["cfg"]
    cfg["blocks"] = tuple(cfg["blocks"])
    Dk = EntropyDeviationKernel(core, DevConfig(**cfg))
    miss = Dk.load_state_dict(ck["state"], strict=False)
    assert not miss.unexpected_keys and all(k.startswith("core.") for k in miss.missing_keys), miss
    return Dk.to(a.device).eval()


# ------------------------------------------------------------------------------------------------------- devflow
def cmd_devflow(a, rest):
    """Flow matching from the entropy deviation kernel (fpk.devflow): the kernel runs D days from the data's state,
    and on every day its deviations are the source and the data's deviation from the mean the target."""
    torch.manual_seed(a.seed)
    pa, cl, core = load_core(a)
    dev = torch.device(a.device)
    Dk = load_devk(a, core, a.dev_tag)
    Fm = DeviationFlow(Dk, DevFlowConfig(hidden=a.hidden, source=a.source, steps=a.steps)).to(dev)
    W = int(getattr(core, "mem", 0))
    years = D.load(pa, a.train_years)
    va = D.load(pa, [a.val_year])
    tr, vr = D.Pairs(years, a.seed, hist=W, ahead=a.days), D.Pairs(va, a.seed + 1, hist=W, ahead=a.days)
    w = D.area_weights(cl.lat).to(dev)
    params = list(Fm.net.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.iters, pct_start=0.05)
    dyn = torch.tensor(DYN)
    vx = [vr(a.batch) for _ in range(a.val_batches)]

    def run(batch, gen=None):
        x, ys, doy, tns, hi = batch
        if ys.dim() == 4:
            ys, tns = ys[:, None], tns[:, None]
        xd, yd, doy, tns = x[:, dyn].to(dev), ys[:, :, dyn].to(dev), doy.to(dev), tns.to(dev)
        with torch.no_grad():
            e = Dk.start(xd, doy, None if hi is None else hi[:, :, dyn].to(dev), gen)
        xb, out = xd, []
        for k in range(a.days):
            with torch.no_grad():
                xb, e, H, _ = Dk.day(xb, e, doy + k, tns[:, k], k, gen)
            out.append(Fm.loss(e, xb, yd[:, k], Fm.cond(xb, H, doy + k + 1, k + 1), w, gen))
        return out

    def val():
        Fm.eval()
        gv = torch.Generator().manual_seed(123)
        with torch.no_grad():
            r = np.mean([[float(v) for v in run(b, gv)] for b in vx], 0)
        Fm.train()
        return [round(float(v), 4) for v in r]

    t0 = time.time()
    log(dict(devflow_val_start=val(), source=a.source))
    for it in range(1, a.iters + 1):
        loss = sum(run(tr(a.batch))) / a.days
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sch.step()
        if it % a.log_every == 0 or it == a.iters:
            log(dict(it=it, loss=round(loss.item(), 4), val=val(), sec=round(time.time() - t0)))
    torch.save(dict(state=Fm.net.state_dict(), cfg=Fm.cfg.__dict__, dev_tag=a.dev_tag, core_tag=a.core_tag),
               os.path.join(a.out, f"{a.tag}.pt"))


def load_devflow(a, core, tag, pairs=0):
    ck = torch.load(os.path.join(a.out, f"{tag}.pt"), map_location="cpu", weights_only=False)
    if ck.get("core_tag") not in (None, a.core_tag):
        raise SystemExit(f"deviation flow {tag} was trained on core {ck['core_tag']}, not --core-tag {a.core_tag}")
    Dk = load_devk(a, core, ck["dev_tag"])
    if pairs:
        Dk.cfg.pairs = pairs  # independent members of the forecast (the kernel's parameters do not depend on P)
    cfg = ck["cfg"]
    cfg["blocks"] = tuple(cfg["blocks"])
    Fm = DeviationFlow(Dk, DevFlowConfig(**cfg))
    Fm.net.load_state_dict(ck["state"])
    return Fm.to(a.device).eval()


# ---------------------------------------------------------------------------------------------------------- tune
def cmd_tune(a, rest):
    """Rollout fine-tuning: M members run D days from the data's state (each day drawing from the kernel, or from
    the flow with --flow-tag), and the fair CRPS of every day against the data (units of s, area-weighted) is
    minimized, plus lam x the one-day loss the model was fitted with (kernel NLL / flow-matching loss) as an anchor.
    The state is detached between days (the gradient of a day reaches its own draw only) unless --through: then the
    gradient of a day's score goes back through the states of the days before it. For the kernel that includes the
    equation's Jacobian (the members' error growth is seen by the earlier draws). For the flow the next day's source
    draw and conditioning are taken without gradient by default, so the state Jacobian is the identity (a day's
    correction is credited as if it were carried unchanged to the later days); --through-core keeps the equation's
    Jacobian there too (the memory of a backward pass through the core on every day). The days follow --days. A memory core's memory runs along each member as in the forecast."""
    torch.manual_seed(a.seed)
    pa, cl, core = load_core(a)
    dev = torch.device(a.device)
    K = load_kernel(a, core, a.kernel_tag)
    Fm = load_flow(a, K, a.flow_tag) if a.flow_tag else None
    if Fm is None:
        for n_, p_ in K.named_parameters():
            p_.requires_grad_(not n_.startswith("core."))
        params, model = [p_ for p_ in K.parameters() if p_.requires_grad], K
    else:
        params, model = list(Fm.net.parameters()), Fm
    model.train()
    W, Dmax, M = int(getattr(core, "mem", 0)), max(a.days), a.members
    years = D.load(pa, a.train_years)
    va = D.load(pa, [a.val_year])
    tr, vr = D.Pairs(years, a.seed, hist=W, ahead=Dmax), D.Pairs(va, a.seed + 1, hist=W, ahead=Dmax)
    w = D.area_weights(cl.lat).to(dev)
    mask = K.keep * w
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
    dyn = torch.tensor(DYN)
    vx = [vr(a.batch) for _ in range(a.val_batches)]
    g = torch.Generator().manual_seed(a.seed)

    def draw(xm, d, tn, obs, gen=None):
        if Fm is not None:
            return Fm.draw(xm, d, tn, gen, steps=a.steps, observe=obs, grad=a.through_core)[0]
        return K.core.finish(K.sample(xm, d, tn, gen, observe=obs)[0], tn)

    def anchor(xd, yd, doy, tn, gen=None):
        if Fm is not None:
            return Fm.loss(xd, yd, doy, tn, w, gen)
        c, h, _, _ = K(xd, doy, tn)
        return K.nll(c, h, yd, w)

    def run(batch, days, train, gen=None):
        x, ys, doy, tns, hi = batch
        xd, yd, doy, tns = x[:, dyn].to(dev), ys[:, :, dyn].to(dev), doy.to(dev), tns.to(dev)
        B = xd.shape[0]
        K.start(xd, doy, None if hi is None else hi[:, :, dyn].to(dev))
        an = anchor(xd, yd[:, 0], doy, tns[:, 0], gen)  # x enters the memory here, before it is copied to the members
        if train and a.lam > 0:
            (a.lam * an).backward()
        K.tile(M)
        xm, dm, out, tot = xd.repeat_interleave(M, 0), doy.repeat_interleave(M, 0), [], 0.0
        for k in range(days):
            tn = tns[:, k].repeat_interleave(M, 0)
            y = draw(xm, dm + k, tn, k > 0, gen)
            cr = crps_torch((y / K.s).view(B, M, *y.shape[1:]), yd[:, k] / K.s)
            cr = (cr * mask).sum() / mask.expand_as(cr).sum()
            if train and a.through:
                tot = tot + cr / days
            elif train:
                (cr / days).backward()
            out.append(float(cr))
            xm = y if (train and a.through) else y.detach()
        if train and a.through:
            tot.backward()
        return float(an), out

    def val():
        model.eval()
        gv = torch.Generator().manual_seed(123)
        with torch.no_grad():
            r = [run(b, Dmax, False, gv) for b in vx]
        model.train()
        return round(float(np.mean([q[0] for q in r])), 4), [round(float(v), 4) for v in np.mean([q[1] for q in r], 0)]

    t0 = time.time()
    an0, cr0 = val()
    log(dict(tune_val_start=dict(anchor=an0, crps=cr0), what="flow" if Fm is not None else "kernel"))
    for it in range(1, a.iters + 1):
        days = a.days[min(len(a.days) - 1, (it - 1) * len(a.days) // a.iters)]
        opt.zero_grad()
        an, cr = run(tr(a.batch), days, True, g)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sch.step()
        if it % a.log_every == 0 or it == a.iters:
            van, vcr = val()
            e = dict(eps=round(float(K.eps().mean()), 4)) if Fm is None else {}
            if Fm is None and K.mem:
                e["eps_m"] = round(float(K.eps_m().mean()), 4)
            log(dict(it=it, days=days, anchor=round(an, 4), crps=round(float(np.mean(cr)), 4), val_anchor=van,
                     val_crps=vcr, **e, sec=round(time.time() - t0)))
    if Fm is None:
        state = {k: v for k, v in K.state_dict().items() if not k.startswith("core.")}
        torch.save(dict(state=state, cfg=K.cfg.__dict__, core_tag=a.core_tag, tuned_from=a.kernel_tag, tune=vars(a)),
                   os.path.join(a.out, f"{a.tag}.pt"))
    else:
        torch.save(dict(state=Fm.net.state_dict(), cfg=Fm.cfg.__dict__, kernel_tag=a.kernel_tag,
                        tuned_from=a.flow_tag), os.path.join(a.out, f"{a.tag}.pt"))


# --------------------------------------------------------------------------------------------------------- score
def cmd_score(a, rest):
    pa, cl, core = load_core(a)
    dev = torch.device(a.device)
    K = load_kernel(a, core, a.kernel_tag) if any(not x.startswith("dev") and x != "persistence" for x in a.arms) \
        else PhysicsKernel(core)
    te = D.load(pa, [a.test_year])[a.test_year]
    L, M = a.leads, a.members
    ics = np.arange(a.ic_start, len(te) - L, a.ic_stride)[: a.max_ics or None]
    w = np.cos(np.deg2rad(cl.lat))
    w = (w / w.mean()).astype(np.float64)
    dyn = np.asarray(DYN)
    W = int(getattr(core, "mem", 0))
    res, raws, t_all = {}, {}, time.time()
    eps = rkt().EPS
    clim_z = ((cl.clim - cl.mean[None, :, None, None]) / (cl.std[None, :, None, None] + eps)).astype(np.float64)
    clim_of = lambda j: np.stack([clim_z[np.minimum(np.arange(k + 1, k + 1 + L), 365)][:, dyn] for k in j])  # noqa: E731
    for arm in a.arms:
        g = torch.Generator().manual_seed(a.seed)
        Dk, Df = None, None
        if arm.startswith("devflow"):  # flow matching from the deviation kernel: M independent members
            Df = load_devflow(a, core, arm, pairs=M)
            step, m = None, M
        elif arm.startswith("devk"):  # the entropy deviation kernel: its own rollout (mean + deviations)
            Dk = load_devk(a, core, arm)
            step, m = None, 2 * Dk.cfg.pairs
        elif arm == "core":  # the core's own day (a memory core: its memory update, then the day)
            step, m = (lambda x, d, tn: core_step(K.core, x, d, tn)), 1
        elif arm == "kernel":
            step, m = (lambda x, d, tn: K.core.finish(K.sample(x, d, tn, g)[0], tn)), M
        elif arm == "persistence":
            step, m = (lambda x, d, tn: x), 1
        else:
            Fm = load_flow(a, K, arm)
            step, m = (lambda x, d, tn, Fm=Fm: Fm.sample(x, d, tn, g, steps=a.steps)[0]), M
        sc = Scorer(L, w, cl.inc, cl.std, pairs=Dk.cfg.pairs if Dk is not None else 0)
        t0 = time.time()
        for i in range(0, len(ics), a.ic_batch):
            j = ics[i:i + a.ic_batch]
            x0 = torch.from_numpy(te[j][:, dyn]).to(dev)
            tis = torch.from_numpy(np.stack([te[k + 1:k + 1 + L, TISR] for k in j])).to(dev)
            truth = np.stack([te[k + 1:k + 1 + L][:, dyn] for k in j])
            hist = (torch.from_numpy(np.stack([te[np.maximum(np.arange(k - W, k), 0)][:, dyn] for k in j])).to(dev)
                    if W else None)
            if Df is not None:
                ens, hk = Df.rollout(x0, torch.tensor(j, device=dev), tis, L, hist=hist, gen=g, steps=a.steps)
                hk = hk[:, None]
            elif Dk is not None:
                ens, hk = Dk.rollout(x0, torch.tensor(j, device=dev), tis, L, hist=hist, gen=g)
                hk = hk[:, None]  # the predicted entropy, in the place of the kernel's (B, 1, L, 66, H, W)
            else:
                ens, hk = rollout(step, x0, torch.tensor(j, device=dev), tis, m, L,
                                  kernel=K if (m > 1 and a.kernel_entropy) else None, gen=g, hist=hist,
                                  start=None if arm == "persistence" else K.start)
            sc.update(ens.cpu().numpy().astype(np.float64), truth.astype(np.float64),
                      None if hk is None else hk.cpu().numpy(), clim=clim_of(j))
        res[arm] = sc.summary()
        raws[arm] = sc.raw()
        res[arm]["members"], res[arm]["sec"] = m, round(time.time() - t0)
        log({arm: {k: res[arm][k] for k in ("nrmse", "crps", "spread_skill", "entropy", "ign") if k in res[arm]}})
    # climatology as a reference (z-scored daily climatology of the training years)
    sc = Scorer(L, w, cl.inc, cl.std)
    for i in range(0, len(ics), a.ic_batch):
        j = ics[i:i + a.ic_batch]
        c = clim_of(j)
        truth = np.stack([te[k + 1:k + 1 + L][:, dyn] for k in j]).astype(np.float64)
        sc.update(c[:, None], truth, clim=c)
    res["climatology"] = sc.summary()
    raws["climatology"] = sc.raw()
    np.savez_compressed(os.path.join(a.out, f"{a.score_tag}_raw.npz"),
                        **{f"{arm}__{k}": v for arm, r in raws.items() for k, v in r.items()},
                        ics=ics, std=cl.std[dyn], inc=cl.inc[dyn])
    meta = dict(test_year=a.test_year, ics=len(ics), channels=[contract.CHANNEL_NAMES[c] for c in DYN], ic_stride=a.ic_stride, members=M, leads=L, steps=a.steps,
                core_tag=a.core_tag, kernel_tag=a.kernel_tag, sec=round(time.time() - t_all))
    path = os.path.join(a.out, f"{a.score_tag}.json")
    with open(path, "w") as f:
        json.dump(dict(meta=meta, scores=res), f, indent=1)
    log(dict(saved=path, **meta))


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("core")
    common(c)
    c.add_argument("--iters", type=int, default=3000)
    k = sub.add_parser("kernel")
    common(k)
    k.add_argument("--tag", default="kernel")
    k.add_argument("--iters", type=int, default=1500)
    k.add_argument("--batch", type=int, default=8)
    k.add_argument("--lr", type=float, default=1e-3)
    k.add_argument("--centres", type=int, default=4)
    k.add_argument("--hidden", type=int, default=32)
    k.add_argument("--log-every", type=int, default=100)
    f = sub.add_parser("flow")
    common(f)
    f.add_argument("--tag", default="flow")
    f.add_argument("--kernel-tag", default="kernel")
    f.add_argument("--source", default="kernel", choices=["kernel", "gauss"])
    f.add_argument("--iters", type=int, default=3000)
    f.add_argument("--batch", type=int, default=8)
    f.add_argument("--lr", type=float, default=5e-4)
    f.add_argument("--hidden", type=int, default=64)
    f.add_argument("--steps", type=int, default=6)
    f.add_argument("--log-every", type=int, default=250)
    d = sub.add_parser("devk")
    common(d)
    d.add_argument("--tag", default="devk")
    d.add_argument("--days", type=int, nargs="+", default=[2, 4, 6, 8, 10], help="rollout days, a curriculum")
    d.add_argument("--pairs", type=int, default=4)
    d.add_argument("--eps0", type=float, default=0.25)
    d.add_argument("--smooth", type=int, default=5)
    d.add_argument("--hidden", type=int, default=32)
    d.add_argument("--iters", type=int, default=800)
    d.add_argument("--batch", type=int, default=4)
    d.add_argument("--lr", type=float, default=1e-3)
    d.add_argument("--val-batches", type=int, default=6)
    d.add_argument("--log-every", type=int, default=50)
    df = sub.add_parser("devflow")
    common(df)
    df.add_argument("--tag", default="devflow")
    df.add_argument("--dev-tag", default="devk")
    df.add_argument("--source", default="kernel", choices=["kernel", "gauss"])
    df.add_argument("--days", type=int, default=10)
    df.add_argument("--iters", type=int, default=600)
    df.add_argument("--batch", type=int, default=4)
    df.add_argument("--lr", type=float, default=5e-4)
    df.add_argument("--hidden", type=int, default=64)
    df.add_argument("--steps", type=int, default=6)
    df.add_argument("--val-batches", type=int, default=4)
    df.add_argument("--log-every", type=int, default=50)
    t = sub.add_parser("tune")
    common(t)
    t.add_argument("--tag", required=True)
    t.add_argument("--kernel-tag", default="kernel")
    t.add_argument("--flow-tag", default="", help="tune this flow (on --kernel-tag); empty = tune the kernel")
    t.add_argument("--days", type=int, nargs="+", default=[2, 3, 4, 6], help="rollout days, a curriculum")
    t.add_argument("--members", type=int, default=4)
    t.add_argument("--iters", type=int, default=600)
    t.add_argument("--batch", type=int, default=2)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--lam", type=float, default=0.5, help="weight of the one-day loss (anchor)")
    t.add_argument("--steps", type=int, default=6)
    t.add_argument("--through", action="store_true", help="gradient through the states of the rollout (no detach)")
    t.add_argument("--through-core", action="store_true",
                   help="flow, with --through: also the equation's Jacobian of the next day's source and conditioning")
    t.add_argument("--val-batches", type=int, default=4)
    t.add_argument("--log-every", type=int, default=50)
    s = sub.add_parser("score")
    common(s)
    s.add_argument("--arms", nargs="+", default=["persistence", "core", "kernel", "flow"])
    s.add_argument("--kernel-tag", default="kernel")
    s.add_argument("--score-tag", default="score")
    s.add_argument("--members", type=int, default=8)
    s.add_argument("--leads", type=int, default=10)
    s.add_argument("--ic-stride", type=int, default=6)
    s.add_argument("--ic-start", type=int, default=0)
    s.add_argument("--max-ics", type=int, default=0)
    s.add_argument("--ic-batch", type=int, default=4)
    s.add_argument("--steps", type=int, default=6)
    s.add_argument("--kernel-entropy", action="store_true", help="also the kernel's entropy along every trajectory")
    a, rest = p.parse_known_args(argv)
    rest = [r for r in rest if r != "--"]
    if a.threads:
        torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    {"core": cmd_core, "kernel": cmd_kernel, "flow": cmd_flow, "tune": cmd_tune, "devk": cmd_devk, "devflow": cmd_devflow, "score": cmd_score}[a.cmd](a, rest)


if __name__ == "__main__":
    main()
