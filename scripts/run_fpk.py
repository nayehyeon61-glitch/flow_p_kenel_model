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

The deviation kernel and its flow (devk, devflow), and two uses of a trained deviation flow:
    # reflow: the student learns the teacher's own (source, endpoint) pairs, its paths straighten
    python scripts/run_fpk.py devflow --dev-tag devk --reflow devflow --tag devflow_r1
    # fine-tuning with the fair energy score of the members, through the sampler (2 steps of it by default; the
    # flow's own steps at the same memory: --steps 0 --chunk 2)
    python scripts/run_fpk.py devtune --flow-tag devflow_r1 --tag devflow_r1_es --norm grid

Core and kernel learning along instead of frozen (both off by default):
    # the core's weights in the kernel's training, through the mean's path of --through; the tuned core is a core
    # checkpoint of its own (--core-out, default <core-tag>_<tag>) and the kernel is recorded on it
    python scripts/run_fpk.py devk --init devk_x --tag devk_u --through 10 --core-lr 2e-5 --core-anchor 1.0 ...
    python scripts/run_fpk.py score --core-tag core_gng_devk_u --arms devk_u
    # the kernel's parameters in the flow's energy-score tuning; the tuned kernel (--kernel-out, default
    # <the flow's kernel>_j) and the tuned flow on it are both written
    python scripts/run_fpk.py devtune --flow-tag devflow --tag devflow_j --kernel-lr 1e-5 --kernel-lam 0.5
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
from fpk.devflow import DevFlowConfig, DeviationFlow  # noqa: E402,F401
from fpk.coreflow import build_flow  # noqa: E402
from fpk.devkernel import DevConfig, EntropyDeviationKernel, TotalEnergy  # noqa: E402
from fpk.escore import energy_score, make_norm, spread_skill  # noqa: E402
import torch.nn.functional as F  # noqa: E402
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
    """Train the tendency core with PG-CFM's own script: plain core, GroupNorm, physical terms G(x).
    ``--no-gfeat``: the control without the physical field, the same core with no G(x) inputs (the kernel and the
    flow built on it then read no G(x) either: fpk.devkernel / fpk.devflow follow the core's gfeat)."""
    argv = ["--arm", "core", "--tag", a.core_tag, "--core-norm", "gn"] + ([] if a.no_gfeat else ["--core-gfeat"]) + [
            "--log-every", "250", "--iters", str(a.iters), "--seed", str(a.seed)] + list(rest)
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


def own_output(path, tuned_from, key, tag):
    """Refuse to write the tuned checkpoint ``path`` over anything but an earlier output of this very run: a file
    that records the same source (tuned_from) and, under ``key`` (the saved options), the same --tag. An original
    core or kernel records neither, so it is never overwritten."""
    if not os.path.exists(path):
        return
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if not (isinstance(ck, dict) and ck.get("tuned_from") == tuned_from and (ck.get(key) or {}).get("tag") == tag):
        raise SystemExit(f"{path} exists and was not written by this run (tuned from {tuned_from}, --tag {tag}): "
                         f"give another name or remove it")


def save_atomic(*pairs):
    """(object, path), ...: every file is written beside its place first and then put there, in the order given
    (a failed write leaves all of them as they were; the files of a pair change together but for a kill between
    the renames)."""
    for obj, path in pairs:
        torch.save(obj, path + ".tmp")
    for _, path in pairs:
        os.replace(path + ".tmp", path)


# ---------------------------------------------------------------------------------------------------------- devk
def cmd_devk(a, rest):
    """The entropy deviation kernel (fpk.devkernel): the mean runs the core's own trajectory for D days, the
    deviations are carried by the equation, and the entropy residual r (and the size h of the new part) is fitted
    by the Gaussian log score of the data around the mean on every day (calibrated entropy: E[IGN] = H).

    --core-lr LR > 0: the core is not frozen. Its trained weights theta (fpk.devkernel.core_weights: what the
    core's own training optimizes; no buffer, statistic or scale of the physical terms) get the gradient of the
    same loss through the mean's path of --through K (required: K >= 1), xbar_(L+1) = Phi_theta(xbar_L, m_L) with
    the memory's update, for the K days of a window; the loss of the iteration is

        sum_L loss_L / D + seq_weight x seq  +  W x sum((theta - theta0)^2) / sum(theta0^2)        (--core-anchor W)

    theta0 = the core as loaded. What the core learns from: the mean's error in the log score and in the sequence
    term, and what the kernel's networks read of the mean's state and memory. Not from the spread: the pairs
    Phi(xbar +- s e) are evaluated without a graph, as are the physical terms G(x) and the memory's warm-up on the
    days before the IC (a constant of the gradient, rebuilt with the current weights for every batch). The core
    stays in eval mode (EntropyDeviationKernel.train). Its weights are an optimizer group of their own (lr LR,
    weight decay 0, the cosine schedule of the other groups) with their own gradient clipping (norm 1): the
    kernel's clipping is what it was. The anchor's gradient 2 W (theta - theta0) / sum(theta0^2) is added AFTER
    that clip (the loss's gradient is cut to norm 1, the pull is not, so a large W does not take the loss's share).
    Its size: it equals a clipped loss gradient at core_rel = |theta0| / (2 W), so on a core of |theta0|^2 ~ 800
    W = 1 does nothing (core_rel 14) and W ~ 1e3 holds the core near 1 %; the low learning rate itself bounds the
    drift to about the sum of the steps' learning rates a weight. Memory: with frozen weights the graph of the mean's path starts at the first
    day whose input has a gradient (day 2 with --correct; without --correct there is none), now at day 1 of every
    window; the weights' own gradients are one more tensor a weight.
    Checkpoints, at every best validation: the tuned core <out>/<--core-out>.pt, the original core's checkpoint
    with the trained weights replaced (the format build_core loads) plus tuned_from and the options (devk), and
    the kernel with core_tag = --core-out (and core_from, train): score / diag_* / devflow --core-tag <--core-out>
    take both, the original core is refused for this kernel and its file is never written. Logged with every
    validation: core_rel = |theta - theta0| / |theta0|, core_anchor = W core_rel^2, core_grad = the norm of the
    loss's gradient on the core's weights before the clip, core_pull = the norm of the anchor's gradient."""
    torch.manual_seed(a.seed)
    if a.core_lr < 0 or (not a.core_lr > 0 and (a.core_anchor or a.core_out)):
        raise SystemExit("--core-anchor / --core-out belong to --core-lr LR > 0 (the core is frozen without it)")
    if a.fresh_best and not a.init:
        raise SystemExit("--fresh-best belongs to --init (without it every checkpoint is a trained state)")
    if a.core_lr > 0:
        if a.through < 1:
            raise SystemExit("--core-lr > 0 needs --through K >= 1: the core's weights learn from the gradient through "
                             "the mean's path of the equation, and --through 0 cuts that path after every day")
        a.core_out = a.core_out or f"{a.core_tag}_{a.tag}"
        if a.core_out == a.core_tag:
            raise SystemExit(f"--core-out {a.core_out} is the core's own tag: the original core is never overwritten")
        if a.core_out in (a.tag, a.init):
            raise SystemExit(f"--core-out {a.core_out} is the tag of a kernel of this run (--tag / --init): two files, two names")
        own_output(os.path.join(a.out, f"{a.core_out}.pt"), a.core_tag, "devk", a.tag)
    pa, cl, core = load_core(a)
    dev = torch.device(a.device)
    Dk = EntropyDeviationKernel(core, DevConfig(pairs=a.pairs, hidden=a.hidden, eps0=a.eps0, smooth=a.smooth,
                                               cloud_mean=a.cloud_mean, correct=a.correct, noise=a.noise,
                                               r_dissip=a.r_dissip, band_scale=a.band_scale,
                                               e0_spec=a.e0_spec, noise_pre=a.noise_pre,
                                               noise_mult=a.noise_mult, noise_spec=a.noise_spec,
                                               bal_vert=a.bal_vert, fdt=a.fdt, fdt_bmin=a.fdt_bmin,
                                               bal_lmin=a.bal_lmin, aam_fix=a.aam_fix),
                                std=cl.std).to(dev)
    if a.fdt_weight and not a.fdt:
        raise SystemExit("--fdt-weight needs the climatological band power: --fdt clim_bands.json")
    if a.init:  # start from a fitted kernel (the correction's zero head keeps its forecasts)
        ck = torch.load(os.path.join(a.out, f"{a.init}.pt"), map_location="cpu", weights_only=False)
        miss, own = Dk.load_init(ck["state"])
        assert not miss.unexpected_keys, miss
        if ck.get("core_tag") not in (None, a.core_tag):  # allowed (a start value), but never silently
            log(dict(init=a.init, fitted_on_core=ck["core_tag"], this_core=a.core_tag))
        if own:  # the options of this run decide the fixed buffers of the balanced noise / fdt, not the checkpoint
            log(dict(init=a.init, buffers_of_this_run_kept=own))
    Dk.through = a.through
    if a.energy_weight:
        names = json.load(open(os.path.join(a.out, "score_devk_m.json")))["meta"]["channels"]
        TE = TotalEnergy(names, cl.mean, cl.std, cl.lat)
    W, Dmax = int(getattr(core, "mem", 0)), max(a.days)
    years = D.load(pa, a.train_years)
    va = D.load(pa, [a.val_year])
    tr, vr = D.Pairs(years, a.seed, hist=W, ahead=Dmax), D.Pairs(va, a.seed + 1, hist=W, ahead=Dmax)
    w = D.area_weights(cl.lat).to(dev)
    if a.energy_weight:  # the spread of the data's daily change of total energy (training years)
        ey = np.concatenate([years[y][:, DYN] for y in sorted(years)[-2:]])
        et = TE(torch.from_numpy(ey).float())
        sig_E = float((et[1:] - et[:-1]).std())
        log(dict(total_energy_daily_change_std=round(sig_E, 3)))
    params = [p_ for p_ in Dk.parameters() if p_.requires_grad]
    cn = [p_ for n_, p_ in Dk.named_parameters() if p_.requires_grad and n_.startswith("cnet.")]
    rest_ = [p_ for n_, p_ in Dk.named_parameters() if p_.requires_grad and not n_.startswith("cnet.")]
    cw, cpar, th0, n0, ck_core = {}, [], [], 1.0, None
    if a.core_lr > 0:  # after the kernel's own lists: the core's weights are a group and a clipping of their own
        ck_core = torch.load(os.path.join(a.out, f"{a.core_tag}.pt"), map_location="cpu", weights_only=False)
        cw = Dk.train_core()
        cpar = list(cw.values())
        bad = [k for k, v in cw.items() if k not in ck_core["state"] or not torch.equal(ck_core["state"][k].to(v), v.detach())]
        if bad:
            raise SystemExit(f"--core-lr: the core in memory is not the checkpoint {a.core_tag}.pt (weights {bad[:4]} ...)")
        th0 = [p_.detach().clone() for p_ in cpar]
        n0 = float(sum(t.pow(2).sum() for t in th0))
        log(dict(core_tuned=a.core_out, tuned_from=a.core_tag, core_weights=len(cpar), core_params=sum(p_.numel() for p_ in cpar),
                 core_lr=a.core_lr, core_anchor=a.core_anchor, through=a.through))
    opt = torch.optim.AdamW([dict(params=rest_, lr=a.lr)] + ([dict(params=cn, lr=a.lr_correct or a.lr)] if cn else [])
                            + ([dict(params=cpar, lr=a.core_lr, weight_decay=0.0)] if cpar else []), weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
    dyn = torch.tensor(DYN)
    vx = [vr(a.batch) for _ in range(a.val_batches)]
    # sum((theta - theta0)^2) / sum(theta0^2) over the core's trained weights: the anchor's term, core_rel^2
    core_d2 = lambda: sum((p_ - q).pow(2).sum() for p_, q in zip(cpar, th0)) / n0  # noqa: E731

    aux = {}  # of the last run: the fdt penalty of every day

    def run(batch, days, gen=None):
        x, ys, doy, tns, hi = batch
        if ys.dim() == 4:
            ys, tns = ys[:, None], tns[:, None]
        xd, yd, doy, tns = x[:, dyn].to(dev), ys[:, :, dyn].to(dev), doy.to(dev), tns.to(dev)
        e = Dk.start(xd, doy, None if hi is None else hi[:, :, dyn].to(dev), gen)
        xb, ign, ent, prod, rm, es, vs = xd, [], [], [], [], [], []
        aux["fdt"] = []
        xprev = xd
        for k in range(days):
            xb, e, H, (st, r) = Dk.day(xb, e, doy + k, tns[:, k], k, gen)
            i, h_ = Dk.lead_loss(H, xb, e, yd[:, k], w, a.band_weight)  # the band term on e before the detach
            if a.fdt_weight:  # the spread about the mean stays below the climate's variance, band by band
                fd = Dk.fdt_penalty(e)
                i = i + a.fdt_weight * fd
                aux["fdt"].append(float(fd.detach()))
            if a.energy_weight:
                # the physical constraint: a member's change of total energy over the day stays within k sigma of
                # the mean's (sigma = the spread of the data's daily change of energy, measured once)
                xm = Dk.members(xb.detach(), e, Dk.s)                                   # (B, 2P, 66, H, W)
                dE = (TE(xm) - TE(xprev[:, None])) - (TE(xb.detach()) - TE(xprev))[:, None]
                i = i + a.energy_weight * F.relu(dE.abs() / sig_E - a.energy_k).pow(2).mean()
            xprev = xb.detach()
            e = e.detach()
            es.append((xb - yd[:, k]) / Dk.s)
            vs.append(torch.exp(2 * H))
            rm.append(float(((((xb - yd[:, k]) / Dk.s).pow(2) * Dk.keep * w).sum() / (Dk.keep * w).expand_as(xb).sum()).detach()))
            if not a.through or (k + 1) % a.through == 0:
                xb = xb.detach()  # the window of the gradient through the chain ends (--through 0: every day)
                Dk.detach_state()
            ign.append(i)
            ent.append(float(h_.detach()))
            m = Dk.keep * w
            pe, pn, pr = Dk.last["prod"]
            prod.append(tuple(float((t * m).sum() / m.expand_as(t).sum()) for t in (pe, pn, pr)))
        # the sequence term: the error of the mean of the whole sequence (what piles up with one sign over the days;
        # the days' own scores see it as independent errors), in units of the sequence's mean variance
        m = Dk.keep * w
        eb, vb = torch.stack(es, 1).mean(1), torch.stack(vs, 1).mean(1).detach()
        seq = (eb.pow(2) / vb * m).sum() / m.expand_as(eb).sum()
        return ign, ent, prod, rm, seq

    def val():
        Dk.eval()
        gv = torch.Generator().manual_seed(123)
        with torch.no_grad():
            r = [run(b, Dmax, gv) for b in vx]
        Dk.train()
        ign = np.mean([[float(v) for v in q[0]] for q in r], 0)
        ent = np.mean([q[1] for q in r], 0)
        # calibrated: IGN - H = 1/2 (E[err^2] = variance); > 1/2 = too sure, < 1/2 = too wide
        mse = np.sqrt(np.mean([q[3] for q in r], 0))  # RMSE of the mean in units of s
        return ([round(float(v), 3) for v in ign], [round(float(v), 3) for v in ign - ent],
                [round(float(v), 3) for v in mse], round(float(np.mean([float(q[4]) for q in r])), 4))

    def save():
        state = {k: v for k, v in Dk.state_dict().items() if not k.startswith("core.")}
        if not cw:
            torch.save(dict(state=state, cfg=Dk.cfg.__dict__, core_tag=a.core_tag), os.path.join(a.out, f"{a.tag}.pt"))
            return
        # the tuned core first (a kernel never names a core that is not there): the original checkpoint's dict with
        # the trained weights replaced, so build_core loads it as it loads the original
        tuned = {**ck_core["state"], **{k: v.detach().cpu().clone() for k, v in cw.items()}}
        save_atomic(({**ck_core, "state": tuned, "tuned_from": a.core_tag, "devk": dict(vars(a))},
                     os.path.join(a.out, f"{a.core_out}.pt")),
                    (dict(state=state, cfg=Dk.cfg.__dict__, core_tag=a.core_out, core_from=a.core_tag, train=dict(vars(a))),
                     os.path.join(a.out, f"{a.tag}.pt")))

    t0 = time.time()
    i0, g0, m0, s0 = val()
    log(dict(devk_val_start=dict(ign=i0, ign_minus_H=g0, rmse_s=m0, seq=s0)))
    score = lambda vi_, vs_: sum(vi_) / len(vi_) + a.seq_weight * vs_  # noqa: E731
    # --init: the start is a candidate checkpoint (a continued run never ends worse than it began), unless
    # --fresh-best: a run whose options change the sampler or the loss (--fdt and its penalty, another noise) is not
    # measured by the start's score, and with it as the bar its training can be discarded as a whole
    best = score(i0, s0) if (a.init and not a.fresh_best) else float("inf")
    if a.init:
        save()
    for it in range(1, a.iters + 1):
        days = a.days[min(len(a.days) - 1, (it - 1) * len(a.days) // a.iters)]
        ign, ent, prod, _, seq = run(tr(a.batch), days)
        fdt = dict(fdt=round(float(np.mean(aux["fdt"])), 5)) if a.fdt_weight else {}
        loss = sum(ign) / days + a.seq_weight * seq
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        if cpar:  # on their own: the core's gradient does not change the clipping of the kernel's
            cg = float(torch.nn.utils.clip_grad_norm_(cpar, 1.0))        # the loss's gradient before the clip
            if a.core_anchor:  # the pull's gradient 2 W (theta - theta0) / sum(theta0^2), added after the clip: it
                with torch.no_grad():  # is not cut with the loss's, and a large W does not take the loss's share
                    for p_, q in zip(cpar, th0):
                        if p_.grad is None:
                            p_.grad = torch.zeros_like(p_)
                        p_.grad.add_(p_ - q, alpha=2.0 * a.core_anchor / n0)
        opt.step()
        sch.step()
        if it % a.log_every == 0 or it == a.iters:
            vi, vg, vm, vs_ = val()
            if score(vi, vs_) < best:  # the checkpoint is the best validation loss of the sequence
                best = score(vi, vs_)
                save()
            d2 = float(core_d2().detach()) if cpar else 0.0
            # core_grad: the norm of the loss's gradient on the core's weights before its clip (to 1); core_pull:
            # the norm of the anchor's, 2 W core_rel / |theta0|. The anchor holds the core where the two are equal
            ct = dict(core_rel=round(d2 ** 0.5, 6), core_anchor=round(a.core_anchor * d2, 6), core_grad=round(cg, 4),
                      core_pull=round(2.0 * a.core_anchor * (d2 / n0) ** 0.5, 6)) if cpar else {}
            log(dict(it=it, days=days, ign=round(float(loss.detach()), 4), best=round(best, 4), val_seq=vs_, val_ign=vi, val_ign_minus_H=vg, val_rmse_s=vm,
                     prod_eq=[round(p_[0], 3) for p_ in prod], prod_noise=[round(p_[1], 3) for p_ in prod],
                     r=[round(p_[2], 3) for p_ in prod],
                     h=round(float(Dk.h().mean()), 4), **fdt, **ct, sec=round(time.time() - t0)))


def load_devk(a, core, tag):
    ck = torch.load(os.path.join(a.out, f"{tag}.pt"), map_location="cpu", weights_only=False)
    if ck.get("core_tag") not in (None, a.core_tag):
        raise SystemExit(f"deviation kernel {tag} was fitted on core {ck['core_tag']}, not --core-tag {a.core_tag}")
    cfg = ck["cfg"]
    cfg["blocks"] = tuple(cfg["blocks"])
    if getattr(a, "cloud_mean", False):  # diagnostic: a kernel fitted on the core's trajectory, scored with the cloud's mean
        cfg["cloud_mean"] = True
    Dk = EntropyDeviationKernel(core, DevConfig(**cfg))
    miss = Dk.load_state_dict(ck["state"], strict=False)
    assert not miss.unexpected_keys and all(k.startswith("core.") for k in miss.missing_keys), miss
    return Dk.to(a.device).eval()


# ------------------------------------------------------------------------------------------------------- devflow
def reflow_teacher(a, Fm):
    """The teacher of ``devflow --reflow TAG``: the trained flow TAG, built on the student's kernel. It must have
    been trained on the same kernel and core (the same sources) with the student's architecture and path; the
    student Fm starts from its weights."""
    if a.reflow == a.tag:
        raise SystemExit(f"--reflow {a.reflow}: the student's --tag is the teacher's (it would be overwritten)")
    ck = torch.load(os.path.join(a.out, f"{a.reflow}.pt"), map_location="cpu", weights_only=False)
    if ck.get("core_tag") not in (None, a.core_tag):
        raise SystemExit(f"--reflow {a.reflow} was trained on core {ck['core_tag']}, not --core-tag {a.core_tag}")
    if ck["dev_tag"] != a.dev_tag:
        raise SystemExit(f"--reflow {a.reflow} was trained on the kernel {ck['dev_tag']}, not --dev-tag {a.dev_tag}")
    cfg = dict(ck["cfg"])
    cfg["blocks"] = tuple(cfg["blocks"])
    tc = DevFlowConfig(**cfg)
    keys = ("hidden", "blocks", "source", "path") + (("core_steps", "core_sub") if tc.path == "core" else ())
    diff = {k: [getattr(Fm.cfg, k), getattr(tc, k)] for k in keys if getattr(Fm.cfg, k) != getattr(tc, k)}
    if diff:
        raise SystemExit(f"--reflow {a.reflow}: the student is not the teacher's flow, [student, teacher] = {diff}")
    Te = build_flow(Fm.dk, tc).to(a.device)
    Te.net.load_state_dict(ck["state"])
    Fm.net.load_state_dict(ck["state"])
    for p_ in Te.net.parameters():
        p_.requires_grad_(False)
    return Te.eval()


@torch.no_grad()
def path_diag(flows, Dk, batches, days, dev, seed=123, steps=None, ref=None):
    """How straight the paths of the flows (name -> flow, all on the kernel Dk) are, on the same kernel days and
    source rows z0 of ``batches``, per lead, in units of sigma. ``steps``: the number of steps n of the full sampler
    of EVERY flow (None = each flow's own; reflow: the teacher's, so both flows are judged with one sampler).
    ``ref``: the name of the flow whose full endpoint R_n(z0) is the reference of drift and ref_m (reflow: the
    teacher).
        steps     n as used (line: Heun steps; core path: Heun steps of the learned part of a grid step, 0 = one
                  Euler step a grid step, which has no step count to vary)
        straight  kinetic energy int |v|^2 dtau / squared displacement |T_n(z0) - z0|^2 of the full sampler
                  (``last_path``; 1 = straight lines at constant speed. Core path: of the learned part)
        end_1, end_2   RMS[T_m(z0) - T_n(z0)] / RMS[T_n(z0)] for a sampler of m = 1, 2 steps: how far the flow's OWN
                  endpoint moves when its sampler is cut to m steps (None when the path has no step count)
        move      RMS[T_n(z0) - z0] / RMS[T_n(z0)]: the size of the transport itself, the scale of end_m (core
                  path: of the learned part)
      and with ``ref``:
        drift     RMS[T_n(z0) - R_n(z0)] / RMS[R_n(z0)]: how far the flow's endpoints are from the reference's on
                  the same rows (0 for the reference). A reflow keeps the law of the endpoints only as far as the
                  student reproduces the teacher's: end_m of a student says nothing about that, drift does
        ref_1, ref_2   RMS[T_m(z0) - R_n(z0)] / RMS[R_n(z0)]: the cut sampler against the REFERENCE's full endpoint
                  (the reference's own ref_m is its end_m). The student's m-step sampler is nearer the teacher's
                  law than the teacher's own m-step sampler only if its ref_m is below the teacher's end_m."""
    dyn = torch.tensor(DYN)
    names = ([ref] if ref else []) + [n for n in flows if n != ref]      # the reference first
    acc = {n: np.zeros((days, 9)) for n in names}
    cut = {n: flows[n].cfg.path == "line" or flows[n].cfg.core_sub > 0 for n in names}   # a step count to vary
    gen = torch.Generator().manual_seed(seed)
    for x, ys, doy, tns, hi in batches:
        if ys.dim() == 4:
            ys, tns = ys[:, None], tns[:, None]
        xd, doy, tns = x[:, dyn].to(dev), doy.to(dev), tns.to(dev)
        e = Dk.start(xd, doy, None if hi is None else hi[:, :, dyn].to(dev), gen)
        xb = xd
        for k in range(days):
            x0 = xb
            xb, e, H, _ = Dk.day(xb, e, doy + k, tns[:, k], k, gen)
            c = flows[names[0]].day_ctx(x0, xb, e, H, doy, tns[:, k], k, gen)
            ms = lambda t: float((t / c.sig.view_as(t)).pow(2).mean())  # noqa: E731
            base = None
            for n in names:
                Fl = flows[n]
                full = Fl.day_run(c, steps if cut[n] else None)
                kin, disp = Fl.last_path
                base = full if n == ref else base
                runs = [Fl.day_run(c, m) if cut[n] else None for m in (1, 2)]
                vs = lambda b: [np.nan if q is None else ms(q - b) for q in runs]  # noqa: E731
                acc[n][k] += [kin, disp, ms(full)] + vs(full) + ([np.nan] * 4 if ref is None else [ms(base), ms(full - base)] + vs(base))
    out = {}
    for n in flows:
        Fl, v = flows[n], acc[n]
        r = lambda t: [None if np.isnan(q) else round(float(q), 4) for q in t]  # noqa: E731
        own = Fl.cfg.steps if Fl.cfg.path == "line" else Fl.cfg.core_sub
        out[n] = dict(steps=(steps or own) if cut[n] else 0,
                      straight=r(v[:, 0] / np.maximum(v[:, 1], 1e-30)), end_1=r(np.sqrt(v[:, 3] / v[:, 2])),
                      end_2=r(np.sqrt(v[:, 4] / v[:, 2])), move=r(np.sqrt(v[:, 1] / v[:, 2])))
        if ref is not None:
            out[n].update(drift=r(np.sqrt(v[:, 6] / v[:, 5])), ref_1=r(np.sqrt(v[:, 7] / v[:, 5])), ref_2=r(np.sqrt(v[:, 8] / v[:, 5])))
    return out


def cmd_devflow(a, rest):
    """Flow matching from the entropy deviation kernel (fpk.devflow): the kernel runs D days from the data's state,
    and on every day its deviations are the source and the data's deviation from the mean the target.
    ``--reflow TEACHER``: the target of every source row is the teacher's transported endpoint of that row (the
    teacher's own number of steps, no gradient) in place of the data's deviation; the student starts from the
    teacher's weights. At the end one json line ``reflow_paths`` (``path_diag``) on the validation batches: both
    flows sampled with the TEACHER's number of steps as the full sampler (logged as ``steps``), their straightness,
    the change of the endpoint when the sampler is cut to 1 and 2 steps (against the flow's own full endpoint,
    end_m, and against the teacher's, ref_m), and the student's drift from the teacher's endpoints."""
    torch.manual_seed(a.seed)
    pa, cl, core = load_core(a)
    dev = torch.device(a.device)
    Dk = load_devk(a, core, a.dev_tag)
    Fm = build_flow(Dk, DevFlowConfig(hidden=a.hidden, source=a.source, steps=a.steps, path=a.path,
                                      core_steps=a.core_steps, core_sub=a.core_sub)).to(dev)
    Te = reflow_teacher(a, Fm) if a.reflow else None
    rf = {} if Te is None else dict(teacher=Te)   # the reflow target of the day's loss
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

    def save():
        cfg = dict(Fm.cfg.__dict__)
        if cfg["path"] == "line":  # the straight line's checkpoint as it always was (the path's keys only with a path)
            cfg.pop("path"), cfg.pop("core_steps"), cfg.pop("core_sub")
        torch.save(dict(state=Fm.net.state_dict(), cfg=cfg, dev_tag=a.dev_tag, core_tag=a.core_tag,
                        **(dict(reflow_of=a.reflow) if a.reflow else {})), os.path.join(a.out, f"{a.tag}.pt"))

    def run(batch, gen=None, train=False):
        x, ys, doy, tns, hi = batch
        if ys.dim() == 4:
            ys, tns = ys[:, None], tns[:, None]
        xd, yd, doy, tns = x[:, dyn].to(dev), ys[:, :, dyn].to(dev), doy.to(dev), tns.to(dev)
        with torch.no_grad():
            e = Dk.start(xd, doy, None if hi is None else hi[:, :, dyn].to(dev), gen)
        xb, out = xd, []
        for k in range(a.days):
            x0 = xb
            with torch.no_grad():
                xb, e, H, _ = Dk.day(xb, e, doy + k, tns[:, k], k, gen)
            sel = torch.randperm(e.shape[1], generator=gen)[:a.sources] if train else None
            lk = Fm.day_loss(x0, xb, e, H, yd[:, k], doy, tns[:, k], k, w, gen, sel, **rf)
            if train:  # a lead's graph is freed before the next one (ten leads at once did not fit the memory)
                (lk / a.days).backward()
            out.append(float(lk))
        return out

    def val():
        Fm.eval()
        gv = torch.Generator().manual_seed(123)
        with torch.no_grad():
            r = np.mean([run(b, gv) for b in vx], 0)
        Fm.train()
        return [round(float(v), 4) for v in r]

    t0 = time.time()
    log(dict(devflow_val_start=val(), source=a.source, **({} if a.path == "line" else dict(path=a.path, core_steps=Fm.N, core_sub=a.core_sub)),
             **(dict(reflow=a.reflow) if a.reflow else {})))
    for it in range(1, a.iters + 1):
        opt.zero_grad()
        loss = sum(run(tr(a.batch), train=True)) / a.days
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sch.step()
        if it % a.log_every == 0 or it == a.iters:
            log(dict(it=it, loss=round(loss, 4), val=val(), sec=round(time.time() - t0)))
            save()  # the state so far (a run that is cut off still leaves a usable flow)
    save()
    if Te is not None:  # both flows with the teacher's sampler: its step count is the full sampler of the diagnostic
        Fm.eval()
        n = Te.cfg.steps if Te.cfg.path == "line" else Te.cfg.core_sub
        log(dict(reflow_paths=path_diag(dict(teacher=Te, student=Fm), Dk, vx, a.days, dev, steps=n or None, ref="teacher"),
                 teacher=a.reflow, steps=n))


def load_devflow(a, core, tag, pairs=0):
    ck = torch.load(os.path.join(a.out, f"{tag}.pt"), map_location="cpu", weights_only=False)
    if ck.get("core_tag") not in (None, a.core_tag):
        raise SystemExit(f"deviation flow {tag} was trained on core {ck['core_tag']}, not --core-tag {a.core_tag}")
    Dk = load_devk(a, core, ck["dev_tag"])
    if pairs:
        Dk.cfg.pairs = pairs  # independent members of the forecast (the kernel's parameters do not depend on P)
    cfg = ck["cfg"]
    cfg["blocks"] = tuple(cfg["blocks"])
    Fm = build_flow(Dk, DevFlowConfig(**cfg))  # the checkpoint's path: the straight line (no key) or the core's flow
    Fm.net.load_state_dict(ck["state"])
    Fm.meta = {k: v for k, v in ck.items() if k not in ("state", "cfg")}  # dev_tag, core_tag, reflow_of, tuned_from
    return Fm.to(a.device).eval()


# ------------------------------------------------------------------------------------------------------- devtune
@torch.no_grad()
def net_call_mb(Fm, hw):
    """MB that ONE call of the flow's network on ONE row keeps in the graph for the backward pass, estimated as its
    input plus the outputs of its leaf modules, 4 bytes a value. Measured (resident memory of a sampler with the
    graph, one thread): 31.3 MB against 34.8 here for hidden 64, blocks (2, 2, 2) on 32 x 64; 13.7 against 14.7
    for hidden 32, blocks (2, 2); 2.2 against 2.3 on 8 x 16: an upper estimate, within 10 %."""
    size = []
    hooks = [m.register_forward_hook(lambda m_, i_, o: size.append(o.numel() * o.element_size()))
             for m in Fm.net.modules() if not list(m.children())]
    x = torch.zeros(1, Fm.c_in, *hw, device=next(Fm.net.parameters()).device)
    try:
        Fm.net(x)
    finally:
        [h.remove() for h in hooks]
    return (sum(size) + x.numel() * x.element_size()) / 2 ** 20


def cmd_devtune(a, rest):
    """Energy-score fine-tuning of a trained deviation flow (line or core path). The kernel runs D days from the
    data's state exactly as in ``devflow`` (no gradient); on every lead L the flow's members z^m = T(e^m) are drawn
    with the gradient through the sampler and the loss of the lead is

        ES_L / n_L  +  lam x (the flow-matching loss of the same batch and sources, the anchor)

    ES_L = the fair energy score (fpk.escore) of the M members against the truth's deviation (truth - xbar) / s in
    the norm --norm, mean over the batch. n_L = the mean over the validation batches of |truth - xbar| in that
    norm at lead L (the energy score of the mean alone, a forecast without spread), computed once at the start:
    a constant per lead, so the leads weigh alike and the score stays proper (a weight that depends on the batch's
    truth or draws would not). ES_L / n_L < 1: the members score better than the mean alone; about 0.71 for a
    calibrated Gaussian in many dimensions. A lead's graph is freed before the next lead, and the score's graph
    before the anchor's is built (two backward passes a lead, the gradients add).

    --norm band: the band-weighted norm of diag_spectra with the climatological band power of --clim. --band-bmin
    (default 1) is the first band that counts: the band l = 0 is the global mean, which is the core's own mass drift
    and has a tiny climatological power, so with it (--band-bmin 0) the norm is mostly the global mean of msl; the
    default is the norm of diag_spectra's "energy_score_b1" (which takes its own c: the values differ by the c).

    Members. The sources are the +e of the kernel's P antithetic pairs, as in the forecast (``rollout``, score):
    the members T(e^1), ..., T(e^P) are independent draws, which the fair score assumes. The mirror sources -e are
    NOT used: T(e) and T(-e) are not independent (fpk.escore, antithetic pairs). --members M: 0 = all P; M < P =
    M of the P pairs, drawn anew for every lead (the validation uses all P); M > P = the kernel is run with M pairs
    (as score --members does).

    The gradient. Line path: the exact gradient of the discrete Heun sampler (2 x steps network calls a member in
    the graph; --steps sets the tuning sampler, default 2; --val-steps the validation's; 0 = the flow's own).
    Core path: the core's steps are constants and their state Jacobian is replaced by the identity (fpk.coreflow,
    the module's docstring; --core-jac keeps it); --steps is the Heun steps of the learned part when the flow has
    core_sub >= 1.

    Memory. One pass keeps batch x M x calls network evaluations of a lead in the graph (calls a member: 2 x steps
    on the line; N, or 2 N x steps with core_sub >= 1, on the core path), about 35 MB each for the hidden 64,
    blocks (2, 2, 2) network on 32 x 64 (``net_call_mb``): 1.1 GB at batch 2, P = 4, --steps 2; 3.3 GB with the
    flow's own 6 steps, 6.5 GB at batch 4. The count and the estimate are logged, and above --max-graph-gb the run
    is refused (before the data are loaded). --chunk R: the same gradient in two passes, the members without a
    graph and d ES / d members first, then the sampler with the graph for R source rows at a time, each backward
    with its rows' d ES / d z (the chain rule: the rows are independent given the network); the graph then holds
    R x calls evaluations whatever the batch, the members and the steps, at the price of one more sampler pass
    without a graph. The core steps of --core-jac in the graph are not in the estimate.

    Logged: the training ES_L / n_L per lead and the anchor; on the validation batches (fixed draws) ES_L / n_L,
    the raw ES_L, the anchor, spread / skill per lead in the grid norm (the members' std about their mean over
    the RMSE of their mean, x sqrt((M + 1) / M): 1 = calibrated, < 1 = the tuning has narrowed the ensemble) and
    val_straight (kinetic energy / squared displacement of the validation sampler, 1 = straight). When the tuning
    sampler is not the validation's (--steps 2 on a 6-step flow), also val_es_tune (ES_L / n_L of the TUNING
    sampler on the same validation draws) and val_end_tune (RMS of the tuning sampler's endpoint minus the
    validation sampler's over the RMS of the latter, units of sigma): the score is tuned through one sampler and
    judged (here, by score --arms and diag_spectra) with another, and the energy score's gradient through a short
    sampler bends the paths again (on the fixture it gave back part of a reflow's straightening, less with the
    anchor); a val_es that falls less than val_es_tune, or a rising val_end_tune, shows the two drifting apart
    (--steps 0 --chunk R tunes through the validation's sampler at the memory of R rows). The checkpoint is the
    state of the best validation mean of ES_L / n_L with the VALIDATION sampler (the start included); it has the
    format of the flow it came from (``load_devflow``, score --arms, diag_spectra --flow-tag) plus tuned_from.

    --kernel-lr LR > 0: the kernel is not frozen. Each lead's kernel day runs with the graph of the kernel's own
    parameters phi (EntropyDeviationKernel.own_parameters; the core without gradient and the states detached
    between the days, as ``devk`` without --through does), and the loss of the lead is

        ES_L / n_L  +  lam x FM_L(theta; sg(kernel))  +  kernel_lam x K_L(phi)

    ES_L: the members' score as above, now also a function of phi through everything the flow takes from the
    kernel's day: the source z0 = e_L (core path: the day's-end offset e_L - sg(p), the source rows being the
    deviations that entered the day), the conditioning's xbar_L and H_L, and the truth's deviation
    (truth - xbar_L) / s, which moves with the corrected mean; sigma = exp(sg(H_L)), the network's unit, stays a
    constant (DeviationFlow.sigma), and the core path's steps keep their straight-through Jacobian (--core-jac:
    their own). FM_L: the flow-matching anchor of the network, with the kernel's e, xbar and H DETACHED. It must not
    reach the kernel: as a function of the source it is minimized by a source without spread (fpk.devflow, the
    module's docstring), so its gradient would collapse the ensemble. K_L = the log score of the lead of ``devk``
    (lead_loss: IGN + band_weight x band_ign; band_weight from the options stored with the kernel (``tune`` or
    ``train``) if there, else 0.5 for a kernel with band_scale and 0 without; --kernel-band-weight sets it): the
    kernel's anchor. It is not the whole of devk's loss: the sequence term, the gradient through the mean's path
    (--through), the fdt penalty and the energy constraint of the kernel's own training are not in it, and
    nothing of them is logged here (score and diag_spectra of the tuned pair show them). The gradient is taken in two stages (the chain rule): the flow's part hangs on leaves of the
    day's xbar, e and H, every backward pass of the score (also --chunk's) adds to their gradients g, and one
    backward pass of kernel_lam K_L / D + <g, (xbar, e, H)> goes through the kernel's day. The kernel's parameters
    are an optimizer group of their own (lr LR, weight decay 1e-4 as in devk) with their own clipping (norm 1); the
    kernel stays in eval mode as in every flow training (its networks have no mode: GroupNorm, no Dropout). The
    graph of a kernel day (its two small networks on the batch, the band transforms of the deviations) is not in
    graph_gb.
    Checkpoints (best validation): the tuned kernel <out>/<--kernel-out>.pt (default <the flow's kernel>_j: the
    original kernel's checkpoint with the new state, plus tuned_from and tune; ``load_devk`` with the same core
    tag) and the tuned flow with dev_tag = --kernel-out. Logged with every validation: val_kernel_ign and
    val_kernel_H per lead (the kernel's Gaussian log score and mean entropy: IGN - H = 1/2 is calibrated; a falling
    H with a rising IGN is a collapsing source, a rising H a blow-up), next to spread / skill of the members; with
    every training line the kernel's anchor."""
    torch.manual_seed(a.seed)
    a.tag = a.tag or a.flow_tag + "_es"
    if a.tag == a.flow_tag:
        raise SystemExit("--tag is the flow's own tag (it would be overwritten)")
    if not a.tag.startswith("devflow"):
        raise SystemExit(f"--tag {a.tag}: score --arms takes a deviation flow by its tag, which must start with 'devflow'")
    pa, cl, core = load_core(a)
    dev = torch.device(a.device)
    Fm = load_devflow(a, core, a.flow_tag)
    Dk = Fm.dk
    kt, kpar, bw, ck_k, dev0 = a.kernel_lr > 0, [], 0.0, None, Fm.meta["dev_tag"]
    if a.kernel_lr < 0 or (not kt and (a.kernel_out or a.kernel_band_weight >= 0)):
        raise SystemExit("--kernel-out / --kernel-band-weight belong to --kernel-lr LR > 0 (the kernel is frozen without it)")
    if kt:
        a.kernel_out = a.kernel_out or dev0 + "_j"
        if a.kernel_out == dev0:
            raise SystemExit(f"--kernel-out {a.kernel_out} is the flow's own kernel: the original kernel is never overwritten")
        if a.kernel_out in (a.tag, a.flow_tag):
            raise SystemExit(f"--kernel-out {a.kernel_out} is the tag of a flow of this run (--tag / --flow-tag): two files, two names")
        own_output(os.path.join(a.out, f"{a.kernel_out}.pt"), dev0, "tune", a.tag)
        ck_k = torch.load(os.path.join(a.out, f"{dev0}.pt"), map_location="cpu", weights_only=False)
        bw = a.kernel_band_weight if a.kernel_band_weight >= 0 else float(
            (ck_k.get("tune") or ck_k.get("train") or {}).get("band_weight", 0.5 if Dk.cfg.band_scale else 0.0))
        if bw and not hasattr(Dk, "sht"):
            raise SystemExit("--kernel-band-weight > 0: this kernel has no spherical transform (no band score)")
        kpar = list(Dk.own_parameters().values())
        for p_ in kpar:  # frozen by the flow's constructor; the core's weights stay frozen
            p_.requires_grad_(True)
    if a.members > Dk.cfg.pairs:
        Dk.cfg.pairs = a.members  # more independent sources (the kernel's parameters do not depend on P)
    P = Dk.cfg.pairs
    M = a.members or P
    if M < 2:
        raise SystemExit("the fair energy score needs at least 2 members (--members, or a kernel with 2 pairs)")
    # the samplers: n_tr steps while tuning, n_va in the validation (0: the path has no step count), and the graph
    line = Fm.cfg.path == "line"
    cut = line or Fm.cfg.core_sub > 0
    own = Fm.cfg.steps if line else Fm.cfg.core_sub
    n_tr, n_va = ((a.steps or own), (a.val_steps or own)) if cut else (0, 0)
    calls = 2 * n_tr if line else Fm.N * (2 * n_tr if cut else 1)                # network calls a member
    rows = a.batch * M
    units = max(min(a.chunk or rows, rows) * calls, rows if a.lam > 0 else 0)    # the sampler's graph, or the anchor's
    gb = units * net_call_mb(Fm, core.static.shape[-2:]) / 1024
    if a.max_graph_gb and gb > a.max_graph_gb:
        raise SystemExit(
            f"devtune: {units} network evaluations a lead in the graph ({min(a.chunk or rows, rows)} rows x {calls} calls a member), "
            f"about {gb:.1f} GB > --max-graph-gb {a.max_graph_gb}: use --chunk (rows a backward pass, the same gradient), "
            f"or lower --steps / --members / --batch, or raise --max-graph-gb (0 = no limit)")
    norm = make_norm(a.norm, Dk.keep, cl.lat, s=Dk.s, std=getattr(cl, "std", None), clim=a.clim, bmin=a.band_bmin).to(dev)
    grid = make_norm("grid", Dk.keep, cl.lat, eps=0.0).to(dev)
    jk = dict(jac=True) if (a.core_jac and not line) else {}
    W = int(getattr(core, "mem", 0))
    years = D.load(pa, a.train_years)
    va = D.load(pa, [a.val_year])
    tr, vr = D.Pairs(years, a.seed, hist=W, ahead=a.days), D.Pairs(va, a.seed + 1, hist=W, ahead=a.days)
    w = D.area_weights(cl.lat).to(dev)
    params = list(Fm.net.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4) if not kt else torch.optim.AdamW(
        [dict(params=params, lr=a.lr), dict(params=kpar, lr=a.kernel_lr)], weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
    dyn = torch.tensor(DYN)
    vx = [vr(a.batch) for _ in range(a.val_batches)]
    scale = np.ones(a.days)  # n_L, set by the first validation

    def save():
        cfg = dict(Fm.cfg.__dict__)
        if cfg["path"] == "line":  # the straight line's checkpoint format (no path keys)
            cfg.pop("path"), cfg.pop("core_steps"), cfg.pop("core_sub")
        keep = {k: v for k, v in Fm.meta.items() if k not in ("tuned_from", "tune")}   # dev_tag, core_tag, reflow_of
        files = []
        if kt:  # the tuned kernel first (the flow never names a kernel that is not there): the original's checkpoint
            kst = {k: v.detach().cpu().clone() for k, v in Dk.state_dict().items() if not k.startswith("core.")}
            files.append(({**ck_k, "state": kst, "tuned_from": dev0, "tune": dict(vars(a), band_weight=bw)},
                          os.path.join(a.out, f"{a.kernel_out}.pt")))
            keep["dev_tag"] = a.kernel_out
        save_atomic(*files, (dict(state=Fm.net.state_dict(), cfg=cfg, **keep, tuned_from=a.flow_tag,
                                  tune=dict(vars(a), scale=scale.tolist())), os.path.join(a.out, f"{a.tag}.pt")))

    def run(batch, gen=None, train=False):
        """Per lead (days, 10), batch means: ES, the anchor, |truth - xbar|, spread^2 and skill^2 (grid norm); and
        of the validation: ES of the tuning sampler, the mean squares (units of sigma) of its endpoint minus the
        validation sampler's and of the latter, the kinetic energy and the squared displacement of the latter.
        --kernel-lr: three more columns, the kernel's IGN, its mean entropy and (training) its anchor."""
        x, ys, doy, tns, hi = batch
        if ys.dim() == 4:
            ys, tns = ys[:, None], tns[:, None]
        xd, yd, doy, tns = x[:, dyn].to(dev), ys[:, :, dyn].to(dev), doy.to(dev), tns.to(dev)
        with torch.no_grad():
            e = Dk.start(xd, doy, None if hi is None else hi[:, :, dyn].to(dev), gen)
        xb, out = xd, []
        kg = kt and train        # the kernel's day with the graph of its parameters
        gk = dict(grad=True) if kg else {}
        for k in range(a.days):
            x0 = xb
            with torch.set_grad_enabled(kg):
                xb, e, H, _ = Dk.day(xb, e, doy + k, tns[:, k], k, gen)
                if kg:  # two stages: the flow's part on leaves fx, fe, fH; the day's own outputs kept for the second
                    kout = (xb, e, H)
                    xb, e, H = (t.detach() for t in kout)   # what the anchor of the network and the next day see
                    fx, fe, fH = leaves = [t.detach().requires_grad_(True) for t in kout]
                else:
                    fx, fe, fH = xb, e, H
                z1 = (yd[:, k] - fx) / Dk.s * Dk.keep
            sel = torch.randperm(P, generator=gen)[:M] if (train and M < P) else None
            c = Fm.day_ctx(x0, fx, fe, fH, doy, tns[:, k], k, gen, sel, **gk)
            wk = 1.0 / (float(scale[k]) * a.days)                     # the lead's weight of the score in the loss
            if train and a.chunk:   # two passes: the members, d ES / d members, then the graph of a.chunk rows at a time
                with torch.no_grad():
                    z = Fm.day_run(c, n_tr or None, **jk)
                zl = z.clone().requires_grad_(True)
                es = energy_score(zl, z1, norm)[0].mean()
                gz = torch.autograd.grad(es * wk, zl, retain_graph=kg)[0].flatten(0, 1)
                if kg:  # the truth's deviation moves with the kernel's mean: d ES / d xbar through z1
                    torch.autograd.backward(es * wk, inputs=[fx])
                for r in torch.arange(gz.shape[0], device=gz.device).split(a.chunk):  # (the context's graph is shared)
                    (Fm.day_run(Fm.ctx_rows(c, r), n_tr or None, grad=True, **jk)[0] * gz[r]).sum().backward(retain_graph=kg)
            else:
                z = Fm.day_run(c, (n_tr if train else n_va) or None, grad=train, **jk)   # (B, M, 66, H, W)
                es = energy_score(z, z1, norm)[0].mean()
                if train:  # the sampler's graph is freed here: before the anchor's is built, and before the next lead
                    (es * wk).backward()
            z, es, more = z.detach(), float(es.detach()), [0.0] * 5
            if not train:
                kin, disp = Fm.last_path
                ms = lambda t: float((t.flatten(0, 1) / c.sig).pow(2).mean())  # noqa: E731
                zt = Fm.day_run(c, n_tr or None, **jk) if n_tr != n_va else z
                more = [float(energy_score(zt, z1, norm)[0].mean()) if n_tr != n_va else es, ms(zt - z), ms(z), kin, disp]
            an = Fm.day_loss(x0, xb, e, H, yd[:, k], doy, tns[:, k], k, w, gen, sel) if (a.lam > 0 or not train) else None
            if train and an is not None:
                (a.lam * an / a.days).backward()
            if kg:  # second stage, once through the kernel's day: its own loss (the anchor) + the score's gradients
                ka = Dk.lead_loss(kout[2], kout[0], kout[1], yd[:, k], w, bw)[0]
                (a.kernel_lam * ka / a.days + sum((t * q.grad).sum() for t, q in zip(kout, leaves)
                                                 if t.requires_grad and q.grad is not None)).backward()
                more = more + [0.0, 0.0, float(ka.detach())]
            with torch.no_grad():
                if kt and not train:
                    more = more + [float(v) for v in Dk.ign(H, xb, yd[:, k], w)] + [0.0]
                sp, sk = spread_skill(z, z1, grid)
                out.append([es, 0.0 if an is None else float(an.detach()), float(norm(z1).mean()),
                            float(sp.mean()), float(sk.mean())] + more)
        return np.array(out)

    rd = lambda t: [round(float(q), 4) for q in t]  # noqa: E731

    def val():
        """The validation mean of ES_L / n_L, and what is logged of the batch means per lead."""
        Fm.eval()
        gv = torch.Generator().manual_seed(123)
        with torch.no_grad():
            r = np.mean([run(b, gv) for b in vx], 0)
        Fm.train()
        if first:  # n_L = E|truth - xbar| on the validation batches, once
            scale[:] = np.maximum(r[:, 2], 1e-30)
            first.clear()
        v = dict(es=rd(r[:, 0] / scale), es_raw=rd(r[:, 0]), anchor=rd(r[:, 1]),
                 spread_skill=rd(np.sqrt((P + 1) / P * r[:, 3] / np.maximum(r[:, 4], 1e-30))),
                 straight=rd(r[:, 8] / np.maximum(r[:, 9], 1e-30)))
        if n_tr != n_va:  # the tuning sampler on the validation draws: its score and its endpoint against the validation's
            v.update(es_tune=rd(r[:, 5] / scale), end_tune=rd(np.sqrt(r[:, 6] / np.maximum(r[:, 7], 1e-30))))
        if kt:  # the source: the kernel's Gaussian log score and mean entropy (IGN - H = 1/2: calibrated)
            v.update(kernel_ign=rd(r[:, 10]), kernel_H=rd(r[:, 11]))
        return float(np.mean(r[:, 0] / scale)), v

    t0 = time.time()
    first = [True]
    best, v0 = val()
    save()
    log(dict(devtune_val_start=v0, scale=rd(scale), norm=a.norm, members=M, pairs=P, flow=a.flow_tag,
             path=Fm.cfg.path, lam=a.lam, steps=n_tr, val_steps=n_va, graph_units=units, graph_gb=round(gb, 2),
             **(dict(kernel=dev0, kernel_out=a.kernel_out, kernel_lr=a.kernel_lr, kernel_lam=a.kernel_lam,
                     kernel_band_weight=bw, kernel_params=sum(p_.numel() for p_ in kpar)) if kt else {})))
    for it in range(1, a.iters + 1):
        opt.zero_grad()
        r = run(tr(a.batch), train=True)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        if kt:  # on their own: the kernel's gradient does not change the clipping of the network's
            torch.nn.utils.clip_grad_norm_(kpar, 1.0)
        opt.step()
        sch.step()
        if it % a.log_every == 0 or it == a.iters:
            sc, v = val()
            if sc < best:  # the checkpoint is the best validation energy score
                best = sc
                save()
            log(dict(it=it, es=rd(r[:, 0] / scale), anchor=round(float(r[:, 1].mean()), 4),
                     **(dict(kernel_anchor=round(float(r[:, 12].mean()), 4)) if kt else {}),
                     **{"val_" + k: q for k, q in v.items()}, best=round(best, 4), sec=round(time.time() - t0)))


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
    K = load_kernel(a, core, a.kernel_tag) if any(not x.startswith("dev") and x not in ("persistence", "core") for x in a.arms) \
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
    c.add_argument("--no-gfeat", action="store_true", help="the control without the physical field G(x)")
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
    d.add_argument("--cloud-mean", action="store_true", help="the mean = the average of the equation over the cloud")
    d.add_argument("--correct", action="store_true",
                   help="chained kernel p(x_t | x_(t-1)): correct the mean (G of the new state) and the entropy at x_t")
    d.add_argument("--energy-weight", type=float, default=0.0,
                   help="penalty on members whose daily change of total energy leaves the mean's by > k sigma")
    d.add_argument("--energy-k", type=float, default=1.0)
    d.add_argument("--noise-pre", action="store_true",
                   help="compose the noise with the equation: add it to the deviations before the equation's day")
    d.add_argument("--noise-mult", type=float, default=0.0,
                   help="the noise multiplies the day's tendency, (1 + a zeta) f, in place of the additive new part")
    d.add_argument("--e0-spec", default="", help="diag_spectra json: initial deviations shaped like its 1-day error")
    d.add_argument("--band-scale", action="store_true", help="entropy by scale: band-wise rescaling of the deviations")
    d.add_argument("--band-weight", type=float, default=0.0, help="weight of the band-wise log score (needs the SHT)")
    d.add_argument("--r-dissip", action="store_true", help="the learned residual r <= 0 (noise makes the positive part)")
    d.add_argument("--noise-spec", default="",
                   help="balanced noise: diag_spectra json, psi starts from its 1-day wind error spectrum (default --e0-spec)")
    d.add_argument("--bal-vert", type=float, default=1.0,
                   help="balanced noise: vertical correlation scale of psi / chi in ln p (0 = independent levels)")
    d.add_argument("--bal-lmin", type=int, default=1,
                   help="balanced noise: psi / chi have no power in the degrees l < this (2 = no solid-body rotation, "
                        "so the balanced wind carries no axial angular momentum). For the angular momentum only: the "
                        "l = 1 band of z and T is NOT changed (linear balance: Phi_1 comes from psi_2)")
    d.add_argument("--aam-fix", action="store_true",
                   help="balanced noise: take the axial angular momentum out of the day's whole new wind, per level "
                        "u - c cos(phi) (the remainder's u too)")
    d.add_argument("--fdt", default="", help="clim_bands.py json: the AR(1) rho per band is its one-day autocorrelation (fixed)")
    d.add_argument("--fdt-weight", type=float, default=0.0,
                   help="weight of the one-sided penalty relu(log V - log c)^2: band spread <= climatological band power")
    d.add_argument("--fdt-bmin", type=int, default=0,
                   help="first band of the fdt penalty (1 = without the band l = 0, the global mean)")
    d.add_argument("--noise", default="smooth", choices=["smooth", "sphere", "balanced"],
                   help="the new part's field: 3 x 3 smoothed, spherical harmonics with learned band amplitudes + AR(1), "
                        "or balanced (psi / chi per level: rotational wind, linear-balance Phi, hydrostatic T)")
    d.add_argument("--init", default="", help="start from this fitted deviation kernel")
    d.add_argument("--fresh-best", action="store_true",
                   help="--init: the checkpoint is the best validation of the TRAINED states (the start is written first "
                        "and replaced at the first validation); default: the start counts, and stays if none is better")
    d.add_argument("--through", type=int, default=0,
                   help="days of the gradient through the mean's path of the chain (a correction answers for later days)")
    d.add_argument("--seq-weight", type=float, default=0.0,
                   help="weight of the sequence term: the squared error of the sequence's mean over its mean variance")
    d.add_argument("--lr-correct", type=float, default=0.0, help="learning rate of the correction net (0 = --lr)")
    d.add_argument("--core-lr", type=float, default=0.0,
                   help="> 0: the core's trained weights learn along (0 = frozen), with this learning rate, through the "
                        "mean's path of --through K (K >= 1 required). The core learns from the mean's error (the log "
                        "score's mean term, the sequence term), not from the spread: the pairs of the deviations are "
                        "evaluated without gradient. Writes the tuned core --core-out; the kernel is recorded on it")
    d.add_argument("--core-anchor", type=float, default=0.0,
                   help="--core-lr: weight W of the pull to the original core, W sum((theta - theta0)^2) / sum(theta0^2); "
                        "its gradient is added after the clip of the loss's (compare the logged core_pull and core_grad)")
    d.add_argument("--core-out", default="",
                   help="--core-lr: tag of the tuned core's checkpoint (default <core-tag>_<tag>); never the core's own")
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
    df.add_argument("--sources", type=int, default=2, help="source draws (of the kernel's P) per IC and lead in the loss")
    df.add_argument("--iters", type=int, default=600)
    df.add_argument("--batch", type=int, default=4)
    df.add_argument("--lr", type=float, default=5e-4)
    df.add_argument("--hidden", type=int, default=64)
    df.add_argument("--steps", type=int, default=6)
    df.add_argument("--path", default="line", choices=["line", "core"],
                    help="line: the straight line in a pseudo time; core: the tendency core's flow over the day "
                         "(fpk.coreflow), the network adds the residual on the day's grid")
    df.add_argument("--core-steps", type=int, default=4,
                    help="--path core: steps of the day's grid (Heun steps of the core's tendency over 1/N day); "
                         "0 = the core's own substeps (the kernel's day itself: zero head = the kernel)")
    df.add_argument("--core-sub", type=int, default=0,
                    help="--path core: 0 = one Euler step of the learned part per grid step (under-dispersed: a perfect "
                         "network gives 0.72 of the spread at 4 steps, a point at 1); M >= 1 = the learned part of each "
                         "grid step is a line flow after the core's step, M Heun steps (2 M network calls)")
    df.add_argument("--val-batches", type=int, default=4)
    df.add_argument("--log-every", type=int, default=50)
    df.add_argument("--reflow", default="",
                    help="reflow: the tag of a trained deviation flow on --dev-tag (the teacher). The student starts "
                         "from it and learns its (source, transported endpoint) pairs: straighter paths, the same "
                         "endpoints; give the teacher's --hidden / --source / --path options")
    dt = sub.add_parser("devtune")
    common(dt)
    dt.add_argument("--flow-tag", required=True, help="the trained deviation flow to tune (line or core path)")
    dt.add_argument("--tag", default="", help="the tuned flow (default <flow-tag>_es); it must start with 'devflow'")
    dt.add_argument("--norm", default="grid", choices=["grid", "band", "energy"],
                    help="the norm of the energy score (fpk.escore): area-weighted RMS in units of s, the "
                         "band-weighted norm of diag_spectra, or the dry + moist total-energy norm")
    dt.add_argument("--clim", default="docs/results/clim_bands.json",
                    help="--norm band: clim_bands.py json, its climatological band power c")
    dt.add_argument("--band-bmin", type=int, default=1,
                    help="--norm band: the first band that counts. Default 1 = without the band l = 0: that band is "
                         "the global mean (the core's own mass drift, and a tiny climatological power, so it would "
                         "dominate the norm); 0 = all bands, diag_spectra's full norm")
    dt.add_argument("--members", type=int, default=0,
                    help="members (source draws, the +e of the kernel's pairs) per IC in the score: 0 = the kernel's P; "
                         "fewer = a random choice per lead; more = the kernel runs that many pairs")
    dt.add_argument("--lam", type=float, default=0.5, help="weight of the flow-matching loss (anchor)")
    dt.add_argument("--iters", type=int, default=300)
    dt.add_argument("--lr", type=float, default=1e-4)
    dt.add_argument("--steps", type=int, default=2,
                    help="sampler steps during tuning (2 x steps network calls a member in the graph); 0 = the flow's "
                         "own. Default 2: the flow's own 6 steps hold 3 x the graph")
    dt.add_argument("--val-steps", type=int, default=0,
                    help="sampler steps of the validation (it picks the checkpoint); 0 = the flow's own (score "
                         "samples with its own --steps, default 6)")
    dt.add_argument("--chunk", type=int, default=0,
                    help="source rows a backward pass: the same gradient in two passes, with the graph of this many "
                         "rows at a time (memory = rows x calls network evaluations, one more sampler pass without "
                         "a graph); 0 = one pass with all batch x members rows in the graph")
    dt.add_argument("--max-graph-gb", type=float, default=2.5,
                    help="refuse to start when the estimated graph of a lead (logged as graph_gb) is larger; 0 = no limit")
    dt.add_argument("--core-jac", action="store_true",
                    help="core path: keep the equation's Jacobian of the core steps in the gradient (default: identity)")
    dt.add_argument("--kernel-lr", type=float, default=0.0,
                    help="> 0: the kernel's own parameters learn along (0 = frozen) with this learning rate, from the "
                         "members' energy score through the sampler (the source e_L, the conditioning's xbar_L and H_L); "
                         "the flow-matching anchor (--lam) is kept away from them, it would collapse the source")
    dt.add_argument("--kernel-lam", type=float, default=0.5,
                    help="--kernel-lr: weight of the kernel's own loss of the lead (IGN + band-weight x band IGN, as devk) "
                         "as the kernel's anchor")
    dt.add_argument("--kernel-out", default="",
                    help="--kernel-lr: tag of the tuned kernel's checkpoint (default <the flow's kernel>_j); never the "
                         "kernel's own. The tuned flow is recorded on it")
    dt.add_argument("--kernel-band-weight", type=float, default=-1.0,
                    help="--kernel-lr: the band weight of the kernel's anchor; < 0 (default) = the one stored with the "
                         "kernel's training options, else 0.5 for a kernel with band_scale and 0 without")
    dt.add_argument("--days", type=int, default=10)
    dt.add_argument("--batch", type=int, default=2)
    dt.add_argument("--val-batches", type=int, default=4)
    dt.add_argument("--log-every", type=int, default=25)
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
    s.add_argument("--cloud-mean", action="store_true", help="devk arms: the cloud's mean (also for a kernel fitted without)")
    s.add_argument("--kernel-entropy", action="store_true", help="also the kernel's entropy along every trajectory")
    a, rest = p.parse_known_args(argv)
    rest = [r for r in rest if r != "--"]
    if a.threads:
        torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    {"core": cmd_core, "kernel": cmd_kernel, "flow": cmd_flow, "tune": cmd_tune, "devk": cmd_devk, "devflow": cmd_devflow,
     "devtune": cmd_devtune, "score": cmd_score}[a.cmd](a, rest)


if __name__ == "__main__":
    main()
