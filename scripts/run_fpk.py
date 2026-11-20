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
from fpk.env import DYN, TISR, pgfm_args, rkt  # noqa: E402
from fpk.ensemble import Scorer, rollout  # noqa: E402
from fpk.flow import FlowConfig, KernelFlow  # noqa: E402
from fpk.kernel import KernelConfig, PhysicsKernel  # noqa: E402


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
    if getattr(core, "mem", 0) or getattr(core, "hist", False) or getattr(core, "entv", False):
        raise SystemExit("the kernel is built on a plain core (gfeat allowed, no memory / history / entropy state)")
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
    tr, vr = D.Pairs(years, a.seed), D.Pairs(va, a.seed + 1)
    w = D.area_weights(cl.lat).to(dev)
    K = PhysicsKernel(core, KernelConfig(n_centres=a.centres, hidden=a.hidden)).to(dev)
    opt = torch.optim.AdamW([p for p in K.parameters() if p.requires_grad], lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
    vx = [vr(a.batch) for _ in range(4)]
    dyn = torch.tensor(DYN)

    def val():
        K.eval()
        tot = 0.0
        with torch.no_grad():
            for x, y, doy, tn in vx:
                c, h, _, _ = K(x[:, dyn].to(dev), doy.to(dev), tn.to(dev))
                tot += float(K.nll(c, h, y[:, dyn].to(dev), w))
        K.train()
        return tot / len(vx)

    t0 = time.time()
    log(dict(kernel_val_nll_start=round(val(), 4)))
    for it in range(1, a.iters + 1):
        x, y, doy, tn = tr(a.batch)
        c, h, _, _ = K(x[:, dyn].to(dev), doy.to(dev), tn.to(dev))
        loss = K.nll(c, h, y[:, dyn].to(dev), w)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(K.parameters(), 1.0)
        opt.step()
        sch.step()
        if it % a.log_every == 0 or it == a.iters:
            log(dict(it=it, nll=round(loss.item(), 4), val_nll=round(val(), 4), eps=round(float(K.eps().mean()), 4),
                     h=round(float(h.mean()), 4), sec=round(time.time() - t0)))
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
    tr, vr = D.Pairs(years, a.seed), D.Pairs(va, a.seed + 1)
    w = D.area_weights(cl.lat).to(dev)
    params = [p for p in Fm.net.parameters()]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.iters, pct_start=0.05)
    vx = [vr(a.batch) for _ in range(4)]
    dyn = torch.tensor(DYN)
    gv = torch.Generator().manual_seed(123)

    def val():
        Fm.eval()
        gv.manual_seed(123)
        tot, err = 0.0, 0.0
        with torch.no_grad():
            for x, y, doy, tn in vx:
                xd, yd = x[:, dyn].to(dev), y[:, dyn].to(dev)
                tot += float(Fm.loss(xd, yd, doy.to(dev), tn.to(dev), w, gv))
        Fm.train()
        return tot / len(vx)

    t0 = time.time()
    log(dict(flow_val_start=round(val(), 4), source=a.source))
    for it in range(1, a.iters + 1):
        x, y, doy, tn = tr(a.batch)
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


# --------------------------------------------------------------------------------------------------------- score
def cmd_score(a, rest):
    pa, cl, core = load_core(a)
    dev = torch.device(a.device)
    K = load_kernel(a, core, a.kernel_tag)
    te = D.load(pa, [a.test_year])[a.test_year]
    L, M = a.leads, a.members
    ics = np.arange(a.ic_start, len(te) - L, a.ic_stride)[: a.max_ics or None]
    w = np.cos(np.deg2rad(cl.lat))
    w = (w / w.mean()).astype(np.float64)
    dyn = np.asarray(DYN)
    res, t_all = {}, time.time()
    for arm in a.arms:
        g = torch.Generator().manual_seed(a.seed)
        if arm == "core":
            step, m = (lambda x, d, tn: K.phi(x, d, tn)), 1
        elif arm == "kernel":
            step, m = (lambda x, d, tn: K.core.finish(K.sample(x, d, tn, g)[0], tn)), M
        elif arm == "persistence":
            step, m = (lambda x, d, tn: x), 1
        else:
            Fm = load_flow(a, K, arm)
            step, m = (lambda x, d, tn, Fm=Fm: Fm.sample(x, d, tn, g, steps=a.steps)[0]), M
        sc = Scorer(L, w, cl.inc, cl.std)
        t0 = time.time()
        for i in range(0, len(ics), a.ic_batch):
            j = ics[i:i + a.ic_batch]
            x0 = torch.from_numpy(te[j][:, dyn]).to(dev)
            tis = torch.from_numpy(np.stack([te[k + 1:k + 1 + L, TISR] for k in j])).to(dev)
            truth = np.stack([te[k + 1:k + 1 + L][:, dyn] for k in j])
            ens, hk = rollout(step, x0, torch.tensor(j, device=dev), tis, m, L,
                              kernel=K if (m > 1 and a.kernel_entropy) else None, gen=g)
            sc.update(ens.cpu().numpy().astype(np.float64), truth.astype(np.float64),
                      None if hk is None else hk.cpu().numpy())
        res[arm] = sc.summary()
        res[arm]["members"], res[arm]["sec"] = m, round(time.time() - t0)
        log({arm: {k: res[arm][k] for k in ("nrmse", "crps", "spread_skill", "entropy", "ign") if k in res[arm]}})
    # climatology as a reference (z-scored daily climatology of the training years)
    clim_z = ((cl.clim - cl.mean[None, :, None, None]) / (cl.std[None, :, None, None] + 1e-8)).astype(np.float64)
    sc = Scorer(L, w, cl.inc, cl.std)
    for i in range(0, len(ics), a.ic_batch):
        j = ics[i:i + a.ic_batch]
        c = np.stack([clim_z[np.minimum(np.arange(k + 1, k + 1 + L), 365)][:, dyn] for k in j])[:, None]
        truth = np.stack([te[k + 1:k + 1 + L][:, dyn] for k in j]).astype(np.float64)
        sc.update(c, truth)
    res["climatology"] = sc.summary()
    meta = dict(test_year=a.test_year, ics=len(ics), ic_stride=a.ic_stride, members=M, leads=L, steps=a.steps,
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
    {"core": cmd_core, "kernel": cmd_kernel, "flow": cmd_flow, "score": cmd_score}[a.cmd](a, rest)


if __name__ == "__main__":
    main()
