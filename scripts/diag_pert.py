"""The Chapman-Kolmogorov condition on the kernel's mean, tested on the cores (no training).

The kernel's L-day law today is the core trajectory plus deviations: its mean is Phi^L(x0). If the L-day kernel is the
composition of the one-day kernels  K(x, .) = N(Phi(x), Sigma)  (the semigroup / Chapman-Kolmogorov condition), the
members go back through the core every day and the mean moves by the noise-induced drift  E Phi(x + eps) - Phi(x)
(1/2 tr Hess Phi Sigma, the Fokker-Planck drift). Here: members through the core,

    ic     x0 + a delta                      (the uncertainty of the initial state only)
    day    every day  x_{k+1} = Phi(x_k) + a delta_k      (the one-day kernel, composed)

delta = one-day errors of the same core on OTHER years (pool years; centred), in antithetic pairs (+delta, -delta) so the
first-order sampling noise cancels. Writes the ensemble mean anomaly per core in the layout of diag_cond.py (its own
--dir with o, x0 and meta of the chosen ICs), plus the sampling variance of the mean (area mean) in <dir>/<year>_sv.npz.

    python scripts/diag_pert.py --year 2016 --ic-stride 6 --mode day --alpha 0.5 --pool 2020 --dir runs/fpk/cond_day05
"""
import argparse
import json
import os
import resource
import sys
import time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--cores", default="core_gngmz8W,core_ngmz8W")
ap.add_argument("--year", type=int, default=2016)
ap.add_argument("--ic-stride", type=int, default=6)
ap.add_argument("--members", type=int, default=8)
ap.add_argument("--mode", choices=["ic", "day"], default="day")
ap.add_argument("--alpha", type=float, default=0.5)
ap.add_argument("--pool", type=int, nargs="+", default=[2020])
ap.add_argument("--src", default="runs/fpk/cond", help="the stride-1 dumps of diag_cond.py (errors of the pool years)")
ap.add_argument("--dir", required=True)
ap.add_argument("--batch", type=int, default=2)
ap.add_argument("--seed", type=int, default=0)
A, PASS = ap.parse_known_args()
assert A.members % 2 == 0
assert A.year not in A.pool, "the errors must come from other years"

sys.path.insert(0, os.path.dirname(__file__))
import run_fpk as R  # noqa: E402
from fpk.ensemble import rollout  # noqa: E402
from fpk.env import DYN, contract  # noqa: E402

tags = A.cores.split(",")
os.makedirs(A.dir, exist_ok=True)
S = []


def grab(a, rest):
    pa, cl, core = R.load_core(a)
    S.append(dict(a=a, pa=pa, cl=cl, core=core, K=R.PhysicsKernel(core)))


R.cmd_score = grab
for t in tags:
    R.main(["score", "--core-tag", t, "--arms", "core", "--threads", "4"] + PASS + ["--test-year", str(A.year), "--ic-stride", str(A.ic_stride)])
a0, cl, pa = S[0]["a"], S[0]["cl"], S[0]["pa"]
dev = torch.device(a0.device)
te = R.D.load(pa, [A.year])[A.year]
L = a0.leads
ics = np.arange(a0.ic_start, len(te) - L, a0.ic_stride)
dyn = np.asarray(DYN)
eps = R.rkt().EPS
cm, cs = cl.mean[dyn][:, None, None], cl.std[dyn][:, None, None] + eps
H, W = te.shape[-2:]
N, C, M, P2 = len(ics), len(dyn), A.members, A.members // 2
keep = torch.ones(1, C, 1, 1)
keep[:, [contract.CHANNEL_NAMES[c] for c in DYN].index("tisr")] = 0                   # tisr is a known forcing

# the pools of one-day errors per core (other years), centred
POOL = {}
for t in tags:
    e = np.concatenate([np.asarray(np.load(f"{A.src}/{y}_{t}.npy", mmap_mode="r")[:, 0], dtype=np.float32)
                        - np.asarray(np.load(f"{A.src}/{y}_o.npy", mmap_mode="r")[:, 0], dtype=np.float32) for y in A.pool])
    POOL[t] = torch.from_numpy(e - e.mean(0, keepdims=True)) * keep
print(json.dumps(dict(pool={t: [int(v.shape[0]), round(float(v.pow(2).mean().sqrt()), 4)] for t, v in POOL.items()})), flush=True)

pre = f"{A.dir}/{A.year}_"
mm = lambda name, shape: np.lib.format.open_memmap(pre + name + ".npy", mode="w+", dtype=np.float16, shape=shape)  # noqa: E731
F = {t: mm(t, (N, L, C, H, W)) for t in tags}
O = mm("o", (N, L, C, H, W))
X0 = mm("x0", (N, C, H, W))
SV = {t: np.zeros((N, L, C)) for t in tags}
SP = {t: np.zeros((N, L, C)) for t in tags}
lat = np.asarray(cl.lat)
w = np.cos(np.deg2rad(lat))
w = torch.tensor(w / w.mean(), dtype=torch.float32)[:, None]
gen = torch.Generator().manual_seed(A.seed)
t0 = time.time()
for i in range(0, N, A.batch):
    j = ics[i:i + A.batch]
    B = len(j)
    truth = np.stack([te[k + 1:k + 1 + L][:, dyn] for k in j])
    clim = np.stack([(cl.clim[np.minimum(np.arange(k + 1, k + 1 + L), 365)][:, dyn] - cm) / cs for k in j]).astype(np.float32)
    O[i:i + B] = truth - clim
    X0[i:i + B] = np.stack([te[k][dyn] - ((cl.clim[min(k, 365)][dyn] - cm) / cs) for k in j])
    tis = torch.from_numpy(np.stack([te[k + 1:k + 1 + L, R.TISR] for k in j])).to(dev)
    for s in S:
        core, K, t = s["core"], s["K"], s["a"].core_tag
        pool = POOL[t]

        def draw():
            d = pool[torch.randint(0, pool.shape[0], (B, P2), generator=gen)]              # (B, P2, C, H, W)
            return (A.alpha * torch.stack([d, -d], 2)).reshape(B * M, C, H, W)            # antithetic pairs

        state = dict(day=0)

        def step(x, d, tn):
            if A.mode == "ic" and state["day"] == 0:
                x = x + draw()
            y = R.core_step(K.core, x, d, tn)
            state["day"] += 1
            if A.mode == "day":
                y = y + draw()
            return y

        Wm = int(getattr(core, "mem", 0))
        hist = torch.from_numpy(np.stack([te[np.maximum(np.arange(k - Wm, k), 0)][:, dyn] for k in j])).to(dev) if Wm else None
        with torch.no_grad():
            ens, _ = rollout(step, torch.from_numpy(te[j][:, dyn]).to(dev), torch.tensor(j, device=dev), tis, M, L,
                             hist=hist, start=K.start)                                    # (B, M, L, C, H, W)
        ens = ens.float().cpu()
        mean = ens.mean(1)
        pm = ens.view(B, P2, 2, L, C, H, W).mean(2)                                       # the pair means
        SV[t][i:i + B] = ((pm.var(1) / P2) * w).mean((-2, -1)).double().numpy()
        SP[t][i:i + B] = ((ens.var(1)) * w).mean((-2, -1)).double().numpy()
        F[t][i:i + B] = mean.numpy() - clim
    if (i // A.batch) % 5 == 0:
        print(json.dumps(dict(year=A.year, ics=int(i + B), of=N, sec=round(time.time() - t0),
                              rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)), flush=True)
for v in list(F.values()) + [O, X0]:
    v.flush()
np.savez(pre + "sv.npz", **{f"sv_{t}": SV[t] for t in tags}, **{f"spread_{t}": SP[t] for t in tags})
json.dump(dict(year=A.year, ics=ics.tolist(), cores=tags, leads=int(L), std=cl.std[dyn].tolist(), lat=lat.tolist(),
               channels=[contract.CHANNEL_NAMES[c] for c in DYN], ic_stride=A.ic_stride, mode=A.mode, alpha=A.alpha,
               members=M, pool=A.pool), open(pre + "meta.json", "w"))
print("done", N, round(time.time() - t0))
