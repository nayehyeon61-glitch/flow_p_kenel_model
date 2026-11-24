"""Mixing the cores inside the rollout (감자 2026-11-25 00:26 KST: the last shot, neither a new core nor a kernel).

Now each core runs its ten days alone and the two forecasts are averaged at the end: (K_W^10 + K_NW^10) / 2. Here the
states are mixed every day, the composition of the mixed kernel's mean: ((K_W + K_NW) / 2)^10. With a relaxation beta,

    m = (x_W + x_NW) / 2,   x_W <- core_W(x_W + beta (m - x_W)),   x_NW <- core_NW(x_NW + beta (m - x_NW)),

beta = 0 is the forecast of now and beta = 1 the full mixture of the states (both cores go on from the mean; a memory
core reads the difference to its own forecast as an innovation, as it does for a corrected state). The dumps have the
layout of scripts/diag_cond.py (the average of the two trajectories as one "core" il<100 beta>) in --dir, next to the
slices of the dumps of the cores alone (--src, the same ICs), for scripts/cond_fit.py:

    python scripts/diag_inloop.py --year 2017 --betas 1 0.5
    python scripts/cond_fit.py --dir runs/fpk/cond_il --cores il100,il100 --exps bands
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
ap.add_argument("--year", type=int, default=2017)
ap.add_argument("--ic-stride", type=int, default=6)
ap.add_argument("--betas", type=float, nargs="+", default=[1.0])
ap.add_argument("--from-day", type=int, default=1, help="the states are mixed from this day of the rollout on")
ap.add_argument("--until-day", type=int, default=0, help="and up to this day (0: to the end); after it the cores run alone")
ap.add_argument("--dir", default="runs/fpk/cond_il")
ap.add_argument("--src", default="runs/fpk/cond", help="the dumps of the cores alone (sliced to the same ICs)")
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--max-ics", type=int, default=0)
A, PASS = ap.parse_known_args()

sys.path.insert(0, os.path.dirname(__file__))
import run_fpk as R  # noqa: E402
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
ics = np.arange(a0.ic_start, len(te) - L, A.ic_stride)[:A.max_ics or None]
dyn = np.asarray(DYN)
eps = R.rkt().EPS
cm, cs = cl.mean[dyn][:, None, None], cl.std[dyn][:, None, None] + eps
H, W = te.shape[-2:]
N, C = len(ics), len(dyn)
pre = f"{A.dir}/{A.year}_"
names = {b: "il%03d" % round(100 * b) + ("" if A.from_day == 1 else f"f{A.from_day}") + (f"u{A.until_day}" if A.until_day else "") for b in A.betas}
F = {b: np.lib.format.open_memmap(pre + names[b] + ".npy", mode="w+", dtype=np.float16, shape=(N, L, C, H, W)) for b in A.betas}
t0 = time.time()
for i in range(0, N, A.batch):
    j = ics[i:i + A.batch]
    clim = np.stack([(cl.clim[np.minimum(np.arange(k + 1, k + 1 + L), 365)][:, dyn] - cm) / cs for k in j]).astype(np.float32)
    tis = torch.from_numpy(np.stack([te[k + 1:k + 1 + L, R.TISR] for k in j])).to(dev)
    x0 = torch.from_numpy(te[j][:, dyn]).to(dev)
    doy = torch.tensor(j, device=dev)
    for b in A.betas:
        xs = []
        for s in S:                                                                   # every core's memory from the days before
            Wm = int(getattr(s["core"], "mem", 0))
            hist = torch.from_numpy(np.stack([te[np.maximum(np.arange(k - Wm, k), 0)][:, dyn] for k in j])).to(dev) if Wm else None
            s["K"].start(x0, doy, hist, 1)
            xs.append(x0)
        out = []
        with torch.no_grad():
            for l in range(L):
                if l + 1 >= A.from_day and l > 0 and (not A.until_day or l <= A.until_day):   # the state of day l; day 0: the same state
                    m = sum(xs) / len(xs)
                    xs = [x + b * (m - x) for x in xs]
                xs = [R.core_step(s["K"].core, x, doy + l, tis[:, l]) for s, x in zip(S, xs)]
                out.append(sum(xs) / len(xs))
        F[b][i:i + len(j)] = torch.stack(out, 1).float().cpu().numpy() - clim
    if (i // A.batch) % 5 == 0:
        print(json.dumps(dict(year=A.year, ics=int(i + len(j)), of=N, sec=round(time.time() - t0),
                              rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)), flush=True)
for v in F.values():
    v.flush()
# the cores alone, the truth and the initial anomaly: the same ICs out of the dumps at hand
sm = json.load(open(f"{A.src}/{A.year}_meta.json"))
row = {k: r for r, k in enumerate(sm["ics"])}
sel = np.array([row[int(k)] for k in ics])
for t in tags + ["o", "x0"]:
    if not os.path.exists(pre + t + ".npy"):
        np.save(pre + t + ".npy", np.load(f"{A.src}/{A.year}_{t}.npy", mmap_mode="r")[sel])
mf = pre + "meta.json"
old = json.load(open(mf)) if os.path.exists(mf) else {}
cores = list(dict.fromkeys(old.get("cores", tags) + list(names.values())))
json.dump(dict(year=A.year, ics=ics.tolist(), cores=cores, leads=int(L), std=cl.std[dyn].tolist(), lat=np.asarray(cl.lat).tolist(),
               channels=[contract.CHANNEL_NAMES[c] for c in DYN], ic_stride=A.ic_stride), open(mf, "w"))
print("done", N, round(time.time() - t0))
