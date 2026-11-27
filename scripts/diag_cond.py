"""What the last condition on the kernel's mean should be (감자 2026-11-24 19:37 KST: z500 10 days toward 712).

The invariant-measure condition (diag_projection.py) reads the mean as  clim + sum_b g_b f_b  with one strength per
Littlewood-Paley band, lead and channel, fitted on 2016. Candidates for what the strength may further depend on, or what
else the mean may be conditioned on, are tested here on the cores' trajectories (no training):

    dump   the fields: every core's forecast anomaly, the truth anomaly and the initial anomaly per IC (z-scored, f16),
           one memmap per (year, field) in --dir; the years the cores never saw are 2016, 2020 (fit) and 2017 (test)
    fit    (scripts/cond_fit.py) the candidates, fitted on the fit years and scored on 2017

    python scripts/diag_cond.py --cores core_gngmz8W,core_ngmz8W,core_gngmz8L,core_gngmz8W_c --year 2016 --ic-stride 1
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
ap.add_argument("--ic-stride", type=int, default=1)
ap.add_argument("--dir", default="runs/fpk/cond")
ap.add_argument("--batch", type=int, default=4)
A, PASS = ap.parse_known_args()

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
N, C = len(ics), len(dyn)
pre = f"{A.dir}/{A.year}_"
mm = lambda name, shape: np.lib.format.open_memmap(pre + name + ".npy", mode="w+", dtype=np.float16, shape=shape)  # noqa: E731
F = {t: mm(t, (N, L, C, H, W)) for t in tags}
O = mm("o", (N, L, C, H, W))
X0 = mm("x0", (N, C, H, W))
t0 = time.time()
for i in range(0, N, A.batch):
    j = ics[i:i + A.batch]
    truth = np.stack([te[k + 1:k + 1 + L][:, dyn] for k in j])
    clim = np.stack([(cl.clim[np.minimum(np.arange(k + 1, k + 1 + L), 365)][:, dyn] - cm) / cs for k in j]).astype(np.float32)
    O[i:i + len(j)] = truth - clim
    X0[i:i + len(j)] = np.stack([te[k][dyn] - ((cl.clim[min(k, 365)][dyn] - cm) / cs) for k in j])
    tis = torch.from_numpy(np.stack([te[k + 1:k + 1 + L, R.TISR] for k in j])).to(dev)
    for s in S:
        core, K = s["core"], s["K"]
        Wm = int(getattr(core, "mem", 0))
        hist = torch.from_numpy(np.stack([te[np.maximum(np.arange(k - Wm, k), 0)][:, dyn] for k in j])).to(dev) if Wm else None
        with torch.no_grad():
            ens, _ = rollout(lambda x, d, tn: R.core_step(K.core, x, d, tn), torch.from_numpy(te[j][:, dyn]).to(dev),
                             torch.tensor(j, device=dev), tis, 1, L, hist=hist, start=K.start)
        F[s["a"].core_tag][i:i + len(j)] = ens[:, 0].float().cpu().numpy() - clim
    if (i // A.batch) % 10 == 0:
        print(json.dumps(dict(year=A.year, ics=int(i + len(j)), of=N, sec=round(time.time() - t0),
                              rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)), flush=True)
for v in list(F.values()) + [O, X0]:
    v.flush()
lat = np.asarray(cl.lat)
json.dump(dict(year=A.year, ics=ics.tolist(), cores=tags, leads=int(L), std=cl.std[dyn].tolist(), lat=lat.tolist(),
               channels=[contract.CHANNEL_NAMES[c] for c in DYN], ic_stride=A.ic_stride), open(pre + "meta.json", "w"))
print("done", N, round(time.time() - t0))
