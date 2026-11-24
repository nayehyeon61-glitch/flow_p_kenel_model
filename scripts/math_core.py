"""Mathematical cores (감자 2026-11-24 20:37 KST: "core 가 물리, 데이터잖아요? 완전히 수학적 core 로 해서 둘이 cover 하지
못한 부분을 cover 한 core 로"). No equation and no network: the transfer (Koopman) operator of the atmosphere estimated
from the record itself (2000-2015, the cores' training years), in the large scales (spherical harmonics l < --lt):

    lim   the operator on linear observables (a linear inverse model): for every lead l the least-squares map
          P_{t+l} = A_l [P_t, P_{t-1}, ...]  of the principal components (ridge chosen on 2014-2015, then refit on all)
    kaf   the empirical transfer operator / kernel analog forecast: K(x, .) = sum_j w_j(x) delta_{x_{j+1}} with Gaussian
          weights on the k nearest delay-embedded states of the record within +-window days of the season; the
          forecast is the mean of their futures (the Markov kernel of the data, no model at all)

Writes <dir>/<year>_<tag>.npy (N, 10, 66, H, W) f16, anomalies at the ICs of <dir>/<year>_meta.json (diag_cond.py), so
cond_fit.py can read it as one more core:

    python scripts/math_core.py --kind kaf --tag kaf --years 2016 2020 2017
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--kind", choices=["lim", "kaf"], required=True)
ap.add_argument("--tag", default="")
ap.add_argument("--years", type=int, nargs="+", default=[2016, 2020, 2017])
ap.add_argument("--dir", default="runs/fpk/cond")
ap.add_argument("--lt", type=int, default=16, help="degrees l < lt")
ap.add_argument("--npc", type=int, default=200, help="lim: PCs of today")
ap.add_argument("--npc-lag", type=int, default=50, help="lim: PCs of each day before")
ap.add_argument("--lags", type=int, default=2, help="lim: days before today in the input; kaf: delay embedding")
ap.add_argument("--q", type=int, default=40, help="kaf: PCs in the distance")
ap.add_argument("--k", type=int, default=50, help="kaf: neighbours")
ap.add_argument("--window", type=int, default=30, help="kaf: +- days of the season")
ap.add_argument("--threads", type=int, default=4)
A, PASS = ap.parse_known_args()
A.tag = A.tag or A.kind
torch.set_num_threads(A.threads)

sys.path.insert(0, os.path.dirname(__file__))
import run_fpk as R  # noqa: E402
from fpk.env import DYN  # noqa: E402
from fpk.sphere import SHT  # noqa: E402

G = {}


def grab(a, rest):
    G["a"], G["pa"] = a, R.pargs(a)
    G["cl"] = R.D.climate(G["pa"])


R.cmd_score = grab
R.main(["score", "--core-tag", "core_gngmz8W", "--arms", "core", "--threads", str(A.threads)] + PASS)
a, pa, cl = G["a"], G["pa"], G["cl"]
dyn = np.asarray(DYN)
eps = R.rkt().EPS
cm, cs = cl.mean[dyn][:, None, None], cl.std[dyn][:, None, None] + eps
CZ = ((cl.clim[:, dyn] - cm) / cs).astype(np.float32)                                 # (366, 66, H, W) climatology, z
sht = SHT(cl.lat)
LT = A.lt
keep = torch.zeros(sht.L, sht.ana.shape[0], dtype=torch.bool)
for l in range(LT):
    keep[l, : l + 1] = True
KI = keep.flatten().nonzero()[:, 0]
im0 = torch.zeros(sht.L, sht.ana.shape[0], dtype=torch.bool)
im0[:, 0] = True                                                                       # the imaginary part of m = 0 is 0
KIm = (keep & ~im0).flatten().nonzero()[:, 0]
C = len(dyn)
DIM = C * (len(KI) + len(KIm))


def to_vec(f):
    """(D, 66, H, W) anomaly -> (D, DIM): the real coefficients l < lt."""
    out = []
    for i in range(0, len(f), 64):
        c = sht.analysis(torch.from_numpy(np.ascontiguousarray(f[i:i + 64]))).flatten(-2)   # (b, 66, L*M) complex
        out.append(torch.cat([c.real[..., KI], c.imag[..., KIm]], -1).flatten(1))
    return torch.cat(out)


def to_field(v):
    """(B, DIM) -> (B, 66, H, W)."""
    B = v.shape[0]
    v = v.view(B, C, -1)
    re = torch.zeros(B, C, sht.L * sht.ana.shape[0])
    im = torch.zeros_like(re)
    re[..., KI] = v[..., : len(KI)]
    im[..., KIm] = v[..., len(KI):]
    c = torch.complex(re, im).view(B, C, sht.L, sht.ana.shape[0])
    return sht.synthesis(c)


def anomalies(year):
    te = R.D.load(pa, [year])[year]
    x = np.asarray(te[:, dyn], dtype=np.float32)
    return x - CZ[np.minimum(np.arange(len(x)), 365)], len(x)


t0 = time.time()
ARC, DOY, YR = [], [], []
for y in a.train_years:
    f, n = anomalies(y)
    ARC.append(to_vec(f))
    DOY.append(np.arange(n))
    YR.append(np.full(n, y))
    del f
    print(json.dumps(dict(archive=y, days=n, sec=round(time.time() - t0))), flush=True)
starts = np.cumsum([0] + [len(d) for d in DOY])
V = torch.cat(ARC)                                                                      # (N, DIM)
DOY, YR = np.concatenate(DOY), np.concatenate(YR)
del ARC
U, S, Vh = torch.linalg.svd(V - V.mean(0), full_matrices=False)
NP = max(A.npc, A.q)
E = Vh[:NP]                                                                            # (NP, DIM) the EOFs
P = V @ E.T                                                                            # (N, NP) principal components
ev = (S[:NP] ** 2 / (S ** 2).sum()).cumsum(0)
print(json.dumps(dict(dim=DIM, explained={k: round(float(ev[k - 1]), 3) for k in (10, 40, 100, 200) if k <= NP}, sec=round(time.time() - t0))), flush=True)
LEADS = 10


def valid(t_lo, t_hi):
    """Archive rows t whose t - lags and t + 10 stay inside the same year."""
    rows = []
    for s, e in zip(starts[:-1], starts[1:]):
        rows.append(np.arange(s + t_lo, e - t_hi))
    return np.concatenate(rows)


def lim_inputs(Pm, t):
    """[P_t[:npc], P_{t-1}[:npc_lag], ...] for rows t of a PC matrix Pm (clamped at the start)."""
    parts = [Pm[t, : A.npc]] + [Pm[np.maximum(t - j, 0), : A.npc_lag] for j in range(1, A.lags + 1)]
    return torch.cat(parts + [torch.ones(len(t), 1)], 1)


if A.kind == "lim":
    rows = valid(A.lags, LEADS)
    X = lim_inputs(P, rows)
    hold = torch.from_numpy(np.isin(YR[rows], [a.train_years[-2], a.train_years[-1]]))

    def fit(m, lam):
        XtX = X[m].T @ X[m]
        Mi = torch.linalg.inv(XtX + lam * torch.trace(XtX) / XtX.shape[0] * torch.eye(XtX.shape[0]))
        return [Mi @ (X[m].T @ V[rows[m.numpy()] + l]) for l in range(1, LEADS + 1)]      # 10 x (k, DIM)

    best = None
    for lam in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        W_ = fit(~hold, lam)
        rh = rows[hold.numpy()]
        err = [float((X[hold] @ W_[l - 1] - V[rh + l]).pow(2).mean()) for l in (1, LEADS)]
        print(json.dumps(dict(ridge=lam, val_mse_day1=round(err[0], 5), val_mse_day10=round(err[1], 5))), flush=True)
        if best is None or err[1] < best[1]:
            best = (lam, err[1])
    Wl = fit(torch.ones(len(rows), dtype=torch.bool), best[0])
    print(json.dumps(dict(ridge_chosen=best[0], sec=round(time.time() - t0))), flush=True)
else:
    cand_all = valid(A.lags, LEADS)
    embA = torch.cat([P[cand_all - j, : A.q] for j in range(A.lags + 1)], 1)

for year in A.years:
    meta = json.load(open(f"{A.dir}/{year}_meta.json"))
    ics = np.asarray(meta["ics"])
    f, n = anomalies(year)
    Vy = to_vec(f)
    del f
    Py = Vy @ E.T
    out = np.lib.format.open_memmap(f"{A.dir}/{year}_{A.tag}.npy", mode="w+", dtype=np.float16, shape=(len(ics), LEADS, C) + CZ.shape[-2:])
    NBR = []
    for i in range(0, len(ics), 16):
        k = ics[i:i + 16]
        if A.kind == "lim":
            xi = lim_inputs(Py, k)
            pred = torch.stack([xi @ Wl[l] for l in range(LEADS)], 1)                      # (b, 10, DIM)
        else:
            emb = torch.cat([Py[np.maximum(k - j, 0), : A.q] for j in range(A.lags + 1)], 1)      # (b, q (lags+1))
            pred = []
            for b, kk in enumerate(k):
                dd = np.abs(DOY[cand_all] - kk)
                dd = np.minimum(dd, 365 - dd)
                m = dd <= A.window
                cand = cand_all[m]
                d2 = (embA[m] - emb[b]).pow(2).sum(1)
                v, j = torch.topk(d2, A.k, largest=False)
                w = torch.exp(-v / v[-1].clamp_min(1e-12))
                w = w / w.sum()
                rows_j = cand[j.numpy()]
                NBR.append(rows_j[:16])
                pred.append(torch.stack([(w[:, None] * V[rows_j + l]).sum(0) for l in range(1, LEADS + 1)]))
            pred = torch.stack(pred)
        out[i:i + len(k)] = to_field(pred.flatten(0, 1)).view(len(k), LEADS, C, *CZ.shape[-2:]).numpy()
    out.flush()
    if NBR:     # the analog kernel's members: archive rows (days counted over the archive years in order), nearest first
        np.savez(f"{A.dir}/{year}_{A.tag}_nbr.npz", rows=np.stack(NBR), starts=starts, years=np.asarray(list(a.train_years)))
    print(json.dumps(dict(year=year, ics=len(ics), sec=round(time.time() - t0))), flush=True)
json.dump(dict(kind=A.kind, lt=A.lt, npc=A.npc, npc_lag=A.npc_lag, lags=A.lags, q=A.q, k=A.k, window=A.window,
               ridge=(best[0] if A.kind == "lim" else None), archive=list(a.train_years)), open(f"{A.dir}/{A.tag}_info.json", "w"))
print("done", round(time.time() - t0))
