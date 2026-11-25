"""A measure core continued from the cores' own forecast (감자 2026-11-25 16:34 KST: "물리도 데이터도 아닌 측도적 core").

The mathematical cores of math_core.py start the transfer operator of the record at the initial state and carry it 10
days (z500 765 alone, nothing on top of the cores). Here the same operator (a linear inverse model in the principal
components of the 16 training years, all 66 channels, l < --lt) starts at the cores' day-k forecast and carries the
remaining 10 - k days: the equations and the network do the part they are good at, the measure does the rest.

    h_k = A_tau P(f_k),   k = day - tau:  tau = day is the pure measure core (from the initial state), tau = 0 only the
                          projection on the measure's leading patterns (no dynamics)

--inp tells what the measure sees of the forecast (all 66 channels, or parts: where the information comes from).
Writes <dir>/<year>_hyb<tag>_<chan>_d<day>.npz with k<k> (N, H, W) anomalies of one channel, read by measure_cond.py
(hyb<k> or hyb<k>@<tag>).

    python scripts/measure_core.py --taus 0 1 2 3 5 7 10
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpk.sphere import SHT  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="runs/fpk/cond")
ap.add_argument("--years", type=int, nargs="+", default=[2016, 2020, 2017])
ap.add_argument("--days", type=int, nargs="+", default=[10])
ap.add_argument("--chans", nargs="+", default=["z500"])
ap.add_argument("--cores", default="core_gngmz8W,core_ngmz8W")
ap.add_argument("--taus", type=int, nargs="+", default=[1, 2, 3, 5, 7, 10], help="days left to the measure (0: only its projection)")
ap.add_argument("--inp", default="all", help="what the measure sees of the forecast: all, chan (the target channel), var (its variable at every level), "
                "novar (everything but its variable), lev (every variable at its level)")
ap.add_argument("--lags", type=int, default=0, help="days before day k in the input (the forecast of day k - 1, ...; the initial state below)")
ap.add_argument("--tag", default="")
ap.add_argument("--dump", default="", help="write every channel as one more core for cond_fit.py (--math): <dir>/<year>_<dump>.npy, only the leads "
                "of --days filled. --taus 0: the projection of the same day's forecast; --taus T: the last min(T, day - 1) days by the measure")
ap.add_argument("--lt", type=int, default=16)
ap.add_argument("--npc", type=int, default=200)
ap.add_argument("--threads", type=int, default=4)
A = ap.parse_args()
torch.set_num_threads(A.threads)
CORES = A.cores.split(",")
meta = json.load(open(f"{A.dir}/{A.years[0]}_meta.json"))
ch, lat = meta["channels"], np.asarray(meta["lat"])
C = len(ch)
sht = SHT(lat)
M = sht.ana.shape[0]
keep = torch.zeros(sht.L, M, dtype=torch.bool)
for l in range(A.lt):
    keep[l, : l + 1] = True
im0 = torch.zeros(sht.L, M, dtype=torch.bool)
im0[:, 0] = True
KI, KIm = keep.flatten().nonzero()[:, 0], (keep & ~im0).flatten().nonzero()[:, 0]
NV = len(KI) + len(KIm)
wl = np.cos(np.deg2rad(lat))
wl = torch.tensor(wl / wl.mean(), dtype=torch.float32)[:, None]
SURF = ("t2m", "sp", "msl", "tcwv", "skt", "tisr")


def to_vec(f):
    """(D, C, H, W) anomaly (numpy, any float) -> (D, C, NV): the real coefficients l < lt."""
    out = []
    for i in range(0, len(f), 64):
        c = sht.analysis(torch.from_numpy(np.ascontiguousarray(f[i:i + 64], dtype=np.float32))).flatten(-2)
        out.append(torch.cat([c.real[..., KI], c.imag[..., KIm]], -1))
    return torch.cat(out)


def to_field(v):
    """(B, NV) of one channel -> (B, H, W)."""
    re, im = torch.zeros(len(v), sht.L * M), torch.zeros(len(v), sht.L * M)
    re[:, KI], im[:, KIm] = v[:, : len(KI)], v[:, len(KI):]
    return sht.synthesis(torch.complex(re, im).view(len(v), sht.L, M))


def inputs(chan):
    """The channels the measure sees for a target channel."""
    if A.inp == "all":
        return list(range(C))
    if A.inp == "chan":
        return [ch.index(chan)]
    surf = chan in SURF
    same_var = [i for i, c in enumerate(ch) if (c == chan if surf else (c not in SURF and c[0] == chan[0]))]
    if A.inp == "var":
        return same_var
    if A.inp == "novar":
        return [i for i in range(C) if i not in same_var]
    if A.inp == "lev":
        return [i for i, c in enumerate(ch) if c == chan or (not surf and c not in SURF and c[1:] == chan[1:])]
    raise SystemExit(f"unknown --inp {A.inp}")


t0 = time.time()
am = json.load(open(f"{A.dir}/archive_meta.json"))
V = to_vec(np.load(f"{A.dir}/archive_anom.npy", mmap_mode="r"))                       # (N, C, NV)
starts = np.cumsum([0] + am["days"])
YR = np.concatenate([np.full(n, y) for y, n in zip(am["years"], am["days"])])
print(json.dumps(dict(archive=len(V), nv=NV, sec=round(time.time() - t0))), flush=True)
FC, EOFS, PK = {}, {}, {}


def fvec(y, k):
    """The cores' average forecast of day k (0: the initial state; below: the days before the start) as coefficients."""
    if (y, k) not in FC:
        ics = np.asarray(json.load(open(f"{A.dir}/{y}_meta.json"))["ics"])
        if k <= 0:
            x0 = np.load(f"{A.dir}/{y}_x0.npy", mmap_mode="r")
            assert k == 0 or np.all(np.diff(ics) == 1), "the days before the start need a dump of every day"
            f = np.asarray(x0[np.maximum(np.arange(len(ics)) + k, 0)])
        else:
            f = sum(np.asarray(np.load(f"{A.dir}/{y}_{t}.npy", mmap_mode="r")[:, k - 1], dtype=np.float32) for t in CORES) / len(CORES)
        FC[(y, k)] = to_vec(f)
    return FC[(y, k)]


if A.dump:                                                                             # the whole state, as one more core
    Vi = V.flatten(1)
    ev, U = torch.linalg.eigh((Vi @ Vi.T).double())
    E = ((U[:, -A.npc:].T.float() @ Vi) / ev[-A.npc:].float().sqrt()[:, None]).flip(0)
    P = Vi @ E.T
    print(json.dumps(dict(dump=A.dump, dim=Vi.shape[1], npc=A.npc, explained=round(float(ev[-A.npc:].sum() / ev.sum()), 3), sec=round(time.time() - t0))), flush=True)
    del U
    WD = {}

    def lim(tau, lags):
        if (tau, lags) not in WD:
            rows = np.concatenate([np.arange(s + lags, e - tau) for s, e in zip(starts[:-1], starts[1:])])
            hold = np.isin(YR[rows], am["years"][-2:])
            X = torch.cat([P[rows - j] for j in range(lags + 1)] + [torch.ones(len(rows), 1)], 1)
            Xt, Xh, Yh = X[~hold], X[hold], Vi[rows[hold] + tau]
            Gt, Bt = Xt.T @ Xt, Xt.T @ Vi[rows[~hold] + tau]
            eye, vh, val = torch.eye(len(Gt)), Yh.pow(2).view(len(Yh), C, NV).mean((0, 2)), {}
            for lam in (1e-3, 1e-2, 1e-1, 1.0):
                Wv = torch.linalg.solve(Gt + lam * torch.trace(Gt) / len(Gt) * eye, Bt)
                val[lam] = float(((Xh @ Wv - Yh).pow(2).view(len(Yh), C, NV).mean((0, 2)) / vh.clamp_min(1e-12)).mean())
            lam = min(val, key=val.get)
            G = Gt + Xh.T @ Xh
            WD[(tau, lags)] = torch.linalg.solve(G + lam * torch.trace(G) / len(G) * eye, Bt + Xh.T @ Yh)
            print(json.dumps(dict(tau=tau, lags=lags, ridge=lam, val_mse_over_var=round(val[lam], 4), sec=round(time.time() - t0))), flush=True)
        return WD[(tau, lags)]

    T0 = A.taus[0]
    for y in A.years:
        n = len(json.load(open(f"{A.dir}/{y}_meta.json"))["ics"])
        out = np.lib.format.open_memmap(f"{A.dir}/{y}_{A.dump}.npy", mode="w+", dtype=np.float16, shape=(n, 10, C, len(lat), 2 * len(lat)))
        for day in A.days:
            tau = 0 if T0 == 0 else (min(T0, day - 1) if day >= 2 else 1)
            k = day - tau
            lags = A.lags if (T0 and k - A.lags >= 0) else 0
            for j in range(lags + 1):
                if (y, k - j) not in PK:
                    PK[(y, k - j)] = fvec(y, k - j).flatten(1) @ E.T
            Pk = torch.cat([PK[(y, k - j)] for j in range(lags + 1)] + [torch.ones(n, 1)], 1)
            Wm = lim(tau, lags)
            for i in range(0, n, 32):
                pr = (Pk[i:i + 32] @ Wm).view(-1, NV)
                out[i:i + 32, day - 1] = to_field(pr).view(-1, C, len(lat), 2 * len(lat)).numpy()
            print(json.dumps(dict(year=y, day=day, k=k, tau=tau, lags=lags, sec=round(time.time() - t0))), flush=True)
        out.flush()
        del out
    print("done", round(time.time() - t0))
    raise SystemExit(0)

for chan in A.chans:
    Z, std = ch.index(chan), float(meta["std"][ch.index(chan)])
    IN = inputs(chan)
    if tuple(IN) not in EOFS:                                                          # the measure's patterns of these inputs
        EOFS.clear()
        PK.clear()
        Vi = V[:, IN].flatten(1)
        ev, U = torch.linalg.eigh((Vi @ Vi.T).double())
        npc = min(A.npc, Vi.shape[1])
        E = ((U[:, -npc:].T.float() @ Vi) / ev[-npc:].float().sqrt()[:, None]).flip(0)   # (npc, DIM) EOFs, leading first
        EOFS[tuple(IN)] = (E, Vi @ E.T)
        print(json.dumps(dict(chan=chan, inputs=len(IN), dim=Vi.shape[1], npc=npc, explained=round(float(ev[-npc:].sum() / ev.sum()), 3),
                              sec=round(time.time() - t0))), flush=True)
        del Vi, U
    E, P = EOFS[tuple(IN)]
    T = V[:, Z]
    Wt = {}
    for tau in sorted(set(A.taus)):
        rows = np.concatenate([np.arange(s + A.lags, e - tau) for s, e in zip(starts[:-1], starts[1:])])
        hold = torch.from_numpy(np.isin(YR[rows], am["years"][-2:]))
        X = torch.cat([P[rows - j] for j in range(A.lags + 1)] + [torch.ones(len(rows), 1)], 1)
        Y = T[rows + tau]

        def fit(m, lam):
            G = X[m].T @ X[m]
            return torch.linalg.solve(G + lam * torch.trace(G) / len(G) * torch.eye(len(G)), X[m].T @ Y[m])

        val = {lam: float((X[hold] @ fit(~hold, lam) - Y[hold]).pow(2).mean()) for lam in (1e-3, 1e-2, 1e-1, 1.0)}
        lam = min(val, key=val.get)
        Wt[tau] = fit(torch.ones(len(rows), dtype=torch.bool), lam)
        print(json.dumps(dict(chan=chan, tau=tau, ridge=lam, val_skill=round(1 - val[lam] / float(Y[hold].pow(2).mean()), 4),
                              sec=round(time.time() - t0))), flush=True)
    for day in A.days:
        for y in A.years:
            out = {}
            o = torch.from_numpy(np.asarray(np.load(f"{A.dir}/{y}_o.npy", mmap_mode="r")[:, day - 1, Z], dtype=np.float32))
            for tau in A.taus:
                k = day - tau
                if k < 0 or (A.lags and k - A.lags < 0 and json.load(open(f"{A.dir}/{y}_meta.json")).get("ic_stride", 1) != 1):
                    continue
                for j in range(A.lags + 1):
                    if (y, k - j) not in PK:
                        PK[(y, k - j)] = fvec(y, k - j)[:, IN].flatten(1) @ E.T
                Pk = torch.cat([PK[(y, k - j)] for j in range(A.lags + 1)], 1)
                h = to_field(torch.cat([Pk, torch.ones(len(Pk), 1)], 1) @ Wt[tau])
                out[f"k{k}"] = h.numpy().astype(np.float16)
                g = float((h * o * wl).mean() / (h * h * wl).mean())                  # the best single strength, in sample
                print(json.dumps(dict(chan=chan, day=day, year=y, k=k, tau=tau, rmse=round(float(((h - o) ** 2 * wl).mean().sqrt()) * std, 2),
                                      rmse_scaled=round(float(((g * h - o) ** 2 * wl).mean().sqrt()) * std, 2), g=round(g, 3),
                                      clim=round(float((o ** 2 * wl).mean().sqrt()) * std, 2), sec=round(time.time() - t0))), flush=True)
            np.savez(f"{A.dir}/{y}_hyb{A.tag}_{chan}_d{day}.npz", **out)
print("done", round(time.time() - t0))
