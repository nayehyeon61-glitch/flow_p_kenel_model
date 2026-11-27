"""What a further core is worth in the mean (감자 2026-11-24 23:56 KST: which cores to mix).

From the dumps of scripts/diag_cond.py (forecast anomalies of every core, the truth, z-scored) the second moments of the
fields per Littlewood-Paley band, lead and channel:  G[c, b, i, j] = < X_i,b X_j,b >  (area weighted, mean over the ICs),
X = (the cores ..., the truth). Everything below is read from G.

  table    every core alone: rmse, the amplitude of its forecast, the correlation of its error with the others
  noise    the signal and noise model of an exchangeable family of cores:  f_i = s + n_i  with the n_i independent of each
           other and of the truth. With the pair (W, NW):  <s s> = <f_W f_NW>,  <n n> = <f f> - <f_W f_NW>,
           <s y> = <fbar y>. The mean of n such cores under the band-wise condition (the best strength per band) leaves
               mse(n) = sum_b  Y_b - c_b^2 / (S_b + N_b / n)
           n = 2 is the mean of now, n -> infinity is what more cores of the same family can reach at most (cores of one
           kind share more than W and NW do, so real sisters do worse than this).
  need     what a new core must bring: with rmse r times the one of the conditioned mean and an error correlation rho to
           it, the best mixture leaves  mse * (1 - (1 - rho r)^2 / (1 + r^2 - 2 rho r)).

    python scripts/core_value.py --years 2016 2020 2017 --days 5 10 --json runs/fpk/core_mix/core_value.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpk.sphere import SHT, bands as lp_bands  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="runs/fpk/cond")
ap.add_argument("--years", type=int, nargs="+", default=[2016, 2020, 2017])
ap.add_argument("--days", type=int, nargs="+", default=[10])
ap.add_argument("--cores", default="core_gngmz8W,core_ngmz8W,core_gngmz8L,core_gngmz8W_c,lim,kaf")
ap.add_argument("--pair", default="core_gngmz8W,core_ngmz8W")
ap.add_argument("--batch", type=int, default=12)
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--json", default="")
ap.add_argument("--cache", default="", help="prefix of the saved second moments (one npz per year and day)")
A = ap.parse_args()
torch.set_num_threads(A.threads)
CORES = A.cores.split(",")
PAIR = A.pair.split(",")
meta = json.load(open(f"{A.dir}/{A.years[0]}_meta.json"))
ch, std, lat = meta["channels"], np.asarray(meta["std"]), np.asarray(meta["lat"])
C, Z = len(ch), ch.index("z500")
SC = [i for i in range(C) if ch[i] != "tisr"]
w = np.cos(np.deg2rad(lat))
w = torch.tensor(w / w.mean(), dtype=torch.float32)[:, None]
sht = SHT(lat)
bj, nb = lp_bands(sht.L)
masks = [(bj == b).float()[:, None] for b in range(nb)]


def banded(f):
    c = sht.analysis(f)
    x = torch.stack([sht.synthesis(c * m) for m in masks], 2)
    x[:, :, -1] += f - x.sum(2)
    return x                                                                          # (B, C, nb, H, W)


def gram(y, L):
    m = json.load(open(f"{A.dir}/{y}_meta.json"))
    F = [t for t in CORES if os.path.exists(f"{A.dir}/{y}_{t}.npy")] + ["o"]
    cf = f"{A.cache}_{y}_d{L + 1}.npz"
    if A.cache and os.path.exists(cf):
        v = np.load(cf)
        if [str(x) for x in v["F"]] == F:
            return F, v["G"], v["G0"], int(v["N"])
    d = {t: np.load(f"{A.dir}/{y}_{t}.npy", mmap_mode="r") for t in F}
    N = len(m["ics"])
    G, G0 = 0, 0
    for i in range(0, N, A.batch):
        X = torch.stack([torch.from_numpy(np.asarray(d[t][i:i + A.batch, L], dtype=np.float32)) for t in F], 1)   # (B, F, C, H, W)
        B = torch.stack([banded(X[:, k]) for k in range(len(F))], 1)                  # (B, F, C, nb, H, W)
        G = G + torch.einsum("bickhw,bjckhw->ckij", B * w, B).double()
        G0 = G0 + torch.einsum("bichw,bjchw->cij", X * w, X).double()
    H, W = X.shape[-2:]
    G, G0 = (G / (N * H * W)).numpy(), (G0 / (N * H * W)).numpy()
    if A.cache:
        np.savez(cf, F=np.array(F), G=G, G0=G0, N=N)
    return F, G, G0, N


def rz(mse, c=Z):
    return float(np.sqrt(max(mse, 0)) * std[c])


out = {}
for day in A.days:
    L = day - 1
    per = {}
    for y in A.years:
        per[y] = gram(y, L)
    groups = {str(y): [y] for y in A.years}
    if len(A.years) > 1:
        groups["+".join(str(y) for y in A.years[:-1])] = A.years[:-1]
    for gname, ys in groups.items():
        F = [t for t in per[ys[0]][0] if all(t in per[y][0] for y in ys)]
        n = sum(per[y][3] for y in ys)
        ix = lambda y: [per[y][0].index(t) for t in F]  # noqa: E731
        G = sum(per[y][1][:, :, ix(y)][:, :, :, ix(y)] * per[y][3] for y in ys) / n   # (C, nb, F, F)
        G0 = sum(per[y][2][:, ix(y)][:, :, ix(y)] * per[y][3] for y in ys) / n        # (C, F, F)
        o = F.index("o")
        res = dict(n=n, fields=F)
        # ---- every core alone
        E = np.zeros((C, len(F) - 1, len(F) - 1))                                     # error second moments
        for i in range(len(F) - 1):
            for j in range(len(F) - 1):
                E[:, i, j] = G0[:, i, j] - G0[:, i, o] - G0[:, j, o] + G0[:, o, o]
        rm = [rz(E[Z, i, i]) for i in range(len(F) - 1)]
        cor = E[Z] / np.sqrt(np.outer(np.diag(E[Z]), np.diag(E[Z])))
        amp = [float(np.sqrt(G0[Z, i, i] / G0[Z, o, o])) for i in range(len(F) - 1)]
        acc = [float(G0[Z, i, o] / np.sqrt(G0[Z, i, i] * G0[Z, o, o])) for i in range(len(F) - 1)]
        res["alone"] = {F[i]: dict(rmse=round(rm[i], 1), amp=round(amp[i], 3), acc=round(acc[i], 3)) for i in range(len(F) - 1)}
        res["err_corr_z500"] = {F[i]: {F[j]: round(float(cor[i, j]), 3) for j in range(len(F) - 1)} for i in range(len(F) - 1)}
        res["clim_rmse"] = round(rz(G0[Z, o, o]), 1)
        # ---- the family model on the pair
        p, q = F.index(PAIR[0]), F.index(PAIR[1])
        S = G[:, :, p, q]                                                             # <s s> per channel and band
        V = 0.5 * (G[:, :, p, p] + G[:, :, q, q])
        Nn = np.maximum(V - S, 0)
        c = 0.5 * (G[:, :, p, o] + G[:, :, q, o])
        Y = G[:, :, o, o]
        fam = {}
        ok = S > 0.1 * V                                                              # a band of noise only tells nothing of s
        sk2 = c ** 2 / np.maximum(S + Nn / 2, 1e-12)
        for k in (1, 2, 3, 4, 6, 8, 16, 0):
            sk = np.where(ok, np.minimum(c ** 2 / np.maximum(S + (Nn / k if k else 0), 1e-12), Y), sk2 if k != 1 else c ** 2 / np.maximum(V, 1e-12))
            mse = (Y - sk).sum(1)                                                     # (C,)
            fam[str(k) if k else "inf"] = dict(z500=round(rz(mse[Z]), 1), t850=round(rz(mse[ch.index("t850")], ch.index("t850")), 3),
                                               t2m=round(rz(mse[ch.index("t2m")], ch.index("t2m")), 3),
                                               nrmse=round(float(np.sqrt(np.maximum(mse, 0))[SC].mean()), 4))
        res["family"] = fam
        res["family_frozen_bands"] = int((~ok)[SC].sum())
        res["family_bands_z500"] = dict(noise_share=[round(float(x), 3) for x in (Nn[Z] / np.maximum(V[Z], 1e-12))],
                                        truth_share=[round(float(x), 3) for x in (Y[Z] / Y[Z].sum())],
                                        skill_2=[round(float(x), 3) for x in (c[Z] ** 2 / np.maximum(S[Z] + Nn[Z] / 2, 1e-12) / Y[Z])],
                                        skill_inf=[round(float(x), 3) for x in (c[Z] ** 2 / np.maximum(S[Z], 1e-12) / Y[Z])])
        # ---- the measured mean of the cores at hand under the band-wise condition (in sample), sets of cores
        sets = {"W": [PAIR[0]], "NW": [PAIR[1]], "W+NW": PAIR}
        for extra in F[:-1]:
            if extra not in PAIR:
                sets["W+NW+" + extra.replace("core_", "")] = PAIR + [extra]
        meas = {}
        for name, ts in sets.items():
            ii = [F.index(t) for t in ts]
            mm = G[:, :, ii][:, :, :, ii].mean((2, 3))                                # <fbar fbar>
            cc = G[:, :, ii, o].mean(2)
            mse_eq = (Y - cc ** 2 / np.maximum(mm, 1e-12)).sum(1)                     # equal weights, one strength per band
            Gs = G[:, :, ii][:, :, :, ii] + 1e-9 * np.eye(len(ii))
            th = np.linalg.solve(Gs, G[:, :, ii, o][..., None])[..., 0]
            mse_fr = (Y - (th * G[:, :, ii, o]).sum(-1)).sum(1)                       # one weight per core and band
            meas[name] = dict(z500_equal=round(rz(mse_eq[Z]), 1), z500_free=round(rz(mse_fr[Z]), 1),
                              nrmse_equal=round(float(np.sqrt(np.maximum(mse_eq, 0))[SC].mean()), 4))
        res["in_sample"] = meas
        out[f"d{day}_{gname}"] = res
        print(f"== day {day}  years {gname}  ICs {n}  clim {res['clim_rmse']}")
        print("   alone:", {k.replace('core_', ''): (v['rmse'], v['amp'], v['acc']) for k, v in res["alone"].items()}, "(rmse, amplitude, acc)")
        print("   error correlation with W | NW:", {F[i].replace('core_', ''): (round(float(cor[i, p]), 3), round(float(cor[i, q]), 3)) for i in range(len(F) - 1)})
        print("   family (n cores, band-wise condition; z500, t850, t2m, nrmse): " + "  ".join(f"{k}: {v['z500']} {v['t850']} {v['t2m']} {v['nrmse']}" for k, v in fam.items()),
              f" [bands held at n=2: {res['family_frozen_bands']} of {len(SC) * nb}]")
        print("   bands z500: noise share", res["family_bands_z500"]["noise_share"], " skill n=2", res["family_bands_z500"]["skill_2"],
              " n=inf", res["family_bands_z500"]["skill_inf"])
        print("   in sample (equal | free):", {k: (v['z500_equal'], v['z500_free']) for k, v in meas.items()}, flush=True)
if A.json:
    json.dump(out, open(A.json, "w"), indent=1)
