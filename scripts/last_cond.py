"""Two conditions on the mean that are not linear features of scripts/cond_fit.py (감자 2026-11-25 00:26 KST: the last shot).

Both stand on the band-wise condition of now (clim + sum_b g_b f_b, g per band, lead and channel, least squares) and read
the dumps of scripts/diag_cond.py at every date (--dir with --ic-stride 1 dumps of the three years).

  sbias    the local first moment: the mean of the errors that are left after the band-wise condition, at every grid
           point, over the dates of the OTHER years within +-win days of the verifying day of the year,
               mean <- mean + alpha * b(x, season).
           For 2016 <-> 2020 the map of one year is applied to the other (cv), for the test year the map of both.
  balance  the mean of one variable read from every variable and level of the forecast: all the channels, banded, as
           features of the target channels (--targets), with a ridge that is chosen by the cv.

    python scripts/last_cond.py --dir runs/fpk/cond_v --days 10 --json runs/fpk/core_mix/last_cond_d10.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpk.sphere import SHT, bands as lp_bands  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="runs/fpk/cond_v")
ap.add_argument("--fit", type=int, nargs=2, default=[2016, 2020])
ap.add_argument("--test", type=int, default=2017)
ap.add_argument("--days", type=int, nargs="+", default=[10])
ap.add_argument("--cores", default="core_gngmz8W,core_ngmz8W")
ap.add_argument("--win", type=int, nargs="+", default=[45])
ap.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
ap.add_argument("--targets", nargs="+", default=["z500", "t850", "t2m"])
ap.add_argument("--ridges", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2, 1e-1])
ap.add_argument("--skip", nargs="*", default=[], help="sbias and / or balance")
ap.add_argument("--ridge", type=float, default=1e-4)
ap.add_argument("--batch", type=int, default=12)
ap.add_argument("--block", type=int, default=15)
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--json", default="")
A = ap.parse_args()
torch.set_num_threads(A.threads)
CORES = A.cores.split(",")
YEARS = A.fit + [A.test]
D = {}
for y in YEARS:
    m = json.load(open(f"{A.dir}/{y}_meta.json"))
    assert m["ic_stride"] == 1, "dumps of every date are needed"
    D[y] = dict(ics=np.asarray(m["ics"]), o=np.load(f"{A.dir}/{y}_o.npy", mmap_mode="r"),
                f=[np.load(f"{A.dir}/{y}_{t}.npy", mmap_mode="r") for t in CORES])
ch, std, lat = m["channels"], np.asarray(m["std"]), np.asarray(m["lat"])
C = len(ch)
SC = [i for i in range(C) if ch[i] != "tisr"]
H, W = D[A.test]["o"].shape[-2:]
w = np.cos(np.deg2rad(lat))
w = torch.tensor(w / w.mean(), dtype=torch.float32)[:, None]
sht = SHT(lat)
bj, nb = lp_bands(sht.L)
masks = [(bj == b).float()[:, None] for b in range(nb)]
TG = [ch.index(t) for t in A.targets]


def banded(f):
    c = sht.analysis(f)
    x = torch.stack([sht.synthesis(c * m_) for m_ in masks], 2)
    x[:, :, -1] += f - x.sum(2)
    return x                                                                          # (B, C, nb, H, W)


def batches(y, L):
    d = D[y]
    for i in range(0, len(d["ics"]), A.batch):
        j = slice(i, min(i + A.batch, len(d["ics"])))
        fm = sum(torch.from_numpy(np.asarray(f[j, L], dtype=np.float32)) for f in d["f"]) / len(d["f"])
        yield j, banded(fm), torch.from_numpy(np.asarray(d["o"][j, L], dtype=np.float32))


def solve(S, c, ridge):
    K = S.shape[-1]
    tr = torch.diagonal(S, dim1=-2, dim2=-1).sum(-1).clamp_min(1e-12)
    return torch.linalg.solve(S + (ridge * tr / K)[..., None, None] * torch.eye(K, dtype=S.dtype), c[..., None])[..., 0].float()


def rz(mse, c):
    return float(np.sqrt(mse[:, c].mean(0)) * std[c])


def nrm(mse):
    return float(np.sqrt(mse.mean(0))[SC].mean())


def boot(a, b, fn, block):
    n = (len(a) // block) * block
    rng = np.random.default_rng(0)
    dr = (rng.integers(0, n // block, size=(2000, n // block))[:, :, None] * block + np.arange(block)).reshape(2000, -1)
    x = np.array([fn(a[i]) - fn(b[i]) for i in dr])
    return [round(float(fn(a) - fn(b)), 3)] + np.percentile(x, [2.5, 97.5]).round(3).tolist()


out = {}
t0 = time.time()
for day in A.days:
    L = day - 1
    res = {}
    # ------------------------------------------------ the band-wise condition: normal equations per year
    NE, NA = {}, {}
    for y in A.fit:
        S, c, Sa, ca, n = 0, 0, 0, 0, 0
        for j, B, o in batches(y, L):
            Bw = B * w
            S = S + torch.einsum("bckhw,bcjhw->ckj", Bw, B).double() / (H * W)
            c = c + torch.einsum("bckhw,bchw->ck", Bw, o).double() / (H * W)
            if "balance" not in A.skip:
                X = B.flatten(1, 2)                                                    # (B, C * nb, H, W): every channel
                X2, Xw2 = X.transpose(0, 1).flatten(1), (X * w).transpose(0, 1).flatten(1)   # (K, B * H * W)
                Sa = Sa + (Xw2 @ X2.T).double() / (H * W)
                ca = ca + (o[:, TG].transpose(0, 1).flatten(1) @ Xw2.T).double() / (H * W)
            n += o.shape[0]
        NE[y], NA[y] = (S, c, n), (Sa, ca, n)
        print(json.dumps(dict(day=day, normal=y, sec=round(time.time() - t0))), flush=True)
    y0, y1 = A.fit
    TH = {"all": solve((NE[y0][0] + NE[y1][0]) / (NE[y0][2] + NE[y1][2]), (NE[y0][1] + NE[y1][1]) / (NE[y0][2] + NE[y1][2]), A.ridge),
          y0: solve(NE[y0][0] / NE[y0][2], NE[y0][1] / NE[y0][2], A.ridge), y1: solve(NE[y1][0] / NE[y1][2], NE[y1][1] / NE[y1][2], A.ridge)}
    THA = {}
    if "balance" not in A.skip:
        for r in A.ridges:
            THA[r] = {"all": solve(((NA[y0][0] + NA[y1][0]) / (NA[y0][2] + NA[y1][2]))[None], (NA[y0][1] + NA[y1][1]) / (NA[y0][2] + NA[y1][2]), r),
                      y0: solve((NA[y0][0] / NA[y0][2])[None], NA[y0][1] / NA[y0][2], r), y1: solve((NA[y1][0] / NA[y1][2])[None], NA[y1][1] / NA[y1][2], r)}
    # ------------------------------------------------ the errors that are left (f16), and the balance's mse
    need = {y0: ["all", y0, y1], y1: ["all", y0, y1], A.test: ["all"]}                # theta of the key applied to the year
    E = {(y, k): np.zeros((len(D[y]["ics"]), C, H, W), dtype=np.float16) for y in YEARS for k in need[y]}
    MB = {}
    for y in YEARS:
        for j, B, o in batches(y, L):
            for k in need[y]:
                E[(y, k)][j] = (o - torch.einsum("ck,bckhw->bchw", TH[k], B)).numpy()
            if "balance" not in A.skip:
                X = B.flatten(1, 2)
                for r in A.ridges:
                    for k in (["all"] if y == A.test else [y0 if y == y1 else y1]):  # theta that never saw the year
                        p = torch.einsum("tk,bkhw->bthw", THA[r][k], X)
                        MB.setdefault((r, y), []).append(((o[:, TG] - p) ** 2 * w).mean((-2, -1)).double().numpy())
        print(json.dumps(dict(day=day, errors=y, sec=round(time.time() - t0))), flush=True)

    def mse_of(e, b=None, alpha=0.0):
        r = []
        for i in range(0, e.shape[0], 32):
            x = torch.from_numpy(e[i:i + 32].astype(np.float32))
            if b is not None:
                x = x - alpha * torch.from_numpy(b[i:i + 32])
            r.append((x ** 2 * w).mean((-2, -1)).double().numpy())
        return np.concatenate(r)                                                      # (N, C)

    def season_map(pool, k, tgt, win):
        """The mean of the errors of the pool years (theta of key k) within +-win days of the verifying day of every IC of tgt."""
        num, den = 0, 0
        vt = (D[tgt]["ics"] + L + 1) % 365
        for py in pool:
            e = E[(py, k)]
            vp = (D[py]["ics"] + L + 1) % 365
            dd = np.abs(vt[:, None] - vp[None])
            sel = (np.minimum(dd, 365 - dd) <= win).astype(np.float32)                # (Nt, Np)
            for c0 in range(0, C, 11):
                ee = torch.from_numpy(e[:, c0:c0 + 11].astype(np.float32)).flatten(1)  # (Np, c * H * W)
                part = (torch.from_numpy(sel) @ ee).view(len(vt), -1, H, W).numpy()
                if isinstance(num, int):
                    num = np.zeros((len(vt), C, H, W), dtype=np.float32)
                num[:, c0:c0 + 11] += part
            den = den + sel.sum(1)
        return num / den[:, None, None, None]

    base = {(A.test, "all"): mse_of(E[(A.test, "all")]), y0: mse_of(E[(y0, y1)]), y1: mse_of(E[(y1, y0)])}   # cv: theta of the other year
    cvb = np.concatenate([base[y0], base[y1]])
    bt = base[(A.test, "all")]
    Zc = ch.index("z500")
    res["bands"] = dict(z500=rz(bt, Zc), t850=rz(bt, ch.index("t850")), t2m=rz(bt, ch.index("t2m")), nrmse=nrm(bt),
                        cv_z500=rz(cvb, Zc), cv_nrmse=nrm(cvb))
    print(f"day {day:2d} bands                 z500 {res['bands']['z500']:6.1f} cv {res['bands']['cv_z500']:6.1f}  t850 {res['bands']['t850']:.3f} "
          f"t2m {res['bands']['t2m']:.3f}  nrmse {res['bands']['nrmse']:.4f} cv {res['bands']['cv_nrmse']:.4f}", flush=True)
    if "sbias" not in A.skip:
        for win in A.win:
            bT = season_map(A.fit, "all", A.test, win)                                # both years' errors -> the test year
            b0 = season_map([y1], y1, y0, win)                                        # the errors of y1 (its own theta) -> y0
            b1 = season_map([y0], y0, y1, win)
            for al in A.alphas:
                mt = mse_of(E[(A.test, "all")], bT, al)
                cv = np.concatenate([mse_of(E[(y0, y1)], b0, al), mse_of(E[(y1, y0)], b1, al)])
                r = dict(z500=rz(mt, Zc), t850=rz(mt, ch.index("t850")), t2m=rz(mt, ch.index("t2m")), nrmse=nrm(mt), cv_z500=rz(cv, Zc),
                         cv_nrmse=nrm(cv), z500_minus_bands=boot(mt, bt, lambda m_: rz(m_, Zc), A.block),
                         nrmse_minus_bands=boot(mt, bt, nrm, A.block),
                         by_var={v: round(100 * (float(np.sqrt(mt.mean(0))[[i for i in SC if ch[i][0] == v and ch[i][1:].isdigit()]].mean())
                                                 / float(np.sqrt(bt.mean(0))[[i for i in SC if ch[i][0] == v and ch[i][1:].isdigit()]].mean()) - 1), 2)
                                 for v in "ztuvq"})
                r["by_var"]["surface"] = round(100 * (float(np.sqrt(mt.mean(0))[[ch.index(x) for x in ("t2m", "sp", "tcwv", "msl", "skt")]].mean())
                                                      / float(np.sqrt(bt.mean(0))[[ch.index(x) for x in ("t2m", "sp", "tcwv", "msl", "skt")]].mean()) - 1), 2)
                res[f"sbias{win}@{al}"] = r
                print(f"day {day:2d} sbias{win:<3d} alpha {al:4.2f}  z500 {r['z500']:6.1f} cv {r['cv_z500']:6.1f}  t850 {r['t850']:.3f} t2m {r['t2m']:.3f}  "
                      f"nrmse {r['nrmse']:.4f} cv {r['cv_nrmse']:.4f}  z500 {r['z500_minus_bands']}  nrmse {r['nrmse_minus_bands']}  {r['by_var']}", flush=True)
    if "balance" not in A.skip:
        for r_ in A.ridges:
            mt = np.concatenate(MB[(r_, A.test)])
            cv = np.concatenate(MB[(r_, y0)] + MB[(r_, y1)])
            r = {}
            for ti, t in enumerate(A.targets):
                c_ = ch.index(t)
                f1 = lambda m_, ti=ti, c_=c_: float(np.sqrt(m_[:, ti].mean()) * std[c_])  # noqa: E731
                f0 = lambda m_, c_=c_: float(np.sqrt(m_[:, c_].mean()) * std[c_])  # noqa: E731
                n = (len(mt) // A.block) * A.block
                rng = np.random.default_rng(0)
                dr = (rng.integers(0, n // A.block, size=(2000, n // A.block))[:, :, None] * A.block + np.arange(A.block)).reshape(2000, -1)
                bs = np.array([f1(mt[i]) - f0(bt[i]) for i in dr])
                r[t] = dict(test=round(f1(mt), 3), cv=round(f1(cv), 3), bands=round(f0(bt), 3), bands_cv=round(f0(cvb), 3),
                            minus_bands=[round(f1(mt) - f0(bt), 3)] + np.percentile(bs, [2.5, 97.5]).round(3).tolist())
            res[f"balance@{r_}"] = r
            print(f"day {day:2d} balance ridge {r_:g}  " + "  ".join(f"{t} {v['test']} (bands {v['bands']}) cv {v['cv']} (bands {v['bands_cv']}) {v['minus_bands']}"
                                                                    for t, v in r.items()), flush=True)
    out[day] = res
if A.json:
    json.dump(out, open(A.json, "w"), indent=1)
