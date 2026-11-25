"""Candidates for the last condition on the mean (fitted on the years the cores never saw, scored on 2017).

Every candidate reads the mean as  clim + sum_k theta_k phi_k  with features phi built from the cores' forecast
anomalies (fields of scripts/diag_cond.py); theta per lead and channel by least squares (area weighted, a small ridge).

    bands      the condition of now: one strength per Littlewood-Paley band of the cores' average  (6)
    free2/3    one weight per core and band (2 or 3 cores); free: one per core of --cores (any number)
    season     the strength varies with the season: bands x [1, cos, sin] of the verifying day
    zones      the strength varies with the latitude: bands x hat functions of the latitude (a partition of unity)
    disagree   the strength varies with the local disagreement of the two cores (smoothed, l <= 7): the uncertainty
    persist    the initial anomaly, banded, as a second predictor
    bias       a constant field per lead (spherical harmonics l <= 4): the drift of the model and the climate trend
    geo        the strength as a smooth field on the sphere: bands x spherical harmonics l <= lg (storm tracks, land)
    prev       the forecast of the day before, banded (a filter in the lead)
    lagK       the lagged forecasts: the cores' average started 1 .. K days earlier, at the same verifying day (lead + k),
               banded (needs dumps of every day, --ic-stride 1, and lead + K inside the dump; the first K starts are dropped
               for every candidate, and the interval is a block bootstrap, --block)
    clim16     the mean of the invariant measure itself: the daily climatology of the 16 training years (the record of
               scripts/mix_matrix.py, +-15 days) minus the climatology in use, at the verifying day, banded
    trend      the drift of that climatology: the linear trend of the 16 years per day of the year (+-45 days) times the
               years from their centre to the year of the forecast, banded
    pastN      the mean of the last N days up to the start (the slow background), banded; needs dumps of every day
    nl         a strength that depends on the size of the forecast anomaly: f |f| per band
    regime     a strength that depends on the state of the flow: bands x the forecast's z500 over the two polar caps
               and the tropics (annular modes, the tropical mean)
    math       the mathematical cores of math_core.py (--math lim,kaf), banded: one weight per core and band
    level      the other variables at the same level, banded (z, t, u, v, q): balance between the fields
    vert       the same variable at the neighbouring levels, banded

Model selection: fit on 2016, score 2020 and the reverse (cv); the test year is 2017 (fit on 2016 + 2020; fit on 2016
alone as the condition of now was).

    python scripts/cond_fit.py --days 10 --exps bands free2 season zones disagree persist bias level vert --json out.json
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpk.sphere import SHT, bands as lp_bands  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="runs/fpk/cond")
ap.add_argument("--fit", type=int, nargs="+", default=[2016, 2020])
ap.add_argument("--test", type=int, default=2017)
ap.add_argument("--days", type=int, nargs="+", default=[10])
ap.add_argument("--exps", nargs="+", default=["bands"])
ap.add_argument("--cores", default="core_gngmz8W,core_ngmz8W")
ap.add_argument("--third", default="core_gngmz8L")
ap.add_argument("--math", default="", help="math cores (math_core.py tags) for the candidate 'math'")
ap.add_argument("--ridge", type=float, default=1e-4)
ap.add_argument("--batch", type=int, default=12)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--zones", type=float, nargs="+", default=[-90, -45, 0, 45, 90], help="hat centres (deg)")
ap.add_argument("--lb", type=int, default=4, help="bias: degrees l <= lb")
ap.add_argument("--lg", type=int, default=2, help="geo: the strength as a field of degree l <= lg")
ap.add_argument("--max-ics", type=int, default=0)
ap.add_argument("--block", type=int, default=0, help="block length (ICs) of the bootstrap; 0: 1, or 15 with lagged forecasts")
ap.add_argument("--ref", default="bands", help="a second reference of the bootstrap (the increment of a candidate over it)")
ap.add_argument("--test-drop-mod", type=int, default=0, help="score the test year without the ICs whose day is a multiple of N "
                "(--test-drop-mod 6 on a daily dump: the starts the 60-IC set never used, an untouched test)")
ap.add_argument("--json", default="")
ap.add_argument("--save-theta", default="")
ap.add_argument("--save-pred", nargs="*", default=[], help="TAG:EXP writes the conditioned mean of the candidate EXP as one more dump "
                "<dir>/<year>_<TAG>.npy (only the leads of --days; the fit years with the strengths that never saw them)")
A = ap.parse_args()
torch.set_num_threads(A.threads)
CORES = A.cores.split(",")
CW, CN = CORES[0], CORES[1]
PARTS = [p for e in A.exps for p in e.split("+")]
NLAG = max([int(p[3:] or 1) for p in PARTS if p.startswith("lag")] + [0])
PAST = sorted({int(p[4:]) for p in PARTS if p.startswith("past")})
KLAG = max([NLAG] + [n - 1 for n in PAST])                                        # the first starts that are dropped
MATH = [t for t in A.math.split(",") if t]
YEARS = list(dict.fromkeys(A.fit + [A.test]))


def load(y):
    m = json.load(open(f"{A.dir}/{y}_meta.json"))
    d = dict(meta=m, o=np.load(f"{A.dir}/{y}_o.npy", mmap_mode="r"), x0=np.load(f"{A.dir}/{y}_x0.npy", mmap_mode="r"),
             ics=np.asarray(m["ics"]))
    for t in dict.fromkeys(m["cores"] + CORES + [A.third] + MATH):
        if os.path.exists(f"{A.dir}/{y}_{t}.npy"):                                    # other dumps may be linked in
            d[t] = np.load(f"{A.dir}/{y}_{t}.npy", mmap_mode="r")
    d["n"] = min(len(d["ics"]), A.max_ics or 10 ** 9)
    d["rows"] = np.arange(KLAG, d["n"])
    if y == A.test and A.test_drop_mod:
        d["rows"] = d["rows"][d["ics"][d["rows"]] % A.test_drop_mod != 0]
    assert not KLAG or np.all(np.diff(d["ics"]) == 1), "lagged forecasts need a dump of every day"
    return d


D = {y: load(y) for y in YEARS}
meta = D[A.test]["meta"]
ch = meta["channels"]
C = len(ch)
std = np.asarray(meta["std"])
lat = np.asarray(meta["lat"])
H, W = D[A.test]["o"].shape[-2:]
w = np.cos(np.deg2rad(lat))
w = torch.tensor(w / w.mean(), dtype=torch.float32)[:, None]               # (H, 1)
sht = SHT(lat)
bj, nb = lp_bands(sht.L)
masks = [(bj == b).float()[:, None] for b in range(nb)]
low = (torch.arange(sht.L) <= 7).float()[:, None]
SC = [i for i in range(C) if ch[i] != "tisr"]
Z = ch.index("z500")


def banded(f):
    """(B, C, H, W) -> (B, C, nb, H, W)."""
    c = sht.analysis(f)
    x = torch.stack([sht.synthesis(c * m) for m in masks], 2)
    x[:, :, -1] += f - x.sum(2)
    return x


# ---------------------------------------------------------------- channel maps for the multivariate candidates
LEV = ["50", "100", "200", "250", "300", "400", "500", "600", "700", "850", "925", "1000"]
SURF = dict(t2m=["t2m", "skt", "t1000", "msl", "sp"], sp=["sp", "msl", "z1000", "t2m", "skt"],
            msl=["msl", "sp", "z1000", "t2m", "skt"], tcwv=["tcwv", "q850", "q700", "q925", "q1000"],
            skt=["skt", "t2m", "t1000", "msl", "sp"], tisr=["tisr"])
lev_map, vert_map = [], []
for c in ch:
    if c in SURF:
        lev_map.append([ch.index(x) for x in SURF[c]] + [-1] * (5 - len(SURF[c])))
        vert_map.append([ch.index(c), -1, -1])
    else:
        v, p = c[0], c[1:]
        lev_map.append([ch.index(c)] + [ch.index(x + p) for x in "ztuvq" if x != v])
        i = LEV.index(p)
        vert_map.append([ch.index(c)] + [ch.index(v + LEV[i + s]) if 0 <= i + s < len(LEV) else -1 for s in (-1, 1)])
lev_map, vert_map = torch.tensor(lev_map), torch.tensor(vert_map)


def gather(Bf, mp):
    """(B, C, nb, H, W) and a (C, k) map (-1 = zeros) -> (B, C, k * nb, H, W)."""
    z = torch.cat([Bf, torch.zeros_like(Bf[:, :1])], 1)
    idx = torch.where(mp < 0, Bf.shape[1], mp)
    g = z[:, idx]                                                            # (B, C, k, nb, H, W)
    return g.reshape(g.shape[0], C, -1, H, W)


# hats in the latitude
zc = np.asarray(A.zones, dtype=float)
hats = np.stack([np.interp(lat, zc, np.eye(len(zc))[k]) for k in range(len(zc))])   # (nz, H)
HATS = torch.tensor(hats, dtype=torch.float32)[:, :, None]                            # (nz, H, 1)
# constant fields: real spherical harmonics l <= lb
Y = []
for l in range(A.lb + 1):
    for m in range(l + 1):
        for part in ((1.0, 0.0), (0.0, 1.0)) if m else ((1.0, 0.0),):
            c = torch.zeros(sht.L, sht.ana.shape[0], dtype=torch.complex64)
            c[l, m] = complex(*part)
            y = sht.synthesis(c)
            Y.append(y / y.pow(2).mul(w).mean().sqrt())
YLM = torch.stack(Y)                                                                   # (nY, H, W)
NG = (A.lg + 1) ** 2                                                                   # geo: the fields l <= lg


C16 = TR = None
if "clim16" in PARTS or "trend" in PARTS:                                                                 # the 16 years' daily mean anomaly, +-15 days
    am = json.load(open(f"{A.dir}/archive_meta.json"))
    arc = np.load(f"{A.dir}/archive_anom.npy", mmap_mode="r")
    acc, cnt, s0 = np.zeros((366, C, H, W), dtype=np.float64), np.zeros(366), 0
    for nd in am["days"]:
        acc[:nd] += np.asarray(arc[s0:s0 + nd], dtype=np.float32)
        cnt[:nd] += 1
        s0 += nd

    def window(x, n, pad):                                                            # the mean over +-pad days of the year
        ext = np.concatenate([x[-pad:], x, x[:pad]])
        cex = np.concatenate([n[-pad:], n, n[:pad]])
        cs_, cc_ = np.cumsum(np.concatenate([np.zeros_like(ext[:1]), ext]), 0), np.cumsum(np.concatenate([[0.0], cex]))
        return ((cs_[2 * pad + 1:] - cs_[:-2 * pad - 1]) / (cc_[2 * pad + 1:] - cc_[:-2 * pad - 1])[:, None, None, None]).astype(np.float32)

    C16 = window(acc, cnt, 15)
    print(json.dumps(dict(clim16_rms_z500=float(np.sqrt((C16[:, Z] ** 2 * w.numpy()).mean())))), flush=True)
    if "trend" in PARTS:                                                              # least squares slope per year, +-45 days
        yc = float(np.mean(am["years"]))
        acc[:], s0, cw = 0, 0, np.zeros(366)
        for yy, nd in zip(am["years"], am["days"]):
            acc[:nd] += (yy - yc) * np.asarray(arc[s0:s0 + nd], dtype=np.float32)
            cw[:nd] += (yy - yc) ** 2
            s0 += nd
        TR, TRC = window(acc, cw, 45), yc
        print(json.dumps(dict(trend_rms_z500_per_decade=float(10 * np.sqrt((TR[:, Z] ** 2 * w.numpy()).mean())))), flush=True)
    del acc
CAPS = [torch.tensor(((lat >= 60) / max((lat >= 60).mean(), 1e-9)), dtype=torch.float32)[:, None] * w,
        torch.tensor(((lat <= -60) / max((lat <= -60).mean(), 1e-9)), dtype=torch.float32)[:, None] * w,
        torch.tensor(((np.abs(lat) <= 20) / max((np.abs(lat) <= 20).mean(), 1e-9)), dtype=torch.float32)[:, None] * w]


def season(doy, k=1):
    ph = 2 * math.pi * doy / 365.25
    return torch.tensor(np.stack([np.ones_like(ph)] + sum([[np.cos(j * ph), np.sin(j * ph)] for j in range(1, k + 1)], []), 1),
                        dtype=torch.float32)                                           # (B, 1 + 2k)


DBAR = {}


def extra(name, P):
    """What a candidate adds to the bands of the cores' average (the span of bands + extra is the candidate)."""
    Bm = P["Bm"]
    if name == "free2":
        return P["B"](CW)
    if name == "free":
        return torch.cat([P["B"](t) for t in CORES[:-1]], 2)
    if name == "math":
        return torch.cat([P["B"](t) for t in MATH], 2)
    if name == "free3":
        return torch.cat([P["B"](CW), P["B"](A.third)], 2)
    if name.startswith("season"):
        s = season(P["doy"], int(name[6:] or 1))[:, 1:]
        return (s[:, None, :, None, None, None] * Bm[:, :, None]).flatten(2, 3)
    if name == "zones":
        return (HATS[:-1, None] * Bm[:, :, None]).flatten(2, 3)
    if name == "disagree":
        return Bm * (P["Dt"] - 1)[:, :, None]
    if name == "persist":
        return P["B0"]
    if name == "bias":
        return YLM.expand(Bm.shape[0], C, -1, H, W)
    if name == "bias_s":
        s = season(P["doy"])
        yl = (YLM[None, :, None] * s[:, None, :, None, None]).flatten(1, 2)           # (B, nY*3, H, W)
        return yl[:, None].expand(-1, C, -1, -1, -1)
    if name == "geo":
        return (YLM[1:NG, None] * Bm[:, :, None]).flatten(2, 3)
    if name == "prev":
        return P["Bp"]
    if name.startswith("lag"):
        return torch.cat([P["Bl"][k] for k in range(int(name[3:] or 1))], 2)
    if name.startswith("past"):
        return P["Bpast"][int(name[4:])]
    if name == "clim16":
        return P["Bc16"]
    if name == "trend":
        return P["Btr"]
    if name == "nl":
        return Bm * Bm.abs()
    if name == "regime":
        return (P["idx"][:, None, :, None, None, None] * Bm[:, :, None]).flatten(2, 3)
    if name == "level":
        return gather(Bm, lev_map[:, 1:])
    if name == "vert":
        return gather(Bm, vert_map[:, 1:])
    raise SystemExit(f"unknown candidate {name}")


def features(name, P):
    if name == "raw":
        return P["fm"][:, :, None]
    return torch.cat([P["Bm"]] + [extra(p, P) for p in name.split("+") if p != "bands"], 2)


def pieces(d, idx, L):
    f = {}
    get = lambda t, l=L: torch.from_numpy(np.asarray(d[t][idx, l], dtype=np.float32))  # noqa: E731
    fw, fn = get(CW), get(CN)
    fm = (fw + fn + sum(get(t) for t in CORES[2:])) / len(CORES)                     # the average of all the cores
    P = dict(fm=fm, doy=d["ics"][idx] + L + 1.0)
    P["Bm"] = banded(P["fm"])

    def B(t):
        if t not in f:
            f[t] = banded(get(t))
        return f[t]
    P["B"] = B
    if any("disagree" in e for e in A.exps):
        dd = (fw - fn) ** 2
        Ds = sht.synthesis(sht.analysis(dd) * low).clamp_min(0)
        P["Dt"] = Ds / torch.as_tensor(DBAR[L], dtype=torch.float32)[None, :, None, None]
    if any("prev" in e for e in A.exps):
        P["Bp"] = banded(sum(get(t, L - 1) for t in CORES) / len(CORES) if L else torch.from_numpy(np.asarray(d["x0"][idx], dtype=np.float32)))
    if NLAG:                                                                          # started k days earlier, lead + k
        P["Bl"] = [banded(sum(torch.from_numpy(np.asarray(d[t][idx - k, L + k], dtype=np.float32)) for t in CORES) / len(CORES))
                   for k in range(1, NLAG + 1)]
    if PAST:                                                                          # the mean of the last N days up to the start
        P["Bpast"] = {n: banded(torch.from_numpy(np.stack([np.asarray(d["x0"][i - n + 1:i + 1], dtype=np.float32).mean(0) for i in idx])))
                      for n in PAST}
    if "clim16" in PARTS:
        P["Bc16"] = banded(torch.from_numpy(C16[np.minimum(d["ics"][idx] + L + 1, 365)]))
    if "trend" in PARTS:
        P["Btr"] = banded(torch.from_numpy(TR[np.minimum(d["ics"][idx] + L + 1, 365)])) * (d["meta"]["year"] - TRC)
    if "regime" in PARTS:
        z = fm[:, Z]                                                                  # (B, H, W)
        P["idx"] = torch.stack([(z * CAP).mean((-2, -1)) for CAP in CAPS], 1)        # (B, 3)
    if any("persist" in e for e in A.exps):
        P["B0"] = banded(torch.from_numpy(np.asarray(d["x0"][idx], dtype=np.float32)))
    return P, torch.from_numpy(np.asarray(d["o"][idx, L], dtype=np.float32))


def batches(d):
    r = d["rows"]
    for i in range(0, len(r), A.batch):
        yield r[i:i + A.batch]


SP = {}
for sp in A.save_pred:
    tg_, ex_ = sp.split(":", 1)
    assert ex_ in A.exps and len(A.fit) == 2, "--save-pred needs its candidate among --exps and two fit years"
    SP[ex_] = {y: np.lib.format.open_memmap(f"{A.dir}/{y}_{tg_}.npy", mode="w+", dtype=np.float16, shape=(len(D[y]["ics"]), 10, C, H, W)) for y in YEARS}
out = dict(meta=dict(fit=A.fit, test=A.test, cores=[CW, CN], ridge=A.ridge, zones=A.zones, lb=A.lb, days=A.days, dir=A.dir, math=A.math,
                     ref=A.ref, test_drop_mod=A.test_drop_mod, block=A.block, test_ics=int(len(D[A.test]["rows"]))), days={})
t0 = time.time()
for day in A.days:
    L = day - 1
    if any("disagree" in e for e in A.exps):                                          # the mean disagreement, fit years
        acc, n = 0, 0
        for y in A.fit:
            d = D[y]
            for idx in batches(d):
                dd = (torch.from_numpy(np.asarray(d[CW][idx, L], dtype=np.float32)) - torch.from_numpy(np.asarray(d[CN][idx, L], dtype=np.float32))) ** 2
                acc = acc + (dd * w).mean((-2, -1)).sum(0).double()
                n += len(idx)
        DBAR[L] = (acc / n).numpy()
    # pass 1: normal equations per year
    NE = {e: {} for e in A.exps}
    for y in A.fit:
        d = D[y]
        for idx in batches(d):
            P, o = pieces(d, idx, L)
            for e in A.exps:
                X = features(e, P)                                                    # (B, C, K, H, W)
                Xw = X * w
                S = torch.einsum("bckhw,bcjhw->ckj", Xw, X).double() / (H * W)
                c = torch.einsum("bckhw,bchw->ck", Xw, o).double() / (H * W)
                a = NE[e].setdefault(y, [0, 0, 0])
                a[0], a[1], a[2] = a[0] + S, a[1] + c, a[2] + len(idx)
        print(json.dumps(dict(day=day, pass1=y, sec=round(time.time() - t0))), flush=True)

    def solve(e, ys):
        S = sum(NE[e][y][0] for y in ys) / sum(NE[e][y][2] for y in ys)
        c = sum(NE[e][y][1] for y in ys) / sum(NE[e][y][2] for y in ys)
        K = S.shape[-1]
        tr = torch.diagonal(S, dim1=-2, dim2=-1).sum(-1).clamp_min(1e-12)
        S = S + (A.ridge * tr / K)[:, None, None] * torch.eye(K, dtype=S.dtype)
        return torch.linalg.solve(S, c[..., None])[..., 0].float()                   # (C, K)

    TH = {}
    for e in A.exps:
        if e == "raw":
            TH[e] = {k: torch.ones(C, 1) for k in ("all", *A.fit, "f0")}
            continue
        TH[e] = {"all": solve(e, A.fit), "f0": solve(e, A.fit[:1])}
        if len(A.fit) == 2:
            TH[e][A.fit[0]], TH[e][A.fit[1]] = solve(e, [A.fit[1]]), solve(e, [A.fit[0]])  # theta that never saw y
    # pass 2: per-IC mse
    MSE = {e: {} for e in A.exps}
    for y in YEARS:
        d = D[y]
        for idx in batches(d):
            P, o = pieces(d, idx, L)
            for e in A.exps:
                X = features(e, P)
                keys = ["all", "f0"] if y == A.test else ([y] if len(A.fit) == 2 else [])
                for k in keys:
                    pred = torch.einsum("ck,bckhw->bchw", TH[e][k], X)
                    if e in SP and k != "f0":
                        SP[e][y][idx, L] = pred.numpy()
                    m = ((o - pred) ** 2 * w).mean((-2, -1)).double().numpy()
                    MSE[e].setdefault((y, k), []).append(m)
        print(json.dumps(dict(day=day, pass2=y, sec=round(time.time() - t0))), flush=True)
    MSE = {e: {k: np.concatenate(v) for k, v in r.items()} for e, r in MSE.items()}
    res = {}
    rz = lambda m, c: float(np.sqrt(m[:, c].mean(0)) * std[c])  # noqa: E731
    nr = lambda m: float(np.sqrt(m.mean(0))[SC].mean())  # noqa: E731
    ref = MSE["bands"][(A.test, "all")] if "bands" in MSE else None
    rng = np.random.default_rng(0)
    nT = len(D[A.test]["rows"])
    ref2 = MSE[A.ref][(A.test, "all")] if A.ref != "bands" and A.ref in MSE else None
    blk = A.block or (15 if KLAG else 1)                                              # daily starts are not independent
    nB = nT // blk
    draws = (rng.integers(0, nB, size=(2000, nB))[:, :, None] * blk + np.arange(blk)).reshape(2000, -1)
    for e in A.exps:
        mt, m0 = MSE[e][(A.test, "all")], MSE[e][(A.test, "f0")]
        r = dict(K=int(TH[e]["all"].shape[1]), z500=rz(mt, Z), z500_fit2016=rz(m0, Z), t850=rz(mt, ch.index("t850")),
                 t2m=rz(mt, ch.index("t2m")), nrmse=nr(mt))
        if len(A.fit) == 2:
            cv = np.concatenate([MSE[e][(y, y)] for y in A.fit])
            r["cv_z500"], r["cv_nrmse"] = rz(cv, Z), nr(cv)
        if ref is not None and e != "bands":
            bs = np.array([rz(mt[i], Z) - rz(ref[i], Z) for i in draws])
            r["z500_minus_bands"] = [round(r["z500"] - rz(ref, Z), 2)] + np.percentile(bs, [2.5, 97.5]).round(2).tolist()
        if ref2 is not None and e != A.ref:
            bs2 = np.array([rz(mt[i], Z) - rz(ref2[i], Z) for i in draws])
            r["z500_minus_ref"] = [round(r["z500"] - rz(ref2, Z), 2)] + np.percentile(bs2, [2.5, 97.5]).round(2).tolist()
            if len(A.fit) == 2:
                cvr = np.concatenate([MSE[A.ref][(y, y)] for y in A.fit])
                r["cv_minus_ref"] = round(r["cv_z500"] - rz(cvr, Z), 2)
        res[e] = r
        print(f"day {day:2d} {e:24s} K {r['K']:3d}  z500 {r['z500']:6.1f} (fit16 {r['z500_fit2016']:6.1f})  cv {r.get('cv_z500', 0):6.1f}  "
              f"t850 {r['t850']:.3f} t2m {r['t2m']:.3f} nrmse {r['nrmse']:.4f} cv {r.get('cv_nrmse', 0):.4f}  "
              f"{r.get('z500_minus_bands', '')}" + (f"  vs {A.ref} {r['z500_minus_ref']} cv {r.get('cv_minus_ref', '')}" if "z500_minus_ref" in r else ""), flush=True)
    out["days"][day] = res
    if A.save_theta:
        torch.save({e: TH[e]["all"] for e in A.exps}, f"{A.save_theta}_d{day}.pt")
for v in SP.values():
    for m in v.values():
        m.flush()
if A.json:
    json.dump(out, open(A.json, "w"), indent=1)
