"""What is left for a condition from the invariant measure on the mean of one channel (감자 2026-11-25 16:34 KST: "물리도
데이터도 아닌 측도적 core 로 cover 할 수 없을까, 불변 측도 조건을 조금 더 조사해 보면"; first dig, then design).

The condition of now reads the mean as  clim + sum_b g_b f_b  in 6 isotropic Littlewood-Paley bands of the cores' average
f: the conditional expectation of the truth under the invariant measure when that measure is isotropic and Gaussian. This
script measures what a finer reading of the same measure adds, one channel and one lead at a time (everything in memory,
seconds per candidate; the fields are those of scripts/diag_cond.py):

    bands     the condition of now (6)
    deg       one strength per degree l: the finest isotropic condition
    mband     bands x classes of the zonal wavenumber (m = 0, 1-3, >= 4): zonal means against waves
    eof       one strength per group of EOFs of the measure itself (the 16 training years): the measure's own coordinates
    phase     the zonal and meridional derivatives of every band: a systematic displacement of the pattern
    clim16    the daily mean of the 16 years minus the climatology in use, banded (cond_fit.py)
    vert      the same variable at the two neighbouring levels, banded (cond_fit.py)
    x0        the initial anomaly, banded
    hyb<k>    the measure core continued from the cores' day-k forecast (scripts/measure_core.py), banded;
              hyb<k>@<tag> reads the variant written with --tag <tag>
    full      a full linear map in the leading R EOFs for what the bands leave (ridge; R and the ridge by cv): the upper
              bound of every linear condition on this channel's forecast (--full-x0 adds the initial anomaly's EOFs)

Candidates join with "+", and every one contains the bands unless it starts with "only:". Fit on 2016 + 2020, cv 2016 <->
2020, test 2017 with a bootstrap against the bands; "oracle" is the same candidate fitted on the test year itself (what a
perfect knowledge of its strengths would give: the ceiling of that family).

    python scripts/measure_cond.py --day 10 --exps bands deg mband eof phase bands+clim16+vert full
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
ap.add_argument("--fit", type=int, nargs="+", default=[2016, 2020])
ap.add_argument("--test", type=int, default=2017)
ap.add_argument("--day", type=int, default=10)
ap.add_argument("--chan", default="z500")
ap.add_argument("--cores", default="core_gngmz8W,core_ngmz8W")
ap.add_argument("--exps", nargs="+", default=["bands"])
ap.add_argument("--ridge", type=float, default=1e-4)
ap.add_argument("--full-r", type=int, nargs="+", default=[10, 20, 40, 80, 160])
ap.add_argument("--full-lam", type=float, nargs="+", default=[0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
ap.add_argument("--full-x0", action="store_true")
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--json", default="")
A = ap.parse_args()
torch.set_num_threads(A.threads)
CORES = A.cores.split(",")
L = A.day - 1
YEARS = list(dict.fromkeys(A.fit + [A.test]))
PARTS = {p for e in A.exps for p in e.replace("only:", "").split("+")}

meta = json.load(open(f"{A.dir}/{A.test}_meta.json"))
ch = meta["channels"]
Z = ch.index(A.chan)
std = float(meta["std"][Z])
lat = np.asarray(meta["lat"])
LEV = ["50", "100", "200", "250", "300", "400", "500", "600", "700", "850", "925", "1000"]
NEI = []
if A.chan[1:] in LEV and A.chan[0] in "ztuvq":
    i = LEV.index(A.chan[1:])
    NEI = [ch.index(A.chan[0] + LEV[i + s]) for s in (-1, 1) if 0 <= i + s < len(LEV)]


def load(y):
    m = json.load(open(f"{A.dir}/{y}_meta.json"))
    g = lambda t, c, l=L: torch.from_numpy(np.asarray(np.load(f"{A.dir}/{y}_{t}.npy", mmap_mode="r")[:, l, c], dtype=np.float32))  # noqa: E731
    d = dict(ics=np.asarray(m["ics"]), o=g("o", Z), f=sum(g(t, Z) for t in CORES) / len(CORES))
    if "vert" in PARTS:
        d["fv"] = [sum(g(t, c) for t in CORES) / len(CORES) for c in NEI]
    if "x0" in PARTS or A.full_x0:
        d["x0"] = torch.from_numpy(np.asarray(np.load(f"{A.dir}/{y}_x0.npy", mmap_mode="r")[:, Z], dtype=np.float32))
    hy = [p for p in PARTS if p.startswith("hyb")]
    if hy:
        d["hyb"] = {p: torch.from_numpy(np.load(f"{A.dir}/{y}_hyb{p.partition('@')[2]}_{A.chan}_d{A.day}.npz")[f"k{p[3:].partition('@')[0]}"]
                                         .astype(np.float32)) for p in hy}
    d["n"] = len(d["ics"])
    return d


D = {y: load(y) for y in YEARS}
H, W = D[A.test]["o"].shape[-2:]
DIM = H * W
wl = np.cos(np.deg2rad(lat))
wl = torch.tensor(wl / wl.mean(), dtype=torch.float32)[:, None]            # (H, 1)
SW = wl.sqrt().expand(H, W).reshape(-1)                                    # the weighted space: a = f sqrt(w)
sht = SHT(lat)
bj, nb = lp_bands(sht.L)
M = sht.ana.shape[0]
masks = [(bj == b).float()[:, None] for b in range(nb)]
mm = torch.arange(M)[None, :]
MCL = [(mm == 0).float(), ((mm >= 1) & (mm <= 3)).float(), (mm >= 4).float()]
ll = torch.arange(sht.L)[:, None]
valid = (mm <= ll).float()


def banded(f, ms=masks, rest=True):
    c = sht.analysis(f)
    x = torch.stack([sht.synthesis(c * m) for m in ms], 1)                 # (N, K, H, W)
    if rest:
        x[:, -1] += f - x.sum(1)
    return x


def tow(x):
    """(N, K, H, W) on the grid -> (N, K, DIM) in the weighted space."""
    return x.flatten(-2) * SW


# ---------------------------------------------------------------- the measure's own coordinates: EOFs of the 16 years
E = EV = C16 = None
if PARTS & {"eof", "full", "clim16"} or any(p.startswith("eof") for p in PARTS):
    am = json.load(open(f"{A.dir}/archive_meta.json"))
    arc = torch.from_numpy(np.asarray(np.load(f"{A.dir}/archive_anom.npy", mmap_mode="r")[:, Z], dtype=np.float32))   # (5844, H, W)
    if "clim16" in PARTS:
        acc, cnt, s0 = np.zeros((366, H, W)), np.zeros(366), 0
        for nd in am["days"]:
            acc[:nd] += arc[s0:s0 + nd].numpy()
            cnt[:nd] += 1
            s0 += nd
        pad = 15
        ext, cex = np.concatenate([acc[-pad:], acc, acc[:pad]]), np.concatenate([cnt[-pad:], cnt, cnt[:pad]])
        cs_, cc_ = np.cumsum(np.concatenate([np.zeros_like(ext[:1]), ext]), 0), np.cumsum(np.concatenate([[0.0], cex]))
        C16 = torch.from_numpy(((cs_[2 * pad + 1:] - cs_[:-2 * pad - 1]) / (cc_[2 * pad + 1:] - cc_[:-2 * pad - 1])[:, None, None]).astype(np.float32))
    a = arc.flatten(1) * SW
    _, S, Vh = torch.linalg.svd(a, full_matrices=False)
    E, EV = Vh, (S ** 2 / len(a) / DIM)                                    # rows: EOFs, orthonormal in the weighted space
    ex = (EV.cumsum(0) / EV.sum())
    print(json.dumps(dict(archive_days=len(a), eof_explained={k: round(float(ex[k - 1]), 4) for k in (1, 4, 16, 64, 256, 1024)},
                          clim_rms=round(float(EV.sum().sqrt()) * std, 1))), flush=True)
    del arc, a
EDGES = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, DIM]

FE = {}


def feat(p, y):
    """One part's features in the weighted space, (N, K, DIM)."""
    if (p, y) in FE:
        return FE[(p, y)]
    d = D[y]
    f = d["f"]
    if p == "bands":
        x = tow(banded(f))
    elif p == "deg":
        x = tow(banded(f, [(ll == l).float() for l in range(sht.L)]))
    elif p == "mband":
        ms = [masks[b] * mc * valid for b in range(nb) for mc in MCL]
        x = tow(banded(f, [m for m in ms if m.sum() > 0], rest=False))
    elif p.startswith("eof"):
        ed = EDGES if p == "eof" else list(range(0, int(p[3:]) + 1)) + [DIM]      # eofN: one strength for each of the first N
        cf = (f.flatten(1) * SW) @ E.T
        x = torch.stack([cf[:, a_:b_] @ E[a_:b_] for a_, b_ in zip(ed[:-1], ed[1:])], 1)
    elif p == "phase":
        c = sht.analysis(f)
        gx, gy = zip(*[sht.grad(c * m) for m in masks[1:]])
        cosl = torch.tensor(np.cos(np.deg2rad(lat)), dtype=torch.float32)[:, None]
        x = tow(torch.cat([torch.stack(gx, 1) * cosl * sht.radius, torch.stack(gy, 1) * sht.radius], 1))
    elif p == "clim16":
        x = tow(banded(C16[np.minimum(d["ics"] + L + 1, 365)]))
    elif p == "vert":
        x = torch.cat([tow(banded(g)) for g in d["fv"]], 1)
    elif p == "x0":
        x = tow(banded(d["x0"]))
    elif p.startswith("hyb"):
        x = tow(banded(d["hyb"][p]))
    else:
        raise SystemExit(f"unknown candidate {p}")
    FE[(p, y)] = x
    return x


def features(e, y):
    only = e.startswith("only:")
    ps = e.replace("only:", "").split("+")
    if not only and "bands" not in ps and not any(p in ("deg", "mband") or p.startswith("eof") for p in ps):
        ps = ["bands"] + ps
    return torch.cat([feat(p, y) for p in ps], 1)


def normal(X, o):
    return (torch.einsum("nkd,njd->kj", X, X).double() / DIM, torch.einsum("nkd,nd->k", X, o).double() / DIM, len(X))


def solve(ne, ridge=A.ridge):
    n = sum(v[2] for v in ne)
    S, c = sum(v[0] for v in ne) / n, sum(v[1] for v in ne) / n
    K = S.shape[0]
    S = S + ridge * torch.trace(S).clamp_min(1e-12) / K * torch.eye(K, dtype=S.dtype)
    return torch.linalg.solve(S, c).float()


OW = {y: D[y]["o"].flatten(1) * SW for y in YEARS}
mse = lambda p, y: ((OW[y] - p) ** 2).mean(1).double().numpy()  # noqa: E731
rz = lambda m: float(np.sqrt(m.mean()) * std)  # noqa: E731
rng = np.random.default_rng(0)
nT = D[A.test]["n"]
draws = rng.integers(0, nT, size=(2000, nT))
out, REF, PRED = {}, None, {}


def report(e, mt, cv, K, orc=None, extra=""):
    global REF
    r = dict(K=K, test=rz(mt), cv=rz(cv))
    if orc is not None:
        r["oracle"] = rz(orc)
    if e == "bands":
        REF = mt
    elif REF is not None:
        bs = np.array([rz(mt[i]) - rz(REF[i]) for i in draws])
        r["minus_bands"] = [round(r["test"] - rz(REF), 2)] + np.percentile(bs, [2.5, 97.5]).round(2).tolist()
    out[e] = r
    print(f"day {A.day:2d} {A.chan} {e:28s} K {K:4d}  test {r['test']:6.1f}  cv {r['cv']:6.1f}  oracle {r.get('oracle', 0):6.1f}  "
          f"{r.get('minus_bands', '')} {extra}", flush=True)


for e in [x for x in A.exps if x != "full"]:
    X = {y: features(e, y) for y in YEARS}
    NE = {y: normal(X[y], OW[y]) for y in YEARS}
    th = solve([NE[y] for y in A.fit])
    mt = mse(torch.einsum("k,nkd->nd", th, X[A.test]), A.test)
    cvy = [mse(torch.einsum("k,nkd->nd", solve([NE[z] for z in A.fit if z != y]), X[y]), y) for y in A.fit]
    cv = np.concatenate(cvy)
    orc = mse(torch.einsum("k,nkd->nd", solve([NE[A.test]]), X[A.test]), A.test)
    report(e, mt, cv, X[A.test].shape[1], orc, "cv " + " ".join(f"{y}: {rz(c):.1f}" for y, c in zip(A.fit, cvy)))
    if e == "bands":
        tho = solve([NE[A.test]])
        Xb = X[A.test]
        fv = (Xb ** 2).mean((0, 2))
        er = ((torch.einsum("k,nkd->nd", th, Xb) - OW[A.test]) ** 2)
        ob = tow(banded(D[A.test]["o"]))
        eb = ((th[None, :, None] * Xb - ob) ** 2).mean((0, 2))
        rho = (Xb * ob).mean((0, 2)) / ((Xb ** 2).mean((0, 2)) * (ob ** 2).mean((0, 2))).sqrt().clamp_min(1e-12)
        out["bands_table"] = dict(g=th.tolist(), g_oracle=tho.tolist(), forecast_var_share=(fv / fv.sum()).tolist(),
                                  error_share=(eb / eb.sum()).tolist(), corr=rho.tolist(), truth_var_share=((ob ** 2).mean((0, 2)) / (ob ** 2).mean((0, 2)).sum()).tolist())
        for b in range(nb):
            print(f"    band {b}: g {th[b]:.3f} (oracle {tho[b]:.3f})  forecast var {fv[b] / fv.sum():.3f}  truth var {out['bands_table']['truth_var_share'][b]:.3f}  "
                  f"error share {eb[b] / eb.sum():.3f}  corr {rho[b]:.3f}", flush=True)
        del er

# ---------------------------------------------------------------- the full linear map in the leading EOFs, on what the bands leave
if "full" in A.exps:
    Xb = {y: features("bands", y) for y in YEARS}
    NEb = {y: normal(Xb[y], OW[y]) for y in YEARS}
    Rm = max(A.full_r)
    co = lambda y, k: (D[y][k].flatten(1) * SW) @ E[:Rm].T  # noqa: E731
    CF = {y: co(y, "f") for y in YEARS}
    CX = {y: co(y, "x0") for y in YEARS} if A.full_x0 else None

    def run(train, test, R, lam):
        th = solve([NEb[y] for y in train])
        P = lambda y: torch.cat([CF[y][:, :R]] + ([CX[y][:, :R]] if CX else []), 1)  # noqa: E731
        G = sum(P(y).T @ P(y) for y in train).double()
        rc = sum(P(y).T @ ((OW[y] - torch.einsum("k,nkd->nd", th, Xb[y])) @ E[:R].T) for y in train).double()
        B = torch.linalg.solve(G + lam * torch.trace(G) / G.shape[0] * torch.eye(G.shape[0], dtype=G.dtype), rc).float()
        return mse(torch.einsum("k,nkd->nd", th, Xb[test]) + (P(test) @ B) @ E[:R], test)

    best, grid = None, {}
    for R in A.full_r:
        for lam in A.full_lam:
            cv = np.concatenate([run([z for z in A.fit if z != y], y, R, lam) for y in A.fit])
            grid[f"{R}:{lam}"] = rz(cv)
            if best is None or rz(cv) < best[0]:
                best = (rz(cv), R, lam, cv)
        print(f"    full R {R:4d}  cv " + "  ".join(f"{lam:g}: {grid[f'{R}:{lam}']:.1f}" for lam in A.full_lam), flush=True)
    _, R, lam, cv = best
    report("full" + ("+x0" if A.full_x0 else ""), run(A.fit, A.test, R, lam), cv, R * R * (2 if CX else 1), run([A.test], A.test, R, lam), f"R {R} ridge {lam:g}")
    out["full_grid"] = grid
if A.json:
    json.dump(dict(meta=vars(A), res=out), open(A.json, "w"), indent=1)
