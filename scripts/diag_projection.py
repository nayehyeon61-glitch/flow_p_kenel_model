"""The invariant-measure (projection) condition on the MEAN of the kernel (감자 2026-11-24 "마지막 한방").

    mu K = mu : the climate pushed through the kernel is the climate again. Second moments: (variance of the mean)
    + (spread^2) = (variance of the climate); with the calibration the kernel already learns (spread^2 = error^2) this
    is  <f (o - f)> = 0 : the error is orthogonal to the forecast anomaly f = mean - climatology, i.e. the mean is the
    conditional expectation (the L2 projection of the truth anomaly o), scale by scale:  activity = ACC.

Measured here on the cores' trajectories (the kernels' means: z500 day 10 within 1 m2/s2 of their cores'), no training:
the anomaly of every core is split into the Littlewood-Paley bands of the degree l (fpk.sphere.bands: l = 0, 1, 2-3,
4-7, 8-15, 16-31), X = [f_core,band], and per IC, lead and channel the exact grid statistics S = <X X^T>, c = <X o>,
<o^2> (area means) are kept. Any linear mean  m = climatology + sum g_i X_i  then has  mse = g S g - 2 g c + <o^2>:

    raw       one core, or the average of the cores (the mixed kernel's mean)
    gain      one strength g of the condition per lead and channel          (m = clim + g f)
    bands     one g per band                                                 (the scales that lost their memory -> clim)
    free      one weight per core and band                                   (the mixture's weights, scale by scale)

fitted by least squares on one year and scored on another (the gains never see the scored year):

    python scripts/diag_projection.py stats --cores core_gngmz8W,core_ngmz8W --year 2016 --ic-stride 3
    python scripts/diag_projection.py stats --cores core_gngmz8W,core_ngmz8W --year 2017
    python scripts/diag_projection.py fit --fit runs/fpk/proj_2016.npz --test runs/fpk/proj_2017.npz --json docs/results/projection.json

stats: options it does not know go to ``run_fpk.py score`` (the + version: --data-root data/era5_1p5 --out
runs/fpk_plus --train-range 1984 2015 --device cuda); the file is --npz, else <out>/proj_<years>.npz. The kernels read
the strengths of such a file with ``run_fpk.py projk`` / ``devk --proj-init`` (a year the kernels were not trained on).
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
ap.add_argument("mode", choices=["stats", "fit"])
ap.add_argument("--cores", default="core_gngmz8W,core_ngmz8W")
ap.add_argument("--year", type=int, nargs="+", default=[2017], help="stats: one year, or several into one file (--npz)")
ap.add_argument("--ic-stride", type=int, default=6)
ap.add_argument("--npz", default="", help="stats: the file (default <out>/proj_<years>.npz)")
ap.add_argument("--fit", default="runs/fpk/proj_2016.npz")
ap.add_argument("--test", default="runs/fpk/proj_2017.npz")
ap.add_argument("--json", default="")
ap.add_argument("--ridge", type=float, default=1e-3, help="relative ridge of the free weights (toward the average of the cores)")
A, PASS = ap.parse_known_args()
if PASS and A.mode != "stats":
    ap.error(f"unrecognized arguments: {' '.join(PASS)}")
Q = [0, 2, 4, 6, 9]

if A.mode == "stats":
    sys.path.insert(0, os.path.dirname(__file__))
    import run_fpk as R  # noqa: E402
    from fpk.ensemble import rollout  # noqa: E402
    from fpk.env import DYN, contract  # noqa: E402
    from fpk.sphere import SHT, bands  # noqa: E402

    tags = A.cores.split(",")
    SS, CC, OO, ICS, YR = [], [], [], [], []
    t0 = time.time()
    am = lambda x: (x * w).mean((-2, -1)).cpu()  # noqa: E731

    def comps(f):
        """(B, L, 66, H, W) anomaly -> (B, L, 66, nb, H, W): its bands; what the fit of the transform leaves goes to the last."""
        c = sht.analysis(f)
        x = torch.stack([sht.synthesis(c * m) for m in masks], 3).double()
        x[:, :, :, -1] += f.double() - x.sum(3)
        return x

    for year in A.year:
        S = []

        def grab(a, rest):
            pa, cl, core = R.load_core(a)
            S.append(dict(a=a, pa=pa, cl=cl, core=core, K=R.PhysicsKernel(core)))

        R.cmd_score = grab
        for t in tags:
            R.main(["score", "--core-tag", t, "--arms", "core", "--threads", "4"] + PASS + ["--test-year", str(year), "--ic-stride", str(A.ic_stride)])
        a0, cl, pa = S[0]["a"], S[0]["cl"], S[0]["pa"]
        dev = torch.device(a0.device)
        te = R.D.load(pa, [a0.test_year])[a0.test_year]
        L = a0.leads
        ics = np.arange(a0.ic_start, len(te) - L, a0.ic_stride)
        dyn = np.asarray(DYN)
        eps = R.rkt().EPS
        cm, cs = cl.mean[dyn][:, None, None], cl.std[dyn][:, None, None] + eps
        w = np.cos(np.deg2rad(cl.lat))
        w = torch.tensor(w / w.mean(), dtype=torch.float64, device=dev)[:, None]
        sht = SHT(cl.lat).to(dev)
        bj, nb = bands(sht.L)
        masks = [(bj == b).float().to(dev)[:, None] for b in range(nb)]     # (L, 1) over the degree
        for i in range(0, len(ics), 2):                                       # 2 ICs at a time: X is 12 fields per channel and lead
            j = ics[i:i + 2]
            truth = torch.from_numpy(np.stack([te[k + 1:k + 1 + L][:, dyn] for k in j])).to(dev)
            clim = torch.from_numpy(np.stack([(cl.clim[np.minimum(np.arange(k + 1, k + 1 + L), 365)][:, dyn] - cm) / cs for k in j]).astype(np.float32)).to(dev)
            tis = torch.from_numpy(np.stack([te[k + 1:k + 1 + L, R.TISR] for k in j])).to(dev)
            X = []
            for s in S:
                core, K = s["core"], s["K"]
                W = int(getattr(core, "mem", 0))
                hist = torch.from_numpy(np.stack([te[np.maximum(np.arange(k - W, k), 0)][:, dyn] for k in j])).to(dev) if W else None
                with torch.no_grad():
                    ens, _ = rollout(lambda x, d, tn: R.core_step(K.core, x, d, tn), torch.from_numpy(te[j][:, dyn]).to(dev),
                                     torch.tensor(j, device=dev), tis, 1, L, hist=hist, start=K.start)
                    X.append(comps(ens[:, 0].float() - clim))
            X = torch.cat(X, 3)                                              # (B, L, 66, n, H, W), n = cores x bands
            o = (truth - clim).double()
            HW, nX = X.shape[-1] * X.shape[-2], X.shape[3]
            Xw = (X * w).reshape(len(j), L * X.shape[2], nX, HW)
            SS.append(np.stack([(Xw[b] @ X[b].reshape(L * X.shape[2], nX, HW).transpose(1, 2)).reshape(L, X.shape[2], nX, nX).cpu().numpy() for b in range(len(j))]) / HW)
            CC.append(am(X * o[:, :, :, None]).numpy())
            OO.append(am(o * o).numpy())
            print(json.dumps(dict(year=year, ics=int(i + len(j)), sec=round(time.time() - t0), rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)), flush=True)
        ICS.append(ics)
        YR.append(np.full(len(ics), year))
        del te, S
    out = A.npz or f"{a0.out}/proj_{'_'.join(str(y) for y in A.year)}.npz"
    np.savez_compressed(out, S=np.concatenate(SS), c=np.concatenate(CC), oo=np.concatenate(OO), ics=np.concatenate(ICS), year=np.concatenate(YR),
                        std=cl.std[dyn], cores=np.array(tags),
                        nb=nb, band=bj.cpu().numpy(), channels=np.array([contract.CHANNEL_NAMES[c] for c in DYN]))
    raise SystemExit(0)

# ------------------------------------------------------------------------------------------------------------ fit
F, T = np.load(A.fit), np.load(A.test)
cores, nb, std = [str(x) for x in T["cores"]], int(T["nb"]), T["std"]
ch = [str(x) for x in T["channels"]]
SC = [i for i in range(len(ch)) if ch[i] != "tisr"]
Z = ch.index("z500")
nc, n = len(cores), len(cores) * nb
lab = ["l=0", "1", "2-3", "4-7", "8-15", "16-31"][:nb]


def mse(d, g):
    """g (L, 66, n) -> (IC, L, 66) mse of clim + sum g X."""
    return np.einsum("lci,blcij,lcj->blc", g, d["S"], g) - 2 * np.einsum("lci,blci->blc", g, d["c"]) + d["oo"]


def solve(d, P, ridge=0.0, g0=None):
    """Least squares over the ICs of d for g = P x (P (n, k)), per lead and channel; the ridge pulls x to g0's."""
    Sm, cm_ = d["S"].mean(0), d["c"].mean(0)                               # (L, 66, n, n), (L, 66, n)
    Aq = np.einsum("ik,lcij,jm->lckm", P, Sm, P)
    bq = np.einsum("ik,lci->lck", P, cm_)
    if ridge:
        lam = ridge * np.trace(Aq, axis1=-2, axis2=-1)[..., None, None] / P.shape[1] * np.eye(P.shape[1])
        x0 = np.linalg.lstsq(P, g0, rcond=None)[0]
        Aq, bq = Aq + lam, bq + np.einsum("lckm,m->lck", lam, x0)
    return np.einsum("ik,lck->lci", P, np.linalg.solve(Aq, bq[..., None])[..., 0])


def split(d, idx):
    return dict(S=d["S"][idx], c=d["c"][idx], oo=d["oo"][idx])


Fd, Td = split(F, slice(None)), split(T, slice(None))
Lh = Td["oo"].shape[1]
one = lambda v: np.broadcast_to(v, (Lh, len(ch), n)).copy()  # noqa: E731
e_core = [np.repeat(np.eye(nc)[k], nb) for k in range(nc)]
e_mean = np.full(n, 1.0 / nc)
arms = {f"raw {c}": (one(e_core[k]), None) for k, c in enumerate(cores)}
arms["raw mean of the cores"] = (one(e_mean), None)
P_band = np.tile(np.eye(nb), (nc, 1)) / nc                                # the average of the cores, one gain per band
designs = {"gain": e_mean[:, None], "bands": P_band, "free": np.eye(n)}
for k, c in enumerate(cores):
    designs[f"{c} gain"] = e_core[k][:, None]
    designs[f"{c} bands"] = np.kron(np.eye(nc)[k][:, None], np.eye(nb))
fits = {}
for name, P in designs.items():
    kw = dict(ridge=A.ridge, g0=e_mean) if name == "free" else {}
    g = solve(Fd, P, **kw)
    fits[name] = g
    # 2-fold check inside the test year (first half / second half), to see that the other year's gains are not luck
    nT = Td["oo"].shape[0]
    h = [np.arange(nT // 2), np.arange(nT // 2, nT)]
    cv = np.concatenate([mse(split(T, h[1 - q]), solve(split(T, h[q]), P, **kw)) for q in (1, 0)])
    arms[name] = (g, cv)
rng = np.random.default_rng(0)
nT = Td["oo"].shape[0]
draws = rng.integers(0, nT, size=(2000, nT))
ref = mse(Td, arms["raw mean of the cores"][0])
out = dict(meta=dict(cores=cores, fit=A.fit, test=A.test, fit_ics=int(F["oo"].shape[0]), test_ics=int(nT), bands=lab, ridge=A.ridge), arms={})
rz = lambda m: np.sqrt(m[:, :, Z].mean(0)) * std[Z]  # noqa: E731
nr = lambda m: np.sqrt(m.mean(0))[:, SC].mean(-1)  # noqa: E731
for name, (g, cv) in arms.items():
    m = mse(Td, g)
    r = dict(z500=rz(m).round(1).tolist(), nrmse=nr(m).round(4).tolist())
    if cv is not None:
        r["z500_cv_in_test_year"] = rz(cv).round(1).tolist()
        r["nrmse_cv_in_test_year"] = nr(cv).round(4).tolist()
        bs = np.array([rz(m[i]) - rz(ref[i]) for i in draws])
        r["z500_minus_raw_mean"] = dict(diff=(rz(m) - rz(ref)).round(1).tolist(), ci=np.percentile(bs, [2.5, 97.5], 0).round(1).tolist())
    out["arms"][name] = r
    print(f"{name:32s} z500 {np.array(r['z500'])[Q]} nrmse {np.array(r['nrmse'])[Q]}" + (f"  cv z500 {np.array(r['z500_cv_in_test_year'])[Q]}  d10 {r['z500_minus_raw_mean']['diff'][9]} {np.array(r['z500_minus_raw_mean']['ci'])[:, 9].tolist()}" if cv is not None else ""))
# the fitted strengths of z500 (rows: lead 1, 3, 5, 7, 10)
out["z500_gain"] = (fits["gain"][:, Z, 0] * nc).round(3).tolist()
out["z500_band_gains"] = {lab[b]: (fits["bands"][:, Z, b] * nc).round(3).tolist() for b in range(nb)}
out["z500_free_weights"] = {c: {lab[b]: fits["free"][:, Z, k * nb + b].round(3).tolist() for b in range(nb)} for k, c in enumerate(cores)}
# what the truth and the mean carry per band (z500, test year): variance shares, ACC and activity of the raw mean
Sm, cm_, oo = Td["S"].mean(0)[:, Z], Td["c"].mean(0)[:, Z], Td["oo"].mean(0)[:, Z]
Pm = P_band
fb = np.einsum("ib,lij,jb->lb", Pm, Sm, Pm)                               # <f_b^2> of the mean (band diagonal)
cb = np.einsum("ib,li->lb", Pm, cm_)                                      # <f_b o>
out["z500_mean_band"] = dict(forecast_var_share=(fb / fb.sum(1, keepdims=True)).round(3)[Q].tolist(),
                             gain_now=(cb / np.maximum(fb, 1e-30)).round(3)[Q].tolist())
print("z500 gain (lead 1,3,5,7,10)", np.array(out["z500_gain"])[Q])
for b in range(nb):
    print(f"  band {lab[b]:6s} gain {np.array(out['z500_band_gains'][lab[b]])[Q]}  share of the mean's variance {np.array(out['z500_mean_band']['forecast_var_share'])[:, b]}")
if A.json:
    json.dump(out, open(A.json, "w"), indent=1)
