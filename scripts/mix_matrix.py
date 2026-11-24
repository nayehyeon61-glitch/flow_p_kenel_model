"""Which kernels to mix (감자 2026-11-24 21:30 KST: "kernel 은 어떠한 kernel 을 섞어야 할지 알아봐 주시고 설계하기").

The CRPS of a mixture  K = sum_i w_i K_i  of Markov kernels is a quadratic form of the weights,

    CRPS(w) = sum_i w_i a_i - 1/2 sum_ij w_i w_j d_ij ,   a_i = E|X_i - y| ,   d_ij = E|X_i - X_j|  (X_i ~ K_i independent)

and the mse of its mean is  w^T C w  with  C_ij = <(m_i - y)(m_j - y)>.  So a kernel is worth mixing in when it is far
from the others (d_ij large) for what it costs in accuracy (a_i): this script measures a, d, C and the moments for a set
of candidate kernels on the same ICs, mix_design.py then finds the weights (fitted on one year, scored on another).

Candidates:
    core:arm            a learned deviation kernel (devk*, e.g. core_gngmz8W:devk_band_sphW_p), its 2 P antithetic members
    clim                the invariant measure itself, K(x, .) = mu: 10-day pieces of the record of the same season
    kaf                 the analog kernel: the futures of the nearest analogs (math_core.py --kind kaf, <year>_kaf_nbr.npz)
    dress:TAG[+TAG]     a mean of diag_cond.py / math_core.py / slow_core.py fields (several: their average) read with one
                        strength g per lead and channel, dressed with its own errors on the --pool years (whole error
                        trajectories of other dates): the empirical-error kernel of that mean

    python scripts/mix_matrix.py --year 2017 --pool 2016 2020 --learned core_gngmz8W:devk_band_sphW_p \\
        core_ngmz8W:devk_band_sphNW_p --synth clim kaf dress:lim dress:slow --npz runs/fpk/cond/mixmat_2017.npz
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
ap.add_argument("--year", type=int, default=2017)
ap.add_argument("--ic-stride", type=int, default=6)
ap.add_argument("--learned", nargs="*", default=[])
ap.add_argument("--synth", nargs="*", default=[])
ap.add_argument("--pool", type=int, nargs="+", default=[2016, 2020])
ap.add_argument("--pool-stride", type=int, default=2)
ap.add_argument("--members", type=int, default=8)
ap.add_argument("--window", type=int, default=15, help="clim: +- days of the season")
ap.add_argument("--dir", default="runs/fpk/cond")
ap.add_argument("--npz", required=True)
ap.add_argument("--max-ics", type=int, default=0)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
A, PASS = ap.parse_known_args()
assert A.year not in A.pool, "the dressing errors must come from other years"
torch.set_num_threads(A.threads)

sys.path.insert(0, os.path.dirname(__file__))
import run_fpk as R  # noqa: E402
from fpk.env import DYN, TISR, contract  # noqa: E402

Q = [0, 2, 4, 6, 9]
LD = [q + 1 for q in Q]
sides, base = [], {}


def grab(a, rest):
    pa, cl, core = R.load_core(a)
    arm = a.arms[0]
    if arm == "core":
        base.update(a=a, pa=pa, cl=cl)
        return
    if not arm.startswith("devk"):
        raise SystemExit(f"{arm}: a deviation kernel (devk*)")
    m = R.net_run(a, R.load_devk(a, core, arm))
    sides.append(dict(a=a, core=core, model=m, pairs=m.cfg.pairs, name=f"{a.core_tag}:{arm}", gen=R.make_gen(a.seed, torch.device(a.device))))


R.cmd_score = grab
common = ["--members", str(A.members), "--threads", str(A.threads), "--test-year", str(A.year), "--ic-stride", str(A.ic_stride)] + PASS
R.main(["score", "--core-tag", "core_gngmz8W", "--arms", "core"] + common)
for n, spec in enumerate(A.learned):
    ct, arm = spec.split(":")
    R.main(["score", "--core-tag", ct, "--arms", arm, "--seed", str(A.seed + n)] + common)    # every side its own noise
a0, pa, cl = base["a"], base["pa"], base["cl"]
dev = torch.device(a0.device)
te = R.D.load(pa, [A.year])[A.year]
L = a0.leads
ics = np.arange(a0.ic_start, len(te) - L, A.ic_stride)[: A.max_ics or None]
dyn = np.asarray(DYN)
eps = R.rkt().EPS
cm, cs = cl.mean[dyn][:, None, None], cl.std[dyn][:, None, None] + eps
CZ = ((cl.clim[:, dyn] - cm) / cs).astype(np.float32)
C, (H, W) = len(dyn), CZ.shape[-2:]
w = np.cos(np.deg2rad(cl.lat))
w = torch.tensor(w / w.mean(), dtype=torch.float32)[:, None]
am = lambda x: (x * w).mean((-2, -1))  # noqa: E731
rng = np.random.default_rng(A.seed + 100)
M = A.members
meta = json.load(open(f"{A.dir}/{A.year}_meta.json"))
ROW = {k: i for i, k in enumerate(meta["ics"])}
f32 = lambda x: torch.from_numpy(np.asarray(x, dtype=np.float32))  # noqa: E731

# ------------------------------------------------------------------------------------------- the synthetic kernels
ARCH = None
if any(s in ("clim", "kaf") for s in A.synth):
    ap_, am_ = f"{A.dir}/archive_anom.npy", f"{A.dir}/archive_meta.json"
    if not os.path.exists(am_):
        yrs = R.D.load(pa, list(a0.train_years))
        n = [len(yrs[y]) for y in a0.train_years]
        arch = np.lib.format.open_memmap(ap_, mode="w+", dtype=np.float16, shape=(sum(n), C, H, W))
        s = 0
        for y, k in zip(a0.train_years, n):
            arch[s:s + k] = np.asarray(yrs[y][:, dyn], dtype=np.float32) - CZ[np.minimum(np.arange(k), 365)]
            s += k
        arch.flush()
        json.dump(dict(years=list(a0.train_years), days=n), open(am_, "w"))
        del yrs, arch
    ameta = json.load(open(am_))
    ARCH = np.load(ap_, mmap_mode="r")
    starts = np.cumsum([0] + ameta["days"])
    a_rows = np.concatenate([np.arange(s, e - L) for s, e in zip(starts[:-1], starts[1:])])   # 10 days ahead stay in the year
    a_doy = np.concatenate([np.arange(0, e - s - L) for s, e in zip(starts[:-1], starts[1:])])
NBR = None
if "kaf" in A.synth:
    nz = np.load(f"{A.dir}/{A.year}_kaf_nbr.npz")
    if nz["years"].tolist() != ameta["years"]:
        raise SystemExit(f"the analogs index the record of {nz['years'][0]}-{nz['years'][-1]}, the archive here is "
                         f"{ameta['years'][0]}-{ameta['years'][-1]}: give the same --train-years (and remove {A.dir}/archive_*)")
    NBR = nz["rows"]
DRESS, POOL_O, POOL_ROWS = {}, None, {}
Zc = [contract.CHANNEL_NAMES[c] for c in DYN].index("z500")
if any(s.startswith("dress:") for s in A.synth):
    for y in A.pool:
        POOL_ROWS[y] = np.arange(0, len(json.load(open(f"{A.dir}/{y}_meta.json"))["ics"]), A.pool_stride)
    POOL_O = np.concatenate([np.load(f"{A.dir}/{y}_o.npy", mmap_mode="r")[POOL_ROWS[y]][:, Q] for y in A.pool])   # (n, 5, 66, H, W) f16
for spec in A.synth:
    if not spec.startswith("dress:"):
        continue
    tags = spec[6:].split("+")
    pf = sum(np.concatenate([np.asarray(np.load(f"{A.dir}/{y}_{t}.npy", mmap_mode="r")[POOL_ROWS[y]][:, Q], dtype=np.float32) for y in A.pool])
             for t in tags) / len(tags)
    sf, sff = 0, 0
    for i in range(0, len(pf), 32):
        f, oo = torch.from_numpy(pf[i:i + 32]), f32(POOL_O[i:i + 32])
        sf, sff = sf + am(f * oo).sum(0).double(), sff + am(f * f).sum(0).double()
    g = (sf / sff.clamp_min(1e-30)).float().clamp(0.0, 1.5)                                 # (5, 66) the strength
    DRESS[spec] = dict(g=g, f=pf.astype(np.float16), cur=[np.load(f"{A.dir}/{A.year}_{t}.npy", mmap_mode="r") for t in tags])
    del pf
    print(json.dumps(dict(dress=spec, pool=int(len(POOL_O)), g_z500=[round(float(v), 3) for v in g[:, Zc]])), flush=True)


def synth(spec, k):
    """Members (M, 5, 66, H, W), the kernel's mean if it is known (else None), kind."""
    if spec == "clim":
        dd = np.abs(a_doy - k)
        cand = a_rows[np.minimum(dd, 365 - dd) <= A.window]
        r = np.sort(rng.choice(cand, M, replace=False))
        return torch.stack([f32(ARCH[r + l]) for l in LD], 1), torch.zeros(len(Q), C, H, W), "indep"
    if spec == "kaf":
        r = np.sort(NBR[ROW[k]][:M])
        return torch.stack([f32(ARCH[r + l]) for l in LD], 1), None, "atoms"
    d = DRESS[spec]
    g = d["g"][:, :, None, None]
    mean = g * sum(f32(x[ROW[k]][Q]) for x in d["cur"]) / len(d["cur"])
    p = np.sort(rng.choice(len(d["f"]), M, replace=False))
    res = f32(POOL_O[p]) - g * f32(d["f"][p])
    return mean[None] + res, mean, "indep"


names = [s["name"] for s in sides] + list(A.synth)
K = len(names)
ST = {k: [] for k in ("a", "D", "C", "Mm", "V", "oo")}
t0 = time.time()
for i in range(0, len(ics), 2):
    j = ics[i:i + 2]
    truth = np.stack([te[k + 1:k + 1 + L][:, dyn] for k in j]).astype(np.float32)
    cj = np.stack([CZ[np.minimum(np.arange(k + 1, k + 1 + L), 365)] for k in j])
    tis = torch.from_numpy(np.stack([te[k + 1:k + 1 + L, TISR] for k in j])).to(dev)
    learned = []
    for s in sides:
        Wm = int(getattr(s["core"], "mem", 0))
        hist = torch.from_numpy(np.stack([te[np.maximum(np.arange(k - Wm, k), 0)][:, dyn] for k in j])).to(dev) if Wm else None
        with torch.no_grad():
            e = s["model"].rollout(torch.from_numpy(te[j][:, dyn]).to(dev), torch.tensor(j, device=dev), tis, L, hist=hist, gen=s["gen"])[0]
        learned.append(e.float().cpu()[:, :, Q] - torch.from_numpy(cj[:, Q])[:, None])           # (B, M, 5, 66, H, W) anomalies
    for b, k in enumerate(j):
        y = torch.from_numpy(truth[b][Q] - cj[b][Q])
        X, MU, KIND = [], [], []
        for n, s in enumerate(sides):
            X.append(learned[n][b]); MU.append(learned[n][b].mean(0)); KIND.append("pairs")
        for spec in A.synth:
            x, mu, kind = synth(spec, int(k))
            X.append(x); MU.append(x.mean(0) if mu is None else mu); KIND.append(kind)
        a_ = torch.stack([am((x - y).abs().mean(0)) for x in X])                                 # (K, 5, 66)
        V = torch.stack([am(((x - mu) ** 2).mean(0)) for x, mu in zip(X, MU)])
        E = [mu - y for mu in MU]
        Cc = torch.stack([torch.stack([am(E[p] * E[q]) for q in range(K)]) for p in range(K)])
        Mm = torch.stack([torch.stack([am(MU[p] * MU[q]) for q in range(K)]) for p in range(K)])
        D = torch.zeros(K, K, len(Q), C)
        for p in range(K):
            for q in range(p, K):
                tot = sum((x[None] - X[q]).abs().sum(0) for x in X[p])                           # sum over all pairs
                n_p, n_q = X[p].shape[0], X[q].shape[0]
                if p != q:
                    D[p, q] = D[q, p] = am(tot / (n_p * n_q))
                elif KIND[p] == "atoms":                                                          # the kernel IS these atoms
                    D[p, p] = am(tot / (n_p * n_p))
                elif KIND[p] == "pairs":                                                          # not the antithetic partners
                    P = n_p // 2
                    D[p, p] = am((tot - 2 * (X[p][:P] - X[p][P:]).abs().sum(0)) / (n_p * (n_p - 2)))
                else:
                    D[p, p] = am(tot / (n_p * (n_p - 1)))
        for key, v in (("a", a_), ("D", D), ("C", Cc), ("Mm", Mm), ("V", V), ("oo", am(y * y))):
            ST[key].append(v.double().numpy())
    print(json.dumps(dict(year=A.year, ics=int(i + len(j)), of=len(ics), sec=round(time.time() - t0),
                          rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)), flush=True)
np.savez_compressed(A.npz, **{k: np.stack(v) for k, v in ST.items()}, names=np.array(names), kinds=np.array(KIND), ics=ics,
                    leads=np.array(LD), std=cl.std[dyn], channels=np.array([contract.CHANNEL_NAMES[c] for c in DYN]),
                    year=A.year, pool=np.array(A.pool), members=M)
print("done", K, names, round(time.time() - t0))
