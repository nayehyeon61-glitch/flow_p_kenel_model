"""Which kernels to mix (감자 2026-11-24 21:30 KST: "kernel 은 어떠한 kernel 을 섞어야 할지 알아봐 주시고 설계하기").

The CRPS of a mixture  K = sum_i w_i K_i  of Markov kernels is a quadratic form of the weights,

    CRPS(w) = sum_i w_i a_i - 1/2 sum_ij w_i w_j d_ij ,   a_i = E|X_i - y| ,   d_ij = E|X_i - X_j|  (X_i ~ K_i independent)

and the mse of its mean is  w^T C w  with  C_ij = <(m_i - y)(m_j - y)>.  So a kernel is worth mixing in when it is far
from the others (d_ij large) for what it costs in accuracy (a_i): this script measures a, d, C and the moments for a set
of candidate kernels on the same ICs, mix_design.py then finds the weights (fitted on one year, scored on another).

Candidates:
    core:arm            a learned deviation kernel (devk*, e.g. core_gngmz8W:devk_band_sphW_p), its 2 P antithetic members,
                        or a deviation flow (devflow*): --members independent members
    clim                the invariant measure itself, K(x, .) = mu: 10-day pieces of the record of the same season
    kaf                 the analog kernel: the futures of the nearest analogs (math_core.py --kind kaf, <year>_kaf_nbr.npz)
    dress:TAG[+TAG]     a mean of diag_cond.py / math_core.py / slow_core.py fields (several: their average) read with one
                        strength g per lead and channel, dressed with its own errors on the --pool years (whole error
                        trajectories of other dates): the empirical-error kernel of that mean
    gdress:TAG[+TAG]    the same mean with Gaussian noise of the errors' standard deviation at every point (lead, channel,
                        grid point): what of dress: is the width map alone, and what the shape of the errors
    sdress:TAG[+TAG]    dress: with the errors of the same season only (+- --season days; sdress60: = +- 60 days)
    gsdress:TAG[+TAG]   the season's errors reduced to their width map (Gaussian), as gdress is to dress
    bgsdress:TAG[+TAG]  gsdress: about the mean + the season's mean error, with the errors' standard deviation about it: the
                        season's first and second moments at every point, without the shape of the errors
    asdress:TAG[+TAG]   sdress: in antithetic pairs, mean +- (e - the season's mean error): the member mean is the kernel's
                        mean exactly (as for a learned kernel's pairs), the odd moments of the errors are given up
    core:flow@p, @pk, @pklat   with --flow-proj: a deviation flow read on the conditioned mean of its core's *_p kernel
                        (the flows were trained on the kernels before the condition; this is the read-out of the condition,
                        as devk --proj is for a kernel). They share the flow's noise: never mix a flow with its own variant.
    core:arm@c{glob,lat,pt}   with --cal-maps: the learned kernel with its deviations from the mean rescaled so that its
                        variance equals the mean's squared error on another year (the second-moment condition
                        E[(y - m)^2] = E[V]), as one factor per lead and channel (cglob), per latitude too (clat) or per
                        grid point (cpt). A run with --save-maps writes these maps (the kernel's variance and the mean's
                        squared error at every point, averaged over its ICs). The variants share their kernel's noise:
                        never mix a kernel with its own variant (their cross distance is that of dependent members).

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
ap.add_argument("--season", type=int, default=45, help="sdress: +- days of the season")
ap.add_argument("--dir", default="runs/fpk/cond")
ap.add_argument("--npz", required=True)
ap.add_argument("--max-ics", type=int, default=0)
ap.add_argument("--save-maps", default="", help="write the learned kernels' variance and error maps here (npz)")
ap.add_argument("--cal-maps", default="", help="maps of another year: add the rescaled variants --cal of the learned kernels found there")
ap.add_argument("--cal", nargs="*", default=["glob", "lat", "pt"], choices=["glob", "lat", "pt"])
ap.add_argument("--cache", default="", help="folder of the learned kernels' and flows' members: rolled out and written at the first run, read afterwards "
                "(<year>_<core>__<arm>_st<stride>_n<ICs>.npy, f16 anomalies at the five leads)")
ap.add_argument("--flow-proj", action="store_true", help="for a deviation flow whose core also has a conditioned kernel (*_p) here: its members re-centred "
                "on that kernel's mean (name@p); with --kappa-maps also their deviations scaled so that the second moment equals the mean's "
                "squared error on another year, one factor per lead and channel (name@pk) or per latitude too (name@pklat)")
ap.add_argument("--kappa-maps", default="", help="maps of another year (--save-maps of a run with the flows and --flow-proj)")
ap.add_argument("--recenter", nargs="*", default=[], help="tags of means in --dir (e.g. the conditioned mean cond_fit.py --save-pred wrote): every "
                "learned kernel also with its members moved onto that mean (name@<tag>): the condition read out on the kernel. They share the "
                "kernel's noise: never mix a kernel with its own variant.")
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
    if arm.startswith("devflow"):                      # a deviation flow: --members independent members, its mean is not known
        m, pairs, kw = R.net_run(a, R.load_devflow(a, core, arm, pairs=a.members)), 0, dict(steps=a.steps)
    elif arm.startswith("devk"):
        m = R.net_run(a, R.load_devk(a, core, arm))
        pairs, kw = m.cfg.pairs, {}
    else:
        raise SystemExit(f"{arm}: a deviation kernel (devk*) or a deviation flow (devflow*)")
    sides.append(dict(a=a, core=core, model=m, pairs=pairs, kw=kw, name=f"{a.core_tag}:{arm}", gen=R.make_gen(a.seed, torch.device(a.device))))


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
rng2 = np.random.default_rng(A.seed + 200)             # the kernels added later draw apart, so the first ones keep their members
tg = torch.Generator().manual_seed(A.seed + 300)
rng3 = np.random.default_rng(A.seed + 400)
tg2 = torch.Generator().manual_seed(A.seed + 500)
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
DRESS, POOL_O, POOL_ROWS, POOL_DOY = {}, None, {}, None
Zc = [contract.CHANNEL_NAMES[c] for c in DYN].index("z500")
is_dress = lambda s: s.split(":")[0].rstrip("0123456789") in ("dress", "gdress", "sdress", "gsdress", "asdress", "bgsdress")  # noqa: E731
if any(is_dress(s) for s in A.synth):
    doy = []
    for y in A.pool:
        pm = json.load(open(f"{A.dir}/{y}_meta.json"))["ics"]
        POOL_ROWS[y] = np.arange(0, len(pm), A.pool_stride)
        doy.append(np.asarray(pm)[POOL_ROWS[y]])
    POOL_DOY = np.concatenate(doy)                                                     # the day of the year of every pool IC
    POOL_O = np.concatenate([np.load(f"{A.dir}/{y}_o.npy", mmap_mode="r")[POOL_ROWS[y]][:, Q] for y in A.pool])   # (n, 5, 66, H, W) f16
for spec in A.synth:
    if not is_dress(spec):
        continue
    if spec.split(":", 1)[1] in DRESS:
        continue
    tags = spec.split(":", 1)[1].split("+")
    pf = sum(np.concatenate([np.asarray(np.load(f"{A.dir}/{y}_{t}.npy", mmap_mode="r")[POOL_ROWS[y]][:, Q], dtype=np.float32) for y in A.pool])
             for t in tags) / len(tags)
    sf, sff = 0, 0
    for i in range(0, len(pf), 32):
        f, oo = torch.from_numpy(pf[i:i + 32]), f32(POOL_O[i:i + 32])
        sf, sff = sf + am(f * oo).sum(0).double(), sff + am(f * f).sum(0).double()
    g = (sf / sff.clamp_min(1e-30)).float().clamp(0.0, 1.5)                                 # (5, 66) the strength
    d = dict(g=g, f=pf.astype(np.float16), cur=[np.load(f"{A.dir}/{A.year}_{t}.npy", mmap_mode="r") for t in tags])
    s1, s2 = 0, 0                                                                          # the errors' mean and width at every point
    for i in range(0, len(pf), 32):
        r = (f32(POOL_O[i:i + 32]) - g[:, :, None, None] * torch.from_numpy(pf[i:i + 32])).double()
        s1, s2 = s1 + r.sum(0), s2 + (r ** 2).sum(0)
    d["bias"] = (s1 / len(pf)).float()                # (5, 66, H, W): the pool's mean error; the kernel's mean is g f + bias
    d["sd"] = (s2 / len(pf)).sqrt().float()           # gdress: N(g f, sd^2), the second moment of the errors about g f
    DRESS[spec.split(":", 1)[1]] = d
    del pf
    print(json.dumps(dict(dress=spec, pool=int(len(POOL_O)), g_z500=[round(float(v), 3) for v in g[:, Zc]])), flush=True)


def synth(spec, k):
    """Members (M, 5, 66, H, W), the kernel's mean if it is known (else None), kind, and the number of atoms of the pool
    the members are drawn from (0: not a pool; mix_design.py scores an empirical kernel as that finite pool)."""
    if spec == "clim":
        dd = np.abs(a_doy - k)
        cand = a_rows[np.minimum(dd, 365 - dd) <= A.window]
        r = np.sort(rng.choice(cand, M, replace=False))
        return torch.stack([f32(ARCH[r + l]) for l in LD], 1), torch.zeros(len(Q), C, H, W), "indep", len(cand)
    if spec == "kaf":
        r = np.sort(NBR[ROW[k]][:M])
        return torch.stack([f32(ARCH[r + l]) for l in LD], 1), None, "atoms", 0
    kind, tag = spec.split(":", 1)
    d = DRESS[tag]
    g = d["g"][:, :, None, None]
    mean = g * sum(f32(x[ROW[k]][Q]) for x in d["cur"]) / len(d["cur"])
    if kind == "gdress":
        return mean[None] + d["sd"][None] * torch.randn((M,) + tuple(d["sd"].shape), generator=tg), mean, "indep", 0
    if kind.startswith(("sdress", "gsdress", "asdress", "bgsdress")):                          # sdress60: +- 60 days
        win = int(kind.lstrip("abgsdres") or A.season)
        dd = np.abs(POOL_DOY - k)
        cand = np.nonzero(np.minimum(dd, 365 - dd) <= win)[0]
        s1, s2 = torch.zeros(len(Q), C, H, W, dtype=torch.float64), torch.zeros(len(Q), C, H, W, dtype=torch.float64)
        for i in range(0, len(cand), 32):                                                     # the season's mean error and width map
            r = (f32(POOL_O[cand[i:i + 32]]) - g * f32(d["f"][cand[i:i + 32]])).double()
            s1, s2 = s1 + r.sum(0), s2 + (r ** 2).sum(0)
        if kind[0] == "g":                                                                    # Gaussian with that width
            sd = (s2 / len(cand)).sqrt().float()
            return mean[None] + sd[None] * torch.randn((M,) + tuple(sd.shape), generator=tg), mean, "indep", 0
        bias = (s1 / len(cand)).float()
        if kind[0] == "b":                                                                    # Gaussian about mean + the season's mean error
            sd = (s2 / len(cand) - (s1 / len(cand)) ** 2).clamp_min(0).sqrt().float()
            return mean[None] + bias[None] + sd[None] * torch.randn((M,) + tuple(sd.shape), generator=tg2), mean + bias, "indep", 0
        if kind[0] == "a":                                                                    # antithetic pairs: mean +- (e - its mean)
            p = np.sort(rng3.choice(cand, M // 2, replace=False))
            res = f32(POOL_O[p]) - g * f32(d["f"][p]) - bias[None]
            return torch.cat([mean[None] + bias[None] + res, mean[None] + bias[None] - res]), mean + bias, "apool", -len(cand)
        p = np.sort(rng2.choice(cand, M, replace=False))
    else:
        cand, bias = np.arange(len(d["f"])), d["bias"]
        p = np.sort(rng.choice(len(d["f"]), M, replace=False))
    res = f32(POOL_O[p]) - g * f32(d["f"][p])
    return mean[None] + res, mean + bias, "indep", len(cand)                                   # the kernel's own mean: with the pool's mean error


CAL = {}
if A.cal_maps:
    cz = np.load(A.cal_maps)
    for s in sides:
        if "V::" + s["name"] not in cz.files:
            continue
        Vm, Em = torch.from_numpy(cz["V::" + s["name"]]).double(), torch.from_numpy(cz["E::" + s["name"]]).double()   # (5, 66, H, W)
        wl = w.double()
        c = dict(glob=((Em * wl).mean((-2, -1), keepdim=True) / (Vm * wl).mean((-2, -1), keepdim=True).clamp_min(1e-30)).sqrt(),
                 lat=(Em.mean(-1, keepdim=True) / Vm.mean(-1, keepdim=True).clamp_min(1e-30)).sqrt(), pt=(Em / Vm.clamp_min(1e-30)).sqrt())
        CAL[s["name"]] = [(k, c[k].clamp(0.3, 3.0).float()) for k in A.cal]
        print(json.dumps(dict(cal=s["name"], maps_ics=int(cz["n"]), z500_glob=[round(float(v), 3) for v in c["glob"][:, Zc, 0, 0]],
                              z500_lat_min_max=[round(float(c["lat"][-1, Zc].min()), 2), round(float(c["lat"][-1, Zc].max()), 2)],
                              pt_q05_q95_day10=[round(float(v), 2) for v in np.quantile(c["pt"][-1].numpy(), [0.05, 0.95])])), flush=True)
FLOWP, KAP, DUMP = {}, {}, {}
if A.flow_proj:
    for n, s in enumerate(sides):
        if not s["pairs"]:
            q = [m for m, t in enumerate(sides) if t["pairs"] and t["a"].core_tag == s["a"].core_tag and t["name"].endswith("_p")]
            if q:
                FLOWP[n] = q[0]
    if A.kappa_maps:
        kz = np.load(A.kappa_maps)
        for n, q in FLOWP.items():
            fn, kn = sides[n]["name"], sides[q]["name"]
            if "VF::" + fn in kz.files and "E::" + kn in kz.files:
                VF, Em = torch.from_numpy(kz["VF::" + fn]).double(), torch.from_numpy(kz["E::" + kn]).double()
                wl = w.double()
                KAP[n] = dict(pk=((Em * wl).mean((-2, -1), keepdim=True) / (VF * wl).mean((-2, -1), keepdim=True).clamp_min(1e-30)).sqrt().clamp(0.3, 3.0).float(),
                              pklat=(Em.mean(-1, keepdim=True) / VF.mean(-1, keepdim=True).clamp_min(1e-30)).sqrt().clamp(0.3, 3.0).float())
                print(json.dumps(dict(kappa=fn, on=kn, maps_ics=int(kz["n"]), z500=[round(float(v), 3) for v in KAP[n]["pk"][:, Zc, 0, 0]],
                                      mean_day10=round(float(KAP[n]["pk"][-1].mean()), 3))), flush=True)


def core_mean(tag, k):
    """The core's own forecast (anomaly, the five leads) from the dump of diag_cond.py."""
    if tag not in DUMP:
        DUMP[tag] = np.load(f"{A.dir}/{A.year}_{tag}.npy", mmap_mode="r")
    return f32(DUMP[tag][ROW[k]][Q])


names = []
for n, s in enumerate(sides):
    names += [s["name"]] + [f"{s['name']}@c{k}" for k, _ in CAL.get(s["name"], [])]
    if s["pairs"]:
        names += [f"{s['name']}@{tg_}" for tg_ in A.recenter]
    if n in FLOWP:
        names += [f"{s['name']}@p"] + [f"{s['name']}@{k}" for k in KAP.get(n, {})]
names += list(A.synth)
K = len(names)
MAPS = {s["name"]: [0, 0, 0] for s in sides} if A.save_maps else {}
CACHE = {}
if A.cache:
    os.makedirs(A.cache, exist_ok=True)
    for n, s in enumerate(sides):
        f = f"{A.cache}/{A.year}_{s['name'].replace(':', '__')}_st{A.ic_stride}_n{len(ics)}.npy"
        if os.path.exists(f + ".done"):
            CACHE[n] = ("r", np.load(f, mmap_mode="r"), f)
        else:
            CACHE[n] = ("w", np.lib.format.open_memmap(f, mode="w+", dtype=np.float16, shape=(len(ics), 2 * s["pairs"] or M, len(Q), C, H, W)), f)
    print(json.dumps(dict(cache={s["name"]: CACHE[n][0] for n, s in enumerate(sides)})), flush=True)
ST = {k: [] for k in ("a", "D", "C", "Mm", "V", "oo", "ea", "eD")}
nrm = lambda z: am(z * z).sqrt()  # noqa: E731   (..., 5, 66): the L2 norm of a field over the sphere (the energy score's distance)
ATOMS = []
t0 = time.time()
for i in range(0, len(ics), 2):
    j = ics[i:i + 2]
    truth = np.stack([te[k + 1:k + 1 + L][:, dyn] for k in j]).astype(np.float32)
    cj = np.stack([CZ[np.minimum(np.arange(k + 1, k + 1 + L), 365)] for k in j])
    tis = torch.from_numpy(np.stack([te[k + 1:k + 1 + L, TISR] for k in j])).to(dev)
    learned = []
    for n, s in enumerate(sides):
        if n in CACHE and CACHE[n][0] == "r":
            learned.append(f32(CACHE[n][1][i:i + len(j)]))
            continue
        Wm = int(getattr(s["core"], "mem", 0))
        hist = torch.from_numpy(np.stack([te[np.maximum(np.arange(k - Wm, k), 0)][:, dyn] for k in j])).to(dev) if Wm else None
        with torch.no_grad():
            e = s["model"].rollout(torch.from_numpy(te[j][:, dyn]).to(dev), torch.tensor(j, device=dev), tis, L, hist=hist, gen=s["gen"], **s["kw"])[0]
        x = e.float().cpu()[:, :, Q] - torch.from_numpy(cj[:, Q])[:, None]                        # (B, M, 5, 66, H, W) anomalies
        if n in CACHE:                                                                             # the same f16 values now and when read back
            CACHE[n][1][i:i + len(j)] = x.numpy()
            x = f32(CACHE[n][1][i:i + len(j)])
        learned.append(x)
    for b, k in enumerate(j):
        y = torch.from_numpy(truth[b][Q] - cj[b][Q])
        X, MU, KIND = [], [], []
        for n, s in enumerate(sides):
            kind = "pairs" if s["pairs"] else "sample"
            x = learned[n][b]
            mu = x.mean(0)
            X.append(x); MU.append(mu); KIND.append(kind)
            for _, c in CAL.get(s["name"], []):                                                  # the same noise, another width
                X.append(mu[None] + c[None] * (x - mu[None])); MU.append(mu); KIND.append(kind)
            if s["pairs"]:
                for tg_ in A.recenter:                                                           # the same deviations about another mean
                    mc = core_mean(tg_, int(k))
                    X.append(mc[None] + (x - mu[None])); MU.append(mc); KIND.append(kind)
            if n in FLOWP:                                                                       # the flow read on the conditioned mean
                fdev = x - core_mean(s["a"].core_tag, int(k))[None]
                mp = learned[FLOWP[n]][b].mean(0)
                for kk in [None] + list(KAP.get(n, {})):
                    xv = mp[None] + (fdev if kk is None else KAP[n][kk][None] * fdev)
                    X.append(xv); MU.append(xv.mean(0)); KIND.append("sample")
                if MAPS:
                    MAPS[s["name"]][2] = MAPS[s["name"]][2] + (fdev ** 2).mean(0).double()
            if MAPS:
                MAPS[s["name"]][0] = MAPS[s["name"]][0] + ((x - mu) ** 2).sum(0).double() / (x.shape[0] - (kind == "sample"))
                MAPS[s["name"]][1] = MAPS[s["name"]][1] + ((mu - y) ** 2).double() - (((x - mu) ** 2).sum(0).double() / (x.shape[0] - 1) / x.shape[0] if kind == "sample" else 0)
        atoms, APOOL = [0] * len(X), {}
        for spec in A.synth:
            x, mu, kind, na = synth(spec, int(k))
            if na < 0:                                   # "apool": - (the dates of its pool); its width is made exact below
                APOOL[len(X)], na = -na, 0
            X.append(x); MU.append(x.mean(0) if mu is None else mu); KIND.append(kind); atoms.append(na)
        ATOMS.append(atoms)
        a_ = torch.stack([am((x - y).abs().mean(0)) for x in X])                                 # (K, 5, 66)
        ea_ = torch.stack([nrm(x - y).mean(0) for x in X])                                       # E||X - y|| per lead and channel
        # the variance inside a kernel; "sample" = independent members and an unknown mean: the unbiased variance, and
        # the noise of the member mean (V / M) leaves the diagonals of C and Mm (E[(m^ - y)^2] = (m - y)^2 + V / M)
        V = torch.stack([am(((x - mu) ** 2).sum(0)) / (x.shape[0] - (kind == "sample")) for x, mu, kind in zip(X, MU, KIND)])
        E = [mu - y for mu in MU]
        Cc = torch.stack([torch.stack([am(E[p] * E[q]) for q in range(K)]) for p in range(K)])
        Mm = torch.stack([torch.stack([am(MU[p] * MU[q]) for q in range(K)]) for p in range(K)])
        for p in range(K):
            if KIND[p] == "sample":
                Cc[p, p] -= V[p] / X[p].shape[0]
                Mm[p, p] -= V[p] / X[p].shape[0]
        D, eD = torch.zeros(K, K, len(Q), C), torch.zeros(K, K, len(Q), C)
        for p in range(K):
            for q in range(p, K):
                tot, tot2 = 0, 0                                                                  # sums over all pairs: |.| and ||.||
                for x in X[p]:
                    df = x[None] - X[q]
                    tot, tot2 = tot + df.abs().sum(0), tot2 + nrm(df).sum(0)
                n_p, n_q = X[p].shape[0], X[q].shape[0]
                if p != q:
                    D[p, q] = D[q, p] = am(tot / (n_p * n_q))
                    eD[p, q] = eD[q, p] = tot2 / (n_p * n_q)
                elif KIND[p] == "atoms":                                                          # the kernel IS these atoms
                    D[p, p], eD[p, p] = am(tot / (n_p * n_p)), tot2 / (n_p * n_p)
                elif KIND[p] in ("pairs", "apool"):                                               # not the antithetic partners
                    P = n_p // 2
                    pd = X[p][:P] - X[p][P:]
                    D[p, p] = am((tot - 2 * pd.abs().sum(0)) / (n_p * (n_p - 2)))
                    eD[p, p] = (tot2 - 2 * nrm(pd).sum(0)) / (n_p * (n_p - 2))
                    if KIND[p] == "apool":
                        # the uniform law on n dates x 2 signs, exactly: two draws are the same date with probability 1 / n,
                        # and then the same atom (distance 0) or its partner, half and half
                        n = APOOL[p]
                        D[p, p] = (1 - 1 / n) * D[p, p] + am(pd.abs().mean(0)) / (2 * n)
                        eD[p, p] = (1 - 1 / n) * eD[p, p] + nrm(pd).mean(0) / (2 * n)
                else:
                    D[p, p], eD[p, p] = am(tot / (n_p * (n_p - 1))), tot2 / (n_p * (n_p - 1))
        for key, v in (("a", a_), ("D", D), ("C", Cc), ("Mm", Mm), ("V", V), ("oo", am(y * y)), ("ea", ea_), ("eD", eD)):
            ST[key].append(v.double().numpy())
    print(json.dumps(dict(year=A.year, ics=int(i + len(j)), of=len(ics), sec=round(time.time() - t0),
                          rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)), flush=True)
np.savez_compressed(A.npz, **{k: np.stack(v) for k, v in ST.items()}, names=np.array(names), kinds=np.array(KIND), ics=ics,
                    leads=np.array(LD), std=cl.std[dyn], channels=np.array([contract.CHANNEL_NAMES[c] for c in DYN]),
                    year=A.year, pool=np.array(A.pool), members=M, atoms=np.array(ATOMS))
if MAPS:
    np.savez_compressed(A.save_maps, n=len(ics), year=A.year, **{f"V::{k}": (v[0] / len(ics)).float().numpy() for k, v in MAPS.items()},
                        **{f"E::{k}": (v[1] / len(ics)).float().numpy() for k, v in MAPS.items()},
                        **{f"VF::{k}": (v[2] / len(ics)).float().numpy() for k, v in MAPS.items() if torch.is_tensor(v[2])})
for n, c in CACHE.items():
    if c[0] == "w":
        c[1].flush()
        open(c[2] + ".done", "w").write(str(len(ics)))
print("done", K, names, round(time.time() - t0))
