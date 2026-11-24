"""The mixture of two forecast distributions as ONE distribution, scored as the arms of ``run_fpk.py score``
(감자 2026-11-24: "한가지만 더" -> the mixed kernel).

    K = 1/2 K_A + 1/2 K_B :  a convex combination of two Markov kernels (each on its own core) is a Markov kernel;
    its mean is the average of the two means, its variance 1/2 (v_A + v_B) + 1/4 (m_A - m_B)^2 (the spread between
    the cores = the uncertainty of the model itself, which neither kernel has).

    python scripts/score_mix.py mixk_W_NW core_gngmz8W:devk_band_sphW core_ngmz8W:devk_band_sphNW
    python scripts/score_mix.py mixg_W_NW core_gngmz8W:devflow_gW core_ngmz8W:devflow_gNW --members 4

Each side is rolled out as its own ``score`` run does (IC batches and members: a deviation kernel its 2 P antithetic
members, a flow ``--members`` independent ones); side A with the scorer's seed, so its numbers written next to the
mixture are those of ``score_<arm>.json`` (--members 8), side B with seed + 1 so that the sides are independent
(--seed-b 0 reproduces side B's stored score too, checked exactly on 4 ICs). The mixture's scores are those of the
distribution, not of the pooled sample:
    mean        1/2 (m_A + m_B)
    variance    1/2 (v_A + v_B) + 1/4 [(m_A - m_B)^2 - n_A - n_B], n = the sampling variance of a side's mean
                (0 for a kernel: its mean is known; v / M for M independent members)
    CRPS        1/2 (E|A - y| + E|B - y|) - 1/2 [1/4 S_A + 1/4 S_B + 1/2 C]: S = E|X - X'| inside a side (its fair
                estimate: all pairs of independent members, the non-partner pairs of antithetic ones), C = the mean
                of |a - b| over all cross pairs. = the mean of the two CRPS - 1/8 x the energy distance of the sides.
    IGN, H      of the Gaussian with that mean and variance; the error term's finite-sample factor of the scorer with
                the pooled count (P_A + P_B pairs, or M_A + M_B members): an approximation, the between-side term of
                the variance being exact. That factor and the bias of log(variance) depend on the count, so IGN is
                compared at the SAME budget as one arm: two kernels also give TAG_h (half the pairs of each side,
                same rollout), two flows are run with --members 4 (4 + 4 = the scorer's 8).
    mean_noise  (n_A + n_B) / 4: E[mse of the finite mean] - mse of the mixture's mean (0 for kernels)
Writes runs/fpk/score_<TAG>.json and score_<TAG>_raw.npz in the scorer's format (scripts/pair_nophys.py pairs it)."""
import argparse
import json
import os
import resource
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import run_fpk as R  # noqa: E402
from fpk.env import DYN, TISR, TISR_DYN, contract  # noqa: E402
from fpk.ensemble import LOG2PI, Scorer, area_mean  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("tag")
ap.add_argument("a")
ap.add_argument("b")
ap.add_argument("--members", type=int, default=8, help="independent members of a flow side (a kernel side: its 2 P)")
ap.add_argument("--max-ics", type=int, default=0)
ap.add_argument("--leads", type=int, default=10)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--proj", default="", help="a proj_<year>.npz of scripts/diag_projection.py stats: every (kernel) side's mean is read out as "
                "climatology + g (.) (mean - climatology), g per lead, channel and band fitted on that file's year for the "
                "side's core, and its deviations are scaled by the fitted error ratio (the invariant-measure condition)")
ap.add_argument("--seed-b", type=int, default=1, help="side B's seed = the scorer's + this (0 = the same draws: only to reproduce score_<arm>)")
A = ap.parse_args()
sides = []


def grab(a, rest):
    pa, cl, core = R.load_core(a)
    arm = a.arms[0]
    if arm.startswith("devflow"):
        m = R.net_run(a, R.load_devflow(a, core, arm, pairs=a.members))
        run = lambda x0, j, tis, hist, g: m.rollout(x0, j, tis, a.leads, hist=hist, gen=g, steps=a.steps)[0]  # noqa: E731
        pairs = 0
    elif arm.startswith("devk"):
        m = R.net_run(a, R.load_devk(a, core, arm))
        run = lambda x0, j, tis, hist, g: m.rollout(x0, j, tis, a.leads, hist=hist, gen=g)[0]  # noqa: E731
        pairs = m.cfg.pairs
    else:
        raise SystemExit(f"{arm}: a deviation kernel (devk*) or a deviation flow (devflow*)")
    sides.append(dict(a=a, pa=pa, cl=cl, core=core, model=m, run=run, pairs=pairs, arm=arm, gen=R.make_gen(a.seed, torch.device(a.device))))


R.cmd_score = grab
for spec in (A.a, A.b):
    ct, arm = spec.split(":")
    # side B on its own noise (seed + 1): with the scorer's seed on both sides the two kernels shape the SAME draws,
    # their members are correlated pair by pair, and the cross term C, the variance's count and the means' noise
    # are all wrong (measured: CRPS of the half mixture 0.0014 above the full one, 2026-11-24)
    R.main(["score", "--core-tag", ct, "--arms", arm, "--members", str(A.members), "--leads", str(A.leads),
            "--max-ics", str(A.max_ics), "--threads", str(A.threads)]
           + (["--seed", str(sides[0]["a"].seed + A.seed_b)] if sides else []))
sa, sb = sides
if bool(sa["pairs"]) != bool(sb["pairs"]):
    raise SystemExit("two kernels or two flows")
a0, cl, pa = sa["a"], sa["cl"], sa["pa"]
te = R.D.load(pa, [a0.test_year])[a0.test_year]
L = a0.leads
ics = np.arange(a0.ic_start, len(te) - L, a0.ic_stride)[: a0.max_ics or None]
w = np.cos(np.deg2rad(cl.lat))
w = (w / w.mean()).astype(np.float64)
dyn = np.asarray(DYN)
eps = R.rkt().EPS
cm, cs = cl.mean[dyn][:, None, None], cl.std[dyn][:, None, None] + eps
clim_of = lambda j: np.stack([(cl.clim[np.minimum(np.arange(k + 1, k + 1 + L), 365)][:, dyn] - cm) / cs for k in j]).astype(np.float64)  # noqa: E731
s_inc = cl.inc[DYN].astype(np.float64)[None, :, None, None]


def side_stats(e, pairs):
    """e (M, L, 66, H, W) -> mean, variance, sampling variance of the mean, S = E|X - X'| of two independent draws."""
    M = e.shape[0]
    m = e.mean(0)
    s = np.sort(e, axis=0)
    i = np.arange(1, M + 1).reshape((M,) + (1,) * (e.ndim - 1))
    tot = 2 * ((2 * i - M - 1) * s).sum(0)
    if pairs:
        P = M // 2
        v = ((e - m) ** 2).mean(0)
        return m, v, np.zeros_like(v), (tot - 2 * np.abs(e[:P] - e[P:]).sum(0)) / (M * (M - 2))
    v = e.var(0, ddof=1)
    return m, v, v / M, tot / (M * (M - 1))


def add_mix(mx, ex, ea, eb, y, c):
    """One IC: the mixture of the members ea, eb (M, L, 66, H, W) into the scorer mx (truth y, climatology c)."""
    ma, va, na, Sa = side_stats(ea, mx.pairs)
    mb, vb, nb, Sb = side_stats(eb, mx.pairs)
    C = sum(np.abs(x[None] - eb).mean(0) for x in ea) / ea.shape[0]
    mu = 0.5 * (ma + mb)
    var = np.maximum(0.5 * (va + vb) + 0.25 * ((ma - mb) ** 2 - na - nb), 0.0)
    crps = 0.5 * (np.abs(ea - y[None]).mean(0) + np.abs(eb - y[None]).mean(0)) - 0.5 * (0.25 * Sa + 0.25 * Sb + 0.5 * C)
    mx.M = ea.shape[0] + eb.shape[0]
    mx.add("mse", area_mean((mu - y) ** 2, w)[None])
    pa_, ta = mu - c, y - c
    mf, mo = area_mean(pa_, w), area_mean(ta, w)
    vf = np.maximum(area_mean(pa_ ** 2, w) - mf ** 2, 0)
    vo = np.maximum(area_mean(ta ** 2, w) - mo ** 2, 1e-30)
    r = (area_mean(pa_ * ta, w) - mf * mo) / np.sqrt(vf * vo + 1e-30)
    act = np.sqrt(vf / vo)
    mx.add("acc", (area_mean(pa_ * ta, w) / np.sqrt(area_mean(pa_ ** 2, w) * area_mean(ta ** 2, w) + 1e-30))[None])
    mx.add("act", act[None])
    mx.add("terms", np.stack([(mf - mo) ** 2, vo * (1 - r ** 2), vo * (act - r) ** 2], -1)[None])
    mx.add("var", area_mean(var, w)[None])
    v = np.maximum(var / s_inc ** 2, 1e-4)
    e2 = (mu - y) ** 2 / s_inc ** 2
    P, M = mx.pairs, mx.M
    cf = ((P - 2) / P if P > 2 else 1.0) if P else (M * (M - 3) / ((M + 1) * (M - 1)) if M > 3 else 1.0)
    mx.add("H", area_mean(0.5 * (LOG2PI + 1 + np.log(v)), w)[None])
    mx.add("IGN", area_mean(0.5 * (LOG2PI + np.log(v)) + cf * e2 / (2 * v), w)[None])
    mx.add("logerr", area_mean(0.5 * np.log(np.maximum(e2, 1e-8)), w)[None])
    mx.add("crps", area_mean(crps, w)[None])
    ex.append(np.stack([area_mean(0.25 * (na + nb), w), area_mean(0.25 * (ma - mb) ** 2, w), area_mean(0.5 * (va + vb), w),
                        area_mean(0.125 * (2 * C - Sa - Sb), w)], -1))


PROJ = None
if A.proj:
    from fpk.sphere import SHT, bands  # noqa: E402
    if not (sa["pairs"] and sb["pairs"]):
        raise SystemExit("--proj: two kernels (their means are known)")
    pz = np.load(A.proj)
    pc, nb = [str(x) for x in pz["cores"]], int(pz["nb"])
    sht = SHT(cl.lat)
    bj, nb2 = bands(sht.L)
    assert nb2 == nb
    masks = [(bj == b).float()[:, None] for b in range(nb)]
    Sm, cm_, oo = pz["S"].mean(0), pz["c"].mean(0), pz["oo"].mean(0)          # (L, 66, n, n), (L, 66, n), (L, 66)
    PROJ = []
    for s in (sa, sb):
        k = pc.index(s["a"].core_tag)
        ix = np.arange(k * nb, (k + 1) * nb)
        S_, c_ = Sm[:, :, ix][:, :, :, ix], cm_[:, :, ix]
        g = np.linalg.solve(S_ + 1e-9 * np.eye(nb), c_[..., None])[..., 0]    # (L, 66, nb): the strength of the condition
        raw = S_.sum((-1, -2)) - 2 * c_.sum(-1) + oo
        new_ = np.einsum("lci,lcij,lcj->lc", g, S_, g) - 2 * (g * c_).sum(-1) + oo
        kap = np.sqrt(np.clip(new_ / np.maximum(raw, 1e-30), 0.25, 1.0))      # the error left: the spread follows it
        g[:, TISR_DYN] = 1.0
        kap[:, TISR_DYN] = 1.0
        PROJ.append((torch.tensor(g, dtype=torch.float32), kap.astype(np.float32)))


def readout(ens, cj, g, kap):
    """ens (B, M, L, 66, H, W) -> the same with the mean climatology + sum_b g_b (mean - climatology)_b, deviations x kap."""
    m = ens.mean(1)
    f = torch.from_numpy(m - cj.astype(np.float32))
    c = sht.analysis(f)
    x = torch.stack([sht.synthesis(c * mk) for mk in masks], 3)               # (B, L, 66, nb, H, W)
    x[:, :, :, -1] += f - x.sum(3)
    m2 = cj.astype(np.float32) + (x * g[None, :, :, :, None, None]).sum(3).numpy()
    return m2[:, None] + kap[None, None, :, :, None, None] * (ens - m[:, None])


sc = {k: Scorer(L, w, cl.inc, cl.std, pairs=s["pairs"]) for k, s in (("a", sa), ("b", sb))}
mixes = {A.tag: (Scorer(L, w, cl.inc, cl.std, pairs=sa["pairs"] + sb["pairs"]), [], None)}
PA, PB = sa["pairs"], sb["pairs"]
if PA >= 4 and PB >= 4:   # the same budget as ONE kernel: half the pairs of each side (TAG_h), no further rollout
    mixes[A.tag + "_h"] = (Scorer(L, w, cl.inc, cl.std, pairs=PA // 2 + PB // 2), [],
                           (np.r_[0:PA // 2, PA:PA + PA // 2], np.r_[0:PB // 2, PB:PB + PB // 2]))
t0 = time.time()
for i in range(0, len(ics), a0.ic_batch):
    j = ics[i:i + a0.ic_batch]
    truth = np.stack([te[k + 1:k + 1 + L][:, dyn] for k in j]).astype(np.float64)
    tis = torch.from_numpy(np.stack([te[k + 1:k + 1 + L, TISR] for k in j]))
    cj = clim_of(j)
    ens = []
    for s in (sa, sb):
        Wm = int(getattr(s["core"], "mem", 0))
        hist = torch.from_numpy(np.stack([te[np.maximum(np.arange(k - Wm, k), 0)][:, dyn] for k in j])) if Wm else None
        with torch.no_grad():
            ens.append(s["run"](torch.from_numpy(te[j][:, dyn]), torch.tensor(j), tis, hist, s["gen"]).cpu().numpy())
        if PROJ is not None:
            ens[-1] = readout(ens[-1], cj, *PROJ[len(ens) - 1])
    for b in range(len(j)):
        ea, eb, y, c = ens[0][b].astype(np.float64), ens[1][b].astype(np.float64), truth[b], cj[b]
        sc["a"].update(ea[None], y[None], clim=c[None])
        sc["b"].update(eb[None], y[None], clim=c[None])
        for mx, ex, sel in mixes.values():
            add_mix(mx, ex, ea if sel is None else ea[sel[0]], eb if sel is None else eb[sel[1]], y, c)
    print(json.dumps(dict(ics=int(i + len(j)), sec=round(time.time() - t0), rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)), flush=True)

clim = Scorer(L, w, cl.inc, cl.std)
for i in range(0, len(ics), a0.ic_batch):
    j = ics[i:i + a0.ic_batch]
    c = clim_of(j)
    clim.update(c[:, None], np.stack([te[k + 1:k + 1 + L][:, dyn] for k in j]).astype(np.float64), clim=c)
Z = list(DYN).index(contract.CHANNEL_INDEX["z500"])
stdz = float(cl.std[DYN][Z])
Q = [0, 2, 4, 6, 9]
for tag, (mx, ex, sel) in mixes.items():
    sfx = "_p" if PROJ is not None else ""
    tags = {tag: mx, sa["arm"] + sfx: sc["a"], sb["arm"] + sfx: sc["b"], "climatology": clim}
    res = {k: v.summary() for k, v in tags.items()}
    ex = np.concatenate([x[None] for x in ex], 0)                     # (IC, L, 66, 4)
    mse = np.concatenate(mx.acc["mse"], 0)
    res[tag].update(
        members=mx.M, pairs=mx.pairs, sides=[A.a, A.b],
        rmse_z500_inf=(np.sqrt(np.maximum(mse[:, :, Z] - ex[:, :, Z, 0], 0).mean(0)) * stdz).tolist(),   # the mixture's own mean
        between_var_share_z500=(ex[:, :, Z, 1].mean(0) / np.maximum(ex[:, :, Z, 1].mean(0) + ex[:, :, Z, 2].mean(0), 1e-30)).tolist())
    raws = {f"{k}__{n}": v for k, s in tags.items() for n, v in s.raw().items()}
    raws[f"{tag}__mix"] = ex.astype(np.float32)                       # mean_noise, between-side variance, within, CRPS gain
    np.savez_compressed(f"{a0.out}/score_{tag}_raw.npz", **raws, ics=ics, std=cl.std[dyn], inc=cl.inc[dyn])
    json.dump(dict(meta=dict(test_year=a0.test_year, ics=len(ics), channels=[contract.CHANNEL_NAMES[c] for c in DYN], ic_stride=a0.ic_stride,
                             members=mx.M, leads=L, sides=[A.a, A.b], sec=round(time.time() - t0)), scores=res),
              open(f"{a0.out}/score_{tag}.json", "w"), indent=1)
    for k in (tag,) + ((sa["arm"] + sfx, sb["arm"] + sfx) if sel is None else ()):
        r = res[k]
        print(k, "nrmse", np.round(np.array(r["nrmse"])[Q], 4), "crps", np.round(np.array(r["crps"])[Q], 4), "ign", np.round(np.array(r["ign"])[Q], 4),
              "ss", np.round(np.array(r["spread_skill"])[Q], 3), "z500", np.round(np.array(r["rmse_phys"]["z500"])[Q], 1), flush=True)
    print(tag, "z500 inf", np.round(np.array(res[tag]["rmse_z500_inf"])[Q], 1), "between share", np.round(np.array(res[tag]["between_var_share_z500"])[Q], 3), flush=True)
