"""Is the error of the mean non-Gaussian at long leads in the large scales, or only mis-scaled? (validation year)

    # 1. rollout (the core, the kernel, the data): the raw spectral coefficients -> .npz
    python scripts/diag_nongauss.py --core-tag core_gngmz8L --dev-tag devk_band_sph --npz runs/fpk/diag_nongauss_devk_band_sph.npz
    # 2. analysis from the .npz alone (numpy and scipy: no core, no data)
    python scripts/diag_nongauss.py --from-npz runs/fpk/diag_nongauss_devk_band_sph.npz --json docs/results/diag_nongauss.json

The .npz, in units of s: per lead, initial condition (N), variable (z500, t850, u250, msl) and mode (l <= 15: per
degree Re c_l0, then Re c_lm, Im c_lm for m = 1..l; fpk.sphere's coefficients)
    err  (leads, N, V, n)        the coefficient of the error of the mean, (truth - xbar) / s
    mem  (leads, N, 2 P, V, n)   the coefficient of every member's deviation (x_m - xbar) / s: +e_1..+e_P, -e_1..-e_P
    l, m, part  (n,)             degree, order, part (0 real, 1 imaginary) of the mode;   meta: a json string

The ensemble is P antithetic pairs +-e_p: its mean is known (0), so the predicted variance of a coefficient on a day,
S^2 = (1/P) sum_p e_p^2, has P degrees of freedom. If the kernel is a calibrated Gaussian (error and e_p independent
N(0, sigma^2), sigma the day's and the mode's own),
    z_ens = err / S  is Student-t with P d.o.f.: variance P / (P - 2) = 2.0, P(|z| > 2) = 11.6 %, P(|z| > 3) = 4.0 %
                     at P = 4 (not the Gaussian's 1, 4.6 %, 0.3 %);
    the rank of err among the 2 P members is P + sign(err) #{p: |e_p| < |err|}: the middle rank P has probability
                     1 / (P + 1), each of the 2 P others 1 / (2 (P + 1)) (the pairs make it symmetric, not uniform).

Statistics per variable, lead and band (l = 1, 2-3, 4-7, 8-15), from the band's n modes x N initial conditions:

  calibration (with the ensemble)
    rms        error rms / predicted rms, the factor of the kernel's spread: sqrt of the mean over the band's modes of
               sum_i err_i^2 / sum_i S_i^2 (i: the initial conditions; per mode, so a spectrum inside the band does
               not weigh the modes). 1 if E[err^2] = E[S^2], whatever the shape of the error. Its null is the cell's
               OWN (``rms_null``): T = err^2 + sum_p e_p^2 is sufficient for the sigma of the day and mode, and given
               T, err^2 = T B with B ~ Beta(1/2, P/2) whatever sigma is; the Monte Carlo draws B for the cell's T. (The
               null of a spread that is the same every day, the band's line in the table, is too narrow when the
               kernel's spread changes from day to day: the days of large sigma dominate the sums.)
    f          the scale of the BULK of z_ens against its t null: the maximum-likelihood f of z ~ f t_P (``tscale``).
               It is the rms factor only for a kernel that is Gaussian up to a factor: tails that the kernel does not
               predict leave f below rms (the kernel's variance is right, the bulk is narrower), regimes above
    var_ens    mean z_ens^2 (null 2.0; the Gaussian log score of the ensemble's S^2 times a factor is minimal at the
               factor var_ens, (P - 2) / P var_ens for the kernel's own sigma^2: the same quantity as rms^2 with every
               day divided by its own S^2, and far noisier, z^2 has no variance at P = 4);   gt2, gt3: the fractions
               |z_ens| > 2, 3
    rank_chi2  chi-square of the rank histogram (2 P + 1 bins, in the json) against the antithetic null above
    kurt_mem   excess kurtosis of the members' own coefficients e_p / S: -6 / (P + 2) if they are Gaussian over the
               pairs. Outside its null, the t null of everything that divides by S is not exact (dagger in the table)
  shape of the error itself (without the ensemble). Every mode is standardized on its own over the initial conditions,
  u = (err - mean) / std, and only then pooled: modes of different variance in a band (a steep spectrum) are a scale
  mixture, with positive kurtosis even if every mode is Gaussian
    bias       mean over the modes of N mean^2 / std^2 (the mode's mean error in units of its standard error, squared:
               1 without a bias). The shape is taken about the mean: a Gaussian with a bias has negative excess
               kurtosis about 0 and would read as two regimes; the bias itself counts in rms and f (it is error)
    kurt_mode  mean u^4 - 3 pooled over the band. Two regimes make it NEGATIVE (two Gaussians d sigma apart:
               -2 d^4 / (4 + d^2)^2 = -0.08, -0.50, -0.96 at d = 1, 2, 3), a spread that changes from day to day POSITIVE
    kurt_min   the smallest excess kurtosis of a single mode (regimes that live in one mode of the band)
  shape after dividing by the predicted spread of the day and mode
    kurt_ens   z = err / S per mode: minus its median, divided by its own scale (``tscale``), mapped to a Gaussian
               through the t null, g = Phi^-1(T_P(z)), standardized; mean g^4 - 3 pooled. A kernel that is Gaussian up
               to a factor per mode gives the null whatever its spread does from day to day. Positive: tails heavier
               than the kernel's day-to-day spread explains; negative: regimes
  bimodality
    p_bim      ONE test of two regimes per band: the smallest of the three lower-tail null probabilities of kurt_mode,
               kurt_min and kurt_ens, and p_bim = the null probability of a smallest one as small (``bim_p``; the three
               are dependent, so their minimum has its own Monte Carlo null). Uniform for a Gaussian kernel of any
               dispersion (the three do not see the scale); not below about 4 / (--null-reps + 1), the null draws
               that hold the smallest value of one of the three (``p_bim_floor``)

Every statistic has its null: --null-reps Monte Carlo bands of the calibrated Gaussian kernel with the file's P and N
and the band's n, through the same code (rms: the cell's own, above). The null's mean and central 95 % interval stand
next to every value, with a flag: -1 / +1 below / above the 95 % interval, -2 / +2 outside the 98 % interval too. The
biases of the estimates (-6 / (N + 1) of the kurtosis of N standardized values, ...) are in the null's mean. The null
takes the initial conditions and the modes of a band as independent (errors that are coherent over several modes or
over initial conditions a week apart make its intervals too narrow).

Verdict of a band, the first that applies (-- / ++: the double flags):
    bimodal signal                                        p_bim < 0.005
    heavy tails beyond the predicted day-to-day spread    kurt_ens ++
    Gaussian, under-dispersed (rms factor, bulk)          rms ++, f not --
    over-dispersed (rms factor, bulk)                     rms --, f not ++
    heavy tails (bulk narrower than the rms)              f --   (and rms not --)
    light tails (bulk wider than the rms)                 f ++   (and rms not ++)
    calibrated Gaussian                                   otherwise (kurt_mode above alone: heavy tails of the pooled
                                                          error that the kernel's day-to-day spread explains)
Under- and over-dispersion are decided by rms, the factor the kernel's spread is to be multiplied with; f decides
nothing about the variance. Where the two disagree the kernel's distribution has another shape than the error's, and
its spread is NOT to be multiplied with f. Measured (4000 null draws, 1000 bands a case, N = 48, P = 4), for a kernel
with the RIGHT variance, E[err^2] = E[S^2], that does not follow a day-to-day sigma = exp(0.5 Z) of the error (rms
0.97 to 0.99 in the median, f = 0.84), in the bands l = 1, 2-3, 4-7:
    heavy tails beyond the predicted day-to-day spread 0.17, 0.73, 1.00;   heavy tails (bulk narrower than the rms)
    0.29, 0.15, 0.00;   over-dispersed 0.12, 0.09, 0.00 (0.41, 0.26, 0.01 if f decided);   under-dispersed 0.05, 0.01,
    0.00;   calibrated 0.38, 0.02, 0.00
The dispersion verdicts that are left are no longer f's: the rms ratio of 48 such days scatters more than its null,
which is the Gaussian kernel's (rms is flagged -- in 0.13 to 0.39 and ++ in 0.08 to 0.30 of these bands), so "over-"
and "under-dispersed" next to flagged tails (f -- with rms inside, kurt_ens +) say less than in a Gaussian kernel.
With exp(0.3 Z) (f = 0.94): over- or under-dispersed 0.07, 0.15, 0.20, one of the two heavy tails 0.11, 0.31, 0.71.
The same error 1.3 x too large has rms = 1.27 to 1.29 and f = 1.09 to 1.10. The other way, a Gaussian kernel whose
dispersion is off is given one of the two "bulk" verdicts where f sees the factor and rms not yet: in at most 0.04
of the bands (factors 0.9, 1.15, 1.3) if its spread is the same every day (rms is then the sharper of the two), in
up to 0.35 (1.3 in l = 1, 0.9 in l = 2-3) if its spread changes as exp(0.6 Z) from day to day, which widens the null
of rms and not that of f; the line then shows rms and f equal.

By chance, in a calibrated Gaussian kernel: a flag is not 0 in 5 % of the bands; the verdict is another one in about
5 % (``false``: 4.5 % of the null's own draws; 4.5 to 6 % of other draws), and "bimodal signal" in about 0.5 % (the
level of p_bim; 0.2 to 0.7 % of other draws, as the null's own smallest values scatter; for a Gaussian kernel of any
dispersion). The printed table has 4 variables x 4 leads x 4 bands = 64 cells, so about 3 of them carry another
verdict and 0.32 "bimodal signal" (0.16 of the 32 cells of leads 7 and 10; 0.22 at 0.7 %): the table's first lines
say so (``summary`` in the json). The question itself, "any regimes at
leads >= 7 in l <= 7", is ONE test over the table's 24 cells there (``summary.regimes``): p = min(1, 24 x the
smallest p_bim), Bonferroni's bound, which holds whatever the dependence between the cells (neighbouring leads and
the variables share the initial conditions). It cannot be below 24 x p_bim's floor, about 0.024 with 4000 null draws
(0.005 with 20000).

``power`` (per band, in the json and the table): the share of bands given "bimodal signal" when the error of every
mode (or of one mode) of the band is a symmetric mixture of two Gaussians d = 1, 2, 3 within-regime sigma apart and
the members are Gaussian with the error's variance (300 bands a case: +-0.03, and the level's own Monte Carlo
scatter). With 48 initial conditions and 4 pairs, for l = 1, 2-3, 4-7, 8-15: regimes in every mode of a band are
found at d = 3 (0.7, 1.0, 1.0, 1.0), at d = 2 in the bands with enough modes (0.07, 0.45, 1.0, 1.0) and not at d = 1
(0.01 to 0.05); regimes in ONE mode are found in 0.13, 0.11, 0.05, 0.01 of the cases at d = 3 (0.92 to 0.97 at d = 6
in l <= 7). This is the price of the level 0.5 % (three one-sided 1 % tests, 3 % by chance, found 0.9 and 0.76 in l = 1
at d = 3 and l = 2-3 at d = 2). Regimes together with a spread that changes from day to day cancel in kurt_mode
(d = 3 and sigma = exp(0.5 Z): +1.1): kurt_ens is then the one that sees them, if the kernel's spread follows the
day's (found in 0.12, 0.41, 0.97 of the cases for l = 1, 2-3, 4-7; "light tails (bulk wider than the rms)" in 0.07,
0.32, 0.03 more). The json also holds, for a histogram, the day-10 values u and z_ens of the modes l <= 3 of z500
(``leading``).
"""

import argparse
import json
import os
import sys
from functools import lru_cache

import numpy as np
from scipy.special import ndtri, stdtr

VARS = ("z500", "t850", "u250", "msl")
BANDS = ((1, 1), (2, 3), (4, 7), (8, 15))
LMAX = 15
LEADS = (1, 5, 7, 10)                                                  # the leads of the printed table
STATS = ("rms", "f", "var_ens", "gt2", "gt3", "rank_chi2", "kurt_mem", "bias", "kurt_mode", "kurt_min", "kurt_ens")
BIM = ("kurt_mode", "kurt_min", "kurt_ens")                            # the bimodality test: their lower tails
ALPHA_BIM = 0.005                                                      # "bimodal signal": p_bim below this
FMT = dict(gt2=".3f", gt3=".3f", rank_chi2=".1f")
VERDICT = ("calibrated Gaussian", "Gaussian, under-dispersed", "over-dispersed",
           "heavy tails beyond the predicted day-to-day spread", "bimodal signal",
           "heavy tails (bulk narrower than the rms)", "light tails (bulk wider than the rms)")
FACTORS = (" (rms factor {rms:.2f}, bulk {f:.2f})", ": rms factor {rms:.2f}, bulk {f:.2f}")   # verdicts 1, 2 and 5, 6
label = lambda b: str(b[0]) if b[0] == b[1] else f"{b[0]}-{b[1]}"  # noqa: E731
p4 = lambda x: np.square(np.square(x))  # noqa: E731


def modes(lmax):
    """(l, m, part), each (n,), of the n = (lmax + 1)^2 real values of the coefficients up to degree lmax: per
    degree Re c_l0, then Re c_lm, Im c_lm for m = 1..l."""
    t = [(l, m, p) for l in range(lmax + 1) for m in range(l + 1) for p in ((0,) if m == 0 else (0, 1))]
    return tuple(np.array(c) for c in zip(*t))


def save(path, err, mem, meta):
    """The .npz of the module's docstring: err (leads, N, V, n), mem (leads, N, 2 P, V, n), n = (lmax + 1)^2 values
    in the order of ``modes``; meta: a dict with at least ``vars`` (the V names)."""
    l, m, part = modes(int(round(np.sqrt(err.shape[-1]))) - 1)
    assert err.shape[-1] == mem.shape[-1] == l.size and len(meta["vars"]) == err.shape[2] == mem.shape[3]
    np.savez_compressed(path, err=np.asarray(err, np.float32), mem=np.asarray(mem, np.float32), l=l, m=m, part=part,
                        meta=json.dumps(meta))


def rank_null(P):
    """The probabilities of the 2 P + 1 ranks of the truth among P antithetic pairs of a calibrated kernel."""
    return np.r_[np.full(P, 0.5 / (P + 1)), 1.0 / (P + 1), np.full(P, 0.5 / (P + 1))]


def tscale(z, P, axis):
    """The maximum-likelihood scale f of z ~ f t_P along ``axis`` (kept): the fixed point of
    f^2 = mean[(P + 1) z^2 / (P + z^2 / f^2)], started at the median of z^2."""
    z2 = z ** 2
    f2 = np.median(z2, axis, keepdims=True) + 1e-300
    for _ in range(30):
        f2 = ((P + 1) * z2 / (P + z2 / f2)).mean(axis, keepdims=True)
    return np.sqrt(f2)


def band_stats(E, e):
    """E (..., N, n) the error's coefficients of a band, e (..., N, P, n) the pairs' deviations -> the statistics
    of the module's docstring, each (...,), and the rank histogram (..., 2 P + 1) (fractions)."""
    N, P, n = e.shape[-3:]
    S2 = (e ** 2).mean(-2)                                             # the predicted variance of the day and mode
    S = np.sqrt(S2) + 1e-300
    z = E / S
    out = dict(rms=np.sqrt(((E ** 2).sum(-2) / (S2.sum(-2) + 1e-300)).mean(-1)),
               f=tscale(z.reshape(*z.shape[:-2], -1), P, -1)[..., 0], var_ens=(z ** 2).mean((-1, -2)),
               gt2=(np.abs(z) > 2).mean((-1, -2)), gt3=(np.abs(z) > 3).mean((-1, -2)))
    rank = P + np.sign(E).astype(int) * (np.abs(e) < np.abs(E)[..., None, :]).sum(-2)
    hist = np.stack([(rank == r).mean((-1, -2)) for r in range(2 * P + 1)], -1)
    out["rank_chi2"] = N * n * ((hist - rank_null(P)) ** 2 / rank_null(P)).sum(-1)
    out["kurt_mem"] = p4(e / S[..., None, :]).mean((-1, -2, -3)) - 3
    m, sd = E.mean(-2, keepdims=True), E.std(-2, keepdims=True) + 1e-300
    k = p4((E - m) / sd).mean(-2) - 3                                  # the excess kurtosis of every mode
    out.update(bias=(N * (m / sd) ** 2).mean((-1, -2)), kurt_mode=k.mean(-1), kurt_min=k.min(-1))
    z = z - np.median(z, -2, keepdims=True)
    z = z / tscale(z, P, -2)
    g = -np.sign(z) * ndtri(stdtr(P, -np.abs(z)))                      # Phi^-1(T_P(z)), through the lower tail
    g = (g - g.mean(-2, keepdims=True)) / (g.std(-2, keepdims=True) + 1e-300)
    out["kurt_ens"] = p4(g).mean((-1, -2)) - 3
    return out, hist


def rms_null(E, e, reps, rng):
    """The null of ``rms`` of the cells E (..., N, n), e (..., N, P, n), each cell's own: (..., 5) = mean, 2.5 %,
    97.5 %, 1 %, 99 % over ``reps`` draws. A calibrated Gaussian kernel with ANY sigma per day and mode: given
    T = err^2 + sum_p e_p^2 (sufficient for sigma), err^2 = T B and sum_p e_p^2 = T (1 - B), B ~ Beta(1/2, P/2)
    independent. So rms^2 = mean over the modes of P A / (W - A), A = sum_i T_i B_i, W = sum_i T_i, with the cell's T
    (the same draws of B for every cell)."""
    N, P, n = e.shape[-3:]
    T = np.moveaxis((E ** 2 + (e ** 2).sum(-2)).reshape(-1, N, n), -1, 0) + 1e-300     # (n, cells, N)
    W, c, s = T.sum(-1, keepdims=True), max(1, 4_000_000 // (max(N, T.shape[1]) * n)), []   # draws at a time
    for i in range(0, reps, c):
        A = T @ rng.beta(0.5, 0.5 * P, (n, N, min(c, reps - i)))                       # (n, cells, draws)
        s.append(np.sqrt((P * A / (W - A)).mean(0)))
    s = np.concatenate(s, -1)
    q = np.stack([s.mean(-1), *np.quantile(s, (0.025, 0.975, 0.01, 0.99), axis=-1)], -1)
    return q.reshape(*E.shape[:-2], 5)


def flags(s, q):
    """The statistic against its null q[stat] (..., 5) = (mean, 2.5 %, 97.5 %, 1 %, 99 %): 0 inside the 95 %
    interval, -1 / +1 below / above it, -2 / +2 also outside the 98 % interval."""
    out = {}
    for k in q:
        v, n = np.asarray(s[k]), np.asarray(q[k])
        out[k] = (v > n[..., 2]).astype(int) + (v > n[..., 4]) - (v < n[..., 1]) - (v < n[..., 3])
    return out


def bim_p(s, nl):
    """The bimodality test of bands, statistics s[k] (...,), against the null ``nl`` of R draws: p_k = (the number
    of null draws of BIM's statistic k at or below s[k], + 1) / (R + 1), and p_bim = (the number of null draws whose
    own smallest p_k is at or below min_k p_k, + 1) / (R + 1). -> (p_bim (...,), the index in BIM of the smallest)."""
    R = nl["bim"].size
    pk = np.stack([(np.searchsorted(nl["sorted"][k], s[k], side="right") + 1) / (R + 1.0) for k in BIM])
    return (np.searchsorted(nl["bim"], pk.min(0), side="right") + 1) / (R + 1.0), pk.argmin(0)


def decide(fl, pb):
    """The index in VERDICT from the double flags and p_bim (the rule of the module's docstring)."""
    r, f = fl["rms"], fl["f"]
    return np.select([pb < ALPHA_BIM, fl["kurt_ens"] > 1, (r > 1) & (f > -2), (r < -1) & (f < 2), f < -1, f > 1],
                     [4, 3, 1, 2, 5, 6], 0)


def draws(rng, R, N, P, n, d=0.0, one=False):
    """R bands of a calibrated Gaussian kernel, unit variance: err (R, N, n), pairs (R, N, P, n). d > 0: the error
    of every mode (``one``: of the first mode) is a symmetric mixture of two Gaussians d within-regime sigma apart,
    the members Gaussian with the error's variance 1 + d^2 / 4."""
    E, e = rng.standard_normal((R, N, n)), rng.standard_normal((R, N, P, n))
    if d:
        k = slice(0, 1) if one else slice(None)
        E[..., k] += 0.5 * d * rng.choice([-1.0, 1.0], E[..., k].shape)
        e[..., k] *= np.sqrt(1 + 0.25 * d * d)
    return E, e


@lru_cache(maxsize=None)
def null(N, P, n, reps=4000, power_reps=300, seed=0):
    """The Monte Carlo reference of a band of n modes x N initial conditions with P pairs, from ``reps`` calibrated
    Gaussian bands: q[stat] = (mean, 2.5 %, 97.5 %, 1 %, 99 %) (rms: with the same sigma every day; a cell has its
    own, ``rms_null``), ``sorted`` and ``bim`` for ``bim_p`` and ``p_bim_floor`` (the smallest p_bim these draws can
    give), ``false`` = the fraction of the draws with another verdict, and ``power`` = the fraction of ``power_reps``
    two-regime bands (d = 1, 2, 3; every mode / one mode) given "bimodal signal"."""
    rng = np.random.default_rng([seed, N, P, n])
    c = max(1, 2_000_000 // (N * P * n))                               # bands at a time

    def sim(R, **kw):
        t = [band_stats(*draws(rng, min(c, R - i), N, P, n, **kw))[0] for i in range(0, R, c)]
        return {k: np.concatenate([p[k] for p in t]) for k in t[0]}
    s = sim(reps)
    q = {k: (float(s[k].mean()), *(float(np.quantile(s[k], p)) for p in (0.025, 0.975, 0.01, 0.99))) for k in STATS}
    rank = np.stack([np.argsort(np.argsort(s[k])) + 1 for k in BIM])    # 1..reps
    nl = dict(q=q, sorted={k: np.sort(s[k]) for k in BIM}, bim=np.sort(rank.min(0) / (reps + 1.0)))
    nl["p_bim_floor"] = float(bim_p({k: -np.inf for k in BIM}, nl)[0])
    verdict = lambda t: decide(flags(t, q), bim_p(t, nl)[0])  # noqa: E731
    nl["false"] = float((verdict(s) > 0).mean())
    nl["power"] = {w: [float((verdict(sim(power_reps, d=d, one=w == "one")) == 4).mean()) for d in (1, 2, 3)]
                   for w in ("all", "one")}
    return nl


def analyse(d, reps=4000, power_reps=300, seed=0):
    """The arrays of the .npz (a dict) -> the json: meta, null[band] = {statistic: [mean, 2.5 %, 97.5 %, 1 %, 99 %],
    n, false, false_bimodal, p_bim_floor, power}, [variable][lead][band] = {statistic: [value, null mean, null 2.5 %,
    null 97.5 %, flag], p_bim, bim_by, rank_hist, v, verdict}, ``leading``, and ``summary`` (of the leads LEADS)."""
    err, mem, ml = np.asarray(d["err"], np.float64), np.asarray(d["mem"], np.float64), np.asarray(d["l"])
    meta = json.loads(str(d["meta"]))
    (L, N, V, _), P = err.shape, mem.shape[2] // 2
    if mem.shape[2] != 2 * P or not np.allclose(mem[:, :, :P], -mem[:, :, P:], atol=1e-6 * np.abs(mem).max()):
        raise SystemExit("mem: P antithetic pairs (+e_1..+e_P, -e_1..-e_P) are expected: the nulls are theirs")
    bands = [b for b in BANDS if b[1] <= ml.max()]
    sel = [(ml >= b[0]) & (ml <= b[1]) for b in bands]
    nl = [null(N, P, int(s.sum()), reps, power_reps, seed) for s in sel]
    out = dict(meta=dict(meta, ics=N, pairs=P, leads=L, bands=[list(b) for b in bands], null_reps=reps,
                         alpha_bimodal=ALPHA_BIM, rank_null=rank_null(P).tolist(), stats=list(STATS),
                         cell="statistic: [value, null mean, null 2.5 %, null 97.5 %, flag]; flag: 0 inside the "
                              "null's 95 % interval, -1 / +1 below / above, -2 / +2 outside its 98 % interval too"),
               null={label(b): dict(n=int(s.sum()), false=r["false"], false_bimodal=ALPHA_BIM,
                                    p_bim_floor=r["p_bim_floor"], power=r["power"],
                                    **{k: list(v) for k, v in r["q"].items()}) for b, s, r in zip(bands, sel, nl)})
    out.update({v: [[None] * len(bands) for _ in range(L)] for v in meta["vars"]})
    for bi, (s, r) in enumerate(zip(sel, nl)):                          # every cell of the band at once: (L, V)
        E, e = np.moveaxis(err[..., s], 2, 1), np.moveaxis(mem[:, :, :P][..., s], 3, 1)
        st, hist = band_stats(E, e)
        q = {k: np.broadcast_to(np.asarray(r["q"][k]), (L, V, 5)) for k in STATS}
        q["rms"] = rms_null(E, e, reps, np.random.default_rng([seed, N, P, int(s.sum()), 1]))
        fl = flags(st, q)
        pb, by = bim_p(st, r)
        vd = decide(fl, pb)
        for k in range(L):
            for vi, v in enumerate(meta["vars"]):
                i = int(vd[k, vi])
                txt = VERDICT[i] + (FACTORS[i > 2].format(rms=float(st["rms"][k, vi]), f=float(st["f"][k, vi]))
                                    if i in (1, 2, 5, 6) else "")
                out[v][k][bi] = dict({n_: [float(st[n_][k, vi]), *map(float, q[n_][k, vi, :3]), int(fl[n_][k, vi])]
                                      for n_ in STATS}, p_bim=float(pb[k, vi]), bim_by=BIM[int(by[k, vi])],
                                     rank_hist=hist[k, vi].tolist(), v=i, verdict=txt)
    # the values themselves: the last lead <= 10 of z500 (else the first variable), the modes l = 1..3
    vi, k, s = (meta["vars"].index("z500") if "z500" in meta["vars"] else 0), min(L, 10) - 1, (ml >= 1) & (ml <= 3)
    E, S = err[k, :, vi][:, s], np.sqrt((mem[k, :, :P, vi][:, :, s] ** 2).mean(1))
    out["leading"] = dict(var=meta["vars"][vi], lead=k + 1, l=ml[s].tolist(), m=np.asarray(d["m"])[s].tolist(),
                          part=np.asarray(d["part"])[s].tolist(), mean=E.mean(0).tolist(), std=E.std(0).tolist(),
                          u=((E - E.mean(0)) / E.std(0)).T.tolist(), z_ens=(E / S).T.tolist())    # (modes, N)
    out["summary"] = summary(out)
    return out


def summary(out, leads=LEADS):
    """The verdicts of the table's cells (variables x ``leads`` x bands) against what a calibrated Gaussian kernel
    gives by chance, for ``table`` (all of them) and ``late`` (leads >= 7): {cells, verdicts: {name: count},
    expected_other = the sum of the bands' ``false``, bimodal (the count), expected_bimodal = ALPHA_BIM x cells}; and
    ``regimes``, the one test of "any regimes at leads >= 7 in l <= 7" over the table's cells there: {cells, p_min
    (the smallest p_bim), where: [variable, lead, band, the statistic of BIM], p = min(1, cells x p_min), p_floor
    (the smallest p these null draws can give)} (Bonferroni: valid whatever the dependence between the cells)."""
    m, res = out["meta"], {}
    ks = [k for k in leads if k <= m["leads"]]
    for name, k0 in (("table", 0), ("late", 7)):
        cs = [(out[v][k - 1][bi]["v"], out["null"][label(b)]) for v in m["vars"] for k in ks if k >= k0
              for bi, b in enumerate(m["bands"])]
        vs = [i for i, _ in cs]
        res[name] = dict(cells=len(cs), verdicts={VERDICT[i]: vs.count(i) for i in sorted(set(vs))},
                         expected_other=float(sum(r["false"] for _, r in cs)), bimodal=vs.count(4),
                         expected_bimodal=ALPHA_BIM * len(cs))
    cs = [(out[v][k - 1][bi]["p_bim"], [v, k, label(b), out[v][k - 1][bi]["bim_by"]]) for v in m["vars"] for k in ks
          if k >= 7 for bi, b in enumerate(m["bands"]) if b[1] <= 7]
    if cs:
        p, where = min(cs, key=lambda t: t[0])
        floor = min(out["null"][label(b)]["p_bim_floor"] for b in m["bands"] if b[1] <= 7)
        res["regimes"] = dict(cells=len(cs), p_min=p, where=where, p=min(1.0, len(cs) * p),
                              p_floor=min(1.0, len(cs) * floor))
    return res


def table(out, leads=LEADS):
    """The printed table: what a calibrated Gaussian kernel gives by chance in its cells, per band the null (mean,
    95 % and 98 % intervals) and a line per variable and lead, then the count of the verdicts and the one test of
    regimes at leads >= 7 (``summary``)."""
    m, w, mark = out["meta"], 12, {0: "  ", 1: "+ ", 2: "++", -1: "- ", -2: "--"}
    fmt = lambda k, v: format(v, FMT.get(k, ".2f"))  # noqa: E731
    sm, what = summary(out, leads), dict(table="the table", late="leads >= 7")
    print(f"{m['ics']} initial conditions, {m['pairs']} antithetic pairs; null: {m['null_reps']} Monte Carlo bands of a "
          "calibrated Gaussian kernel. - / +: below / above its 95 % interval, -- / ++: its 98 % interval too (the "
          "verdict's flags; rms: against the cell's own null, given its days' predicted spreads, in the json); "
          f"'bimodal signal': p_bim < {ALPHA_BIM}; dagger: the members are not Gaussian over the pairs (kurt_mem -- / ++)")
    print("A calibrated Gaussian kernel shows by chance: " + "; ".join(
        f"in {what[k]} ({t['cells']} cells) {t['expected_other']:.1f} verdicts other than 'calibrated Gaussian', "
        f"{t['expected_bimodal']:.2f} of them 'bimodal signal'" for k, t in sm.items() if k in what and t["cells"]))
    for bi, b in enumerate(m["bands"]):
        r = out["null"][label(b)]
        pw = lambda t: " ".join(f"{p:.2f}" for p in t)  # noqa: E731
        print(f"\nband l = {label(b)}: {r['n']} modes. Calibrated Gaussian given another verdict: {r['false']:.3f}; "
              f"two regimes d = 1, 2, 3 sigma apart given 'bimodal signal': every mode {pw(r['power']['all'])}, "
              f"one mode {pw(r['power']['one'])}")
        print(f"{'':14s}" + "".join(f"{k:>{w}s}  " for k in m["stats"]) + f"{'p_bim':>8s}")
        print(f"{'null mean':14s}" + "".join(f"{fmt(k, r[k][0]):>{w}s}  " for k in m["stats"]))
        for name, i in (("null 95 %", 1), ("null 98 %", 3)):
            print(f"{name:14s}" + "".join(f"{fmt(k, r[k][i]) + '..' + fmt(k, r[k][i + 1]):>{w}s}  " for k in m["stats"]))
        for v in m["vars"]:
            for k in leads:
                if k > m["leads"]:
                    continue
                c = out[v][k - 1][bi]
                print(f"{v:5s} lead {k:2d} " + "".join(f"{fmt(n, c[n][0]):>{w}s}{mark[c[n][4]]}" for n in m["stats"])
                      + f"{c['p_bim']:8.4f}  {c['verdict']}" + (" (dagger)" if abs(c["kurt_mem"][4]) > 1 else ""))
    print()
    for k, t in sm.items():
        if k in what and t["cells"]:
            print(f"{what[k]}, {t['cells']} cells: " + ", ".join(f"{n} {c}" for n, c in t["verdicts"].items())
                  + f" (by chance: {t['expected_other']:.1f} other than 'calibrated Gaussian', "
                  f"{t['expected_bimodal']:.2f} 'bimodal signal')")
    if "regimes" in sm:
        t = sm["regimes"]
        print(f"any regimes at leads >= 7, l <= 7 ({t['cells']} cells): smallest p_bim {t['p_min']:.4f} ({t['where'][0]} "
              f"lead {t['where'][1]}, l = {t['where'][2]}, {t['where'][3]}); p = {t['p']:.3f} (Bonferroni over the cells: "
              f"holds whatever their dependence; not below {t['p_floor']:.3f} with these null draws)")


def rollout(a, R):
    """Stage 1: the kernel's rollout from --ics initial conditions of the validation year -> the .npz."""
    import torch
    from fpk.sphere import SHT
    a.cloud_mean = False
    if a.threads:
        torch.set_num_threads(a.threads)
    pa, cl, core = R.load_core(a)
    Dk = R.load_devk(a, core, a.dev_tag)
    S = SHT(cl.lat)
    ml, mm, mp = modes(min(LMAX, S.L - 1))
    il, im, re = torch.tensor(ml), torch.tensor(mm), torch.tensor(mp == 0)

    def coef(x):                                                       # (..., H, W) -> (..., n) in the order of modes
        c = S.analysis(x)[..., il, im]
        return torch.where(re, c.real, c.imag)
    va = R.D.load(pa, [a.val_year])[a.val_year]
    names = [R.contract.CHANNEL_NAMES[c] for c in R.DYN]
    ci, dyn = [names.index(v) for v in VARS], np.asarray(R.DYN)
    W, L = int(getattr(core, "mem", 0)), a.leads
    ics = np.linspace(8, len(va) - L - 1, a.ics).astype(int)
    g = torch.Generator().manual_seed(7)
    err, mem = [], []                                                  # per batch: (L, B, V, n), (L, B, 2 P, V, n)
    for i in range(0, len(ics), a.batch):
        j = ics[i:i + a.batch]
        x, doy = torch.from_numpy(va[j][:, dyn]), torch.tensor(j)
        tis = torch.from_numpy(np.stack([va[k + 1:k + 1 + L][:, R.TISR] for k in j]))
        hist = torch.from_numpy(np.stack([va[np.maximum(np.arange(k - W, k), 0)][:, dyn] for k in j])) if W else None
        truth = torch.from_numpy(np.stack([va[k + 1:k + 1 + L][:, dyn] for k in j]))
        eb, mb = [], []
        with torch.no_grad():
            e = Dk.start(x, doy, hist, g)
            for k in range(L):
                x, e, _, _ = Dk.day(x, e, doy + k, tis[:, k], k, g)
                eb.append(coef(((truth[:, k] - x) / Dk.s)[:, ci]))
                mb.append(coef(torch.cat([e, -e], 1)[:, :, ci]))       # the members are xbar +- s e
        err.append(torch.stack(eb))
        mem.append(torch.stack(mb))
        print(json.dumps(dict(ics=i + len(j))), flush=True)
    path = a.npz or os.path.join(a.out, f"diag_nongauss_{a.dev_tag}.npz")
    save(path, torch.cat(err, 1).numpy(), torch.cat(mem, 1).numpy(),
         dict(core_tag=a.core_tag, dev_tag=a.dev_tag, year=a.val_year, vars=list(VARS), ic_days=ics.tolist()))
    print(f"saved {path}; next: python scripts/diag_nongauss.py --from-npz {path} --json {a.json}")


def main(argv=None):
    p = argparse.ArgumentParser(description="stage 1 (rollout -> --npz) takes the options of run_fpk.py as well "
                                            "(--core-tag, --out, --val-year, ...); stage 2 is --from-npz")
    p.add_argument("--from-npz", default="", help="stage 2: analyse this file (no core, no data)")
    p.add_argument("--json", default="docs/results/diag_nongauss.json")
    p.add_argument("--null-reps", type=int, default=4000,
                   help="Monte Carlo bands of the null, per band (p_bim is not below 1 / (this + 1): at least 1000)")
    p.add_argument("--power-reps", type=int, default=300, help="Monte Carlo bands of every power estimate")
    p.add_argument("--null-seed", type=int, default=0)
    p.add_argument("--npz", default="", help="stage 1: the coefficients' file (default <out>/diag_nongauss_<dev-tag>.npz)")
    p.add_argument("--dev-tag", default="devk_band_sph")
    p.add_argument("--ics", type=int, default=48)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--leads", type=int, default=10)
    a, rest = p.parse_known_args(argv)
    if not a.from_npz:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import run_fpk as R
        R.common(p)
        return rollout(p.parse_args(argv), R)
    if rest:
        p.error(f"not an option of --from-npz: {' '.join(rest)}")
    out = analyse(dict(np.load(a.from_npz)), a.null_reps, a.power_reps, a.null_seed)
    out["meta"]["npz"] = a.from_npz
    json.dump(out, open(a.json, "w"))
    table(out)


if __name__ == "__main__":
    main()
