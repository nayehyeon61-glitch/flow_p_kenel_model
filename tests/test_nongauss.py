"""scripts/diag_nongauss.py: the analysis stage on synthetic .npz files, and the rollout stage on the toy core.

Synthetic kernels of P = 4 antithetic pairs and 48 initial conditions; every (lead, variable) of a file is an
independent sample (a repeated seed) of a band. The null of a band is the script's own Monte Carlo (REPS draws).
The bounds on shares are about 4 standard deviations (binomial) or more from the expected share, so that they do not
depend on the seeds (the file passes with seven other sets of seeds: every seed of ``synth`` shifted)."""

import json
import math
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from test_fpk import H, W, mcore  # noqa: E402,F401  (the fixture)
import diag_nongauss as G  # noqa: E402

REPS, PREPS, N, P = 2000, 100, 48, 4


def synth(path, seed, leads=4, lmax=3, f=1.0, mix=0.0, known=True, d=0.0, dband=(2, 3), one=False, steep=0.0, bias=0.0):
    """A synthetic .npz. Per lead, initial condition and variable a day's sigma = exp(mix x Gaussian); the error of a
    mode is amp (bias + f sigma (Z + regimes)): regimes = +-d/2 in the modes of ``dband`` (``one``: in its first
    mode only); the members are amp sigma x Gaussian with the error's variance at f = 1 (``known`` False: they do
    not know the day's sigma, their variance is 1); amp = (1 + index of the mode)^-steep, a spectrum inside the bands."""
    rng = np.random.default_rng(seed)
    l, m, part = G.modes(lmax)
    n, V = l.size, len(G.VARS)
    amp = (1.0 + np.arange(n)) ** -steep
    sig = np.exp(mix * rng.standard_normal((leads, N, V, 1)))
    two = d * ((l >= dband[0]) & (l <= dband[1]))                                   # the separation per mode
    if one:
        two = two * (np.cumsum(two > 0) == 1)
    z = rng.standard_normal((leads, N, V, n))
    err = amp * (bias + f * sig * (z + 0.5 * two * rng.choice([-1.0, 1.0], (leads, N, V, n))))
    s_mem = (sig if known else np.ones_like(sig))[:, :, None]                       # what the members know
    e = amp * np.sqrt(1 + 0.25 * two ** 2) * s_mem * rng.standard_normal((leads, N, P, V, n))
    G.save(path, err, np.concatenate([e, -e], 2), dict(vars=list(G.VARS), seed=seed))
    return dict(np.load(path))


def cells(out, band):
    """The cells (dicts) of a band of all variables and leads."""
    bi = [G.label(b) for b in out["meta"]["bands"]].index(band)
    return [row[bi] for v in out["meta"]["vars"] for row in out[v]]


def share(cs, verdict):
    return float(np.mean([c["verdict"].startswith(verdict) for c in cs]))


def rate(cs, k, test=lambda fl: fl != 0):
    """The share of the cells whose flag of the statistic k passes ``test``."""
    return float(np.mean([test(c[k][4]) for c in cs]))


def med(cs, k):
    return float(np.median([c[k][0] for c in cs]))


def test_modes_ranks_and_the_t_null_of_antithetic_pairs():
    l, m, part = G.modes(3)
    assert l.size == 16 and [int((l == k).sum()) for k in range(4)] == [1, 3, 5, 7]
    assert l[:4].tolist() == [0, 1, 1, 1] and m[:4].tolist() == [0, 0, 1, 1] and part[:4].tolist() == [0, 0, 0, 1]
    # the rank of the truth among P antithetic pairs, by sorting: the middle rank twice as likely as the others
    rng = np.random.default_rng(0)
    e, x = rng.standard_normal((200000, P)), rng.standard_normal(200000)
    rank = (np.concatenate([e, -e], 1) < x[:, None]).sum(1)
    freq = np.bincount(rank, minlength=2 * P + 1) / rank.size
    assert np.allclose(G.rank_null(P), [0.1] * 4 + [0.2] + [0.1] * 4) and np.abs(freq - G.rank_null(P)).max() < 0.004
    st, hist = G.band_stats(x[None, :, None], e[None, :, :, None])
    assert np.allclose(hist[0], freq)                                               # the statistic's own ranks
    # z_ens is Student-t with P d.o.f.: var P / (P - 2) = 2, 11.6 % beyond 2, 4.0 % beyond 3 (not 1, 4.6 %, 0.3 %)
    assert abs(float(st["gt2"][0]) - 0.1161) < 0.003 and abs(float(st["gt3"][0]) - 0.0399) < 0.002
    assert abs(float(st["f"][0]) - 1.0) < 0.01 and abs(float(st["kurt_mem"][0]) + 6 / (P + 2)) < 0.01
    # the rms ratio: sqrt(sum err^2 / sum S^2) per mode, the mean of the modes' ratios (of the variances)
    assert abs(float(st["rms"][0]) - 1.0) < 0.01
    E2, e2 = rng.standard_normal((5, 3)) * [1.0, 10.0, 100.0], rng.standard_normal((5, P, 3)) * [1.0, 10.0, 100.0]
    want = math.sqrt(np.mean((E2 ** 2).sum(0) / (e2 ** 2).mean(1).sum(0)))
    assert abs(float(G.band_stats(E2, e2)[0]["rms"]) - want) < 1e-12
    nl = G.null(N, P, 12, REPS, PREPS)
    q = nl["q"]
    assert abs(q["var_ens"][0] - 2.0) < 0.1 and abs(q["gt2"][0] - 0.1161) < 0.003 and abs(q["gt3"][0] - 0.0399) < 0.002
    assert abs(q["f"][0] - 1) < 0.01 and abs(q["rank_chi2"][0] - 2 * P) < 0.6 and abs(q["kurt_mem"][0] + 1) < 0.01
    # with the same spread every day the rms ratio is the sharper of the two scales (it pools the P pairs of the days)
    assert abs(q["rms"][0] - 1) < 0.01 and q["rms"][2] - q["rms"][1] < q["f"][2] - q["f"][1] < 0.2
    assert abs(q["kurt_mode"][0] + 6 / (N + 1)) < 0.02 and abs(q["bias"][0] - 1.07) < 0.05   # the estimates' biases
    assert all(q[k][3] <= q[k][1] < q[k][0] < q[k][2] <= q[k][4] for k in G.STATS)
    # the scale of a scaled t: 1.3 x t_4
    assert abs(float(G.tscale(1.3 * rng.standard_t(P, 100000), P, -1)[0]) - 1.3) < 0.015
    # the bimodality test: the smallest lower-tail probability of the three statistics against its own null
    assert list(nl["sorted"]) == list(G.BIM) and nl["bim"].shape == (REPS,) and np.all(np.diff(nl["bim"]) >= 0)
    low, mid = {k: np.array([-9.0]) for k in G.BIM}, {k: np.array([q[k][0]]) for k in G.BIM}
    assert 2 / (REPS + 1) <= float(G.bim_p(low, nl)[0][0]) == nl["p_bim_floor"] <= 4 / (REPS + 1)   # the three smallest
    assert float(G.bim_p(mid, nl)[0][0]) > 0.5
    one = dict(mid, kurt_min=np.array([nl["sorted"]["kurt_min"][9]]))               # the 10th smallest of one of them
    p, by = G.bim_p(one, nl)
    assert int(by[0]) == 1 and 10 / (REPS + 1) < float(p[0]) < 3.5 * 11 / (REPS + 1)   # the price of three tests
    assert 0 < 32 * G.ALPHA_BIM < 0.3 and G.ALPHA_BIM > 5 / (REPS + 1)


def test_rms_null_is_the_cells_own():
    """The null of the rms ratio given T = err^2 + sum_p e_p^2: with the same sigma every day it is the band's
    (unconditional) one; with a sigma that changes from day to day it is wider, as the statistic is."""
    rng = np.random.default_rng(1)
    q = np.array(G.null(N, P, 12, REPS, PREPS)["q"]["rms"])
    E, e = rng.standard_normal((300, N, 12)), rng.standard_normal((300, N, P, 12))
    flat = G.rms_null(E, e, REPS, np.random.default_rng(2))
    assert flat.shape == (300, 5) and np.abs(flat.mean(0) - q).max() < 0.015 and flat.std(0).max() < 0.01
    sig = np.exp(0.6 * rng.standard_normal((300, N, 1)))
    own = G.rms_null(sig * E, sig[:, :, None] * e, REPS, np.random.default_rng(2))
    width = lambda t: float((t[:, 2] - t[:, 1]).mean())  # noqa: E731
    assert width(own) > 1.3 * width(flat)
    # the statistic of these calibrated cells against the two nulls: the cell's own holds its level, the band's does not
    s = G.band_stats(sig * E, sig[:, :, None] * e)[0]
    out = lambda t: float(((s["rms"] < t[..., 1]) | (s["rms"] > t[..., 2])).mean())  # noqa: E731
    assert out(own) < 0.11 and out(q) > 0.15


def test_calibrated_gaussian_has_no_flag(tmp_path):
    """2 seeds x 40 leads x 4 variables = 320 samples of each band: every flag is outside its 95 % interval in about
    5 % of them (binomial sd 1.2 %, plus the scatter of the null's own quantiles), the verdict is another one in
    about 5 %, and 'bimodal signal' in 0.5 % (1.6 of 320 expected: at most 8)."""
    outs = [G.analyse(synth(tmp_path / f"c{s}.npz", s, leads=40), REPS, PREPS) for s in (1, 2)]
    assert outs[0]["meta"]["bands"] == [[1, 1], [2, 3]] and outs[0]["meta"]["pairs"] == P and outs[0]["meta"]["ics"] == N
    for band in ("1", "2-3"):
        cs = cells(outs[0], band) + cells(outs[1], band)
        rates = {k: rate(cs, k) for k in G.STATS}
        assert all(r <= 0.14 for r in rates.values()), rates                         # 0.03 to 0.07 with this null
        assert 0.025 <= np.mean(list(rates.values())) <= 0.08, rates
        assert share(cs, "calibrated Gaussian") >= 0.89 and abs(outs[0]["null"][band]["false"] - 0.05) < 0.03
        assert share(cs, "bimodal signal") <= 8 / 320 and outs[0]["null"][band]["false_bimodal"] == G.ALPHA_BIM
        p = np.array([c["p_bim"] for c in cs])                 # uniform (mean 0.49, 12 % below 0.1 with this null)
        assert abs(p.mean() - 0.5) < 0.09 and p.min() >= 2 / (REPS + 1) and (p < 0.1).mean() < 0.2
        c = cs[0]                                                                   # the null next to every value
        assert all(len(c[k]) == 5 and c[k][2] < c[k][1] < c[k][3] for k in G.STATS)
        assert len(c["rank_hist"]) == 2 * P + 1 and abs(sum(c["rank_hist"]) - 1) < 1e-9
        assert all(c["verdict"] == G.VERDICT[c["v"]] for c in cs if c["v"] in (0, 3, 4)) and c["bim_by"] in G.BIM
    lead = outs[0]["leading"]                                                       # the values for a histogram
    assert lead["var"] == "z500" and lead["lead"] == 10 and lead["l"] == [1] * 3 + [2] * 5 + [3] * 7
    u = np.array(lead["u"])
    assert u.shape == (15, N) == np.array(lead["z_ens"]).shape and np.allclose(u.mean(1), 0, atol=1e-6) and np.allclose(u.std(1), 1)


def test_under_dispersion_factor_is_recovered(tmp_path):
    """The error 1.3 x the kernel's spread, 40 samples of each band: the rms ratio and the bulk's scale f are both
    1.3, within 10 % in every sample of l = 4-7 (48 modes: sd 0.02), in 90 % of l = 2-3 (sd 0.04, 0.05) and in the
    mean of l = 1 (3 modes x 48 values: sd 0.09, 0.10; within 10 % in 87 %, 81 %); the verdict names the factors, and
    the shape is not flagged more often than in a calibrated kernel."""
    out = G.analyse(synth(tmp_path / "u.npz", 3, leads=10, lmax=7, f=1.3), REPS, PREPS)
    for band, least in (("1", 0.5), ("2-3", 0.9), ("4-7", 1.0)):
        cs = cells(out, band)
        for k in ("rms", "f"):
            v = np.array([c[k][0] for c in cs])
            assert abs(v.mean() / 1.3 - 1) < 0.05 and (np.abs(v / 1.3 - 1) < 0.1).mean() >= least, (band, k, v)
        if band != "1":
            assert share(cs, "Gaussian, under-dispersed (rms factor 1.") >= 0.8
            assert rate(cs, "rms", lambda fl: fl == 2) >= 0.9 and rate(cs, "f", lambda fl: fl == 2) >= 0.9
            assert rate(cs, "rank_chi2", lambda fl: fl > 0) >= 0.8 and np.mean([c["var_ens"][0] for c in cs]) > 2.8
            c = next(c for c in cs if c["v"] == 1)
            assert c["verdict"] == f"Gaussian, under-dispersed (rms factor {c['rms'][0]:.2f}, bulk {c['f'][0]:.2f})"
        else:
            assert share(cs, "Gaussian, under-dispersed") >= 0.6                    # 0.94: the rms ratio sees it
        assert rate(cs, "kurt_mode") <= 0.25 and rate(cs, "kurt_ens") <= 0.25
    over = G.analyse(synth(tmp_path / "o.npz", 4, leads=10, f=0.7), REPS, PREPS)
    cs = cells(over, "2-3")
    assert share(cs, "over-dispersed (rms factor 0.") >= 0.8 and rate(cs, "rms", lambda fl: fl == -2) >= 0.9
    assert abs(med(cs, "rms") - 0.7) < 0.03 and abs(med(cs, "f") - 0.7) < 0.03


def test_day_to_day_scale_mixture(tmp_path):
    """A spread that changes from day to day (sigma = exp(0.6 Z)): the pooled error has heavy tails (kurt_mode above).
    Members that carry the day's true sigma: nothing is left after dividing by it (kurt_ens, f, and rms against its
    own null inside): calibrated. Members that do not: the tails are beyond the predicted spread."""
    known = G.analyse(synth(tmp_path / "k.npz", 5, leads=10, lmax=7, mix=0.6), REPS, PREPS)
    blind = G.analyse(synth(tmp_path / "b.npz", 5, leads=10, lmax=7, mix=0.6, known=False), REPS, PREPS)
    for band in ("2-3", "4-7"):
        ck, cb = cells(known, band), cells(blind, band)
        assert rate(ck + cb, "kurt_mode", lambda fl: fl == 2) >= 0.9 and med(ck + cb, "kurt_mode") > 1.0   # before: heavy tails
        assert all(rate(ck, k) <= 0.25 for k in ("kurt_ens", "f", "rms"))                        # after: none
        assert share(ck, "calibrated Gaussian") >= 0.75
        assert share(cb, "heavy tails beyond the predicted day-to-day spread") >= (0.9 if band == "4-7" else 0.6)
        assert share(ck + cb, "bimodal signal") <= 0.1


def test_a_right_variance_with_heavy_tails_is_not_over_dispersed(tmp_path):
    """A kernel with the RIGHT variance that does not follow the error's day-to-day sigma = exp(0.5 Z): the bulk of
    z_ens is narrower than the kernel says (f = 0.84), the rms ratio is 1 in the mean (its median 0.97). The verdict
    is not 'over-dispersed (factor 0.84)' (41 % and 26 % of the bands l = 1 and 2-3 when f decided; now the 12 % and
    9 % where the rms ratio of the sample itself is below its interval, never from f alone): it is one of the two
    'heavy tails'. 160 samples of each band."""
    mix = 0.5
    out = G.analyse(synth(tmp_path / "t.npz", 10, leads=40, mix=mix, known=False, f=math.exp(-mix ** 2)), REPS, PREPS)
    for band, most, least in (("1", 0.26, 0.25), ("2-3", 0.22, 0.7)):
        cs = cells(out, band)
        assert 0.9 < med(cs, "rms") < 1.05 and abs(np.mean([c["rms"][0] ** 2 for c in cs]) - 1) < 0.15
        assert abs(med(cs, "f") - 0.84) < 0.04 and rate(cs, "f", lambda fl: fl == -2) >= 0.3
        assert share(cs, "over-dispersed") <= most and share(cs, "Gaussian, under-dispersed") <= 0.15
        assert share(cs, "heavy tails") >= least
        # the dispersion is the rms ratio's verdict: never from the bulk's scale alone
        assert all(c["rms"][4] == -2 for c in cs if c["v"] == 2) and all(c["rms"][4] == 2 for c in cs if c["v"] == 1)
        bulk = [c for c in cs if c["f"][4] == -2 and c["rms"][4] > -2 and c["kurt_ens"][4] < 2 and c["p_bim"] >= G.ALPHA_BIM]
        assert len(bulk) >= 4 and all(c["v"] == 5 for c in bulk) and sum(c["v"] == 5 for c in cs) == len(bulk)
        c = bulk[0]
        assert c["verdict"] == f"heavy tails (bulk narrower than the rms): rms factor {c['rms'][0]:.2f}, bulk {c['f'][0]:.2f}"
    # the same error 1.3 x too large: the rms ratio is the factor of the spread, the bulk's scale is not
    out = G.analyse(synth(tmp_path / "u.npz", 11, leads=40, mix=mix, known=False, f=1.3 * math.exp(-mix ** 2)), REPS, PREPS)
    cs = cells(out, "2-3")
    assert abs(med(cs, "rms") / 1.3 - 1) < 0.05 and med(cs, "f") < 1.17 and share(cs, "over-dispersed") == 0
    assert share(cs, "Gaussian, under-dispersed") + share(cs, "heavy tails") >= 0.9


def test_two_regimes_in_one_band(tmp_path):
    """The modes l = 2-3 a mixture of two Gaussians 3 sigma apart, the spread right: negative kurtosis, bimodal
    signal there and not in l = 1; the one test of the table's cells at leads >= 7; the script's own power estimate."""
    out = G.analyse(synth(tmp_path / "m.npz", 6, leads=10, d=3.0), REPS, PREPS)
    cs = cells(out, "2-3")
    assert share(cs, "bimodal signal") == 1.0 and all(c["kurt_mode"][4] == -2 for c in cs)
    assert np.mean([c["bim_by"] == "kurt_mode" for c in cs]) >= 0.9
    floor = out["null"]["2-3"]["p_bim_floor"]
    assert all(c["p_bim"] < G.ALPHA_BIM for c in cs) and np.median([c["p_bim"] for c in cs]) == floor <= 4 / (REPS + 1)
    assert abs(np.mean([c["kurt_mode"][0] for c in cs]) + 2 * 81 / 169) < 0.1               # -2 d^4 / (4 + d^2)^2
    assert abs(med(cs, "rms") - 1) < 0.03 and med(cs, "f") > 1.05                            # a wide bulk, the rms right
    assert share(cells(out, "1"), "calibrated Gaussian") >= 0.75
    t = out["summary"]["regimes"]                                                            # leads 7, 10 x 4 x 2 bands
    assert t["cells"] == 16 and t["p_min"] == floor and t["p"] == 16 * floor < 0.05 and t["where"][2:] == ["2-3", "kurt_mode"]
    assert t["p_floor"] == 16 * min(floor, out["null"]["1"]["p_bim_floor"])
    assert out["summary"]["late"]["bimodal"] >= 8 and out["summary"]["table"]["verdicts"]["bimodal signal"] >= 16
    pw = out["null"]["2-3"]["power"]
    assert pw["all"][2] > 0.9 and 0.25 < pw["all"][1] < 0.85 and pw["all"][0] < 0.15 and pw["one"][2] < 0.3
    assert out["null"]["1"]["power"]["all"][2] > 0.4


def test_regimes_in_one_mode_and_under_a_changing_spread(tmp_path):
    """The two other statistics of the bimodality test. kurt_min: regimes 6 sigma apart in ONE mode of l = 2-3 (the
    pooled kurt_mode moves by -1.6 / 12 only). kurt_ens: regimes 3 sigma apart in every mode of l = 4-7 under a spread
    that changes from day to day and that the kernel follows: the pooled kurtosis is positive, the regimes show
    after dividing by the day's predicted spread."""
    out = G.analyse(synth(tmp_path / "1.npz", 12, leads=10, d=6.0, one=True), REPS, PREPS)
    cs = cells(out, "2-3")
    assert share(cs, "bimodal signal") >= 0.7 and np.mean([c["bim_by"] == "kurt_min" for c in cs]) >= 0.8
    assert np.median([c["kurt_min"][0] for c in cs]) < -1.4 and rate(cs, "kurt_mode", lambda fl: fl == -2) <= 0.5
    out = G.analyse(synth(tmp_path / "2.npz", 13, leads=10, lmax=7, d=3.0, dband=(4, 7), mix=0.5), REPS, PREPS)
    cs = cells(out, "4-7")
    assert share(cs, "bimodal signal") >= 0.8 and np.mean([c["bim_by"] == "kurt_ens" for c in cs if c["v"] == 4]) >= 0.9
    assert med(cs, "kurt_mode") > 0.3 and rate(cs, "kurt_mode", lambda fl: fl < 0) <= 0.1      # no regimes in the pooled error
    assert rate(cs, "kurt_ens", lambda fl: fl == -2) >= 0.8
    assert share(cells(out, "2-3"), "bimodal signal") <= 0.1


def test_the_table_says_what_chance_gives(tmp_path, capsys):
    """20 calibrated files of 10 leads, 4 variables and 3 bands. 'bimodal signal' is given to 0.5 % of ALL cells
    (2400: 12 expected, at most 30; 3 % = 72 with three one-sided 1 % tests); the one test of regimes at leads >= 7,
    l <= 7 (24 cells, Bonferroni) is below 0.05 in at most 5 % of the files (at most 5 of 20); the table's header
    gives the expected counts of a calibrated kernel."""
    outs = [G.analyse(synth(tmp_path / f"{s}.npz", 100 + s, leads=10, lmax=7), REPS, PREPS) for s in range(20)]
    bim = sum(c["v"] == 4 for o in outs for b in ("1", "2-3", "4-7") for c in cells(o, b))
    assert bim <= 30
    ps = np.array([o["summary"]["regimes"]["p"] for o in outs])
    assert (ps < 0.05).sum() <= 5 and ps.min() >= 48 / (REPS + 1) and ps.max() <= 1.0
    sm = outs[0]["summary"]
    assert sm["table"]["cells"] == 48 and sm["late"]["cells"] == 24 and sm["regimes"]["cells"] == 24
    assert abs(sm["table"]["expected_bimodal"] - 0.24) < 1e-12 and abs(sm["late"]["expected_bimodal"] - 0.12) < 1e-12
    assert abs(sm["table"]["expected_other"] - 16 * sum(outs[0]["null"][b]["false"] for b in ("1", "2-3", "4-7"))) < 1e-9
    assert 1.0 < sm["table"]["expected_other"] < 4.0 and sum(sm["table"]["verdicts"].values()) == 48
    t = sm["regimes"]
    c = cells(outs[0], t["where"][2])[list(G.VARS).index(t["where"][0]) * 10 + t["where"][1] - 1]
    assert t["p_min"] == c["p_bim"] and t["where"][3] == c["bim_by"] and t["where"][1] in (7, 10)
    assert t["p"] == min(1.0, 24 * t["p_min"]) and 48 / (REPS + 1) <= t["p_floor"] <= 96 / (REPS + 1)
    G.table(outs[0])
    txt = capsys.readouterr().out
    assert (f"in the table (48 cells) {sm['table']['expected_other']:.1f} verdicts other than 'calibrated Gaussian', "
            "0.24 of them 'bimodal signal'") in txt
    assert "in leads >= 7 (24 cells)" in txt and "0.12 of them 'bimodal signal'" in txt
    assert f"any regimes at leads >= 7, l <= 7 (24 cells): smallest p_bim {t['p_min']:.4f}" in txt and f"p = {t['p']:.3f}" in txt
    assert txt.count("z500  lead ") == 12 and txt.count("p_bim") >= 4
    # all four bands (l <= 15, few null draws): the table's 64 cells, 32 at leads >= 7, the test of regimes over the
    # 24 with l <= 7, whatever the band l = 8-15 shows (here regimes in every mode of it)
    full = G.analyse(synth(tmp_path / "f.npz", 99, leads=10, lmax=15, d=3.0, dband=(8, 15)), 200, 5)
    sm = full["summary"]
    assert (sm["table"]["cells"], sm["late"]["cells"], sm["regimes"]["cells"]) == (64, 32, 24)
    assert abs(sm["table"]["expected_bimodal"] - 0.32) < 1e-12 and abs(sm["late"]["expected_bimodal"] - 0.16) < 1e-12
    assert all(c["p_bim"] == full["null"]["8-15"]["p_bim_floor"] and c["kurt_mode"][4] == -2 for c in cells(full, "8-15"))
    assert sm["regimes"]["where"][2] in ("1", "2-3", "4-7") and sm["regimes"]["p"] == min(1.0, 24 * sm["regimes"]["p_min"])


def test_a_steep_spectrum_inside_a_band_is_no_flag(tmp_path):
    """Modes of different variance in a band (a factor 30 in l = 2-3), every one Gaussian and calibrated: pooled
    without the per-mode standardization they are a scale mixture (excess kurtosis > 1, the first version's
    kurt_pool); the statistics do not see the spectrum at all."""
    flat = G.analyse(synth(tmp_path / "f.npz", 7, leads=10), REPS, PREPS)
    d = synth(tmp_path / "s.npz", 7, leads=10, steep=1.5)
    steep = G.analyse(d, REPS, PREPS)
    sel = (d["l"] >= 2) & (d["l"] <= 3)
    E = d["err"][..., sel].astype(np.float64)
    assert E.std((0, 1, 2)).max() / E.std((0, 1, 2)).min() > 5
    pooled = ((E / np.sqrt((E ** 2).mean((1, 3), keepdims=True))) ** 4).mean((1, 3)) - 3     # per lead and variable
    assert pooled.mean() > 1.0
    for band in ("1", "2-3"):
        cf, cst = cells(flat, band), cells(steep, band)
        for a, b in zip(cf, cst):
            assert a["verdict"] == b["verdict"] and all(a[k][4] == b[k][4] and abs(a[k][0] - b[k][0]) < 1e-3 for k in G.STATS)
            assert a["p_bim"] == b["p_bim"] and abs(a["rms"][3] - b["rms"][3]) < 1e-3
        assert share(cst, "calibrated Gaussian") >= 0.75 and rate(cst, "kurt_mode") <= 0.25


def test_a_bias_is_not_read_as_regimes(tmp_path):
    """A mean error of one sigma in every mode: Gaussian about its mean (the kurtosis about 0 would be -0.5, as for
    regimes 2 sigma apart). It is error, so the kernel is under-dispersed by sqrt(2), and ``bias`` shows why."""
    out = G.analyse(synth(tmp_path / "b.npz", 8, leads=10, bias=1.0), REPS, PREPS)
    cs = cells(out, "2-3")
    assert all(c["bias"][4] == 2 and c["bias"][0] > 20 for c in cs) and share(cs, "Gaussian, under-dispersed") >= 0.75
    assert share(cs, "bimodal signal") <= 0.1 and rate(cs, "kurt_mode") <= 0.25
    assert abs(med(cs, "f") - math.sqrt(2)) < 0.12 and abs(med(cs, "rms") - math.sqrt(2)) < 0.06


def test_members_in_another_layout_are_refused(tmp_path):
    d = synth(tmp_path / "x.npz", 9, leads=1)
    d["mem"] = np.abs(d["mem"])
    with pytest.raises(SystemExit):
        G.analyse(d, 50, 10)


@pytest.mark.parametrize("noise", ["sphere", "balanced"])
def test_rollout_stage_is_the_kernels_rollout(mcore, tmp_path, monkeypatch, capsys, noise):
    """Stage 1 on the toy core, 7 initial conditions in batches of 3, 3, 1 and 5 leads: EVERY value of the .npz
    (all leads, batches, members, variables and modes) is the coefficient of EntropyDeviationKernel.rollout from the
    same initial conditions, history, tisr days, truth days and generator (seed 7, carried over the batches)."""
    import run_fpk as R
    from fpk.devkernel import DevConfig, EntropyDeviationKernel
    from fpk.sphere import SHT
    torch.manual_seed(3)
    Dk = EntropyDeviationKernel(mcore, DevConfig(pairs=3, hidden=8, blocks=(1,), smooth=3, noise=noise, band_scale=True))
    with torch.no_grad():                    # away from the start values (zero head, r = 0, g_band = 0)
        for n, p in Dk.named_parameters():
            if not n.startswith("core."):
                p.add_(0.1 * torch.randn(p.shape))
    torch.save(dict(state={k: v for k, v in Dk.state_dict().items() if not k.startswith("core.")},
                    cfg=dict(Dk.cfg.__dict__), core_tag="c"), tmp_path / "devk.pt")
    rng = np.random.default_rng(0)
    lat, L, nic, nb = np.linspace(-78.75, 78.75, H), 5, 7, 3
    year = rng.standard_normal((40, 67, H, W)).astype(np.float32)
    monkeypatch.setattr(R, "load_core", lambda a: (None, SimpleNamespace(lat=lat), mcore))
    monkeypatch.setattr(R.D, "load", lambda pa, years: {y: year for y in years})
    npz, js = str(tmp_path / "ng.npz"), str(tmp_path / "ng.json")
    G.main(["--out", str(tmp_path), "--core-tag", "c", "--dev-tag", "devk", "--ics", str(nic), "--batch", str(nb),
            "--leads", str(L), "--npz", npz, "--json", js])
    assert f"--from-npz {npz}" in capsys.readouterr().out and not os.path.exists(js)
    d = np.load(npz)
    meta = json.loads(str(d["meta"]))
    l, m, part = G.modes(H - 1)
    assert d["err"].shape == (L, nic, 4, H * H) and d["mem"].shape == (L, nic, 6, 4, H * H) and d["err"].dtype == np.float32
    assert all(np.array_equal(d[k], v) for k, v in (("l", l), ("m", m), ("part", part)))
    assert meta["vars"] == list(G.VARS) and meta["dev_tag"] == "devk"
    assert meta["ic_days"] == np.linspace(8, 40 - L - 1, nic).astype(int).tolist() and len(set(meta["ic_days"])) == nic
    assert np.array_equal(d["mem"][:, :, :3], -d["mem"][:, :, 3:]) and np.abs(d["mem"]).min() > 0
    # the reference: the kernel's own rollout, batch by batch with one generator
    dyn, names = np.asarray(R.DYN), [R.contract.CHANNEL_NAMES[c] for c in R.DYN]
    ci, days = [names.index(v) for v in G.VARS], np.asarray(meta["ic_days"])
    assert names[ci[0]] == "z500" and len(set(ci)) == 4
    D2 = R.load_devk(SimpleNamespace(out=str(tmp_path), core_tag="c", device="cpu"), mcore, "devk")
    S, g, mem_len = SHT(lat), torch.Generator().manual_seed(7), int(mcore.mem)

    def coef(x):                             # (..., H, W) -> (..., n): Re / Im c_lm in the order of ``modes``
        c = S.analysis(x)
        return torch.stack([(c[..., a, b].real if q == 0 else c[..., a, b].imag) for a, b, q in zip(l, m, part)], -1)
    err, mem = [], []
    for i in range(0, nic, nb):
        j = days[i:i + nb]
        tis = torch.from_numpy(np.stack([year[k + 1:k + 1 + L, R.TISR] for k in j]))                # days k+1 .. k+L
        hist = torch.from_numpy(np.stack([year[k - mem_len:k][:, dyn] for k in j]))                 # days k-W .. k-1
        truth = torch.from_numpy(np.stack([year[k + 1:k + 1 + L][:, dyn] for k in j]))              # (B, L, 66, H, W)
        ens, _ = D2.rollout(torch.from_numpy(year[j][:, dyn]), torch.tensor(j), tis, L, hist=hist, gen=g)
        xbar = ens.mean(1)                                                                           # (B, L, 66, H, W)
        err.append(coef(((truth - xbar) / D2.s)[:, :, ci]).transpose(0, 1))                          # (L, B, V, n)
        mem.append(coef(((ens - xbar[:, None]) / D2.s)[:, :, :, ci]).permute(2, 0, 1, 3, 4))         # (L, B, 2 P, V, n)
    err, mem = torch.cat(err, 1).numpy(), torch.cat(mem, 1).numpy()
    assert np.abs(d["err"] - err).max() < 1e-4 and np.abs(d["mem"] - mem).max() < 1e-4
    # the comparison has teeth: the size of the values, and the change from lead to lead, batch to batch, member to member
    assert np.abs(err).mean() > 0.05 and np.abs(mem).mean() > 1e-3
    assert np.abs(err[1:] - err[:-1]).mean() > 0.05 and np.abs(mem[1:] - mem[:-1]).mean() > 1e-3
    assert np.abs(err[:, nb:2 * nb] - err[:, :nb]).mean() > 0.05 and np.abs(mem[:, :, 1] - mem[:, :, 0]).mean() > 1e-3
    if noise != "sphere":
        return
    # stage 2 in its own process: the table, the json, and no torch / run_fpk / fpk in it
    code = ("import sys, runpy; sys.argv = ['diag_nongauss.py', '--from-npz', sys.argv[1], '--json', sys.argv[2], "
            "'--null-reps', '100', '--power-reps', '10']; "
            f"runpy.run_path(r'{os.path.join(ROOT, 'scripts', 'diag_nongauss.py')}', run_name='__main__'); "
            "assert not {'torch', 'run_fpk', 'fpk'} & set(sys.modules)")
    r = subprocess.run([sys.executable, "-c", code, npz, js], capture_output=True, text=True, cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr
    out = json.load(open(js))
    assert out["meta"]["bands"] == [[1, 1], [2, 3], [4, 7]] and out["meta"]["ics"] == nic and out["meta"]["pairs"] == 3
    assert len(out["z500"]) == L and len(out["z500"][0]) == 3 and out["leading"]["lead"] == L
    assert all(math.isfinite(c[k][0]) for v in G.VARS for row in out[v] for c in row for k in G.STATS)
    assert all(1 / 101 <= c["p_bim"] <= 1 and c["v"] != 4 for v in G.VARS for row in out[v] for c in row)   # 100 null draws
    assert "regimes" not in out["summary"] and out["summary"]["late"]["cells"] == 0 and "any regimes" not in r.stdout
    assert "band l = 4-7: 48 modes" in r.stdout and r.stdout.count("z500  lead  1") == 3 and "lead  7" not in r.stdout
    assert r.stdout.count("z500  lead  5") == 3 and "in the table (24 cells)" in r.stdout
