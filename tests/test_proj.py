"""The invariant-measure condition in the deviation kernel (fpk.devkernel ``proj`` / ``read``, 감자 2026-11-24):
the forecast mean is read out as climatology + g (.) (xbar - climatology), g = the strength of the condition per lead,
channel and band, learned by the log score; the chain (the core's trajectory, the deviations) is not changed."""
import json
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.dirname(__file__))
from fpk.env import TISR_DYN  # noqa: E402
from test_eqkernel import H, W, _core, _kernel  # noqa: E402
from test_fpk import mcore  # noqa: E402,F401

KEPT = [i for i in range(66) if i != TISR_DYN]


def _clim(seed=0):
    return (0.5 * np.random.default_rng(seed).standard_normal((366, 66, H, W))).astype(np.float32)


def _anomaly(Dk, B=2, seed=1):
    """A field the transform holds exactly: the synthesis of random coefficients of every degree."""
    g = torch.Generator().manual_seed(seed)
    L = Dk.sht.L
    c = torch.complex(torch.randn(B, 66, L, L, generator=g), torch.randn(B, 66, L, L, generator=g))
    return Dk.sht.synthesis(c * (torch.arange(L)[:, None] >= torch.arange(L)[None, :]))


def test_the_readout_starts_as_the_chains_mean_and_the_rollout_as_the_kernel_without_it():
    core = _core(False)
    Pl, Pr = _kernel(core), _kernel(core, proj=True)
    assert float(Pr.g_mean.abs().sum()) == 0.0 and tuple(Pr.g_mean.shape) == (10, 66, Pr.n_bands)
    x = torch.randn(2, 66, H, W)
    assert Pl.read(x, torch.tensor([3, 4]), 0) is x                      # no proj: the chain's mean itself
    with pytest.raises(RuntimeError):
        Pr.read(x, torch.tensor([3, 4]), 0)                              # the climatology is not there
    Pr.set_clim(_clim())
    assert torch.equal(Pr.read(x, torch.tensor([3, 4]), 0), x)
    xd, doy, tis = torch.randn(2, 66, H, W), torch.tensor([5, 6]), torch.zeros(2, 3, H, W)
    e0, h0 = Pl.rollout(xd, doy, tis, 3, gen=torch.Generator().manual_seed(2))
    e1, h1 = Pr.rollout(xd, doy, tis, 3, gen=torch.Generator().manual_seed(2))
    assert torch.equal(e0, e1) and torch.equal(h0, h1)
    assert Pl.proj_report() is None
    assert Pr.proj_report() == dict(mean={k: [1.0] * Pr.n_bands for k in ("day1", "day5", "day10")}, spread=[1.0] * 10)
    assert torch.equal(Pl.spread(3), torch.ones(1, 66, 1, 1)) and torch.equal(Pr.spread(3), torch.ones(1, 66, 1, 1))


def test_the_readout_scales_the_anomaly_of_one_band_and_one_lead():
    core = _core(False)
    Dk = _kernel(core, proj=True)
    cl = _clim()
    Dk.set_clim(cl)
    anom = _anomaly(Dk)
    doy = torch.tensor([10, 364])
    c = torch.from_numpy(cl[[10, 364]])
    x = c + anom
    with torch.no_grad():
        Dk.g_mean[2, :, 1] = math.log(0.3)                               # day 3, the band l = 1
    y = Dk.read(x, doy, 2)
    band = Dk.sht.synthesis(Dk.sht.analysis(anom) * (Dk.band_of == 1).float()[:, None])
    assert torch.allclose((y - c)[:, KEPT], (anom - 0.7 * band)[:, KEPT], atol=1e-5)
    assert float(band[:, KEPT].abs().max()) > 0.1
    assert torch.equal(y[:, TISR_DYN], x[:, TISR_DYN])                   # tisr is a known forcing
    assert torch.equal(Dk.read(x, doy, 0), x) and torch.equal(Dk.read(x, doy, 9), x)   # the other leads: untouched
    # the climatology of the forecast's own day, the last row beyond the table (day 366 of a leap year's forecast)
    far = Dk.read(torch.from_numpy(cl[[365, 365]]) + anom, torch.tensor([365, 900]), 2)
    assert torch.allclose((far - torch.from_numpy(cl[[365, 365]]))[:, KEPT], (anom - 0.7 * band)[:, KEPT], atol=1e-5)
    # a lead beyond day 10 reads the strength of day 10
    with torch.no_grad():
        Dk.g_mean[9] = math.log(0.5)
    assert torch.allclose((Dk.read(x, doy, 14) - c)[:, KEPT], 0.5 * anom[:, KEPT], atol=1e-5)


def test_the_rollouts_members_are_around_the_read_out_mean_and_the_chain_is_the_same():
    core = _core(False)
    Pl, Pr = _kernel(core), _kernel(core, proj=True)
    cl = _clim()
    Pr.set_clim(cl)
    with torch.no_grad():
        Pr.g_mean.fill_(math.log(0.5))
    xd, doy, tis = torch.randn(2, 66, H, W), torch.tensor([5, 6]), torch.zeros(2, 3, H, W)
    e0, h0 = Pl.rollout(xd, doy, tis, 3, gen=torch.Generator().manual_seed(2))
    e1, h1 = Pr.rollout(xd, doy, tis, 3, gen=torch.Generator().manual_seed(2))
    m0, m1 = e0.mean(1), e1.mean(1)                                       # (B, L, 66, H, W)
    assert torch.allclose(e0 - m0[:, None], e1 - m1[:, None], atol=1e-5) and torch.equal(h0, h1)   # the deviations, H
    with torch.no_grad():                                                 # the read-out spread: x 0.8 on day 2
        Pr.g_spread[1] = math.log(0.8)
    e2, h2 = Pr.rollout(xd, doy, tis, 3, gen=torch.Generator().manual_seed(2))
    d1, d2 = e1 - m1[:, None], e2 - e2.mean(1)[:, None]
    assert torch.allclose(e2.mean(1), m1, atol=1e-5)
    assert torch.allclose(d2[:, :, 1, KEPT], 0.8 * d1[:, :, 1, KEPT], atol=1e-5) and torch.allclose(d2[:, :, 1, TISR_DYN], d1[:, :, 1, TISR_DYN])
    assert torch.allclose(d2[:, :, [0, 2]], d1[:, :, [0, 2]], atol=1e-6)    # the chain's day 3 is the same
    assert torch.allclose(h2[:, 1, KEPT], h1[:, 1, KEPT] + math.log(0.8), atol=1e-5) and torch.equal(h2[:, [0, 2]], h1[:, [0, 2]])
    for L in range(3):                                                    # day L + 1 of the IC's doy
        c = torch.from_numpy(np.stack([cl[5 + L + 1], cl[6 + L + 1]]))
        want = m0[:, L] - 0.5 * (m0[:, L] - c) * Pr.keep                   # every band and the grid's rest: x 0.5
        assert torch.allclose(m1[:, L], want, atol=1e-5)
    assert float((m1 - m0)[:, :, KEPT].abs().max()) > 0.05


def test_the_strength_learns_the_projection_from_the_log_score():
    """The truth's anomaly is half the forecast's: the log score's minimum is the least-squares strength 0.5."""
    core = _core(False)
    Dk = _kernel(core, proj=True)
    cl = _clim()
    Dk.set_clim(cl)
    anom = _anomaly(Dk, B=4)
    doy = torch.tensor([1, 2, 3, 4])
    c = torch.from_numpy(cl[[1, 2, 3, 4]])
    x, truth = c + anom, c + 0.5 * anom
    w = torch.ones(1, 1, H, 1)
    H0 = torch.zeros(4, 66, H, W)
    opt = torch.optim.Adam([Dk.g_mean], lr=0.05)
    first = None
    for _ in range(300):
        loss = Dk.ign(H0, Dk.read(x, doy, 0), truth, w)[0]
        first = float(loss) if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    g = torch.exp(Dk.g_mean.detach())
    assert float(loss) < 0.3 * first
    assert float((g[0][KEPT] - 0.5).abs().max()) < 0.05
    assert float((g[1:] - 1.0).abs().max()) == 0.0                         # the other leads saw no gradient
    assert float(Dk.g_mean.grad[0, TISR_DYN].abs().max()) == 0.0            # tisr is not read out


def test_devk_cli_proj_trains_the_strength_starts_from_a_fit_and_reloads(mcore, tmp_path, monkeypatch, capsys):
    import run_fpk as R
    rng = np.random.default_rng(0)
    clim = (0.3 * rng.standard_normal((366, 67, H, W))).astype(np.float32)
    cl = SimpleNamespace(lat=np.linspace(-78.75, 78.75, H), std=np.ones(67), mean=np.zeros(67), clim=clim)
    data = rng.standard_normal((12, 67, H, W)).astype(np.float32)
    monkeypatch.setattr(R, "load_core", lambda a: (None, cl, mcore))
    monkeypatch.setattr(R.D, "load", lambda pa, years: {y: data for y in years})
    monkeypatch.setattr(R.D, "climate", lambda pa: cl)
    monkeypatch.setattr(R, "pargs", lambda a, extra=None: None)
    base = ["devk", "--out", str(tmp_path), "--core-tag", "c", "--days", "2", "--iters", "2", "--batch", "2", "--pairs", "2",
            "--val-batches", "1", "--hidden", "8", "--smooth", "3", "--log-every", "1", "--band-scale"]
    R.main(base + ["--tag", "a"])
    a = torch.load(tmp_path / "a.pt", weights_only=False)
    assert a["cfg"]["proj"] is False and "g_mean" not in a["state"]
    capsys.readouterr()
    R.main(base + ["--tag", "b", "--init", "a", "--proj", "--proj-lr", "500", "--fresh-best"])
    logs = [json.loads(q) for q in capsys.readouterr().out.strip().splitlines() if q.startswith("{")]
    assert sorted(logs[-1]["proj"]["mean"]) == ["day1", "day10", "day5"] and all(math.isfinite(v) for v in logs[-1]["val_ign"])
    b = torch.load(tmp_path / "b.pt", weights_only=False)
    assert b["cfg"]["proj"] is True
    for k in ("g_mean", "g_spread"):                                       # days 1, 2 were trained, the rest not
        assert float(b["state"][k][:2].abs().max()) > 1e-3 and float(b["state"][k][2:].abs().max()) == 0.0
    ns = SimpleNamespace(out=str(tmp_path), core_tag="c", device="cpu")
    Dk = R.load_devk(ns, mcore, "b")
    assert Dk.cfg.proj and Dk._clim.shape == (366, 66, H, W) and Dk._clim.dtype == np.float32
    assert np.allclose(Dk._clim, clim[:, np.asarray(R.DYN)] / (1.0 + R.rkt().EPS), atol=1e-6)
    xd, doy, tis = torch.randn(2, 66, H, W), torch.tensor([5, 6]), torch.zeros(2, 2, H, W)
    hist = torch.randn(2, int(mcore.mem), 66, H, W)
    ens, _ = Dk.rollout(xd, doy, tis, 2, hist=hist, gen=torch.Generator().manual_seed(2))
    assert torch.isfinite(ens).all()
    with pytest.raises(SystemExit):
        R.no_proj_flow(Dk, "b")                                           # a flow on a read-out kernel: not wired yet
    R.no_proj_flow(R.load_devk(ns, mcore, "a"), "a")
    # --proj-init: the least-squares strength of this core from a stats file, the band scales follow the error left
    nb = Dk.n_bands
    S = np.broadcast_to(np.kron(np.diag([7.0, 1.0]), np.eye(nb)), (3, 10, 66, 2 * nb, 2 * nb)).copy()   # cores x, c
    np.savez(tmp_path / "proj.npz", S=S, oo=np.full((3, 10, 66), 0.25 * nb + 1.0), cores=np.array(["x", "c"]), nb=nb,
             c=np.concatenate([np.full((3, 10, 66, nb), 0.1), np.full((3, 10, 66, nb), 0.5)], -1))
    with pytest.raises(SystemExit):
        R.proj_fit(str(tmp_path / "proj.npz"), "y")                       # not a core of the file
    g, kap = R.proj_fit(str(tmp_path / "proj.npz"), "c")
    # raw mse = nb - nb + oo = 0.25 nb + 1, with g = 0.5: 0.25 nb - 0.5 nb + oo = 1
    assert np.allclose(g[:, KEPT], 0.5) and np.allclose(g[:, TISR_DYN], 1.0)
    assert np.allclose(kap[:, KEPT], math.sqrt(1.0 / (0.25 * nb + 1.0))) and np.allclose(kap[:, TISR_DYN], 1.0)
    capsys.readouterr()
    R.main(base + ["--tag", "f", "--init", "a", "--proj", "--proj-init", str(tmp_path / "proj.npz"), "--lr", "0", "--fresh-best"])
    logs = [json.loads(q) for q in capsys.readouterr().out.strip().splitlines() if q.startswith("{")]
    assert any("proj_init" in q for q in logs)
    f = torch.load(tmp_path / "f.pt", weights_only=False)
    assert torch.allclose(f["state"]["g_mean"][:, KEPT], torch.tensor(math.log(0.5))) and float(f["state"]["g_mean"][:, TISR_DYN].abs().max()) == 0.0
    assert torch.allclose(f["state"]["g_spread"][:, KEPT], torch.tensor(0.5 * math.log(1.0 / (0.25 * nb + 1.0))), atol=1e-6)
    assert float(f["state"]["g_spread"][:, TISR_DYN].abs().max()) == 0.0
    assert torch.equal(f["state"]["g_band"], a["state"]["g_band"])          # the chain's own band scales: untouched
    with pytest.raises(SystemExit):
        R.main(base + ["--tag", "g", "--proj-init", str(tmp_path / "proj.npz")])   # the start of --proj only
    # projk: the same start written onto a fitted kernel without training; the chain's parameters untouched
    pk = ["projk", "--out", str(tmp_path), "--core-tag", "c", "--dev-tag", "a", "--proj-init", str(tmp_path / "proj.npz")]
    R.main(pk + ["--tag", "a_p"])
    p = torch.load(tmp_path / "a_p.pt", weights_only=False)
    assert p["cfg"]["proj"] is True and p["proj_of"] == "a" and {k: v for k, v in p["cfg"].items() if k != "proj"} == {
        k: v for k, v in a["cfg"].items() if k != "proj"}
    assert sorted(p["state"]) == sorted(list(a["state"]) + ["g_mean", "g_spread"])
    assert all(torch.equal(p["state"][k], a["state"][k]) for k in a["state"])
    assert torch.equal(p["state"]["g_mean"], f["state"]["g_mean"]) and torch.equal(p["state"]["g_spread"], f["state"]["g_spread"])
    Dp = R.load_devk(ns, mcore, "a_p")
    e2, _ = Dp.rollout(xd, doy, tis, 2, hist=hist, gen=torch.Generator().manual_seed(2))
    e0, _ = R.load_devk(ns, mcore, "a").rollout(xd, doy, tis, 2, hist=hist, gen=torch.Generator().manual_seed(2))
    d0, d2 = e0 - e0.mean(1, keepdim=True), e2 - e2.mean(1, keepdim=True)
    kap0 = math.sqrt(1.0 / (0.25 * nb + 1.0))
    assert torch.allclose(d2[:, :, :, KEPT], kap0 * d0[:, :, :, KEPT], atol=1e-5)
    for bad in (["--tag", "a_p"], ["--tag", "a"]):                        # never over an existing checkpoint
        with pytest.raises(SystemExit):
            R.main(pk + bad)
    with pytest.raises(SystemExit):                                       # not twice
        R.main(pk[:6] + ["--dev-tag", "a_p", "--proj-init", str(tmp_path / "proj.npz"), "--tag", "a_pp"])


def test_the_readout_on_the_pole_grid_of_the_plus_version(tmp_path):
    """On a grid with the poles (as 121 x 240): every band and the grid's rest read out at strength g (all 0.5 here:
    clim + 0.5 anomaly exactly), the spread readout, and the projection file's band count is the kernel's."""
    from test_plus import HP, WP, _spec
    from fpk.devkernel import DevConfig, EntropyDeviationKernel
    from fpk.sphere import bands
    from pgcfm.core import static_features
    from pgcfm.kai_tend import make_core
    from fpk.env import contract
    torch.manual_seed(1)
    lat = np.linspace(-90, 90, HP)
    lon = np.linspace(0, 360, WP, endpoint=False)
    contract.register_latitudes(lat)
    st = static_features(lat, lon, np.random.rand(HP, WP).astype(np.float32), np.random.rand(HP, WP).astype(np.float32))
    core = make_core(st, np.full(67, 0.3), np.zeros(67), np.ones(67), lat, hidden=8, blocks=(1,), norm="gn").eval()
    core.pm = None
    Dk = EntropyDeviationKernel(core, DevConfig(pairs=2, hidden=8, blocks=(1,), smooth=5, noise="sphere", band_scale=True,
                                                e0_spec=_spec(tmp_path, HP - 2), noise_cells=5, proj=True))
    assert Dk.sht.L == HP - 2 and Dk.n_bands == bands(HP - 2)[1] and tuple(Dk.g_mean.shape) == (10, 66, Dk.n_bands)
    cl = (0.5 * np.random.default_rng(0).standard_normal((366, 66, HP, WP))).astype(np.float32)
    Dk.set_clim(cl)
    x = torch.randn(2, 66, HP, WP)
    doy = torch.tensor([3, 200])
    with torch.no_grad():
        Dk.g_mean.fill_(math.log(0.5))
        Dk.g_spread.fill_(math.log(0.9))
    c = torch.from_numpy(cl[[3, 200]])
    y = Dk.read(x, doy, 4)
    assert torch.allclose(y[:, KEPT], (c + 0.5 * (x - c))[:, KEPT], atol=1e-5) and torch.equal(y[:, TISR_DYN], x[:, TISR_DYN])
    xd, tis = torch.randn(2, 66, HP, WP), torch.zeros(2, 2, HP, WP)
    ens, ent = Dk.rollout(xd, doy, tis, 2, gen=torch.Generator().manual_seed(1))
    assert ens.shape == (2, 4, 2, 66, HP, WP) and torch.isfinite(ens).all() and torch.isfinite(ent).all()
