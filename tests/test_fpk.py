import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fpk.env import contract  # noqa: E402
from fpk.ensemble import fair_crps  # noqa: E402
from fpk.flow import FlowConfig, KernelFlow  # noqa: E402
from fpk.kernel import KernelConfig, PhysicsKernel, smooth_noise  # noqa: E402
from pgcfm.core import static_features  # noqa: E402
from pgcfm.kai_tend import make_core  # noqa: E402

H, W = 8, 16


@pytest.fixture(scope="module")
def core():
    torch.manual_seed(0)
    lat = np.linspace(-78.75, 78.75, H)
    lon = np.linspace(0, 360, W, endpoint=False)
    contract.register_latitudes(lat)
    st = static_features(lat, lon, np.random.rand(H, W).astype(np.float32), np.random.rand(H, W).astype(np.float32))
    mean = np.asarray(contract.REFERENCE_MEAN, dtype=np.float64) if hasattr(contract, "REFERENCE_MEAN") else np.zeros(67)
    std = np.asarray(contract.REFERENCE_STD, dtype=np.float64) if hasattr(contract, "REFERENCE_STD") else np.ones(67)
    c = make_core(st, np.full(67, 0.3), np.asarray(mean).reshape(-1)[:67], np.asarray(std).reshape(-1)[:67], lat,
                  hidden=8, blocks=(1,), norm="gn")
    c.pm = None  # the positivity map needs physical units of a real state
    return c.eval()


def test_smooth_noise_shape_and_variance():
    x = torch.zeros(3, 2, 5, 32, 64)
    xi = smooth_noise(tuple(x.shape), x)
    assert xi.shape == x.shape
    assert abs(float(xi.var()) - 1.0) < 0.1
    big = torch.zeros(400, 1, 32, 64)
    v = smooth_noise(tuple(big.shape), big).var((0, 1, 3))
    assert float((v - 1).abs().max()) < 0.1  # every latitude row, the edge rows included


def test_kernel_nll_is_the_kernel_sum(core):
    K = PhysicsKernel(core, KernelConfig(n_centres=4, hidden=8, blocks=(1,)))
    xd = torch.randn(2, 66, H, W)
    y = xd + 0.3 * torch.randn_like(xd)
    doy = torch.tensor([10, 200])
    c, h, _, phi0 = K(xd, doy, None)
    assert c.shape == (2, 4, 66, H, W) and h.shape == (2, 66, H, W)
    # the antithetic pairs: the centres start symmetric around x, so their mean is close to Phi(x)
    assert float((c.mean(1) - phi0).abs().max().detach()) < 0.5
    w = torch.ones(1, 1, H, 1)
    got = float(K.nll(c, h, y, w))
    s = core.t6
    dens = 0
    for k in range(4):
        z = (y - c[:, k]) / s / h
        dens = dens + torch.exp(-0.5 * z ** 2) / (h * math.sqrt(2 * math.pi))
    lp = torch.log(dens / 4) * core.keep
    want = float(-(lp.sum()) / (core.keep.sum() * 2 * H * W))
    assert abs(got - want) < 1e-3 * max(1.0, abs(want))
    ent = K.entropy(c, h)
    assert torch.isfinite(ent).all()


def test_zero_velocity_flow_is_the_kernel(core):
    K = PhysicsKernel(core, KernelConfig(n_centres=2, hidden=8, blocks=(1,)))
    Fm = KernelFlow(K, FlowConfig(hidden=8, blocks=(1,), steps=3))
    xd = torch.randn(2, 66, H, W)
    doy = torch.tensor([3, 4])
    y, z0 = Fm.sample(xd, doy, None, torch.Generator().manual_seed(1))
    assert torch.allclose(y, core.finish(xd + core.t6 * z0, None), atol=1e-5)
    loss = Fm.loss(xd, xd + 0.1, doy, None, torch.ones(1, 1, H, 1))
    loss.backward()
    assert all(p.grad is None for p in K.parameters())


def test_fair_crps():
    rng = np.random.default_rng(0)
    ens = rng.normal(size=(3, 5, 4))
    obs = rng.normal(size=(3, 4))
    got = fair_crps(ens, obs)
    M = 5
    want = np.abs(ens - obs[:, None]).mean(1) - np.abs(ens[:, :, None] - ens[:, None]).sum((1, 2)) / (2 * M * (M - 1))
    assert np.allclose(got, want)


def test_calibrated_ensemble_scores_zero_gap_and_unit_spread_skill():
    from fpk.ensemble import Scorer
    rng = np.random.default_rng(1)
    B, M, L, Hh, Ww = 40, 8, 2, 8, 16
    ens = rng.normal(size=(B, M, L, 66, Hh, Ww))
    truth = rng.normal(size=(B, L, 66, Hh, Ww))
    sc = Scorer(L, np.ones(Hh), np.ones(67), np.ones(67))
    sc.update(ens, truth)
    out = sc.summary()
    assert abs(out["ign"][0] - out["entropy"][0]) < 0.02
    assert abs(out["spread_skill"][0] - 1.0) < 0.02
