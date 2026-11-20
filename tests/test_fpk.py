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


@pytest.fixture(scope="module")
def mcore():
    torch.manual_seed(1)
    lat = np.linspace(-78.75, 78.75, H)
    lon = np.linspace(0, 360, W, endpoint=False)
    contract.register_latitudes(lat)
    st = static_features(lat, lon, np.random.rand(H, W).astype(np.float32), np.random.rand(H, W).astype(np.float32))
    c = make_core(st, np.full(67, 0.3), np.zeros(67), np.ones(67), lat, hidden=8, blocks=(1,), norm="gn",
                  mem=2, mem_ch=4, mz=True)
    c.pm = None
    with torch.no_grad():  # the head starts at zero: give the memory a say in the tendency
        c.net.head.weight.normal_(0, 0.1)
    return c.eval()


def test_memory_kernel_keeps_one_memory_per_member(mcore):
    from fpk.ensemble import rollout
    K = PhysicsKernel(mcore, KernelConfig(n_centres=4, hidden=8, blocks=(1,)))
    B, M = 2, 3
    xd = torch.randn(B, 66, H, W)
    hist = torch.randn(B, 2, 66, H, W)
    doy = torch.tensor([30, 31])
    # forward: the centres perturb a copy of the memory, the memory itself is the observed one
    K.start(xd, doy, hist)
    m_ref = mcore.m.clone()
    c, h, _, phi0 = K(xd, doy, None, observe=False)
    assert c.shape == (B, 4, 66, H, W) and torch.equal(mcore.m, m_ref) and torch.equal(mcore.fc, phi0)
    assert float((c[:, 0] - c[:, 1]).abs().max()) > 0  # the cloud is a cloud
    # a 3-day ensemble: every member carries its own memory, updated once a day with its own state
    K.start(xd, doy, hist)
    m_ic = mcore.m.clone()
    seen = []

    def step(x, d, tn):
        y0, _, _, _ = K.sample(x, d, tn, torch.Generator().manual_seed(int(d[0])))
        seen.append(mcore.m.clone())
        return mcore.finish(y0, tn)
    ens, hk = rollout(step, xd, doy, torch.zeros(B, 3, H, W), M, 3, kernel=K, start=K.start, hist=hist)
    assert ens.shape == (B, M, 3, 66, H, W) and hk.shape == (B, M, 3, 66, H, W)
    assert seen[0].shape[0] == B * M
    # day 1: the members' memories are the IC's after the IC was remembered (all members alike)
    K.start(xd, doy, hist)
    mcore.remember(xd)
    assert torch.allclose(seen[0], mcore.m.repeat_interleave(M, 0), atol=1e-6)
    assert not torch.allclose(seen[0], m_ic.repeat_interleave(M, 0))
    # day 2: the members differ (each saw its own state and innovation)
    assert float((seen[1].view(B, M, -1)[:, 0] - seen[1].view(B, M, -1)[:, 1]).abs().max()) > 0
    # the memory cloud's spread is learned
    K.start(xd, doy, hist)
    c, h, _, _ = K(xd, doy, None)
    K.nll(c, h, xd, torch.ones(1, 1, H, 1)).backward()
    assert K.eps_m_raw.grad is not None and float(K.eps_m_raw.grad.abs().sum()) > 0


def test_memory_flow_reads_the_memory(mcore):
    K = PhysicsKernel(mcore, KernelConfig(n_centres=2, hidden=8, blocks=(1,)))
    Fm = KernelFlow(K, FlowConfig(hidden=8, blocks=(1,), steps=2))
    xd, doy = torch.randn(2, 66, H, W), torch.tensor([3, 4])
    K.start(xd, doy, torch.randn(2, 2, 66, H, W))
    z0, cond = Fm.source(xd, doy, None)
    assert torch.equal(cond[:, -K.mem_ch:], mcore.m)


def test_crps_torch_matches_numpy():
    from fpk.ensemble import crps_torch
    rng = np.random.default_rng(2)
    ens, obs = rng.normal(size=(3, 6, 5)), rng.normal(size=(3, 5))
    got = crps_torch(torch.from_numpy(ens), torch.from_numpy(obs)).numpy()
    assert np.allclose(got, fair_crps(ens, obs))


def test_pairs_history_and_days():
    from fpk.data import Pairs
    arr = np.arange(20, dtype=np.float32)[:, None, None, None] * np.ones((1, 67, 2, 3), np.float32)
    P = Pairs({2000: arr}, seed=0, hist=3, ahead=4)
    for _ in range(20):
        x, y, d, tn, h = P(5)
        assert y.shape == (5, 4, 67, 2, 3) and h.shape == (5, 3, 67, 2, 3) and tn.shape == (5, 4, 2, 3)
        i = d.numpy()
        assert np.all(i + 4 <= 19)
        assert np.allclose(y[:, :, 0, 0, 0].numpy(), i[:, None] + np.arange(1, 5))
        assert np.allclose(h[:, :, 0, 0, 0].numpy(), np.maximum(i[:, None] + np.arange(-3, 0), 0))
