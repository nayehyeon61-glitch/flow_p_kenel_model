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


@pytest.mark.parametrize("which", ["core", "mcore"])
def test_entropy_deviation_kernel(which, request):
    from fpk.devkernel import DevConfig, EntropyDeviationKernel
    c = request.getfixturevalue(which)
    D = EntropyDeviationKernel(c, DevConfig(pairs=3, hidden=8, blocks=(1,), smooth=3))
    B = 2
    xd, doy = torch.randn(B, 66, H, W), torch.tensor([5, 6])
    hist = torch.randn(B, 2, 66, H, W) if which == "mcore" else None
    ens, ent = D.rollout(xd, doy, torch.zeros(B, 3, H, W), 3, hist=hist)
    assert ens.shape == (B, 6, 3, 66, H, W) and ent.shape == (B, 3, 66, H, W)
    # the ensemble mean is the equation's own trajectory (antithetic pairs around it)
    if which == "mcore":
        c.reset(hist, doy.float() * 24.0, xd)
    x = xd
    for L in range(3):
        x = c.step(x, (doy + L).float() * 24.0, torch.zeros(B, H, W))
        assert torch.allclose(ens[:, :, L].mean(1), x, atol=1e-4)
    # the predicted entropy is the size of the members' deviations (r = 0 at the start), and the residual moves it
    e = D.start(xd, doy, hist)
    x1, e1, H1, (stretch, r) = D.day(xd, e, doy, None, 0)
    assert torch.allclose(H1, D.size(e1), atol=1e-5) and float(r.abs().max()) == 0
    with torch.no_grad():
        D.r_bias.fill_(0.5)
    e = D.start(xd, doy, hist, torch.Generator().manual_seed(0))
    _, e2, H2, _ = D.day(xd, e, doy, None, 0, torch.Generator().manual_seed(1))
    assert torch.allclose(H2, D.size(e2), atol=1e-5)
    loss, _ = D.ign(H2, x1, x1 + 0.1, torch.ones(1, 1, H, 1))
    loss.backward()
    assert D.r_bias.grad is not None and D.h_raw.grad is not None and float(D.h_raw.grad.abs().sum()) > 0
    with torch.no_grad():
        D.r_bias.zero_()


def test_antithetic_pairs_are_scored_without_bias():
    """A calibrated ensemble of P antithetic pairs around the true mean: CRPS = the Gaussian's, spread / skill = 1,
    IGN = H (the statistics of M independent members would say 6 % better CRPS and spread / skill 1.13)."""
    from fpk.ensemble import Scorer, crps_pairs
    rng = np.random.default_rng(3)
    B, P, L, Hh, Ww = 60, 4, 1, 8, 16
    d = rng.normal(size=(B, P, L, 66, Hh, Ww))
    ens = np.concatenate([d, -d], 1)
    truth = rng.normal(size=(B, L, 66, Hh, Ww))
    assert abs(crps_pairs(ens[:, :, 0], truth[:, 0]).mean() - 1 / math.sqrt(math.pi)) < 0.005
    assert abs(fair_crps(ens[:, :, 0], truth[:, 0]).mean() - 1 / math.sqrt(math.pi)) > 0.02
    sc = Scorer(L, np.ones(Hh), np.ones(67), np.ones(67), pairs=P)
    sc.update(ens, truth)
    out = sc.summary()
    assert abs(out["spread_skill"][0] - 1.0) < 0.01
    assert abs(out["ign"][0] - out["entropy"][0]) < 0.02


def test_deviation_flow_starts_as_the_kernel(core):
    from fpk.devflow import DevFlowConfig, DeviationFlow
    from fpk.devkernel import DevConfig, EntropyDeviationKernel
    D = EntropyDeviationKernel(core, DevConfig(pairs=3, hidden=8, blocks=(1,), smooth=3))
    Fm = DeviationFlow(D, DevFlowConfig(hidden=8, blocks=(1,), steps=2))
    xd, doy, tis = torch.randn(2, 66, H, W), torch.tensor([5, 6]), torch.zeros(2, 2, H, W)
    ens, ent = Fm.rollout(xd, doy, tis, 2, gen=torch.Generator().manual_seed(4))
    ref, ent2 = D.rollout(xd, doy, tis, 2, gen=torch.Generator().manual_seed(4))
    assert ens.shape == (2, 3, 2, 66, H, W)
    assert torch.allclose(ens, ref[:, :3], atol=1e-5) and torch.allclose(ent, ent2)  # zero velocity = the kernel's +e
    e = D.start(xd, doy)
    xb, e1, H1, _ = D.day(xd, e, doy, None, 0)
    loss = Fm.loss(e1.detach(), xb, xb + 0.2, Fm.cond(xb, H1.detach(), doy + 1, 1), H1.detach(), torch.ones(1, 1, H, 1))
    loss.backward()
    assert all(p.grad is None for p in D.parameters()) and any(p.grad is not None for p in Fm.net.parameters())


@pytest.mark.parametrize("which", ["core", "mcore"])
def test_correction_kernel_starts_as_the_chain_and_learns_the_mean(which, request):
    from fpk.devkernel import DevConfig, EntropyDeviationKernel
    c = request.getfixturevalue(which)
    W_ = int(getattr(c, "mem", 0))
    torch.manual_seed(0)
    A = EntropyDeviationKernel(c, DevConfig(pairs=2, hidden=8, blocks=(1,), smooth=3))
    Bk = EntropyDeviationKernel(c, DevConfig(pairs=2, hidden=8, blocks=(1,), smooth=3, correct=True))
    Bk.load_state_dict({k: v for k, v in A.state_dict().items()}, strict=False)
    xd, doy, tis = torch.randn(2, 66, H, W), torch.tensor([5, 6]), torch.zeros(2, 3, H, W)
    hist = torch.randn(2, W_, 66, H, W) if W_ else None
    a, ha = A.rollout(xd, doy, tis, 3, hist, torch.Generator().manual_seed(4))
    b, hb = Bk.rollout(xd, doy, tis, 3, hist, torch.Generator().manual_seed(4))
    assert torch.allclose(a, b, atol=1e-5) and torch.allclose(ha, hb, atol=1e-5)  # zero head: the uncorrected chain
    e = Bk.start(xd, doy, hist)
    xb, e1, H1, _ = Bk.day(xd, e, doy, tis[:, 0], 0)
    ign, _ = Bk.ign(H1, xb, xb.detach() + 0.3 * Bk.s, torch.ones(1, 1, H, 1))
    ign.backward()
    g = Bk.cnet.head.weight.grad if hasattr(Bk.cnet, "head") else [p for p in Bk.cnet.parameters()][-2].grad
    assert g is not None and g[:66].abs().sum() > 0 and g[66:].abs().sum() > 0  # the mean and the entropy both learn


def test_gradient_through_the_chain_reaches_an_earlier_correction(mcore):
    from fpk.devkernel import DevConfig, EntropyDeviationKernel
    torch.manual_seed(0)
    Dk = EntropyDeviationKernel(mcore, DevConfig(pairs=2, hidden=8, blocks=(1,), smooth=3, correct=True))
    xd, doy, tis = torch.randn(2, 66, H, W), torch.tensor([5, 6]), torch.zeros(2, 2, H, W)
    hist = torch.randn(2, int(mcore.mem), 66, H, W)

    def grad_of_day2(through):
        Dk.through = through
        Dk.zero_grad()
        e = Dk.start(xd, doy, hist)
        x1, e1, _, _ = Dk.day(xd, e, doy, tis[:, 0], 0)
        probe = torch.zeros_like(x1, requires_grad=True)  # a correction of day 1
        x1 = x1 + probe
        if not through:
            x1 = x1.detach()
            Dk.detach_state()
        x2, _, H2, _ = Dk.day(x1, e1.detach(), doy + 1, tis[:, 1], 1)
        Dk.ign(H2, x2, x2.detach() + 0.3 * Dk.s, torch.ones(1, 1, H, 1))[0].backward()
        return probe.grad

    assert grad_of_day2(0) is None
    g = grad_of_day2(2)
    assert g is not None and float(g.abs().sum()) > 0
    assert all(p.grad is None for p in mcore.parameters())  # the equation's weights stay fixed


def test_spherical_harmonics_roundtrip_parseval_and_isotropic_noise():
    import numpy as np
    from fpk.sphere import SHT
    lat = np.linspace(-90 + 2.8125, 90 - 2.8125, 32)
    S = SHT(lat)
    torch.manual_seed(0)
    c = torch.complex(torch.randn(2, 32, 32), torch.randn(2, 32, 32)) * torch.tril(torch.ones(32, 32))
    c[..., 0] = torch.complex(c[..., 0].real, torch.zeros(2, 32))
    assert torch.allclose(S.analysis(S.synthesis(c)), c, atol=1e-4)  # exact for band-limited fields
    amp = torch.zeros(32)
    amp[6] = 1.0
    n = S.noise(amp, (1000,), torch.Generator().manual_seed(1))
    v = n.pow(2).mean((0, 2))
    assert abs(float(v.mean()) - 1.0) < 0.05 and float(v.max() / v.min()) < 1.15  # unit variance, isotropic
    assert int(S.spectrum(n).mean(0).argmax()) == 6


def test_sphere_noise_starts_like_the_smooth_noise_and_learns(core):
    from fpk.devkernel import DevConfig, EntropyDeviationKernel
    Dk = EntropyDeviationKernel(core, DevConfig(pairs=2, hidden=8, blocks=(1,), smooth=3, noise="sphere"))
    xi = Dk.new_noise((50, 2, 66, H, W), torch.zeros(1), torch.Generator().manual_seed(0))
    assert abs(float(xi.pow(2).mean()) - 1.0) < 0.1                     # unit variance like smooth_noise
    xd, doy, tis = torch.randn(2, 66, H, W), torch.tensor([5, 6]), torch.zeros(2, 2, H, W)
    with torch.no_grad():
        Dk.rho_raw.fill_(2.0)
    e = Dk.start(xd, doy)
    x1, e1, H1, _ = Dk.day(xd, e, doy, tis[:, 0], 0)
    xi1 = Dk._xi.clone()
    x2, e2, H2, _ = Dk.day(x1, e1, doy + 1, tis[:, 1], 1)
    c = float((Dk._xi * xi1).mean() / (xi1.pow(2).mean().sqrt() * Dk._xi.pow(2).mean().sqrt()))
    assert c > 0.8                                                       # AR(1) with rho = tanh(2) = 0.96
    Dk.ign(H2, x2, x2.detach() + 0.3 * Dk.s, torch.ones(1, 1, H, 1))[0].backward()
    assert Dk.a_log.grad is not None and float(Dk.a_log.grad.abs().sum()) > 0


def test_band_scale_starts_as_the_kernel_and_learns_from_the_band_score(core):
    from fpk.devkernel import DevConfig, EntropyDeviationKernel
    torch.manual_seed(0)
    A = EntropyDeviationKernel(core, DevConfig(pairs=2, hidden=8, blocks=(1,), smooth=3))
    Bk = EntropyDeviationKernel(core, DevConfig(pairs=2, hidden=8, blocks=(1,), smooth=3, band_scale=True))
    Bk.load_state_dict(A.state_dict(), strict=False)
    xd, doy, tis = torch.randn(2, 66, H, W), torch.tensor([5, 6]), torch.zeros(2, 2, H, W)
    a, _ = A.rollout(xd, doy, tis, 2, gen=torch.Generator().manual_seed(4))
    b, _ = Bk.rollout(xd, doy, tis, 2, gen=torch.Generator().manual_seed(4))
    assert torch.allclose(a, b, atol=1e-5)
    e = Bk.start(xd, doy)
    x1, e1, H1, _ = Bk.day(xd, e, doy, tis[:, 0], 0)
    Bk.band_ign(e1, x1.detach(), x1.detach() + 0.5 * Bk.s).backward()
    assert Bk.g_band.grad is not None and float(Bk.g_band.grad[0].abs().sum()) > 0


def test_initial_deviations_follow_the_given_spectrum(core, tmp_path):
    import json
    from fpk.devkernel import DevConfig, EntropyDeviationKernel
    sp = [0.0] * H
    sp[2] = 1.0
    f = tmp_path / "d.json"
    f.write_text(json.dumps({"err_deg": {"all": [sp]}}))
    Dk = EntropyDeviationKernel(core, DevConfig(pairs=2, hidden=8, blocks=(1,), eps0=1.0, e0_spec=str(f)))
    e = Dk.start(torch.zeros(20, 66, H, W), torch.zeros(20, dtype=torch.long), gen=torch.Generator().manual_seed(0))
    k = Dk.keep.reshape(-1) > 0
    spec = Dk.sht.spectrum(e[:, :, k]).mean((0, 1, 2))
    assert int(spec.argmax()) == 2 and abs(float(spec.sum()) - 1.0) < 0.15


def test_noise_composed_with_the_equation(core):
    from fpk.devkernel import DevConfig, EntropyDeviationKernel
    torch.manual_seed(0)
    Dk = EntropyDeviationKernel(core, DevConfig(pairs=2, hidden=8, blocks=(1,), smooth=3, noise_pre=True))
    xd, doy, tis = torch.randn(2, 66, H, W), torch.tensor([5, 6]), torch.zeros(2, 1, H, W)
    e = torch.zeros(2, 2, 66, H, W)                      # no deviations: the day's spread is all M (h xi)
    x1, e1, H1, _ = Dk.day(xd, e, doy, tis[:, 0], 0)
    assert float(e1.abs().mean()) > 0 and torch.isfinite(H1).all()
    assert float(Dk.last["hxi"].abs().max()) < 1e-6     # nothing is added after the equation


def test_noise_multiplying_the_tendency(mcore):
    from fpk.devkernel import DevConfig, EntropyDeviationKernel
    torch.manual_seed(0)
    Dk = EntropyDeviationKernel(mcore, DevConfig(pairs=2, hidden=8, blocks=(1,), smooth=3, noise_mult=0.5))
    xd, doy, tis = torch.randn(2, 66, H, W), torch.tensor([5, 6]), torch.zeros(2, 1, H, W)
    e = Dk.start(xd, doy, torch.randn(2, int(mcore.mem), 66, H, W)) * 0
    x1, e1, H1, _ = Dk.day(xd, e, doy, tis[:, 0], 0)
    tend = (x1 - xd) / Dk.s * Dk.keep
    new = Dk.last["hxi"]                                   # a zeta tend: proportional to the day's tendency
    k = Dk.keep.reshape(-1) > 0
    a = torch.nn.functional.softplus(Dk.a_raw)[k].view(1, 1, -1, 1, 1)
    zeta = new[:, :, k] / (a * tend[:, None][:, :, k])     # the same field in every channel
    assert float(tend.abs().mean()) > 0 and float((zeta - zeta[:, :, :1]).abs().median()) < 1e-3
    Dk.ign(H1, x1, x1.detach() + 0.3 * Dk.s, torch.ones(1, 1, H, 1))[0].backward()
    assert float(Dk.a_raw.grad.abs().sum()) > 0


def _gc_dphi(F):
    """d/dphi of a field (H, W) that is smooth over the poles, by the FFT along the great circles (the meridian
    lon and its continuation lon + 180 deg form 2 H equally spaced points of a circle)."""
    Hn = F.shape[-2]
    G = torch.cat([F, F.roll(F.shape[-1] // 2, -1).flip(-2)], -2)
    k = torch.fft.fftfreq(2 * Hn, 1.0 / (2 * Hn))
    return torch.fft.ifft(1j * k[:, None] * torch.fft.fft(G, dim=-2), dim=-2).real[..., :Hn, :]


def test_sht_derivatives_on_analytic_fields():
    """Tolerances on the 32 x 64 grid in float32: 1e-4 relative for the analytic fields, 1e-3 for the pointwise
    divergence / vorticity of random fields of degree <= 20 against the size of the other one."""
    from fpk.sphere import OMEGA, SHT
    lat = np.linspace(-90 + 2.8125, 90 - 2.8125, 32)
    S = SHT(lat)
    assert list(S.state_dict()) == ["syn", "ana", "deg"]                 # the checkpoints keep their keys
    a = S.radius
    phi = torch.tensor(np.deg2rad(lat), dtype=torch.float32)[:, None]
    lam = torch.arange(64)[None] * (2 * math.pi / 64)
    cos, sin = torch.cos(phi), torch.sin(phi)
    # psi = a cos(phi) cos(lam) (l = 1): lap psi = -2 psi / a^2, u = sin(phi) cos(lam), v = -sin(lam)
    psi = a * cos * torch.cos(lam)
    c = S.analysis(psi)
    assert float((S.synthesis(S.lap(c)) + 2 * psi / a ** 2).abs().max()) < 1e-4 * 2 / a
    assert torch.allclose(S.inv_lap(S.lap(c)), c, atol=1e-4 * a)
    u, v = S.wind(c)
    assert float((u - sin * torch.cos(lam)).abs().max()) < 1e-4 and float((v + torch.sin(lam) * cos / cos).abs().max()) < 1e-4
    # f = cos^2(phi) cos(2 lam) (l = 2, m = 2): the gradient, and the eigenvalue -l (l + 1) / a^2 = -6 / a^2
    f = cos ** 2 * torch.cos(2 * lam)
    cf = S.analysis(f)
    gx, gy = S.grad(cf)
    assert float((a * gx + 2 * cos * torch.sin(2 * lam)).abs().max()) < 1e-4
    assert float((a * gy + 2 * cos * sin * torch.cos(2 * lam)).abs().max()) < 1e-4
    assert float((a ** 2 * S.synthesis(S.lap(cf)) + 6 * f).abs().max()) < 1e-3
    # random psi, chi of degree <= 20: the wind of psi has no divergence, the wind of chi no vorticity (pointwise,
    # d/dlam and d/dphi by FFT along the latitude and the great circles), and helmholtz inverts wind
    g = torch.Generator().manual_seed(0)
    cp, cc = S.noise_coef((2,), torch.zeros(1), g), S.noise_coef((2,), torch.zeros(1), g)
    cp[..., 21:, :], cc[..., 21:, :], cp[..., 0, :], cc[..., 0, :] = 0, 0, 0, 0
    dlam = lambda F: torch.fft.irfft(1j * torch.arange(33) * torch.fft.rfft(F, dim=-1), n=64, dim=-1)  # noqa: E731
    div = lambda u, v: (dlam(u) + _gc_dphi(v * cos)) / (a * cos)  # noqa: E731
    vor = lambda u, v: (dlam(v) - _gc_dphi(u * cos)) / (a * cos)  # noqa: E731
    rms = lambda t: float(t.pow(2).mean().sqrt())  # noqa: E731
    up, vp = S.wind(cp)
    uc, vc = S.wind(torch.zeros_like(cp), cc)
    assert rms(div(up, vp)) < 1e-3 * rms(vor(up, vp)) and rms(vor(uc, vc)) < 1e-3 * rms(div(uc, vc))
    assert rms(vor(up, vp) - S.synthesis(S.lap(cp))) < 1e-3 * rms(vor(up, vp))
    p2, c2 = S.helmholtz(up + uc, vp + vc)
    assert torch.allclose(p2, cp, atol=1e-4) and torch.allclose(c2, cc, atol=1e-4)
    zeta, dv = S.vortdiv(up, vp)
    assert rms(dv) < 1e-4 * rms(zeta)
    # linear balance: the solid-body rotation u = U cos(phi), psi = -a U sin(phi): Phi = -a Omega U (sin^2 - 1/3);
    # and lap Phi = f lap psi + (2 Omega cos(phi) / a) (1 / a) dpsi/dphi on the grid for the random psi
    U = 10.0
    Phi = S.synthesis(S.linear_balance(S.analysis(-a * U * sin.expand(32, 64))))
    assert float((Phi + a * OMEGA * U * (sin ** 2 - 1 / 3)).abs().max()) < 1e-4 * a * OMEGA * U
    rhs = 2 * OMEGA * sin * S.synthesis(S.lap(cp)) + 2 * OMEGA * cos / a * S.grad(cp)[1]
    assert rms(S.synthesis(S.lap(S.linear_balance(cp))) - rhs) < 1e-4 * rms(rhs)


def _balanced(mcore, **kw):
    from fpk.devkernel import DevConfig, EntropyDeviationKernel
    return EntropyDeviationKernel(mcore, DevConfig(pairs=2, hidden=8, blocks=(1,), smooth=3, noise="balanced", **kw))


def test_default_kernel_has_no_new_state(core):
    from fpk.devkernel import DevConfig, EntropyDeviationKernel
    own = lambda D: sorted(k for k in D.state_dict() if not k.startswith(("core.", "net.", "cnet.")))  # noqa: E731
    cfg = dict(pairs=2, hidden=8, blocks=(1,))
    assert own(EntropyDeviationKernel(core, DevConfig(**cfg))) == ["h_raw", "r_bias"]
    assert own(EntropyDeviationKernel(core, DevConfig(noise="sphere", band_scale=True, **cfg))) == sorted(
        ["h_raw", "r_bias", "a_log", "rho_raw", "g_band", "band_of", "base_amp", "sht.syn", "sht.ana", "sht.deg"])
    assert DevConfig().noise == "smooth" and DevConfig().fdt == "" and DevConfig().noise_spec == ""


def test_balanced_noise_is_balanced(mcore):
    """Tolerances (8 x 16 grid, float32): divergence < 1e-3 of the vorticity (RMS) without chi and remainder;
    hydrostatic and linear-balance residuals < 1e-3 of the terms."""
    from fpk.devkernel import R_DRY, level_channels
    from fpk.sphere import OMEGA
    torch.manual_seed(0)
    Dk = _balanced(mcore)
    S, (iz, it, iu, iv) = Dk.sht, Dk.bal_idx
    rms = lambda t: float(t.pow(2).mean().sqrt())  # noqa: E731
    with torch.no_grad():
        x = Dk.new_part((40, 2, 66, H, W), torch.zeros(1), torch.Generator().manual_seed(0))
        si = lambda k: x[:, :, Dk.bal_idx[k]] * Dk.bal_unit[k].view(1, 1, -1, 1, 1)  # noqa: E731
        zeta, dv = S.vortdiv(si(2), si(3))
        r0 = rms(dv) / rms(zeta)                 # chi = 0.1 psi and the remainder 0.3: clearly divergent
        Dk.chi_raw.fill_(-30.0)
        Dk.w_raw.fill_(-30.0)
        Dk.h_raw.fill_(math.log(math.e - 1))     # h = 1
        x = Dk.new_part((40, 2, 66, H, W), torch.zeros(1), torch.Generator().manual_seed(0))
        zeta, dv = S.vortdiv(si(2), si(3))
        assert rms(dv) < 1e-3 * rms(zeta) and r0 > 0.05
        # h = 1: the wind channels have about unit variance; q and the surface keep the unit sphere noise
        assert abs(rms(x[:, :, torch.cat([iu, iv])]) - 1.0) < 0.15 and abs(rms(x[:, :, 12:24]) - 1.0) < 0.1
        assert float(x[:, :, Dk.keep.reshape(-1) == 0].abs().max()) == 0
        # hydrostatic: R T = -dPhi/dln p (centred in ln p, one-sided at the ends), on the fields of the noise
        phi, T = si(0), si(1)
        lnp = torch.tensor(np.log(level_channels()[1]), dtype=torch.float32)
        d = torch.stack([(phi[:, :, min(k + 1, 11)] - phi[:, :, max(k - 1, 0)]) / (lnp[min(k + 1, 11)] - lnp[max(k - 1, 0)])
                         for k in range(12)], 2)
        assert rms(R_DRY * T + d) < 1e-3 * rms(d)
        # the levels are correlated in the vertical (T of independent levels would be several times larger)
        assert float((phi[:, :, 5] * phi[:, :, 6]).mean()) > 0.8 * rms(phi[:, :, 5]) * rms(phi[:, :, 6])
        # linear balance on a psi of degree <= L - 2 (lap Phi is then inside the truncation): lap Phi = div(f grad psi)
        cp = S.noise_coef((3, 12), torch.zeros(1), torch.Generator().manual_seed(1)) * 1e6
        cp[..., S.L - 1:, :], cp[..., 0, :] = 0, 0
        u, v, phi, T = Dk.balance(cp)
        ph = torch.tensor(np.deg2rad([-90.0 + 180.0 / H * (i + 0.5) for i in range(H)]), dtype=torch.float32)[:, None]
        rhs = 2 * OMEGA * torch.sin(ph) * S.synthesis(S.lap(cp)) + 2 * OMEGA * torch.cos(ph) / S.radius * S.grad(cp)[1]
        assert rms(S.synthesis(S.lap(S.analysis(phi))) - rhs) < 1e-3 * rms(rhs)
        assert rms(S.vortdiv(u, v)[1]) < 1e-3 * rms(S.vortdiv(u, v)[0])


def test_balanced_noise_day_gradients_and_time_correlation(mcore, tmp_path):
    import json
    from fpk.devkernel import DevConfig, EntropyDeviationKernel
    torch.manual_seed(0)
    Dk = _balanced(mcore)
    xd, doy, tis = torch.randn(2, 66, H, W), torch.tensor([5, 6]), torch.zeros(2, 2, H, W)
    hist = torch.randn(2, int(mcore.mem), 66, H, W)
    with torch.no_grad():
        Dk.rho_raw.fill_(2.0)
    e = Dk.start(xd, doy, hist)
    x1, e1, H1, _ = Dk.day(xd, e, doy, tis[:, 0], 0)
    assert x1.shape == (2, 66, H, W) and e1.shape == (2, 2, 66, H, W) and H1.shape == (2, 66, H, W)
    assert torch.isfinite(e1).all() and torch.isfinite(H1).all() and Dk.last["hxi"].shape == e1.shape
    assert float(e1[:, :, Dk.keep.reshape(-1) == 0].abs().max()) == 0
    b1, n1 = Dk._bc.clone(), Dk.last["hxi"].detach().clone()
    x2, e2, H2, _ = Dk.day(x1, e1.detach(), doy + 1, tis[:, 1], 1)
    cor = lambda a, b: float((a.conj() * b).real.sum() / (a.abs().pow(2).sum().sqrt() * b.abs().pow(2).sum().sqrt()))  # noqa: E731
    assert cor(b1, Dk._bc) > 0.8 and cor(n1, Dk.last["hxi"].detach()) > 0.8   # AR(1), rho = tanh(2) = 0.96
    Dk.ign(H2, x2, x2.detach() + 0.3 * Dk.s, torch.ones(1, 1, H, 1))[0].backward()
    for n in ("p_log", "chi_raw", "w_raw", "a_log", "h_raw", "rho_raw"):
        g = getattr(Dk, n).grad
        assert g is not None and torch.isfinite(g).all() and float(g.abs().sum()) > 0, n
    assert all(p.grad is None for p in mcore.parameters())
    # composed with the equation (noise_pre) it runs; with the multiplicative noise it is refused
    Dp = _balanced(mcore, noise_pre=True)
    x1, e1, H1, _ = Dp.day(xd, Dp.start(xd, doy, hist) * 0, doy, tis[:, 0], 0)
    assert torch.isfinite(H1).all() and float(e1.abs().mean()) > 0 and float(Dp.last["hxi"].abs().max()) < 1e-6
    with pytest.raises(ValueError):
        _balanced(mcore, noise_mult=0.5)
    # the start of psi's spectrum from a 1-day error spectrum of the wind: amp_psi^2 ~ E(l) / (l (l + 1))
    E = [0.0, 1.0, 2.0, 4.0, 2.0, 1.0, 0.5, 0.25]
    f = tmp_path / "d.json"
    f.write_text(json.dumps({"err_deg": {"all": [[1.0] * H], "u250": [E]}}))
    Ds = _balanced(mcore, noise_spec=str(f))
    l = torch.arange(H, dtype=torch.float32)
    ke = Ds.psi_amp.pow(2) * l * (l + 1)                       # the wind's spectrum of the start
    assert torch.allclose(ke, torch.tensor(E) / sum(E), atol=1e-5) and float(Ds.psi_amp[0]) == 0
    with torch.no_grad():
        Ds.w_raw.fill_(-30.0)
        Ds.chi_raw.fill_(-30.0)
        x = Ds.new_part((60, 2, 66, H, W), torch.zeros(1), torch.Generator().manual_seed(0))
    sp = Ds.sht.spectrum(x[:, :, Ds.bal_idx[2]]).mean((0, 1, 2)) + Ds.sht.spectrum(x[:, :, Ds.bal_idx[3]]).mean((0, 1, 2))
    assert int(sp.argmax()) == 3
    # a checkpoint carries the constants: a kernel built without the statistics loads them
    Dl = _balanced(mcore)
    Dl.load_state_dict({k: v for k, v in Ds.state_dict().items() if not k.startswith("core.")}, strict=False)
    assert torch.equal(Dl.psi_amp, Ds.psi_amp) and torch.equal(Dl.bal_k, Ds.bal_k)


def test_fdt_bound_and_fixed_time_correlation(core, tmp_path):
    import json
    from fpk.devkernel import DevConfig, EntropyDeviationKernel
    from fpk.sphere import bands
    nb = bands(H)[1]
    f = tmp_path / "clim.json"
    f.write_text(json.dumps(dict(c=[[1.0] * nb] * 66, rho=[[0.5] * nb] * 66, rho_band=[0.9, 0.9, 0.5, 0.0])))
    Dk = EntropyDeviationKernel(core, DevConfig(pairs=2, hidden=8, blocks=(1,), smooth=3, noise="sphere", fdt=str(f)))
    assert not Dk.rho_raw.requires_grad and torch.allclose(Dk.rho()[Dk.band_of == 1], torch.tensor(0.9))
    e = smooth_noise((2, 2, 66, H, W), torch.zeros(1), torch.Generator().manual_seed(0)) * Dk.keep.unsqueeze(0)
    V = Dk.band_power(e)
    assert float(V[:, Dk.keep.reshape(-1) > 0].max()) < 1.0 and float(Dk.fdt_penalty(e)) == 0   # below the climate
    big = (3.0 * e).requires_grad_(True)
    pen = Dk.fdt_penalty(big)
    k = Dk.keep.reshape(-1) > 0
    want = torch.relu(torch.log(Dk.band_power(big.detach()).mean(0)[k])).pow(2).mean()   # V: mean over the batch
    assert float(pen) > 0 and abs(float(pen) - float(want)) < 1e-6
    pen.backward()
    assert float(big.grad.abs().sum()) > 0
    # the AR(1) of the noise uses the file's rho: band 1 keeps 0.9 of the last draw, the last band nothing
    xd, doy = torch.randn(2, 66, H, W), torch.tensor([5, 6])
    e0 = Dk.start(xd, doy)
    Dk.new_noise(tuple(e0.shape), xd, torch.Generator().manual_seed(1))
    a = Dk.sht.analysis(Dk._xi)
    Dk.new_noise(tuple(e0.shape), xd, torch.Generator().manual_seed(2))
    b = Dk.sht.analysis(Dk._xi)
    cor = lambda j: float((a.conj() * b).real[..., Dk.band_of == j, :].sum()  # noqa: E731
                          / (a.abs().pow(2)[..., Dk.band_of == j, :].sum().sqrt() * b.abs().pow(2)[..., Dk.band_of == j, :].sum().sqrt()))
    assert abs(cor(1) - 0.9) < 0.08 and abs(cor(nb - 1)) < 0.1
    # without sphere noise the bound alone works (smooth noise), and a wrong file is refused
    Dn = EntropyDeviationKernel(core, DevConfig(pairs=2, hidden=8, blocks=(1,), fdt=str(f)))
    assert float(Dn.fdt_penalty(3.0 * e)) > 0 and not hasattr(Dn, "rho_raw")
    f.write_text(json.dumps(dict(c=[[1.0] * (nb + 1)] * 66, rho_band=[0.5] * (nb + 1))))
    with pytest.raises(ValueError):
        EntropyDeviationKernel(core, DevConfig(pairs=2, hidden=8, blocks=(1,), fdt=str(f)))


def test_balanced_amplitude_scales_psi_and_start_values(mcore):
    """h of the wind channels is the amplitude of psi BEFORE the derivatives (u, v, Phi and T follow it together);
    the start values of the spec (chi 0.1 of psi, remainder 0.3) and the unit std s of the balanced fields."""
    import copy
    import torch.nn.functional as Fn
    from fpk.devkernel import DevConfig, EntropyDeviationKernel, R_DRY, level_channels
    from fpk.env import DYN
    from fpk.sphere import A_EARTH
    Dk = _balanced(mcore)
    iz, it, iu, iv = Dk.bal_idx
    # the start: chi = 0.1 psi, the remainder 0.3 on z, t, u, v of the levels and 1 on every other channel
    assert torch.allclose(Fn.softplus(Dk.chi_raw), torch.tensor(0.1)) and Dk.chi_raw.shape == (12, Dk.n_bands)
    assert torch.allclose(Fn.softplus(Dk.w_raw), torch.tensor(0.3)) and float(Dk.p_log.abs().max()) == 0
    w = Dk.remainder_weight()
    lev = torch.zeros(66, dtype=torch.bool)
    lev[Dk.bal_idx.reshape(-1)] = True
    assert int(lev.sum()) == 48 and torch.allclose(w[lev], torch.tensor(0.3)) and bool((w[~lev] == 1).all())

    def draw(D, seed=0, n=40):
        D._xi, D._bc = None, None
        with torch.no_grad():
            return D.new_part((n, 2, 66, H, W), torch.zeros(1), torch.Generator().manual_seed(seed))

    # the size of the start: the remainder is 0.3 h of unit noise, the divergent wind about 0.1 of the rotational
    with torch.no_grad():
        Dk.h_raw.fill_(math.log(math.e - 1))             # h = 1
    x = draw(Dk)
    rms = lambda t: float(t.pow(2).mean().sqrt())  # noqa: E731
    with torch.no_grad():
        Dk.p_log.fill_(-30.0)                            # psi and chi off: the remainder alone
    rem = draw(Dk)
    assert abs(rms(rem[:, :, lev]) - 0.3) < 0.03 and torch.allclose(rem[:, :, 12:24], x[:, :, 12:24])
    with torch.no_grad():
        Dk.p_log.fill_(0.0)
        Dk.w_raw.fill_(-30.0)
    xr = draw(Dk)
    si = lambda t, k: t[:, :, Dk.bal_idx[k]] * Dk.bal_unit[k].view(1, 1, -1, 1, 1)  # noqa: E731
    zeta, dv = Dk.sht.vortdiv(si(xr, 2), si(xr, 3))
    assert 0.05 < rms(dv) / rms(zeta) < 0.2
    # h: twice the h of u and v on every level doubles u, v, z and T; the h of z and T does not enter
    h1 = math.log(math.e - 1)
    with torch.no_grad():
        Dk.chi_raw.fill_(-30.0)
    x1 = draw(Dk)
    with torch.no_grad():
        Dk.h_raw[torch.cat([iu, iv])] = math.log(math.expm1(2.0))
    x2 = draw(Dk)
    assert torch.allclose(x2[:, :, lev], 2 * x1[:, :, lev], rtol=1e-4, atol=1e-5) and rms(x1[:, :, it]) > 0.1
    with torch.no_grad():
        Dk.h_raw[torch.cat([iu, iv])] = h1
        Dk.h_raw[torch.cat([iz, it])] = math.log(math.expm1(5.0))
    assert torch.allclose(draw(Dk), x1, atol=1e-5)
    # one level (k = 4), a = sqrt(h_u h_v) = 2: its u, v and Phi double, the other levels keep theirs, and T stays
    # the hydrostatic T of the new Phi (the levels next to it change, R T = -dPhi/dln p)
    with torch.no_grad():
        Dk.h_raw[torch.cat([iz, it])] = h1
        Dk.h_raw[iu[4]], Dk.h_raw[iv[4]] = math.log(math.expm1(4.0)), math.log(math.expm1(1.0))
    x3 = draw(Dk)
    oth = torch.tensor([k for k in range(12) if k != 4])
    for i in (iz, iu, iv):
        assert torch.allclose(x3[:, :, i[4]], 2 * x1[:, :, i[4]], rtol=1e-4, atol=1e-5)
        assert torch.allclose(x3[:, :, i[oth]], x1[:, :, i[oth]], atol=1e-5)
    assert rms(x3[:, :, it[3]] - x1[:, :, it[3]]) > 0.1 * rms(x1[:, :, it[3]])
    phi, T = si(x3, 0), si(x3, 1)
    lnp = torch.tensor(np.log(level_channels()[1]), dtype=torch.float32)
    d = torch.stack([(phi[:, :, min(k + 1, 11)] - phi[:, :, max(k - 1, 0)]) / (lnp[min(k + 1, 11)] - lnp[max(k - 1, 0)])
                     for k in range(12)], 2)
    assert rms(R_DRY * T + d) < 1e-3 * rms(d)
    # the unit of a core whose s differs by channel and of the run's own std: SI = std s e, and the wind of every
    # level has about unit variance in these units at h = 1
    c2 = copy.deepcopy(mcore)
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        c2.t6.copy_((0.1 + 0.5 * torch.rand(66, generator=g)).view_as(c2.t6))
    std = 0.5 + 10 * torch.rand(67, generator=g).numpy()
    D2 = EntropyDeviationKernel(c2, DevConfig(pairs=2, hidden=8, blocks=(1,), smooth=3, noise="balanced"), std=std)
    want = (torch.tensor(std[np.asarray(DYN)], dtype=torch.float32) * c2.t6.reshape(-1))[D2.bal_idx]
    assert torch.allclose(D2.bal_unit, want, rtol=1e-6) and float(want.std() / want.mean()) > 0.3
    assert torch.allclose(D2.bal_k, A_EARTH * torch.sqrt(4.0 / (want[2] ** -2 + want[3] ** -2)), rtol=1e-5)
    with torch.no_grad():
        D2.h_raw.fill_(h1)
        D2.w_raw.fill_(-30.0)
        D2.chi_raw.fill_(-30.0)
    y = draw(D2, n=80)
    var = 0.5 * (y[:, :, D2.bal_idx[2]].pow(2).mean((0, 1, 3, 4)) + y[:, :, D2.bal_idx[3]].pow(2).mean((0, 1, 3, 4)))
    assert float((var - 1).abs().max()) < 0.2, var


def test_balanced_time_state_spectrum_fallback_and_init(mcore, tmp_path):
    """The AR(1) state of psi / chi is cleared by start() and uses the fdt file's rho; noise_spec falls back to
    e0_spec; load_init keeps the fixed buffers of this run's options."""
    import json
    from fpk.sphere import bands
    nb = bands(H)[1]
    xd, doy = torch.randn(2, 66, H, W), torch.tensor([5, 6])
    hist = torch.randn(2, int(mcore.mem), 66, H, W)
    shape = (2, 2, 66, H, W)
    gen = lambda i: torch.Generator().manual_seed(i)  # noqa: E731
    # (a) a new trajectory does not carry the last draw of the one before (rho = tanh(2) = 0.96)
    Dk = _balanced(mcore)
    with torch.no_grad():
        Dk.rho_raw.fill_(2.0)
        Dk.start(xd, doy, hist)
        Dk.new_part(shape, xd, gen(1))
        carried = Dk.new_part(shape, xd, gen(2))
        Dk.new_part(shape, xd, gen(1))
        Dk.start(xd, doy, hist)
        assert Dk._bc is None and Dk._xi is None
        fresh = Dk.new_part(shape, xd, gen(2))
        Dk.start(xd, doy, hist)
        again = Dk.new_part(shape, xd, gen(2))
    assert torch.equal(fresh, again) and float((fresh - carried).abs().max()) > 0.1
    # (b) fdt: the AR(1) of the psi / chi draws is the file's rho per band (0.9 at l = 1, 0 in the last band), not
    # the frozen rho_raw = 0
    f = tmp_path / "clim.json"
    f.write_text(json.dumps(dict(c=[[1.0] * nb] * 66, rho_band=[0.9, 0.9, 0.5, 0.0])))
    Df = _balanced(mcore, fdt=str(f))
    assert not Df.rho_raw.requires_grad and float(Df.rho_raw.abs().max()) == 0
    with torch.no_grad():
        Df.new_part((40, 2, 66, H, W), xd, gen(1))
        a = Df._bc.clone()
        Df.new_part((40, 2, 66, H, W), xd, gen(2))
        b = Df._bc.clone()
    bj = Df.band_of
    cor = lambda j: float((a.conj() * b).real[..., bj == j, :].sum()  # noqa: E731
                          / (a.abs().pow(2)[..., bj == j, :].sum().sqrt() * b.abs().pow(2)[..., bj == j, :].sum().sqrt()))
    assert abs(cor(1) - 0.9) < 0.03 and abs(cor(2) - 0.5) < 0.05 and abs(cor(nb - 1)) < 0.05
    # (c) the spectrum of psi: noise_spec, else e0_spec, else the base spectrum of the sphere noise
    E1, E2 = [0.0, 1.0, 2.0, 4.0, 2.0, 1.0, 0.5, 0.25], [0.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.1]
    f1, f2 = tmp_path / "s1.json", tmp_path / "s2.json"
    f1.write_text(json.dumps({"err_deg": {"all": [[1.0] * H], "u250": [E1]}}))
    f2.write_text(json.dumps({"err_deg": {"all": [[1.0] * H], "u250": [E2]}}))
    l = torch.arange(H, dtype=torch.float32)
    ke = lambda D: D.psi_amp.pow(2) * l * (l + 1)  # noqa: E731
    D1, D12, D0 = _balanced(mcore, e0_spec=str(f1)), _balanced(mcore, e0_spec=str(f1), noise_spec=str(f2)), _balanced(mcore)
    assert torch.allclose(ke(D1), torch.tensor(E1) / sum(E1), atol=1e-5)
    assert torch.allclose(ke(D12), torch.tensor(E2) / sum(E2), atol=1e-5)
    assert torch.allclose(ke(D0)[1:], (D0.base_amp.pow(2) / D0.base_amp.pow(2)[1:].sum())[1:], atol=1e-5)
    # a file without u250: the spectrum of all channels
    f1.write_text(json.dumps({"err_deg": {"all": [E2]}}))
    assert torch.allclose(ke(_balanced(mcore, noise_spec=str(f1))), torch.tensor(E2) / sum(E2), atol=1e-5)
    # (d) --init: the learned parts come from the checkpoint, the fixed buffers stay those of this run's options
    A = _balanced(mcore, fdt=str(f))
    with torch.no_grad():
        A.p_log.fill_(0.5)
        A.h_raw.fill_(0.7)
    state = {k: v.clone() for k, v in A.state_dict().items() if not k.startswith("core.")}
    f.write_text(json.dumps(dict(c=[[2.0] * nb] * 66, rho_band=[0.1] * nb)))
    C = _balanced(mcore, fdt=str(f), bal_vert=0.0, noise_spec=str(f2))
    miss, own = C.load_init(state)
    assert not miss.unexpected_keys and sorted(own) == ["bal_vert", "fdt_c", "fdt_rho", "psi_amp"]
    assert torch.equal(C.bal_vert, torch.eye(12)) and torch.allclose(ke(C), torch.tensor(E2) / sum(E2), atol=1e-5)
    assert float(C.fdt_c.min()) == 2.0 and torch.allclose(C.fdt_rho, torch.tensor(0.1))
    assert torch.equal(C.p_log, A.p_log) and torch.equal(C.h_raw, A.h_raw)
    assert C.load_init({k: v for k, v in C.state_dict().items() if not k.startswith("core.")})[1] == []


def test_fdt_penalty_at_the_climate_and_first_band(core, tmp_path):
    """Deviations whose band power IS the climate's: the bound is not active beyond the scatter of the batch's
    estimate (the penalty of every sample alone would be several times larger), and fdt_bmin leaves l = 0 out."""
    import json
    import torch.nn.functional as Fn
    from fpk.devkernel import DevConfig, EntropyDeviationKernel
    from fpk.sphere import SHT, bands
    bj, nb = bands(H)
    S = SHT([-90.0 + 180.0 / H * (i + 0.5) for i in range(H)])
    amp = torch.ones(H) / math.sqrt(H)
    c = torch.stack([(amp ** 2)[bj == b].sum() for b in range(nb)])                 # the band power of the noise
    f = tmp_path / "clim.json"
    f.write_text(json.dumps(dict(c=[c.tolist()] * 66, rho_band=[0.5] * nb)))
    cfg = dict(pairs=2, hidden=8, blocks=(1,), fdt=str(f))
    Dk, D1 = EntropyDeviationKernel(core, DevConfig(**cfg)), EntropyDeviationKernel(core, DevConfig(fdt_bmin=1, **cfg))
    e = S.noise(amp, (64, 2, 66), torch.Generator().manual_seed(0)) * Dk.keep.unsqueeze(0)
    k = Dk.keep.reshape(-1) > 0
    V = Dk.band_power(e)
    assert abs(float(V[:, k].mean((0, 1))[-1] / c[-1]) - 1) < 0.05
    each = Fn.relu(torch.log(V[:, k]) - torch.log(c)).pow(2).mean()                # sample by sample: biased
    pen = float(Dk.fdt_penalty(e))
    assert pen < 0.004 and float(each) > 0.03 and pen < 0.1 * float(each)
    # twice the climate's power in the band l = 0 alone: seen by the bound, not with fdt_bmin = 1
    co = S.analysis(e)
    co[..., 0, :] = co[..., 0, :] * math.sqrt(2.0)
    e2 = S.synthesis(co)
    assert float(Dk.fdt_penalty(e2)) > 0.5 * math.log(2.0) ** 2 / nb and float(D1.fdt_penalty(e2)) < 0.004
    assert abs(float(D1.fdt_penalty(3.0 * e)) - math.log(9.0) ** 2) < 0.2 and DevConfig().fdt_bmin == 0


def test_daily_climatology_of_years():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from clim_bands import daily_clim
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(365, 2, 2, 3)).astype(np.float32), rng.normal(size=(366, 2, 2, 3)).astype(np.float32)
    m = np.concatenate([0.5 * (a + b[:365]), b[365:]], 0)        # the mean per day; day 366 from the leap year alone
    assert np.allclose(daily_clim(iter([a, b]), half=0, step=50), m, atol=1e-6)
    cl = daily_clim([a, b], half=15)
    want = lambda i: m[np.arange(i - 15, i + 16) % 366].mean(0)  # noqa: E731
    assert cl.shape == (366, 2, 2, 3) and all(np.allclose(cl[i], want(i), atol=1e-5) for i in (0, 7, 200, 365))
    # more years: the climatology's own sampling error, which the anomaly's variance would carry, shrinks as 1 / n
    ys = [rng.normal(size=(365, 1, 4, 4)).astype(np.float32) for _ in range(16)]
    v2, v16 = daily_clim(ys[:2], half=0)[:365].var(), daily_clim(ys, half=0)[:365].var()
    assert abs(v2 - 0.5) < 0.05 and abs(v16 - 1 / 16) < 0.01


def test_climate_band_power_and_autocorrelation():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from clim_bands import band_stats
    from fpk.sphere import SHT, bands
    lat = np.linspace(-90 + 5.625, 90 - 5.625, 16)
    S = SHT(lat)
    bj, nb = bands(S.L)
    g = torch.Generator().manual_seed(0)
    amp = torch.ones(S.L) / math.sqrt(S.L)                     # unit variance, equal power per degree
    rho = torch.tensor([0.9, 0.8, 0.6, 0.3, 0.0])[bj]
    x = [S.noise(amp, (3,), g)]
    for _ in range(399):                                       # an AR(1) in time with a rho per band
        c = rho[:, None] * S.analysis(x[-1]) + torch.sqrt(1 - rho ** 2)[:, None] * S.analysis(S.noise(amp, (3,), g))
        x.append(S.synthesis(c))
    c, r = band_stats(torch.stack(x), S, chunk=64)
    want = torch.stack([(amp ** 2)[bj == b].sum() for b in range(nb)])
    assert c.shape == (3, nb) and float((c / want - 1).abs()[:, 2:].max()) < 0.15
    assert float((r - torch.tensor([0.9, 0.8, 0.6, 0.3, 0.0])).abs()[:, 1:].max()) < 0.1


def test_conservation_diagnostic(mcore):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import diag_conserve as C
    from fpk.devkernel import DevConfig, EntropyDeviationKernel
    from fpk.env import DYN
    from fpk.sphere import A_EARTH
    lat = np.linspace(-78.75, 78.75, H)
    mean, std = np.asarray(contract.REFERENCE_MEAN)[DYN], np.asarray(contract.REFERENCE_STD)[DYN]
    inv = C.Invariants(contract.REFERENCE_MEAN, contract.REFERENCE_STD, lat)
    names = [contract.CHANNEL_NAMES[c] for c in DYN]
    # a state at rest at the mean, plus 100 Pa of msl and a solid-body wind u = 10 cos(phi) on every level
    x = torch.zeros(2, 66, H, W)
    cosl = torch.tensor(np.cos(np.deg2rad(lat)), dtype=torch.float32).view(H, 1)
    for i, n in enumerate(names):
        if n[0] == "u" and n[1:].isdigit():
            x[1, i] = (10.0 * cosl - mean[i]) / std[i]
            x[0, i] = -mean[i] / std[i]
        if n[0] == "v" and n[1:].isdigit():
            x[:, i] = -mean[i] / std[i]
    x[1, names.index("msl")] = 100.0 / std[names.index("msl")]
    v = inv(x)
    w = np.cos(np.deg2rad(lat))
    want = A_EARTH / C.G0 * 10.0 * (w ** 3).sum() / w.sum() * (100000.0 - 5000.0)    # sum of the dp = 950 hPa
    assert abs(float(v["mass"][1] - v["mass"][0]) - 100.0) < 0.5 and abs(float(v["aam"][0])) < 1e-6 * want
    assert abs(float(v["aam"][1]) / want - 1) < 1e-3
    ke = 0.5 * 100.0 * (w ** 3).sum() / w.sum()                                       # 1/2 u^2, area mean
    assert abs(float(v["energy"][1] - v["energy"][0]) - ke) < 0.02 * ke
    d = C.daily_change(inv, torch.stack([x[0], x[1], x[1]], 0)[None])                 # (1, 3 days) -> (1, 2)
    assert d["mass"].shape == (1, 2) and abs(float(d["mass"][0, 0]) - 100.0) < 0.5 and float(d["mass"][0, 1]) == 0
    assert abs(C.rms(d["mass"]) - 100.0 / math.sqrt(2)) < 0.5
    # the trajectories of a kernel: the members average to the mean, the mean is the core's own path
    Dk = EntropyDeviationKernel(mcore, DevConfig(pairs=2, hidden=8, blocks=(1,), smooth=3))
    xd, doy, tis = torch.randn(2, 66, H, W), torch.tensor([5, 6]), torch.zeros(2, 3, H, W)
    hist = torch.randn(2, int(mcore.mem), 66, H, W)
    memb, mn, det = C.paths(Dk, mcore, xd, doy, tis, 3, hist, torch.Generator().manual_seed(0))
    assert memb.shape == (2, 4, 4, 66, H, W) and mn.shape == det.shape == (2, 4, 66, H, W)
    assert torch.allclose(memb.mean(1), mn, atol=1e-4) and torch.allclose(mn, det, atol=1e-4)
    ch = C.daily_change(inv, memb)
    assert all(ch[k].shape == (2, 4, 3) and torch.isfinite(ch[k]).all() for k in C.KEYS)
