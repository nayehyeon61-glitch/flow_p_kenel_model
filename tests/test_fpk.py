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
