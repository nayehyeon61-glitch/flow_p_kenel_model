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


# ------------------------------------------------------------------------------- the core's flow as the path
def _core_flow(mcore, N, source="kernel", pairs=3, sigma_min=0.0, sub=0, r_bias=0.3, **kw):
    from fpk.coreflow import build_flow
    from fpk.devflow import DevFlowConfig
    from fpk.devkernel import DevConfig, EntropyDeviationKernel
    Dk = EntropyDeviationKernel(mcore, DevConfig(pairs=pairs, hidden=8, blocks=(1,), smooth=3, **kw))
    with torch.no_grad():
        Dk.r_bias.fill_(r_bias)                           # the day's end exp(r) is part of what is tested
        if kw.get("band_scale"):
            Dk.g_band.normal_(0, 0.3)
    Fm = build_flow(Dk, DevFlowConfig(hidden=8, blocks=(1,), path="core", core_steps=N, source=source,
                                      sigma_min=sigma_min, core_sub=sub))
    return Dk, Fm


def _mem_core(substeps):
    torch.manual_seed(1)
    lat = np.linspace(-78.75, 78.75, H)
    lon = np.linspace(0, 360, W, endpoint=False)
    st = static_features(lat, lon, np.random.rand(H, W).astype(np.float32), np.random.rand(H, W).astype(np.float32))
    c = make_core(st, np.full(67, 0.3), np.zeros(67), np.ones(67), lat, hidden=8, blocks=(1,), norm="gn",
                  mem=2, mem_ch=4, mz=True, substeps=substeps)
    c.pm = None
    with torch.no_grad():
        c.net.head.weight.normal_(0, 0.1)
    return c.eval()


def _day_inputs():
    g = torch.Generator().manual_seed(11)
    return (torch.randn(2, 66, H, W, generator=g), torch.tensor([5, 6]), torch.randn(2, 3, H, W, generator=g),
            torch.randn(2, 2, 66, H, W, generator=g))


@pytest.mark.parametrize("kw", [dict(noise_pre=True), dict(), dict(band_scale=True), dict(noise_mult=0.3),
                                dict(noise_pre=True, correct=True)])
def test_core_flow_with_a_zero_head_is_the_kernels_day(mcore, kw):
    """On the core's own solver (core_steps = 0: one Heun step a day on this core, as on the trained cores) the
    zero-head members are the kernel's e' for every noise mode: the noise composed with the equation
    (z_0 = e + h xi), the additive one (h xi added at the day's end, before exp(r)), band_scale, noise_mult."""
    from fpk.coreflow import CoreFlow
    xd, doy, tis, hist = _day_inputs()
    Dk, Fm = _core_flow(mcore, 0, **kw)
    assert isinstance(Fm, CoreFlow) and Fm.N == 1
    ens, ent = Fm.rollout(xd, doy, tis, 3, hist=hist, gen=torch.Generator().manual_seed(4))
    ref, ent2 = Dk.rollout(xd, doy, tis, 3, hist=hist, gen=torch.Generator().manual_seed(4))
    assert ens.shape == (2, 3, 3, 66, H, W) and torch.equal(ent, ent2)
    assert torch.allclose(ens, ref[:, :3], atol=1e-5)
    assert Fm.last_path == (0.0, 0.0) and Fm.last_split[1] == 0.0 and Fm.last_split[0] != 0.0


def test_core_flow_zero_head_on_a_finer_grid_is_the_kernel_to_second_order(mcore):
    """The two approximations of a grid of 4 steps, on the fixtures: another solver step than the core's own
    (one-step core, 4 steps: relative rms 1.5e-3 here) and the odd part around the mean path instead of the pair's
    own midpoint (4-substep core on its own grid: 8e-5 here)."""
    xd, doy, tis, hist = _day_inputs()
    rel = []
    for c in (mcore, _mem_core(4)):
        Dk, Fm = _core_flow(c, 4, noise_pre=True)
        ens, _ = Fm.rollout(xd, doy, tis, 3, hist=hist, gen=torch.Generator().manual_seed(4))
        ref, _ = Dk.rollout(xd, doy, tis, 3, hist=hist, gen=torch.Generator().manual_seed(4))
        dev = (ref[:, :3] - ref[:, 3:]) / 2
        rel.append(float((ens - ref[:, :3]).pow(2).mean().sqrt() / dev.pow(2).mean().sqrt()))
    assert 1e-5 < rel[0] < 1e-2 and rel[1] < 1e-3 and rel[1] < rel[0]


def _one_day(Dk, lead=1, seed=3):
    xd, doy, tis, hist = _day_inputs()
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        e = Dk.start(xd, doy, hist, g)
        x = xd
        for L in range(lead + 1):
            x0 = x
            x, e, Hn, _ = Dk.day(x, e, doy + L, tis[:, L], L, g)
    return x0, x, e, Hn, doy, tis[:, lead], lead


@pytest.mark.parametrize("kw,N,source,sub", [
    (dict(noise_pre=True), 4, "kernel", 0), (dict(band_scale=True), 4, "kernel", 0), (dict(), 1, "kernel", 0),
    (dict(noise_pre=True), 4, "gauss", 0), (dict(), 4, "kernel", 0), (dict(), 4, "kernel", 2),
    (dict(noise_pre=True), 2, "kernel", 3)])
def test_core_flow_target_is_exact_for_the_sampler(mcore, kw, N, source, sub):
    """An oracle that returns the training target u*_k of the path's state (whatever the sampler's own state is)
    takes the sampler from z_0 along the path onto z1: the target is the one of this sampler."""
    Dk, Fm = _core_flow(mcore, N, source=source, sub=sub, **kw)
    x0, xb, e, Hn, doy, tisr, lead = _one_day(Dk)
    c = Fm.day_start(x0, e, Hn, doy, tisr, lead, torch.Generator().manual_seed(0))
    z1 = 1.5 * torch.randn(c.z0.shape, generator=torch.Generator().manual_seed(1)) * Dk.keep
    zc = Fm.zero_path(c)
    full = lambda k: torch.full((c.z0.shape[0],), k)  # noqa: E731
    seen = []

    def oracle(z, k, tau):  # core_sub: the state after the core's step, on the line of the grid step at tau
        zk, ut = Fm.target(c, zc, z1, full(k), tau=torch.full((c.z0.shape[0],), tau) if sub else None)
        seen.append(float((z - zk).abs().max()))
        return ut
    out = Fm.run(c, oracle).flatten(0, 1)
    assert max(seen) < 1e-4 and torch.allclose(out, z1, atol=1e-4)
    assert float((zc[-1] - z1).abs().max()) > 1.0                          # ... and the zero head does not
    # at k = 0 the target is the residual itself, later the core's stretching of the part already added is taken off
    assert torch.allclose(Fm.target(c, zc, z1, full(0))[1], z1 - zc[-1], atol=1e-5)
    if N > 1:
        assert float((Fm.target(c, zc, z1, full(N - 1))[1] - (z1 - zc[-1])).abs().max()) > 1e-3
    # the day's growth of 1/2 log variance is split into the core's and the learned part, and the two add up
    var = lambda v: float(((v / c.sig) ** 2 * Dk.keep).sum() / Dk.keep.expand_as(v).sum())  # noqa: E731
    assert abs(sum(Fm.last_split) - 0.5 * math.log(var(z1) / var(c.z0))) < 1e-4 and Fm.last_split[1] != 0
    assert Fm.last_path[0] >= Fm.last_path[1] * (1 - 1e-5) > 0                # Cauchy-Schwarz: kinetic >= displacement


def test_core_flow_loss_gradients_and_mixed_steps(mcore):
    Dk, Fm = _core_flow(mcore, 4, sigma_min=0.02, noise_pre=True)
    x0, xb, e, Hn, doy, tisr, lead = _one_day(Dk)
    w = torch.ones(1, 1, H, 1)
    truth = xb + 0.3 * torch.randn(xb.shape, generator=torch.Generator().manual_seed(2))
    m_ref = mcore.m.clone()
    loss = Fm.day_loss(x0, xb, e, Hn, truth, doy, tisr, lead, w, torch.Generator().manual_seed(5), sel=torch.tensor([2, 0]))
    assert torch.isfinite(loss) and float(loss) > 0 and torch.equal(mcore.m, m_ref)   # the core's memory is put back
    loss.backward()
    assert all(p.grad is None for p in Dk.parameters()) and all(p.grad is None for p in mcore.parameters())
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in Fm.net.parameters())
    # zero head, no noise: the loss of given steps is the mean square of the targets in units of sigma, and at k = 0
    # the target is the residual of the kernel's day, (z1 - e') / sigma
    Dk, Fm = _core_flow(mcore, 4, noise_pre=True)
    x0, xb, e, Hn, doy, tisr, lead = _one_day(Dk)
    c = Fm.day_start(x0, e, Hn, doy, tisr, lead)
    z1 = ((truth - xb) / Dk.s * Dk.keep)[c.bidx]
    zc = Fm.zero_path(c)
    k = torch.tensor([0, 1, 2, 3, 3, 0])
    ut = Fm.target(c, zc, z1, k)[1]
    want = float(((ut / c.sig) ** 2 * Dk.keep).sum() / Dk.keep.expand_as(ut).sum())
    assert abs(float(Fm.day_loss(x0, xb, e, Hn, truth, doy, tisr, lead, w, k=k)) - want) < 1e-5 * want
    assert torch.allclose(ut[k == 0], (z1 - zc[-1])[k == 0], atol=1e-5)
    for j in range(4):   # the rows of a mixed draw are the rows of the same step drawn for all
        assert torch.allclose(ut[k == j], Fm.target(c, zc, z1, torch.full((6,), j))[1][k == j], atol=1e-6)
    with pytest.raises(NotImplementedError):
        Fm.transport(e, None, Hn)


def test_core_flow_rollout_and_gauss_control(mcore):
    xd, doy, tis, hist = _day_inputs()
    out = {}
    for source in ("kernel", "gauss"):
        Dk, Fm = _core_flow(mcore, 4, source=source, noise_pre=True)
        with torch.no_grad():   # a head that does something
            torch.manual_seed(0)
            [p.normal_(0, 0.05) for p in list(Fm.net.parameters())[-2:]]
        ens, ent = Fm.rollout(xd, doy, tis, 3, hist=hist, gen=torch.Generator().manual_seed(4))
        ref, ent2 = Dk.rollout(xd, doy, tis, 3, hist=hist, gen=torch.Generator().manual_seed(4))
        assert ens.shape == (2, 3, 3, 66, H, W) and torch.isfinite(ens).all()
        # the entropy is the kernel's (the control's own draws shift the kernel's random numbers after day 1)
        assert torch.allclose(ent, ent2) if source == "kernel" else torch.allclose(ent[:, 0], ent2[:, 0])
        assert all(math.isfinite(v) for v in Fm.last_path + Fm.last_split) and Fm.last_path[0] > 0
        out[source] = (ens, ref)
    # the kernel source without a learned part would be the kernel; the control starts from a Gaussian of the
    # kernel's predicted size (its own draws), carried by the same core steps
    assert float((out["kernel"][0] - out["kernel"][1][:, :3]).abs().max()) > 1e-3
    assert float((out["gauss"][0] - out["kernel"][0]).abs().max()) > 1e-2
    Dk, Fm = _core_flow(mcore, 1, source="gauss")
    x0, xb, e, Hn, doy, tisr, lead = _one_day(Dk)
    c = Fm.day_start(x0, e, Hn, doy, tisr, lead, torch.Generator().manual_seed(9))
    want = smooth_noise(tuple(e.shape), e, torch.Generator().manual_seed(9)) * Dk.keep.unsqueeze(0) * Fm.sigma(Hn)[:, None]
    assert torch.equal(c.z0, want.flatten(0, 1)) and c.add is None
    assert torch.allclose(Fm.run(c).flatten(0, 1), Fm.S(0, x0, c.z0, c.bidx, c.hours, c.tisr), atol=1e-6)


def test_deviation_flow_checkpoints_and_the_default_path(mcore, tmp_path):
    """The loaders build the flow of the checkpoint's path; a checkpoint without the key (every one so far) and the
    defaults are the straight line, whose day calls are the old calls bit for bit."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from types import SimpleNamespace
    import run_fpk as R
    from fpk.coreflow import CoreFlow, build_flow
    from fpk.devflow import DevFlowConfig, DeviationFlow
    xd, doy, tis, hist = _day_inputs()
    a = SimpleNamespace(out=str(tmp_path), core_tag="c", device="cpu")
    Dk, Fm = _core_flow(mcore, 4, noise_pre=True)
    with torch.no_grad():
        [p.normal_(0, 0.05) for p in list(Fm.net.parameters())[-2:]]
    torch.save(dict(state={k: v for k, v in Dk.state_dict().items() if not k.startswith("core.")},
                    cfg=dict(Dk.cfg.__dict__), core_tag="c"), tmp_path / "devk.pt")
    torch.save(dict(state=Fm.net.state_dict(), cfg=dict(Fm.cfg.__dict__), dev_tag="devk", core_tag="c"),
               tmp_path / "devflow_core.pt")
    F2 = R.load_devflow(a, mcore, "devflow_core", pairs=2)
    assert type(F2) is CoreFlow and F2.N == 4 and F2.cfg.path == "core" and F2.dk.cfg.pairs == 2
    Fm.dk.cfg.pairs = 2
    one = Fm.rollout(xd, doy, tis, 2, hist=hist, gen=torch.Generator().manual_seed(4))[0]
    two = F2.rollout(xd, doy, tis, 2, hist=hist, gen=torch.Generator().manual_seed(4), steps=6)[0]
    assert one.shape == (2, 2, 2, 66, H, W) and torch.equal(one, two)
    # the straight line: defaults, the old checkpoint format (no path key), the day calls
    assert DevFlowConfig().path == "line" and type(build_flow(Dk, DevFlowConfig(hidden=8, blocks=(1,)))) is DeviationFlow
    Fl = DeviationFlow(Dk, DevFlowConfig(hidden=8, blocks=(1,), steps=2))
    with torch.no_grad():
        [p.normal_(0, 0.05) for p in list(Fl.net.parameters())[-2:]]
    torch.save(dict(state=Fl.net.state_dict(), cfg=dict(hidden=8, blocks=[1], sigma_min=0.02, steps=2, source="kernel"),
                    dev_tag="devk", core_tag="c"), tmp_path / "devflow_old.pt")
    F3 = R.load_devflow(a, mcore, "devflow_old", pairs=2)
    assert type(F3) is DeviationFlow and F3.cfg.path == "line"
    got = F3.rollout(xd, doy, tis, 2, hist=hist, gen=torch.Generator().manual_seed(4))[0]
    g = torch.Generator().manual_seed(4)          # the rollout as it was written before the day calls
    with torch.no_grad():
        e = Dk.start(xd, doy, hist, g)
        x, old = xd, []
        for L in range(2):
            x, e, Hn, _ = Dk.day(x, e, doy + L, tis[:, L], L, g)
            old.append(x[:, None] + Dk.s.unsqueeze(0) * Fl.transport(e, Fl.cond(x, Hn, doy + L + 1, L + 1), Hn, g, None))
        w = torch.ones(1, 1, H, 1)
        sel = torch.tensor([1])
        l_old = Fl.loss(e[:, sel], x, x + 0.2, Fl.cond(x, Hn, doy + 2, 2), Hn, w, torch.Generator().manual_seed(8))
        l_new = Fl.day_loss(xd, x, e, Hn, x + 0.2, doy, tis[:, 1], 1, w, torch.Generator().manual_seed(8), sel)
    assert torch.equal(got, torch.stack(old, 2)) and torch.equal(l_old, l_new)
    assert float((got - x[:, None, None]).abs().max()) > 0


def _net_inputs(Fm, call):
    """The inputs the flow's network saw during ``call()`` (one tensor per network call)."""
    seen = []
    h = Fm.net.register_forward_pre_hook(lambda m, a: seen.append(a[0].detach().clone()))
    try:
        out = call()
    finally:
        h.remove()
    return seen, out


@pytest.mark.parametrize("r_bias", [-0.5, 0.3])
def test_core_flow_loss_is_on_the_scale_of_the_line_at_every_step(mcore, r_bias):
    """The kernel's end of the day (exp(r)) must not rescale the residual already added: with a zero head the loss
    of every grid step is about the line path's, (z1 - e')^2 / sigma^2, whatever r is (folded into the last step it
    was x 4.7 at r = -0.5 and about 0 at r = +0.3)."""
    from fpk.devflow import DevFlowConfig, DeviationFlow
    w = torch.ones(1, 1, H, 1)
    for kw in (dict(noise_pre=True), dict()):
        Dk, Fm = _core_flow(mcore, 4, r_bias=r_bias, **kw)
        x0, xb, e, Hn, doy, tisr, lead = _one_day(Dk)
        truth = xb + Dk.s * (torch.exp(Hn) * torch.randn(xb.shape, generator=torch.Generator().manual_seed(2)))
        Fl = DeviationFlow(Dk, DevFlowConfig(hidden=8, blocks=(1,), sigma_min=0.0))
        line = float(Fl.day_loss(x0, xb, e, Hn, truth, doy, tisr, lead, w, torch.Generator().manual_seed(0)))
        per_k = [float(Fm.day_loss(x0, xb, e, Hn, truth, doy, tisr, lead, w, k=torch.full((6,), j))) for j in range(4)]
        assert abs(per_k[0] - line) < 0.01 * line and all(abs(v - line) < 0.05 * line for v in per_k), (line, per_k)
        assert float(Dk.last["r"].abs().mean()) > 0.2


def test_core_flow_conditioning_is_the_equation_at_the_intermediate_time(mcore):
    """What the network is given at step k, in the loss and in the sampler: the mean path's state xbar(t_k) and its
    G, the time of year and the lead advanced by t_k, t_k's features, the rows of the right IC."""
    from fpk.flow import tau_features
    N = 4
    Dk, Fm = _core_flow(mcore, N, noise_pre=True)
    with torch.no_grad():
        [p.normal_(0, 0.05) for p in list(Fm.net.parameters())[-2:]]
    x0, xb, e, Hn, doy, tisr, lead = _one_day(Dk)
    hours = (doy + lead).float() * 24.0
    xs = [x0]
    with torch.no_grad():
        for k in range(N - 1):                     # the core's own steps over 1 / N day, the memory of the day
            xs.append(mcore.substep(xs[-1], hours, k, N))
    want = [Fm.cond(xs[k], Hn, doy + lead + k / N, lead + k / N) for k in range(N)]
    assert float((want[0] - want[1]).abs().max()) > 1e-3 and float((want[0][0] - want[0][1]).abs().max()) > 1e-3
    assert float((xs[1] - x0).abs().max()) > 1e-4 and want[0].shape[1] > 66 + 66   # G(xbar) is among the channels
    assert torch.allclose(want[2][:, -1], torch.full((2, H, W), (lead + 2 / N) / 10))
    tf = lambda t, n: tau_features(torch.full((n,), t), H, W)  # noqa: E731
    w, truth = torch.ones(1, 1, H, 1), xb + 0.3
    sel = torch.tensor([2, 0])
    bidx = torch.tensor([0, 0, 1, 1])
    for j in range(N):
        seen, _ = _net_inputs(Fm, lambda: Fm.day_loss(x0, xb, e, Hn, truth, doy, tisr, lead, w, sel=sel, k=torch.full((4,), j)))
        assert len(seen) == 1 and torch.allclose(seen[0][:, 66:-4], want[j][bidx], atol=1e-6)
        assert torch.allclose(seen[0][:, -4:], tf(j / N, 4))
    # mixed steps: every row its own step's conditioning
    k = torch.tensor([3, 0, 1, 2])
    seen, _ = _net_inputs(Fm, lambda: Fm.day_loss(x0, xb, e, Hn, truth, doy, tisr, lead, w, sel=sel, k=k))
    assert torch.allclose(seen[0][:, 66:-4], torch.stack([want[int(k[i])][bidx[i]] for i in range(4)]), atol=1e-6)
    assert torch.allclose(seen[0][:, -4:], torch.cat([tf(int(q) / N, 1) for q in k]))
    # the sampler: N calls, the same conditioning, and its state at step 0 is the source
    b3 = torch.arange(2).repeat_interleave(3)
    seen, z = _net_inputs(Fm, lambda: Fm.day_transport(x0, xb, e, Hn, doy, tisr, lead))
    assert len(seen) == N
    for j in range(N):
        assert torch.allclose(seen[j][:, 66:-4], want[j][b3], atol=1e-6) and torch.allclose(seen[j][:, -4:], tf(j / N, 6))
    sig = Fm.sigma(Hn)[b3]
    assert torch.allclose(seen[0][:, :66] * sig, Dk.last["zin"].flatten(0, 1), atol=1e-5)
    # core_sub = 2: Heun on the line of every grid step, times t_k + tau / N, the conditioning of t_k
    Fm.cfg.core_sub = 2
    seen, _ = _net_inputs(Fm, lambda: Fm.day_transport(x0, xb, e, Hn, doy, tisr, lead))
    times = [float(q[0, -4, 0, 0]) for q in seen]
    assert len(seen) == 4 * N and np.allclose(times, [(j + t) / N for j in range(N) for t in (0, .5, .5, 1)])
    assert all(torch.allclose(seen[4 * j + i][:, 66:-4], want[j][b3], atol=1e-6) for j in range(N) for i in range(4))
    r = Fm.day_rows(Fm.day_start(x0, e, Hn, doy, tisr, lead), xb, truth, k=torch.full((6,), 2), tau=torch.full((6,), 0.25))
    seen, _ = _net_inputs(Fm, lambda: Fm.day_loss(x0, xb, e, Hn, truth, doy, tisr, lead, w, k=torch.full((6,), 2),
                                                  tau=torch.full((6,), 0.25)))
    assert torch.allclose(seen[0][:, -4:], tf(2.25 / N, 6)) and torch.allclose(seen[0][:, 66:-4], want[2][b3], atol=1e-6)
    assert torch.allclose(seen[0][:, :66] * sig, r.z, atol=1e-5) and torch.allclose(r.t, torch.full((6,), 2.25 / N))


@pytest.mark.parametrize("sub", [0, 2])
def test_core_flow_sigma_min_noise(mcore, sub):
    """sigma_min: noise of sigma_min sigma on the network's input. core_sub = 0: the target leads from the PERTURBED
    z_k to the path's z_(k+1) (exact for the sampler from that state); core_sub >= 1: noise on the line of the grid
    step, the target is the line's (as in DeviationFlow)."""
    N = 4
    Dk, Fm = _core_flow(mcore, N, sigma_min=0.05, sub=sub)
    x0, xb, e, Hn, doy, tisr, lead = _one_day(Dk)
    truth, w = xb + 0.3, torch.ones(1, 1, H, 1)
    k, tau = torch.tensor([0, 1, 2, 3, 3, 0]), torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 0.5])
    c = Fm.day_start(x0, e, Hn, doy, tisr, lead)
    r = Fm.day_rows(c, xb, truth, torch.Generator().manual_seed(5), k, tau)
    zc = Fm.zero_path(c)
    clean, ut0 = Fm.target(c, zc, r.z1, k, tau=tau if sub else None)
    kp = Dk.keep.expand_as(r.z) > 0
    n = ((r.z - clean) / c.sig)[kp]
    assert 0.04 < float(n.std()) < 0.06 and float((r.z - clean)[~kp].abs().max()) == 0
    nxt = torch.stack([zc[int(q) + 1][i] + (int(q) + 1) / N * (r.z1 - zc[N])[i] for i, q in enumerate(k)])   # z_(k+1)^path
    if sub:
        assert torch.equal(r.ut, ut0)
    else:
        Tz = torch.stack([Fm.T(c, int(q), r.z[i:i + 1], torch.tensor([i]))[0] for i, q in enumerate(k)])
        assert torch.allclose(Tz + r.ut / N, nxt, atol=1e-5) and float((r.ut - ut0).abs().max()) > 1e-3
    # the loss: the network sees that perturbed state, and (zero head) the loss is the mean square of that target
    seen, loss = _net_inputs(Fm, lambda: Fm.day_loss(x0, xb, e, Hn, truth, doy, tisr, lead, w,
                                                     torch.Generator().manual_seed(5), k=k, tau=tau))
    assert torch.allclose(seen[0][:, :66] * c.sig, r.z, atol=1e-5)
    want = float(((r.ut / c.sig) ** 2 * Dk.keep).sum() / Dk.keep.expand_as(r.ut).sum())
    assert abs(float(loss) - want) < 1e-5 * want


@pytest.mark.parametrize("source", ["kernel", "gauss"])
def test_core_flow_day_end_offset_and_source_rows(mcore, source):
    """An additive kernel on a 4-step grid with selected pairs: the source rows are the selected pairs' deviations
    that entered the equation, the offset b = e' - p of the same pairs enters once, after the LAST core step, and
    the zero head is the kernel's e' up to the other solver step."""
    N, sel = 4, torch.tensor([2, 0])
    Dk, Fm = _core_flow(mcore, N, source=source)
    x0, xb, e, Hn, doy, tisr, lead = _one_day(Dk)
    c = Fm.day_start(x0, e, Hn, doy, tisr, lead, torch.Generator().manual_seed(3), sel)
    assert c.z0.shape[0] == 4 and c.P == 2 and c.bidx.tolist() == [0, 0, 1, 1]
    zc = Fm.zero_path(c)
    chain = c.z0
    for k in range(N):
        chain = Fm.S(k, c.xs[k], chain, c.bidx, c.hours, c.tisr)
    if source == "gauss":
        assert c.add is None and torch.allclose(zc[N], chain, atol=1e-6)
        return
    b = (e - Dk.last["p"])[:, sel].flatten(0, 1)
    assert float(Dk.last["hxi"].abs().max()) > 1e-2 and float((b[0] - b[1]).abs().max()) > 1e-2
    assert torch.equal(c.z0, Dk.last["zin"][:, sel].flatten(0, 1)) and torch.equal(c.add, b)
    assert torch.allclose(zc[N], chain + b, atol=1e-6)
    for k in range(1, N):      # ... and not before the last step
        assert torch.allclose(zc[k], Fm.S(k - 1, c.xs[k - 1], zc[k - 1], c.bidx, c.hours, c.tisr), atol=1e-6)
    ek = e[:, sel].flatten(0, 1)
    assert float((zc[N] - ek).pow(2).mean().sqrt() / ek.pow(2).mean().sqrt()) < 1e-2
    # the rows of a selection are the rows of the full day
    cf = Fm.day_start(x0, e, Hn, doy, tisr, lead)
    full = Fm.zero_path(cf)[N].view(2, 3, 66, H, W)
    assert torch.allclose(zc[N], full[:, sel].flatten(0, 1), atol=1e-6)


def test_core_flow_one_euler_step_is_under_dispersed_and_the_sub_flow_is_not(core):
    """Independent Gaussian source and target of one variance on a core with zero tendency (S_k = identity): the
    optimum of the regression is u(z, t) = z (2 t - 1) / ((1 - t)^2 + t^2). One Euler step a grid point (core_sub = 0)
    gives members with (1 - 1/4)(1 - 0.8/4)(1 + 0.8/4) = 0.72 of the target's spread at N = 4; the line flow of every
    grid step (core_sub = 3, Heun) gives the target's spread."""
    from fpk.coreflow import build_flow
    from fpk.devflow import DevFlowConfig
    from fpk.devkernel import DevConfig, EntropyDeviationKernel
    N, P = 4, 48
    Dk = EntropyDeviationKernel(core, DevConfig(pairs=P, hidden=8, blocks=(1,), smooth=3))
    x0, Hn, doy = torch.randn(2, 66, H, W), torch.zeros(2, 66, H, W), torch.tensor([5, 6])
    e = torch.zeros(2, P, 66, H, W)
    z1 = (smooth_noise(tuple(e.shape), e, torch.Generator().manual_seed(1)) * Dk.keep.unsqueeze(0)).flatten(0, 1)
    kp = Dk.keep.expand_as(z1) > 0
    coef = lambda t: (2 * t - 1) / ((1 - t) ** 2 + t ** 2)  # noqa: E731
    ratio = {}
    for sub in (0, 3):
        Fm = build_flow(Dk, DevFlowConfig(hidden=8, blocks=(1,), path="core", core_steps=N, source="gauss", core_sub=sub))
        c = Fm.day_start(x0, e, Hn, doy, None, 0, torch.Generator().manual_seed(0))
        zc = Fm.zero_path(c)
        assert torch.allclose(zc[N], c.z0, atol=1e-6)
        # the implementation's own target on its own path has that optimum (least squares of a pointwise slope)
        for k, tau in ((1, 0.0), (3, 0.5 if sub else 0.0)):
            zk, ut = Fm.target(c, zc, z1, torch.full((2 * P,), k), tau=torch.full((2 * P,), tau) if sub else None)
            assert abs(float((ut[kp] * zk[kp]).sum() / (zk[kp] ** 2).sum()) - coef((k + tau) / N)) < 0.05
        out = Fm.run(c, lambda z, k, tau: coef((k + tau) / N) * z).flatten(0, 1)
        ratio[sub] = float(out[kp].std() / z1[kp].std())
    assert 0.68 < ratio[0] < 0.76 and 0.96 < ratio[3] < 1.04, ratio


@pytest.mark.parametrize("path,sub", [("core", 2), ("core", 0), ("line", 0)])
def test_devflow_trainer_wiring(mcore, tmp_path, monkeypatch, capsys, path, sub):
    """``devflow --path/--core-steps/--core-sub`` through the trainer itself: the flow it builds, the day calls it
    makes (the mean BEFORE the day, the IC's day of year, the lead), and the checkpoint it writes and reloads."""
    import json
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from types import SimpleNamespace
    import run_fpk as R
    from fpk.coreflow import CoreFlow
    from fpk.devflow import DeviationFlow
    from fpk.devkernel import DevConfig, EntropyDeviationKernel
    Dk = EntropyDeviationKernel(mcore, DevConfig(pairs=3, hidden=8, blocks=(1,), smooth=3, noise_pre=True))
    torch.save(dict(state={k: v for k, v in Dk.state_dict().items() if not k.startswith("core.")},
                    cfg=dict(Dk.cfg.__dict__), core_tag="c"), tmp_path / "devk.pt")
    rng = np.random.default_rng(0)
    lat = np.linspace(-78.75, 78.75, H)
    monkeypatch.setattr(R, "load_core", lambda a: (None, SimpleNamespace(lat=lat), mcore))
    monkeypatch.setattr(R.D, "load", lambda pa, years: {y: rng.standard_normal((12, 67, H, W)).astype(np.float32) for y in years})
    calls, built = [], []
    real = R.build_flow

    def build(dk, cfg, gop=None):
        Fm = real(dk, cfg)
        inner = Fm.day_loss

        def day_loss(x0, xb, e, Hn, truth, doy0, tisr, lead, w, gen=None, sel=None):
            calls.append((x0, xb, doy0.clone(), lead, None if sel is None else len(sel), tisr, dk.core.fc.clone()))
            return inner(x0, xb, e, Hn, truth, doy0, tisr, lead, w, gen, sel)
        Fm.day_loss = day_loss
        built.append(Fm)
        return Fm
    monkeypatch.setattr(R, "build_flow", build)
    argv = ["devflow", "--out", str(tmp_path), "--core-tag", "c", "--tag", "t", "--dev-tag", "devk", "--days", "2",
            "--iters", "1", "--batch", "2", "--sources", "1", "--val-batches", "1", "--hidden", "8", "--steps", "2"]
    if path == "core":
        argv += ["--path", "core", "--core-steps", "2", "--core-sub", str(sub)]
    R.main(argv)
    Fm = built[0]
    assert type(Fm) is (CoreFlow if path == "core" else DeviationFlow) and Fm.cfg.path == path and Fm.cfg.core_sub == sub
    assert path == "line" or Fm.N == 2
    logs = [json.loads(q) for q in capsys.readouterr().out.strip().splitlines()]
    assert all(math.isfinite(v) for q in logs for v in q.get("devflow_val_start", []) + q.get("val", []))
    assert ("path" in logs[0]) == (path == "core")
    # validation at the start (1 batch), the training iteration, the validation of the log: 2 leads each
    assert [q[3] for q in calls] == [0, 1] * 3 and [q[4] for q in calls] == [None, None, 1, 1, None, None]
    for a0, a1 in zip(calls[0::2], calls[1::2]):
        assert torch.equal(a0[2], a1[2])                      # the IC's day of year at every lead
        assert a1[0] is a0[1] and not torch.equal(a0[0], a0[1])  # the day starts from the mean before it
        assert torch.equal(a0[1], a0[6]) and torch.equal(a1[1], a1[6])   # xb = the mean right after the kernel's day
        assert a0[5].shape == (2, H, W) and not torch.equal(a0[5], a1[5])
    ck = torch.load(tmp_path / "t.pt", weights_only=False)
    if path == "core":
        assert ck["cfg"]["path"] == "core" and ck["cfg"]["core_steps"] == 2 and ck["cfg"]["core_sub"] == sub
    else:   # the straight line's checkpoint keeps the format it always had
        assert sorted(ck["cfg"]) == ["blocks", "hidden", "sigma_min", "source", "steps"]
    monkeypatch.setattr(R, "build_flow", real)
    F2 = R.load_devflow(SimpleNamespace(out=str(tmp_path), core_tag="c", device="cpu"), mcore, "t")
    assert type(F2) is type(Fm) and F2.cfg == Fm.cfg
    assert all(torch.equal(a_, b_) for a_, b_ in zip(F2.net.state_dict().values(), Fm.net.state_dict().values()))


# ------------------------------------------------- the fair energy score, its fine-tuning (devtune) and reflow
def test_fair_energy_score_is_unbiased_and_proper():
    """Members N(0, a^2 I) and truth N(0, I) in d = 12 dimensions with |f| = |f|_2 / sqrt(d): E|X - y| =
    c sqrt(1 + a^2) and E|X - X'| = c sqrt(2) a, c = sqrt(2) Gamma((d + 1) / 2) / (Gamma(d / 2) sqrt(d)), so
    ES(a) = c (sqrt(1 + a^2) - a / sqrt(2)): 0.707 c at the truth's a = 1, 0.765 c at a = 1/2, 0.822 c at a = 2.
    The fair score has that mean for M = 2, 4, 8; the form with 1 / (2 M^2) is high by E|X - X'| / (2 M) and
    prefers the too narrow ensemble at M = 2."""
    from fpk.escore import GridNorm, energy_score
    d = 12
    c = math.sqrt(2) * math.exp(math.lgamma((d + 1) / 2) - math.lgamma(d / 2)) / math.sqrt(d)
    nrm = GridNorm(torch.ones(3), torch.ones(2))
    g = torch.Generator().manual_seed(0)
    es = {}
    for M in (2, 4, 8):
        for a in (0.5, 1.0, 2.0):
            x, y = a * torch.randn(40000, M, 3, 2, 2, generator=g), torch.randn(40000, 3, 2, 2, generator=g)
            fair, d1, d2 = energy_score(x, y, nrm)
            es[M, a] = float(fair.mean()), float((d1 - (M - 1) / M * d2).mean())
            assert abs(es[M, a][0] - c * (math.sqrt(1 + a * a) - a / math.sqrt(2))) < 0.01, (M, a, es[M, a])
            assert abs(float(d1.mean()) - c * math.sqrt(1 + a * a)) < 0.01 and abs(float(d2.mean()) - c * a / math.sqrt(2)) < 0.01
        assert es[M, 1.0][0] < es[M, 0.5][0] - 0.03 and es[M, 1.0][0] < es[M, 2.0][0] - 0.08
    assert abs(es[2, 1.0][1] - es[2, 1.0][0] - c / (2 * math.sqrt(2))) < 0.01 and es[2, 0.5][1] < es[2, 1.0][1] - 0.05
    with pytest.raises(ValueError):
        energy_score(torch.zeros(2, 1, 3, 2, 2), torch.zeros(2, 3, 2, 2), nrm)
    # eps inside the root: two members that coincide have a finite gradient (the plain root has none)
    for eps, finite in ((1e-12, True), (0.0, False)):
        x = torch.randn(2, 3, 3, 2, 2, generator=g)
        x[:, 1] = x[:, 0]
        x.requires_grad_(True)
        energy_score(x, torch.zeros(2, 3, 2, 2), GridNorm(torch.ones(3), torch.ones(2), eps))[0].sum().backward()
        assert bool(torch.isfinite(x.grad).all()) == finite


def test_spread_skill_of_a_calibrated_ensemble():
    from fpk.escore import GridNorm, spread_skill
    g = torch.Generator().manual_seed(1)
    M = 4
    x, y = torch.randn(3000, M, 3, 2, 2, generator=g), torch.randn(3000, 3, 2, 2, generator=g)
    sp, sk = spread_skill(x, y, GridNorm(torch.ones(3), torch.ones(2)))
    assert sp.shape == sk.shape == (3000,) and abs(float(sp.mean()) - 1.0) < 0.02       # the unbiased variance
    assert abs(math.sqrt((M + 1) / M * float(sp.mean()) / float(sk.mean())) - 1.0) < 0.02


def test_energy_score_of_antithetic_pairs_is_unbiased():
    """The kernel's members [e, -e] are P antithetic pairs. With the spread term over members of different pairs
    (pairs=True) the score has the mean of the fair score of independent members, P = 2 and 4: against the closed
    form of the Gaussian case (members N(0, a^2 I), truth N(0, I), d = 12) and, for a law that is not Gaussian
    (Laplace members, a uniform truth), against a large-sample reference, the fair score of 2 independent members.
    The fair form over ALL pairs of members (diag_spectra's es_version 1) is low by c a (2 - sqrt 2) / (2 (2 P - 1)):
    14 % and 6 % of the calibrated score at P = 2 and 4, and the more the wider the ensemble, so that at P = 2 it
    hardly charges an ensemble twice too wide."""
    from fpk.escore import GridNorm, energy_score
    d, n = 12, 60000
    c = math.sqrt(2) * math.exp(math.lgamma((d + 1) / 2) - math.lgamma(d / 2)) / math.sqrt(d)
    true = lambda a: c * (math.sqrt(1 + a * a) - a / math.sqrt(2))  # noqa: E731
    nrm = GridNorm(torch.ones(3), torch.ones(2))
    g = torch.Generator().manual_seed(0)
    lap = lambda *sh: (lambda u: -torch.sign(u) * (1 - 2 * u.abs()).clamp_min(1e-12).log())(torch.rand(*sh, generator=g) - 0.5)  # noqa: E731
    uni = lambda *sh: 3.0 * (torch.rand(*sh, generator=g) - 0.5)  # noqa: E731
    ref = float(energy_score(lap(300000, 2, 3, 2, 2), uni(300000, 3, 2, 2), nrm)[0].mean())
    v1 = {}
    for P in (2, 4):
        for a in (1.0, 2.0):
            e, y = a * torch.randn(n, P, 3, 2, 2, generator=g), torch.randn(n, 3, 2, 2, generator=g)
            mem = torch.cat([e, -e], 1)
            es, d1, d2 = energy_score(mem, y, nrm, pairs=True)
            assert abs(float(es.mean()) - true(a)) < 0.004 and abs(float(d2.mean()) - c * a / math.sqrt(2)) < 0.004, (P, a)
            assert abs(float(d1.mean()) - c * math.sqrt(1 + a * a)) < 0.004
            v1[P, a] = float(energy_score(mem, y, nrm)[0].mean())
            assert abs(v1[P, a] - true(a) + c * a * (2 - math.sqrt(2)) / (2 * (2 * P - 1))) < 0.004, (P, a, v1[P, a])
        # by hand on a few rows: the mean over the ordered pairs of members that are not partners
        M, x, yy = 2 * P, mem[:40], y[:40]
        cross = sum(nrm(x[:, i] - x[:, j]) for i in range(M) for j in range(M) if i % P != j % P) / (M * (M - 2))
        assert torch.allclose(energy_score(x, yy, nrm, pairs=True)[0], nrm(x - yy[:, None]).mean(1) - 0.5 * cross, atol=1e-5)
        # not Gaussian: Laplace members and their mirrors against a uniform truth
        e, y = lap(n, P, 3, 2, 2), uni(n, 3, 2, 2)
        mem = torch.cat([e, -e], 1)
        assert abs(float(energy_score(mem, y, nrm, pairs=True)[0].mean()) - ref) < 0.006, (P, ref)
        assert float(energy_score(mem, y, nrm)[0].mean()) < ref - 0.02
    assert v1[2, 1.0] < 0.88 * true(1.0) and v1[4, 1.0] < 0.95 * true(1.0)
    assert v1[2, 2.0] - v1[2, 1.0] < 0.25 * (true(2.0) - true(1.0))        # version 1 hardly sees the too wide one
    for bad in (mem[:, :3], mem[:, :2]):                                     # an odd number of members, one pair
        with pytest.raises(ValueError):
            energy_score(bad, y, nrm, pairs=True)


def test_band_norm_is_the_norm_of_diag_spectra(tmp_path):
    """The shared implementation gives the numbers of scripts/diag_spectra.py as it computed them itself (the lines
    of the script before the norm was shared, on the same fields), bit for bit."""
    import json
    from fpk.env import TISR_DYN
    from fpk.escore import BandNorm, band_power, energy_score, make_norm
    from fpk.sphere import SHT, bands
    lat = np.linspace(-78.75, 78.75, H)
    S = SHT(lat)
    g = torch.Generator().manual_seed(0)
    keep = torch.ones(1, 66, 1, 1)
    keep[0, TISR_DYN] = 0
    B, M = 3, 4
    mem, dev = torch.randn(B, M, 66, H, W, generator=g), 1.5 * torch.randn(B, 66, H, W, generator=g)
    an = torch.randn(20, 66, H, W, generator=g)
    # ---- scripts/diag_spectra.py at c6d6cd6 (cpw from the anomalies an, the members mem, the error dev)
    bj0, nb0 = bands(S.L)
    band_pw = lambda f: torch.stack([S.spectrum(f)[..., bj0 == b].sum(-1) for b in range(nb0)], -1)  # noqa: E731
    cpw = band_pw(an).mean(0).clamp(min=1e-8)
    nj0 = torch.tensor([float((2 * torch.arange(S.L)[bj0 == b] + 1).sum()) for b in range(nb0)])
    wES = (1.0 / (nj0 * cpw / nj0.sum())) * (keep.reshape(-1, 1) > 0) / 65.0 / nb0
    er = dev[:, None]
    nrm = lambda f: (band_pw(f) * wES).sum((-1, -2)).sqrt()  # noqa: E731
    d1 = nrm(mem - er).mean(1)
    d2 = sum(nrm(mem - mem.roll(q, 1)).mean(1) for q in range(1, M)) / (M - 1)
    old = [float(d1.sum()), float(0.5 * d2.sum()), float((d1 - 0.5 * d2).sum())]
    # ---- the shared code, as the script calls it now
    N = BandNorm(S, band_power(S, an).mean(0).clamp(min=1e-8), keep)
    fair, e1, e2 = energy_score(mem, dev, N)
    assert [float(e1.sum()), float(e2.sum()), float(fair.sum())] == old
    assert torch.equal(N.w, wES) and torch.equal(e1, d1) and torch.equal(fair, d1 - 0.5 * d2)
    # the fair score is the mean over the pairs i != j
    pair = torch.stack([N(mem[:, i] - mem[:, j]) for i in range(M) for j in range(M) if i != j], 1)
    assert torch.allclose(fair, e1 - pair.sum(1) / (2 * M * (M - 1)), atol=1e-5)
    # the bands' powers add up to the area mean of the square (a field inside the truncation; 8 latitudes: to 6 %)
    sm = S.synthesis(S.analysis(mem[0, 0]))
    cosw = torch.tensor(np.cos(np.deg2rad(lat)) / np.cos(np.deg2rad(lat)).sum(), dtype=torch.float32).view(H, 1)
    assert torch.allclose(band_power(S, sm).sum(-1), (sm.pow(2) * cosw).sum(-2).mean(-1), rtol=0.06)
    # from a clim_bands json (devtune --norm band --clim), and without the band l = 0
    f = tmp_path / "clim.json"
    f.write_text(json.dumps(dict(c=cpw.tolist())))
    assert torch.equal(make_norm("band", keep, lat, clim=str(f), eps=0.0)(mem), N(mem))
    one = torch.ones(66, H, W)
    assert float(N(one)) > 1.0 and float(make_norm("band", keep, lat, clim=str(f), bmin=1)(one)) < 1e-3 * float(N(one))
    with pytest.raises(ValueError):
        make_norm("band", keep, lat)
    with pytest.raises(ValueError):
        BandNorm(S, cpw[:, :-1], keep)
    # differentiable, no gradient to tisr
    x = mem.clone().requires_grad_(True)
    energy_score(x, dev, make_norm("band", keep, lat, clim=str(f)))[0].sum().backward()
    assert torch.isfinite(x.grad).all() and float(x.grad[:, :, TISR_DYN].abs().max()) == 0 and float(x.grad.abs().sum()) > 0


def test_energy_norm_by_hand_and_no_gradient_to_the_mass_field():
    """|f|^2 = < sum_k (dp_k / g) [1/2 (u'^2 + v'^2) + c_p / (2 T_r) T'^2 + L^2 / (2 c_p T_r) q'^2]
    + R T_r / (2 g p_r) p_s'^2 >, x' = std s f: single channels by hand (dp = 100 hPa at 500 hPa, half layers of
    37.5 and 25 hPa at 1000 and 50 hPa), the area mean, and zero weight (and gradient) on z, msl, t2m, skt, tcwv."""
    from fpk.env import DYN
    from fpk.escore import CP, G0, LV, P_REF, R_DRY, T_REF, EnergyNorm, make_norm
    assert (CP, G0, LV, P_REF, R_DRY, T_REF) == (1004.0, 9.80665, 2.5e6, 1.0e5, 287.05, 270.0)
    names = [contract.CHANNEL_NAMES[c] for c in DYN]
    lat = np.linspace(-78.75, 78.75, H)
    g = torch.Generator().manual_seed(0)
    s, std = 0.1 + torch.rand(1, 66, 1, 1, generator=g), 0.5 + 10 * torch.rand(67, generator=g).numpy().astype(np.float64)
    E = EnergyNorm(s, std, lat)
    unit = lambda n: float(std[DYN][names.index(n)]) * float(s.reshape(-1)[names.index(n)])  # noqa: E731

    def one(n, amp=1.0):
        f = torch.zeros(66, H, W)
        f[names.index(n)] = amp
        return f
    by_hand = {"u500": 1.0e4 / 9.80665 * 0.5, "v500": 1.0e4 / 9.80665 * 0.5, "t1000": 3750.0 / 9.80665 * 1004.0 / 540.0,
               "q50": 2500.0 / 9.80665 * 2.5e6 ** 2 / (2 * 1004.0 * 270.0), "sp": 287.05 * 270.0 / (2 * 9.80665 * 1.0e5)}
    for n, a in by_hand.items():
        assert abs(float(E.sq(one(n, 2.0))) / (a * (2.0 * unit(n)) ** 2) - 1) < 1e-5, n
    # two channels add; the area mean: u = cos(lat) has the mean square sum cos^3 / sum cos
    w = np.cos(np.deg2rad(lat))
    f = one("u250") * torch.tensor(w, dtype=torch.float32).view(H, 1) + one("t850", 3.0)
    want = 5000.0 / 9.80665 * 0.5 * unit("u250") ** 2 * (w ** 3).sum() / w.sum() + 11250.0 / 9.80665 * 1004.0 / 540.0 * (3.0 * unit("t850")) ** 2
    assert abs(float(E.sq(f)) / want - 1) < 1e-5 and abs(float(E(f)) / math.sqrt(want) - 1) < 1e-5
    zero = [n for n in names if (n[0] == "z" and n[1:].isdigit()) or n in ("t2m", "msl", "skt", "tcwv", "tisr")]
    assert len(zero) == 17 and int((E.a > 0).sum()) == 49 and all(float(E.a[names.index(n)]) == 0 for n in zero)
    assert float(E.sq(sum(one(n, 5.0) for n in zero))) == 0
    x = torch.randn(2, 3, 66, H, W, generator=g).requires_grad_(True)
    N = make_norm("energy", torch.ones(66), lat, s=s, std=std)
    assert torch.allclose(N(x), (E.sq(x) + 1e-12).sqrt()) and N(x).shape == (2, 3)
    N(x).sum().backward()
    zi = torch.tensor([names.index(n) for n in zero])
    oth = torch.tensor([i for i in range(66) if i not in zi.tolist()])
    assert float(x.grad[:, :, zi].abs().max()) == 0 and float(x.grad[:, :, oth].abs().amax((0, 1, 3, 4)).min()) > 0


def _flat_grad(loss, params):
    g = torch.autograd.grad(loss, params, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if q is None else q).reshape(-1) for p, q in zip(params, g)])


def _fd(loss, params, d, h=1e-3):
    """The central difference of ``loss()`` along the unit direction d / |d| of the parameters, and that direction."""
    d, saved = d / d.norm(), [p.detach().clone() for p in params]
    out = []
    for a in (h, -h):
        i = 0
        with torch.no_grad():
            for p, q in zip(params, saved):
                p.copy_(q + a * d[i:i + p.numel()].view_as(p))
                i += p.numel()
            out.append(float(loss()))
    with torch.no_grad():
        for p, q in zip(params, saved):
            p.copy_(q)
    return (out[0] - out[1]) / (2 * h), d


def _flow_with_a_head(Dk, std=0.05, seed=0, **kw):
    from fpk.coreflow import build_flow
    from fpk.devflow import DevFlowConfig
    Fm = build_flow(Dk, DevFlowConfig(**{**dict(hidden=8, blocks=(1,)), **kw}))
    torch.manual_seed(seed)
    with torch.no_grad():
        [p.normal_(0, std) for p in list(Fm.net.parameters())[-2:]]
    return Fm


def _kernel(mcore, pairs=3, **kw):
    from fpk.devkernel import DevConfig, EntropyDeviationKernel
    return EntropyDeviationKernel(mcore, DevConfig(pairs=pairs, hidden=8, blocks=(1,), smooth=3, **(kw or dict(noise_pre=True))))


@pytest.mark.parametrize("which", ["line", "gauss", "core", "core_sub"])
def test_sampler_with_the_gradient_is_the_sampler(mcore, which):
    """grad=True: the members of the sampler without a graph, bit for bit, with a graph that reaches the flow's
    network and nothing else (kernel and core stay without gradient); inside no_grad no graph is built."""
    kw = dict(line=dict(steps=3), gauss=dict(steps=3, source="gauss"), core=dict(path="core", core_steps=4),
              core_sub=dict(path="core", core_steps=2, core_sub=2))[which]
    Dk = _kernel(mcore)
    Fm = _flow_with_a_head(Dk, **kw)
    x0, xb, e, Hn, doy, tisr, lead = _one_day(Dk)
    gen = lambda: torch.Generator().manual_seed(3)  # noqa: E731
    ref = Fm.day_transport(x0, xb, e, Hn, doy, tisr, lead, gen())
    stat = (Fm.last_path, getattr(Fm, "last_split", None))
    out = Fm.day_transport(x0, xb, e, Hn, doy, tisr, lead, gen(), grad=True)
    assert out.requires_grad and not ref.requires_grad and torch.equal(out.detach(), ref)
    assert (Fm.last_path, getattr(Fm, "last_split", None)) == stat and stat[0][0] > 0
    # the two calls: the day's source rows, then runs from them (all pairs, or a choice of them)
    c = Fm.day_ctx(x0, xb, e, Hn, doy, tisr, lead, gen())
    assert torch.equal(Fm.day_run(c), ref) and torch.equal(Fm.day_run(c, grad=True).detach(), ref)
    sel = torch.tensor([2, 0])
    part = Fm.day_run(Fm.day_ctx(x0, xb, e, Hn, doy, tisr, lead, gen(), sel))
    assert part.shape == (2, 2, 66, H, W) and (which == "gauss" or torch.allclose(part, ref[:, sel], atol=1e-5))
    if which.startswith("core"):   # the core's own Jacobian in the graph: the same members
        assert torch.allclose(Fm.day_run(c, grad=True, jac=True).detach(), ref, atol=1e-6)
    with torch.no_grad():
        assert not Fm.day_run(c, grad=True).requires_grad
    out.pow(2).mean().backward()
    assert all(p.grad is None for p in Dk.parameters()) and all(p.grad is None for p in mcore.parameters())
    gr = [p.grad for p in Fm.net.parameters() if p.grad is not None]
    assert gr and all(torch.isfinite(q).all() for q in gr) and sum(float(q.abs().sum()) for q in gr) > 0


def _sampler_by_hand(Fm, c, jac=False):
    """The sampler of the day's context c written out, every operation in the graph: the Heun steps of the line;
    on the core path the day's grid with the core's own Jacobian of its steps (jac) or the identity for it."""
    full = lambda v: torch.full((c.z0.shape[0],), float(v))  # noqa: E731
    if Fm.cfg.path == "line":
        z, dt = c.z0, 1.0 / Fm.cfg.steps
        for i in range(Fm.cfg.steps):
            k1 = Fm.velocity(z, full(i * dt), c.cd, c.sig)
            k2 = Fm.velocity(z + dt * k1, full(i * dt + dt), c.cd, c.sig)
            z = z + dt * 0.5 * (k1 + k2)
        return z.view(c.B, c.P, *z.shape[1:])
    N, M = Fm.N, Fm.cfg.core_sub
    u = lambda y, k, tau: Fm.velocity(y, full((k + tau) / N), Fm.cond_k(c, k)[c.bidx], c.sig)  # noqa: E731
    z = c.z0
    for k in range(N):
        a = Fm.T(c, k, z, grad=True) if (jac or not z.requires_grad) else Fm.T(c, k, z.detach()) + (z - z.detach())
        if M == 0:
            z = a + u(z, k, 0.0) / N
            continue
        z, h = a, 1.0 / (N * M)
        for j in range(M):
            k1 = u(z, k, j / M)
            z = z + h * 0.5 * (k1 + u(z + h * k1, k, (j + 1) / M))
    return z.view(c.B, c.P, *z.shape[1:])


def test_sampler_gradient_exact_on_the_line_and_straight_through_on_the_core_path(mcore):
    """The gradient of the energy score of the sampled members.

    Exact (line; core path with jac): autograd through ``day_run`` is the gradient of the sampler written out by
    hand with every operation in the graph (1e-4 of its norm: a Jacobian dropped inside a Heun step, such as
    k1.detach() in the corrector, is 7 % and 4 % of the norm with this head, std 0.3; with the head of std 0.05
    it is 1.5 % and nearly orthogonal to the gradient, which a finite difference along the gradient alone does not
    see), and finite differences agree along the gradient (1 %) and along three random directions.

    Core path, default: the identity for the Jacobian of the core steps AFTER an increment, measured against the
    gradient with the core's Jacobian (jac). On this fixture the core's steps change a deviation by 7 % (rms) a
    day: with the head of std 0.05 the default is off by 1.9 % in norm (cosine 0.9998) on a grid of 4 steps (2.2 %
    on 8), 1.3 % with 2 steps and core_sub = 3, and exact with 1 step. With 3 x the core's tendency (22 % a day):
    9 % (cosine 0.996; 6 % with 2 steps); with 10 x (66 % a day) it was 33 % (cosine 0.95; 22 % and 0.98 with 2
    steps): about 0.45 x the day's change of a deviation, 0.3 x on a grid of 2 steps. (Half of that with the head
    of std 0.3: 1.0 %, 0.7 % and 4.6 %.)"""
    from fpk.escore import energy_score, make_norm
    lat = np.linspace(-78.75, 78.75, H)

    def case(cr, std, **kw):
        Dk = _kernel(cr)
        Fm = _flow_with_a_head(Dk, std, **kw)
        x0, xb, e, Hn, doy, tisr, lead = _one_day(Dk)
        nrm = make_norm("grid", Dk.keep, lat)
        z1 = 0.5 * torch.randn(xb.shape, generator=torch.Generator().manual_seed(2)) * Dk.keep
        c = Fm.day_ctx(x0, xb, e, Hn, doy, tisr, lead)
        return Fm, c, list(Fm.net.parameters()), lambda z: energy_score(z, z1, nrm)[0].mean()

    # ---- exact: the sampler written out, and finite differences
    for kw in (dict(steps=3), dict(path="core", core_steps=2, core_sub=3), dict(path="core", core_steps=4)):
        Fm, c, params, score = case(mcore, 0.3, **kw)
        jk = dict(jac=True) if "path" in kw else {}
        loss = lambda: score(Fm.day_run(c, grad=True, **jk))  # noqa: E731
        g = _flat_grad(loss(), params)
        hand = _sampler_by_hand(Fm, c, **jk)
        assert torch.allclose(hand, Fm.day_run(c), atol=1e-5)
        gh = _flat_grad(score(hand), params)
        assert float((g - gh).norm()) < 1e-4 * float(gh.norm()) and float(gh.norm()) > 0.1, (kw, float((g - gh).norm() / gh.norm()))
        if jk:   # the default's graph is the one with the identity for the core's Jacobian
            g0, gh0 = _flat_grad(score(Fm.day_run(c, grad=True)), params), _flat_grad(score(_sampler_by_hand(Fm, c)), params)
            assert float((g0 - gh0).norm()) < 1e-4 * float(gh0.norm()) and float((g0 - g).norm()) > 3e-3 * float(g.norm())
        fd, _ = _fd(loss, params, g)
        assert abs(fd - float(g.norm())) < 0.01 * float(g.norm()), (kw, fd, float(g.norm()))
        gen = torch.Generator().manual_seed(0)
        for _ in range(3):
            fd, d = _fd(loss, params, torch.randn(g.shape, generator=gen))
            assert abs(fd - float(g @ d)) < 0.02 * abs(float(g @ d)) + 5e-5, (kw, fd, float(g @ d))
    # ---- the core path's default against the exact gradient
    strong = _mem_core(1)
    with torch.no_grad():
        strong.net.head.weight.mul_(3.0)
    cases = [(mcore, dict(path="core", core_steps=1), (0.0, 1e-6)), (mcore, dict(path="core", core_steps=4), (0.01, 0.05)),
             (mcore, dict(path="core", core_steps=2, core_sub=3), (0.005, 0.04)), (strong, dict(path="core", core_steps=4), (0.05, 0.2))]
    for cr, kw, band in cases:
        Fm, c, params, score = case(cr, 0.05, **kw)
        loss = lambda **k: score(Fm.day_run(c, grad=True, **k))  # noqa: E731
        g, gx = _flat_grad(loss(), params), _flat_grad(loss(jac=True), params)
        fd, _ = _fd(loss, params, gx)
        assert abs(fd - float(gx.norm())) < 0.02 * float(gx.norm()), (kw, fd, float(gx.norm()))
        rel = float((g - gx).norm() / gx.norm())
        cos = float(g @ gx / (g.norm() * gx.norm()))
        assert band[0] <= rel <= band[1] and cos > 0.99, (kw, rel, cos)
        if band[0] > 0:   # along the difference of the two, the finite difference sides with the core's Jacobian
            fd, d = _fd(loss, params, gx - g)
            a, b = float(gx @ d), float(g @ d)
            assert abs(fd - a) < 0.3 * abs(a - b), (kw, fd, a, b)


def _cli(mcore, tmp_path, monkeypatch, fixed=False):
    """A trainer of scripts/run_fpk.py on the fixture: the kernel checkpoint devk.pt in tmp_path, the core and the
    data replaced (12 days of noise for every year). ``fixed``: every batch is the same batch (training and
    validation). Returns (run_fpk, the kernel, the common arguments)."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from types import SimpleNamespace
    import run_fpk as R
    Dk = _kernel(mcore)
    torch.save(dict(state={k: v for k, v in Dk.state_dict().items() if not k.startswith("core.")},
                    cfg=dict(Dk.cfg.__dict__), core_tag="c"), tmp_path / "devk.pt")
    lat = np.linspace(-78.75, 78.75, H)
    data = np.random.default_rng(0).standard_normal((12, 67, H, W)).astype(np.float32)
    monkeypatch.setattr(R, "load_core", lambda a: (None, SimpleNamespace(lat=lat, std=np.ones(67)), mcore))
    monkeypatch.setattr(R.D, "load", lambda pa, years: {y: data for y in years})
    if fixed:
        real, kept = R.D.Pairs, {}

        class Fixed:
            def __init__(self, years, seed=0, hist=0, ahead=1, **kw):  # device, prefetch: the real Pairs on the CPU
                self.p = real(years, 0, hist=hist, ahead=ahead)

            def __call__(self, B):
                if B not in kept:
                    kept[B] = self.p(B)
                return kept[B]
        monkeypatch.setattr(R.D, "Pairs", Fixed)
    return R, Dk, ["--out", str(tmp_path), "--core-tag", "c"]


def _logs(capsys):
    import json
    return [json.loads(q) for q in capsys.readouterr().out.strip().splitlines() if q.startswith("{")]


@pytest.mark.parametrize("path,norm", [("line", "grid"), ("line", "band"), ("core", "energy"), ("core0", "grid")])
def test_devtune_lowers_the_energy_score_and_keeps_the_checkpoint_format(mcore, tmp_path, monkeypatch, capsys, path, norm):
    """``devtune`` through the trainer on one fixed batch (training and validation; the validation's draws are
    fixed), with its defaults (2 sampler steps while tuning, the band norm without l = 0, one pass): a few
    iterations lower the energy score of that batch in every norm and on every path, the start line names the
    samplers and the size of the graph, and the checkpoint is the flow's own format (path, reflow_of kept) plus
    tuned_from, loaded by ``load_devflow``."""
    import json
    from types import SimpleNamespace
    from fpk.sphere import bands
    R, Dk, base = _cli(mcore, tmp_path, monkeypatch, fixed=True)
    kw = dict(line=dict(steps=3), core=dict(path="core", core_steps=2, core_sub=2), core0=dict(path="core", core_steps=3))[path]
    Fm = _flow_with_a_head(Dk, **kw)
    cfg = {k: v for k, v in Fm.cfg.__dict__.items() if k not in ("gcond", "gmember", "gdirect", "gop", "unet", "unet_reps")}  # only when used
    if path == "line":   # the straight line's checkpoint has no path keys
        cfg = {k: cfg[k] for k in ("hidden", "blocks", "sigma_min", "steps", "source")}
    src = {k: v.clone() for k, v in Fm.net.state_dict().items()}
    torch.save(dict(state=src, cfg=dict(cfg), dev_tag="devk", core_tag="c", reflow_of="devflow_t0"), tmp_path / "devflow_t.pt")
    clim = tmp_path / "clim.json"
    clim.write_text(json.dumps(dict(c=(0.2 + np.random.default_rng(1).random((66, bands(H)[1]))).tolist())))
    argv = ["devtune"] + base + ["--flow-tag", "devflow_t", "--norm", norm, "--clim", str(clim), "--days", "2", "--batch", "2",
                                 "--val-batches", "1", "--lam", "0"]
    capsys.readouterr()
    R.main(argv + ["--iters", "6", "--lr", "3e-3", "--log-every", "3"])
    logs = _logs(capsys)
    v0, mid, end = logs[0]["devtune_val_start"], logs[1], logs[2]
    assert logs[0]["norm"] == norm and logs[0]["members"] == logs[0]["pairs"] == 3 and len(logs[0]["scale"]) == 2
    # the samplers (tuning: --steps, default 2; validation: the flow's own) and the graph of a lead: batch 2 x 3
    # members x the network calls a member (2 a Heun step; the core path: on each of its grid steps)
    steps, vsteps, calls = dict(line=(2, 3, 4), core=(2, 2, 8), core0=(0, 0, 3))[path]
    assert (logs[0]["steps"], logs[0]["val_steps"], logs[0]["graph_units"]) == (steps, vsteps, 6 * calls) and logs[0]["graph_gb"] < 0.1
    assert ("es_tune" in v0) == ("end_tune" in v0) == ("val_es_tune" in end) == ("val_end_tune" in end) == (steps != vsteps)
    assert all(1.0 <= q < 1.2 for q in v0["straight"] + end["val_straight"])
    assert sum(end["val_es"]) < sum(mid["val_es"]) < sum(v0["es"]) and sum(end["val_es"]) < sum(v0["es"]) - 1e-3
    assert abs(end["best"] - sum(end["val_es"]) / 2) < 1e-3
    for q in (v0["es"], v0["es_raw"], v0["anchor"], v0["spread_skill"], end["es"], end["val_es_raw"], end["val_anchor"],
              end["val_spread_skill"], logs[0]["scale"]):
        assert len(q) == 2 and all(math.isfinite(v) and v > 0 for v in q)
    # ES / n: n = |truth - xbar| of the validation batch, so the start is below 1 only by what the spread earns
    assert all(abs(r / n - v) < 2e-4 for r, n, v in zip(v0["es_raw"], logs[0]["scale"], v0["es"])) and max(v0["es"]) < 1
    ck = torch.load(tmp_path / "devflow_t_es.pt", weights_only=False)
    assert ck["tuned_from"] == "devflow_t" and ck["reflow_of"] == "devflow_t0" and ck["dev_tag"] == "devk" and ck["core_tag"] == "c"
    assert ck["cfg"] == cfg and ck["tune"]["norm"] == norm and len(ck["tune"]["scale"]) == 2
    assert ck["tune"]["band_bmin"] == 1 and ck["tune"]["steps"] == 2 and ck["tune"]["chunk"] == 0     # the defaults
    assert any(not torch.equal(ck["state"][k], src[k]) for k in src)
    a = SimpleNamespace(out=str(tmp_path), core_tag="c", device="cpu")
    F2 = R.load_devflow(a, mcore, "devflow_t_es", pairs=2)
    assert type(F2) is type(Fm) and F2.cfg == Fm.cfg and F2.meta["tuned_from"] == "devflow_t"
    assert all(torch.equal(v, ck["state"][k]) for k, v in F2.net.state_dict().items())
    xd, doy, tis, hist = _day_inputs()
    ens, ent = F2.rollout(xd, doy, tis, 2, hist=hist, gen=torch.Generator().manual_seed(4), steps=6)   # as score --arms
    assert ens.shape == (2, 2, 2, 66, H, W) and torch.isfinite(ens).all() and torch.isfinite(ent).all()
    if (path, norm) != ("line", "grid"):
        return
    # --lr 0: nothing learned, the source's weights (a run that learns: test_devtune_keeps_the_best_validation_state)
    R.main(argv + ["--iters", "1", "--lr", "0", "--tag", "devflow_same", "--lam", "0.5"])
    same = torch.load(tmp_path / "devflow_same.pt", weights_only=False)
    assert all(torch.equal(same["state"][k], src[k]) for k in src)
    # fewer members than the kernel's pairs (a choice per lead), more (the kernel runs that many), other steps
    capsys.readouterr()
    R.main(argv + ["--iters", "1", "--members", "2", "--steps", "1", "--val-steps", "2", "--tag", "devflow_m2"])
    R.main(argv + ["--iters", "1", "--members", "5", "--tag", "devflow_m5"])
    lg = [q for q in _logs(capsys) if "devtune_val_start" in q]
    assert [(q["members"], q["pairs"]) for q in lg] == [(2, 3), (5, 5)]
    # a tag that score --arms would not take as a deviation flow, the flow's own tag, one member
    for bad in (["--tag", "tuned"], ["--tag", "devflow_t"], ["--members", "1"]):
        with pytest.raises(SystemExit):
            R.main(argv + ["--iters", "1"] + bad)


def _devtune(mcore, tmp_path, monkeypatch, path="line", std=0.05, r_bias=None):
    """``devtune`` on the fixture with one fixed batch and the flow devflow_t.pt (a head of ``std``). Returns
    run_fpk, the arguments, the flows the trainer loaded (the live networks), the gradients it handed to the clip,
    and the rows of every network call made with a graph (one list a flow loaded). ``r_bias``: the kernel's own
    (the fixture's members have a tenth of the error's size, spread / skill 0.1, and the score hardly sees them;
    r_bias = 2 gives 0.5 to 0.9)."""
    R, Dk, base = _cli(mcore, tmp_path, monkeypatch, fixed=True)
    if r_bias is not None:
        ck = torch.load(tmp_path / "devk.pt", weights_only=False)
        ck["state"]["r_bias"].fill_(r_bias)
        torch.save(ck, tmp_path / "devk.pt")
    kw = dict(line=dict(steps=3), core=dict(path="core", core_steps=2, core_sub=2), core0=dict(path="core", core_steps=3))[path]
    Fm = _flow_with_a_head(Dk, std, **kw)
    cfg = dict(Fm.cfg.__dict__)
    if path == "line":
        cfg = {k: cfg[k] for k in ("hidden", "blocks", "sigma_min", "steps", "source")}
    torch.save(dict(state=Fm.net.state_dict(), cfg=cfg, dev_tag="devk", core_tag="c"), tmp_path / "devflow_t.pt")
    live, grads, rows = [], [], []
    load, clip = R.load_devflow, torch.nn.utils.clip_grad_norm_

    def loaded(*a_, **k_):
        live.append(load(*a_, **k_))
        rows.append([])
        live[-1].net.register_forward_hook(lambda m, i, o, q=rows[-1]: q.append(i[0].shape[0]) if torch.is_grad_enabled() else None)
        return live[-1]

    def spy(params, mx):
        params = list(params)
        grads.append(torch.cat([q.grad.reshape(-1).clone() for q in params]))
        return clip(params, mx)
    monkeypatch.setattr(R, "load_devflow", loaded)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
    return R, ["devtune"] + base + ["--flow-tag", "devflow_t", "--batch", "2", "--val-batches", "1"], live, grads, rows


@pytest.mark.parametrize("path", ["line", "core", "core0"])
def test_devtune_gradient_is_the_score_plus_lam_times_the_anchor(mcore, tmp_path, monkeypatch, capsys, path):
    """The gradient ``devtune`` hands to the optimiser, on one fixed batch and one lead (the same draws for every
    lam: the anchor's are made after the lead's members): g(lam) = g(0) + lam A at the default lam = 0.5, to 1 % of
    the smaller of the score's gradient g(0) and the anchor's 0.5 A, so both are there; the same gradient in two
    passes with the graph of 2 or 4 source rows at a time (--chunk), also with the core's Jacobian (--core-jac);
    and the network evaluations made with a graph are the rows x calls of the log (graph_units)."""
    R, argv, live, grads, rows = _devtune(mcore, tmp_path, monkeypatch, path, std=0.3, r_bias=1.0)   # e' = e^r p: an offset
    argv += ["--days", "1", "--iters", "1", "--lr", "0", "--tag", "devflow_g"]
    capsys.readouterr()
    for extra in (["--lam", "0"], [], ["--lam", "1"], ["--chunk", "2"], ["--chunk", "4"]):
        R.main(argv + extra)
    g0, gh, g1, c2, c4 = grads
    A = g1 - g0
    small = min(float(g0.norm()), 0.5 * float(A.norm()))    # on this fixture (a truth of noise) the anchor's is 100 x the score's
    assert small > 0.01 and float((gh - g0 - 0.5 * A).norm()) < 0.01 * small, (small, float((gh - g0 - 0.5 * A).norm()))
    assert float((c2 - gh).norm()) < 1e-5 * float(gh.norm()) and float((c4 - gh).norm()) < 1e-5 * float(gh.norm())
    calls = dict(line=4, core=8, core0=3)[path]          # 2 Heun steps (--steps): 2 calls each; the core path: a grid step
    logs = [q for q in _logs(capsys) if "devtune_val_start" in q]
    assert [q["graph_units"] for q in logs] == [6 * calls, 6 * calls, 6 * calls, max(2 * calls, 6), 4 * calls]
    # with a graph: the sampler's 6 rows x calls, and the anchor's 6 rows (lam > 0); the chunks' first pass has none
    assert [sum(q) for q in rows] == [6 * calls] + [6 * calls + 6] * 4 and rows[3][:calls] == [2] * calls and max(rows[1]) == 6
    if path == "core":
        R.main(argv + ["--core-jac"])
        R.main(argv + ["--core-jac", "--chunk", "3"])
        assert float((grads[6] - grads[5]).norm()) < 1e-5 * float(grads[5].norm())
        assert 1e-3 * float(g0.norm()) < float((grads[5] - gh).norm()) < 0.1 * float(g0.norm())      # another score gradient
    if path == "line":   # ES_L / n_L: the score's gradient does not depend on the unit of the norm (4 x c: half the norm)
        import json
        from fpk.sphere import bands
        c = 0.2 + np.random.default_rng(1).random((66, bands(H)[1]))
        for name, f in (("c1", 1.0), ("c4", 4.0)):
            (tmp_path / f"{name}.json").write_text(json.dumps(dict(c=(f * c).tolist())))
            R.main(argv + ["--lam", "0", "--norm", "band", "--clim", str(tmp_path / f"{name}.json")])
        b1, b4 = grads[-2:]
        assert float(b1.norm()) > 0.01 and float((b1 - b4).norm()) < 1e-4 * float(b1.norm()) and float((b1 - g0).norm()) > 0.01 * float(b1.norm())
    # the graph's size: refused above --max-graph-gb, and --chunk is what the refusal offers
    gb = 6 * calls * R.net_call_mb(live[0], (H, W)) / 1024
    with pytest.raises(SystemExit, match="--chunk"):
        R.main(argv + ["--max-graph-gb", str(0.6 * gb)])
    R.main(argv + ["--max-graph-gb", str(0.6 * gb), "--chunk", "1" if path == "core0" else "2"])
    R.main(argv + ["--max-graph-gb", "0"])                                                           # 0 = no limit


@pytest.mark.parametrize("path", ["line", "core"])
def test_devtune_score_gradient_by_hand(mcore, tmp_path, monkeypatch, capsys, path):
    """The gradient of a training iteration without the anchor (--lam 0), two leads, by hand: the kernel's days and
    the members' sources with the draws the trainer made (the state of the random generator when it asked for its
    batch), the sampler of --steps with the graph, and sum_L ES_L / (n_L D) with the n_L of the log and D = 2 days."""
    from fpk.env import DYN
    from fpk.escore import energy_score, make_norm
    R, argv, live, grads, rows = _devtune(mcore, tmp_path, monkeypatch, path, std=0.3, r_bias=1.0)
    states, Fixed = [], R.D.Pairs

    class Rec(Fixed):
        def __call__(self, B):
            states.append(torch.get_rng_state())
            return Fixed.__call__(self, B)
    monkeypatch.setattr(R.D, "Pairs", Rec)
    capsys.readouterr()
    R.main(argv + ["--days", "2", "--iters", "1", "--lr", "0", "--lam", "0", "--steps", "1", "--tag", "devflow_h"])
    scale = _logs(capsys)[0]["scale"]
    Fm = live[0]
    x, ys, doy, tns, hi = Fixed(R.D.load(None, [0]), 0, hist=2, ahead=2)(2)          # the fixed batch
    nrm = make_norm("grid", Fm.dk.keep, np.linspace(-78.75, 78.75, H))
    torch.set_rng_state(states[-1])                                                  # the training batch was the last asked for
    with torch.no_grad():
        e = Fm.dk.start(x[:, DYN], doy, hi[:, :, DYN])
    xb, loss = x[:, DYN], 0.0
    for k in range(2):
        x0 = xb
        with torch.no_grad():
            xb, e, Hn, _ = Fm.dk.day(xb, e, doy + k, tns[:, k], k)
        c = Fm.day_ctx(x0, xb, e, Hn, doy, tns[:, k], k)
        z1 = (ys[:, k][:, DYN] - xb) / Fm.dk.s * Fm.dk.keep
        loss = loss + energy_score(Fm.day_run(c, 1, grad=True), z1, nrm)[0].mean() / (scale[k] * 2)
    g = _flat_grad(loss, list(Fm.net.parameters()))
    assert len(states) == 2 and float(g.norm()) > 0.01 and float((g - grads[0]).norm()) < 1e-3 * float(g.norm()), (
        float(g.norm()), float(grads[0].norm()))


def test_network_call_memory_estimate(mcore):
    """``net_call_mb``: the input and the outputs of the network's leaf modules for one row, 4 bytes a value; it
    grows with the grid (not quite as its points: the padded copies have a rim) and with the network, is at least
    the input, the trunk's first layer and the output, and leaves no hook."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import run_fpk as R
    Fm = _flow_with_a_head(_kernel(mcore), steps=3)
    mb = R.net_call_mb(Fm, (H, W))
    assert mb > (Fm.c_in + 8 + 66) * H * W * 4 / 2 ** 20 and 3.0 < R.net_call_mb(Fm, (2 * H, 2 * W)) / mb <= 4.0
    assert not any(m._forward_hooks for m in Fm.net.modules())
    big = _flow_with_a_head(_kernel(mcore), steps=3, hidden=16, blocks=(1, 1))
    assert R.net_call_mb(big, (H, W)) > 1.2 * mb


def test_devtune_keeps_the_best_validation_state(mcore, tmp_path, monkeypatch, capsys):
    """The checkpoint is the state of the best validation score, the start included. The validation's scores are
    scripted (x 1, 0.9, 0.7, 1.2, 0.8 at the start and at the four logs; four iterations move the score itself by
    far less) and the network's state is kept at every log line: the checkpoint is the state of the third
    validation, not the last one and not the first improvement, and ``best`` is the running minimum. With x 1,
    1.3, 1.1 it stays the flow it started from. The score that decides is the VALIDATION sampler's (3 steps): the
    tuning sampler's on the same draws (2 steps, val_es_tune) is scripted with the inverse factors, its best is
    another validation."""
    R, argv, live, grads, rows = _devtune(mcore, tmp_path, monkeypatch)
    real, factor, snaps, calls = R.energy_score, [1.0], [], [0]

    def scripted(*a_, **k_):   # a validation (no graph) scores the validation sampler, then the tuning one, lead by lead
        if torch.is_grad_enabled():
            return real(*a_, **k_)
        calls[0] += 1
        return tuple((factor[0] if calls[0] % 2 else 1.0 / factor[0]) * q for q in real(*a_, **k_))
    monkeypatch.setattr(R, "energy_score", scripted)
    for script, best in (([1.0, 0.9, 0.7, 1.2, 0.8], 2), ([1.0, 1.3, 1.1], 0)):
        seq, snaps[:], factor[0], calls[0] = iter(script[1:]), [], script[0], 0

        def log(d):   # the line of a validation: the state it saw; the next validation's factor
            snaps.append((d, {k: v.clone() for k, v in live[-1].net.state_dict().items()}))
            factor[0] = next(seq, 1.0)
        monkeypatch.setattr(R, "log", log)
        R.main(argv + ["--days", "2", "--iters", str(len(script) - 1), "--log-every", "1", "--lr", "1e-3", "--tag", "devflow_b"])
        ck = torch.load(tmp_path / "devflow_b.pt", weights_only=False)
        same = [all(torch.equal(ck["state"][k], st[k]) for k in st) for _, st in snaps]
        assert same == [i == best for i in range(len(script))], same
        vals, tune = ([sum(d["devtune_val_start"][k] if i == 0 else d["val_" + k]) / 2 for i, (d, _) in enumerate(snaps)]
                      for k in ("es", "es_tune"))
        assert int(np.argmin(vals)) == best and all(abs(v / vals[0] - f) < 0.02 for v, f in zip(vals, script))
        assert int(np.argmin(tune)) != best and all(abs(v / tune[0] * f - 1) < 0.02 for v, f in zip(tune, script))
        assert all(abs(d["best"] - min(vals[:i + 1])) < 2e-4 for i, (d, _) in enumerate(snaps) if i)


def test_devtune_logs_the_tuning_sampler_against_the_validations(mcore, tmp_path, monkeypatch, capsys):
    """The validation's lines by hand, lead 1 of the fixed batch with the validation's draws, a flow of 3 Heun steps
    tuned through 1 (--steps 1): ES / n of the validation sampler and of the tuning one (val_es_tune), the rms of
    the endpoints' difference over the validation endpoint's (val_end_tune, units of sigma) and the validation
    sampler's kinetic energy over its squared displacement (val_straight). With the same sampler for both
    (--val-steps 1, or --steps 0) there is nothing to compare and the two keys are not written."""
    from fpk.env import DYN
    from fpk.escore import energy_score, make_norm
    R, argv, live, grads, rows = _devtune(mcore, tmp_path, monkeypatch, std=0.3, r_bias=2.0)
    argv += ["--days", "2", "--iters", "1", "--lr", "0", "--tag", "devflow_l"]
    capsys.readouterr()
    R.main(argv + ["--steps", "1"])
    R.main(argv + ["--steps", "1", "--val-steps", "1"])
    R.main(argv + ["--steps", "0"])
    a, b, c0 = (q for q in _logs(capsys) if "devtune_val_start" in q)
    va, vb, vc = (q["devtune_val_start"] for q in (a, b, c0))
    assert (a["steps"], a["val_steps"], b["steps"], b["val_steps"], c0["steps"], c0["val_steps"]) == (1, 3, 1, 1, 3, 3)
    assert "es_tune" not in vb and "end_tune" not in vb and "es_tune" not in vc and a["scale"] == b["scale"] == c0["scale"]
    assert va["es_tune"] == vb["es"] and va["es"] == vc["es"] and va["straight"] == vc["straight"]
    assert max(abs(p_ - q_) for p_, q_ in zip(va["es_tune"], va["es"])) > 5e-4, (va["es_tune"], va["es"])    # two samplers
    Fm = live[0]
    x, ys, doy, tns, hi = R.D.Pairs(R.D.load(None, [0]), 0, hist=2, ahead=2)(2)      # the fixed batch
    gv = torch.Generator().manual_seed(123)
    nrm = make_norm("grid", Fm.dk.keep, np.linspace(-78.75, 78.75, H))
    with torch.no_grad():
        e = Fm.dk.start(x[:, DYN], doy, hi[:, :, DYN], gv)
        xb, e, Hn, _ = Fm.dk.day(x[:, DYN], e, doy, tns[:, 0], 0, gv)
        c = Fm.day_ctx(x[:, DYN], xb, e, Hn, doy, tns[:, 0], 0, gv)
        z = Fm.day_run(c)
        kin, disp = Fm.last_path
        zt, z1 = Fm.day_run(c, 1), (ys[:, 0][:, DYN] - xb) / Fm.dk.s * Fm.dk.keep
        ms = lambda t: float((t.flatten(0, 1) / c.sig).pow(2).mean())  # noqa: E731
        n = float(nrm(z1).mean())
        want = dict(es=float(energy_score(z, z1, nrm)[0].mean()) / n, es_tune=float(energy_score(zt, z1, nrm)[0].mean()) / n,
                    end_tune=math.sqrt(ms(zt - z) / ms(z)), straight=kin / disp)
    assert abs(a["scale"][0] - n) < 2e-4 and all(abs(va[k][0] - v) < 2e-4 for k, v in want.items()), (va, want)
    assert want["end_tune"] > 0.01 and want["straight"] > 1.001


def test_reflow_targets_are_the_teachers_endpoints(mcore):
    """Reflow: the end z1 of a source row is the teacher's endpoint of that row. A zero-head teacher's endpoint is
    the source row itself (line) or the end of the zero-head path (core path), and a zero-head student's loss is 0;
    with a teacher that moves, the loss is the mean square of its displacement (line) and the targets take the
    sampler onto the teacher's ends (core path)."""
    from fpk.devflow import DevFlowConfig, DeviationFlow
    w = torch.ones(1, 1, H, 1)
    Dk = _kernel(mcore, band_scale=True)     # an additive kernel: the day's end enters the core path as an offset
    x0, xb, e, Hn, doy, tisr, lead = _one_day(Dk)
    truth = xb + 0.3
    msq = lambda t, sig: float(((t / sig) ** 2 * Dk.keep).sum() / Dk.keep.expand_as(t).sum())  # noqa: E731
    gen = lambda: torch.Generator().manual_seed(5)  # noqa: E731
    sel = torch.tensor([2, 0])
    # ---- the straight line
    for source in ("kernel", "gauss"):
        St = DeviationFlow(Dk, DevFlowConfig(hidden=8, blocks=(1,), steps=3, source=source))
        T0 = DeviationFlow(Dk, DevFlowConfig(hidden=8, blocks=(1,), steps=3, source=source))
        c = T0.day_ctx(x0, xb, e, Hn, doy, tisr, lead, gen(), sel)
        assert torch.equal(T0.day_run(c).flatten(0, 1), c.z0)
        assert float(St.day_loss(x0, xb, e, Hn, truth, doy, tisr, lead, w, gen(), sel, teacher=T0)) == 0
        assert float(St.day_loss(x0, xb, e, Hn, truth, doy, tisr, lead, w, gen(), sel)) > 0.1
        T1 = _flow_with_a_head(Dk, 0.3, steps=3, source=source, sigma_min=0.0)
        S1 = DeviationFlow(Dk, DevFlowConfig(hidden=8, blocks=(1,), steps=5, source=source, sigma_min=0.0))
        seen, inner = [], T1.integrate
        T1.integrate = lambda z, cd, sig, *a_, **k_: seen.append((z, inner(z, cd, sig, *a_, **k_), a_, k_)) or seen[-1][1]
        loss = S1.day_loss(x0, xb, e, Hn, truth, doy, tisr, lead, w, gen(), sel, teacher=T1)
        z0, z1, a_, k_ = seen[0]
        c = T1.day_ctx(x0, xb, e, Hn, doy, tisr, lead, gen(), sel)
        assert len(seen) == 1 and a_ == () and k_ == {} and not z1.requires_grad            # its own steps, no gradient
        assert torch.equal(z0, c.z0) and torch.equal(z1, inner(c.z0, c.cd, c.sig, 3)) and loss.requires_grad
        assert abs(float(loss) - msq(z1 - z0, c.sig)) < 1e-5 * float(loss) and float(loss) > 1e-3
        loss.backward()
        assert all(p.grad is None for p in T1.net.parameters()) and all(p.grad is None for p in Dk.parameters())
    # ---- the core's flow
    for sub in (0, 2):
        T0 = _flow_with_a_head(Dk, 0.0, path="core", core_steps=3, core_sub=sub, sigma_min=0.0)
        St = _flow_with_a_head(Dk, 0.0, path="core", core_steps=3, core_sub=sub, sigma_min=0.0)
        c = St.day_start(x0, e, Hn, doy, tisr, lead, gen(), sel)
        zc = St.zero_path(c)
        end0 = T0.run(c).flatten(0, 1)
        assert torch.allclose(end0, zc[-1], atol=1e-6) and c.add is not None
        r = St.day_rows(c, xb, truth, gen(), z1=end0)
        assert torch.equal(r.z1, end0) and float(r.ut.abs().max()) < 1e-5
        assert float(St.day_loss(x0, xb, e, Hn, truth, doy, tisr, lead, w, gen(), sel, teacher=T0)) < 1e-10
        assert float(St.day_loss(x0, xb, e, Hn, truth, doy, tisr, lead, w, gen(), sel)) > 0.1
        T1 = _flow_with_a_head(Dk, 0.3, path="core", core_steps=3, core_sub=sub, sigma_min=0.0)
        end1 = T1.run(c).flatten(0, 1)
        k = torch.tensor([0, 1, 2, 2])
        tau = torch.tensor([0.0, 0.3, 0.6, 0.9]) if sub else None
        want = St.day_rows(c, xb, truth, k=k, tau=tau, z1=end1)
        loss = St.day_loss(x0, xb, e, Hn, truth, doy, tisr, lead, w, gen(), sel, k=k, tau=tau, teacher=T1)
        assert torch.equal(want.z1, end1) and abs(float(loss) - msq(want.ut, c.sig)) < 1e-5 * float(loss) and float(loss) > 1e-3
        full = lambda j: torch.full((4,), j)  # noqa: E731
        out = St.run(c, lambda z, j, t: St.target(c, zc, end1, full(j), tau=torch.full((4,), t) if sub else None)[1])
        assert torch.allclose(out.flatten(0, 1), end1, atol=1e-4) and float((end1 - zc[-1]).abs().max()) > 0.1


@pytest.mark.parametrize("path", ["line", "core"])
def test_reflow_through_the_trainer_straightens_the_paths(mcore, tmp_path, monkeypatch, capsys, path):
    """``devflow --reflow``: the student starts as the teacher (same kernel, architecture and path: checked), its
    checkpoint records reflow_of, and the last log line compares both flows on the validation batches, each sampled
    with the TEACHER's number of steps as the full sampler (line: 4 Heun steps, although the student was given
    --steps 6; core path: 2 Heun steps of a grid step), logged as ``steps``.

    What the fixture shows after 40 iterations (a random head as the teacher, nearly straight already; lr 3e-4:
    with 3e-3 the reflow loss rises from the teacher's weights; the numbers move with the fixture's random kernel,
    ranges over a few of them). The student's OWN paths are straighter: kinetic energy / squared displacement
    1.02 to 1.03 -> 1.01 (line), 1.02 to 1.05 -> 1.01 (core path), and its own endpoint moves less when the
    sampler is cut to 1 step (end_1: 8 to 13 % -> 5 to 6 % of the endpoint's rms; core path 2 to 3 % -> 1 to 1.5 %).

    What it does NOT show: that the student's cut sampler is nearer the teacher's endpoints than the teacher's own
    cut sampler. The student's full endpoints have drifted from the teacher's by 6 to 8 % of their rms on both
    paths (``drift``; the reflow loss only halves in 40 iterations, and 150 do not close it), which is as much as
    the curvature error that was removed. Against the teacher's full endpoint (ref_m) the student's 1-step
    sampler on the line is between no better and a quarter better than the teacher's (8 to 10 % against 8 to
    13 %), its 2-step sampler 2.5 to 3 times worse (8 % against 3 %), and on the core path its 1-step sampler 2.5
    to 3 times worse (7 to 9 % against 2 to 3 %). The test asserts the straightening of the student's own paths,
    that the drift is reported, and that ref_m is the drift and end_m composed; it does not assert a gain against
    the teacher's law."""
    import shutil
    from types import SimpleNamespace
    from fpk.devflow import DevFlowConfig
    R, Dk, base = _cli(mcore, tmp_path, monkeypatch)
    extra = [] if path == "line" else ["--path", "core", "--core-steps", "2", "--core-sub", "2"]
    kw = dict(steps=4) if path == "line" else dict(path="core", core_steps=2, core_sub=2)
    Te = _flow_with_a_head(Dk, 0.3, seed=1, blocks=DevFlowConfig().blocks, **kw)      # the trainer's architecture
    torch.save(dict(state=Te.net.state_dict(), cfg=dict(Te.cfg.__dict__), dev_tag="devk", core_tag="c"), tmp_path / "devflow_t.pt")
    started, real = [], R.reflow_teacher

    def spy(a, Fm):
        T_ = real(a, Fm)
        started.append(T_.dk is Fm.dk and type(T_) is type(Fm) and not any(p.requires_grad for p in T_.net.parameters())
                       and all(torch.equal(u, v) for u, v in zip(T_.net.state_dict().values(), Fm.net.state_dict().values()))
                       and all(torch.equal(u, v) for u, v in zip(T_.net.state_dict().values(), Te.net.state_dict().values())))
        return T_
    monkeypatch.setattr(R, "reflow_teacher", spy)
    argv = ["devflow"] + base + ["--reflow", "devflow_t", "--dev-tag", "devk", "--days", "2", "--batch", "2", "--sources", "3",
                                 "--val-batches", "2", "--hidden", "8", "--steps", "6", "--seed", "1"]
    capsys.readouterr()
    R.main(argv + ["--tag", "devflow_s", "--iters", "40", "--lr", "3e-4", "--log-every", "40"] + extra)
    logs = _logs(capsys)
    assert started == [True] and logs[0]["reflow"] == "devflow_t" and logs[-1]["teacher"] == "devflow_t"
    t, s = logs[-1]["reflow_paths"]["teacher"], logs[-1]["reflow_paths"]["student"]
    mean = lambda v: sum(v) / len(v)  # noqa: E731
    assert t["steps"] == s["steps"] == logs[-1]["steps"] == (4 if path == "line" else 2)       # the teacher's, for both
    assert all(len(q[k]) == 2 for q in (t, s) for k in ("straight", "end_1", "end_2", "move", "drift", "ref_1", "ref_2"))
    assert 1.0 <= mean(s["straight"]) < mean(t["straight"]) and mean(s["end_1"]) <= mean(t["end_1"]), (t, s)
    # against the teacher's full endpoint: the teacher's own cut error is its end_m; the student's is its drift and
    # its end_m composed (the triangle; end_m is relative to the student's own rms, the same within the drift)
    assert t["drift"] == [0.0, 0.0] and t["ref_1"] == t["end_1"] and t["ref_2"] == t["end_2"] and min(s["drift"]) > 0.01
    for m in ("1", "2"):
        assert all(abs(r - d) <= 1.1 * e + 2e-4 for r, d, e in zip(s["ref_" + m], s["drift"], s["end_" + m])), (m, s)
    assert mean(logs[-2]["val"]) < mean(logs[0]["devflow_val_start"])          # the reflow loss itself
    assert max(t["end_2"]) < min(t["end_1"]) and min(t["move"]) > 0.1
    ck = torch.load(tmp_path / "devflow_s.pt", weights_only=False)
    assert ck["reflow_of"] == "devflow_t" and ck["dev_tag"] == "devk"
    monkeypatch.setattr(R, "reflow_teacher", real)
    F2 = R.load_devflow(SimpleNamespace(out=str(tmp_path), core_tag="c", device="cpu"), mcore, "devflow_s")
    assert type(F2) is type(Te) and F2.meta["reflow_of"] == "devflow_t"
    # another architecture, path, kernel or core than the teacher's, or the teacher's own tag
    shutil.copy(tmp_path / "devk.pt", tmp_path / "devk2.pt")
    other = ["--path", "core", "--core-steps", "2"] if path == "line" else ["--core-steps", "3", "--core-sub", "2", "--path", "core"]
    for bad in (["--tag", "x", "--hidden", "16"] + extra, ["--tag", "x"] + other, ["--tag", "x", "--dev-tag", "devk2"] + extra,
                ["--tag", "devflow_t"] + extra, ["--tag", "x", "--source", "gauss"] + extra):
        with pytest.raises(SystemExit):
            R.main(argv + ["--iters", "1"] + bad)


def test_path_diagnostic_by_hand(mcore):
    """``path_diag``: kinetic energy / squared displacement of the full sampler, and the relative rms change of the
    endpoint with 1 and 2 steps, for a line flow and for the core path (which has steps to cut with core_sub only).
    With a reference flow (the teacher of a reflow) and its number of steps as the full sampler of every flow: the
    other flow's drift from the reference's endpoints and its cut samplers against them, on the same source rows."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import run_fpk as R
    from fpk.env import DYN
    Dk = _kernel(mcore)
    flows = dict(line=_flow_with_a_head(Dk, 0.3, steps=4), core=_flow_with_a_head(Dk, 0.3, path="core", core_steps=2, core_sub=3),
                 core0=_flow_with_a_head(Dk, 0.3, path="core", core_steps=2))
    data = np.random.default_rng(0).standard_normal((12, 67, H, W)).astype(np.float32)
    batch = R.D.Pairs({0: data}, 0, hist=2, ahead=2)(2)
    for name, Fl in flows.items():
        out = R.path_diag({name: Fl}, Dk, [batch], 2, torch.device("cpu"))[name]
        x, ys, doy, tns, hi = batch
        gen = torch.Generator().manual_seed(123)
        with torch.no_grad():
            e = Dk.start(x[:, DYN], doy, hi[:, :, DYN], gen)
            xb, e, Hn, _ = Dk.day(x[:, DYN], e, doy, tns[:, 0], 0, gen)
            c = Fl.day_ctx(x[:, DYN], xb, e, Hn, doy, tns[:, 0], 0, gen)
            full = Fl.day_run(c).flatten(0, 1)
            kin, disp = Fl.last_path
            ms = lambda t: float((t / c.sig).pow(2).mean())  # noqa: E731
            cut = [math.sqrt(ms(Fl.day_run(c, m).flatten(0, 1) - full) / ms(full)) for m in (1, 2)]
        assert out["steps"] == dict(line=4, core=3, core0=0)[name] and len(out["straight"]) == 2
        assert abs(out["straight"][0] - kin / disp) < 1e-3 and abs(out["move"][0] - math.sqrt(disp / ms(full))) < 1e-3
        assert "drift" not in out and "ref_1" not in out          # no reference: the flow against itself only
        if name == "core0":
            assert out["end_1"] == out["end_2"] == [None, None] and cut == [0.0, 0.0]
        else:
            assert abs(out["end_1"][0] - cut[0]) < 1e-3 and abs(out["end_2"][0] - cut[1]) < 1e-3 and cut[0] > cut[1] > 0
    # ---- with a reference (reflow: the teacher) and ITS number of steps as the full sampler of both flows
    core, core0 = dict(path="core", core_steps=2, core_sub=3), dict(path="core", core_steps=2)
    for name, n, kt, ks in (("line", 2, dict(steps=2), dict(steps=6)), ("core", 3, core, core), ("core0", 0, core0, core0)):
        Te, St = _flow_with_a_head(Dk, 0.3, seed=0, **kt), _flow_with_a_head(Dk, 0.3, seed=5, **ks)
        out = R.path_diag(dict(student=St, teacher=Te), Dk, [batch], 2, torch.device("cpu"), steps=n or None, ref="teacher")
        t, s = out["teacher"], out["student"]
        gen = torch.Generator().manual_seed(123)
        with torch.no_grad():
            e = Dk.start(x[:, DYN], doy, hi[:, :, DYN], gen)
            xb, e, Hn, _ = Dk.day(x[:, DYN], e, doy, tns[:, 0], 0, gen)
            c = Te.day_ctx(x[:, DYN], xb, e, Hn, doy, tns[:, 0], 0, gen)
            ms = lambda q: float((q.flatten(0, 1) / c.sig).pow(2).mean())  # noqa: E731
            Tn, Sn = Te.day_run(c, n or None), St.day_run(c, n or None)
            kin, disp = St.last_path
            S1, S2 = (St.day_run(c, m) for m in (1, 2))
        near = lambda got, want: abs(got[0] - want) < 1e-3  # noqa: E731
        assert t["steps"] == s["steps"] == n and t["drift"] == [0.0, 0.0] and t["ref_1"] == t["end_1"] and t["ref_2"] == t["end_2"]
        assert near(s["straight"], kin / disp) and near(s["drift"], math.sqrt(ms(Sn - Tn) / ms(Tn))) and s["drift"][0] > 0.05
        if name == "core0":
            assert s["end_1"] == s["ref_1"] == s["ref_2"] == [None, None]
            continue
        assert near(s["end_1"], math.sqrt(ms(S1 - Sn) / ms(Sn))) and near(s["ref_1"], math.sqrt(ms(S1 - Tn) / ms(Tn)))
        assert near(s["end_2"], math.sqrt(ms(S2 - Sn) / ms(Sn))) and near(s["ref_2"], math.sqrt(ms(S2 - Tn) / ms(Tn)))
        if name == "line":   # the full sampler is the teacher's 2 steps, not the student's own 6: its 2-step cut is itself
            own = R.path_diag(dict(student=St), Dk, [batch], 2, torch.device("cpu"))["student"]
            assert s["end_2"] == [0.0, 0.0] and own["steps"] == 6 and min(own["end_2"]) > 0.003, own


_BASE = "c6d6cd602b9d6f6a48cc3e9d7469ce7c26b186c1"   # the commit before fpk.escore, devtune and --reflow
_DEFAULTS_DRIVER = r'''
import contextlib, io, json, os, sys
from types import SimpleNamespace
import numpy as np, torch
ROOT, OUT = sys.argv[1], sys.argv[2]
sys.path[:0] = [ROOT, ROOT + "/scripts"]
import run_fpk as R
import diag_spectra as DS
import fpk.devflow, fpk.coreflow
assert all(os.path.realpath(m.__file__).startswith(os.path.realpath(ROOT) + os.sep) for m in (R, DS, fpk.devflow, fpk.coreflow))
from fpk.devkernel import DevConfig, EntropyDeviationKernel
from fpk.env import DYN, contract
from pgcfm.core import static_features
from pgcfm.kai_tend import make_core
H, W = 8, 16
torch.manual_seed(1); np.random.seed(1)
lat, lon = np.linspace(-78.75, 78.75, H), np.linspace(0, 360, W, endpoint=False)
contract.register_latitudes(lat)
st = static_features(lat, lon, np.random.rand(H, W).astype(np.float32), np.random.rand(H, W).astype(np.float32))
core = make_core(st, np.full(67, 0.3), np.zeros(67), np.ones(67), lat, hidden=8, blocks=(1,), norm="gn", mem=2, mem_ch=4, mz=True)
core.pm = None
with torch.no_grad():
    core.net.head.weight.normal_(0, 0.1)
core.eval()
torch.manual_seed(3)
for tag, kw in (("devk", dict(noise_pre=True)), ("devk_b", dict(noise="sphere", band_scale=True))):
    Dk = EntropyDeviationKernel(core, DevConfig(pairs=3, hidden=8, blocks=(1,), smooth=3, **kw))
    with torch.no_grad():
        Dk.r_bias.fill_(0.2)
    torch.save(dict(state={k: v for k, v in Dk.state_dict().items() if not k.startswith("core.")}, cfg=dict(Dk.cfg.__dict__),
                    core_tag="c"), f"{OUT}/{tag}.pt")
json.dump(dict(meta=dict(channels=[contract.CHANNEL_NAMES[c] for c in DYN])), open(f"{OUT}/score_devk_m.json", "w"))
data = np.random.default_rng(0).standard_normal((30, 67, H, W)).astype(np.float32)
R.load_core = lambda a: (None, SimpleNamespace(lat=lat, std=np.ones(67)), core)
R.D.load = DS.D.load = lambda pa, years: {y: data + 0.01 * (y - 2014) for y in years}
res, base = {}, ["--out", OUT, "--core-tag", "c"]


def quiet(f, argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        f(argv)
    return [{k: v for k, v in json.loads(q).items() if k != "sec"} for q in buf.getvalue().splitlines() if q.startswith("{")]


def spectra(argv):
    sys.argv = ["diag_spectra.py"] + argv
    DS.main()


for tag, dk, extra in (("devflow_line", "devk", []), ("devflow_gauss", "devk_b", ["--source", "gauss"]),
                       ("devflow_core2", "devk", ["--path", "core", "--core-steps", "2", "--core-sub", "2"]),
                       ("devflow_core0", "devk_b", ["--path", "core", "--core-steps", "3"])):
    res[tag + "/log"] = quiet(R.main, ["devflow"] + base + [
        "--tag", tag, "--dev-tag", dk, "--days", "2", "--iters", "3", "--batch", "2", "--sources", "2", "--val-batches", "1",
        "--hidden", "8", "--steps", "3", "--lr", "3e-3", "--log-every", "2", "--seed", "5"] + extra)
    ck = torch.load(f"{OUT}/{tag}.pt", weights_only=False)
    res[tag + "/ckpt"] = sorted(ck.keys()), sorted(ck["cfg"].items(), key=str)
    res.update({f"{tag}/state/{k}": v for k, v in ck["state"].items()})
    Fm = R.load_devflow(SimpleNamespace(out=OUT, core_tag="c", device="cpu"), core, tag, pairs=4)
    g = torch.Generator().manual_seed(11)
    xd, doy, tis = torch.randn(2, 66, H, W, generator=g), torch.tensor([5, 6]), torch.randn(2, 2, H, W, generator=g)
    hist = torch.randn(2, 2, 66, H, W, generator=g)
    res[tag + "/rollout"], res[tag + "/entropy"] = Fm.rollout(xd, doy, tis, 2, hist=hist, gen=torch.Generator().manual_seed(4), steps=4)
    res[tag + "/path"] = torch.tensor(list(Fm.last_path) + list(getattr(Fm, "last_split", None) or []))
    gg = torch.Generator().manual_seed(9)
    with torch.no_grad():
        e = Fm.dk.start(xd, doy, hist, gg)
        xb, e, Hn, _ = Fm.dk.day(xd, e, doy, tis[:, 0], 0, gg)
    res[tag + "/day_loss"] = Fm.day_loss(xd, xb, e, Hn, xb + 0.3, doy, tis[:, 0], 0, torch.ones(1, 1, H, 1),
                                         torch.Generator().manual_seed(2), torch.tensor([2, 0])).detach()
    res[tag + "/day_transport"] = Fm.day_transport(xd, xb, e, Hn, doy, tis[:, 0], 0, torch.Generator().manual_seed(2))
    for ft in ([], ["--flow-tag", tag]):
        if ft or tag in ("devflow_line", "devflow_gauss"):      # the kernel's own members: once per kernel
            js = f"{OUT}/diag_{tag}_{len(ft)}.json"
            quiet(spectra, base + ["--dev-tag", dk, "--ics", "2", "--batch", "2", "--leads", "2", "--json", js] + ft)
            out = json.load(open(js))
            if out["meta"].pop("es_version", 1) == 2:   # the new keys apart; the json as version 1 wrote it
                res[f"new/{tag}/es{len(ft)}"] = {k: out[k] for k in out if k.startswith("energy_score")}
                out.pop("energy_score_b1")
                out["energy_score"], out["energy_score_parts"] = out.pop("energy_score_v1"), out.pop("energy_score_parts_v1")
            res[f"{tag}/spectra{len(ft)}"] = json.dumps(out, sort_keys=True)
torch.save(res, f"{OUT}/dump.pt")
'''


def test_defaults_are_bit_for_bit_the_base_commit(tmp_path):
    """Every new option off (no --reflow, no devtune, the samplers without the gradient): the deviation-flow trainer
    on the line, the Gaussian source and both core paths (its logs and checkpoints), ``load_devflow`` with a rollout,
    ``day_loss``, ``day_transport``, and ``diag_spectra`` with the kernel's members and with a flow's (its json, the
    band-weighted energy score now through fpk.escore; what that commit wrote as energy_score is energy_score_v1 of
    es_version 2, whose three new keys are checked apart) are bit for bit those of the commit this patch is made on:
    ``git archive`` of that commit into a temporary directory, the same driver run from both trees (one thread).
    Skipped where git or that commit is not there."""
    import subprocess
    root = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
    trees = dict(base=tmp_path / "base", new=root)
    trees["base"].mkdir()
    try:
        ar = subprocess.run(["git", "-C", root, "archive", _BASE, "fpk", "scripts", "third_party"], capture_output=True, check=True)
        subprocess.run(["tar", "-x", "-C", str(trees["base"])], input=ar.stdout, check=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("no git, or the base commit is not in this repository")
    drv = tmp_path / "driver.py"
    drv.write_text(_DEFAULTS_DRIVER)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    procs = {}
    for name, tree in trees.items():
        (tmp_path / ("out_" + name)).mkdir()
        procs[name] = subprocess.Popen([sys.executable, str(drv), str(tree), str(tmp_path / ("out_" + name))], env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for name, p in procs.items():
        out = p.communicate(timeout=600)[0]
        assert p.returncode == 0, (name, out[-3000:])
    a, b = (torch.load(tmp_path / ("out_" + name) / "dump.pt", weights_only=False) for name in trees)
    es = {k: b.pop(k) for k in list(b) if k.startswith("new/")}
    assert a.keys() == b.keys() and len(a) > 150
    assert [k for k in a if not (torch.equal(a[k], b[k]) if torch.is_tensor(a[k]) else a[k] == b[k])] == []
    assert not os.path.exists(trees["base"] / "fpk" / "escore.py") and any(k.endswith("spectra2") for k in a)
    # es_version 2: what the base commit wrote as energy_score is energy_score_v1 (compared above). A flow's members
    # are independent: the same number. The kernel's [e, -e]: the same E|X - y|, a smaller spread term without the
    # partners (2 |e| apart), so a higher score; and the score without the band l = 0 is another number
    assert sorted(es) == sorted(f"new/{t}/es{n}" for t, n in [("devflow_line", 0), ("devflow_gauss", 0)] + [
        (t, 2) for t in ("devflow_line", "devflow_gauss", "devflow_core2", "devflow_core0")])
    for k, v in es.items():
        new, old, p2, p1, b1 = (np.array(v[q]) for q in ("energy_score", "energy_score_v1", "energy_score_parts",
                                                         "energy_score_parts_v1", "energy_score_b1"))
        assert new.shape == b1.shape == (2,) and (b1 > 0).all() and (b1 != new).all() and np.allclose(p2[:, 0] - p2[:, 1], new)
        if k.endswith("es2"):
            assert (new == old).all() and (p2 == p1).all()
        else:
            assert (p2[:, 0] == p1[:, 0]).all() and (p2[:, 1] < 0.99 * p1[:, 1]).all() and (new > old + 0.01 * p1[:, 1]).all()
