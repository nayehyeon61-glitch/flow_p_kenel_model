"""G-flow (fpk.devflow gmember / gdirect / gcond): the member's own physics in the deviation flow.

The cores here carry a stand-in operator (a smooth nonlinear function of the state's first 48 channels, the real
operator needs a real grid and real units); what is tested is the flow's handling of G, not the operator."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from fpk.env import DYN, contract  # noqa: E402
from pgcfm.core import static_features  # noqa: E402
from pgcfm.kai_tend import make_core  # noqa: E402

H, W = 8, 16


def _fake_operator(x):
    return torch.tanh(2.0 * x[:, :48]) + 0.1 * x[:, 12:60] ** 2


def _core(gfeat):
    torch.manual_seed(2)
    lat = np.linspace(-78.75, 78.75, H)
    lon = np.linspace(0, 360, W, endpoint=False)
    contract.register_latitudes(lat)
    st = static_features(lat, lon, np.random.rand(H, W).astype(np.float32), np.random.rand(H, W).astype(np.float32))
    c = make_core(st, np.full(67, 0.3), np.zeros(67), np.ones(67), lat, hidden=8, blocks=(1,), norm="gn", gfeat=gfeat)
    c.pm = None
    if gfeat:
        c.features = _fake_operator
    return c.eval()


@pytest.fixture(scope="module")
def gcore():
    return _core(True)


def _kernel(core):
    from fpk.devkernel import DevConfig, EntropyDeviationKernel
    torch.manual_seed(3)
    return EntropyDeviationKernel(core, DevConfig(pairs=3, hidden=8, blocks=(1,), smooth=3))


def _flow(Dk, **kw):
    from fpk.devflow import DevFlowConfig, DeviationFlow
    torch.manual_seed(5)
    return DeviationFlow(Dk, DevFlowConfig(**{**dict(hidden=8, blocks=(1,), steps=2), **kw}))


def _ctx(Dk, Fm):
    g = torch.Generator().manual_seed(7)
    xd, doy, tis = torch.randn(2, 66, H, W, generator=g), torch.tensor([5, 6]), torch.zeros(2, 2, H, W)
    e = Dk.start(xd, doy, None, g)
    xb, e1, H1, _ = Dk.day(xd, e, doy, tis[:, 0], 0, g)
    return xd, xb, e1, H1, doy, tis, Fm.day_ctx(xd, xb, e1, H1, doy, tis[:, 0], 0, g)


def test_the_physics_channel_maps_each_operator_term_onto_its_own_variable_and_level():
    from fpk.devflow import g_direct_index
    src, dst = g_direct_index(48)
    assert len(src) == len(dst) == 48 and len(set(dst.tolist())) == 48
    for name, k in (("u", 0), ("v", 1), ("T", 2), ("q", 3)):
        for j, p in enumerate(contract.LEVELS_HPA):
            i = k * 12 + j
            assert int(src[i]) == i and int(dst[i]) == DYN.index(contract.CHANNEL_INDEX[f"{name.lower()}{p}"])
    with pytest.raises(ValueError):
        g_direct_index(40)


def test_the_default_flow_is_the_flow_as_it_was(gcore):
    """Defaults: G(xbar) in the conditioning (the kernel's core has G), no dG, no A, the same input channels."""
    Dk = _kernel(gcore)
    Fm = _flow(Dk)
    assert Fm.c_in == 66 + 66 + 48 + 66 + gcore.static.shape[0] + 4 + 1 + 4
    assert not hasattr(Fm.net, "g_direct") and Fm.g_report() is None
    assert Fm.n_gc == 48 and Fm.n_gm == 0
    No = _flow(Dk, gcond=False)
    assert No.c_in == Fm.c_in - 48 and No.n_gc == 0
    xb, H1 = torch.randn(2, 66, H, W), torch.zeros(2, 66, H, W)
    assert No.cond(xb, H1, torch.tensor([3, 4]), 1).shape[1] == Fm.cond(xb, H1, torch.tensor([3, 4]), 1).shape[1] - 48


def test_dG_is_the_change_of_the_operator_at_the_member(gcore):
    Dk = _kernel(gcore)
    for gcond in (True, False):
        Fm = _flow(Dk, gmember=True, gcond=gcond)
        assert Fm.c_in == 66 + 66 + (48 if gcond else 0) + 66 + gcore.static.shape[0] + 4 + 1 + 48 + 4
        xd, xb, e1, H1, doy, tis, c = _ctx(Dk, Fm)
        z = c.z0
        xbar = c.cd[:, :66]
        want = _fake_operator(xbar + Dk.s * z * Dk.keep) - _fake_operator(xbar)
        assert torch.allclose(Fm.member_g(z, c.cd), want, atol=1e-6)
        assert float(Fm.member_g(torch.zeros_like(z), c.cd).abs().max()) == 0.0   # the mean's own state: no change
        v = Fm.velocity(z, torch.full((z.shape[0],), 0.3), c.cd, c.sig)
        assert v.shape == z.shape and float(v.abs().max()) == 0.0     # zero head: the flow starts as the kernel


def test_the_physics_channel_starts_at_zero_and_then_adds_A_dG_on_u_v_T_q_only(gcore):
    Dk = _kernel(gcore)
    Pl, Gd = _flow(Dk), _flow(Dk, gdirect=True)
    assert Gd.c_in == Pl.c_in and Gd.net.g_direct.shape == (1, 66, 1, 1) and float(Gd.net.g_direct.abs().sum()) == 0
    sd = Pl.net.state_dict()
    with torch.no_grad():
        for k in list(sd)[-2:]:
            sd[k].normal_(0, 0.05)    # a head that moves
    Pl.net.load_state_dict(sd)
    Gd.net.load_state_dict({**sd, "g_direct": Gd.net.g_direct.detach().clone()})
    xd, xb, e1, H1, doy, tis, c = _ctx(Dk, Pl)
    t = torch.full((c.z0.shape[0],), 0.4)
    v0, v1 = Pl.velocity(c.z0, t, c.cd, c.sig), Gd.velocity(c.z0, t, c.cd, c.sig)
    assert torch.allclose(v0, v1, atol=1e-6)                       # A = 0: the flow without the channel
    with torch.no_grad():
        Gd.net.g_direct.normal_(0, 1.0)
    d = Gd.velocity(c.z0, t, c.cd, c.sig) - v0
    dg = Gd.member_g(c.z0, c.cd)
    ref = torch.zeros_like(d)
    ref[:, Gd.g_dst] = dg[:, Gd.g_src]
    assert torch.allclose(d, Gd.net.g_direct * ref * Dk.keep, atol=1e-5)
    others = [i for i in range(66) if i not in set(Gd.g_dst.tolist())]
    assert float(d[:, others].abs().max()) == 0.0                  # z, the surface fields: no operator tendency
    rep = Gd.g_report()
    assert rep["levels"] == contract.LEVELS_HPA and set(rep) == {"levels", "u", "v", "T", "q"} and len(rep["T"]) == 12
    j = DYN.index(contract.CHANNEL_INDEX["t850"])
    assert abs(rep["T"][contract.LEVELS_HPA.index(850)] - float(Gd.net.g_direct.flatten()[j])) < 1e-4


def test_A_learns_from_the_flow_matching_loss_and_the_kernel_does_not(gcore):
    Dk = _kernel(gcore)
    Fm = _flow(Dk, gmember=True, gdirect=True)
    xd, xb, e1, H1, doy, tis, c = _ctx(Dk, Fm)
    truth = xb + 0.3 * Dk.s * torch.randn_like(xb)
    loss = Fm.day_loss(xd, xb, e1, H1, truth, doy, tis[:, 0], 0, torch.ones(1, 1, H, 1), torch.Generator().manual_seed(1))
    loss.backward()
    assert Fm.net.g_direct.grad is not None and float(Fm.net.g_direct.grad.abs().sum()) > 0
    assert all(p.grad is None for p in Dk.parameters())
    assert any(n == "g_direct" for n, _ in Fm.net.named_parameters())   # the optimizer of devflow takes Fm.net's


def test_a_G_flow_checkpoint_keeps_its_options_and_a_plain_one_is_saved_as_before(gcore, tmp_path):
    from types import SimpleNamespace
    import run_fpk as R
    Dk = _kernel(gcore)
    torch.save(dict(state={k: v for k, v in Dk.state_dict().items() if not k.startswith("core.")},
                    cfg=dict(Dk.cfg.__dict__), core_tag="c"), tmp_path / "devk.pt")
    a = SimpleNamespace(out=str(tmp_path), core_tag="c", device="cpu")
    plain = R.flow_ckpt_cfg(_flow(Dk))
    assert not {"path", "core_steps", "core_sub", "gcond", "gmember", "gdirect"} & set(plain)
    Fm = _flow(Dk, gmember=True, gdirect=True)
    with torch.no_grad():
        Fm.net.g_direct.normal_(0, 0.5)
        [p.normal_(0, 0.05) for p in list(Fm.net.parameters())[-2:]]
    cfg = R.flow_ckpt_cfg(Fm)
    assert cfg["gmember"] is True and cfg["gdirect"] is True and "gcond" not in cfg
    assert R.flow_ckpt_cfg(_flow(Dk, gcond=False))["gcond"] is False
    torch.save(dict(state=Fm.net.state_dict(), cfg=cfg, dev_tag="devk", core_tag="c"), tmp_path / "devflow_g.pt")
    F2 = R.load_devflow(a, gcore, "devflow_g", pairs=3)
    assert F2.cfg.gmember and F2.cfg.gdirect and torch.equal(F2.net.g_direct, Fm.net.g_direct)
    xd, doy, tis = torch.randn(2, 66, H, W), torch.tensor([5, 6]), torch.zeros(2, 2, H, W)
    one = Fm.rollout(xd, doy, tis, 2, gen=torch.Generator().manual_seed(4))[0]
    two = F2.rollout(xd, doy, tis, 2, gen=torch.Generator().manual_seed(4))[0]
    assert torch.allclose(one, two, atol=1e-6) and float((one - one[:, :1]).abs().max()) > 0


def test_G_flow_needs_the_operator_and_the_straight_line():
    from fpk.devflow import DevFlowConfig, DeviationFlow
    Dn = _kernel(_core(False))
    for kw in (dict(gmember=True), dict(gdirect=True)):
        with pytest.raises(ValueError):
            DeviationFlow(Dn, DevFlowConfig(hidden=8, blocks=(1,), **kw))
    DeviationFlow(Dn, DevFlowConfig(hidden=8, blocks=(1,), gcond=False))   # no G anyway: allowed, the same flow
    Dg = _kernel(_core(True))
    with pytest.raises(ValueError):
        DeviationFlow(Dg, DevFlowConfig(hidden=8, blocks=(1,), path="core", gdirect=True))


def test_G_flow_on_the_U_Net_of_the_plus_version(gcore):
    """G-flow and the + version's U-Net together: the dG input and A on the multi-scale trunk, a zero start."""
    Dk = _kernel(gcore)
    Fm = _flow(Dk, unet=(8, 16), gmember=True, gdirect=True)
    assert Fm.net.g_direct.shape == (1, 66, 1, 1)
    xd, xb, e1, H1, doy, tis, c = _ctx(Dk, Fm)
    v = Fm.velocity(c.z0, torch.full((c.z0.shape[0],), 0.5), c.cd, c.sig)
    assert v.shape == c.z0.shape and float(v.abs().max()) == 0.0
    import run_fpk as R
    cfg = R.flow_ckpt_cfg(Fm)
    assert cfg["unet"] == (8, 16) and cfg["gmember"] and cfg["gdirect"]
