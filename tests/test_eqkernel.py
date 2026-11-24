"""The kernelized equation in the deviation kernel (fpk.devkernel eq_kernel / EqOperator, 감자 2026-11-24).

The cores carry a stand-in operator (a smooth nonlinear function of the state's first 48 channels with the attributes
the real OperatorFeatures has: ``phys.std``); what is tested is the kernel's handling of the operator."""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from fpk.env import DYN, contract  # noqa: E402
from pgcfm.core import static_features  # noqa: E402
from pgcfm.kai_tend import make_core  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from test_fpk import _cli, _flow_with_a_head, _logs, mcore  # noqa: E402,F401

H, W = 8, 16


class _FakeOp:
    """G / its RMS (as OperatorFeatures returns it), with the z-score std of the 66 channels."""

    def __init__(self):
        self.phys = SimpleNamespace(std=torch.linspace(1.0, 2.0, 66, dtype=torch.float64).view(66, 1, 1))

    def __call__(self, x):
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
        c.features = _FakeOp()
        c.gscale.fill_(1.0 / 86400.0)   # G's RMS in SI per second: one z-score std a day over std
    return c.eval()


@pytest.fixture(scope="module")
def gcore():
    return _core(True)


def _kernel(core, eqop=None, **kw):
    from fpk.devkernel import DevConfig, EntropyDeviationKernel
    torch.manual_seed(3)
    return EntropyDeviationKernel(core, DevConfig(**{**dict(pairs=3, hidden=8, blocks=(1,), smooth=3), **kw}), eqop=eqop)


def _day(Dk, seed=7):
    g = torch.Generator().manual_seed(seed)
    xd, doy, tis = torch.randn(2, 66, H, W, generator=g), torch.tensor([5, 6]), torch.zeros(2, 2, H, W)
    e = Dk.start(xd, doy, None, g)
    return xd, e, Dk.day(xd, e, doy, tis[:, 0], 0, torch.Generator().manual_seed(seed + 1))


def test_the_operator_is_G_per_day_in_z_score_units_on_its_own_channels(gcore):
    from fpk.devkernel import EqOperator
    op = EqOperator(gcore)
    x = torch.randn(3, 66, H, W)
    out = op(x)
    g = _FakeOp()(x)
    std = _FakeOp().phys.std.reshape(-1).float()
    for name, k in (("u", 0), ("v", 1), ("T", 2), ("q", 3)):
        for j, p in enumerate(contract.LEVELS_HPA):
            c = DYN.index(contract.CHANNEL_INDEX[f"{name.lower()}{p}"])
            assert torch.allclose(out[:, c], g[:, k * 12 + j] / std[c], atol=1e-5)
    others = [i for i in range(66) if i not in set(op.dst.tolist())]
    assert len(others) == 18 and float(out[:, others].abs().max()) == 0.0
    with pytest.raises(ValueError):
        EqOperator(_core(False))


def test_the_equation_kernel_starts_as_the_kernel_without_it(gcore):
    """A = 0 and the nets' zero heads: the same day as the kernel without the kernelized equation."""
    Pl, Eq = _kernel(gcore), _kernel(gcore, eq_kernel=True)
    assert float(Eq.eq_a.abs().sum()) == 0.0
    xd, e, (x1, e1, H1, _) = _day(Pl)
    _, _, (x2, e2, H2, _) = _day(Eq)
    assert torch.allclose(x1, x2) and torch.allclose(e1, e2, atol=1e-6) and torch.allclose(H1, H2, atol=1e-6)
    assert sum(p.numel() for p in Eq.net.parameters()) > sum(p.numel() for p in Pl.net.parameters())   # 132 more inputs


def test_p_G_is_the_operators_day_of_the_pairs_softly_bounded(gcore):
    from fpk.devkernel import EqOperator
    Eq = _kernel(gcore, eq_kernel=True, eq_clamp=1e6)
    xd, e, _ = _day(Eq)
    op = EqOperator(gcore)
    s = Eq.s
    gp = torch.stack([op(xd + s * e[:, i]) for i in range(e.shape[1])], 1)
    gm = torch.stack([op(xd - s * e[:, i]) for i in range(e.shape[1])], 1)
    want = (gp - gm) / (2 * s.unsqueeze(0)) * Eq.keep.unsqueeze(0)
    assert torch.allclose(Eq._pG, want, atol=1e-4, rtol=1e-4)
    Ec = _kernel(gcore, eq_kernel=True, eq_clamp=0.01)
    _day(Ec)
    bound = 0.01 * torch.exp(Ec.size(e))[:, None]
    assert bool((Ec._pG.abs() <= bound + 1e-6).all())


def test_A_moves_the_direction_on_the_operators_channels_and_learns_from_the_log_score(gcore):
    Eq = _kernel(gcore, eq_kernel=True)
    with torch.no_grad():
        Eq.eq_a.fill_(0.5)
    xd, e, (x1, e1, H1, _) = _day(Eq)
    q_add = Eq.last["p"]
    pG = Eq._pG
    Pl = _kernel(gcore, eq_kernel=True)
    _day(Pl)
    d = q_add - Pl.last["p"]
    assert torch.allclose(d, 0.5 * Eq.eq_mask * pG, atol=1e-5)
    others = [i for i in range(66) if Eq.eq_mask[0, i, 0, 0] == 0]
    assert float(d[:, :, others].abs().max()) == 0.0
    truth = x1 + 0.5 * Eq.s * torch.randn_like(x1)
    loss, _ = Eq.lead_loss(H1, x1, e1, truth, torch.ones(1, 1, H, 1))
    loss.backward()
    assert Eq.eq_a.grad is not None and float(Eq.eq_a.grad.abs().sum()) > 0
    rep = Eq.eq_report()
    assert set(rep) == {"levels", "u", "v", "T", "q"} and all(abs(v - 0.5) < 1e-6 for v in rep["T"])


def test_the_correction_net_reads_the_equations_fields(gcore):
    Pl, Eq = _kernel(gcore, correct=True), _kernel(gcore, correct=True, eq_kernel=True)
    xd, e, (x1, e1, H1, _) = _day(Eq)
    assert torch.isfinite(H1).all() and torch.isfinite(e1).all()
    _, _, (x0, e0, H0, _) = _day(Pl)
    assert torch.allclose(x0, x1) and torch.allclose(H0, H1, atol=1e-6)     # zero heads: the same start


def test_an_outside_operator_gives_the_kernelized_equation_to_a_core_without_G(gcore, tmp_path, monkeypatch):
    from fpk.devkernel import EqOperator
    import run_fpk as R
    nog = _core(False)
    op = EqOperator(gcore)
    with pytest.raises(ValueError):
        _kernel(nog, eq_kernel=True)                                   # no operator at all
    with pytest.raises(ValueError):
        _kernel(nog, eq_kernel=True, eq_core="g")                      # the tag without the operator
    with pytest.raises(ValueError):
        _kernel(nog, eqop=op, eq_kernel=True)                          # the operator without the tag
    with pytest.raises(ValueError):
        _kernel(gcore, eqop=op, eq_kernel=True, eq_core="g")           # a core with its own operator
    Dn = _kernel(nog, eqop=op, eq_kernel=True, eq_core="g")
    assert Dn.eqop is op and not Dn.gfeat
    xd, e, (x1, e1, H1, _) = _day(Dn)
    gp = torch.stack([op(xd + Dn.s * e[:, i]) for i in range(e.shape[1])], 1)
    gm = torch.stack([op(xd - Dn.s * e[:, i]) for i in range(e.shape[1])], 1)
    c = Dn.cfg.eq_clamp * torch.exp(Dn.size(e))[:, None]
    want = c * torch.tanh((gp - gm) / (2 * Dn.s.unsqueeze(0)) * Dn.keep.unsqueeze(0) / c)
    assert torch.allclose(Dn._pG, want, atol=1e-4, rtol=1e-4)
    with torch.no_grad():
        Dn.eq_a.normal_(0, 0.3)
    torch.save(dict(state={k: v for k, v in Dn.state_dict().items() if not k.startswith("core.")},
                    cfg=dict(Dn.cfg.__dict__), core_tag="c"), tmp_path / "devk_eq.pt")
    monkeypatch.setattr(R, "eq_operator", lambda a, tag: op if tag == "g" else None)
    a = SimpleNamespace(out=str(tmp_path), core_tag="c", device="cpu")
    D2 = R.load_devk(a, nog, "devk_eq")
    assert D2.eqop is op and D2.cfg.eq_core == "g" and torch.equal(D2.eq_a, Dn.eq_a)
    _, _, (y1, f1, G1, _) = _day(Dn.eval())
    _, _, (y2, f2, G2, _) = _day(D2)
    assert torch.allclose(f1, f2, atol=1e-6) and torch.allclose(G1, G2, atol=1e-6)


def test_devtune_moves_A_with_the_flow_at_kernel_lr_times_eq_lr(gcore, mcore, tmp_path, monkeypatch, capsys):
    """``devtune --kernel-lr`` on a kernel with the kernelized equation (the flow-aware kernel, 감자 2026-11-24 11:32
    KST): A is an optimizer group of its own with lr = --kernel-lr x --eq-lr and no weight decay, as in devk, so the
    members' score through the flow moves how much the kernel follows the equation; the nets keep --kernel-lr. Adam's
    first step is +-lr per element: A moves by kernel_lr x eq_lr, every net parameter by at most kernel_lr (plus
    the weight decay's lr x 1e-4 x |p|). The start line and every log line carry A."""
    from fpk.devkernel import EqOperator
    core = mcore                                       # a core without G(x), with a memory (as core_ngmz8W)
    op = EqOperator(gcore)
    R0, _, base = _cli(core, tmp_path, monkeypatch, fixed=True)
    Dk = _kernel(core, eqop=op, eq_kernel=True, eq_core="g", correct=True, noise="sphere", band_scale=True)
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for n, p in Dk.named_parameters():
            if not n.startswith("core."):
                p.add_(0.05 * torch.randn(p.shape, generator=g))
        Dk.r_bias.fill_(1.0)
    torch.save(dict(state={k: v for k, v in Dk.state_dict().items() if not k.startswith("core.")},
                    cfg=dict(Dk.cfg.__dict__), core_tag="c"), tmp_path / "devk.pt")
    Fm = _flow_with_a_head(Dk, 0.3, steps=3)
    cfg = {k: Fm.cfg.__dict__[k] for k in ("hidden", "blocks", "sigma_min", "steps", "source")}
    torch.save(dict(state=Fm.net.state_dict(), cfg=cfg, dev_tag="devk", core_tag="c"), tmp_path / "devflow_t.pt")
    monkeypatch.setattr(R0, "eq_operator", lambda a, tag: op if tag == "g" else None)
    live, load = [], R0.load_devflow
    monkeypatch.setattr(R0, "load_devflow", lambda *a_, **k_: live.append(load(*a_, **k_)) or live[-1])
    start = {n: p.detach().clone() for n, p in Dk.own_parameters().items()}
    argv = ["devtune"] + base + ["--flow-tag", "devflow_t", "--batch", "2", "--val-batches", "1", "--days", "2",
                                 "--lam", "0", "--iters", "1", "--lr", "1e-3", "--kernel-lr", "1e-4"]
    for k, (eq_lr, tag) in enumerate(((20.0, "devflow_a"), (1.0, "devflow_b"))):
        capsys.readouterr()
        R0.main(argv + ["--eq-lr", str(eq_lr), "--tag", tag, "--kernel-out", f"devk_{tag[-1]}"])
        logs = _logs(capsys)
        Dt = live[-1].dk
        assert Dt.eqop is op
        moved = {n: (p.detach() - start[n]).abs() for n, p in Dt.own_parameters().items()}
        a_moves = moved["eq_a"][Dt.eq_mask.bool()]
        assert float(a_moves.max()) == pytest.approx(1e-4 * eq_lr, rel=1e-3)          # A: +-lr x eq_lr
        assert float(a_moves.median()) > 0.5 * 1e-4 * eq_lr                            # on most of its channels
        assert float(moved["eq_a"][~Dt.eq_mask.bool().expand_as(moved["eq_a"])].abs().max()) == 0.0   # no grad, no decay
        nets = max(float(v.max()) for n, v in moved.items() if n != "eq_a")
        assert 0.5e-4 < nets < 1e-4 * 1.01 + 1e-8                                     # the nets: kernel_lr
        v0 = [q for q in logs if "devtune_val_start" in q][0]
        assert v0["kernel_eq_lr"] == pytest.approx(1e-4 * eq_lr) and set(v0["eq_a"]) == {"levels", "u", "v", "T", "q"}
        assert "eq_a" in logs[-1] and logs[-1]["it"] == 1
