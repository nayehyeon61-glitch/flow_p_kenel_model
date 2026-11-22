"""Balanced noise without axial angular momentum: DevConfig.bal_lmin and DevConfig.aam_fix (fpk.devkernel).

The axial angular momentum of a level is the area integral of u cos(phi). Two ways to take it are used below:

    aam_exact   the integral of the truncated field itself: for u = -(1/a) dpsi/dphi + (1/(a cos phi)) dchi/dlam it is
                -(2/a) <psi sin(phi)> = -(2 / (a sqrt 3)) Re psi_10, psi from ``SHT.helmholtz`` (exact for a wind
                in the truncation)
    aam_grid    sum_i cos(phi_i) [u cos(phi)]_i / sum_i cos(phi_i), the grid's sum that scripts/diag_conserve.py takes.
                On the equiangular grid this is not an exact quadrature: the zonal psi modes of odd degree l >= 3
                leave 0.3 to 3 % of the weight of l = 1 on 8 latitudes (1e-5 to 5e-3 on 32)

bal_lmin = 2 makes aam_exact of the balanced wind zero (to rounding); aam_fix makes aam_grid of the whole new part
zero (to rounding).
"""

import importlib
import json
import math
import os
import subprocess
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from test_fpk import H, W, core, mcore  # noqa: E402,F401  (the fixtures)
from fpk.devkernel import DevConfig, EntropyDeviationKernel  # noqa: E402
from fpk.env import contract  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# the keys of the balanced checkpoints of the runs (runs/fpk/devk_band_bal.pt, devk_band_bal_fdt.pt), net / cnet aside
KEYS_BAL = sorted(["a_log", "bal_dlnp", "bal_idx", "bal_k", "bal_unit", "bal_vert", "band_of", "base_amp", "chi_raw",
                   "e0_amp", "g_band", "h_raw", "p_log", "psi_amp", "r_bias", "rho_raw", "sht.ana", "sht.deg", "sht.syn",
                   "w_raw"])
KEYS_FDT = sorted(KEYS_BAL + ["fdt_c", "fdt_rho"])
CFG_OLD = ["pairs", "eps0", "h0", "smooth", "r_max", "hidden", "blocks", "cloud_mean", "correct", "d_max", "r_dissip",
           "band_scale", "noise_pre", "noise_mult", "e0_spec", "noise", "noise_spec", "bal_vert", "fdt", "fdt_bmin"]
H1 = math.log(math.e - 1)   # h_raw of h = 1
REF = "c6d6cd6"             # the commit before DevConfig.bal_lmin / aam_fix: the defaults are bit for bit its kernel

COS = torch.cos(torch.deg2rad(-90.0 + 180.0 / H * (torch.arange(H) + 0.5))).view(H, 1)
rms = lambda t: float(t.double().pow(2).mean().sqrt())  # noqa: E731
own = lambda D: sorted(k for k in D.state_dict() if not k.startswith(("core.", "net.", "cnet.")))  # noqa: E731


def aam_grid(u):
    """(..., H, W) -> (...): sum_i cos(phi_i) (mean over lon of u cos(phi))_i / sum_i cos(phi_i)."""
    return (u * COS ** 2).mean(-1).sum(-1) / COS.sum()


def aam_exact(S, u, v):
    """(..., H, W) x 2 -> (...): the area mean of u cos(phi) of a wind in the truncation, -(2 / (a sqrt 3)) Re psi_10."""
    return -2.0 / (S.radius * math.sqrt(3.0)) * S.helmholtz(u, v)[0][..., 1, 0].real


def kern(c, **kw):
    torch.manual_seed(0)
    return EntropyDeviationKernel(c, DevConfig(pairs=2, hidden=8, blocks=(1,), smooth=3, noise="balanced", **kw))


def draw(D, n=40, seed=0):
    D._xi, D._bc = None, None
    with torch.no_grad():
        return D.new_part((n, 2, 66, H, W), torch.zeros(1), torch.Generator().manual_seed(seed))


def si(D, x, k):
    """The channels of z, t, u or v (k = 0..3) of x in SI units: (B, P, K, H, W)."""
    return x[:, :, D.bal_idx[k]] * D.bal_unit[k].view(1, 1, -1, 1, 1)


def test_the_two_area_means_on_a_solid_body_rotation(mcore):
    S = kern(mcore).sht
    u = 7.0 * COS.expand(H, W)
    assert abs(float(aam_exact(S, u, torch.zeros(H, W))) - 7.0 * 2 / 3) < 1e-4       # <cos^2> = 2/3
    assert abs(float(aam_grid(u)) - 7.0 * float(COS.pow(3).sum() / COS.sum())) < 1e-5
    assert abs(float(aam_grid(u)) / (7.0 * 2 / 3) - 1) < 0.01                        # the grid's sum: 0.7 % off


@pytest.mark.parametrize("chi", [True, False])
def test_bal_lmin_2_has_no_solid_body_rotation(mcore, chi):
    """Remainder weight 0, h = 1: the balanced wind alone. Tolerances (8 x 16, float32): the exact area mean of
    u cos(phi) and the l = 1 coefficients of psi and chi 1e-5 of their size with bal_lmin = 1 (measured 4e-8); the
    grid's sum 1e-2 (measured 1.7e-3: the quadrature, see the module's docstring)."""
    out = {}
    for lmin in (1, 2):
        D = kern(mcore, bal_lmin=lmin)
        with torch.no_grad():
            D.h_raw.fill_(H1)
            D.w_raw.fill_(-30.0)
            if not chi:
                D.chi_raw.fill_(-30.0)
        x = draw(D)
        u, v = si(D, x, 2), si(D, x, 3)
        psi, ch = D.sht.helmholtz(u, v)
        out[lmin] = dict(exact=rms(aam_exact(D.sht, u, v)), grid=rms(aam_grid(u)), psi1=rms(psi[..., 1, :2].abs()),
                         chi1=rms(ch[..., 1, :2].abs()), psi2=rms(psi[..., 2, :3].abs()),
                         wind=rms(x[:, :, torch.cat([D.bal_idx[2], D.bal_idx[3]])]), ucos=rms(u * COS), D=D)
        # the wind's variance at h = 1 is still about one, and psi_amp is normalized over the degrees that are left
        l = torch.arange(H, dtype=torch.float32)
        assert abs(out[lmin]["wind"] - 1.0) < 0.15 and abs(float((D.psi_amp.pow(2) * l * (l + 1)).sum()) - 1) < 1e-5
        # by construction: the coefficients of psi and chi below bal_lmin are zero, the others are not
        with torch.no_grad():
            cp, cc = D.psi_chi(D.sht.noise_coef((2, 3, 2, 12), torch.zeros(1), torch.Generator().manual_seed(1)))
        assert float(cp[..., :lmin, :].abs().max()) == 0 and float(cc[..., :lmin, :].abs().max()) == 0
        assert float(cp[..., lmin, :lmin + 1].abs().min()) > 0
    a, b = out[1], out[2]
    assert a["exact"] > 0.1 * a["ucos"] and a["grid"] > 0.1 * a["ucos"]              # bal_lmin = 1: clearly not zero
    assert b["exact"] < 1e-5 * a["exact"] and b["grid"] < 1e-2 * a["grid"]
    assert b["psi1"] < 1e-5 * a["psi1"] and b["psi2"] > a["psi2"]                    # the equatorial axes (m = 1) too
    if chi:
        assert b["chi1"] < 1e-5 * a["chi1"] and a["chi1"] > 0.05 * a["psi1"]
    assert float(b["D"].psi_amp[1]) == 0 and float(a["D"].psi_amp[1]) > 0 and torch.equal(a["D"].bal_k, b["D"].bal_k)
    assert torch.equal(a["D"].psi_amp, kern(mcore).psi_amp)                           # 1 is the default


def test_bal_lmin_2_leaves_the_degree_1_of_z_and_t(mcore):
    """bal_lmin is for the angular momentum only. Linear balance: Phi_n is made of psi_(n-1) and psi_(n+1), so
    Phi_1 comes from psi_2 and psi_1 goes into Phi_2. The same draws, remainder weight 0: the degree-1 power of z and
    T with bal_lmin = 2 is that of bal_lmin = 1 times c^2, the renormalization of psi_amp^2 over the degrees left
    (1.32 on this grid's spectrum, 1.0044 on the runs'), on every level; their degree-2 power falls (to 0.39 to 0.47
    of c^2 here), the degrees >= 3 are again c^2. So the spread of the band l = 1 of z and T is not reduced."""
    sp, amp = {}, {}
    for lmin in (1, 2):
        D = kern(mcore, bal_lmin=lmin)
        with torch.no_grad():
            D.h_raw.fill_(H1)
            D.w_raw.fill_(-30.0)
        x = draw(D)
        sp[lmin] = [D.sht.spectrum(x[:, :, D.bal_idx[k]]).double().mean((0, 1)) for k in range(4)]   # z, t, u, v: (K, L)
        amp[lmin] = D.psi_amp.double()
    c2 = float((amp[2][2] / amp[1][2]) ** 2)
    assert c2 > 1.05 and torch.allclose(amp[2][2:], amp[1][2:] * math.sqrt(c2), rtol=1e-6)
    assert abs(c2 * (1 - 2 * float(amp[1][1]) ** 2) - 1) < 1e-5                   # sum_l l (l + 1) psi_amp^2 = 1 in both
    for k in (0, 1):                                                              # z and T
        r = sp[2][k] / sp[1][k] / c2                                              # (K, L)
        assert float(sp[1][k][:, 1].sum() / sp[1][k].sum()) > 0.2                  # the degree 1 is a large part of them
        assert float((r[:, 1] - 1).abs().max()) < 1e-4 and float((r[:, 3:] - 1).abs().max()) < 1e-4
        assert float(r[:, 2].max()) < 0.6 and float(r[:, 2].min()) > 0.2
    # the wind's own degree 1 is what the option removes: v (from psi_1's m = 1, the equatorial axes) all but vanishes
    assert float((sp[2][3][:, 1] / sp[1][3][:, 1]).max()) < 0.05


def test_bal_lmin_and_aam_fix_are_checked(mcore, core):
    for bad in (0, H, -1):
        with pytest.raises(ValueError):
            kern(mcore, bal_lmin=bad)
    assert float(kern(mcore, bal_lmin=H - 1).psi_amp[:H - 1].abs().max()) == 0
    for kw in (dict(bal_lmin=2), dict(aam_fix=True)):
        for noise in ("sphere", "smooth"):
            with pytest.raises(ValueError):
                EntropyDeviationKernel(core, DevConfig(pairs=2, hidden=8, blocks=(1,), noise=noise, **kw))
    assert DevConfig().bal_lmin == 1 and DevConfig().aam_fix is False


@pytest.mark.parametrize("lmin", [1, 2])
def test_aam_fix_removes_the_whole_new_parts_axial_angular_momentum(mcore, lmin):
    """The remainder on (start values: weight 0.3, chi 0.1): the grid's sum of u cos(phi) of every level of the
    new part is zero to 1e-5 of its size without the fix (float32: measured 2e-7); only u changes, by c cos(phi)."""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import diag_conserve as C
    D0, D1 = kern(mcore, bal_lmin=lmin), kern(mcore, bal_lmin=lmin, aam_fix=True)
    x0, x1 = draw(D0), draw(D1)
    a0, a1 = aam_grid(si(D0, x0, 2)), aam_grid(si(D1, x1, 2))                        # (B, P, K)
    assert rms(a0) > 0.05 * rms(si(D0, x0, 2) * COS) and rms(a1) < 1e-5 * rms(a0)
    iu = D0.bal_idx[2]
    rest = torch.tensor([c for c in range(66) if c not in iu.tolist()])
    assert torch.equal(x0[:, :, rest], x1[:, :, rest])
    d = (x0 - x1)[:, :, iu]                                                          # c cos(phi) / unit of the level
    c = d[..., H // 2:H // 2 + 1, :1] / COS[H // 2]
    assert torch.allclose(d, c * COS, atol=1e-5 * float(x0[:, :, iu].abs().max())) and rms(d) > 1e-3 * rms(x0[:, :, iu])
    # the diagnostic's own number (scripts/diag_conserve.py): the vertical integral of the levels' angular momentum
    inv = C.Invariants(np.zeros(67), np.asarray(contract.REFERENCE_STD), np.linspace(-78.75, 78.75, H))
    assert rms(inv(x1 * D1.s)["aam"]) < 1e-5 * rms(inv(x0 * D0.s)["aam"])
    # the fix alone: a solid-body wind is removed whole, a second pass changes nothing, the rest is untouched
    y = torch.zeros(1, 1, 66, H, W)
    y[:, :, iu] = 3.0 * COS
    y[:, :, D1.bal_idx[3]] = 1.0
    z = D1.aam_neutral(y)
    assert float(z[:, :, iu].abs().max()) < 1e-5 and torch.equal(z[:, :, D1.bal_idx[3]], y[:, :, D1.bal_idx[3]])
    assert torch.allclose(D1.aam_neutral(x1), x1, atol=1e-6)
    # a day of the kernel: the new part of the day carries none, and the gradients reach the amplitudes through it
    xd, doy = torch.randn(2, 66, H, W), torch.tensor([5, 6])
    hist = torch.randn(2, int(mcore.mem), 66, H, W)
    e = D1.start(xd, doy, hist)
    x2, e2, H2, _ = D1.day(xd, e, doy, torch.zeros(2, H, W), 0)
    hx, h0 = D1.last["hxi"], draw(D0, n=2)
    assert rms(aam_grid(si(D1, hx.detach(), 2))) < 1e-5 * rms(aam_grid(si(D0, h0, 2)))
    D1.ign(H2, x2, x2.detach() + 0.3 * D1.s, torch.ones(1, 1, H, 1))[0].backward()
    for n in ("p_log", "chi_raw", "w_raw", "a_log", "h_raw"):
        g = getattr(D1, n).grad
        assert g is not None and torch.isfinite(g).all() and float(g.abs().sum()) > 0, n
    # the gradient THROUGH the fix: a loss of the new part's u alone (the score above reaches the amplitudes through
    # the other channels too). It is the gradient of the same projection u - c cos(phi), c = <u cos> / <cos^2>,
    # written out here on the kernel without the fix (same parameters, same draws)
    names, grads = ("p_log", "chi_raw", "w_raw", "a_log", "h_raw"), []
    for D in (D1, D0):
        D._xi, D._bc = None, None
        u = D.new_part((2, 2, 66, H, W), torch.zeros(1), torch.Generator().manual_seed(3))[:, :, iu]
        if D is D0:
            unit = D.bal_unit[2].view(-1, 1, 1)
            c = (u * unit * COS ** 2).mean(-1, keepdim=True).sum(-2, keepdim=True) / COS.pow(3).sum()
            u = u - c * COS / unit
        grads.append(torch.autograd.grad(u.pow(2).mean(), [getattr(D, n) for n in names]))
    for n, g1, g0 in zip(names, *grads):
        assert float(g1.abs().max()) > 0 and torch.allclose(g1, g0, rtol=1e-3, atol=1e-6 * float(g0.abs().max())), n
    assert float(grads[0][2][[0, 1, 3]].abs().max()) == 0                         # w_raw: the remainder of u alone


def test_time_correlation_and_init_from_a_kernel_with_every_degree(mcore):
    """bal_lmin = 2 keeps the AR(1) state of psi / chi; a kernel fitted with bal_lmin = 1 is continued with 2:
    the learned parts come from the checkpoint, psi_amp (a CFG_BUFFER) stays this run's."""
    xd, doy, tis = torch.randn(2, 66, H, W), torch.tensor([5, 6]), torch.zeros(2, 2, H, W)
    hist = torch.randn(2, int(mcore.mem), 66, H, W)
    A = kern(mcore)
    with torch.no_grad():
        A.rho_raw.fill_(2.0)                 # rho = tanh(2) = 0.96
        A.p_log.fill_(0.5)
        A.h_raw.fill_(0.7)
    state = {k: v.clone() for k, v in A.state_dict().items() if not k.startswith("core.")}
    C = kern(mcore, bal_lmin=2, aam_fix=True)
    assert own(C) == own(A)                  # no new key in the state
    miss, kept = C.load_init(state)
    assert not miss.unexpected_keys and all(k.startswith("core.") for k in miss.missing_keys) and kept == ["psi_amp"]
    assert float(C.psi_amp[1]) == 0 and float(A.psi_amp[1]) > 0
    assert all(torch.equal(getattr(C, k), getattr(A, k)) for k in ("p_log", "h_raw", "rho_raw", "chi_raw", "w_raw", "bal_k"))
    assert C.load_init({k: v for k, v in C.state_dict().items() if not k.startswith("core.")})[1] == []
    cor = lambda a, b: float((a.conj() * b).real.sum() / (a.abs().pow(2).sum().sqrt() * b.abs().pow(2).sum().sqrt()))  # noqa: E731
    with torch.no_grad():
        e = C.start(xd, doy, hist)
        x1, e1, _, _ = C.day(xd, e, doy, tis[:, 0], 0)
        b1, n1 = C._bc.clone(), C.last["hxi"].clone()
        C.day(x1, e1, doy + 1, tis[:, 1], 1)
        assert cor(b1, C._bc) > 0.8 and cor(n1, C.last["hxi"]) > 0.8
        C.start(xd, doy, hist)
        assert C._bc is None and C._xi is None
    u, v = si(C, C.last["hxi"], 2), si(C, C.last["hxi"], 3)
    assert rms(aam_grid(u)) < 1e-5 * rms(u * COS)


def _files(tmp_path):
    """The files of the balanced runs on the test grid: a spectrum (diag_spectra.py) and the climate's bands."""
    from fpk.sphere import bands
    nb = bands(H)[1]
    E = [0.0, 1.0, 2.0, 4.0, 2.0, 1.0, 0.5, 0.25]
    sp, cl = tmp_path / "spec.json", tmp_path / "clim.json"
    sp.write_text(json.dumps({"err_deg": {"all": [[0.1] + [1.0] * (H - 1)], "u250": [E]}}))
    cl.write_text(json.dumps(dict(c=[[1.0] * nb] * 66, rho_band=[0.9, 0.9, 0.5, 0.0])))
    return str(sp), str(cl)


def test_checkpoint_key_sets(mcore, tmp_path):
    """The state of a balanced kernel has the keys of the runs' checkpoints with and without the new options, an
    old state (no new key, a cfg without the options) loads, and the cfg of a new checkpoint adds the two options."""
    sp, cl = _files(tmp_path)
    run = dict(correct=True, band_scale=True, e0_spec=sp, noise_spec=sp)
    for kw, keys in ((run, KEYS_BAL), (dict(run, fdt=cl, fdt_bmin=1), KEYS_FDT)):
        A, B = kern(mcore, **kw), kern(mcore, bal_lmin=2, aam_fix=True, **kw)
        assert own(A) == keys and own(B) == keys
        assert all(a.shape == b.shape for a, b in zip(A.state_dict().values(), B.state_dict().values()))
        state = {k: v for k, v in A.state_dict().items() if not k.startswith("core.")}
        old = {k: v for k, v in A.cfg.__dict__.items() if k in CFG_OLD}                # the cfg of an old checkpoint
        assert sorted(old) == sorted(CFG_OLD) and sorted(set(A.cfg.__dict__) - set(old)) == ["aam_fix", "bal_lmin"]
        L = EntropyDeviationKernel(mcore, DevConfig(**old))
        miss = L.load_state_dict(state, strict=False)
        assert not miss.unexpected_keys and all(k.startswith("core.") for k in miss.missing_keys)
        assert L.cfg.bal_lmin == 1 and not L.cfg.aam_fix and torch.equal(L.psi_amp, A.psi_amp)
        assert B.load_init(state)[1] == ["psi_amp"]


@pytest.fixture(scope="module")
def head(tmp_path_factory):
    """fpk/devkernel.py of the commit before bal_lmin / aam_fix (c6d6cd6; another one with $FPK_REF) as the module
    ``fpk_head.devkernel``: git archive REF fpk | tar -x. A fixed commit, not HEAD: once the change is committed
    HEAD is the change itself. Skipped where the commit is absent (a shallow clone, no git)."""
    dst = tmp_path_factory.mktemp("head")
    ref = os.environ.get("FPK_REF", REF)
    subprocess.run(f"git -C '{ROOT}' archive {ref} fpk | tar -x -C '{dst}'", shell=True, capture_output=True)
    if not os.path.exists(os.path.join(dst, "fpk", "devkernel.py")):
        pytest.skip(f"no git commit {ref} to compare with")
    pkg = types.ModuleType("fpk_head")
    pkg.__path__ = [os.path.join(dst, "fpk")]
    sys.modules["fpk_head"] = pkg
    return importlib.import_module("fpk_head.devkernel")


@pytest.mark.parametrize("case", ["smooth", "sphere", "balanced", "run", "run_fdt", "pre"])
def test_default_options_are_bit_for_bit_the_reference_commit(mcore, head, tmp_path, case):
    """Fixed seeds: the state after the construction, a 3-day rollout (states and entropy) and the day's new part
    are equal bit for bit to the reference commit's, for the configurations of the runs."""
    sp, cl = _files(tmp_path)
    kw = dict(smooth=dict(), sphere=dict(noise="sphere", band_scale=True), balanced=dict(noise="balanced"),
              run=dict(noise="balanced", correct=True, band_scale=True, e0_spec=sp, noise_spec=sp),
              run_fdt=dict(noise="balanced", correct=True, band_scale=True, e0_spec=sp, noise_spec=sp, fdt=cl, fdt_bmin=1,
                           bal_vert=0.5),
              pre=dict(noise="balanced", noise_pre=True, cloud_mean=True, r_dissip=True))[case]
    out = []
    for mod in (sys.modules[EntropyDeviationKernel.__module__], head):
        assert (mod is head) == (mod.__name__ == "fpk_head.devkernel")
        torch.manual_seed(0)
        D = mod.EntropyDeviationKernel(mcore, mod.DevConfig(pairs=2, hidden=8, blocks=(1,), smooth=3, **kw))
        state = {k: v.clone() for k, v in D.state_dict().items() if not k.startswith("core.")}
        g = torch.Generator().manual_seed(1)
        with torch.no_grad():                # away from the start values (zero head, r = 0), the same in both
            for n, p in D.named_parameters():
                if not n.startswith("core."):
                    p.add_(0.1 * torch.randn(p.shape, generator=g))
        xd, doy, tis = torch.randn(2, 66, H, W, generator=g), torch.tensor([5, 6]), torch.zeros(2, 3, H, W)
        hist = torch.randn(2, int(mcore.mem), 66, H, W, generator=g)
        ens, ent = D.rollout(xd, doy, tis, 3, hist=hist, gen=torch.Generator().manual_seed(2))
        out.append((state, ens, ent, D.last["hxi"], D.last["r"]))
    (s0, *r0), (s1, *r1) = out
    assert list(s0) == list(s1) and all(torch.equal(s0[k], s1[k]) for k in s0)
    assert all(torch.equal(a, b) for a, b in zip(r0, r1)) and torch.isfinite(r0[0]).all()
    assert float(r0[3].abs().max()) > 0 and (case == "pre" or float(r0[2].abs().max()) > 0)   # r and the new part: not zero


def test_devk_cli_bal_lmin_aam_fix_and_init(mcore, tmp_path, monkeypatch, capsys):
    """``devk --bal-lmin 2 --aam-fix --init`` through the trainer: the options reach the kernel and its checkpoint,
    the fitted kernel with every degree is continued (this run's psi_amp is kept), and the checkpoint reloads."""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import run_fpk as R
    rng = np.random.default_rng(0)
    cl = SimpleNamespace(lat=np.linspace(-78.75, 78.75, H), std=np.asarray(contract.REFERENCE_STD))
    monkeypatch.setattr(R, "load_core", lambda a: (None, cl, mcore))
    monkeypatch.setattr(R.D, "load", lambda pa, years: {y: rng.standard_normal((12, 67, H, W)).astype(np.float32) for y in years})
    base = ["devk", "--out", str(tmp_path), "--core-tag", "c", "--days", "2", "--iters", "1", "--batch", "2", "--pairs", "2",
            "--val-batches", "1", "--hidden", "8", "--smooth", "3", "--log-every", "1"]
    R.main(base + ["--noise", "balanced", "--tag", "a"])
    a = torch.load(tmp_path / "a.pt", weights_only=False)
    assert a["cfg"]["bal_lmin"] == 1 and a["cfg"]["aam_fix"] is False and float(a["state"]["psi_amp"][1]) > 0
    capsys.readouterr()
    R.main(base + ["--noise", "balanced", "--tag", "b", "--init", "a", "--bal-lmin", "2", "--aam-fix"])
    logs = [json.loads(q) for q in capsys.readouterr().out.strip().splitlines()]
    assert logs[0] == dict(init="a", buffers_of_this_run_kept=["psi_amp"])
    assert all(math.isfinite(v) for q in logs for v in q.get("val_ign", []))
    b = torch.load(tmp_path / "b.pt", weights_only=False)
    assert b["cfg"]["bal_lmin"] == 2 and b["cfg"]["aam_fix"] is True and float(b["state"]["psi_amp"][1]) == 0
    assert sorted(b["state"]) == sorted(a["state"])
    Dk = R.load_devk(SimpleNamespace(out=str(tmp_path), core_tag="c", device="cpu"), mcore, "b")
    assert Dk.cfg.bal_lmin == 2 and Dk.cfg.aam_fix and float(Dk.psi_amp[1]) == 0
    x = draw(Dk, n=4)
    assert rms(aam_grid(si(Dk, x, 2))) < 1e-5 * rms(si(Dk, x, 2) * COS)
    with pytest.raises(ValueError):          # the options belong to the balanced noise
        R.main(base + ["--noise", "sphere", "--tag", "x", "--bal-lmin", "2"])
