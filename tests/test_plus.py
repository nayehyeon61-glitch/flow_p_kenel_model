"""The + version (1.5 deg grid with the poles, lazy years, GPU options), on small grids on the CPU.

    grid_lat / SHT     the contract's two grids from the shape; on a grid with the poles the transform is the exact
                       inverse of its synthesis for every band-limited field, and refuses a singular truncation
    smooth_noise       the k x k smoothing (k = 11 on the 1.5 deg grid) has unit variance on every row
    LazyYear / Pairs   the years on disk read as the in-memory z-scores, the one-read-per-sample path and the
                       prefetching thread draw the same batches
    devkernel          on a grid with the poles: the kernel's day, the e0 spectrum of the grid's length, the U-Net nets
                       (zero head: the same kernel as the plain nets), bf16 and recomputation of the nets
"""

import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fpk.env import contract  # noqa: E402
from fpk.kernel import smooth_noise  # noqa: E402
from fpk.sphere import SHT, grid_lat, has_poles  # noqa: E402
from pgcfm.core import static_features  # noqa: E402
from pgcfm.kai_tend import make_core  # noqa: E402

HP, WP = 13, 24  # a grid with the poles, W = 2 (H - 1), as 121 x 240


def _coef(B, L, seed=0):
    g = torch.Generator().manual_seed(seed)
    c = torch.complex(torch.randn(B, L, L, generator=g), torch.randn(B, L, L, generator=g)) * torch.tril(torch.ones(L, L))
    c[..., 0] = torch.complex(c[..., 0].real, torch.zeros(B, L))
    return c


def test_grid_latitudes_of_the_two_grids():
    lat = grid_lat(121, 240)
    assert lat[0] == -90.0 and lat[-1] == 90.0 and np.allclose(np.diff(lat), 1.5) and has_poles(lat)
    lat = grid_lat(32, 64)
    assert np.allclose(lat, -90 + 5.625 * (np.arange(32) + 0.5)) and not has_poles(lat)
    with pytest.raises(ValueError):
        grid_lat(32, 60)


def test_sht_on_the_pole_grid_is_exact_and_has_the_data_width():
    S = SHT(grid_lat(HP, WP))
    assert (S.W, S.L, S.poles) == (WP, HP - 2, True)      # lmax = H - 3, below the Nyquist order W / 2 - 1
    c = _coef(3, S.L)
    x = S.synthesis(c)
    assert x.shape == (3, HP, WP)
    assert torch.allclose(S.analysis(x), c, atol=1e-4)
    assert torch.isfinite(S.spectrum(x)).all()
    n = S.noise(torch.ones(S.L), (2, 5), torch.Generator().manual_seed(0))
    assert n.shape == (2, 5, HP, WP) and torch.isfinite(n).all()
    with pytest.raises(ValueError):                         # the pole rows carry no fit: lmax = H - 2 is singular
        SHT(grid_lat(HP, WP), lmax=HP - 2)
    # the derivatives stay finite on the pole rows (cos = 6e-17 there)
    gx, gy = S.grad(c)
    assert torch.isfinite(gx).all() and torch.isfinite(gy).all()


def test_sht_without_poles_is_unchanged():
    S = SHT(grid_lat(8, 16))
    assert (S.W, S.L, S.poles) == (16, 8, False)
    c = _coef(2, 8, 1)
    assert torch.allclose(S.analysis(S.synthesis(c)), c, atol=1e-4)


@pytest.mark.parametrize("cells", [3, 5, 11])
def test_smooth_noise_has_unit_variance_on_every_row(cells):
    shape = (600, 1, 25, 48)
    z = smooth_noise(shape, torch.zeros(1), torch.Generator().manual_seed(0), cells)
    v = z.var((0, 1, 3))
    assert float((v - 1).abs().max()) < 0.12
    # neighbouring cells correlate more the wider the window
    r = float((z[..., :-1] * z[..., 1:]).mean() / z.pow(2).mean())
    assert r > {3: 0.6, 5: 0.75, 11: 0.88}[cells]


def _year(tmp_path, D=30, H=6, W=10):
    rng = np.random.default_rng(0)
    raw = (rng.standard_normal((D, 67, H, W)) * 3 + 10).astype(np.float32)
    p = tmp_path / "2000.npy"
    np.save(p, raw)
    mean, std = raw.astype(np.float64).mean((0, 2, 3)), raw.astype(np.float64).std((0, 2, 3))
    return np.load(p, mmap_mode="r"), mean, std


def test_lazy_year_reads_the_zscores(tmp_path):
    from fpk.data import LazyYear
    from fpk.env import rkt
    R = rkt()
    mm, mean, std = _year(tmp_path)
    assert isinstance(mm, np.memmap)
    ly = LazyYear(mm, mean, std, R.EPS)
    full = R.zscore(np.asarray(mm), mean, std)
    assert len(ly) == 30 and ly.shape == full.shape
    assert np.allclose(ly[3], full[3], atol=1e-5)
    assert np.allclose(ly[4:9], full[4:9], atol=1e-5)
    j = np.array([2, 0, 7])
    assert np.allclose(ly[j], full[j], atol=1e-5)
    assert np.allclose(ly[5:8, 62], full[5:8, 62], atol=1e-5)          # days, then a channel (tisr)
    assert np.allclose(ly[2][[0, 3]], full[2][[0, 3]], atol=1e-5)


def test_pairs_read_once_per_sample_and_prefetch_the_same_batches(tmp_path):
    from fpk.data import LazyYear, Pairs
    from fpk.env import rkt
    R = rkt()
    mm, mean, std = _year(tmp_path)
    full = {2000: R.zscore(np.asarray(mm), mean, std)}
    lazy = {2000: LazyYear(mm, mean, std, R.EPS)}
    ref = Pairs(full, seed=3, hist=4, ahead=5)
    one = Pairs(lazy, seed=3, hist=4, ahead=5, device="cpu")
    pre = Pairs(lazy, seed=3, hist=4, ahead=5, device="cpu", prefetch=True)
    assert one.dev is not None
    for _ in range(3):
        a, b, c = ref(4), one(4), pre(4)
        for u, v, w in zip(a, b, c):
            assert torch.allclose(u.float(), v.float(), atol=1e-5) and torch.equal(v, w)


@pytest.fixture(scope="module")
def pole_core():
    torch.manual_seed(0)
    lat, lon = grid_lat(HP, WP), np.linspace(0, 360, WP, endpoint=False)
    contract.register_latitudes(lat)
    st = static_features(lat, lon, np.random.rand(HP, WP).astype(np.float32), np.random.rand(HP, WP).astype(np.float32))
    c = make_core(st, np.full(67, 0.3), np.zeros(67), np.ones(67), lat, hidden=8, blocks=(1,), norm="gn")
    c.pm = None
    return c.eval()


def _spec(tmp_path, L, name="e0.json"):
    sp = [0.0] + [1.0 / (l * l) for l in range(1, L)]
    f = tmp_path / name
    f.write_text(json.dumps({"err_deg": {"all": [sp]}}))
    return str(f)


def test_kernel_on_the_pole_grid(pole_core, tmp_path):
    from fpk.devkernel import DevConfig, EntropyDeviationKernel
    L = HP - 2
    cfg = dict(pairs=2, hidden=8, blocks=(1,), smooth=5, noise="sphere", band_scale=True, correct=True,
               e0_spec=_spec(tmp_path, L), noise_cells=5)
    Dk = EntropyDeviationKernel(pole_core, DevConfig(**cfg))
    assert (Dk.sht.W, Dk.sht.L) == (WP, L)
    xd, doy, tis = torch.randn(2, 66, HP, WP), torch.tensor([5, 6]), torch.zeros(2, 2, HP, WP)
    ens, ent = Dk.rollout(xd, doy, tis, 2, gen=torch.Generator().manual_seed(1))
    assert ens.shape == (2, 4, 2, 66, HP, WP) and torch.isfinite(ens).all() and torch.isfinite(ent).all()
    with pytest.raises(ValueError):                                # a spectrum of another grid
        EntropyDeviationKernel(pole_core, DevConfig(**{**cfg, "e0_spec": _spec(tmp_path, 32, "x.json")}))


def test_unet_nets_start_as_the_plain_nets_and_run_in_bf16(pole_core, tmp_path):
    from fpk.devkernel import DevConfig, EntropyDeviationKernel
    from fpk.devflow import DevFlowConfig, DeviationFlow
    from fpk.nets import set_run
    cfg = dict(pairs=2, hidden=8, blocks=(1,), smooth=5, noise="sphere", correct=True, noise_cells=5)
    A = EntropyDeviationKernel(pole_core, DevConfig(**cfg))
    U = EntropyDeviationKernel(pole_core, DevConfig(**cfg, unet=(8, 16, 16), unet_reps=1))
    U.load_state_dict({k: v for k, v in A.state_dict().items() if not k.startswith(("net.", "cnet."))}, strict=False)
    xd, doy, tis = torch.randn(2, 66, HP, WP), torch.tensor([5, 6]), torch.zeros(2, 2, HP, WP)
    a, _ = A.rollout(xd, doy, tis, 2, gen=torch.Generator().manual_seed(2))
    b, _ = U.rollout(xd, doy, tis, 2, gen=torch.Generator().manual_seed(2))
    assert torch.allclose(a, b, atol=1e-5)                           # zero heads: the nets add nothing yet
    # bf16 nets and their recomputation in the backward pass: finite values, gradients to the nets
    with torch.no_grad():
        U.net.head.weight.normal_(0, 0.01)
    set_run(U, torch.bfloat16, True)
    U.through = 2
    e = U.start(xd, doy)
    x1, e1, H1, _ = U.day(xd, e, doy, tis[:, 0], 0)
    loss = U.ign(H1, x1, xd, torch.ones(1, 1, HP, 1))[0]
    loss.backward()
    assert torch.isfinite(loss) and H1.dtype == torch.float32
    assert U.net.head.weight.grad is not None and float(U.net.head.weight.grad.abs().sum()) > 0
    F_ = DeviationFlow(U, DevFlowConfig(hidden=8, blocks=(1,), unet=(8, 16), unet_reps=1))
    set_run(F_, torch.bfloat16, True)
    with torch.no_grad():
        F_.net.head.weight.normal_(0, 0.01)
    lk = F_.day_loss(xd, x1.detach(), e1.detach(), H1.detach(), xd, doy, tis[:, 0], 0, torch.ones(1, 1, HP, 1))
    lk.backward()
    assert torch.isfinite(lk) and float(F_.net.head.weight.grad.abs().sum()) > 0
