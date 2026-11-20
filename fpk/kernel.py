"""Equation -> distribution: the physical kernel K(x, y) built on the tendency core.

The tendency core of PG-CFM integrates the primitive-equation tendency for one day,

    Phi(x) = x + int_0^1day f_theta(x, G(x)) dt          G(x) = -R0(x): the operator's 48 physical terms
                                                          (u, v, T, q x 12 levels), f_theta: the learned closure

and gives ONE next state. The kernel turns that equation into a distribution of the next state by summing kernels
centred on the equation's own trajectories from a cloud of states around x:

    K(x, y) = (1/N) sum_k  N( y ; Phi(x + eps * s * delta_k) , diag (s * h(x))^2 )          (kernel sum, KDE form)

    delta_k   spatially smooth unit noise (3 x 3 cells), antithetic pairs +-delta (the centres' mean stays at Phi(x))
    eps       learned per channel: how far the equation is started from x (the analysis uncertainty)
    h(x)      learned bandwidth per channel and grid point, read off (x, G(x), Phi(x) - x): the part of the next day the
              equation does not determine (the closure's error, sub-grid noise), in units of s
    s         std of the daily increment per channel (``core.t6``)

So the centres carry the equation's flow-dependent growth (where Phi stretches the cloud, the kernel is wide), and
the bandwidth the rest. The kernel is fitted by the composite (per channel and grid point) negative log-likelihood
of the observed next day; the equation itself (the core) stays frozen. The kernel is then the source distribution of
flow matching (fpk.flow), and its entropy is the one-day physical entropy production (``entropy``).
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import DYN, TISR_DYN, time_features
from .env import PGFM  # noqa: F401  (sys.path)
from pgcfm.core import Net  # noqa: E402
from pgcfm.kai_tend import core_step, stateless_norm  # noqa: E402
from pgcfm.seqdct import _NetCfg  # noqa: E402

LOG2PI = math.log(2 * math.pi)


def smooth_noise(shape, like: torch.Tensor, gen: Optional[torch.Generator] = None) -> torch.Tensor:
    """White noise smoothed over 3 x 3 cells (lon circular, lat reflect), rescaled to unit variance."""
    xi = torch.randn(shape, generator=gen, device="cpu").to(like)
    xi = xi.reshape(-1, *shape[-3:])
    xi = F.pad(F.pad(xi, (1, 1, 0, 0), mode="circular"), (0, 0, 1, 1), mode="reflect")
    out = F.avg_pool2d(xi, 3, 1) * 3.0
    # the reflected halo repeats the neighbour row in the two edge rows: variance 15/9 there, back to 1
    out[..., 0, :] *= 3.0 / math.sqrt(15.0)
    out[..., -1, :] *= 3.0 / math.sqrt(15.0)
    return out.reshape(shape)


@dataclass
class KernelConfig:
    n_centres: int = 4        # N (even: antithetic pairs)
    eps0: float = 0.3         # start of eps (units of s)
    h0: float = 0.6           # start of the bandwidth (units of s)
    h_min: float = 0.02
    hidden: int = 32
    blocks: tuple = (2, 2)


class PhysicsKernel(nn.Module):
    """K(x, .) of a frozen tendency core ``core`` (pgcfm DynCore / memory core without memory use).

    States are the 66 dynamic channels, normalized; ``doy`` (B,) day of year of x; ``tisr`` (B, H, W) the forcing
    of the next day."""

    def __init__(self, core, cfg: KernelConfig = KernelConfig()):
        super().__init__()
        self.core, self.cfg = core, cfg
        for p in core.parameters():
            p.requires_grad_(False)
        self.gfeat = bool(getattr(core.cfg, "gfeat", False))
        n_g = core.gscale.shape[1] if self.gfeat else 0
        n_static = core.static.shape[0]
        c_in = 66 + n_g + 66 + n_static + 4
        # GroupNorm, no dropout (ClimODE's block has BatchNorm + Dropout: batch-dependent, and train != eval)
        self.bw = stateless_norm(Net(c_in, 66, _NetCfg(cfg.hidden, tuple(cfg.blocks))))
        # zero head: the bandwidth starts at h0 everywhere
        self.h_bias = nn.Parameter(torch.full((1, 66, 1, 1), math.log(math.expm1(cfg.h0 - cfg.h_min))))
        self.eps_raw = nn.Parameter(torch.full((66,), math.log(math.expm1(cfg.eps0))))

    def train(self, mode: bool = True):
        super().train(mode)
        self.core.eval()  # the equation is frozen
        return self

    # ---- the pieces
    @property
    def s(self) -> torch.Tensor:
        return self.core.t6  # (1, 66, 1, 1)

    @property
    def keep(self) -> torch.Tensor:
        return self.core.keep  # (1, 66, 1, 1), 0 on tisr (a known forcing)

    def eps(self) -> torch.Tensor:
        return F.softplus(self.eps_raw).view(1, 66, 1, 1)

    def G(self, xd: torch.Tensor) -> Optional[torch.Tensor]:
        return self.core.features(xd) if self.gfeat else None

    def phi(self, xd: torch.Tensor, doy: torch.Tensor, tisr: Optional[torch.Tensor]) -> torch.Tensor:
        """The equation's day: the core's one-day step (B, 66, H, W)."""
        return core_step(self.core, xd, doy, tisr)

    def bandwidth(self, xd: torch.Tensor, g: Optional[torch.Tensor], tend: torch.Tensor,
                  doy: torch.Tensor) -> torch.Tensor:
        """h(x) (B, 66, H, W) in units of s, from the state, its physical terms and the equation's tendency."""
        B, _, H, W = xd.shape
        parts = [xd] + ([g] if g is not None else []) + [tend, self.core.static.expand(B, -1, -1, -1),
                                                         time_features(doy.float() * 24.0, H, W)]
        return F.softplus(self.bw(torch.cat(parts, 1)) + self.h_bias) + self.cfg.h_min

    def centres(self, xd: torch.Tensor, doy: torch.Tensor, tisr: Optional[torch.Tensor], n: int,
                gen: Optional[torch.Generator] = None, antithetic: bool = True) -> torch.Tensor:
        """(B, n, 66, H, W): Phi(x + eps s delta_k)."""
        B = xd.shape[0]
        if antithetic and n % 2 == 0:
            d = smooth_noise((B, n // 2) + tuple(xd.shape[1:]), xd, gen)
            d = torch.cat([d, -d], 1)
        else:
            d = smooth_noise((B, n) + tuple(xd.shape[1:]), xd, gen)
        x0 = xd[:, None] + self.eps().unsqueeze(0) * self.s.unsqueeze(0) * self.keep.unsqueeze(0) * d
        rep = lambda t: None if t is None else t.repeat_interleave(n, 0)  # noqa: E731
        c = self.phi(x0.flatten(0, 1), rep(doy), rep(tisr))
        return c.view(B, n, *xd.shape[1:])

    def tend(self, xd, phi0) -> torch.Tensor:
        """The equation's own day from x, (Phi(x) - x) / s, as an input (no gradient)."""
        return ((phi0 - xd) / self.s * self.keep).detach()

    def forward(self, xd, doy, tisr, n: Optional[int] = None, gen=None):
        """(centres (B, n, 66, H, W), bandwidth h (B, 66, H, W), G(x) or None, Phi(x) (B, 66, H, W))."""
        n = n or self.cfg.n_centres
        c = self.centres(xd, doy, tisr, n, gen)
        with torch.no_grad():
            phi0 = self.phi(xd, doy, tisr)
        g = self.G(xd)
        h = self.bandwidth(xd, g, self.tend(xd, phi0), doy)
        return c, h, g, phi0

    # ---- the distribution
    def nll(self, c: torch.Tensor, h: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Composite negative log-likelihood of y (B, 66, H, W) under the kernel sum, per channel and grid point,
        area-weighted (w (1, 1, H, 1)), in nats per value and in units of s (tisr excluded)."""
        z = (y[:, None] - c) / self.s.unsqueeze(0)                                  # (B, n, 66, H, W)
        lp = -0.5 * (z / h[:, None]) ** 2 - torch.log(h[:, None]) - 0.5 * LOG2PI
        lp = torch.logsumexp(lp, 1) - math.log(c.shape[1])                         # (B, 66, H, W)
        m = self.keep * w
        return -(lp * m).sum() / (m.expand_as(lp).sum())

    def variance(self, c: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Variance of K(x, .) per value (B, 66, H, W), units of s^2: spread of the centres + bandwidth^2."""
        vc = ((c - c.mean(1, keepdim=True)) / self.s.unsqueeze(0)).pow(2).mean(1)
        return vc + h ** 2

    def entropy(self, c: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Gaussian (moment) entropy of the kernel per value, nats in units of s: 0.5 log(2 pi e var). An upper
        bound of the kernel sum's own marginal entropy; the one-day entropy production of the equation."""
        return 0.5 * (LOG2PI + 1.0 + torch.log(self.variance(c, h)))

    def sample(self, xd, doy, tisr, gen=None):
        """One draw of K(x, .) per row of xd (a member): a fresh centre Phi(x + eps s delta) and the bandwidth
        noise s h xi (xi smooth unit noise). Returns (y0 (B, 66, H, W), h, G(x), Phi(x))."""
        c = self.centres(xd, doy, tisr, 1, gen, antithetic=False)[:, 0]
        phi0 = self.phi(xd, doy, tisr)
        g = self.G(xd)
        h = self.bandwidth(xd, g, self.tend(xd, phi0), doy)
        y0 = c + self.s * h * smooth_noise(tuple(xd.shape), xd, gen) * self.keep
        return y0, h, g, phi0
