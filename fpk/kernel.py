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
MEM_STATE = ("m", "prev", "gprev", "fc", "z", "M", "PD")  # a memory core's tensors of one trajectory (pgcfm)
ADV_STATE = ("m", "z", "M")                                # ... the ones its day (``_advance``) reads


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
    eps_m0: float = 0.1       # start of eps_m: the spread of the memory cloud (memory core only)


class PhysicsKernel(nn.Module):
    """K(x, .) of a frozen tendency core ``core`` (pgcfm DynCore, or its memory core).

    States are the 66 dynamic channels, normalized; ``doy`` (B,) day of year of x; ``tisr`` (B, H, W) the forcing
    of the next day.

    Memory core (Mori-Zwanzig memory m, e.g. core_gngmz8): the kernel sum runs over a cloud of (state, memory),

        K(x, m, y) = (1/N) sum_k N( y ; Phi(x + eps s delta_k ; m + eps_m delta'_k) , diag (s h(x, G, m))^2 )

    Every trajectory (member) carries its own memory: ``start`` fills it from the W days before the IC (teacher
    forcing, as the core was trained), and every day ``observe`` (called by ``forward`` / ``sample``) updates it
    once with the member's state; the innovation it sees is the member's state minus the equation's forecast
    Phi(x; m) of the day before (no perturbation)."""

    def __init__(self, core, cfg: KernelConfig = KernelConfig()):
        super().__init__()
        self.core, self.cfg = core, cfg
        for p in core.parameters():
            p.requires_grad_(False)
        self.gfeat = bool(getattr(core.cfg, "gfeat", False))
        n_g = core.gscale.shape[1] if self.gfeat else 0
        n_static = core.static.shape[0]
        self.mem = int(getattr(core, "mem", 0))
        self.mem_ch = int(core.mem_ch) if self.mem else 0
        c_in = 66 + n_g + 66 + n_static + 4 + self.mem_ch
        # GroupNorm, no dropout (ClimODE's block has BatchNorm + Dropout: batch-dependent, and train != eval)
        self.bw = stateless_norm(Net(c_in, 66, _NetCfg(cfg.hidden, tuple(cfg.blocks))))
        # zero head: the bandwidth starts at h0 everywhere
        self.h_bias = nn.Parameter(torch.full((1, 66, 1, 1), math.log(math.expm1(cfg.h0 - cfg.h_min))))
        self.eps_raw = nn.Parameter(torch.full((66,), math.log(math.expm1(cfg.eps0))))
        if self.mem:
            self.eps_m_raw = nn.Parameter(torch.full((self.mem_ch,), math.log(math.expm1(cfg.eps_m0))))

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

    def eps_m(self) -> torch.Tensor:
        return F.softplus(self.eps_m_raw).view(1, -1, 1, 1)

    def G(self, xd: torch.Tensor) -> Optional[torch.Tensor]:
        return self.core.features(xd) if self.gfeat else None

    # ---- the memory of a trajectory (no-ops for a plain core)
    @torch.no_grad()
    def start(self, xd: torch.Tensor, doy: torch.Tensor, hist: Optional[torch.Tensor] = None, members: int = 1):
        """A trajectory starts at x (B, 66, H, W): the memory from the W days before it (hist (B, W, 66, H, W),
        oldest first, None = empty), then copied to ``members`` rows per IC (repeat_interleave order)."""
        if not self.mem:
            return
        self.core.reset(hist, doy.float() * 24.0, xd)
        self.tile(members)

    def tile(self, members: int):
        if not self.mem or members == 1:
            return
        for k in MEM_STATE:
            v = getattr(self.core, k, None)
            if torch.is_tensor(v):
                setattr(self.core, k, v.repeat_interleave(members, 0))
        self.core.past = [v.repeat_interleave(members, 0) for v in getattr(self.core, "past", [])]
        self.core.seq = None

    @torch.no_grad()
    def observe(self, xd: torch.Tensor):
        """The day's state enters the memory (once a day, before the day's step)."""
        if self.mem:
            self.core.remember(xd)
            self.core.started = True

    def memory(self) -> Optional[torch.Tensor]:
        return self.core.m if self.mem else None

    def phi(self, xd: torch.Tensor, doy: torch.Tensor, tisr: Optional[torch.Tensor]) -> torch.Tensor:
        """The equation's day: the core's one-day step (B, 66, H, W); a memory core with its present memory
        (no memory update here, see ``observe``)."""
        if self.mem:
            return self.core._advance(xd, doy.float() * 24.0, tisr)
        return core_step(self.core, xd, doy, tisr)

    def bandwidth(self, xd: torch.Tensor, g: Optional[torch.Tensor], tend: torch.Tensor,
                  doy: torch.Tensor) -> torch.Tensor:
        """h(x) (B, 66, H, W) in units of s, from the state, its physical terms and the equation's tendency."""
        B, _, H, W = xd.shape
        parts = [xd] + ([g] if g is not None else []) + [tend, self.core.static.expand(B, -1, -1, -1),
                                                         time_features(doy.float() * 24.0, H, W)]
        if self.mem:
            parts.append(self.core.m)
        return F.softplus(self.bw(torch.cat(parts, 1)) + self.h_bias) + self.cfg.h_min

    def centres(self, xd: torch.Tensor, doy: torch.Tensor, tisr: Optional[torch.Tensor], n: int,
                gen: Optional[torch.Generator] = None, antithetic: bool = True) -> torch.Tensor:
        """(B, n, 66, H, W): Phi(x + eps s delta_k) (memory core: with the memory m + eps_m delta'_k)."""
        B = xd.shape[0]

        def noise(shape):
            if antithetic and n % 2 == 0:
                d = smooth_noise((B, n // 2) + shape, xd, gen)
                return torch.cat([d, -d], 1)
            return smooth_noise((B, n) + shape, xd, gen)
        d = noise(tuple(xd.shape[1:]))
        x0 = xd[:, None] + self.eps().unsqueeze(0) * self.s.unsqueeze(0) * self.keep.unsqueeze(0) * d
        rep = lambda t: None if t is None else t.repeat_interleave(n, 0)  # noqa: E731
        if not self.mem:
            c = self.phi(x0.flatten(0, 1), rep(doy), rep(tisr))
            return c.view(B, n, *xd.shape[1:])
        keep = {k: getattr(self.core, k, None) for k in ADV_STATE}
        try:
            for k, v in keep.items():
                if v is not None:
                    setattr(self.core, k, rep(v))
            m = keep["m"]
            dm = noise(tuple(m.shape[1:]))
            self.core.m = self.core.m + (self.eps_m().unsqueeze(0) * dm).flatten(0, 1)
            c = self.phi(x0.flatten(0, 1), rep(doy), rep(tisr))
        finally:
            for k, v in keep.items():
                setattr(self.core, k, v)
        return c.view(B, n, *xd.shape[1:])

    def tend(self, xd, phi0) -> torch.Tensor:
        """The equation's own day from x, (Phi(x) - x) / s, as an input (no gradient)."""
        return ((phi0 - xd) / self.s * self.keep).detach()

    def _phi0(self, xd, doy, tisr):
        """Phi(x) without perturbation (no gradient); a memory core takes it as the forecast that the next day's
        state is compared with (the innovation of its memory)."""
        with torch.no_grad():
            phi0 = self.phi(xd, doy, tisr)
        if self.mem:
            self.core.fc = phi0
        return phi0

    def forward(self, xd, doy, tisr, n: Optional[int] = None, gen=None, observe: bool = True):
        """(centres (B, n, 66, H, W), bandwidth h (B, 66, H, W), G(x) or None, Phi(x) (B, 66, H, W)).
        ``observe``: x enters the memory first (memory core; False when it already has today)."""
        n = n or self.cfg.n_centres
        if observe:
            self.observe(xd)
        c = self.centres(xd, doy, tisr, n, gen)
        phi0 = self._phi0(xd, doy, tisr)
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

    def sample(self, xd, doy, tisr, gen=None, observe: bool = True):
        """One draw of K(x, .) per row of xd (a member): a fresh centre Phi(x + eps s delta) and the bandwidth
        noise s h xi (xi smooth unit noise). Returns (y0 (B, 66, H, W), h, G(x), Phi(x))."""
        if observe:
            self.observe(xd)
        c = self.centres(xd, doy, tisr, 1, gen, antithetic=False)[:, 0]
        phi0 = self._phi0(xd, doy, tisr)
        g = self.G(xd)
        h = self.bandwidth(xd, g, self.tend(xd, phi0), doy)
        y0 = c + self.s * h * smooth_noise(tuple(xd.shape), xd, gen) * self.keep
        return y0, h, g, phi0
