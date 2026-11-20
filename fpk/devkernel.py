"""The deviation kernel shaped by entropy: mean = the equation's trajectory, spread = an entropy budget.

The physical kernel of fpk.kernel lets every member run its own noisy trajectory; the ensemble mean then loses to the
deterministic core and the spread overshoots at long leads (docs/design_ko.md, sections 5 and 6). Here the two jobs
are separated:

    mean        xbar_(L+1) = Phi(xbar_L)                     the tendency core's own trajectory (with its memory)
    members     x^m_L = xbar_L +- s e^p_L                    P antithetic pairs of deviations e (units of s)

    direction   p^p = [Phi(xbar + s e^p) - Phi(xbar - s e^p)] / (2 s)        the equation carries the deviations:
                                                                             the perturbation pairs of the kernel
    new part    q^p = p^p + h xi^p                                           what the equation does not determine
    size        e^p_(L+1) = q^p exp(r)                                       the entropy budget:

    H_(L+1) = 1/2 log <q^2>  +  r(xbar, G, m, ...)       per channel and grid point, nats in units of s (+ const)
              ---------------    ------------------
              the equation's     the learned residual
              stretching of      of the day's entropy
              the cloud          production

so the day's entropy production is H_(L+1) - H_L = (stretching rate of the equation) + (residual), and the kernel
of the deviations is N(e; 0, Sigma) with the equation's directions and the volume set by the accumulated entropy.
<q^2> is the mean over the pairs, smoothed over ``smooth`` x ``smooth`` cells.

Training: the Gaussian log score (IGN) of the data around the mean with the predicted entropy,
    IGN = H + err^2 / (2 exp(2 H)),    err = (truth - xbar) / s,     (H written without the constant 1/2 log 2 pi e)
on every day of a rollout. Its minimum is the calibrated entropy, E[IGN] = H: the spread is the size of the actual
error of the mean, so the long-lead overshoot is penalized directly. The core is frozen and no gradient goes through
it (r and h are the trained parts, the deviations are detached between days).
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import time_features
from .env import PGFM  # noqa: F401  (sys.path)
from .kernel import ADV_STATE, smooth_noise
from pgcfm.core import Net  # noqa: E402
from pgcfm.kai_tend import core_step, stateless_norm  # noqa: E402
from pgcfm.seqdct import _NetCfg  # noqa: E402

HALF_LOG_2PIE = 0.5 * (math.log(2 * math.pi) + 1.0)


@dataclass
class DevConfig:
    pairs: int = 4            # P antithetic pairs = 2 P members
    eps0: float = 0.25        # size of the initial deviations (units of s); its entropy is corrected by r on day 1
    h0: float = 0.3           # start of the new part's size (units of s), learned per channel
    smooth: int = 5           # cells of the smoothing of <q^2>
    r_max: float = 2.0        # |r| <= r_max (a day cannot change the size by more than e^r_max)
    hidden: int = 32
    blocks: tuple = (2, 2)


class EntropyDeviationKernel(nn.Module):
    def __init__(self, core, cfg: DevConfig = DevConfig()):
        super().__init__()
        self.core, self.cfg = core, cfg
        for p in core.parameters():
            p.requires_grad_(False)
        self.gfeat = bool(getattr(core.cfg, "gfeat", False))
        n_g = core.gscale.shape[1] if self.gfeat else 0
        self.mem = int(getattr(core, "mem", 0))
        self.mem_ch = int(core.mem_ch) if self.mem else 0
        # xbar, G(xbar), (Phi(xbar) - xbar) / s, log size before / after the equation's day, static, time, lead, m
        c_in = 66 + n_g + 66 + 66 + 66 + core.static.shape[0] + 4 + 1 + self.mem_ch
        self.net = stateless_norm(Net(c_in, 66, _NetCfg(cfg.hidden, tuple(cfg.blocks))))  # zero head: r = bias
        self.r_bias = nn.Parameter(torch.zeros(1, 66, 1, 1))
        self.h_raw = nn.Parameter(torch.full((66,), math.log(math.expm1(cfg.h0))))

    def train(self, mode: bool = True):
        super().train(mode)
        self.core.eval()
        return self

    @property
    def s(self):
        return self.core.t6

    @property
    def keep(self):
        return self.core.keep

    def h(self):
        return F.softplus(self.h_raw).view(1, 66, 1, 1)

    # ---- pieces
    def size(self, e: torch.Tensor) -> torch.Tensor:
        """1/2 log of the smoothed mean square over the pairs: (B, P, 66, H, W) -> (B, 66, H, W)."""
        v, k = e.pow(2).mean(1), self.cfg.smooth
        if k > 1:
            r = k // 2
            v = F.avg_pool2d(F.pad(F.pad(v, (r, r, 0, 0), mode="circular"), (0, 0, r, r), mode="reflect"), k, 1)
        return 0.5 * torch.log(v + 1e-8)

    @torch.no_grad()
    def start(self, xd, doy, hist=None, gen=None):
        """The trajectory starts at x (B, 66, H, W): the mean's memory (memory core) and the initial deviations
        e_0 (B, P, 66, H, W) = eps0 x smooth unit noise."""
        if self.mem:
            self.core.reset(hist, doy.float() * 24.0, xd)
        B, P = xd.shape[0], self.cfg.pairs
        return self.cfg.eps0 * smooth_noise((B, P) + tuple(xd.shape[1:]), xd, gen) * self.keep.unsqueeze(0)

    @torch.no_grad()
    def _equation(self, xbar, e, doy, tisr):
        """The mean's day and the deviations carried by the equation: (Phi(xbar), p (B, P, 66, H, W), G(xbar))."""
        B, P = e.shape[:2]
        core, hours = self.core, doy.float() * 24.0
        if self.mem:
            core.remember(xbar)
            core.started = True
            phi0 = core._advance(xbar, hours, tisr)
        else:
            phi0 = core_step(core, xbar, doy, tisr)
        g = core.features(xbar) if self.gfeat else None
        n = 2 * P
        xs = (xbar[:, None] + self.s.unsqueeze(0) * torch.cat([e, -e], 1)).flatten(0, 1)
        rep = lambda t: None if t is None else t.repeat_interleave(n, 0)  # noqa: E731
        if self.mem:
            keep = {k: getattr(core, k, None) for k in ADV_STATE}
            try:
                for k, v in keep.items():
                    if v is not None:
                        setattr(core, k, rep(v))
                y = core._advance(xs, rep(hours), rep(tisr))
            finally:
                for k, v in keep.items():
                    setattr(core, k, v)
            core.fc = phi0  # the mean is the core's own trajectory
        else:
            y = core_step(core, xs, rep(doy), rep(tisr))
        y = y.view(B, n, *xbar.shape[1:])
        p = (y[:, :P] - y[:, P:]) / (2 * self.s.unsqueeze(0)) * self.keep.unsqueeze(0)
        return phi0, p, g

    def day(self, xbar, e, doy, tisr, lead: int, gen=None):
        """One day. xbar (B, 66, H, W) the mean, e (B, P, 66, H, W) the deviations of the day, ``lead`` = days since
        the IC (0 for the first day). Returns (xbar_next, e_next, H_next (B, 66, H, W), parts): H_next = 1/2 log of
        the variance of the next day in units of s^2; parts = (stretch, r): the day's entropy production split
        into the equation's stretching 1/2 log <p^2> / <e^2> (+ the new part) and the learned residual."""
        B, _, Hh, Ww = xbar.shape
        phi0, p, g = self._equation(xbar, e, doy, tisr)
        xi = smooth_noise(tuple(p.shape), xbar, gen) * self.keep.unsqueeze(0)
        q = p + self.h().unsqueeze(0) * xi
        le, lq = self.size(e), self.size(q)
        tend = ((phi0 - xbar) / self.s * self.keep).detach()
        parts = [xbar] + ([g] if g is not None else []) + [tend, le.detach(), lq.detach(),
                                                           self.core.static.expand(B, -1, -1, -1),
                                                           time_features(doy.float() * 24.0, Hh, Ww),
                                                           xbar.new_full((B, 1, Hh, Ww), lead / 10.0)]
        if self.mem:
            parts.append(self.core.m)
        r = self.cfg.r_max * torch.tanh((self.net(torch.cat(parts, 1)) + self.r_bias) / self.cfg.r_max) * self.keep
        e_next = q * torch.exp(r)[:, None]
        return phi0, e_next, lq + r, (lq - le, r)

    def ign(self, H, xbar, truth, w):
        """The Gaussian log score of the truth around the mean with entropy H (without 1/2 log 2 pi), area-weighted,
        tisr excluded; and the mean entropy itself: calibrated when the two are 1/2 apart (E[err^2] = exp(2 H))."""
        err = (truth - xbar) / self.s
        v = H + 0.5 * err.pow(2) * torch.exp(-2 * H)
        m = self.keep * w
        n = m.expand_as(v).sum()
        return (v * m).sum() / n, (H * m).sum() / n

    @staticmethod
    def members(xbar, e, s):
        """(B, 2 P, 66, H, W): the member states xbar +- s e."""
        return xbar[:, None] + s.unsqueeze(0) * torch.cat([e, -e], 1)

    @torch.no_grad()
    def rollout(self, xd0, doy0, tisr, leads: int, hist=None, gen=None):
        """States (B, 2 P, L, 66, H, W) and the predicted entropy (B, L, 66, H, W) in nats per value in units of s
        (the full Gaussian entropy, 1/2 log 2 pi e var)."""
        e = self.start(xd0, doy0, hist, gen)
        x, out, ent = xd0, [], []
        for L in range(leads):
            x, e, H, _ = self.day(x, e, doy0 + L, tisr[:, L], L, gen)
            out.append(self.members(x, e, self.s))
            ent.append(H + HALF_LOG_2PIE)
        return torch.stack(out, 2), torch.stack(ent, 1)
