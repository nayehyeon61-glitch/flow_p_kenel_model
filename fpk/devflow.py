"""Flow matching from the entropy deviation kernel: the flow learns the distribution the equation produced.

The deviation kernel (fpk.devkernel) gives, for every lead L, the mean xbar_L (the core's trajectory) and deviations
e_L ~ N(0, Sigma_L): directions carried by the equation, size from the entropy budget H_L. That Gaussian is right in
its second moment (E[IGN] = H) but the true error of the mean is not Gaussian and its structure in space is not just
the equation's. So the kernel is the SOURCE of conditional flow matching, in units of s:

    source   z0 = e_L                      a draw of the deviation kernel at lead L
    target   z1 = (x_L - xbar_L) / s        the data's deviation from the mean
    path     z_tau = (1 - tau) z0 + tau z1 (+ sigma_min noise),   velocity target z1 - z0
    v_theta(z_tau, tau | xbar_L, G(xbar_L), H_L, lead, time of year, static)       zero head: starts as the kernel

The net works in units of the predicted spread sigma_L = exp(H_L) (input z / sigma, output x sigma) and the loss is
the velocity error in those units, so every lead weighs the same (in units of s the day-10 error is 8 x the day-1's).

    member   x^m_L = xbar_L + s T(e^m_L)    T = the flow's transport (Heun steps)

The kernel's own deviations go on through the equation (the flow transports each lead's distribution, it is not fed
back), so training (sources from the kernel's rollout) and forecasting see the same source distribution.
``source="gauss"``: the control, smooth unit Gaussian noise as the source with the same conditioning (H included).
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .devkernel import HALF_LOG_2PIE, EntropyDeviationKernel
from .env import time_features
from .flow import tau_features
from .kernel import smooth_noise
from pgcfm.core import Net  # noqa: E402
from pgcfm.kai_tend import stateless_norm  # noqa: E402
from pgcfm.seqdct import _NetCfg  # noqa: E402


@dataclass
class DevFlowConfig:
    hidden: int = 64
    blocks: tuple = (2, 2, 2)
    sigma_min: float = 0.02
    steps: int = 6
    source: str = "kernel"   # kernel | gauss
    path: str = "line"       # line: the straight line in a pseudo time (this class); core: the tendency core's own
    #                          flow over the physical day as the path (fpk.coreflow.CoreFlow, built by build_flow)
    core_steps: int = 4      # path = core: the steps of the day's grid t_k = k / N, each one Heun step of the core's
    #                          tendency over 1 / N day; 0 = the core's own substeps (its trained solver)
    core_sub: int = 0        # path = core: 0 = one Euler step of the learned part a grid step (the network sees z_k);
    #                          M >= 1 = the learned part of a grid step is a line flow after the core's step, M Heun steps


class DeviationFlow(nn.Module):
    def __init__(self, dk: EntropyDeviationKernel, cfg: DevFlowConfig = DevFlowConfig()):
        super().__init__()
        self.dk, self.cfg = dk, cfg
        for p in dk.parameters():
            p.requires_grad_(False)
        core = dk.core
        n_g = core.gscale.shape[1] if dk.gfeat else 0
        # z_tau, xbar, G(xbar), H, static, time of year, lead, tau
        c_in = 66 + 66 + n_g + 66 + core.static.shape[0] + 4 + 1 + 4
        self.net = stateless_norm(Net(c_in, 66, _NetCfg(cfg.hidden, tuple(cfg.blocks))))

    def train(self, mode: bool = True):
        super().train(mode)
        self.dk.eval()
        return self

    @torch.no_grad()
    def cond(self, xbar, H, doy, lead: int):
        """The conditioning at a lead: xbar (B, 66, H, W) the mean of that day, H its predicted entropy field,
        doy (B,) the day of year of that day, lead = days since the IC."""
        B, _, Hh, Ww = xbar.shape
        core = self.dk.core
        parts = [xbar] + ([core.features(xbar)] if self.dk.gfeat else []) + [
            H * self.dk.keep, core.static.expand(B, -1, -1, -1), time_features(doy.float() * 24.0, Hh, Ww),
            xbar.new_full((B, 1, Hh, Ww), lead / 10.0)]
        return torch.cat(parts, 1)

    def source(self, e, gen=None, sig=None):
        """e (B, P, 66, H, W) the kernel's deviations -> the source draws. The control ``gauss``: smooth Gaussian
        noise of the kernel's predicted size sig (B, 1, 66, H, W) (the same variance, none of the equation's
        directions)."""
        if self.cfg.source == "gauss":
            return smooth_noise(tuple(e.shape), e, gen) * self.dk.keep.unsqueeze(0) * (1.0 if sig is None else sig)
        return e

    def sigma(self, H):
        """exp(H) (B, 66, H, W), 1 on tisr (no deviation there)."""
        k = self.dk.keep
        return torch.exp(H.detach().clamp(-6.0, 6.0)) * k + (1 - k)

    def velocity(self, z, tau, cond, sig):
        Hh, Ww = z.shape[-2:]
        return self.net(torch.cat([z / sig, cond, tau_features(tau, Hh, Ww)], 1)) * sig * self.dk.keep

    def loss(self, e, xbar, truth, cond, H, w, gen=None):
        """Flow-matching loss of one lead: every one of the P deviations is a source draw for the same target."""
        B, P = e.shape[:2]
        z0 = self.source(e, gen, self.sigma(H)[:, None]).flatten(0, 1)
        z1 = ((truth - xbar) / self.dk.s * self.dk.keep).repeat_interleave(P, 0)
        c, sig = cond.repeat_interleave(P, 0), self.sigma(H).repeat_interleave(P, 0)
        tau = torch.rand(B * P, generator=gen).to(z0)
        t = tau.view(-1, 1, 1, 1)
        zt = (1 - t) * z0 + t * z1 + self.cfg.sigma_min * sig * torch.randn(z0.shape, generator=gen).to(z0) * self.dk.keep
        sq = ((self.velocity(zt, tau, c, sig) - (z1 - z0)) / sig) ** 2
        m = self.dk.keep * w
        return (sq * m).sum() / m.expand_as(sq).sum()

    @torch.no_grad()
    def transport(self, e, cond, H, gen=None, steps: Optional[int] = None):
        """(B, P, 66, H, W) -> the transported deviations (Heun)."""
        B, P = e.shape[:2]
        z = self.source(e, gen, self.sigma(H)[:, None]).flatten(0, 1)
        c, sig = cond.repeat_interleave(P, 0), self.sigma(H).repeat_interleave(P, 0)
        n = steps or self.cfg.steps
        dt = 1.0 / n
        z0, kin = z, 0.0
        for i in range(n):
            t = torch.full((z.shape[0],), i * dt, device=z.device)
            k1 = self.velocity(z, t, c, sig)
            k2 = self.velocity(z + dt * k1, t + dt, c, sig)
            v = 0.5 * (k1 + k2)
            kin = kin + dt * (v / sig).pow(2).mean()
            z = z + dt * v
        # how straight the paths are (Benamou-Brenier): the kinetic energy int |v|^2 dtau over the squared
        # displacement |T(z) - z|^2 (units of sigma); 1 = straight lines at constant speed, the OT geodesic
        disp = ((z - z0) / sig).pow(2).mean()
        self.last_path = (float(kin), float(disp))
        return z.view(B, P, *z.shape[1:])

    # ---- one day of the kernel -> its loss / its members: the calls a trainer or a rollout makes right after
    # ``dk.day`` (x0 the mean before the day, xb the mean after it, e and H what the day returned, doy0 the IC's day
    # of year, tisr the day's forcing, lead = days since the IC of the day's start). The same for every path.
    def day_loss(self, x0, xb, e, H, truth, doy0, tisr, lead: int, w, gen=None, sel=None):
        """The flow-matching loss of the day; ``sel``: the indices of the kernel's pairs used as source draws."""
        es = e if sel is None else e[:, sel]
        return self.loss(es, xb, truth, self.cond(xb, H, doy0 + lead + 1, lead + 1), H, w, gen)

    @torch.no_grad()
    def day_transport(self, x0, xb, e, H, doy0, tisr, lead: int, gen=None, steps: Optional[int] = None):
        """The members' deviations of the day (B, P, 66, H, W), units of s around xb."""
        return self.transport(e, self.cond(xb, H, doy0 + lead + 1, lead + 1), H, gen, steps)

    @torch.no_grad()
    def rollout(self, xd0, doy0, tisr, leads: int, hist=None, gen=None, steps: Optional[int] = None):
        """Members (B, P, L, 66, H, W) = xbar_L + s T(e^p_L) (P independent members, P = the kernel's pairs) and the
        kernel's predicted entropy (B, L, 66, H, W)."""
        dk = self.dk
        e = dk.start(xd0, doy0, hist, gen)
        x, out, ent = xd0, [], []
        for L in range(leads):
            x0 = x
            x, e, H, _ = dk.day(x, e, doy0 + L, tisr[:, L], L, gen)
            z = self.day_transport(x0, x, e, H, doy0, tisr[:, L], L, gen, steps)
            out.append(x[:, None] + dk.s.unsqueeze(0) * z)
            ent.append(H + HALF_LOG_2PIE)
        return torch.stack(out, 2), torch.stack(ent, 1)
