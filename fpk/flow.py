"""Flow matching from the physical kernel to the data: K(x, .) -> p(x_{t+1} | x_t).

Conditional flow matching with an informed source. For one day pair (x, y):

    z0 = (y0 - x) / s,   y0 ~ K(x, .)          the source: a draw of the physical kernel (fpk.kernel)
    z1 = (y  - x) / s                           the target: the observed next day
    z_tau = (1 - tau) z0 + tau z1 + sigma_min n,   u = z1 - z0
    loss  = || v_theta(z_tau, tau | x, G(x), Phi(x), h(x)) - u ||^2   (area-weighted, tisr excluded)

Source and target are drawn independently given x, so the learned flow transports the kernel distribution K(x, .)
to the conditional law of the next day (the marginal-preserving property of conditional flow matching). The
velocity head starts at zero: before training the flow is the identity and the model IS the physical kernel. What
the flow learns is only the part the equation-kernel gets wrong (its bias, its shape), so the transport is short.

``source = "gauss"`` is the control: the same network and conditioning, the source a smooth N(0, I) in units of s
around x (standard flow matching): the physics then enters as conditioning only, not as the distribution.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .env import time_features
from .kernel import PhysicsKernel, smooth_noise
from pgcfm.core import Net  # noqa: E402
from pgcfm.seqdct import _NetCfg  # noqa: E402


@dataclass
class FlowConfig:
    hidden: int = 64
    blocks: tuple = (2, 2, 2)
    sigma_min: float = 0.02
    steps: int = 6           # Heun steps of the sampler
    source: str = "kernel"   # kernel | gauss


def tau_features(tau: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """(B, 4, H, W): tau, sin / cos of pi tau, sin 2 pi tau."""
    f = torch.stack([tau, torch.sin(math.pi * tau), torch.cos(math.pi * tau), torch.sin(2 * math.pi * tau)], 1)
    return f[:, :, None, None].expand(-1, -1, H, W)


class KernelFlow(nn.Module):
    def __init__(self, kernel: PhysicsKernel, cfg: FlowConfig = FlowConfig()):
        super().__init__()
        self.kernel, self.cfg = kernel, cfg
        for p in kernel.parameters():
            p.requires_grad_(False)
        core = kernel.core
        n_g = core.gscale.shape[1] if kernel.gfeat else 0
        # z_tau, x, G(x), (Phi(x) - x) / s, log h, static, time of year, tau
        c_in = 66 + 66 + n_g + 66 + 66 + core.static.shape[0] + 4 + 4
        self.net = Net(c_in, 66, _NetCfg(cfg.hidden, tuple(cfg.blocks)))

    @property
    def s(self):
        return self.kernel.s

    @property
    def keep(self):
        return self.kernel.keep

    def cond(self, xd, g, phi0, h, doy):
        B, _, H, W = xd.shape
        parts = [xd] + ([g] if g is not None else []) + [self.kernel.tend(xd, phi0), torch.log(h),
                                                         self.kernel.core.static.expand(B, -1, -1, -1),
                                                         time_features(doy.float() * 24.0, H, W)]
        return torch.cat(parts, 1)

    def velocity(self, z, tau, cond):
        H, W = z.shape[-2:]
        return self.net(torch.cat([z, cond, tau_features(tau, H, W)], 1)) * self.keep

    def source(self, xd, doy, tisr, gen=None):
        """(z0, cond): a draw of the source in increment units, and the conditioning of the velocity."""
        with torch.no_grad():
            y0, h, g, phi0 = self.kernel.sample(xd, doy, tisr, gen)
            if self.cfg.source == "gauss":
                z0 = smooth_noise(tuple(xd.shape), xd, gen) * self.keep
            else:
                z0 = (y0 - xd) / self.s * self.keep
            return z0, self.cond(xd, g, phi0, h, doy)

    def loss(self, xd, yd, doy, tisr, w, gen=None) -> torch.Tensor:
        z0, cond = self.source(xd, doy, tisr, gen)
        z1 = (yd - xd) / self.s * self.keep
        B = xd.shape[0]
        tau = torch.rand(B, generator=gen).to(xd)
        t = tau.view(B, 1, 1, 1)
        zt = (1 - t) * z0 + t * z1 + self.cfg.sigma_min * torch.randn_like(z0) * self.keep
        v = self.velocity(zt, tau, cond)
        m = self.keep * w
        sq = (v - (z1 - z0)) ** 2
        return (sq * m).sum() / m.expand_as(sq).sum()

    @torch.no_grad()
    def sample(self, xd, doy, tisr, gen=None, steps: Optional[int] = None, flow: bool = True):
        """One draw of the next day per row of xd: the kernel draw transported by the learned flow (Heun);
        ``flow=False``: the kernel draw itself. Returns (y (B, 66, H, W), z0 the source draw)."""
        z, cond = self.source(xd, doy, tisr, gen)
        z0 = z
        if flow:
            n = steps or self.cfg.steps
            dt = 1.0 / n
            for i in range(n):
                t = torch.full((xd.shape[0],), i * dt, device=xd.device)
                k1 = self.velocity(z, t, cond)
                k2 = self.velocity(z + dt * k1, t + dt, cond)
                z = z + 0.5 * dt * (k1 + k2)
        y = xd + self.s * z
        return self.kernel.core.finish(y, tisr), z0
