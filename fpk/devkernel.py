"""The deviation kernel shaped by entropy: mean = the equation's trajectory, spread = an entropy budget.

The physical kernel of fpk.kernel lets every member run its own noisy trajectory; the ensemble mean then loses to the
deterministic core and the spread overshoots at long leads (docs/design_ko.md, sections 5 and 6). Here the two jobs
are separated:

    mean        xbar_(L+1) = Phi(xbar_L)                     the tendency core's own trajectory (with its memory);
                                                             ``cloud_mean``: the average of Phi over the members
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

``correct``: the chain of transition kernels p(x_t | x_(t-1)) trained as a 10-day sequence. Every link is
    transport   the equation's day of the mean and of the deviations (above)
    correction  at the new x_t: mean += s exp(H) delta(x_t, G(x_t), cloud shift, m, ...), H += r2(...)
and the sum of the days' log scores is the log likelihood of the sequence under the chain's Gaussian marginals. The
links see the states the chain itself produced (not the data's), so the correction learns what the equation's
day-L state still tells about the truth: the unpredictable part of the mean is removed and goes to the entropy.
"""

import math

import numpy as np
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import time_features
from .env import PGFM  # noqa: F401  (sys.path)
from .kernel import ADV_STATE, MEM_STATE, smooth_noise
from .sphere import SHT, bands
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
    cloud_mean: bool = False  # the mean = the average of the equation over the cloud (not the core's one trajectory)
    correct: bool = False     # the chained kernel p(x_t | x_(t-1)): at the new x_t a correction of the mean from the
    #                           primitive-equation terms G(x_t) and a correction of the entropy (sequence training)
    d_max: float = 0.25       # the mean's correction of one day is at most d_max x the cloud's size exp(H)
    r_dissip: bool = False    # r = -softplus(-u) <= 0: the noise spectrum makes all the positive production
    #                           (r_plus = 1/2 log(1 + Q / <p^2>)), the learned residual only removes (r_minus)
    band_scale: bool = False  # entropy by scale: the deviations are rescaled per Littlewood-Paley band of degrees,
    #                           channel and lead, exp(g[lead, c, j]) (starts at 1); H is then the size of the result
    noise: str = "smooth"     # the new part's field: "smooth" (3 x 3 mean of white noise) or "sphere": spherical
    #                           harmonics with a learned amplitude per Littlewood-Paley band and channel (starts as
    #                           the isotropic copy of "smooth") and an AR(1) correlation in time per band


class EntropyDeviationKernel(nn.Module):
    def __init__(self, core, cfg: DevConfig = DevConfig()):
        super().__init__()
        self.core, self.cfg = core, cfg
        self.through = 0  # training: days of the gradient through the mean's path of the equation (0 = none)
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
        if cfg.noise == "sphere" or cfg.band_scale:
            H = int(core.static.shape[-2])
            lat = [-90.0 + 180.0 / H * (i + 0.5) for i in range(H)]  # the data's equiangular grid, south to north
            self.sht = SHT(lat)
            bj, nb = bands(self.sht.L)
            self.register_buffer("band_of", bj)
            self.n_bands = nb
        if cfg.band_scale:
            self.g_band = nn.Parameter(torch.zeros(10, 66, nb))  # log scale per lead (day 1..10), channel, band
        if cfg.noise == "sphere":
            # the degree spectrum of smooth_noise (unit variance): the start of the band amplitudes
            with torch.no_grad():
                z = smooth_noise((512, 1, H, 2 * H), self.sht.syn, torch.Generator().manual_seed(0))
                base = self.sht.spectrum(z).mean(0).squeeze(0)
            self.register_buffer("base_amp", base.clamp(min=1e-12).sqrt())  # (L,)
            self.a_log = nn.Parameter(torch.zeros(66, nb))      # log amplitude per channel and band
            self.rho_raw = nn.Parameter(torch.zeros(nb))         # AR(1) in time per band, rho = tanh(raw)
        self._xi = None
        if cfg.correct:
            # at the new state: x_t, G(x_t), the cloud's shift of the mean (mean of Phi over the members - Phi(xbar),
            # the equation's own nonlinear correction), the day's tendency, log size before / after, the entropy
            # H, static, time, lead, m. Out: delta (mean, units of the cloud's size exp(H)) and r2 (entropy); zero
            # head, so the chain starts as the uncorrected kernel
            c2 = 66 + n_g + 66 + 66 + 66 + 66 + 66 + core.static.shape[0] + 4 + 1 + self.mem_ch
            self.cnet = stateless_norm(Net(c2, 132, _NetCfg(cfg.hidden, tuple(cfg.blocks))))

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

    def amp(self):
        """(66, L): the new part's amplitude per channel and degree (sphere noise)."""
        return torch.exp(self.a_log)[:, self.band_of] * self.base_amp

    def new_noise(self, shape, like, gen=None):
        """The new part's unit field xi (shape (B, P, 66, H, W)); sphere noise keeps its last draw for the AR(1)."""
        if self.cfg.noise != "sphere":
            return smooth_noise(shape, like, gen)
        eta = self.sht.noise(self.amp(), tuple(shape[:2]) + (66,), gen)  # differentiable in the amplitudes
        if self._xi is None or self._xi.shape != eta.shape:
            xi = eta
        else:  # band by band: xi_t = rho xi_(t-1) + sqrt(1 - rho^2) eta_t, done in spectral space
            rho = torch.tanh(self.rho_raw)[self.band_of]                      # (L,)
            cp, ce = self.sht.analysis(self._xi), self.sht.analysis(eta)
            xi = self.sht.synthesis(rho[:, None] * cp + torch.sqrt(1 - rho ** 2)[:, None] * ce)
        self._xi = xi.detach()
        return xi

    @torch.no_grad()
    def start(self, xd, doy, hist=None, gen=None):
        """The trajectory starts at x (B, 66, H, W): the mean's memory (memory core) and the initial deviations
        e_0 (B, P, 66, H, W) = eps0 x smooth unit noise."""
        if self.mem:
            self.core.reset(hist, doy.float() * 24.0, xd)
        B, P = xd.shape[0], self.cfg.pairs
        self._xi = None
        return self.cfg.eps0 * smooth_noise((B, P) + tuple(xd.shape[1:]), xd, gen) * self.keep.unsqueeze(0)

    def _equation(self, xbar, e, doy, tisr):
        """The mean's day and the deviations carried by the equation: (Phi(xbar), p (B, P, 66, H, W), G(xbar)).
        ``self.through``: the mean's own day keeps its graph (the core's weights are frozen, the gradient goes to
        xbar and the memory), so a correction of day t is scored by the days after it."""
        B, P = e.shape[:2]
        core, hours = self.core, doy.float() * 24.0
        with torch.set_grad_enabled(bool(self.through) and torch.is_grad_enabled()):
            if self.mem:
                core.remember(xbar)
                core.started = True
                phi0 = core._advance(xbar, hours, tisr)
            else:
                phi0 = core_step(core, xbar, doy, tisr)
        with torch.no_grad():
            out = self._cloud(xbar.detach(), e, doy, tisr, phi0)
        return (out[3] if self.cfg.cloud_mean else phi0,) + out[:3]

    def detach_state(self):
        """Cut the graph of the mean's memory (between the windows of the gradient)."""
        for k in MEM_STATE:
            v = getattr(self.core, k, None)
            if torch.is_tensor(v):
                setattr(self.core, k, v.detach())

    def _cloud(self, xbar, e, doy, tisr, phi0):
        B, P = e.shape[:2]
        core, hours = self.core, doy.float() * 24.0
        phi0 = phi0.detach()
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
        shift = (y.mean(1) - phi0) / self.s * self.keep
        if self.cfg.cloud_mean:
            # xbar_(L+1) = (1 / 2P) sum_p [Phi(xbar + s e_p) + Phi(xbar - s e_p)]: the mean of the kernel sum. Where the
            # equation is nonlinear over the cloud (the scales the cloud has decorrelated) this differs from
            # Phi(xbar): the unpredictable part is averaged out by the equation itself, not by the core's training
            phi0 = y.mean(1)
            if self.mem:
                core.fc = phi0
        p = (y[:, :P] - y[:, P:]) / (2 * self.s.unsqueeze(0)) * self.keep.unsqueeze(0)
        return p, g, shift, phi0

    def day(self, xbar, e, doy, tisr, lead: int, gen=None):
        """One day. xbar (B, 66, H, W) the mean, e (B, P, 66, H, W) the deviations of the day, ``lead`` = days since
        the IC (0 for the first day). Returns (xbar_next, e_next, H_next (B, 66, H, W), parts): H_next = 1/2 log of
        the variance of the next day in units of s^2; parts = (stretch, r): the day's entropy production split
        into the equation's stretching 1/2 log <p^2> / <e^2> (+ the new part) and the learned residual."""
        B, _, Hh, Ww = xbar.shape
        phi0, p, g, shift = self._equation(xbar, e, doy, tisr)
        g = None if g is None else g.detach()
        xi = self.new_noise(tuple(p.shape), xbar, gen) * self.keep.unsqueeze(0)
        q = p + self.h().unsqueeze(0) * xi
        le, lq = self.size(e), self.size(q)
        tend = ((phi0 - xbar) / self.s * self.keep).detach()
        parts = [xbar] + ([g] if g is not None else []) + [tend, le.detach(), lq.detach(),
                                                           self.core.static.expand(B, -1, -1, -1),
                                                           time_features(doy.float() * 24.0, Hh, Ww),
                                                           xbar.new_full((B, 1, Hh, Ww), lead / 10.0)]
        if self.mem:
            parts.append(self.core.m)
        u = self.net(torch.cat(parts, 1)) + self.r_bias
        if self.cfg.r_dissip:
            r = -self.cfg.r_max * torch.tanh(F.softplus(-u) / self.cfg.r_max) * self.keep
        else:
            r = self.cfg.r_max * torch.tanh(u / self.cfg.r_max) * self.keep
        if self.cfg.correct:
            # the correction kernel at the new x_t: the primitive-equation terms of the new state move the mean,
            # and the entropy is corrected once more there
            H0 = (lq + r).detach()
            new = [phi0] + ([self.core.features(phi0).detach()] if self.gfeat else []) + [
                shift, tend, le.detach(), lq.detach(), H0, self.core.static.expand(B, -1, -1, -1),
                time_features((doy.float() + 1.0) * 24.0, Hh, Ww), xbar.new_full((B, 1, Hh, Ww), (lead + 1) / 10.0)]
            if self.mem:
                new.append(self.core.m)
            out = self.cnet(torch.cat(new, 1))
            # bounded: the corrected x_t goes back into the equation, so an unbounded correction compounds over
            # the chain (first run: day-10 RMSE 1.50 -> 1.71 s after 150 iterations)
            delta = self.cfg.d_max * torch.tanh(out[:, :66] / self.cfg.d_max) * torch.exp(H0).clamp(max=20.0) * self.keep
            r = r + self.cfg.r_max * torch.tanh(out[:, 66:] / self.cfg.r_max) * self.keep
            phi0 = phi0 + self.s * delta
            if self.mem:
                self.core.fc = phi0.detach()  # the corrected mean is the forecast the memory's innovation refers to
        e_next = q * torch.exp(r)[:, None]
        Hn = lq + r
        if self.cfg.band_scale:
            # the identity plus the band-wise change, so the start is exactly the kernel without it
            fac = torch.expm1(self.g_band[min(lead, 9)])[:, self.band_of]          # (66, L)
            e_next = e_next + self.sht.synthesis(self.sht.analysis(e_next) * fac[:, :, None]) * self.keep.unsqueeze(0)
            Hn = self.size(e_next)
        # the parts of the day (diagnostics): the day's production H_(L+1) - H_L split into the equation's stretching
        # 1/2 log <p^2> - H_L, the noise's 1/2 log <q^2> - 1/2 log <p^2> (>= 0) and the learned residual r
        lp = self.size(p)
        self.last = dict(p=p, hxi=q - p, r=r, prod=((lp - le).detach(), (lq - lp).detach(), r.detach()))
        return phi0, e_next, Hn, (lq - le, r)

    def ign(self, H, xbar, truth, w):
        """The Gaussian log score of the truth around the mean with entropy H (without 1/2 log 2 pi), area-weighted,
        tisr excluded; and the mean entropy itself: calibrated when the two are 1/2 apart (E[err^2] = exp(2 H))."""
        err = (truth - xbar) / self.s
        v = H + 0.5 * err.pow(2) * torch.exp(-2 * H)
        m = self.keep * w
        n = m.expand_as(v).sum()
        return (v * m).sum() / n, (H * m).sum() / n

    def band_ign(self, e, xbar, truth):
        """The Gaussian log score by scale: per channel and band j of degrees, the error's power E_j against the
        deviations' power V_j (mean over the pairs), over the band's n_j coefficients,
        n_j (1/2 log (V_j / n_j) + E_j / (2 V_j)) / sum_j n_j, averaged over the channels (tisr excluded).
        Its minimum is V_j = E_j: the spread in every band the size of the error there."""
        bj = self.band_of
        n = torch.stack([(2 * torch.arange(self.sht.L, device=bj.device)[bj == b] + 1).sum()
                         for b in range(self.n_bands)]).to(e)
        band = lambda t: torch.stack([t[..., bj == b].sum(-1) for b in range(self.n_bands)], -1)  # noqa: E731
        V = band(self.sht.spectrum(e).mean(1)).clamp(min=1e-10)                    # (B, 66, nb)
        E = band(self.sht.spectrum((truth - xbar) / self.s))
        v = (n * (0.5 * torch.log(V / n) + E / (2 * V))).sum(-1) / n.sum()          # (B, 66)
        k = self.keep.reshape(-1) > 0
        return v[:, k].mean()

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


class TotalEnergy:
    """The dry total energy per unit mass of a state, area- and level-mean: c_p T + 1/2 (u^2 + v^2) over the 13
    pressure levels (J/kg), from the z-scored dynamic channels (66). The functional of the physical entropy
    constraint (design, step 8): the day's change of the energy of a member should stay near the mean's."""

    CP = 1004.0

    def __init__(self, names, mean, std, lat):
        dyn = [n for n in names if n != "tisr"] if len(names) == 67 else list(names)
        idx = lambda pre: [i for i, n in enumerate(dyn) if n[0] == pre and n[1:].isdigit()]  # noqa: E731
        self.t, self.u, self.v = idx("t"), idx("u"), idx("v")
        keep = [i for i, n in enumerate(names) if n != "tisr"] if len(names) == 67 else list(range(len(names)))
        m, s = np.asarray(mean)[keep], np.asarray(std)[keep]
        self.m, self.s = torch.tensor(m, dtype=torch.float32), torch.tensor(s, dtype=torch.float32)
        w = np.cos(np.deg2rad(np.asarray(lat)))
        self.w = torch.tensor(w / w.mean(), dtype=torch.float32).view(-1, 1)

    def __call__(self, x):
        """x (..., 66, H, W) z-scored -> (...,) J/kg."""
        phys = lambda i: x[..., i, :, :] * self.s[i, None, None] + self.m[i, None, None]  # noqa: E731
        e = self.CP * phys(self.t) + 0.5 * (phys(self.u) ** 2 + phys(self.v) ** 2)
        return (e * self.w).mean((-1, -2, -3))
