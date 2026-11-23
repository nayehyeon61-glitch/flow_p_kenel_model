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

G-flow (``gmember`` / ``gdirect``): G(xbar) above is the operator's tendency at the MEAN, one field for all members.
The tendency core read G at its own state; the flow's analogue is G at the member's state along the path,

    dG(z_tau) = G(xbar + s z_tau) - G(xbar)        (no gradient, units of G's ERA5 RMS, as the core's input)
    v = sigma v_theta(z_tau, tau | c, dG) + A (.) dG        (gdirect: A per u, v, T, q channel, zero at the start)

dG is how the equation's tendency of this member differs from the mean's: to first order J s z_tau, the
tangent-linear dynamics of the deviation. It is still a function of the inputs (z_tau, xbar), a hint and no new
information, as G was for the core. A is read after training (``g_report``): how much of the transport the
operator's own term carries, per variable and level. ``gcond=False``: no G at all in the flow (the control).

Two uses of a trained flow (scripts/run_fpk.py):
  * reflow (``devflow --reflow TEACHER``): the same loss with the target of every source row replaced by the
    teacher's transported endpoint of that row, z1 = T_teacher(z0). The coupling (z0, z1) is then a map, the
    regression target z1 - z0 is a function of the source row, and the learned paths straighten toward the lines
    z0 -> T_teacher(z0) without changing the law of the endpoints (rectified flow).
  * energy-score fine-tuning (``devtune``): the members x^m = xbar + s T(e^m) with the gradient through the sampler
    (``integrate(..., grad=True)``: the Heun steps as written, every network call in the graph), scored with the
    fair energy score of fpk.escore. ``ctx_rows``: the sampler on a part of the day's source rows, for a gradient
    taken a few rows at a time (the memory of the graph of those rows only).
  * the kernel learns along (``devtune --kernel-lr``): the day's context with the graph of what the kernel gave it
    (``day_ctx(..., grad=True)``: the source z0 = e_L, the conditioning's xbar_L and H_L; sigma = exp(sg(H)) stays
    a constant, ``sigma``), so the members' energy score reaches the kernel's parameters through the sampler.

The flow-matching loss never sends a gradient to the kernel (``loss`` detaches e, xbar and H). It is a regression
of z1 - z0 on (z_tau, tau, conditioning), bounded below by E Var[z1 - z0 | z_tau, tau, c]; as a function of the
SOURCE's parameters that bound, and the loss, fall with the source's spread and reach zero for a point source
(z_tau = (1 - tau) m + tau z1 then determines z1): its gradient collapses the ensemble's source. The kernel is
anchored by its own log score instead (``EntropyDeviationKernel.lead_loss``).
"""

import math
from dataclasses import dataclass
from types import SimpleNamespace
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
    # The physical field in the flow (G-flow, 감자 2026-11-23; the straight line only, a kernel whose core has G(x)):
    gcond: bool = True       # G(xbar) of the day's mean in the conditioning, the same for every member (as it always
    #                          was); False = the flow reads no G of the mean (the control of G in the flow, same kernel)
    gmember: bool = False    # the member's own physics as an input at every tau: dG = G(xbar + s z_tau) - G(xbar)
    gdirect: bool = False    # the physics channel in the velocity: + A (.) dG on the u, v, T, q channels, A per channel,
    #                          zero at the start (the flow starts as without it), read after training (``g_direct``)


def g_direct_index(n_g: int):
    """The physics channel's map: (src, dst) with dG channel src[i] (G's own order, dyncore.G_EQS x the 12 levels:
    u, v, T, q from 50 to 1000 hPa) onto the dynamic channel dst[i] of the same variable and level (index into DYN).
    The first 48 of the core's n_g physical terms are those (extra terms of a core, if any, have no channel)."""
    from pgcfm import contract
    from pgcfm.dyncore import G_EQS
    from .env import DYN
    if n_g < len(G_EQS) * len(contract.LEVELS_HPA):
        raise ValueError(f"the physics channel needs the 48 operator tendencies, the core has {n_g} physical terms")
    src, dst = [], []
    for i, q in enumerate(G_EQS):
        for j, p in enumerate(contract.LEVELS_HPA):
            src.append(i * len(contract.LEVELS_HPA) + j)
            dst.append(DYN.index(contract.CHANNEL_INDEX[f"{q.lower()}{p}"]))
    return torch.tensor(src), torch.tensor(dst)


class DeviationFlow(nn.Module):
    def __init__(self, dk: EntropyDeviationKernel, cfg: DevFlowConfig = DevFlowConfig()):
        super().__init__()
        self.dk, self.cfg = dk, cfg
        for p in dk.parameters():
            p.requires_grad_(False)
        core = dk.core
        g_opts = (not cfg.gcond) or cfg.gmember or cfg.gdirect
        if g_opts and cfg.path != "line":
            raise ValueError("gcond / gmember / gdirect are options of the straight line (path = line)")
        if (cfg.gmember or cfg.gdirect) and not dk.gfeat:
            raise ValueError("gmember / gdirect read the core's operator G(x): the kernel's core has no G(x) input")
        n_g = core.gscale.shape[1] if dk.gfeat else 0
        self.n_gc = n_g if cfg.gcond else 0       # G(xbar) channels of the conditioning
        self.n_gm = n_g if cfg.gmember else 0     # dG channels of the network's input
        # z_tau, xbar, G(xbar), H, static, time of year, lead, [dG], tau
        self.c_in = c_in = 66 + 66 + self.n_gc + 66 + core.static.shape[0] + 4 + 1 + self.n_gm + 4
        self.net = stateless_norm(Net(c_in, 66, _NetCfg(cfg.hidden, tuple(cfg.blocks))))
        if cfg.gdirect:
            # A sits on the network module so that everything that handles the flow's weights (the optimizer, the
            # checkpoint's state, a reflow teacher, devtune) carries it unchanged; the network's forward ignores it
            self.net.register_parameter("g_direct", nn.Parameter(torch.zeros(1, 66, 1, 1)))
            src, dst = g_direct_index(n_g)
            self.register_buffer("g_src", src, persistent=False)
            self.register_buffer("g_dst", dst, persistent=False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.dk.eval()
        return self

    def cond(self, xbar, H, doy, lead: int, grad: bool = False):
        """The conditioning at a lead: xbar (B, 66, H, W) the mean of that day, H its predicted entropy field,
        doy (B,) the day of year of that day, lead = days since the IC. ``grad``: with the graph of xbar and H (the
        kernel's gradient through what the flow reads; G(xbar) is the operator's, without gradient); else none."""
        B, _, Hh, Ww = xbar.shape
        core = self.dk.core
        with torch.set_grad_enabled(grad and torch.is_grad_enabled()):
            parts = [xbar] + ([core.features(xbar)] if self.n_gc else []) + [
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
        """exp(sg(H)) (B, 66, H, W), 1 on tisr (no deviation there): the unit of the network's input and output,
        a constant of every gradient (also of the kernel's, ``day_ctx(..., grad=True)``)."""
        k = self.dk.keep
        return torch.exp(H.detach().clamp(-6.0, 6.0)) * k + (1 - k)

    def member_g(self, z, cond):
        """dG (R, n_g, H, W) = G(xbar + s z) - G(xbar): the change of the operator's tendency (units of its ERA5 RMS,
        clamped as the core's input) at the member's state z (units of s around the mean xbar = cond[:, :66]). No
        gradient (the operator's, as everywhere). G(xbar) is the conditioning's own channels when it has them."""
        core = self.dk.core
        with torch.no_grad():
            xbar = cond[:, :66].detach()
            g0 = cond[:, 66:66 + self.n_gc].detach() if self.n_gc else core.features(xbar)
            return core.features(xbar + self.dk.s * (z.detach() * self.dk.keep)) - g0

    def physics(self, dg):
        """The physics channel A (.) dG on the dynamic channels (R, 66, H, W), units of s (no factor sigma: dG
        already grows with the member's deviation; zero on z, sp, t2m, ..., which have no operator tendency)."""
        out = dg.new_zeros(dg.shape[0], 66, *dg.shape[-2:])
        out[:, self.g_dst] = dg[:, self.g_src]
        return self.net.g_direct * out

    def velocity(self, z, tau, cond, sig):
        Hh, Ww = z.shape[-2:]
        if not (self.cfg.gmember or self.cfg.gdirect):
            return self.net(torch.cat([z / sig, cond, tau_features(tau, Hh, Ww)], 1)) * sig * self.dk.keep
        dg = self.member_g(z, cond)
        v = self.net(torch.cat([z / sig, cond] + ([dg] if self.cfg.gmember else []) + [tau_features(tau, Hh, Ww)], 1)) * sig
        if self.cfg.gdirect:
            v = v + self.physics(dg)
        return v * self.dk.keep

    def g_report(self):
        """What the physics channel learned: A per variable and level (u, v, T, q x 12 levels, 50 -> 1000 hPa)."""
        if not self.cfg.gdirect:
            return None
        from pgcfm import contract
        from pgcfm.dyncore import G_EQS
        a = self.net.g_direct.detach().flatten()[self.g_dst].view(len(G_EQS), -1).cpu()
        return {"levels": contract.LEVELS_HPA, **{q: [round(float(v), 5) for v in a[i]] for i, q in enumerate(G_EQS)}}

    def loss(self, e, xbar, truth, cond, H, w, gen=None, teacher=None):
        """Flow-matching loss of one lead: every one of the P deviations is a source draw for the same target.
        ``teacher`` (reflow): the target of a source row is the teacher's transported endpoint of that row,
        z1 = T_teacher(z0) (its own steps, no gradient), in place of the data's deviation. The kernel's e, xbar and
        H are detached: this loss trains the network only (for the source it is minimized by a collapsed source,
        the module's docstring)."""
        e, xbar, H = e.detach(), xbar.detach(), H.detach()
        B, P = e.shape[:2]
        z0 = self.source(e, gen, self.sigma(H)[:, None]).flatten(0, 1)
        z1 = ((truth - xbar) / self.dk.s * self.dk.keep).repeat_interleave(P, 0)
        c, sig = cond.repeat_interleave(P, 0), self.sigma(H).repeat_interleave(P, 0)
        if teacher is not None:
            z1 = teacher.integrate(z0, c, sig)
        tau = torch.rand(B * P, generator=gen).to(z0)
        t = tau.view(-1, 1, 1, 1)
        zt = (1 - t) * z0 + t * z1 + self.cfg.sigma_min * sig * torch.randn(z0.shape, generator=gen).to(z0) * self.dk.keep
        sq = ((self.velocity(zt, tau, c, sig) - (z1 - z0)) / sig) ** 2
        m = self.dk.keep * w
        return (sq * m).sum() / m.expand_as(sq).sum()

    def integrate(self, z, c, sig, steps: Optional[int] = None, grad: bool = False):
        """The transport T of the source rows z (R, 66, H, W) with their conditioning c and sigma: ``steps`` Heun
        steps of dz / dtau = v_theta(z, tau). ``grad``: with the gradient to the network's parameters (the exact
        gradient of this discrete sampler: all 2 x steps network calls stay in the graph); else no graph is built."""
        n = steps or self.cfg.steps
        dt = 1.0 / n
        z0, kin = z, 0.0
        with torch.set_grad_enabled(grad and torch.is_grad_enabled()):
            for i in range(n):
                t = torch.full((z.shape[0],), i * dt, device=z.device)
                k1 = self.velocity(z, t, c, sig)
                k2 = self.velocity(z + dt * k1, t + dt, c, sig)
                v = 0.5 * (k1 + k2)
                kin = kin + dt * (v.detach() / sig).pow(2).mean()
                z = z + dt * v
        # how straight the paths are (Benamou-Brenier): the kinetic energy int |v|^2 dtau over the squared
        # displacement |T(z) - z|^2 (units of sigma); 1 = straight lines at constant speed, the OT geodesic
        disp = ((z.detach() - z0) / sig).pow(2).mean()
        self.last_path = (float(kin), float(disp))
        return z

    def transport(self, e, cond, H, gen=None, steps: Optional[int] = None, grad: bool = False):
        """(B, P, 66, H, W) -> the transported deviations (Heun); ``grad``: see ``integrate``."""
        B, P = e.shape[:2]
        with torch.no_grad():
            z = self.source(e, gen, self.sigma(H)[:, None]).flatten(0, 1)
            c, sig = cond.repeat_interleave(P, 0), self.sigma(H).repeat_interleave(P, 0)
        z = self.integrate(z, c, sig, steps, grad)
        return z.view(B, P, *z.shape[1:])

    # ---- one day of the kernel -> its loss / its members: the calls a trainer or a rollout makes right after
    # ``dk.day`` (x0 the mean before the day, xb the mean after it, e and H what the day returned, doy0 the IC's day
    # of year, tisr the day's forcing, lead = days since the IC of the day's start). The same for every path.
    def day_loss(self, x0, xb, e, H, truth, doy0, tisr, lead: int, w, gen=None, sel=None, teacher=None):
        """The flow-matching loss of the day; ``sel``: the indices of the kernel's pairs used as source draws;
        ``teacher``: the reflow target (see ``loss``)."""
        es = e if sel is None else e[:, sel]
        return self.loss(es, xb, truth, self.cond(xb, H, doy0 + lead + 1, lead + 1), H, w, gen, teacher)

    def day_transport(self, x0, xb, e, H, doy0, tisr, lead: int, gen=None, steps: Optional[int] = None,
                      grad: bool = False):
        """The members' deviations of the day (B, P, 66, H, W), units of s around xb."""
        return self.transport(e, self.cond(xb, H, doy0 + lead + 1, lead + 1), H, gen, steps, grad)

    # ---- the sampler in two calls (the same for every path): the day's source rows once, then any number of runs
    # from them (other step counts, another flow on the same rows, with or without the gradient)
    def day_ctx(self, x0, xb, e, H, doy0, tisr, lead: int, gen=None, sel=None, grad: bool = False):
        """The sampler's context of the day: the source rows z0 (R = B P', 66, H, W) of the kernel's pairs ``sel``
        (None = all P; the +e of every antithetic pair, as in ``rollout``), their conditioning and sigma.
        ``grad``: with the graph of e, xb and H (the kernel learns from the members' score: z0 = e and the
        conditioning's xbar and H carry it; sigma does not, and the Gaussian control's source has only sigma);
        else no graph is built."""
        with torch.set_grad_enabled(grad and torch.is_grad_enabled()):
            es = e if sel is None else e[:, sel]
            B, P = es.shape[:2]
            sig = self.sigma(H)
            return SimpleNamespace(B=B, P=P, z0=self.source(es, gen, sig[:, None]).flatten(0, 1), sig=sig.repeat_interleave(P, 0),
                                   cd=self.cond(xb, H, doy0 + lead + 1, lead + 1, grad).repeat_interleave(P, 0))

    def day_run(self, c, steps: Optional[int] = None, grad: bool = False):
        """The members' deviations (B, P', 66, H, W) from the context of ``day_ctx``; ``steps``, ``grad``: see
        ``integrate``."""
        z = self.integrate(c.z0, c.cd, c.sig, steps, grad)
        return z.view(c.B, c.P, *z.shape[1:])

    def ctx_rows(self, c, rows):
        """The context of the source rows ``rows`` (indices into the R rows of c) alone, as one batch element:
        ``day_run`` of it is (1, len(rows), 66, H, W), those rows of the full run (the rows do not see each other).
        For a gradient in pieces: sum over the pieces of [d L / d z of the rows] . z(rows) has the gradient of L."""
        return SimpleNamespace(B=1, P=len(rows), z0=c.z0[rows], sig=c.sig[rows], cd=c.cd[rows])

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
