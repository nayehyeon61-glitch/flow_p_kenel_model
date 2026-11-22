"""Flow matching whose path is the tendency core's own flow: the pseudo time of the straight line becomes the day.

fpk.devflow moves a draw of the deviation kernel to the data's deviation along the straight line
z_tau = (1 - tau) z0 + tau z1 in a pseudo time tau that has no physical meaning: the equation enters as the source
and as conditioning at the END of the day only. Here t in [0, 1] is the physical time across the day L -> L + 1 on a
grid t_k = k / N, and the path is the core's flow of the deviation plus the residual, added linearly in t:

    mean path   xbar(t_(k+1)) = step_k(xbar(t_k))            step_k = one Heun step of the core's tendency
                                                             f_theta(x, G(x)) over 1 / N day (``core.substep``),
                                                             with the memory of the day (a memory core)
    core step   S_k(z) = [step_k(xbar(t_k) + s z) - step_k(xbar(t_k) - s z)] / 2s      the odd part around the mean
                                                             path: how p is formed in the kernel, per step
    source      z_0 = the deviations that enter the kernel's day (e, or e + h xi for a noise_pre kernel)
    day's end   T_k = S_k (k < N - 1),   T_(N-1)(z) = S_(N-1)(z) + b,   b = e' - p     the part of the kernel's day
                                                             that is not the equation's (its additive noise, exp(r),
                                                             band_scale, ...: whatever the kernel does), as an
                                                             offset of the member: it does NOT rescale the residual
                                                             already added (see below)
    zero head   zc_(k+1) = T_k(zc_k), zc_0 = z_0             ends at e' + [S_(N-1)(...S_0(z_0)) - p]: the kernel's
                                                             own e', exactly when the grid is the core's own solver
                                                             and N = 1 (the real cores)
    target      z1 = (truth_(L+1) - xbar_(L+1)) / s          xbar_(L+1) = the kernel's mean (with its correction)
    path        z_k = zc_k + t_k rr,   rr = z1 - zc_N        the core's path + the residual, linear in the day
    flow        z_(k+1) = T_k(z_k) + (1 / N) u_theta(z_k, t_k | xbar(t_k), G(xbar(t_k)), H, lead + t_k, ...)
    target of u u*_k = N [z_(k+1)^path - T_k(z_k^path)]      exact for this sampler (= rr at k = 0): what the day
                                                             still needs beyond the core's own step from z_k

So the network is conditioned on the equation at the intermediate physical time, and what it learns is the residual
of the core's flow, not a transport in a time of its own. One k per sample is drawn; every core step runs without
gradient (the core is frozen), the gradient goes through u_theta at the drawn k only.

The day's end as an offset. Folding the kernel's end map E (exp(r), band_scale) into the last step, E(S(z) + a),
multiplies the part (N - 1) / N rr of the residual that is already in the state by e^r: the last step's target is
then about (N - (N - 1) e^r) rr (x 4.7 in the loss at r = -0.5, about 0 at r = +0.3) and the loss is no longer on
the scale of the other steps or of DeviationFlow.loss. With the offset b every step's target is rr minus the CORE's
stretching of the part already added, and nothing of the kernel's end has to be known or inverted here.

``core_sub`` = M >= 1: the learned part of every grid step as a continuous flow. The one Euler step a grid point
above cannot carry a distribution: the network learns E[u*_k | z_k], and a source draw and the truth's deviation are
independent given the conditioning, so the optimum contracts at early k and expands at late k (independent
Gaussians of one variance, S = identity: slopes -1, -0.8, 0, 0.8 at N = 4, members with 0.72 of the target's
spread; N = 1 collapses to the conditional mean). With M >= 1, after the core's step w = T_k(z_k) the learned part
is flow matching on the straight line of that grid step,

    y(tau) = w^path + tau d_k,   d_k = z_(k+1)^path - T_k(z_k^path),   tau in [0, 1],   target N d_k (= u*_k)
    u_theta(y, t_k + tau / N | the conditioning of t_k),   integrated with M Heun steps (2 M network calls)

which keeps the path's marginal at every grid point (the same oracle: 1.00 of the target's spread at N = 4, M = 3).
N = 1 with M >= 1 is DeviationFlow's line from the kernel's e', conditioned on the day's start. ``core_sub`` = 0 is
the one-step sampler above (the network sees z_k BEFORE the core's step).

The members x = xbar_(L+1) + s z_N are not fed back: the kernel's own deviations go on through the equation, as in
DeviationFlow. ``source="gauss"``: z_0 = smooth Gaussian noise of the kernel's predicted size, carried by the same
core steps (no offset: T_k = S_k for every k).

The two approximations, both zero when N = 1 on a one-step core (the grid IS the kernel's day):
  * the odd part is taken around the mean path, the kernel's p around the pair's own midpoint
    [x+(t_k) + x-(t_k)] / 2, which leaves the mean path at second order in s e once a step has been made;
  * N other than core.n_sub() integrates the frozen tendency with another step than the core was trained with.
rr is defined from the zero-head end of THIS path, so the target stays exact for the sampler in either case; only
"zero head = the kernel's e'" becomes approximate.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Optional

import torch

from .devflow import DevFlowConfig, DeviationFlow
from .devkernel import EntropyDeviationKernel
from .kernel import ADV_STATE


class CoreFlow(DeviationFlow):
    """The same network and conditioning channels as DeviationFlow (tau's features hold t_k, the lead channel
    lead + t_k, the time of year advances with t_k); the calls of one day are ``day_loss`` / ``day_transport``."""

    def __init__(self, dk: EntropyDeviationKernel, cfg: DevFlowConfig = DevFlowConfig(path="core")):
        super().__init__(dk, cfg)
        self.N = int(cfg.core_steps) or int(dk.core.n_sub())
        self.last_split = None

    # ---- the core on the day's grid (no gradient: the core is frozen)
    @contextmanager
    def _state(self, bidx):
        """A memory core reads its state as attributes: for rows of the batch elements ``bidx`` every row gets the
        state of its IC (the memory of the day, as the kernel's cloud does), put back afterwards."""
        core = self.dk.core
        if not self.dk.mem or bidx is None:
            yield
            return
        keep = {k: getattr(core, k, None) for k in ADV_STATE}
        try:
            for k, v in keep.items():
                if v is not None:
                    setattr(core, k, v[bidx])
            yield
        finally:
            for k, v in keep.items():
                setattr(core, k, v)

    @torch.no_grad()
    def _step(self, k: int, x, hours, tisr, bidx=None):
        """step_k: Heun step k of the N of the day from the states x (rows of the batch elements ``bidx``; None =
        the batch itself); the last one ends the day (the forcing of the new day, positive mass)."""
        core = self.dk.core
        if bidx is not None:
            hours, tisr = hours[bidx], None if tisr is None else tisr[bidx]
        with self._state(bidx):
            y = core.substep(x, hours, k, self.N)
        return core.finish(y, tisr) if k == self.N - 1 else y

    @torch.no_grad()
    def mean_path(self, x0, hours, tisr):
        """[xbar(t_0), ..., xbar(t_(N-1))]: the mean of the day carried by the core's steps."""
        xs = [x0]
        for k in range(self.N - 1):
            xs.append(self._step(k, xs[-1], hours, tisr))
        return xs

    @torch.no_grad()
    def S(self, k: int, xk, z, bidx, hours, tisr):
        """The core's step of the deviations z (R, 66, H, W) around the mean path's state xk (B, 66, H, W)."""
        R, s, c = z.shape[0], self.dk.s, xk[bidx]
        b2 = torch.cat([bidx, bidx])
        y = self._step(k, torch.cat([c + s * z, c - s * z], 0), hours, tisr, b2)
        return (y[:R] - y[R:]) / (2 * s) * self.dk.keep

    @torch.no_grad()
    def T(self, c, k: int, z, rows=None):
        """T_k on the rows ``rows`` (None = all) of the day c: the core's step and, at the last one, the offset
        b = e' - p of the kernel's day (what the kernel did beside the equation)."""
        y = self.S(k, c.xs[k], z, c.bidx if rows is None else c.bidx[rows], c.hours, c.tisr)
        if k < self.N - 1 or c.add is None:
            return y
        return y + (c.add if rows is None else c.add[rows])

    # ---- one day
    @torch.no_grad()
    def day_start(self, x0, e, H, doy0, tisr, lead: int, gen=None, sel=None):
        """The day's context, right after ``dk.day`` (it reads the kernel's ``last`` and the core's memory of that
        day; e is what the day returned): the mean path, the source rows z0 (R = B P', 66, H, W) and the offset of
        the kernel's day."""
        dk = self.dk
        B = x0.shape[0]
        sig = self.sigma(H)
        if self.cfg.source == "gauss":
            z0 = self.source(e if sel is None else e[:, sel], gen, sig[:, None])
            add = None
        else:
            pick = lambda t: t.detach() if sel is None else t.detach()[:, sel]  # noqa: E731
            z0, add = pick(dk.last["zin"]), pick(e - dk.last["p"]).flatten(0, 1)
        P = z0.shape[1]
        hours = (doy0 + lead).float() * 24.0
        c = SimpleNamespace(B=B, P=P, lead=lead, hours=hours, tisr=tisr, add=add, H=H, z0=z0.flatten(0, 1),
                            bidx=torch.arange(B, device=x0.device).repeat_interleave(P),
                            doy=(doy0 + lead).float(), xs=self.mean_path(x0, hours, tisr), _cond={})
        c.sig = sig[c.bidx]
        return c

    def cond_k(self, c, k: int):
        """The conditioning at t_k (B, C, H, W): the mean path's state and its G, the time advanced by t_k."""
        if k not in c._cond:
            t = k / self.N
            c._cond[k] = self.cond(c.xs[k], c.H, c.doy + t, c.lead + t)
        return c._cond[k]

    @torch.no_grad()
    def zero_path(self, c):
        """[zc_0, ..., zc_N]: the zero-head path of every source row (2 N core steps a row)."""
        zc = [c.z0]
        for k in range(self.N):
            zc.append(self.T(c, k, zc[-1]))
        return zc

    @torch.no_grad()
    def target(self, c, zc, z1, k, noise=None, tau=None):
        """The network's input state and the exact target of u at the steps k (R,) of the rows: (z, u*_k).
        ``tau`` None (core_sub = 0): z = the path's z_k, + ``noise`` (R, 66, H, W) (the target then leads from the
        perturbed state to the path's z_(k+1)). ``tau`` (R,) (core_sub >= 1): z = y(tau) = w + tau d_k (+ noise)
        after the core's step w = T_k(z_k^path) of the path's own state, target N d_k."""
        N = self.N
        rr = z1 - zc[N]
        pick = lambda off: torch.stack(zc, 0)[k + off, torch.arange(k.shape[0], device=k.device)]  # noqa: E731
        t = (k.float() / N).view(-1, 1, 1, 1)
        zk, zk1 = pick(0) + t * rr, pick(1) + (t + 1.0 / N) * rr
        Tz = pick(1).clone()            # k = 0 from the path's own z_0: T_0(z_0) is the zero-head path, u* = rr
        before = tau is None and noise is not None
        if before:
            zk = zk + noise
        for j in range(N):
            rows = (k == j).nonzero().squeeze(1) if (j or before) else k.new_zeros(0)
            if rows.numel():
                Tz[rows] = self.T(c, j, zk[rows], rows)
        ut = N * (zk1 - Tz)
        if tau is None:
            return zk, ut
        y = Tz + tau.view(-1, 1, 1, 1) * (zk1 - Tz)
        return (y if noise is None else y + noise), ut

    @torch.no_grad()
    def day_rows(self, c, xb, truth, gen=None, k=None, tau=None):
        """The training rows of the day c: the network's input state z, its time t (R,) in the day, the grid step
        k (R,) and the target ut; one k per source row (and one tau in [0, 1) with core_sub >= 1), drawn unless
        given (a test)."""
        dk, N, R = self.dk, self.N, c.z0.shape[0]
        z1 = ((truth - xb) / dk.s * dk.keep)[c.bidx]
        if k is None:
            k = torch.randint(0, N, (R,), generator=gen)
        k = k.to(c.z0.device)
        if self.cfg.core_sub > 0:
            tau = (torch.rand(R, generator=gen) if tau is None else tau).to(c.z0)
        else:
            tau = None
        noise = None
        if self.cfg.sigma_min > 0:
            noise = self.cfg.sigma_min * c.sig * torch.randn(c.z0.shape, generator=gen).to(c.z0) * dk.keep
        z, ut = self.target(c, self.zero_path(c), z1, k, noise, tau)
        t = (k.float() + (0.0 if tau is None else tau)) / N
        return SimpleNamespace(z=z, t=t, k=k, ut=ut, z1=z1)

    def day_loss(self, x0, xb, e, H, truth, doy0, tisr, lead: int, w, gen=None, sel=None, k=None, tau=None):
        """The loss of the day in units of sigma = exp(H): one step k per source row (``k`` (R,), ``tau`` (R,):
        given, a test). Core steps: N - 1 (mean path, per IC) + 2 N (zero-head path) + 2 (the target; not for
        k = 0, unless core_sub = 0 with sigma_min noise) per source row."""
        dk = self.dk
        c = self.day_start(x0, e, H, doy0, tisr, lead, gen, sel)
        r = self.day_rows(c, xb, truth, gen, k, tau)
        with torch.no_grad():
            cd = torch.stack([self.cond_k(c, j) for j in range(self.N)], 0)[r.k, c.bidx]
        sq = ((self.velocity(r.z, r.t, cd, c.sig) - r.ut) / c.sig) ** 2
        m = dk.keep * w
        return (sq * m).sum() / m.expand_as(sq).sum()

    @torch.no_grad()
    def run(self, c, vel=None):
        """The sampler from the day's source rows. core_sub = 0: z_(k+1) = T_k(z_k) + u(z_k, t_k) / N.
        core_sub = M: w = T_k(z_k), then M Heun steps of dy / dtau = u(y, t_k + tau / N) / N from y(0) = w.
        ``vel(z, k, tau)``: another velocity than the network's (a test's oracle). Core steps: N - 1 per IC + 2 N
        per member; network calls per member: N, or 2 N M."""
        N, M, keep = self.N, int(self.cfg.core_sub), self.dk.keep
        var = lambda v: (v / c.sig).pow(2).mul(keep).sum() / keep.expand_as(v).sum()  # noqa: E731
        if vel is None:
            def vel(z, k, tau):
                t = torch.full((z.shape[0],), (k + tau) / N, device=z.device)
                return self.velocity(z, t, self.cond_k(c, k)[c.bidx], c.sig)
        z, kin, tot, core, learned = c.z0, 0.0, 0.0, 0.0, 0.0
        for k in range(N):
            if M == 0:
                u = vel(z, k, 0.0)
                a = self.T(c, k, z)
                zn = a + u / N
                kin = kin + (u / c.sig).pow(2).mean() / N
            else:
                a = self.T(c, k, z)
                zn, h = a, 1.0 / (N * M)
                for j in range(M):
                    k1 = vel(zn, k, j / M)
                    v = 0.5 * (k1 + vel(zn + h * k1, k, (j + 1) / M))
                    kin = kin + h * (v / c.sig).pow(2).mean()
                    zn = zn + h * v
            v0, va, vn = var(z), var(a), var(zn)
            core = core + 0.5 * float(torch.log(va / v0))
            learned = learned + 0.5 * float(torch.log(vn / va))
            tot = tot + (zn - a)
            z = zn
        # the learned part's own path: kinetic energy int |u|^2 dt over the squared sum of its increments (1 = the
        # same u all day); and the day's growth of 1/2 log variance (units of sigma), split into the core's steps
        # (with the offset of the kernel's day) and the learned increments
        self.last_path = (float(kin), float((tot / c.sig).pow(2).mean()))
        self.last_split = (core, learned)
        return z.view(c.B, c.P, *z.shape[1:])

    @torch.no_grad()
    def day_transport(self, x0, xb, e, H, doy0, tisr, lead: int, gen=None, steps: Optional[int] = None):
        """The members' deviations of the day (B, P, 66, H, W). ``steps`` is not used: the grid is the day's."""
        return self.run(self.day_start(x0, e, H, doy0, tisr, lead, gen))

    def loss(self, *a, **k):
        raise NotImplementedError("the core path needs the day's start: use day_loss")

    def transport(self, *a, **k):
        raise NotImplementedError("the core path needs the day's start: use day_transport")


def build_flow(dk: EntropyDeviationKernel, cfg: DevFlowConfig):
    """The deviation flow of a configuration: the straight line (DeviationFlow) or the core's flow (CoreFlow)."""
    if cfg.path == "line":
        return DeviationFlow(dk, cfg)
    if cfg.path == "core":
        return CoreFlow(dk, cfg)
    raise ValueError(f"path = {cfg.path!r}: line or core")
