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

Reflow (``day_loss(..., teacher=)``): z1 = the teacher's sampled end z_N of the same source row, in place of the
truth's deviation; the path and the target are built from it as above (rr = z1 - zc_N is then a function of the row).

The sampler with the gradient (``run(c, grad=True)``, the energy-score fine-tuning). The core's steps stay without
gradient, so the Jacobian of T_k with respect to the state is not available. It is replaced by the identity
(straight-through: T_k is evaluated at the detached state and z - sg(z) is added), the network's own du / dz is kept:

    exact    dz_N / dtheta = sum_k  prod_(j > k) [ dT_j / dz + (1 / N) du_j / dz ]  (1 / N) du_k / dtheta
    here                            prod_(j > k) [    I      + (1 / N) du_j / dz ]

What is dropped is dT_j / dz - I: the core's stretching, over the LATER grid steps j > k, of the increment the network
added at step k (the last step's increment is exact, and nothing is dropped at N = 1). ``jac=True`` keeps dT_j / dz
(the core's steps run with the graph of their input: the exact gradient, at the memory of a backward pass through
two core steps a member and grid step). core_sub >= 1: the same with the M Heun steps of a grid step in the graph.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Optional

import torch

from .devflow import DevFlowConfig, DeviationFlow
from .devkernel import EntropyDeviationKernel
from .kernel import ADV_STATE, rand, randn


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

    def _step(self, k: int, x, hours, tisr, bidx=None, grad: bool = False):
        """step_k: Heun step k of the N of the day from the states x (rows of the batch elements ``bidx``; None =
        the batch itself); the last one ends the day (the forcing of the new day, positive mass). ``grad``: with the
        graph of x (the core's weights stay frozen)."""
        core = self.dk.core
        with torch.set_grad_enabled(grad and torch.is_grad_enabled()):
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

    def S(self, k: int, xk, z, bidx, hours, tisr, grad: bool = False):
        """The core's step of the deviations z (R, 66, H, W) around the mean path's state xk (B, 66, H, W).
        ``grad``: with the graph of z (dS / dz, the equation's Jacobian of the step)."""
        with torch.set_grad_enabled(grad and torch.is_grad_enabled()):
            R, s, c = z.shape[0], self.dk.s, xk[bidx]
            b2 = torch.cat([bidx, bidx])
            y = self._step(k, torch.cat([c + s * z, c - s * z], 0), hours, tisr, b2, grad)
            return (y[:R] - y[R:]) / (2 * s) * self.dk.keep

    def T(self, c, k: int, z, rows=None, grad: bool = False):
        """T_k on the rows ``rows`` (None = all) of the day c: the core's step and, at the last one, the offset
        b = e' - p of the kernel's day (what the kernel did beside the equation)."""
        y = self.S(k, c.xs[k], z, c.bidx if rows is None else c.bidx[rows], c.hours, c.tisr, grad)
        if k < self.N - 1 or c.add is None:
            return y
        return y + (c.add if rows is None else c.add[rows])

    def _T(self, c, k: int, z, jac: bool = False):
        """T_k inside the sampler: as ``T`` for a state without a graph; for a state with one, the value of T_k
        with the identity as its Jacobian (straight-through), or with the core's own Jacobian (``jac``)."""
        if not z.requires_grad:
            return self.T(c, k, z)
        if jac:
            return self.T(c, k, z, grad=True)
        return self.T(c, k, z.detach()) + (z - z.detach())

    # ---- one day
    def day_start(self, x0, e, H, doy0, tisr, lead: int, gen=None, sel=None, grad: bool = False):
        """The day's context, right after ``dk.day`` (it reads the kernel's ``last`` and the core's memory of that
        day; e is what the day returned): the mean path, the source rows z0 (R = B P', 66, H, W) and the offset of
        the kernel's day. ``grad`` (the kernel learns from the members' score): with the graph of e in the offset
        b = e - sg(p) and of H in the conditioning; the source rows (the deviations that ENTERED the day), the pairs'
        p, the mean path and sigma stay constants. So d z_N / d (kernel) = d e' / d (kernel) through the offset,
        added at the day's end, and the network's own dependence on H; else no graph is built."""
        dk = self.dk
        B = x0.shape[0]
        with torch.set_grad_enabled(grad and torch.is_grad_enabled()):
            sig = self.sigma(H)
            if self.cfg.source == "gauss":
                z0 = self.source(e if sel is None else e[:, sel], gen, sig[:, None])
                add = None
            else:
                pick = lambda t: t if sel is None else t[:, sel]  # noqa: E731
                z0, add = pick(dk.last["zin"].detach()), pick(e - dk.last["p"].detach()).flatten(0, 1)
            P = z0.shape[1]
            hours = (doy0 + lead).float() * 24.0
            c = SimpleNamespace(B=B, P=P, lead=lead, hours=hours, tisr=tisr, add=add, H=H, z0=z0.flatten(0, 1),
                                bidx=torch.arange(B, device=x0.device).repeat_interleave(P), grad=grad,
                                doy=(doy0 + lead).float(), xs=self.mean_path(x0, hours, tisr), _cond={})
            c.sig = sig[c.bidx]
        return c

    def cond_k(self, c, k: int):
        """The conditioning at t_k (B, C, H, W): the mean path's state and its G, the time advanced by t_k (with
        the graph of H for a context made with ``grad``)."""
        g = bool(getattr(c, "grad", False)) and torch.is_grad_enabled()   # a pass without a graph keeps its own copy
        if (k, g) not in c._cond:
            t = k / self.N
            c._cond[(k, g)] = self.cond(c.xs[k], c.H, c.doy + t, c.lead + t, g)
        return c._cond[(k, g)]

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
    def day_rows(self, c, xb, truth, gen=None, k=None, tau=None, z1=None):
        """The training rows of the day c: the network's input state z, its time t (R,) in the day, the grid step
        k (R,) and the target ut; one k per source row (and one tau in [0, 1) with core_sub >= 1), drawn unless
        given (a test). ``z1`` (R, 66, H, W): the end of every row in place of the truth's deviation (reflow)."""
        dk, N, R = self.dk, self.N, c.z0.shape[0]
        if z1 is None:
            z1 = ((truth - xb) / dk.s * dk.keep)[c.bidx]
        if k is None:
            k = torch.randint(0, N, (R,), generator=gen, device=gen.device if gen is not None else c.z0.device)
        k = k.to(c.z0.device)
        if self.cfg.core_sub > 0:
            tau = (rand(R, c.z0, gen) if tau is None else tau).to(c.z0)
        else:
            tau = None
        noise = None
        if self.cfg.sigma_min > 0:
            noise = self.cfg.sigma_min * c.sig * randn(c.z0.shape, c.z0, gen) * dk.keep
        z, ut = self.target(c, self.zero_path(c), z1, k, noise, tau)
        t = (k.float() + (0.0 if tau is None else tau)) / N
        return SimpleNamespace(z=z, t=t, k=k, ut=ut, z1=z1)

    def day_loss(self, x0, xb, e, H, truth, doy0, tisr, lead: int, w, gen=None, sel=None, k=None, tau=None,
                 teacher=None):
        """The loss of the day in units of sigma = exp(H): one step k per source row (``k`` (R,), ``tau`` (R,):
        given, a test). Core steps: N - 1 (mean path, per IC) + 2 N (zero-head path) + 2 (the target; not for
        k = 0, unless core_sub = 0 with sigma_min noise) per source row. ``teacher`` (reflow): the end z1 of a
        source row is the teacher's sampled end of that row (its sampler from the same day's start, no gradient:
        2 N more core steps a row)."""
        dk = self.dk
        c = self.day_start(x0, e, H, doy0, tisr, lead, gen, sel)
        r = self.day_rows(c, xb, truth, gen, k, tau, None if teacher is None else teacher.run(c).flatten(0, 1))
        with torch.no_grad():
            cd = torch.stack([self.cond_k(c, j) for j in range(self.N)], 0)[r.k, c.bidx]
        sq = ((self.velocity(r.z, r.t, cd, c.sig) - r.ut) / c.sig) ** 2
        m = dk.keep * w
        return (sq * m).sum() / m.expand_as(sq).sum()

    def run(self, c, vel=None, grad: bool = False, jac: bool = False, sub: Optional[int] = None):
        """The sampler from the day's source rows. core_sub = 0: z_(k+1) = T_k(z_k) + u(z_k, t_k) / N.
        core_sub = M: w = T_k(z_k), then M Heun steps of dy / dtau = u(y, t_k + tau / N) / N from y(0) = w.
        ``vel(z, k, tau)``: another velocity than the network's (a test's oracle). Core steps: N - 1 per IC + 2 N
        per member; network calls per member: N, or 2 N M. ``sub``: another number of Heun steps than core_sub
        (a flow trained with core_sub >= 1 only: its network saw the state after the core's step).
        ``grad``: with the gradient to the network's parameters, the core's steps as constants with the identity as
        their state Jacobian; ``jac``: with their own Jacobian (the module's docstring)."""
        N, M, keep = self.N, int(self.cfg.core_sub if sub is None else sub), self.dk.keep
        var = lambda v: (v / c.sig).pow(2).mul(keep).sum() / keep.expand_as(v).sum()  # noqa: E731
        if vel is None:
            def vel(z, k, tau):
                t = torch.full((z.shape[0],), (k + tau) / N, device=z.device)
                return self.velocity(z, t, self.cond_k(c, k)[c.bidx], c.sig)
        z, kin, tot, core, learned = c.z0, 0.0, 0.0, 0.0, 0.0
        with torch.set_grad_enabled(grad and torch.is_grad_enabled()):
            for k in range(N):
                if M == 0:
                    u = vel(z, k, 0.0)
                    a = self._T(c, k, z, jac)
                    zn = a + u / N
                    kin = kin + (u.detach() / c.sig).pow(2).mean() / N
                else:
                    a = self._T(c, k, z, jac)
                    zn, h = a, 1.0 / (N * M)
                    for j in range(M):
                        k1 = vel(zn, k, j / M)
                        v = 0.5 * (k1 + vel(zn + h * k1, k, (j + 1) / M))
                        kin = kin + h * (v.detach() / c.sig).pow(2).mean()
                        zn = zn + h * v
                v0, va, vn = var(z.detach()), var(a.detach()), var(zn.detach())
                core = core + 0.5 * float(torch.log(va / v0))
                learned = learned + 0.5 * float(torch.log(vn / va))
                tot = tot + (zn - a).detach()
                z = zn
        # the learned part's own path: kinetic energy int |u|^2 dt over the squared sum of its increments (1 = the
        # same u all day); and the day's growth of 1/2 log variance (units of sigma), split into the core's steps
        # (with the offset of the kernel's day) and the learned increments
        self.last_path = (float(kin), float((tot / c.sig).pow(2).mean()))
        self.last_split = (core, learned)
        return z.view(c.B, c.P, *z.shape[1:])

    def day_transport(self, x0, xb, e, H, doy0, tisr, lead: int, gen=None, steps: Optional[int] = None,
                      grad: bool = False):
        """The members' deviations of the day (B, P, 66, H, W). ``steps`` is not used: the grid is the day's."""
        return self.run(self.day_start(x0, e, H, doy0, tisr, lead, gen), grad=grad)

    def day_ctx(self, x0, xb, e, H, doy0, tisr, lead: int, gen=None, sel=None, grad: bool = False):
        """The sampler's context of the day (``day_start``; xb is not used: the path starts from the day's start)."""
        return self.day_start(x0, e, H, doy0, tisr, lead, gen, sel, grad)

    def day_run(self, c, steps: Optional[int] = None, grad: bool = False, jac: bool = False):
        """The members' deviations (B, P', 66, H, W) from the day's context. ``steps``: the Heun steps of the learned
        part of a grid step when the flow has core_sub >= 1 (None = core_sub); not used with core_sub = 0 (one
        Euler step a grid step: the grid is the day's). ``grad``, ``jac``: see ``run``."""
        return self.run(c, grad=grad, jac=jac, sub=steps if self.cfg.core_sub > 0 else None)

    def ctx_rows(self, c, rows):
        """The context of the source rows ``rows`` alone, as one batch element (see DeviationFlow.ctx_rows): the mean
        path, the conditioning and the core's memory stay those of the day (per IC, addressed by the rows' bidx)."""
        q = SimpleNamespace(**vars(c))
        q.B, q.P, q.z0, q.sig, q.bidx = 1, len(rows), c.z0[rows], c.sig[rows], c.bidx[rows]
        q.add = None if c.add is None else c.add[rows]
        return q

    def integrate(self, *a, **k):
        raise NotImplementedError("the core path needs the day's start: use day_ctx / day_run")

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
