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
from .sphere import A_EARTH, SHT, bands
from pgcfm.core import Net  # noqa: E402
from pgcfm.kai_tend import core_step, stateless_norm  # noqa: E402
from pgcfm.seqdct import _NetCfg  # noqa: E402

HALF_LOG_2PIE = 0.5 * (math.log(2 * math.pi) + 1.0)
R_DRY = 287.05  # J / (kg K)


def _softplus_inv(v: float) -> float:
    return math.log(math.expm1(v))


def level_channels():
    """The dynamic-channel indices (4, K) of z, t, u, v on the K pressure levels (top first) and the levels (Pa)."""
    from .env import DYN, contract
    names = [contract.CHANNEL_NAMES[c] for c in DYN]
    idx = [[names.index(f"{v}{p}") for p in contract.LEVELS_HPA] for v in ("z", "t", "u", "v")]
    return torch.tensor(idx), np.asarray(contract.LEVELS_PA, dtype=np.float64)


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
    noise_pre: bool = False   # the new part composed with the equation: the Gaussian field is added to the deviations
    #                           BEFORE the equation's day, e_in = e + h xi, so the day's new variance is M (h xi): it
    #                           lies in the region the equation reaches (balanced, on the equation's scales)
    noise_mult: float = 0.0   # > 0: the noise multiplies the physics. The day's tendency of a member is (1 + a zeta)
    #                           times the equation's, zeta one large-scale field (degrees <= 7) for all channels (the
    #                           balance between the variables is kept), a per channel (start noise_mult). The new
    #                           part is then a zeta (Phi(xbar) - xbar) / s in place of the additive h xi
    e0_spec: str = ""         # the initial deviations' degree spectrum: "" = smooth_noise; a json of
    #                           diag_spectra.py: its 1-day error spectrum (err_deg, all channels), unit variance
    noise: str = "smooth"     # the new part's field: "smooth" (3 x 3 mean of white noise) or "sphere": spherical
    #                           harmonics with a learned amplitude per Littlewood-Paley band and channel (starts as
    #                           the isotropic copy of "smooth") and an AR(1) correlation in time per band;
    #                           "balanced": per pressure level a streamfunction psi and a velocity potential chi
    #                           (spherical harmonics, learned amplitude per level and band), the wind
    #                           k x grad psi + grad chi, the geopotential of the linear balance
    #                           lap Phi = div(f grad psi) and the hydrostatic T = -(1/R) dPhi/dln p; the other
    #                           channels, and a learned small remainder of u, v, T, z, are "sphere" noise
    noise_spec: str = ""      # balanced noise: a json of diag_spectra.py whose 1-day error spectrum of the wind
    #                           (err_deg: u250, else all) starts psi's degree spectrum, amp_psi(l)^2 ~ E(l) / (l (l+1));
    #                           "" = e0_spec if given, else the base spectrum of the sphere noise
    bal_vert: float = 1.0     # balanced noise: the vertical correlation of psi and chi, exp(-(dln p)^2 / (2 d^2))
    #                           (0 = independent levels, whose hydrostatic T is far too large)
    fdt: str = ""             # fluctuation-dissipation: a json of clim_bands.py. The AR(1) rho per band is its
    #                           one-day autocorrelation (fixed, not learned) and ``fdt_penalty`` holds the deviations'
    #                           band power below its climatological band power
    fdt_bmin: int = 0         # the first band of ``fdt_penalty`` (1: without the band l = 0, the global mean, whose
    #                           power is one coefficient a pair and whose climatological value is about 1e-4 s^2)


class EntropyDeviationKernel(nn.Module):
    # the fixed buffers that the balanced noise and the fdt settings derive from the configuration of the run
    CFG_BUFFERS = ("bal_unit", "bal_k", "psi_amp", "bal_vert", "bal_dlnp", "fdt_c", "fdt_rho")

    def __init__(self, core, cfg: DevConfig = DevConfig(), std=None):
        """``std`` (66 or 67 values): the z-score std of the channels, needed by the balanced noise (SI units);
        None = the contract's reference values. A checkpoint carries the constants derived from it."""
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
        if cfg.noise_mult > 0:
            self.a_raw = nn.Parameter(torch.full((66,), math.log(math.expm1(cfg.noise_mult))))
        if cfg.noise not in ("smooth", "sphere", "balanced"):
            raise ValueError(f"noise = {cfg.noise!r}: smooth, sphere or balanced")
        if cfg.noise == "balanced" and cfg.noise_mult > 0:
            raise ValueError("noise_mult replaces the additive new part: it cannot be combined with balanced noise")
        sphere = cfg.noise in ("sphere", "balanced")
        if sphere or cfg.band_scale or cfg.e0_spec or cfg.noise_mult > 0 or cfg.fdt:
            H = int(core.static.shape[-2])
            lat = [-90.0 + 180.0 / H * (i + 0.5) for i in range(H)]  # the data's equiangular grid, south to north
            self.sht = SHT(lat)
            bj, nb = bands(self.sht.L)
            self.register_buffer("band_of", bj)
            self.n_bands = nb
        if cfg.e0_spec:
            import json
            sp = torch.tensor(json.load(open(cfg.e0_spec))["err_deg"]["all"][0], dtype=torch.float32)
            self.register_buffer("e0_amp", (sp / sp.sum()).clamp(min=0).sqrt())  # (L,), unit total variance
        if cfg.band_scale:
            self.g_band = nn.Parameter(torch.zeros(10, 66, nb))  # log scale per lead (day 1..10), channel, band
        if sphere:
            # the degree spectrum of smooth_noise (unit variance): the start of the band amplitudes
            with torch.no_grad():
                z = smooth_noise((512, 1, H, 2 * H), self.sht.syn, torch.Generator().manual_seed(0))
                base = self.sht.spectrum(z).mean(0).squeeze(0)
            self.register_buffer("base_amp", base.clamp(min=1e-12).sqrt())  # (L,)
            self.a_log = nn.Parameter(torch.zeros(66, nb))      # log amplitude per channel and band
            self.rho_raw = nn.Parameter(torch.zeros(nb))         # AR(1) in time per band, rho = tanh(raw)
        if cfg.noise == "balanced":
            self._init_balanced(std)
        if cfg.fdt:
            import json
            d = json.load(open(cfg.fdt))
            c, rho = torch.tensor(d["c"], dtype=torch.float32), torch.tensor(d["rho_band"], dtype=torch.float32)
            if tuple(c.shape) != (66, nb) or tuple(rho.shape) != (nb,):
                raise ValueError(f"{cfg.fdt}: c {tuple(c.shape)} and rho_band {tuple(rho.shape)}, not (66, {nb}) and ({nb},)")
            self.register_buffer("fdt_c", c.clamp(min=1e-10))            # climatological band power, units of s^2
            self.register_buffer("fdt_rho", rho.clamp(0.0, 0.999))       # one-day autocorrelation per band
            if sphere:
                self.rho_raw.requires_grad_(False)                       # kept for the checkpoints, not used
        self._xi, self._bc = None, None
        if cfg.correct:
            # at the new state: x_t, G(x_t), the cloud's shift of the mean (mean of Phi over the members - Phi(xbar),
            # the equation's own nonlinear correction), the day's tendency, log size before / after, the entropy
            # H, static, time, lead, m. Out: delta (mean, units of the cloud's size exp(H)) and r2 (entropy); zero
            # head, so the chain starts as the uncorrected kernel
            c2 = 66 + n_g + 66 + 66 + 66 + 66 + 66 + core.static.shape[0] + 4 + 1 + self.mem_ch
            self.cnet = stateless_norm(Net(c2, 132, _NetCfg(cfg.hidden, tuple(cfg.blocks))))

    def _init_balanced(self, std):
        """The constants of the balanced noise. psi_k = K_k a_k A_k(l) sum_j V_kj eta_j: eta unit coefficients,
        V the vertical correlation, A = exp(p_log)[band] psi_amp(l) / sqrt(2 l + 1) with sum_l l (l + 1) psi_amp^2
        = 1, so the wind of level k has the area mean u^2 + v^2 = (K_k a_k / a)^2. K_k makes the mean of the
        variances of u and v in the kernel's units (wind / (std s)) one at a_k = 1."""
        from .env import DYN, contract
        cfg, L, nb = self.cfg, self.sht.L, self.n_bands
        idx, p = level_channels()
        K = idx.shape[1]
        sd = np.asarray(contract.REFERENCE_STD if std is None else std, dtype=np.float64).reshape(-1)
        sd = sd[DYN] if sd.shape[0] == 67 else sd
        unit = (torch.tensor(sd, dtype=torch.float32) * self.core.t6.reshape(-1).cpu())[idx]     # (4, K) SI per e
        self.register_buffer("bal_idx", idx)
        self.register_buffer("bal_unit", unit)
        self.register_buffer("bal_k", A_EARTH * torch.sqrt(4.0 / (unit[2] ** -2 + unit[3] ** -2)))   # (K,) m2/s
        # the wind's degree spectrum E(l): the 1-day error of a diag_spectra json, else the sphere noise's base
        spec = cfg.noise_spec or cfg.e0_spec
        if spec:
            import json
            d = json.load(open(spec))["err_deg"]
            E = torch.tensor(d.get("u250", d["all"])[0], dtype=torch.float32)
            if E.shape[0] < L:
                raise ValueError(f"{spec}: {E.shape[0]} degrees, the grid has {L}")
            E = E[:L].clamp(min=0)
        else:
            E = self.base_amp.pow(2).cpu()
        l = torch.arange(L, dtype=torch.float32)
        a2 = torch.where(l > 0, E / (l * (l + 1)).clamp(min=1), torch.zeros(L))    # the kinetic energy of a psi mode
        self.register_buffer("psi_amp", (a2 / (a2 * l * (l + 1)).sum()).sqrt())    # is l (l + 1) / a^2 |psi_lm|^2
        # the vertical correlation C_kj = exp(-(ln p_k - ln p_j)^2 / (2 d^2)): V = C^(1/2) with unit rows
        lnp = np.log(p)
        if cfg.bal_vert > 0:
            ev, U = np.linalg.eigh(np.exp(-0.5 * ((lnp[:, None] - lnp[None]) / cfg.bal_vert) ** 2))
            V = (U * np.sqrt(np.maximum(ev, 0.0))) @ U.T
            V = V / np.sqrt((V ** 2).sum(1, keepdims=True))
        else:
            V = np.eye(K)
        self.register_buffer("bal_vert", torch.tensor(V, dtype=torch.float32))
        # hydrostatic: T = -(1/R) dPhi/dln p, centred differences in ln p, one-sided at the two ends
        Dp = np.zeros((K, K))
        for k in range(K):
            lo, hi = max(k - 1, 0), min(k + 1, K - 1)
            Dp[k, hi] += 1.0 / (lnp[hi] - lnp[lo])
            Dp[k, lo] -= 1.0 / (lnp[hi] - lnp[lo])
        self.register_buffer("bal_dlnp", torch.tensor(Dp, dtype=torch.float32))
        self.p_log = nn.Parameter(torch.zeros(K, nb))                               # log amplitude of psi
        self.chi_raw = nn.Parameter(torch.full((K, nb), _softplus_inv(0.1)))        # chi / psi amplitude
        self.w_raw = nn.Parameter(torch.full((4, K), _softplus_inv(0.3)))           # the remainder of z, t, u, v

    def load_init(self, state):
        """Start from the state of a fitted kernel (strict=False): the learned parts are taken, the fixed buffers
        of ``CFG_BUFFERS`` stay those of this kernel's own configuration (its cfg is what a checkpoint records).
        Returns (the keys' result of load_state_dict, the names of the buffers where the state differed)."""
        own = {k: getattr(self, k).clone() for k in self.CFG_BUFFERS if hasattr(self, k)}
        miss = self.load_state_dict(state, strict=False)
        diff = [k for k, v in own.items() if k in state and not torch.equal(state[k].to(v), v)]
        with torch.no_grad():
            for k, v in own.items():
                getattr(self, k).copy_(v)
        return miss, diff

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

    def rho(self):
        """(L,): the AR(1) coefficient in time of every degree (per band): learned, or the climate's (fdt)."""
        return (self.fdt_rho if self.cfg.fdt else torch.tanh(self.rho_raw))[self.band_of]

    def new_noise(self, shape, like, gen=None):
        """The new part's unit field xi (shape (B, P, 66, H, W)); sphere noise keeps its last draw for the AR(1)."""
        if self.cfg.noise == "smooth":
            return smooth_noise(shape, like, gen)
        eta = self.sht.noise(self.amp(), tuple(shape[:2]) + (66,), gen)  # differentiable in the amplitudes
        if self._xi is None or self._xi.shape != eta.shape:
            xi = eta
        else:  # band by band: xi_t = rho xi_(t-1) + sqrt(1 - rho^2) eta_t, done in spectral space
            rho = self.rho()                                                  # (L,)
            cp, ce = self.sht.analysis(self._xi), self.sht.analysis(eta)
            xi = self.sht.synthesis(rho[:, None] * cp + torch.sqrt(1 - rho ** 2)[:, None] * ce)
        self._xi = xi.detach()
        return xi

    def balance(self, psi, chi=None):
        """The balanced fields of a streamfunction and a velocity potential, coefficients (..., K, L, M) in m2/s
        on the K levels: u, v (m/s) = k x grad psi + grad chi, Phi (m2/s2) with lap Phi = div(f grad psi), and
        T (K) = -(1/R) dPhi/dln p. Each (..., K, H, W)."""
        u, v = self.sht.wind(psi, chi)
        phi = self.sht.synthesis(self.sht.linear_balance(psi))
        T = -torch.einsum("kj,...jhw->...khw", self.bal_dlnp, phi) / R_DRY
        return u, v, phi, T

    def psi_chi(self, c):
        """The coefficients of psi and chi (B, P, K, L, M), m2/s, from the unit draws c (2, B, P, K, L, M). The
        amplitude a_k = sqrt(h_u h_v) of the level is applied to psi before any derivative, so the wind, Phi and
        T of the level stay balanced with each other."""
        h = self.h().reshape(-1)
        a = torch.sqrt(h[self.bal_idx[2]] * h[self.bal_idx[3]]) * self.bal_k                       # (K,)
        A = torch.exp(self.p_log)[:, self.band_of] * self.psi_amp / torch.sqrt(2 * self.sht.deg + 1)  # (K, L)
        A = (a[:, None] * A)[:, :, None]
        mix = lambda t: torch.complex(torch.einsum("kj,...jlm->...klm", self.bal_vert, t.real),  # noqa: E731
                                      torch.einsum("kj,...jlm->...klm", self.bal_vert, t.imag))
        return A * mix(c[0]), A * F.softplus(self.chi_raw)[:, self.band_of][:, :, None] * mix(c[1])

    def remainder_weight(self):
        """(66,): the weight of the independent sphere noise: learned (start 0.3) for u, v, T, z, 1 elsewhere."""
        w = torch.ones(66, device=self.w_raw.device)
        return w.index_put((self.bal_idx.reshape(-1),), F.softplus(self.w_raw).reshape(-1))

    def new_part(self, shape, like, gen=None):
        """h xi, the day's new part (B, P, 66, H, W) in units of s, tisr excluded. Balanced noise:
        (balanced fields of a_k psi, chi) / (std s) + h_c w_c (sphere noise)_c, the draws of psi, chi and of the
        remainder each with the AR(1) in time per band."""
        hxi = self.h().unsqueeze(0) * (self.new_noise(shape, like, gen) * self.keep.unsqueeze(0))
        if self.cfg.noise != "balanced":
            return hxi
        c = self.sht.noise_coef((2,) + tuple(shape[:2]) + (self.bal_idx.shape[1],), like, gen)
        if self._bc is not None and self._bc.shape == c.shape:
            rho = self.rho()[:, None]
            c = rho * self._bc + torch.sqrt(1 - rho ** 2) * c
        self._bc = c.detach()
        u, v, phi, T = self.balance(*self.psi_chi(c))
        bal = torch.cat([phi, T, u, v], 2) / self.bal_unit.reshape(1, 1, -1, 1, 1)
        return (hxi * self.remainder_weight().view(1, 1, 66, 1, 1)).index_add(2, self.bal_idx.reshape(-1), bal)

    @torch.no_grad()
    def start(self, xd, doy, hist=None, gen=None):
        """The trajectory starts at x (B, 66, H, W): the mean's memory (memory core) and the initial deviations
        e_0 (B, P, 66, H, W) = eps0 x smooth unit noise."""
        if self.mem:
            self.core.reset(hist, doy.float() * 24.0, xd)
        B, P = xd.shape[0], self.cfg.pairs
        self._xi, self._bc = None, None
        if self.cfg.e0_spec:  # shaped like the 1-day error: the variance where the error grows from, not at l 16-31
            z = self.sht.noise(self.e0_amp.to(xd), (B, P, xd.shape[1]), gen)
        else:
            z = smooth_noise((B, P) + tuple(xd.shape[1:]), xd, gen)
        return self.cfg.eps0 * z * self.keep.unsqueeze(0)

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
        hxi = self.new_part(tuple(e.shape), xbar, gen)
        if self.cfg.noise_pre:
            # K_phys o G: the noise goes through the equation with the deviations (linear part M (e + h xi)); the
            # gradient to h and to the noise amplitudes goes through the straight-through copy of the injected part
            inj = hxi
            phi0, p, g, shift = self._equation(xbar, e + inj.detach(), doy, tisr)
            q = p + (inj - inj.detach())
        else:
            phi0, p, g, shift = self._equation(xbar, e, doy, tisr)
            q = p + hxi
        if self.cfg.noise_mult > 0:
            L_ = self.sht.L
            amp = torch.zeros(L_, device=xbar.device)
            amp[1:8] = torch.sqrt((2 * torch.arange(1, 8, device=xbar.device) + 1) / 63.0)  # unit variance, l = 1..7
            zeta = self.sht.noise(amp, (B, e.shape[1], 1), gen)                               # (B, P, 1, H, W)
            tend0 = ((phi0 - xbar) / self.s * self.keep).detach()
            q = p + F.softplus(self.a_raw).view(1, 1, 66, 1, 1) * zeta * tend0[:, None]
        g = None if g is None else g.detach()
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

    def _band(self, t):
        """(..., L) degree power -> (..., nb) band power."""
        return torch.stack([t[..., self.band_of == b].sum(-1) for b in range(self.n_bands)], -1)

    def band_power(self, e):
        """V_cj (B, 66, nb): the band power of the deviations (B, P, 66, H, W), mean over the pairs = the
        ensemble's variance about its mean in the band (units of s^2)."""
        return self._band(self.sht.spectrum(e).mean(1)).clamp(min=1e-10)

    def fdt_penalty(self, e):
        """The one-sided fluctuation-dissipation bound: the spread about the mean cannot exceed the climate's
        variance, mean over the channels (tisr excluded) and bands j >= fdt_bmin of relu(log V_cj - log c_cj)^2.
        V_cj is the mean over the pairs AND the batch before the log: the log of a band power estimated from few
        coefficients scatters, and a one-sided penalty of every sample is then positive at V = c (0.20 in the band
        l = 0 with 2 pairs), which would push the large scales below the climate instead of capping them."""
        over = F.relu(torch.log(self.band_power(e).mean(0)) - torch.log(self.fdt_c))       # (66, nb)
        return over[self.keep.reshape(-1) > 0, self.cfg.fdt_bmin:].pow(2).mean()

    def band_ign(self, e, xbar, truth):
        """The Gaussian log score by scale: per channel and band j of degrees, the error's power E_j against the
        deviations' power V_j (mean over the pairs), over the band's n_j coefficients,
        n_j (1/2 log (V_j / n_j) + E_j / (2 V_j)) / sum_j n_j, averaged over the channels (tisr excluded).
        Its minimum is V_j = E_j: the spread in every band the size of the error there."""
        bj = self.band_of
        n = torch.stack([(2 * torch.arange(self.sht.L, device=bj.device)[bj == b] + 1).sum()
                         for b in range(self.n_bands)]).to(e)
        V = self.band_power(e)                                                     # (B, 66, nb)
        E = self._band(self.sht.spectrum((truth - xbar) / self.s))
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
