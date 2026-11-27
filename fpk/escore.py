"""The fair energy score of an ensemble in a Hilbert norm, differentiable: the loss of ``run_fpk.py devtune``.

    ES(F, y) = E|X - y| - 1/2 E|X - X'|                    X, X' independent draws of the forecast F, y the truth;
                                                           strictly proper when |.| is a Hilbert norm
    fair     = (1 / M) sum_i |X_i - y|  -  1 / (2 M (M - 1)) sum_(i != j) |X_i - X_j|        M >= 2 members

The fair form is unbiased for M independent members (the form with 1 / (2 M^2) is high by E|X - X'| / (2 M): it
charges the spread, which rewards a too narrow ensemble at small M). |f| = sqrt(|f|^2 + eps): eps inside the root
keeps the gradient finite where two members coincide (eps = 0: the plain root, the numbers of
scripts/diag_spectra.py).

Antithetic pairs (``energy_score(..., pairs=True)``): the kernel's 2 P members xbar +- s e_p are P independent pairs,
not 2 P independent members. A member and its mirror are 2 |e| apart, two independent draws sqrt(2) E|e| on average
(Gaussian), so the fair form over all pairs of members overstates 1/2 E|X - X'| by the factor
1 + (sqrt(2) - 1) / (2 P - 1) and the score comes out low by (2 - sqrt(2)) E|e| / (2 (2 P - 1)): 14, 6 and 3 % of the
score of a calibrated Gaussian at P = 2, 4, 8. The spread term is therefore taken over members of DIFFERENT pairs only,

    pairs    = (1 / 2P) sum_i |X_i - y|  -  1 / (2 * 2P (2P - 2)) sum_(i, j in different pairs) |X_i - X_j|      P >= 2

which is unbiased when every member has the forecast's law (a law symmetric about the mean, as the kernel's) and the
pairs are independent: any two members of different pairs are then two independent draws.

The norms, on fields f (..., 66, H, W) in units of s (<.> = the area mean, k_c = 0 on tisr and 1 elsewhere, K = 65):

    grid     |f|^2 = (1 / K) sum_c k_c <f_c^2>                                  the RMS in units of s
    band     |f|^2 = sum_c sum_j w_cj |P_j f_c|^2,     w_cj = k_c / (K nb) * (sum_i n_i) / (n_j c_cj)
             P_j f the part of f in the Littlewood-Paley band j of degrees (fpk.sphere.bands), |P_j f_c|^2 its power
             (area mean of its square), n_j = sum over the band of 2 l + 1, c_cj the climatological band power ("c"
             of scripts/clim_bands.py, units of s^2), nb the number of bands: the band-weighted norm of
             scripts/diag_spectra.py (one implementation; diag_spectra passes the c it computes itself). The power
             of a band relative to the climate's is weighted by (sum n) / n_j = 1024, 341, 85, 21, 5.3, 1.3 on the
             32 x 64 grid: the norm is mostly the bands l = 0 and l = 1. ``bmin``: the first band that counts.
             bmin = 1 leaves out l = 0, the global mean: that band is the core's own mass drift and its
             climatological power is tiny (so 1 / c is huge: with clim_bands.json the global mean of msl alone
             is most of the norm). It is the default of ``devtune --band-bmin`` and the norm of diag_spectra's
             "energy_score_b1"; bmin = 0 (the default here) is diag_spectra's "energy_score".
    energy   |f|^2 = < sum_k (dp_k / g) [ 1/2 (u_k'^2 + v_k'^2) + c_p / (2 T_r) T_k'^2 + L^2 / (2 c_p T_r) q_k'^2 ]
                       + R T_r p_r / (2 g) (p_s' / p_r)^2 >                    J / m^2, x' = std s f in SI units
             the dry + moist total-energy norm: dp_k the trapezoid weights of the pressure levels (sum 950 hPa),
             T_r = 270 K, p_r = 1000 hPa, p_s the surface pressure channel "sp". The geopotential z, msl, t2m,
             skt, tcwv (and tisr) have ZERO weight: a score in this norm does not see them and gives them no
             gradient (the energy of the mass field is in T and p_s).
"""

import json
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .env import DYN, contract
from .sphere import SHT, bands

G0 = 9.80665          # m / s^2
CP = 1004.0           # J / (kg K)
R_DRY = 287.05        # J / (kg K)
LV = 2.5e6            # J / kg
T_REF = 270.0         # K
P_REF = 1.0e5         # Pa


def band_power(S: SHT, f: torch.Tensor, bj: Optional[torch.Tensor] = None) -> torch.Tensor:
    """(..., H, W) -> (..., nb): the power of the field in every Littlewood-Paley band of degrees (their sum is the
    area mean of f^2 for a field inside the truncation)."""
    if bj is None:
        bj = bands(S.L)[0]
    sp = S.spectrum(f)
    return torch.stack([sp[..., bj == b].sum(-1) for b in range(int(bj.max()) + 1)], -1)


def band_weights(c: torch.Tensor, keep: torch.Tensor, L: int) -> torch.Tensor:
    """w_cj (C, nb) = k_c / (K nb) * (sum_i n_i) / (n_j c_cj) from the climatological band power c (C, nb) and the
    channel mask keep (C values): the weights of the band norm."""
    bj, nb = bands(L)
    nj = torch.tensor([float((2 * torch.arange(L)[bj == b] + 1).sum()) for b in range(nb)])
    k = keep.reshape(-1, 1).cpu() > 0
    return (1.0 / (nj * c / nj.sum())) * k / float(k.sum()) / nb


class Norm(nn.Module):
    """A Hilbert norm of fields (..., 66, H, W) -> (...,): ``sq`` the squared norm, the call sqrt(sq + eps)."""

    def __init__(self, eps: float = 0.0):
        super().__init__()
        self.eps = float(eps)

    def sq(self, f: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return (self.sq(f) + self.eps).sqrt()


class GridNorm(Norm):
    """|f|^2 = sum k_c w_x f_c(x)^2 / sum k_c w_x: keep (66 values) the channel mask, w (H values) area weights."""

    def __init__(self, keep, w, eps: float = 0.0):
        super().__init__(eps)
        self.register_buffer("m", keep.reshape(-1, 1, 1).float() * w.reshape(1, -1, 1).float())   # (66, H, 1)

    def sq(self, f):
        return (f.pow(2) * self.m).sum((-3, -2, -1)) / self.m.expand(f.shape[-3:]).sum()


class BandNorm(Norm):
    """|f|^2 = sum_c sum_j w_cj |P_j f_c|^2 with the weights of ``band_weights``; ``bmin``: the bands j < bmin do
    not count (1 = without l = 0, the global mean)."""

    def __init__(self, S: SHT, c, keep, bmin: int = 0, eps: float = 0.0):
        super().__init__(eps)
        self.S = S
        bj, nb = bands(S.L)
        c = torch.as_tensor(c, dtype=torch.float32)
        if tuple(c.shape) != (keep.numel(), nb):
            raise ValueError(f"climatological band power {tuple(c.shape)}, not ({keep.numel()}, {nb})")
        w = band_weights(c, keep, S.L)
        if bmin:
            w = w.clone()
            w[:, :bmin] = 0.0
        self.register_buffer("bj", bj)
        self.register_buffer("w", w)

    def power(self, f):
        return band_power(self.S, f, self.bj)

    def sq(self, f):
        return (self.power(f) * self.w).sum((-1, -2))


class EnergyNorm(Norm):
    """The dry + moist total-energy norm (J / m^2, the formula of the module's docstring). ``s`` (66 values): the unit
    of the fields in z-score units (core.t6); ``std`` (66 or 67 values): the z-score std of the channels, so the SI
    perturbation of channel c is std_c s_c f_c; ``lat`` (H,) degrees. ``a`` (66,): the weight of <f_c^2>, zero on
    z, t2m, msl, skt, tcwv and tisr."""

    def __init__(self, s, std, lat, eps: float = 0.0):
        super().__init__(eps)
        names = [contract.CHANNEL_NAMES[c] for c in DYN]
        p = np.asarray(contract.LEVELS_PA, dtype=np.float64)
        dp = np.gradient(p)
        dp[0], dp[-1] = 0.5 * (p[1] - p[0]), 0.5 * (p[-1] - p[-2])            # trapezoid: half a layer at the ends
        a = np.zeros(len(names))
        for k, lev in enumerate(contract.LEVELS_HPA):
            m = dp[k] / G0                                                    # kg / m^2 of the layer
            a[names.index(f"u{lev}")] = a[names.index(f"v{lev}")] = 0.5 * m
            a[names.index(f"t{lev}")] = m * CP / (2.0 * T_REF)
            a[names.index(f"q{lev}")] = m * LV ** 2 / (2.0 * CP * T_REF)
        if "sp" in names:
            a[names.index("sp")] = R_DRY * T_REF / (2.0 * G0 * P_REF)         # (R T_r p_r / 2 g) / p_r^2
        sd = np.asarray(std, dtype=np.float64).reshape(-1)
        sd = sd[DYN] if sd.shape[0] == 67 else sd
        unit = sd * torch.as_tensor(s).detach().double().reshape(-1).cpu().numpy()
        w = np.cos(np.deg2rad(np.asarray(lat, dtype=np.float64)))
        self.register_buffer("a", torch.tensor(a * unit ** 2, dtype=torch.float32))
        self.register_buffer("m", self.a.view(-1, 1, 1) * torch.tensor(w / w.mean(), dtype=torch.float32).view(1, -1, 1))

    def sq(self, f):
        return (f.pow(2) * self.m).sum((-3, -2, -1)) / (f.shape[-2] * f.shape[-1])


def make_norm(kind: str, keep, lat, s=None, std=None, clim: str = "", bmin: int = 0, eps: float = 1e-12) -> Norm:
    """The norm of a name: grid | band (``clim``: a json of scripts/clim_bands.py, its "c") | energy (``s``, ``std``)."""
    if kind == "grid":
        w = np.cos(np.deg2rad(np.asarray(lat, dtype=np.float64)))
        return GridNorm(keep, torch.tensor(w / w.mean(), dtype=torch.float32), eps)
    if kind == "band":
        if not clim:
            raise ValueError("the band norm needs the climatological band power: a json of scripts/clim_bands.py")
        return BandNorm(SHT(lat), json.load(open(clim))["c"], keep, bmin, eps)
    if kind == "energy":
        return EnergyNorm(s, std, lat, eps)
    raise ValueError(f"norm = {kind!r}: grid, band or energy")


def energy_score(x: torch.Tensor, y: torch.Tensor, norm, pairs: bool = False):
    """The fair energy score of the members x (B, M, 66, H, W) against y (B, 66, H, W) in ``norm`` (fields -> their
    norm over the last three axes). Returns (ES, E|X - y|, 1/2 E|X - X'|), each (B,): the mean over the members of
    |X_i - y| and half the mean over the ordered pairs i != j of |X_i - X_j| (every shift q = 1 .. M - 1 of the
    members against themselves), ES their difference. ``pairs``: the members are P = M / 2 antithetic pairs, x[:, p]
    and x[:, P + p] the two of pair p (the kernel's [e, -e]); the shift q = P, a member against its own partner, is
    left out of the second term (the module's docstring), M - 2 shifts instead of M - 1."""
    M = x.shape[1]
    if M < 2:
        raise ValueError("the fair energy score needs at least 2 members")
    if pairs and (M % 2 or M < 4):
        raise ValueError("antithetic pairs: an even number of members, at least 2 pairs")
    d1 = norm(x - y[:, None]).mean(1)
    d2 = sum(norm(x - x.roll(q, 1)).mean(1) for q in range(1, M) if not (pairs and 2 * q == M)) / (M - 1 - bool(pairs))
    return d1 - 0.5 * d2, d1, 0.5 * d2


def spread_skill(x: torch.Tensor, y: torch.Tensor, grid: Norm):
    """(spread^2, skill^2), each (B,), in the squared norm ``grid``: the members' variance about their mean
    (unbiased, 1 / (M - 1)) and the squared error of their mean. A calibrated ensemble of M independent members has
    E[spread^2] (M + 1) / M = E[skill^2]."""
    m = x.mean(1)
    return grid.sq(x - m[:, None]).sum(1) / (x.shape[1] - 1), grid.sq(m - y)


__all__ = ["Norm", "GridNorm", "BandNorm", "EnergyNorm", "make_norm", "energy_score", "spread_skill", "band_power",
           "band_weights"]
