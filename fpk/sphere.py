"""Spherical harmonic transform on the lat-lon grid of the data, and fields built degree by degree.

The grid is equiangular (H latitudes, W = 2 H longitudes, no poles). The transform keeps the triangular truncation
l <= lmax = H - 1 and uses, per order m, the weighted least-squares inverse of the synthesis matrix: exact for every
field of degree <= lmax (the analysis of a synthesis is the identity), and the area-weighted best fit otherwise.

    SHT(lat).analysis(x)        (..., H, W)  -> complex (..., lmax + 1, lmax + 1)     [l, m], m >= 0
    SHT(lat).synthesis(c)       complex (..., L, L) -> (..., H, W)
    SHT(lat).spectrum(x)        (..., H, W)  -> (..., L)   the power of every degree l, summed over m; its sum is
                                                           the area mean of x^2 (Parseval on the unit sphere / 4 pi)
    SHT(lat).noise(amp, ...)    a Gaussian field with degree spectrum amp(l)^2 (isotropic on the sphere)

Differential operators on the sphere of radius a (``radius``, default the Earth's), on the coefficients:
    lap(c), inv_lap(c)          -l (l + 1) / a^2 and its inverse (l = 0 -> 0)
    grad(c)                     ((1 / (a cos phi)) d/dlam, (1 / a) d/dphi) on the grid: d/dlam = i m, d/dphi through
                                the derivative of the Legendre basis (exact for the truncated field at every point)
    wind(psi, chi)              (u, v) = k x grad psi + grad chi
    helmholtz(u, v)             the (psi, chi) of a wind (weighted least squares, the inverse of ``wind``)
    vortdiv(u, v)               (lap psi, lap chi) on the grid
    linear_balance(psi)         Phi with lap Phi = div(f grad psi), f = 2 Omega sin(phi)

Normalization: Y_lm orthonormal over the sphere with measure dOmega / (4 pi), so a field's area mean of x^2 equals
sum_l sum_m |c_lm|^2 (with m > 0 counted twice for a real field).
"""

import math
from typing import Optional

import numpy as np
import torch

A_EARTH = 6.371e6     # m
OMEGA = 7.292e-5      # 1/s


def _legendre(lmax: int, x: np.ndarray) -> np.ndarray:
    """Normalized associated Legendre functions P[l, m, i] at x = sin(lat), with int over the sphere of
    (P_lm(x) e^{i m lon})^2 dOmega / (4 pi) = 1 (Condon-Shortley phase left out). Stable recursion in l."""
    n = x.shape[0]
    P = np.zeros((lmax + 1, lmax + 1, n))
    c = np.sqrt(np.maximum(0.0, 1.0 - x ** 2))
    P[0, 0] = 1.0
    for m in range(1, lmax + 1):
        P[m, m] = P[m - 1, m - 1] * c * math.sqrt((2 * m + 1) / (2 * m))
    for m in range(0, lmax):
        P[m + 1, m] = x * math.sqrt(2 * m + 3) * P[m, m]
    for m in range(0, lmax + 1):
        for l in range(m + 2, lmax + 1):
            a = math.sqrt((4 * l * l - 1) / (l * l - m * m))
            b = math.sqrt(((l - 1) ** 2 - m * m) / (4 * (l - 1) ** 2 - 1))
            P[l, m] = a * (x * P[l - 1, m] - b * P[l - 2, m])
    return P


class SHT(torch.nn.Module):
    def __init__(self, lat_deg, lmax: Optional[int] = None, radius: float = A_EARTH):
        super().__init__()
        self.radius = float(radius)
        lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
        H = lat.shape[0]
        L = (H - 1 if lmax is None else int(lmax)) + 1
        self.H, self.W, self.L = H, 2 * H, L
        P = _legendre(L - 1, np.sin(lat))  # (L, L, H)
        w = np.cos(lat)
        w = w / w.sum()  # area weights of the rows (the mean over a row is taken by the FFT)
        syn = np.zeros((L, H, L))  # per m: (H, L) synthesis matrix over l (zero for l < m)
        ana = np.zeros((L, L, H))
        for m in range(L):
            A = P[m:, m, :].T  # (H, L - m)
            Aw = A * w[:, None]
            # weighted least squares: c = (A^T W A)^-1 A^T W f
            G = A.T @ Aw
            ana[m, m:, :] = np.linalg.solve(G, Aw.T)
            syn[m, :, m:] = A
        self.register_buffer("syn", torch.tensor(syn, dtype=torch.float32))   # (M, H, L)
        self.register_buffer("ana", torch.tensor(ana, dtype=torch.float32))   # (M, L, H)
        self.register_buffer("deg", torch.arange(L, dtype=torch.float32))
        # ---- derivatives (buffers outside the state dict: checkpoints keep their keys)
        # eps[l, m] = sqrt((l^2 - m^2) / (4 l^2 - 1)):  x P_l = eps_(l+1) P_(l+1) + eps_l P_(l-1)  and
        # cos(phi) dP_l/dphi = (l + 1) eps_l P_(l-1) - l eps_(l+1) P_(l+1)      (x = sin(phi), fixed m)
        P1 = _legendre(L, np.sin(lat))                                         # one degree more: (L + 1, L + 1, H)
        ll, mm = np.meshgrid(np.arange(L + 1), np.arange(L), indexing="ij")
        eps = np.sqrt(np.maximum(ll ** 2 - mm ** 2, 0) / np.maximum(4.0 * ll ** 2 - 1, 1.0))   # (L + 1, M)
        dsyn, csyn = np.zeros((L, H, L)), np.zeros((L, H, L))
        hel = np.zeros((L, 2 * L, 2 * H), dtype=np.complex128)
        for m in range(L):
            for l in range(m, L):
                lo = P1[l - 1, m] if l > m else 0.0
                dsyn[m, :, l] = ((l + 1) * eps[l, m] * lo - l * eps[l + 1, m] * P1[l + 1, m]) / np.cos(lat)
                csyn[m, :, l] = m * P1[l, m] / np.cos(lat)
            # the wind of (psi_l, chi_l) of order m at the H latitudes, radius 1:
            #   u = -sum psi_l dP_l + i sum chi_l C_l,   v = i sum psi_l C_l + sum chi_l dP_l,   C = m P / cos(phi)
            k = np.arange(max(m, 1), L)                                         # l = 0 has no gradient
            A = np.block([[-dsyn[m][:, k], 1j * csyn[m][:, k]], [1j * csyn[m][:, k], dsyn[m][:, k]]])
            Aw = A * np.concatenate([w, w])[:, None]
            sol = np.linalg.solve(A.conj().T @ Aw, Aw.conj().T)                 # (2 n, 2 H)
            hel[m, k] = sol[: len(k)]
            hel[m, L + k] = sol[len(k):]
        f32 = lambda t: torch.tensor(t, dtype=torch.float32)  # noqa: E731
        self.register_buffer("dsyn", f32(dsyn), persistent=False)              # (M, H, L)  dP_lm / dphi
        self.register_buffer("csyn", f32(csyn), persistent=False)              # (M, H, L)  m P_lm / cos(phi)
        self.register_buffer("hel", torch.tensor(hel, dtype=torch.complex64), persistent=False)   # (M, 2 L, 2 H)
        self.register_buffer("eps", f32(eps), persistent=False)                # (L + 1, M)

    def analysis(self, x: torch.Tensor) -> torch.Tensor:
        """(..., H, W) real -> (..., L, M) complex c[l, m]."""
        f = torch.fft.rfft(x.float(), dim=-1, norm="forward")[..., : self.L]  # (..., H, M): the mean over lon
        re = torch.einsum("mlh,...hm->...lm", self.ana, f.real)
        im = torch.einsum("mlh,...hm->...lm", self.ana, f.imag)
        return torch.complex(re, im)

    def synthesis(self, c: torch.Tensor) -> torch.Tensor:
        """(..., L, M) complex -> (..., H, W) real."""
        return self._lon(self._lat(self.syn, c))

    @staticmethod
    def _lat(mat: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """The sum over l with a real (M, H, L) matrix: complex (..., L, M) -> complex (..., H, M)."""
        re = torch.einsum("mhl,...lm->...hm", mat, c.real)
        im = torch.einsum("mhl,...lm->...hm", mat, c.imag)
        return torch.complex(re, im)

    def _lon(self, f: torch.Tensor) -> torch.Tensor:
        """The Fourier sum over m: complex (..., H, M) -> real (..., H, W)."""
        pad = self.W // 2 + 1 - self.L
        if pad > 0:
            f = torch.nn.functional.pad(f, (0, pad))
        return torch.fft.irfft(f, n=self.W, dim=-1, norm="forward")

    def spectrum(self, x: torch.Tensor) -> torch.Tensor:
        """(..., H, W) -> (..., L) power of degree l: |c_l0|^2 + 2 sum_{m>0} |c_lm|^2."""
        p = self.analysis(x).abs().pow(2)
        p = torch.cat([p[..., :1], 2 * p[..., 1:]], -1)
        return p.sum(-1)

    def noise(self, amp: torch.Tensor, shape, gen: Optional[torch.Generator] = None) -> torch.Tensor:
        """A real Gaussian field (*shape, H, W) whose degree-l power is amp[..., l]^2 (amp broadcasts against
        (*shape, L)). Isotropic: the power of degree l is spread evenly over its 2 l + 1 orders."""
        L = self.L
        # E|c_lm|^2 = 1 for every (l, m) present; degree power = 2 l + 1 -> scale to amp^2
        scale = amp / torch.sqrt(2 * self.deg + 1)
        c = self.noise_coef(shape, amp, gen) * scale[..., :, None]
        return self.synthesis(c)

    def noise_coef(self, shape, like: torch.Tensor, gen: Optional[torch.Generator] = None) -> torch.Tensor:
        """Unit Gaussian coefficients (*shape, L, M) of a real field: E|c_lm|^2 = 1 for m <= l (m = 0 real), zero
        above; the power of degree l is 2 l + 1."""
        L = self.L
        z = torch.randn(tuple(shape) + (L, L, 2), generator=gen, device="cpu").to(like)
        c = torch.complex(z[..., 0], z[..., 1]) / math.sqrt(2.0)
        c[..., 0] = torch.complex(z[..., 0, 0], torch.zeros_like(z[..., 0, 0]))  # m = 0 real
        tri = (torch.arange(L)[None, :] <= torch.arange(L)[:, None]).to(like)   # m <= l
        return c * tri

    # ---- differential operators (coefficients (..., L, M), radius a)
    def lap(self, c: torch.Tensor) -> torch.Tensor:
        """The Laplacian: -l (l + 1) / a^2 c_lm."""
        return c * (-self.deg * (self.deg + 1) / self.radius ** 2)[:, None]

    def inv_lap(self, c: torch.Tensor) -> torch.Tensor:
        """The inverse Laplacian: -a^2 / (l (l + 1)) c_lm, and 0 for l = 0."""
        k = self.deg * (self.deg + 1)
        return c * torch.where(k > 0, -self.radius ** 2 / k.clamp(min=1), torch.zeros_like(k))[:, None]

    def grad(self, c: torch.Tensor):
        """The gradient on the grid, ((1 / (a cos phi)) df/dlam, (1 / a) df/dphi): d/dlam is i m on the
        coefficients, d/dphi the sum over dP_lm/dphi. Exact at every grid point for the truncated field."""
        fx = self._lat(self.csyn, c)
        gx = self._lon(torch.complex(-fx.imag, fx.real))                       # i m P / cos(phi)
        return gx / self.radius, self._lon(self._lat(self.dsyn, c)) / self.radius

    def wind(self, psi: torch.Tensor, chi: Optional[torch.Tensor] = None):
        """(u, v) = k x grad psi + grad chi:  u = -(1/a) dpsi/dphi + (1/(a cos phi)) dchi/dlam,
        v = (1/(a cos phi)) dpsi/dlam + (1/a) dchi/dphi. Coefficients (..., L, M) -> two fields (..., H, W)."""
        px, py = self.grad(psi)
        if chi is None:
            return -py, px
        cx, cy = self.grad(chi)
        return cx - py, px + cy

    def helmholtz(self, u: torch.Tensor, v: torch.Tensor):
        """The streamfunction and velocity potential (coefficients, l = 0 zero) of a wind (..., H, W): per order m
        the area-weighted least-squares fit of ``wind`` to (u, v); exact for the wind of a truncated (psi, chi)."""
        f = torch.cat([torch.fft.rfft(t.float(), dim=-1, norm="forward")[..., : self.L] for t in (u, v)], -2)
        z = torch.einsum("mkh,...hm->...km", self.hel, f) * self.radius       # (..., 2 L, M)
        return z[..., : self.L, :], z[..., self.L:, :]

    def vortdiv(self, u: torch.Tensor, v: torch.Tensor):
        """The vorticity lap psi and the divergence lap chi of a wind, on the grid (1/s)."""
        psi, chi = self.helmholtz(u, v)
        return self.synthesis(self.lap(psi)), self.synthesis(self.lap(chi))

    def linear_balance(self, psi: torch.Tensor, omega: float = OMEGA) -> torch.Tensor:
        """The geopotential of the linear balance lap Phi = div(f grad psi), f = 2 Omega sin(phi), l = 0 zero:
        Phi_n = 2 Omega [(n - 1) / n eps_n psi_(n-1) + (n + 2) / (n + 1) eps_(n+1) psi_(n+1)]   (every m),
        from x P_l and cos(phi) dP_l/dphi in the neighbouring degrees; the degree lmax + 1 is cut."""
        n = self.deg[:, None]
        lo = torch.roll(psi, 1, -2) * ((n - 1) / n.clamp(min=1) * self.eps[:-1])          # psi_(n-1); eps_n = 0 at n = m
        up = torch.roll(psi, -1, -2) * ((n + 2) / (n + 1) * self.eps[1:])
        up = torch.cat([up[..., :-1, :], torch.zeros_like(up[..., :1, :])], -2)           # psi_(lmax+1) = 0
        return 2 * omega * (lo + up) * (n > 0)


def bands(L: int, n: Optional[int] = None):
    """Littlewood-Paley bands of the degrees: j = 0 is l = 0, then 2^(j-1) <= l < 2^j, the last band up to L - 1.
    Returns the band index of every degree (L,) and the number of bands."""
    j = np.zeros(L, dtype=np.int64)
    for l in range(1, L):
        j[l] = int(math.floor(math.log2(l))) + 1
    if n is not None:
        j = np.minimum(j, n - 1)
    return torch.tensor(j), int(j.max()) + 1


__all__ = ["SHT", "bands", "A_EARTH", "OMEGA"]
