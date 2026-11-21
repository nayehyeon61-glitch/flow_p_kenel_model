"""Spherical harmonic transform on the lat-lon grid of the data, and fields built degree by degree.

The grid is equiangular (H latitudes, W = 2 H longitudes, no poles). The transform keeps the triangular truncation
l <= lmax = H - 1 and uses, per order m, the weighted least-squares inverse of the synthesis matrix: exact for every
field of degree <= lmax (the analysis of a synthesis is the identity), and the area-weighted best fit otherwise.

    SHT(lat).analysis(x)        (..., H, W)  -> complex (..., lmax + 1, lmax + 1)     [l, m], m >= 0
    SHT(lat).synthesis(c)       complex (..., L, L) -> (..., H, W)
    SHT(lat).spectrum(x)        (..., H, W)  -> (..., L)   the power of every degree l, summed over m; its sum is
                                                           the area mean of x^2 (Parseval on the unit sphere / 4 pi)
    SHT(lat).noise(amp, ...)    a Gaussian field with degree spectrum amp(l)^2 (isotropic on the sphere)

Normalization: Y_lm orthonormal over the sphere with measure dOmega / (4 pi), so a field's area mean of x^2 equals
sum_l sum_m |c_lm|^2 (with m > 0 counted twice for a real field).
"""

import math
from typing import Optional

import numpy as np
import torch


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
    def __init__(self, lat_deg, lmax: Optional[int] = None):
        super().__init__()
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

    def analysis(self, x: torch.Tensor) -> torch.Tensor:
        """(..., H, W) real -> (..., L, M) complex c[l, m]."""
        f = torch.fft.rfft(x.float(), dim=-1, norm="forward")[..., : self.L]  # (..., H, M): the mean over lon
        re = torch.einsum("mlh,...hm->...lm", self.ana, f.real)
        im = torch.einsum("mlh,...hm->...lm", self.ana, f.imag)
        return torch.complex(re, im)

    def synthesis(self, c: torch.Tensor) -> torch.Tensor:
        """(..., L, M) complex -> (..., H, W) real."""
        re = torch.einsum("mhl,...lm->...hm", self.syn, c.real)
        im = torch.einsum("mhl,...lm->...hm", self.syn, c.imag)
        f = torch.complex(re, im)
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
        z = torch.randn(tuple(shape) + (L, L, 2), generator=gen, device="cpu").to(amp)
        c = torch.complex(z[..., 0], z[..., 1]) / math.sqrt(2.0)
        c[..., 0] = torch.complex(z[..., 0, 0], torch.zeros_like(z[..., 0, 0]))  # m = 0 real
        tri = (torch.arange(L)[None, :] <= torch.arange(L)[:, None]).to(amp)    # m <= l
        # E|c_lm|^2 = 1 for every (l, m) present; degree power = 2 l + 1 -> scale to amp^2
        scale = amp / torch.sqrt(2 * self.deg + 1)
        c = c * tri * scale[..., :, None]
        return self.synthesis(c)


def bands(L: int, n: Optional[int] = None):
    """Littlewood-Paley bands of the degrees: j = 0 is l = 0, then 2^(j-1) <= l < 2^j, the last band up to L - 1.
    Returns the band index of every degree (L,) and the number of bands."""
    j = np.zeros(L, dtype=np.int64)
    for l in range(1, L):
        j[l] = int(math.floor(math.log2(l))) + 1
    if n is not None:
        j = np.minimum(j, n - 1)
    return torch.tensor(j), int(j.max()) + 1


__all__ = ["SHT", "bands"]
