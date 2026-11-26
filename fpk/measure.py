"""The coordinates of the measure (scripts/measure_core.py, scripts/plus_archive.py): the real spherical-harmonic
coefficients of degree l < lt of a field, the m = 0 imaginary parts left out (they are zero for a real field).

    to_vec   (D, C, H, W) anomalies -> (D, C, NV)
    to_field (B, NV) of one channel -> (B, H, W) on the grid of the latitudes (the contract's width: 2 (H - 1)
             longitudes with the poles, 2 H without), the field's projection on l < lt

The record of the training years in these coordinates (archive_vec.npy) is all the measure core needs of it: on the
1.5 deg grid the record's fields are 45 GB (f16), its coefficients l < 32 3 GB.
"""
import numpy as np
import torch

from .sphere import SHT


class MeasureCoords:
    def __init__(self, lat, lt: int):
        self.sht = sht = SHT(np.asarray(lat))
        if lt > sht.L:
            raise ValueError(f"lt {lt} above the grid's degrees ({sht.L})")
        self.lt, self.H, self.W = int(lt), sht.H, sht.W
        M = self.M = sht.ana.shape[0]
        keep = torch.zeros(sht.L, M, dtype=torch.bool)
        for l in range(lt):
            keep[l, : l + 1] = True
        im0 = torch.zeros(sht.L, M, dtype=torch.bool)
        im0[:, 0] = True
        self.KI = keep.flatten().nonzero()[:, 0]
        self.KIm = (keep & ~im0).flatten().nonzero()[:, 0]
        self.nv = len(self.KI) + len(self.KIm)

    def to_vec(self, f, chunk: int = 64) -> torch.Tensor:
        """(D, C, H, W) anomaly (numpy or torch, any float) -> (D, C, NV) float32."""
        out = []
        for i in range(0, len(f), chunk):
            x = f[i:i + chunk]
            x = x.float() if torch.is_tensor(x) else torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
            c = self.sht.analysis(x).flatten(-2)
            out.append(torch.cat([c.real[..., self.KI], c.imag[..., self.KIm]], -1))
        return torch.cat(out)

    def to_field(self, v: torch.Tensor) -> torch.Tensor:
        """(B, NV) of one channel -> (B, H, W)."""
        L, M = self.sht.L, self.M
        re, im = torch.zeros(len(v), L * M), torch.zeros(len(v), L * M)
        re[:, self.KI], im[:, self.KIm] = v[:, : len(self.KI)], v[:, len(self.KI):]
        return self.sht.synthesis(torch.complex(re, im).view(len(v), L, M))


__all__ = ["MeasureCoords"]
