"""Daily ERA5 on the KAI contract (67 channels, z-scored), through the vendored PG-CFM loader.

Everything is in the normalized units of PG-CFM's ``stats.npz`` (the same file the tendency core was trained
with). ``s`` = std of the daily increment per channel (``inc``) is the unit of every distribution here: a kernel
bandwidth of 1 means "one typical day of change".
"""

import os
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch

from .env import DYN, OROG, TISR, rkt


@dataclass
class Climate:
    mean: np.ndarray   # (67,)
    std: np.ndarray    # (67,)
    inc: np.ndarray    # (67,) std of the daily increment, normalized units
    clim: np.ndarray   # (366, 67, H, W) daily climatology, physical units
    lat: np.ndarray    # (H,)


def load(a, years) -> Dict[int, np.ndarray]:
    """year -> (D, 67, H, W) z-scored float32 (in memory: the 5.625 deg grid is 0.2 GB a year)."""
    R = rkt()
    raw = R.load_years(a, years)
    mean, std, _, _ = R.stats(a, raw if not os.path.exists(os.path.join(a.out, "stats.npz")) else None)
    return {y: R.zscore(np.asarray(raw[y]), mean, std) for y in years}


def climate(a) -> Climate:
    R = rkt()
    os.makedirs(a.out, exist_ok=True)
    if not os.path.exists(os.path.join(a.out, "stats.npz")):
        R.stats(a, R.load_years(a, a.train_years))
    mean, std, inc, clim = R.stats(a)
    y = a.val_year
    lat = R.lat_of(os.path.join(a.data, f"{y}.nc"), clim.shape[-2], a.era5)
    return Climate(mean, std, inc, clim, lat)


def area_weights(lat: np.ndarray) -> torch.Tensor:
    """(1, 1, H, 1) cos(lat) normalized to mean 1."""
    w = np.cos(np.deg2rad(lat))
    return torch.from_numpy((w / w.mean()).astype(np.float32)).view(1, 1, -1, 1)


class Pairs:
    """Random (x_t, x_{t+1}) day pairs from the z-scored years (no pair across a year boundary).

    Returns x (B, 67, H, W), y (B, 67, H, W), doy (B,) of x, tisr_next (B, H, W) = the forcing of y's day."""

    def __init__(self, years: Dict[int, np.ndarray], seed: int = 0):
        self.years = years
        self.keys = sorted(years)
        self.n = np.array([len(years[k]) - 1 for k in self.keys])
        self.rng = np.random.default_rng(seed)

    def __call__(self, B: int):
        ys = self.rng.choice(len(self.keys), size=B, p=self.n / self.n.sum())
        x, y, d = [], [], []
        for k in ys:
            arr = self.years[self.keys[k]]
            i = int(self.rng.integers(0, len(arr) - 1))
            x.append(arr[i])
            y.append(arr[i + 1])
            d.append(i)
        x, y = torch.from_numpy(np.stack(x)), torch.from_numpy(np.stack(y))
        return x, y, torch.tensor(d), y[:, TISR]


__all__ = ["Climate", "load", "climate", "area_weights", "Pairs", "DYN", "OROG", "TISR"]
