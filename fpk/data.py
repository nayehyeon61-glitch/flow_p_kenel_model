"""Daily ERA5 on the KAI contract (67 channels, z-scored), through the vendored PG-CFM loader.

Everything is in the normalized units of PG-CFM's ``stats.npz`` (the same file the tendency core was trained
with). ``s`` = std of the daily increment per channel (``inc``) is the unit of every distribution here: a kernel
bandwidth of 1 means "one typical day of change".

The 5.625 deg years (0.2 GB each) are z-scored in memory. A year the vendored loader keeps on disk (np.memmap: the
1.5 deg grid, 2.85 GB a year, 91 GB for 32 years) stays there as a ``LazyYear``: indexing reads only the days asked
for and z-scores them. ``Pairs(..., device="cuda")`` reads the raw days of a sample in one slab and z-scores them on
the GPU, and ``prefetch`` draws the next batch in a thread while the model trains on the present one.
"""

import os
from concurrent.futures import ThreadPoolExecutor
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


class LazyYear:
    """A year of daily fields that stays on disk (np.memmap (D, 67, H, W), raw float32), read as z-scores: the index
    is days [, then indices of the (67, H, W) field], as of the in-memory array. float32 arithmetic (the in-memory
    path z-scores in float64; the two differ by ~1e-6 of a z-score). ``raw``, ``m``, ``s``: for ``Pairs`` on a GPU."""

    def __init__(self, raw, mean, std, eps):
        self.raw = raw
        self.m = np.asarray(mean, dtype=np.float32)[:, None, None]
        self.s = (np.asarray(std, dtype=np.float64) + eps).astype(np.float32)[:, None, None]

    def __len__(self):
        return len(self.raw)

    @property
    def shape(self):
        return tuple(self.raw.shape)

    def __getitem__(self, i):
        t, rest = (i[0], tuple(i[1:])) if isinstance(i, tuple) else (i, ())
        y = (np.asarray(self.raw[t], dtype=np.float32) - self.m) / self.s
        if rest:
            y = y[(slice(None),) * (y.ndim - 3) + rest]
        return y


def load(a, years) -> Dict[int, np.ndarray]:
    """year -> (D, 67, H, W) z-scored float32: in memory (the 5.625 deg grid is 0.2 GB a year), or a ``LazyYear``
    for a year the vendored loader keeps on disk (the 1.5 deg grid, or --lazy)."""
    R = rkt()
    raw = R.load_years(a, years)
    mean, std, _, _ = R.stats(a, raw if not os.path.exists(os.path.join(a.out, "stats.npz")) else None)
    return {y: LazyYear(raw[y], mean, std, R.EPS) if isinstance(raw[y], np.memmap) else R.zscore(np.asarray(raw[y]), mean, std)
            for y in years}


def climate(a) -> Climate:
    R = rkt()
    os.makedirs(a.out, exist_ok=True)
    if not os.path.exists(os.path.join(a.out, "stats.npz")):
        R.stats(a, R.load_years(a, a.train_years))
    mean, std, inc, clim = R.stats(a)
    y = a.val_year
    lat = R.lat_of(os.path.join(a.data, f"{y}.nc"), clim.shape[-2], a.era5)
    # a stats.npz of another grid (an --out of the 5.625 deg runs) would be used without a word: refuse it
    H, W = clim.shape[-2:]
    # the validation year's file (a score or a diagnosis reads it too); a folder without it is not checked here
    src, npy = os.path.join(a.data, f"{y}.nc"), os.path.join(a.cache, f"{y}.npy")
    first = R.daily(src, npy) if (os.path.exists(src) or os.path.exists(npy)) else None   # memmap: the shape only
    if first is not None and tuple(first.shape[-2:]) != (H, W):
        raise SystemExit(f"{a.out}/stats.npz is for a {H} x {W} grid, the data is {tuple(first.shape[-2:])}: "
                         f"use another --out for this grid")
    from .sphere import grid_lat
    if len(lat) != H or not np.allclose(lat, grid_lat(H, W), atol=1e-3):
        raise SystemExit(f"latitudes of the data ({lat[0]:.2f} .. {lat[-1]:.2f}, {len(lat)}) are not the contract's "
                         f"{H} x {W} grid, south to north")
    return Climate(mean, std, inc, clim, lat)


def area_weights(lat: np.ndarray) -> torch.Tensor:
    """(1, 1, H, 1) cos(lat) normalized to mean 1."""
    w = np.cos(np.deg2rad(lat))
    return torch.from_numpy((w / w.mean()).astype(np.float32)).view(1, 1, -1, 1)


class Pairs:
    """Random (x_t, x_{t+1}) day pairs from the z-scored years (no pair across a year boundary).

    Returns x (B, 67, H, W), y (B, 67, H, W), doy (B,) of x, tisr_next (B, H, W) = the forcing of y's day and, with
    ``hist`` = W > 0, the W days before x (B, W, 67, H, W), oldest first (clamped at the start of the year, as
    PG-CFM does), else None. ``ahead`` = K > 1: y (B, K, 67, H, W) the K days after x, tisr_next (B, K, H, W)."""

    def __init__(self, years: Dict[int, np.ndarray], seed: int = 0, hist: int = 0, ahead: int = 1, device=None,
                 prefetch: bool = False):
        """``device`` (lazy years): x, y and hist come back on the device, z-scored there from the raw days of one
        read per sample (the same float32 numbers as indexing the ``LazyYear``). ``prefetch``: the next batch of the
        same size is drawn in a thread meanwhile. The batches are those without it, in the same order: a call with
        another size first undoes the draws of the batch drawn ahead, and ``state`` is the generator's state where
        the next batch to be returned begins (``set_state`` puts it back for a continued run)."""
        self.years, self.hist, self.ahead = years, int(hist), int(ahead)
        self.keys = sorted(years)
        self.n = np.array([len(years[k]) - self.ahead for k in self.keys])
        self.rng = np.random.default_rng(seed)
        self.dev = None
        if device is not None and all(isinstance(v, LazyYear) for v in years.values()):
            self.dev = torch.device(device)
            v = years[self.keys[0]]
            self.m = torch.from_numpy(v.m).to(self.dev)
            self.s = torch.from_numpy(v.s).to(self.dev)
        self._pool = ThreadPoolExecutor(1) if prefetch else None
        self._next = None

    def __call__(self, B: int):
        if self._pool is None:
            return self.draw(B)
        if self._next is not None and self._next[0] == B:
            out = self._next[1].result()
        else:
            self._drop()                      # another size: the batch drawn ahead is undone, drawn now at this size
            out = self.draw(B)
        # the worker is idle here: the state where the next batch begins, read before it is drawn
        self._next = (B, self._pool.submit(self.draw, B), self.rng.bit_generator.state)
        return out

    def _drop(self):
        if self._next is not None:
            self._next[1].result()
            self.rng.bit_generator.state = self._next[2]
            self._next = None

    def state(self):
        """The generator's state where the next batch to be returned begins (a batch drawn ahead is not counted)."""
        return self._next[2] if self._next is not None else self.rng.bit_generator.state

    def set_state(self, st):
        self._drop()
        self.rng.bit_generator.state = st

    def draw(self, B: int):
        ys = self.rng.choice(len(self.keys), size=B, p=self.n / self.n.sum())
        x, y, d, h = [], [], [], []
        K, W = self.ahead, self.hist
        for k in ys:
            arr = self.years[self.keys[k]]
            i = int(self.rng.integers(0, len(arr) - K))
            if self.dev is not None:
                # one read of the sample's days [i - W, i + K] from the disk, raw
                lo = max(i - W, 0)
                arr = np.asarray(arr.raw[lo:i + 1 + K])
                i0 = i - lo
                x.append(arr[i0])
                y.append(arr[i0 + 1] if K == 1 else arr[i0 + 1:i0 + 1 + K])
                if W:
                    h.append(arr[np.maximum(np.arange(i - W, i), 0) - lo])
            else:
                x.append(arr[i])
                y.append(arr[i + 1] if K == 1 else arr[i + 1:i + 1 + K])
                if W:
                    h.append(arr[np.maximum(np.arange(i - W, i), 0)])
            d.append(i)
        x, y = torch.from_numpy(np.stack(x)), torch.from_numpy(np.stack(y))
        hist = torch.from_numpy(np.stack(h)) if W else None
        if self.dev is not None:
            z = lambda t: None if t is None else (t.to(self.dev) - self.m) / self.s  # noqa: E731
            x, y, hist = z(x), z(y), z(hist)
        return x, y, torch.tensor(d), y[..., TISR, :, :], hist


__all__ = ["Climate", "LazyYear", "load", "climate", "area_weights", "Pairs", "DYN", "OROG", "TISR"]
