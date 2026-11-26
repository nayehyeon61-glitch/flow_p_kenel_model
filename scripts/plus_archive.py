"""The record of the training years in the coordinates of the measure (the + version, 감자 2026-11-26: the 711 stack).

The measure core (scripts/measure_core.py --dump) estimates the transfer operator of the record: a linear inverse model
in the leading principal components of all 66 channels' coefficients l < --lt. On the 5.625 deg grid it read the
record's anomaly fields (archive_anom.npy, 1.6 GB); on the 1.5 deg grid with 32 years those are 45 GB, so the record is
streamed here, year by year, straight into its coefficients:

    <dir>/archive_vec.npy   (days, 66, NV) float32, the z-scored anomaly (truth - the daily climatology of stats.npz)
    <dir>/archive_meta.json years, days, lt, grid

The anomalies are those of scripts/diag_cond.py (the same climatology), so the record and the cores' fields are in the
same coordinates.

    python scripts/plus_archive.py --dir runs/fpk_plus/cond --lt 32 --data-root data/era5_1p5 --out runs/fpk_plus \\
        --train-range 1984 2015
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--lt", type=int, default=32)
ap.add_argument("--years", type=int, nargs="*", default=None, help="the record's years (default: the training years)")
ap.add_argument("--chunk", type=int, default=32)
A, PASS = ap.parse_known_args()

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_fpk as R  # noqa: E402
from fpk.env import DYN  # noqa: E402
from fpk.measure import MeasureCoords  # noqa: E402

G = {}


def grab(a, rest):
    G["a"] = a


R.cmd_score = grab                       # the options of run_fpk.py (data, folder, years, device) without loading a core
R.main(["score", "--arms", "core"] + PASS)
a = G["a"]
pa = R.pargs(a)
cl = R.D.climate(pa)
years = A.years or list(a.train_years)
dyn = np.asarray(DYN)
CZ = R.clim_z(cl)                                                                      # (366, 66, H, W) z-scored
MC = MeasureCoords(cl.lat, A.lt)
dev = torch.device(a.device)
MC.sht.to(dev)
os.makedirs(A.dir, exist_ok=True)
days = []
for y in years:                                                                        # the lengths first (memmaps: no read)
    days.append(len(R.D.load(pa, [y])[y]))
out = np.lib.format.open_memmap(f"{A.dir}/archive_vec.npy.part", mode="w+", dtype=np.float32, shape=(sum(days), len(dyn), MC.nv))
t0, s = time.time(), 0
for y, n in zip(years, days):
    yr = R.D.load(pa, [y])[y]
    for i in range(0, n, A.chunk):
        k = np.arange(i, min(i + A.chunk, n))
        z = np.asarray(yr[i:i + len(k)])[:, dyn] - CZ[np.minimum(k, 365)]
        out[s + i:s + i + len(k)] = MC.to_vec(torch.from_numpy(z).to(dev)).cpu().numpy()
    s += n
    print(json.dumps(dict(year=y, days=n, sec=round(time.time() - t0))), flush=True)
    del yr
out.flush()
del out
os.replace(f"{A.dir}/archive_vec.npy.part", f"{A.dir}/archive_vec.npy")
json.dump(dict(years=years, days=days, lt=A.lt, nv=MC.nv, grid=[MC.H, MC.W]), open(f"{A.dir}/archive_meta.json", "w"))
print("done", sum(days), MC.nv, round(time.time() - t0))
