"""The mathematical cores on their own: RMSE per lead and the correlation of their errors with the cores' (2017).

    python scripts/math_eval.py kaf lim
"""
import argparse
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("tags", nargs="+")
ap.add_argument("--dir", default="runs/fpk/cond")
ap.add_argument("--year", type=int, default=2017)
ap.add_argument("--json", default="")
A = ap.parse_args()
m = json.load(open(f"{A.dir}/{A.year}_meta.json"))
ch, std = m["channels"], np.asarray(m["std"])
w = np.cos(np.deg2rad(np.asarray(m["lat"])))
w = (w / w.mean())[:, None]
Q = [0, 2, 4, 6, 9]
o = np.load(f"{A.dir}/{A.year}_o.npy", mmap_mode="r")
cores = ["core_gngmz8W", "core_ngmz8W"]
out = {}
for c in ("z500", "t850", "t2m"):
    i = ch.index(c)
    oo = np.asarray(o[:, :, i], dtype=np.float32)
    F = {t: np.asarray(np.load(f"{A.dir}/{A.year}_{t}.npy", mmap_mode="r")[:, :, i], dtype=np.float32) for t in cores + A.tags}
    F["core mean"] = (F[cores[0]] + F[cores[1]]) / 2
    E = {t: f - oo for t, f in F.items()}
    r = {}
    for t in ["core mean", *cores, *A.tags]:
        rm = np.sqrt((E[t] ** 2 * w).mean((-2, -1)).mean(0)) * std[i]
        r[t] = dict(rmse=rm[Q].round(3 if c != "z500" else 1).tolist())
        if t in A.tags:
            cc = (E[t] * E["core mean"] * w).sum((0, 2, 3)) / np.sqrt((E[t] ** 2 * w).sum((0, 2, 3)) * (E["core mean"] ** 2 * w).sum((0, 2, 3)))
            r[t]["err_corr_with_core_mean"] = cc[Q].round(3).tolist()
    clim = np.sqrt((oo ** 2 * w).mean((-2, -1)).mean(0)) * std[i]
    r["climatology"] = dict(rmse=clim[Q].round(3 if c != "z500" else 1).tolist())
    out[c] = r
    print(c)
    for t, v in r.items():
        print(f"  {t:14s} {v}")
if A.json:
    json.dump(out, open(A.json, "w"), indent=1)
