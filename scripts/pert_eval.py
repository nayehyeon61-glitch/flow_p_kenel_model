"""Raw scores of the members-through-the-core means (diag_pert.py) against the deterministic cores on the same ICs.

    python scripts/pert_eval.py --year 2016 runs/fpk/cond_day05 runs/fpk/cond_day10 runs/fpk/cond_ic10

The deterministic fields come from the stride-1 dump (--src, diag_cond.py) at the same ICs. 'inf': the mean's mse
minus the sampling variance of the mean (antithetic pair means; the two cores' draws are independent).
"""
import argparse
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("dirs", nargs="+")
ap.add_argument("--year", type=int, default=2016)
ap.add_argument("--src", default="runs/fpk/cond")
ap.add_argument("--json", default="")
A = ap.parse_args()
Q = [0, 2, 4, 6, 9]
src = json.load(open(f"{A.src}/{A.year}_meta.json"))
ch = src["channels"]
std = np.asarray(src["std"])
lat = np.asarray(src["lat"])
w = np.cos(np.deg2rad(lat))
w = (w / w.mean())[:, None]
Z, T8, T2 = ch.index("z500"), ch.index("t850"), ch.index("t2m")
SC = [i for i in range(len(ch)) if ch[i] != "tisr"]
cores = src["cores"][:2]


def amse(f, o):
    return (((f - o) ** 2) * w).mean((-2, -1))                                    # (N, L, C)


def summary(m):
    r = lambda c: np.sqrt(m[:, :, c].mean(0)) * std[c]  # noqa: E731
    return dict(z500=r(Z)[Q].round(1).tolist(), t850=r(T8)[Q].round(3).tolist(), t2m=r(T2)[Q].round(3).tolist(),
                nrmse=np.sqrt(m.mean(0))[:, SC].mean(-1)[Q].round(4).tolist())


out = {}
for d in A.dirs:
    meta = json.load(open(f"{d}/{A.year}_meta.json"))
    pos = {k: i for i, k in enumerate(src["ics"])}
    idx = np.array([pos[k] for k in meta["ics"]])
    o = np.asarray(np.load(f"{d}/{A.year}_o.npy", mmap_mode="r"), dtype=np.float32)
    o_src = np.asarray(np.load(f"{A.src}/{A.year}_o.npy", mmap_mode="r")[idx], dtype=np.float32)
    assert np.abs(o - o_src).max() < 1e-2, "the ICs do not line up"
    sv = np.load(f"{d}/{A.year}_sv.npz")
    det = {t: np.asarray(np.load(f"{A.src}/{A.year}_{t}.npy", mmap_mode="r")[idx], dtype=np.float32) for t in cores}
    per = {t: np.asarray(np.load(f"{d}/{A.year}_{t}.npy", mmap_mode="r"), dtype=np.float32) for t in cores}
    res = {}
    for t in cores:
        res[f"det {t}"] = summary(amse(det[t], o))
        res[f"pert {t}"] = summary(amse(per[t], o))
        res[f"pert {t} inf"] = summary(amse(per[t], o) - sv[f"sv_{t}"])
    res["det mean"] = summary(amse((det[cores[0]] + det[cores[1]]) / 2, o))
    mp = amse((per[cores[0]] + per[cores[1]]) / 2, o)
    res["pert mean"] = summary(mp)
    res["pert mean inf"] = summary(mp - (sv[f"sv_{cores[0]}"] + sv[f"sv_{cores[1]}"]) / 4)
    sp = (sv[f"spread_{cores[0]}"] + sv[f"spread_{cores[1]}"]) / 2
    res["spread z500"] = (np.sqrt(sp[:, :, Z].mean(0)) * std[Z])[Q].round(1).tolist()
    res["sv share z500 (mean)"] = (((sv[f"sv_{cores[0]}"] + sv[f"sv_{cores[1]}"]) / 4)[:, :, Z].mean(0) / mp[:, :, Z].mean(0))[Q].round(4).tolist()
    out[d] = dict(meta={k: meta[k] for k in ("mode", "alpha", "members", "pool", "ic_stride")}, n=len(idx), scores=res)
    print(f"== {d} {out[d]['meta']} n={len(idx)}")
    for k, v in res.items():
        print(f"  {k:28s} {v}")
if A.json:
    json.dump(out, open(A.json, "w"), indent=1)
