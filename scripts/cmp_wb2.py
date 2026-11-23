"""NeuralGCM (WeatherBench2 public forecasts) against the fpk models on the same 2020 ICs, in ERA5 daily means.

fpk forecasts the ERA5 daily mean (the mean of 00/06/12/18 UTC, scripts wb2_to_kai --daily-mean). NeuralGCM's
public 2020 forecasts (WB2, regridded conservatively to the same 64x32 grid) start at 00 and 12 UTC and are stored
every 12 h, so for fpk's IC day k (whose daily mean already holds 18 UTC of day k) NeuralGCM starts at 12 UTC of
day k (6 h less information than fpk), and its daily mean of day k + j is the mean of its 00 and 12 UTC of that
day (leads 24 j - 12 and 24 j h). The truth is fpk's own: the 4-time ERA5 daily mean. The 2-time mean differs from
the 4-time mean by itself; ``era5`` measures that floor (the 2-time ERA5 mean scored as a forecast of the
4-time mean) so it can be read off.

    python scripts/cmp_wb2.py fetch --model det          # ~5 min
    python scripts/cmp_wb2.py fetch --model ens          # ~30 min (50 members)
    python scripts/cmp_wb2.py fetch --model era5         # 00/06/12/18 UTC of 2020, 8 variables
    python scripts/cmp_wb2.py score --fpk runs/fpk/score2020_raw.npz --json docs/results/cmp_neuralgcm_2020.json

Scores per variable and lead with the definitions of fpk.ensemble.Scorer (latitude-weighted MSE of the ensemble
mean, fair CRPS, spread / skill with sqrt((M + 1) / M), ACC of anomalies to the fpk daily climatology), so the
numbers sit beside ``run_fpk.py score``'s.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, R)

# the variables NeuralGCM and fpk share (fpk.ensemble.KEY_VARS without t2m and msl)
SHARED = [("z500", "geopotential", 500), ("z850", "geopotential", 850), ("t850", "temperature", 850),
          ("u850", "u_component_of_wind", 850), ("v850", "v_component_of_wind", 850),
          ("u250", "u_component_of_wind", 250), ("v250", "v_component_of_wind", 250),
          ("q700", "specific_humidity", 700)]
NAMES = [s[0] for s in SHARED]
URL = {"det": "gs://weatherbench2/datasets/neuralgcm_deterministic/2020-64x32_equiangular_conservative.zarr",
       "ens": "gs://weatherbench2/datasets/neuralgcm_ens/2020-64x32_equiangular_conservative.zarr",
       "era5": "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"}
YEAR, LEADS, DAYS = 2020, 10, 366


def ics_of(stride=6, start=0, n_days=DAYS, leads=LEADS):
    """fpk score's ICs: day indices of the test year (run_fpk.py score: arange(ic_start, len(te) - L, stride))."""
    return np.arange(start, n_days - leads, stride)


def open_zarr(model):
    import logging
    import xarray as xr
    logging.getLogger("gcsfs").setLevel(logging.ERROR)
    return xr.open_zarr(URL[model], chunks=None)


def fetch_ic(ds, model, k, out):
    """NeuralGCM from 12 UTC of day k: daily means of days k+1..k+10 (members, 10, 8, lat, lon) and the start
    state (8, lat, lon), float32, physical units, latitude S -> N."""
    path = os.path.join(out, f"ic{k:03d}.npz")
    if os.path.exists(path):
        return path
    ti = 2 * k + 1                       # 2020-01-01T00 + k days + 12 h
    t = np.datetime64(f"{YEAR}-01-01T12") + np.timedelta64(int(k), "D")
    assert ds.time.values[ti] == t, (ds.time.values[ti], t)
    td = ds.prediction_timedelta.values[:2 * LEADS + 1]
    lead_h = (td / np.timedelta64(1, "h") if np.issubdtype(td.dtype, np.timedelta64) else td).astype(int)  # ens: hours
    assert (lead_h == 12 * np.arange(2 * LEADS + 1)).all(), lead_h
    chans = []
    for name, var, lev in SHARED:
        x = ds[var].isel(time=ti, prediction_timedelta=slice(0, 2 * LEADS + 1)).sel(level=lev)
        x = x.transpose(*(["realization"] if model == "ens" else []), "prediction_timedelta", "latitude",
                        "longitude").values.astype(np.float32)
        chans.append(x)
    x = np.stack(chans, -3)              # ([M,] 21, 8, lat, lon)
    if model == "det":
        x = x[None]
    daily = 0.5 * (x[:, 1:2 * LEADS:2] + x[:, 2:2 * LEADS + 1:2])   # leads (24 j - 12, 24 j) -> day k + j
    np.savez(path + ".tmp.npz", daily=daily, start=x[0, 0], k=k)
    os.replace(path + ".tmp.npz", path)
    return path


def cmd_fetch(a):
    out = os.path.join(a.data, a.model)
    os.makedirs(out, exist_ok=True)
    ds = open_zarr(a.model)
    if a.model == "era5":
        sl = ds.sel(time=slice(f"{YEAR}-01-01T00", f"{YEAR}-12-31T18"))
        assert len(sl.time) == 4 * DAYS, len(sl.time)
        x = np.stack([sl[var].sel(level=lev).transpose("time", "latitude", "longitude").values.astype(np.float32)
                      for _, var, lev in SHARED], 1)                                  # (4 D, 8, lat, lon)
        x = x.reshape(DAYS, 4, *x.shape[1:])                                          # 00 06 12 18 UTC
        np.savez(os.path.join(out, "era5_2020.npz"), mean4=x.mean(1), mean2=0.5 * (x[:, 0] + x[:, 2]),
                 at12=x[:, 2])
        print("era5 done", x.shape, flush=True)
        return
    ics = ics_of(a.ic_stride)[: a.max_ics or None]
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        for i, p in enumerate(ex.map(lambda k: fetch_ic(ds, a.model, int(k), out), ics)):
            print(f"{i + 1}/{len(ics)} {os.path.basename(p)} {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------------------------------------- score
ALLV = ["z500", "z850", "t850", "t2m", "msl", "u850", "v850", "u250", "v250", "q700"]   # fpk.ensemble.KEY_VARS
CLIMODE_V = ["z500", "t850", "t2m"]                                                        # ClimODE's z, t, t2m


def fpk_setup(test_year=YEAR):
    """fpk's normalization, climatology and truth (run_fpk.py score's own)."""
    from fpk import data as D
    from fpk.env import pgfm_args
    # run_fpk.py's common defaults (stats.npz of runs/fpk holds the normalization and the climatology)
    pa = pgfm_args(os.path.join(R, "data", "era5_5p625"), os.path.join(R, "runs", "fpk"), [2014, 2015], 2016,
                   test_year, "cpu", "core_gngmz8W")
    cl = D.climate(pa)
    te = D.load(pa, [test_year])[test_year]
    return pa, cl, te


def gauss_crps(mu, sd, y):
    from scipy.stats import norm
    sd = np.maximum(sd, 1e-12)
    z_ = (y - mu) / sd
    return sd * (z_ * (2 * norm.cdf(z_) - 1) + 2 * norm.pdf(z_) - 1 / np.sqrt(np.pi))


def scores(fc, truth, clim, w, sd=None):
    """fc (N, M, L, C, H, W), truth / clim (N, L, C, H, W), z-scored. Per-IC arrays (N, L, C) as Scorer.update:
    mse of the ensemble mean, fair CRPS (|error| for one member, Gaussian with ``sd`` (N, L, C, H, W) if given),
    ensemble variance (ddof 1), ACC of the anomalies."""
    from fpk.ensemble import area_mean, fair_crps
    M = fc.shape[1]
    mu = fc.mean(1)
    r = {"mse": area_mean((mu - truth) ** 2, w)}
    pa, ta = mu - clim, truth - clim
    r["acc"] = area_mean(pa * ta, w) / np.sqrt(area_mean(pa ** 2, w) * area_mean(ta ** 2, w) + 1e-30)
    if M > 1:
        r["var"] = area_mean(fc.var(1, ddof=1), w)
        r["crps"] = np.stack([area_mean(fair_crps(fc[:, :, l], truth[:, l]), w) for l in range(fc.shape[2])], 1)
    elif sd is not None:
        r["crps"] = area_mean(gauss_crps(mu, sd, truth), w)
    else:
        r["crps"] = area_mean(np.abs(mu - truth), w)
    return r


def summary(r, names, std, M, pairs=False):
    """Per-variable lead curves in physical units, as Scorer.summary. ``r`` arrays (N, L, C) over ``names``."""
    rmse = np.sqrt(r["mse"].mean(0))                                           # (L, C)
    out = {"n_ic": int(r["mse"].shape[0]), "members": int(M),
           "rmse_phys": {v: (rmse[:, i] * std[v]).tolist() for i, v in enumerate(names)},
           "crps_vars": {v: (r["crps"].mean(0)[:, i] * std[v]).tolist() for i, v in enumerate(names)},
           "acc_vars": {v: r["acc"].mean(0)[:, i].tolist() for i, v in enumerate(names)}}
    if "var" in r:
        sp = np.sqrt(r["var"].mean(0))
        c = 1.0 if pairs else np.sqrt((M + 1) / M)
        out["spread_skill_vars"] = {v: (c * sp[:, i] / rmse[:, i]).tolist() for i, v in enumerate(names)}
    return out


def paired(rA, rB, std, names, leads=(0, 2, 4, 6, 9), n_boot=2000, seed=0):
    """RMSE(A) - RMSE(B) per variable and lead on the same ICs (N, L, C mse), 95% bootstrap interval over ICs."""
    g = np.random.default_rng(seed)
    N = rA.shape[0]
    idx = g.integers(0, N, (n_boot, N))
    out = {}
    for i, v in enumerate(names):
        d = {}
        for l in leads:
            a_, b_ = rA[:, l, i], rB[:, l, i]
            diff = (np.sqrt(a_.mean()) - np.sqrt(b_.mean())) * std[v]
            bs = (np.sqrt(a_[idx].mean(1)) - np.sqrt(b_[idx].mean(1))) * std[v]
            d[str(l + 1)] = [float(diff), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        out[v] = d
    return out


def cmd_score(a):
    from fpk.env import contract, rkt
    from fpk.ensemble import dyn_index
    pa, cl, te = fpk_setup()
    eps = rkt().EPS
    std = {v: float(cl.std[contract.CHANNEL_INDEX[v]]) for v in ALLV}
    w = np.cos(np.deg2rad(cl.lat))
    w = (w / w.mean()).astype(np.float64)
    ics = ics_of(a.ic_stride)
    pos = {int(k): n for n, k in enumerate(ics)}

    def chans(names):
        return np.array([contract.CHANNEL_INDEX[v] for v in names])

    def zf(x, names):  # physical (..., C, H, W) -> z-scored
        c = chans(names)
        return (x - cl.mean[c][:, None, None]) / (cl.std[c][:, None, None] + eps)

    def truth_of(names, days):
        return np.stack([te[k + 1:k + 1 + LEADS][:, chans(names)] for k in days]).astype(np.float64)

    def clim_of(names, days):
        c = chans(names)
        cz = (cl.clim[:, c] - cl.mean[c][None, :, None, None]) / (cl.std[c][None, :, None, None] + eps)
        return np.stack([cz[np.minimum(np.arange(k + 1, k + 1 + LEADS), 365)] for k in days]).astype(np.float64)

    res, raws, sets = {}, {}, {}                     # sets: model -> (names, ic days)
    T8, C8 = truth_of(NAMES, ics), clim_of(NAMES, ics)
    # ERA5 checks: the 4-time mean of the fetched 6-hourly ERA5 is fpk's truth; the 2-time (00 + 12 UTC) mean scored
    # as a forecast of the 4-time mean is the floor of any 2-time daily mean (NeuralGCM's)
    p5 = os.path.join(a.data, "era5", "era5_2020.npz")
    T8_2 = None
    if os.path.exists(p5):
        era = np.load(p5)
        m4 = np.stack([zf(era["mean4"][k + 1:k + 1 + LEADS], NAMES) for k in ics])
        res["check_era5_mean4_vs_fpk_truth_rmse_phys"] = {
            v: float(np.sqrt(((m4 - T8)[:, :, i] ** 2).mean()) * std[v]) for i, v in enumerate(NAMES)}
        T8_2 = np.stack([zf(era["mean2"][k + 1:k + 1 + LEADS], NAMES) for k in ics])
        r = scores(T8_2[:, None], T8, C8, w)
        res["era5_2time_mean"], raws["era5_2time_mean"], sets["era5_2time_mean"] = summary(r, NAMES, std, 1), r, (NAMES, ics)
    # NeuralGCM: deterministic and the 50-member ensemble (and its first 8 members, fpk's ensemble size)
    for model in ("det", "ens"):
        files = [os.path.join(a.data, model, f"ic{k:03d}.npz") for k in ics]
        if not all(os.path.exists(f) for f in files):
            print(model, "missing ICs:", sum(not os.path.exists(f) for f in files), flush=True)
            continue
        mems = [None] if model == "det" else [8, None]
        acc, acc2, st = {m: [] for m in mems}, [], []
        for n, f in enumerate(files):
            x = np.load(f)
            st.append(zf(x["start"], NAMES))
            fc = zf(x["daily"].astype(np.float64), NAMES)                            # (M, L, 8, H, W)
            for m in mems:
                acc[m].append(scores(fc[None, :m], T8[[n]], C8[[n]], w))
            if model == "det" and T8_2 is not None:                                 # against the 2-time truth
                acc2.append(scores(fc[None], T8_2[[n]], C8[[n]], w))
        if model == "det":
            st = np.stack(st)
            ic_t = np.stack([te[k][chans(NAMES)] for k in ics]).astype(np.float64)
            res["check_start_vs_ic_daymean_corr_z500"] = float(np.corrcoef(st[:, 0].ravel(), ic_t[:, 0].ravel())[0, 1])
            if acc2:
                r = {q: np.concatenate([u[q] for u in acc2], 0) for q in acc2[0]}
                res["neuralgcm_det_vs_2time_truth"] = summary(r, NAMES, std, 1)
        for m in mems:
            r = {q: np.concatenate([u[q] for u in acc[m]], 0) for q in acc[m][0]}
            M = 1 if model == "det" else (m or 50)
            name = "neuralgcm_det" if model == "det" else f"neuralgcm_ens{M}"
            res[name], raws[name], sets[name] = summary(r, NAMES, std, M), r, (NAMES, ics)
            print(name, "z500 RMSE", [round(v) for v in res[name]["rmse_phys"]["z500"]], flush=True)
    # ClimODE (its own subset of the ICs)
    dc = os.path.join(a.data, "climode")
    cfiles = sorted(f for f in os.listdir(dc) if f.startswith("ic") and f.endswith(".npz")) if os.path.isdir(dc) else []
    if cfiles:
        days = np.array([int(f[2:5]) for f in cfiles])
        assert all(int(k) in pos for k in days), days
        sel = [0, 1, 2]                                                             # z, t, t2m of ClimODE's 5
        fc, sd = [], []
        for f in cfiles:
            x = np.load(os.path.join(dc, f))
            fc.append(zf(x["daily"][:, sel].astype(np.float64), CLIMODE_V))
            sd.append(x["daily_std"][:, sel] / (cl.std[chans(CLIMODE_V)][:, None, None] + eps))
        fc, sd = np.stack(fc), np.stack(sd).astype(np.float64)
        ok = ~np.isnan(fc).any((1, 2, 3, 4))
        res["climode_nan_ics"] = days[~ok].tolist()
        days, fc, sd = days[ok], fc[ok], sd[ok]
        Tc, Cc = truth_of(CLIMODE_V, days), clim_of(CLIMODE_V, days)
        r = scores(fc[:, None], Tc, Cc, w, sd=sd)
        res["climode"], raws["climode"], sets["climode"] = summary(r, CLIMODE_V, std, 1), r, (CLIMODE_V, days)
        print("climode", len(days), "ICs, z500 RMSE", [round(v) for v in res["climode"]["rmse_phys"]["z500"]], flush=True)
    # the fpk arms of run_fpk.py score --test-year 2020 on the same ICs (raw per-IC arrays over the DYN channels)
    for fp in a.fpk:
        z_ = np.load(fp)
        assert np.array_equal(z_["ics"], ics), (z_["ics"], ics)
        sc = json.load(open(fp.replace("_raw.npz", ".json")))["scores"]
        for arm in sorted({k.split("__")[0] for k in z_.files if "__" in k}):
            dch = np.array([dyn_index(v) for v in ALLV])
            r = {q: z_[f"{arm}__{q}"][:, :, dch].astype(np.float64) for q in ("mse", "crps", "acc", "var")
                 if f"{arm}__{q}" in z_.files}
            M = int(sc.get(arm, {}).get("members", 1))
            res[arm] = summary(r, ALLV, std, M, pairs=arm.startswith("devk"))
            raws[arm], sets[arm] = r, (ALLV, ics)
    # paired differences on the shared variables and ICs, fpk arm minus each baseline
    pairs_out = {}
    for A in [k for k in sets if k.startswith(("dev", "core", "persistence"))]:
        for B in [k for k in sets if k.startswith(("neuralgcm", "climode"))]:
            nB, dB = sets[B]
            names = [v for v in nB if v in sets[A][0]]
            iA = [pos[int(k)] for k in dB]
            rA = raws[A]["mse"][iA][:, :, [sets[A][0].index(v) for v in names]]
            rB = raws[B]["mse"][:, :, [nB.index(v) for v in names]]
            pairs_out[f"{A} - {B}"] = paired(rA, rB, std, names)
    res["paired_rmse_diff_day"] = pairs_out
    res["meta"] = dict(year=YEAR, ics=ics.tolist(), leads_days=LEADS,
                       climode_ics=sets["climode"][1].tolist() if "climode" in sets else [],
                       neuralgcm_start="12 UTC of the IC day", neuralgcm_daily_mean="mean of 00 and 12 UTC",
                       climode_start="18 UTC of the IC day", climode_daily_mean="mean of 00/06/12/18 UTC",
                       truth="ERA5 daily mean of 00/06/12/18 UTC (fpk kaiday)")
    os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
    with open(a.json, "w") as f:
        json.dump(res, f, indent=1)
    np.savez_compressed(a.json.replace(".json", "_raw.npz"),
                        **{f"{k}__{q}": v.astype(np.float32) for k, r in raws.items() for q, v in r.items()},
                        ics=ics)
    print("wrote", a.json, flush=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--model", choices=["det", "ens", "era5"], required=True)
    f.add_argument("--workers", type=int, default=6)
    f.add_argument("--max-ics", type=int, default=0)
    s = sub.add_parser("score")
    s.add_argument("--fpk", nargs="*", default=[], help="run_fpk.py score --test-year 2020 raw npz files")
    s.add_argument("--json", default=os.path.join(R, "docs", "results", "cmp_neuralgcm_2020.json"))
    for q in (f, s):
        q.add_argument("--data", default=os.path.join(R, "data", "wb2_2020"))
        q.add_argument("--ic-stride", type=int, default=6)
    a = p.parse_args(argv)
    if a.cmd == "fetch":
        cmd_fetch(a)
    else:
        cmd_score(a)


if __name__ == "__main__":
    main()
