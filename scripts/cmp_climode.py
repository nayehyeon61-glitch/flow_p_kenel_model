"""ClimODE's public checkpoint on fpk's 2020 ICs, as ERA5 daily means (the comparison of scripts/cmp_wb2.py).

ClimODE (Verma et al., ICLR 2024) integrates 6-hourly fields of z500, t850, t2m, u10, v10. fpk's IC day k is the
daily mean of 00/06/12/18 UTC, so ClimODE starts at 18 UTC of day k (the same information) and its daily mean of
day k + j is the mean of its steps 4 j - 3 .. 4 j (00/06/12/18 UTC of that day), exactly the 4-time mean of the
truth. The run follows the earlier project's run_climode_2020.py (physical_guide_flowmatching): the authors' code
(vendored in third_party/pgfm/third_party/climode), min-max normalization over 2006-2018, the test velocity fitted
with upstream fit_velocity, and 2020 batched with 2018 at the same time index (upstream never calls model.eval(),
so its batch norm sees the two years of the batch); only 2020 is kept. 2020 keeps Feb 29 (time index = 6 h steps
from Jan 1, as the earlier run).

    python scripts/cmp_climode.py --every 5        # 12 of the 60 ICs, spread over the year (~15 min each on CPU)

Writes <out>/ic<k>.npz: daily (10, 5, H, W) mean, daily_std (10, 5, H, W) the mean of the 4 step stds (the
fully correlated bound), steps (40, 5, H, W) the 6-hourly mean, physical units, latitude S -> N.
"""

import argparse
import os
import sys
import time
import urllib.request

import numpy as np
import torch

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CLIMODE = os.path.join(R, "third_party", "pgfm", "third_party", "climode")
CKPT_URL = "https://raw.githubusercontent.com/Aalto-QuML/ClimODE/e729d23e8799ce0e075699e76d60227d848d8d0c/checkpoints/ClimODE_global.pt"
LEVELS = ["z", "t", "t2m", "u10", "v10"]
FOLDERS = ["geopotential_500", "temperature_850", "2m_temperature", "10m_u_component_of_wind",
           "10m_v_component_of_wind"]
FPK_NAMES = ["z500", "t850", "t2m", None, None]    # fpk channels of the ClimODE variables (u10 / v10: none)
YEAR, PARTNER, STEPS = 2020, 2018, 1460


def year_fields(E, y):
    import xarray as xr
    out, lat, lon = [], None, None
    for lev, f in zip(LEVELS, FOLDERS):
        with xr.open_dataset(os.path.join(E, f, f"{f}_{y}_5.625deg.nc")) as d:
            out.append(d[lev].values.astype(np.float32))
            lat, lon = d.lat.values, d.lon.values
    return np.stack(out, 1), lat, lon                       # (T, 5, H, W)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(R, "data", "era5_5p625", "era5_data"))
    p.add_argument("--out", default=os.path.join(R, "data", "wb2_2020", "climode"))
    p.add_argument("--ic-stride", type=int, default=6, help="fpk score's IC stride (days)")
    p.add_argument("--every", type=int, default=5, help="every n-th of fpk's ICs")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--max-ics", type=int, default=0)
    p.add_argument("--lead-days", type=int, default=10)
    p.add_argument("--solver", default="euler")
    p.add_argument("--threads", type=int, default=os.cpu_count())
    a = p.parse_args(argv)
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    sys.path.insert(0, CLIMODE)
    import model_function  # noqa: F401  (classes of the pickled checkpoint)
    from model_function import Optim_velocity
    import utils as cu
    import xarray as xr
    # upstream opens the one constants file with open_mfdataset (needs dask, not installed here): same values
    cu.xr.open_mfdataset = lambda paths, **kw: xr.open_dataset(paths[0] if isinstance(paths, list) else paths)
    torch.set_num_threads(a.threads)
    torch.backends.mkldnn.enabled = False  # 2x faster for these 128-channel 32x64 convs on CPU
    ckpt = os.path.join(out, "ClimODE_global.pt")
    if not os.path.exists(ckpt):
        urllib.request.urlretrieve(CKPT_URL, ckpt + ".tmp")
        os.replace(ckpt + ".tmp", ckpt)

    # min-max over 2006-2018 (upstream get_train_test_data_without_scales_batched)
    mm = os.path.join(out, "minmax.npz")
    if not os.path.exists(mm):
        lo, hi = np.full(5, np.inf), np.full(5, -np.inf)
        for y in range(2006, 2019):
            x, _, _ = year_fields(a.data, y)
            lo, hi = np.minimum(lo, x.min((0, 2, 3))), np.maximum(hi, x.max((0, 2, 3)))
        np.savez(mm, mx=hi, mn=lo)
    mx, mn = np.load(mm)["mx"], np.load(mm)["mn"]
    scale = mx - mn
    arrs = []
    for y in (YEAR, PARTNER):
        x, lat, lon = year_fields(a.data, y)
        arrs.append((x[:STEPS] - mn[None, :, None, None]) / scale[None, :, None, None])
    test = torch.from_numpy(np.stack(arrs, 1).astype(np.float32))          # (1460, 2 years, 5, H, W)
    time_steps = torch.arange(STEPS).view(-1, 1)
    const, lat_map, lon_map = cu.add_constant_info([os.path.join(a.data, "constants", "constants_5.625deg.nc")])
    H, W = test.shape[3], test.shape[4]
    L = 4 * a.lead_days

    days = np.arange(0, 366 - a.lead_days, a.ic_stride)[a.offset::a.every]
    if a.max_ics:
        days = days[:a.max_ics]
    model = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.method = a.solver
    cwd = os.getcwd()
    os.chdir(out)
    try:
        if not os.path.exists("kernel.npy"):
            cu.get_gauss_kernel((H, W), lat, lon)
        kernel = torch.from_numpy(np.load("kernel.npy"))
        t0 = time.time()
        for n, k in enumerate(days):
            path = os.path.join(out, f"ic{k:03d}.npz")
            if os.path.exists(path):
                continue
            ic = 4 * int(k) + 3                                            # 18 UTC of day k
            assert ic + L < STEPS
            name = f"vel_ic{k:03d}"
            if not os.path.exists(name + "_vel.npy"):
                sel = torch.tensor([ic])
                dl = lambda t: torch.utils.data.DataLoader(t, batch_size=1, shuffle=False)  # noqa: E731
                cu.fit_velocity(dl(torch.arange(STEPS).view(-1, 1)[sel]), dl(time_steps[sel]), test, dl(test[sel]),
                                torch.device("cpu"), 2, LEVELS, 0, H, W, types=name, vel_model=Optim_velocity,
                                kernel=kernel, lat=lat, lon=lon)
            vel = torch.from_numpy(np.load(name + "_vel.npy"))[0]
            batch = test[ic:ic + L + 1]
            T = time_steps[ic:ic + L + 1].float().flatten()
            model.update_param([vel.view(2, 2 * 5, H, W), const, lat_map, lon_map])
            with torch.no_grad():
                mean, std, _ = model(T, batch[0].view(2, 1, 5, H, W))
            m = mean[1:, 0].numpy() * scale[None, :, None, None] + mn[None, :, None, None]   # (L, 5, H, W)
            s = std[1:, 0].numpy() * scale[None, :, None, None]
            daily = m.reshape(a.lead_days, 4, 5, H, W).mean(1)
            dstd = s.reshape(a.lead_days, 4, 5, H, W).mean(1)
            np.savez(path + ".tmp.npz", daily=daily.astype(np.float32), daily_std=dstd.astype(np.float32),
                     steps=m.astype(np.float32), k=k, ic_index=ic, lat=lat, lon=lon)
            os.replace(path + ".tmp.npz", path)
            print(f"{n + 1}/{len(days)} day {k} {time.time() - t0:.0f}s nan={bool(np.isnan(daily).any())}", flush=True)
    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    main()
