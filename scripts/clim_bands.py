"""The climate's variance and memory by scale: the reference of the fluctuation-dissipation settings of the kernel.

    python scripts/clim_bands.py --json docs/results/clim_bands.json

From the cached validation year, the anomaly a_t = (x_t - climatology) / s in units of s (s = the std of the daily
increment, the unit of the kernel's deviations), per dynamic channel c and Littlewood-Paley band j of degrees
(fpk.sphere.bands, the bands of ``band_ign``):

    c_cj     = mean over the days of the band power of a_t       (area mean of the square of the band's part)
    rho_cj   = sum_t <a_t, a_(t+1)>_j / sqrt(sum_t |a_t|_j^2  sum_t |a_(t+1)|_j^2)      the one-day autocorrelation
    rho_band = mean over the channels (tisr excluded) of rho_cj

<., .>_j is the band's part of the inner product on the sphere (sum over the band's coefficients, m > 0 twice).
``--anom clim`` (default): the anomaly from the daily climatology (mean per day of year, 31-day running mean) of
``--clim-years`` (default: every cached year before the validation year; the seasonal cycle is not weather variance,
and a climatology of few years leaves its own sampling error in c: with the 2-year one of stats.npz the channel
mean of c is 3 - 12 % larger in the bands l >= 1 and 17 % smaller at l = 0, single entries off by a factor 0.1 to
2.5); ``--anom mean``: from the year's time mean (the "cpw" of diag_spectra.py).

The kernel reads "c" (run_fpk.py devk --fdt-weight: the ensemble's band power stays below it) and "rho_band"
(--fdt: the AR(1) of the noise in time per band, fixed).
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import run_fpk as R  # noqa: E402
from fpk import data as D  # noqa: E402
from fpk.env import DYN, TISR_DYN, contract, rkt  # noqa: E402
from fpk.sphere import SHT, bands  # noqa: E402


def band_stats(an: torch.Tensor, S: SHT, chunk: int = 32):
    """an (T, C, H, W) the anomalies of consecutive days -> (c (C, nb), rho (C, nb)): the mean band power and the
    one-day autocorrelation of the band's coefficients."""
    bj, nb = bands(S.L)
    band = lambda t: torch.stack([t[..., bj == b, :].sum((-1, -2)) for b in range(nb)], -1)  # noqa: E731  (.., L, M) -> (.., nb)
    two = torch.cat([torch.ones(1), 2 * torch.ones(S.L - 1)])                            # m > 0 counted twice
    T = an.shape[0]
    p0, p1, cr, pw = 0.0, 0.0, 0.0, 0.0
    prev = None
    for i in range(0, T, chunk):
        co = S.analysis(an[i:i + chunk])                                                  # (t, C, L, M)
        pw = pw + band(co.abs().pow(2) * two).sum(0)
        al = co if prev is None else torch.cat([prev, co], 0)
        a, b = al[:-1], al[1:]
        cr = cr + band((a.conj() * b).real * two).sum(0)
        p0 = p0 + band(a.abs().pow(2) * two).sum(0)
        p1 = p1 + band(b.abs().pow(2) * two).sum(0)
        prev = co[-1:]
    return pw / T, cr / torch.sqrt(p0 * p1).clamp(min=1e-30)


def daily_clim(years, half: int = 15, step: int = 64) -> np.ndarray:
    """The daily climatology (366, C, H, W) of an iterable of years (D, C, H, W) (arrays or memmaps, read in slabs,
    one year held at a time): the
    mean per day of the year (a day no year has: its neighbour), then the running mean over 2 half + 1 days,
    periodic in the year. The formula of the vendored ``stats`` (run_kai_tendency.py)."""
    nd = 366
    clim, cnt = None, np.zeros(nd)
    for x in years:
        n = min(len(x), nd)
        if clim is None:
            clim = np.zeros((nd,) + tuple(x.shape[1:]), dtype=np.float32)
        for i in range(0, n, step):
            clim[i:min(i + step, n)] += x[i:min(i + step, n)]
        cnt[:n] += 1
    for i in range(nd):
        clim[i] = clim[i] / cnt[i] if cnt[i] else clim[i - 1]
    out = np.empty_like(clim)
    for i in range(nd):
        out[i] = clim[np.arange(i - half, i + half + 1) % nd].mean(0)
    return out


def main():
    p = argparse.ArgumentParser()
    R.common(p)
    p.add_argument("--anom", default="clim", choices=["clim", "mean"])
    p.add_argument("--clim-years", type=int, nargs="*", default=None,
                   help="years of the daily climatology (default: every cached year before --val-year; "
                        "'--clim-years' alone: the climatology of stats.npz, the first run's training years)")
    p.add_argument("--json", default="docs/results/clim_bands.json")
    a = R.years_of(p.parse_args())
    torch.set_num_threads(a.threads or 1)
    pa = R.pargs(a)
    cl = D.climate(pa)
    va = D.load(pa, [a.val_year])[a.val_year]
    dyn = np.asarray(DYN)
    x = torch.from_numpy(np.asarray(va[:, dyn], dtype=np.float32))
    del va
    cy = []
    if a.anom == "clim":
        if a.clim_years is None:
            cy = sorted(int(f[:4]) for f in os.listdir(pa.cache) if f[:4].isdigit() and f.endswith(".npy")
                        and len(f) == 8 and int(f[:4]) < a.val_year)
        else:
            cy = list(a.clim_years)
        clim = cl.clim if not cy else daily_clim(
            rkt().daily(os.path.join(pa.data, f"{y}.nc"), os.path.join(pa.cache, f"{y}.npy")) for y in cy)
        z = lambda v: v.astype(np.float32)[None, :, None, None]  # noqa: E731
        ref = torch.from_numpy(((clim[:len(x), dyn] - z(cl.mean[dyn])) / z(cl.std[dyn] + rkt().EPS)))
        del clim
    else:
        ref = x.mean(0, keepdim=True)
    s = torch.tensor(cl.inc[dyn], dtype=torch.float32).view(1, -1, 1, 1)                  # = core.t6
    S = SHT(cl.lat)
    with torch.no_grad():
        c, rho = band_stats(x.sub_(ref).div_(s), S)
    bj, nb = bands(S.L)
    keep = np.arange(66) != TISR_DYN
    out = dict(meta=dict(year=a.val_year, days=len(x), anom=a.anom, clim_years=cy or "stats.npz", channels=[contract.CHANNEL_NAMES[i] for i in DYN],
                         bands=[[int(np.where(bj.numpy() == b)[0].min()), int(np.where(bj.numpy() == b)[0].max())]
                                for b in range(nb)]),
               c=c.tolist(), rho=rho.tolist(), rho_band=rho[keep].mean(0).tolist())
    json.dump(out, open(a.json, "w"))
    f = lambda v: " ".join(f"{q:9.3f}" for q in v)  # noqa: E731
    print("bands (degrees):", out["meta"]["bands"])
    print("rho_band        ", f(out["rho_band"]))
    for n in ("z500", "t850", "u250", "msl"):
        k = out["meta"]["channels"].index(n)
        print(f"{n:5s} c   {f(out['c'][k])}\n      rho {f(out['rho'][k])}")


if __name__ == "__main__":
    main()
