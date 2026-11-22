"""Is the error of the mean non-Gaussian at long leads in the large scales? (validation year)

    python scripts/diag_nongauss.py --core-tag core_gngmz8L --dev-tag devk_band_sph --json docs/results/diag_nongauss.json

For z500, t850, u250 and msl, every lead and the bands l = 1, 2-3, 4-7, 8-15: the spherical-harmonic coefficients
of the error (truth - xbar) / s (real and imaginary parts, m >= 0), standardized in two ways:

    z_ens  = coefficient / the ensemble's standard deviation of the same coefficient (the 2 P members)
    z_pool = coefficient / the root mean square of that band's coefficients over all initial conditions

and pooled over the initial conditions and the band's modes. A Gaussian predictive distribution with the right
spread gives var(z_ens) = 1 (times (M + 1) / (M - 3) for the M-member estimate of the variance: 1.8 at M = 8),
excess kurtosis near 0 (heavier with few members) and 4.6 % beyond 2. z_pool does not use the ensemble: its excess
kurtosis measures how far the error itself is from one Gaussian (a mixture of regimes or a spread that changes from
day to day both make it positive); kurt_day is the same after dividing each initial condition's band by its own
root mean square (the day-to-day change of the spread taken out: what is left is non-Gaussian shape within a day).
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
from fpk.sphere import SHT, bands  # noqa: E402

VARS = ("z500", "t850", "u250", "msl")
BANDS = (1, 2, 3, 4)  # l = 1, 2-3, 4-7, 8-15


def parts(c):
    """complex (..., L, M) -> real (..., L, 2 M): real and imaginary parts."""
    return torch.cat([c.real, c.imag], -1)


def stats(z):
    z = np.asarray(z, dtype=np.float64)
    z = z[np.isfinite(z)]
    m, v = z.mean(), z.var()
    k = ((z - m) ** 4).mean() / max(v, 1e-30) ** 2 - 3.0
    return dict(n=int(z.size), var=float(v), kurt=float(k), gt2=float((np.abs(z) > 2 * np.sqrt(v)).mean()),
                gt3=float((np.abs(z) > 3 * np.sqrt(v)).mean()))


def main():
    p = argparse.ArgumentParser()
    R.common(p)
    p.add_argument("--dev-tag", default="devk_band_sph")
    p.add_argument("--ics", type=int, default=48)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--leads", type=int, default=10)
    p.add_argument("--json", default="docs/results/diag_nongauss.json")
    a = p.parse_args()
    a.cloud_mean = False
    if a.threads:
        torch.set_num_threads(a.threads)
    pa, cl, core = R.load_core(a)
    Dk = R.load_devk(a, core, a.dev_tag)
    S = SHT(cl.lat)
    bj, _ = bands(S.L)
    va = D.load(pa, [a.val_year])[a.val_year]
    names = json.load(open(os.path.join(a.out, "score_devk_m.json")))["meta"]["channels"]
    ci = [names.index(v) for v in VARS]
    dyn = np.asarray(R.DYN)
    W, L = int(getattr(core, "mem", 0)), a.leads
    ics = np.linspace(8, len(va) - L - 1, a.ics).astype(int)
    g = torch.Generator().manual_seed(7)
    err = [[] for _ in range(L)]   # per lead: (B, V, L, 2M) coefficients of the error
    sig = [[] for _ in range(L)]   # per lead: the ensemble's std of the same coefficients
    for i in range(0, len(ics), a.batch):
        j = ics[i:i + a.batch]
        x = torch.from_numpy(va[j][:, dyn])
        doy = torch.tensor(j)
        tis = torch.from_numpy(np.stack([va[k + 1:k + 1 + L][:, D.TISR] for k in j]))
        hist = torch.from_numpy(np.stack([va[np.maximum(np.arange(k - W, k), 0)][:, dyn] for k in j])) if W else None
        truth = torch.from_numpy(np.stack([va[k + 1:k + 1 + L][:, dyn] for k in j]))
        with torch.no_grad():
            e = Dk.start(x, doy, hist, g)
            for k in range(L):
                x, e, H, _ = Dk.day(x, e, doy + k, tis[:, k], k, g)
                er = ((truth[:, k] - x) / Dk.s)[:, ci]                       # (B, V, H, W)
                mem = torch.cat([e, -e], 1)[:, :, ci]                        # (B, 2P, V, H, W)
                err[k].append(parts(S.analysis(er)))
                sig[k].append(parts(S.analysis(mem)).pow(2).mean(1).sqrt())  # the pairs are symmetric: mean 0
        print(json.dumps(dict(ics=i + len(j))), flush=True)
    out = dict(meta=dict(core_tag=a.core_tag, dev_tag=a.dev_tag, year=a.val_year, ics=len(ics), leads=L,
                         members=int(2 * Dk.cfg.pairs), vars=list(VARS),
                         bands=[[int(np.where(bj.numpy() == b)[0].min()), int(np.where(bj.numpy() == b)[0].max())]
                                for b in BANDS]))
    for vi, v in enumerate(VARS):
        out[v] = []
        for k in range(L):
            E, Sg = torch.cat(err[k])[:, vi], torch.cat(sig[k])[:, vi]       # (N, L, 2M)
            row = []
            for b in BANDS:
                sel = (bj == b)
                Eb, Sb = E[:, sel], Sg[:, sel]
                ok = Sb > 1e-8                                               # drops the imaginary part of m = 0, m > l
                ze = (Eb / Sb.clamp(min=1e-8))[ok].numpy()
                pool = (Eb.pow(2) * ok).sum((1, 2), keepdim=True) / ok.sum((1, 2), keepdim=True)   # per IC
                zp = (Eb / (Eb.pow(2) * ok).sum() .div(ok.sum()).sqrt())[ok].numpy()
                zd = (Eb / pool.sqrt())[ok].numpy()
                se, sp, sd = stats(ze), stats(zp), stats(zd)
                row.append(dict(var_ens=se["var"], kurt_ens=se["kurt"], gt2_ens=float((np.abs(ze) > 2).mean()),
                                gt3_ens=float((np.abs(ze) > 3).mean()), kurt_pool=sp["kurt"], gt3_pool=sp["gt3"],
                                kurt_day=sd["kurt"], n=se["n"]))
            out[v].append(row)
    json.dump(out, open(a.json, "w"))
    print("bands (degrees):", out["meta"]["bands"], " members:", out["meta"]["members"])
    for v in VARS:
        for k in (0, 4, 6, 9):
            if k >= L:
                continue
            r = out[v][k]
            print(f"{v} lead {k + 1:2d}  var_ens " + " ".join(f"{q['var_ens']:5.2f}" for q in r)
                  + "  |z|>3 " + " ".join(f"{q['gt3_ens']:.3f}" for q in r)
                  + "  kurt_pool " + " ".join(f"{q['kurt_pool']:5.2f}" for q in r)
                  + "  kurt_day " + " ".join(f"{q['kurt_day']:5.2f}" for q in r))


if __name__ == "__main__":
    main()
