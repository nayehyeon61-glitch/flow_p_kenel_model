"""Diagnostics 1 and 2 of the function-space design (docs/design_ko.md, section 11), on the validation year.

    python scripts/diag_spectra.py --core-tag core_gngmz8 --dev-tag devk_seq_m --json docs/results/diag_spectra.json

1. Spectra by spherical-harmonic degree l (fpk.sphere, units of s, area mean of the square per degree, summed over
   the degree's orders): the error of the mean e = (truth - xbar) / s, the ensemble's deviations e_next (the
   member's offset x_m - xbar in units of s) and the kernel's new part h xi, at every lead. Calibrated: the
   deviations' spectrum equals the error's at every degree.
2. rho_eq: the share of the deviations' variance carried by the equation's directions,
   rho_eq = <|p e^r|^2> / <|e_next|^2>, e_next = (p + h xi) e^r, per lead (area-weighted, tisr excluded). A small
   rho_eq means the flow's source is mostly the isotropic new part, which a Gaussian source reproduces.
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


def main():
    p = argparse.ArgumentParser()
    R.common(p)
    p.add_argument("--dev-tag", default="devk_seq_m")
    p.add_argument("--ics", type=int, default=24)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--leads", type=int, default=10)
    p.add_argument("--json", default="docs/results/diag_spectra.json")
    a = p.parse_args()
    a.cloud_mean = False
    if a.threads:
        torch.set_num_threads(a.threads)
    pa, cl, core = R.load_core(a)
    Dk = R.load_devk(a, core, a.dev_tag)
    S = SHT(cl.lat)
    va = D.load(pa, [a.val_year])[a.val_year]
    names = json.load(open(os.path.join(a.out, "score_devk_m.json")))["meta"]["channels"]
    dyn = np.asarray(R.DYN)
    kk = torch.tensor(np.where(Dk.keep.reshape(-1).numpy() > 0)[0])
    ci = [names.index(v) for v in VARS]
    W, L = int(getattr(core, "mem", 0)), a.leads
    ics = np.linspace(8, len(va) - L - 1, a.ics).astype(int)
    w = D.area_weights(cl.lat)
    sums = {k: torch.zeros(L, 66, S.L) for k in ("err", "dev", "new", "eq")}
    rho = torch.zeros(L, 66, 2)
    g = torch.Generator().manual_seed(7)
    n = 0
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
                er = torch.exp(Dk.last["r"])[:, None]
                eq, new = Dk.last["p"] * er, Dk.last["hxi"] * er          # (B, P, 66, H, W)
                sums["err"][k] += S.spectrum((truth[:, k] - x) / Dk.s).sum(0)
                sums["dev"][k] += S.spectrum(e).mean(1).sum(0)
                sums["eq"][k] += S.spectrum(eq).mean(1).sum(0)
                sums["new"][k] += S.spectrum(new).mean(1).sum(0)
                rho[k, :, 0] += (eq.pow(2) * w).mean((1, 3, 4)).sum(0)
                rho[k, :, 1] += (e.pow(2) * w).mean((1, 3, 4)).sum(0)
        n += len(j)
        print(json.dumps(dict(ics=n)), flush=True)
    sp = {k: v / n for k, v in sums.items()}
    bj, nb = bands(S.L)
    band = lambda t: torch.stack([t[..., bj == b].sum(-1) for b in range(nb)], -1)  # noqa: E731
    out = dict(meta=dict(core_tag=a.core_tag, dev_tag=a.dev_tag, year=a.val_year, ics=n, leads=L,
                         bands=[[int(np.where(bj.numpy() == b)[0].min()), int(np.where(bj.numpy() == b)[0].max())]
                                for b in range(nb)]))
    r = rho[..., 0] / rho[..., 1]
    out["rho_eq"] = dict(all=r[:, kk].mean(1).tolist(), **{v: r[:, c].tolist() for v, c in zip(VARS, ci)})
    for key in sp:
        out[key] = dict(all=band(sp[key][:, kk].mean(1)).tolist(), **{v: band(sp[key][:, c]).tolist() for v, c in zip(VARS, ci)})
        out[key + "_deg"] = dict(all=sp[key][:, kk].mean(1).tolist(), **{v: sp[key][:, c].tolist() for v, c in zip(VARS, ci)})
    json.dump(out, open(a.json, "w"))
    f = lambda v: " ".join(f"{x:6.3f}" for x in v)  # noqa: E731
    print("bands (degrees):", out["meta"]["bands"])
    for v in ("all",) + VARS:
        print(f"== {v}: rho_eq at leads 1/5/10 = {f([out['rho_eq'][v][q] for q in (0, 4, 9) if q < L])}")
        for q in (0, 4, 9):
            if q >= L:
                continue
            print(f"  lead {q + 1:2d}  err {f(out['err'][v][q])}\n           dev {f(out['dev'][v][q])}"
                  f"\n           eq  {f(out['eq'][v][q])}\n           new {f(out['new'][v][q])}")


if __name__ == "__main__":
    main()
