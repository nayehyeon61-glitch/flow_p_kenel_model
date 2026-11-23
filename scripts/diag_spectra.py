"""Diagnostics 1 and 2 of the function-space design (docs/design_ko.md, section 11), on the validation year.

    python scripts/diag_spectra.py --core-tag core_gngmz8 --dev-tag devk_seq_m --json docs/results/diag_spectra.json

1. Spectra by spherical-harmonic degree l (fpk.sphere, units of s, area mean of the square per degree, summed over
   the degree's orders): the error of the mean e = (truth - xbar) / s, the ensemble's deviations e_next (the
   member's offset x_m - xbar in units of s) and the kernel's new part h xi, at every lead. Calibrated: the
   deviations' spectrum equals the error's at every degree.
2. rho_eq: the share of the deviations' variance carried by the equation's directions,
   rho_eq = <|p e^r|^2> / <|e_next|^2>, e_next = (p + h xi) e^r, per lead (area-weighted, tisr excluded). A small
   rho_eq means the flow's source is mostly the isotropic new part, which a Gaussian source reproduces.

Also the fair energy score of the members in the band-weighted norm (fpk.escore; c = the band power of the validation
year's anomalies, computed here). The json has "es_version": 2:
    energy_score      a flow's P members (--flow-tag): the fair score of independent members. The kernel's 2 P
                      members xbar +- s e: P antithetic pairs, the spread term over members of different pairs only
    energy_score_v1   what version 1 (no "es_version" in meta) wrote as "energy_score": the kernel's 2 P members as
                      independent ones, biased low (a member and its mirror are 2 |e| apart) and the more the wider
                      the kernel; for a flow the same number as energy_score. Compare a version-1 file with this key
    energy_score_b1   energy_score in the norm without the band l = 0 (the global mean: the core's own mass drift
                      and a tiny climatological power, so it dominates the full norm)
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
from fpk.escore import BandNorm, band_power, energy_score  # noqa: E402
from fpk.kernel import make_gen  # noqa: E402
from fpk.sphere import SHT, bands  # noqa: E402

VARS = ("z500", "t850", "u250", "msl")


def main():
    p = argparse.ArgumentParser()
    R.common(p)
    p.add_argument("--dev-tag", default="devk_seq_m")
    p.add_argument("--ics", type=int, default=24)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--leads", type=int, default=10)
    p.add_argument("--flow-tag", default="", help="a deviation flow on --dev-tag: its members instead of the kernel's")
    p.add_argument("--json", default="docs/results/diag_spectra.json")
    a = R.years_of(p.parse_args())
    a.cloud_mean = False
    if a.threads:
        torch.set_num_threads(a.threads)
    pa, cl, core = R.load_core(a)
    Dk = R.load_devk(a, core, a.dev_tag)
    Fm = R.load_devflow(a, core, a.flow_tag) if a.flow_tag else None
    if Fm is not None:
        if Fm.meta["dev_tag"] != a.dev_tag:  # (a flow of devtune --kernel-lr is on its own tuned kernel)
            raise SystemExit(f"--flow-tag {a.flow_tag} is on the kernel {Fm.meta['dev_tag']}, not --dev-tag {a.dev_tag}")
        Dk = Fm.dk
    path, split = np.zeros((a.leads, 2)), np.zeros((a.leads, 2))
    # band-weighted energy score (fpk.escore), sums over the ICs: E|X - y|, 1/2 E|X - X'| and their difference of
    # "energy_score"; the same of es_version 1 ("energy_score_v1"); the score without the band l = 0 ("energy_score_b1")
    es = np.zeros((a.leads, 7))
    dv = torch.device(a.device)
    if Fm is not None:
        R.net_run(a, Fm)
    else:
        R.net_run(a, Dk)
    S = SHT(cl.lat, nlon=int(core.static.shape[-1])).to(dv)   # the data's grid (the 1.5 deg one has the poles)
    va = D.load(pa, [a.val_year])[a.val_year]
    names = [R.contract.CHANNEL_NAMES[c] for c in R.DYN]
    dyn = np.asarray(R.DYN)
    kk = torch.tensor(np.where(Dk.keep.reshape(-1).cpu().numpy() > 0)[0])
    ci = [names.index(v) for v in VARS]
    W, L = int(getattr(core, "mem", 0)), a.leads
    ics = np.linspace(8, len(va) - L - 1, a.ics).astype(int)
    w = D.area_weights(cl.lat).to(dv)
    sums = {k: torch.zeros(L, 66, S.L, device=dv) for k in ("err", "dev", "new", "eq")}
    with torch.no_grad():  # the climatological band power of the day-to-day anomaly (units of s), per channel
        # the year's mean in slabs of days (a 1.5 deg year read at once is 2.85 GB, and more in its z-scoring)
        ym = sum(torch.from_numpy(np.asarray(va[i:i + 32][:, dyn], dtype=np.float64)).sum(0) for i in range(0, len(va), 32))
        ym = (ym / len(va)).float()[None].to(dv)
        an = torch.from_numpy(va[::7][:, dyn]).to(dv) - ym
        cpw = torch.cat([band_power(S, an[i:i + 8] / Dk.s) for i in range(0, len(an), 8)]).mean(0).clamp(min=1e-8)  # (66, nb)
        nrm = BandNorm(S, cpw, Dk.keep)   # |f|_w (fpk.escore), w = 1 / (n_j cpw / sum n) / 65 / nb, the norm of devtune
        nrm1 = BandNorm(S, cpw, Dk.keep, bmin=1)   # the same weights without the band l = 0 (the global mean)
    rho = torch.zeros(L, 66, 2, device=dv)
    g = make_gen(7, dv)
    n = 0
    for i in range(0, len(ics), a.batch):
        j = ics[i:i + a.batch]
        x = torch.from_numpy(va[j][:, dyn]).to(dv)
        doy = torch.tensor(j, device=dv)
        tis = torch.from_numpy(np.stack([va[k + 1:k + 1 + L][:, D.TISR] for k in j])).to(dv)
        hist = torch.from_numpy(np.stack([va[np.maximum(np.arange(k - W, k), 0)][:, dyn] for k in j])).to(dv) if W else None
        truth = torch.from_numpy(np.stack([va[k + 1:k + 1 + L][:, dyn] for k in j])).to(dv)
        with torch.no_grad():
            e = Dk.start(x, doy, hist, g)
            for k in range(L):
                xp = x
                x, e, H, _ = Dk.day(x, e, doy + k, tis[:, k], k, g)
                dev = e                                                   # the members' offsets, units of s
                if Fm is not None:
                    dev = Fm.day_transport(xp, x, e, H, doy, tis[:, k], k, g)
                    path[k] += np.array(Fm.last_path) * len(j)
                    if getattr(Fm, "last_split", None) is not None:      # the core's flow: core / learned growth
                        split[k] += np.array(Fm.last_split) * len(j)
                # the energy score in the band-weighted norm |f|_w^2 = sum_c sum_j w_cj |P_j f_c|^2, w = 1 / (the
                # error's own band power of that lead is not known beforehand: the climatological band power c_cj).
                # A flow's P members are independent draws: the fair score. The kernel's 2 P members [e, -e] are P
                # antithetic pairs: the spread term over members of different pairs only (es_version 2; version 1
                # took them as 2 P independent members, which is low by (2 - sqrt 2) E|e| / (2 (2 P - 1)) for
                # Gaussian deviations: 6 % of a calibrated score at P = 4, an estimated 7 to 10 % on the stored
                # kernel runs, and the more the wider the kernel)
                anti = Fm is None
                mem, yk = dev if Fm is not None else torch.cat([e, -e], 1), (truth[:, k] - x) / Dk.s
                if anti and e.shape[1] < 2:   # one pair alone has no two independent members: no score (nan)
                    score = lambda n_: (torch.full((len(j),), float("nan")),) * 3  # noqa: E731
                else:
                    score = lambda n_: energy_score(mem, yk, n_, pairs=anti)  # noqa: E731
                fair, d1, d2 = score(nrm)                                            # ES, E|X - y|, 1/2 E|X - X'|: (B,)
                old = energy_score(mem, yk, nrm) if anti else (fair, d1, d2)         # es_version 1
                es[k] += np.array([float(q.sum()) for q in (d1, d2, fair, old[1], old[2], old[0], score(nrm1)[0])])
                er = torch.exp(Dk.last["r"])[:, None]
                eq, new = Dk.last["p"] * er, Dk.last["hxi"] * er          # (B, P, 66, H, W)
                sums["err"][k] += S.spectrum((truth[:, k] - x) / Dk.s).sum(0)
                sums["dev"][k] += S.spectrum(dev).mean(1).sum(0)
                sums["eq"][k] += S.spectrum(eq).mean(1).sum(0)
                sums["new"][k] += S.spectrum(new).mean(1).sum(0)
                rho[k, :, 0] += (eq.pow(2) * w).mean((1, 3, 4)).sum(0)
                rho[k, :, 1] += (e.pow(2) * w).mean((1, 3, 4)).sum(0)
        n += len(j)
        print(json.dumps(dict(ics=n)), flush=True)
    sp = {k: v.cpu() / n for k, v in sums.items()}
    rho = rho.cpu()
    bj, nb = bands(S.L)
    band = lambda t: torch.stack([t[..., bj == b].sum(-1) for b in range(nb)], -1)  # noqa: E731
    out = dict(meta=dict(core_tag=a.core_tag, dev_tag=a.dev_tag, year=a.val_year, ics=n, leads=L, es_version=2,
                         bands=[[int(np.where(bj.numpy() == b)[0].min()), int(np.where(bj.numpy() == b)[0].max())]
                                for b in range(nb)]))
    r = rho[..., 0] / rho[..., 1]
    out["rho_eq"] = dict(all=r[:, kk].mean(1).tolist(), **{v: r[:, c].tolist() for v, c in zip(VARS, ci)})
    for key in sp:
        out[key] = dict(all=band(sp[key][:, kk].mean(1)).tolist(), **{v: band(sp[key][:, c]).tolist() for v, c in zip(VARS, ci)})
        out[key + "_deg"] = dict(all=sp[key][:, kk].mean(1).tolist(), **{v: sp[key][:, c].tolist() for v, c in zip(VARS, ci)})
    # spread / error per band (sqrt of the power ratio): the judge of the scale allocation, 0.8-1.25 in every band
    out["band_ratio"] = {v: (np.sqrt(np.array(out["dev"][v]) / np.maximum(np.array(out["err"][v]), 1e-12))).tolist()
                         for v in out["dev"]}
    # the Bures-Wasserstein distance by band between the spread and the error (diagonal Gaussians per mode):
    # W2^2 = sum_j n_j (sigma_j - sqrt(S_j))^2, sigma_j^2 = spread power / n_j, S_j = error power / n_j
    nj = np.array([sum(2 * l + 1 for l in range(lo, hi + 1)) for lo, hi in out["meta"]["bands"]], dtype=float)
    out["w2_band"] = {v: (nj * (np.sqrt(np.array(out["dev"][v]) / nj) - np.sqrt(np.array(out["err"][v]) / nj)) ** 2).tolist()
                      for v in out["dev"]}
    out["w2"] = {v: np.sum(out["w2_band"][v], -1).tolist() for v in out["dev"]}
    f = lambda v: " ".join(f"{x:6.3f}" for x in v)  # noqa: E731
    # es_version 2. "energy_score": the fair score of the members (kernel: of its antithetic pairs, the spread term
    # over members of different pairs); "energy_score_v1": what es_version 1 wrote as "energy_score" (the kernel's
    # 2 P members as independent ones: biased low; a flow's members: the same number as "energy_score");
    # "energy_score_b1": "energy_score" in the norm without the band l = 0
    out["energy_score"] = (es[:, 2] / n).tolist()
    out["energy_score_parts"] = (es[:, :2] / n).tolist()
    out["energy_score_v1"] = (es[:, 5] / n).tolist()
    out["energy_score_parts_v1"] = (es[:, 3:5] / n).tolist()
    out["energy_score_b1"] = (es[:, 6] / n).tolist()
    if Fm is not None:
        out["path_kinetic_over_displacement"] = (path[:, 0] / np.maximum(path[:, 1], 1e-12)).tolist()
        if getattr(Fm, "last_split", None) is not None:
            out["day_growth_core_learned"] = (split / n).tolist()   # 1/2 log variance of the day, units of sigma
            print("day's growth (core, learned) at leads 1/5/10:", [f(split[q] / n) for q in (0, 4, 9) if q < L])
        print("path kinetic / displacement at leads 1/5/10:", f([out["path_kinetic_over_displacement"][q] for q in (0, 4, 9) if q < L]))
    for key, what in (("energy_score", "band-weighted energy score"), ("energy_score_v1", "    the same, es_version 1"),
                      ("energy_score_b1", "    without the band l = 0")):
        print(f"{what} at leads 1/5/10:", " ".join(f"{out[key][q]:.4f}" for q in (0, 4, 9) if q < L))
    json.dump(out, open(a.json, "w"))
    print("bands (degrees):", out["meta"]["bands"])
    for v in ("all",) + VARS:
        print(f"== {v}: rho_eq at leads 1/5/10 = {f([out['rho_eq'][v][q] for q in (0, 4, 9) if q < L])}"
              f"   Bures-W2^2 = {f([out['w2'][v][q] for q in (0, 4, 9) if q < L])}")
        for q in (0, 4, 9):
            if q >= L:
                continue
            print(f"  lead {q + 1:2d}  err {f(out['err'][v][q])}\n           dev {f(out['dev'][v][q])}"
                  f"\n           eq  {f(out['eq'][v][q])}\n           new {f(out['new'][v][q])}"
                  f"\n         ratio {f(out['band_ratio'][v][q])}")


if __name__ == "__main__":
    main()
