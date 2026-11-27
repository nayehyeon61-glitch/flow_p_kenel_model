"""Conservation diagnostic (record only): how much the global invariants change from day to day.

    python scripts/diag_conserve.py --core-tag core_gngmz8 --dev-tag devk_seq_m --json docs/results/diag_conserve.json

For a few validation initial conditions and --leads days, three global numbers of every state

    mass      the area mean of the mean sea level pressure (Pa): a proxy of the surface pressure, the dry mass
    energy    the dry total energy c_p T + 1/2 (u^2 + v^2), area and level mean (J/kg; fpk.devkernel.TotalEnergy)
    aam       the axial angular momentum of the wind per unit area, (a / g) sum_k u_k cos(phi) dp_k, area mean
              (kg/s; dp_k the trapezoid weights of the pressure levels)

and the RMS over the initial conditions and days of their daily change X_(t+1) - X_t (the day-0 state included), for
    members   every member of the kernel's ensemble (--dev-tag), along its own trajectory
    mean      the kernel's mean path
    core      the core alone, from the same initial condition
    era5      the truth
The atmosphere's own daily change (era5) is the scale: a member that changes much more does not conserve.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fpk.devkernel import TotalEnergy  # noqa: E402
from fpk.env import DYN, TISR, contract  # noqa: E402
from fpk.sphere import A_EARTH  # noqa: E402

G0 = 9.80665
KEYS = ("mass", "energy", "aam")


class Invariants:
    """x (..., 66, H, W) z-scored dynamic channels -> dict of (...,) tensors: mass (Pa), energy (J/kg), aam (kg/s).
    ``mean``, ``std``: the z-score statistics of the 66 dynamic (or all 67) channels; ``lat`` in degrees."""

    def __init__(self, mean, std, lat):
        names = [contract.CHANNEL_NAMES[c] for c in DYN]
        m, s = np.asarray(mean, dtype=np.float64).reshape(-1), np.asarray(std, dtype=np.float64).reshape(-1)
        m, s = (m[DYN], s[DYN]) if m.shape[0] == 67 else (m, s)
        self.m, self.s = torch.tensor(m, dtype=torch.float32), torch.tensor(s, dtype=torch.float32)
        self.te = TotalEnergy(names, m, s, lat)
        self.msl = names.index("msl")
        self.u = [names.index(f"u{p}") for p in contract.LEVELS_HPA]
        p = np.asarray(contract.LEVELS_PA, dtype=np.float64)
        dp = np.gradient(p)                      # centred half-differences
        dp[0], dp[-1] = 0.5 * (p[1] - p[0]), 0.5 * (p[-1] - p[-2])   # trapezoid: half a layer at the ends
        self.dp = torch.tensor(dp, dtype=torch.float32).view(-1, 1, 1)
        c = np.cos(np.deg2rad(np.asarray(lat, dtype=np.float64)))
        self.cos = torch.tensor(c, dtype=torch.float32).view(-1, 1)
        self.w = torch.tensor(c / c.mean(), dtype=torch.float32).view(-1, 1)

    def __call__(self, x):
        x = x.float()
        si = lambda i: x[..., i, :, :] * self.s[i, None, None] + self.m[i, None, None]  # noqa: E731
        mass = (si([self.msl]) * self.w).mean((-1, -2, -3))
        aam = A_EARTH / G0 * ((si(self.u) * self.dp).sum(-3) * self.cos * self.w).mean((-1, -2))
        return dict(mass=mass, energy=self.te(x), aam=aam)


def daily_change(inv: Invariants, path: torch.Tensor):
    """path (..., T, 66, H, W), the states of consecutive days -> dict of (..., T - 1): X_(t+1) - X_t."""
    return {k: v[..., 1:] - v[..., :-1] for k, v in inv(path).items()}


def rms(d: torch.Tensor) -> float:
    return float(d.double().pow(2).mean().sqrt())


@torch.no_grad()
def paths(Dk, core, x0, doy, tis, leads, hist=None, gen=None):
    """The trajectories from x0 (B, 66, H, W), each with the day-0 state in front: the kernel's members
    (B, 2 P, leads + 1, 66, H, W), its mean (B, leads + 1, ...) and the core alone (B, leads + 1, ...)."""
    from pgcfm.kai_tend import core_step
    mem = int(getattr(core, "mem", 0))
    e = Dk.start(x0, doy, hist, gen)
    x, mean, memb = x0, [x0], [x0[:, None].expand(-1, 2 * Dk.cfg.pairs, -1, -1, -1)]
    for k in range(leads):
        x, e, _, _ = Dk.day(x, e, doy + k, tis[:, k], k, gen)
        mean.append(x)
        memb.append(Dk.members(x, e, Dk.s))
    if mem:
        core.reset(hist, doy.float() * 24.0, x0)
    x, det = x0, [x0]
    for k in range(leads):
        x = core_step(core, x, doy + k, tis[:, k])
        det.append(x)
    return torch.stack(memb, 2), torch.stack(mean, 1), torch.stack(det, 1)


def main():
    import run_fpk as R
    from fpk import data as D
    p = argparse.ArgumentParser()
    R.common(p)
    p.add_argument("--dev-tag", default="devk_seq_m")
    p.add_argument("--ics", type=int, default=8)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--leads", type=int, default=10)
    p.add_argument("--json", default="docs/results/diag_conserve.json")
    a = p.parse_args()
    a.cloud_mean = False
    if a.threads:
        torch.set_num_threads(a.threads)
    pa, cl, core = R.load_core(a)
    Dk = R.load_devk(a, core, a.dev_tag)
    va = D.load(pa, [a.val_year])[a.val_year]
    inv = Invariants(cl.mean, cl.std, cl.lat)
    dyn, W, L = np.asarray(DYN), int(getattr(core, "mem", 0)), a.leads
    ics = np.linspace(8, len(va) - L - 1, a.ics).astype(int)
    g = torch.Generator().manual_seed(7)
    ch = {r: {k: [] for k in KEYS} for r in ("members", "mean", "core", "era5")}
    for i in range(0, len(ics), a.batch):
        j = ics[i:i + a.batch]
        x0 = torch.from_numpy(va[j][:, dyn])
        tis = torch.from_numpy(np.stack([va[k + 1:k + 1 + L][:, TISR] for k in j]))
        hist = torch.from_numpy(np.stack([va[np.maximum(np.arange(k - W, k), 0)][:, dyn] for k in j])) if W else None
        truth = torch.from_numpy(np.stack([va[k:k + 1 + L][:, dyn] for k in j]))
        memb, mean, det = paths(Dk, core, x0, torch.tensor(j), tis, L, hist, g)
        for r, t in (("members", memb), ("mean", mean), ("core", det), ("era5", truth)):
            for k, v in daily_change(inv, t).items():
                ch[r][k].append(v.reshape(-1, L))                     # (cases, L)
        print(json.dumps(dict(ics=i + len(j))), flush=True)
    cat = {r: {k: torch.cat(v, 0) for k, v in d.items()} for r, d in ch.items()}
    out = dict(meta=dict(core_tag=a.core_tag, dev_tag=a.dev_tag, year=a.val_year, ics=len(ics), leads=L,
                         members=2 * Dk.cfg.pairs, units=dict(mass="Pa", energy="J/kg", aam="kg/s")),
               rms={r: {k: rms(v) for k, v in d.items()} for r, d in cat.items()},
               rms_by_lead={r: {k: v.double().pow(2).mean(0).sqrt().tolist() for k, v in d.items()} for r, d in cat.items()})
    json.dump(out, open(a.json, "w"))
    print("RMS of the daily change      " + " ".join(f"{k + ' (' + out['meta']['units'][k] + ')':>16s}" for k in KEYS))
    for r in ("members", "mean", "core", "era5"):
        print(f"  {r:26s} " + " ".join(f"{out['rms'][r][k]:16.4g}" for k in KEYS))
    print("  members / era5             " + " ".join(f"{out['rms']['members'][k] / max(out['rms']['era5'][k], 1e-30):16.3f}" for k in KEYS))


if __name__ == "__main__":
    main()
