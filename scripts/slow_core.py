"""A core of the slow variable (감자 2026-11-24 21:13 KST: "어떤 core 의 형태가 마지막으로 cover 할 것 같나요?").

The day-10 error of the two cores splits into a fast part (day 10 minus the mean of days 6-10: 48 % of the error, 10 % of
its variance explained, error correlation 0.96: unpredictable) and a slow part (the mean of days 6-10: 54 %, 34 %
explained, error correlation 0.90). This core predicts the slow variable directly, with no rollout:

    input   today's anomaly and the means of days t-4..t, t-9..t-5, t-14..t-10 (4 x 66 channels), the season, the latitude
    target  the mean anomaly of days t+a .. t+b   (--target-days 6 10)
    net     the residual trunk of the kernel (fpk.nets.make_net), zero head: it starts at the climatology

Trained on the cores' training years but the last two (validation, the best state is kept); then its forecast at the
ICs of <dir>/<year>_meta.json is written as <dir>/<year>_<tag>.npy in the layout of diag_cond.py (the same field at
every lead), so cond_fit.py reads it as one more core (--math <tag>).

    python scripts/slow_core.py --tag slow --iters 2000 --train-years 2000 ... 2015
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="slow")
ap.add_argument("--target-days", type=int, nargs=2, default=[6, 10])
ap.add_argument("--years", type=int, nargs="+", default=[2016, 2020, 2017])
ap.add_argument("--dir", default="runs/fpk/cond")
ap.add_argument("--hidden", type=int, default=72)
ap.add_argument("--blocks", type=int, nargs="+", default=[2, 2])
ap.add_argument("--iters", type=int, default=2000)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--lr", type=float, default=4e-4)
ap.add_argument("--wd", type=float, default=1e-2)
ap.add_argument("--val-every", type=int, default=200)
ap.add_argument("--val-years", type=int, default=2, help="the last years of the training range kept for validation")
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
A, PASS = ap.parse_known_args()
torch.set_num_threads(A.threads)
torch.manual_seed(A.seed)
rng = np.random.default_rng(A.seed)

sys.path.insert(0, os.path.dirname(__file__))
import run_fpk as R  # noqa: E402
from fpk.env import DYN  # noqa: E402
from fpk.nets import make_net  # noqa: E402

G = {}


def grab(a, rest):
    G["a"], G["pa"] = a, R.pargs(a)
    G["cl"] = R.D.climate(G["pa"])


R.cmd_score = grab
R.main(["score", "--core-tag", "core_gngmz8W", "--arms", "core", "--threads", str(A.threads)] + PASS)
a, pa, cl = G["a"], G["pa"], G["cl"]
dyn = np.asarray(DYN)
eps = R.rkt().EPS
cm, cs = cl.mean[dyn][:, None, None], cl.std[dyn][:, None, None] + eps
CZ = ((cl.clim[:, dyn] - cm) / cs).astype(np.float32)
C, (H, W) = len(dyn), CZ.shape[-2:]
lat = np.deg2rad(np.asarray(cl.lat))
wl = torch.tensor(np.cos(lat) / np.cos(lat).mean(), dtype=torch.float32).view(1, 1, H, 1)
GEO = torch.tensor(np.stack([np.broadcast_to(np.sin(lat)[:, None], (H, W)), np.broadcast_to(np.cos(lat)[:, None], (H, W))]), dtype=torch.float32)
TA, TB = A.target_days
PAST = 15


def anomalies(year):
    te = R.D.load(pa, [year])[year]
    x = np.asarray(te[:, dyn], dtype=np.float32)
    return x - CZ[np.minimum(np.arange(len(x)), 365)]


def inputs(An, t):
    """An (D, 66, H, W) anomalies of one year, t (b,) days -> (b, 4 * 66 + 4, H, W)."""
    t = np.asarray(t)
    day = lambda j: An[np.maximum(t - j, 0)]  # noqa: E731  (clamped at the start of the year, as the cores' memory is)
    parts = [day(0)] + [np.mean([day(j) for j in range(s, s + 5)], 0) for s in (0, 5, 10)]
    x = torch.from_numpy(np.concatenate(parts, 1))
    ph = torch.tensor(2 * math.pi * t / 365.25, dtype=torch.float32)
    sea = torch.stack([ph.sin(), ph.cos()], 1)[:, :, None, None].expand(-1, -1, H, W)
    return torch.cat([x, sea, GEO.expand(len(t), -1, -1, -1)], 1)


def target(An, t):
    return torch.from_numpy(np.mean([An[np.asarray(t) + j] for j in range(TA, TB + 1)], 0))


t0 = time.time()
TR = {y: anomalies(y) for y in a.train_years}
val_y = list(a.train_years)[-A.val_years:]
fit_y = [y for y in a.train_years if y not in val_y]
pool = [(y, t) for y in fit_y for t in range(PAST - 1, len(TR[y]) - TB)]
vpool = [(y, t) for y in val_y for t in range(PAST - 1, len(TR[y]) - TB, 5)]
print(json.dumps(dict(fit=[fit_y[0], fit_y[-1]], val=val_y, samples=len(pool), val_samples=len(vpool), sec=round(time.time() - t0))), flush=True)


def batch(items):
    xs, ys = [], []
    for y in sorted(set(i[0] for i in items)):
        tt = [i[1] for i in items if i[0] == y]
        xs.append(inputs(TR[y], tt))
        ys.append(target(TR[y], tt))
    return torch.cat(xs), torch.cat(ys)


net = make_net(4 * C + 4, C, A.hidden, A.blocks)
opt = torch.optim.AdamW(net.parameters(), lr=A.lr, weight_decay=A.wd)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=A.lr, total_steps=A.iters, pct_start=max(0.05, 2.0 / A.iters))
Z = [R.contract.CHANNEL_NAMES[c] for c in DYN].index("z500")


def validate():
    net.eval()
    se, sv = 0, 0
    with torch.no_grad():
        for i in range(0, len(vpool), 32):
            x, y = batch(vpool[i:i + 32])
            p = net(x)
            se = se + ((p - y) ** 2 * wl).mean((-2, -1)).sum(0)
            sv = sv + (y ** 2 * wl).mean((-2, -1)).sum(0)
    net.train()
    return float(se.sum() / sv.sum()), float(se[Z] / sv[Z])


best, state = 9.0, None
for it in range(1, A.iters + 1):
    x, y = batch([pool[i] for i in rng.integers(0, len(pool), A.batch)])
    loss = ((net(x) - y) ** 2 * wl).mean()
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    sched.step()
    if it % A.val_every == 0 or it == A.iters:
        v, vz = validate()
        if v < best:
            best, state = v, {k: p.detach().clone() for k, p in net.state_dict().items()}
        print(json.dumps(dict(it=it, loss=round(float(loss), 4), val_mse_over_var=round(v, 4), val_z500=round(vz, 4), best=round(best, 4),
                              sec=round(time.time() - t0))), flush=True)
net.load_state_dict(state)
net.eval()
torch.save(dict(state=state, args=vars(A), val=best), f"{A.dir}/{A.tag}.pt")
del TR

for year in A.years:
    meta = json.load(open(f"{A.dir}/{year}_meta.json"))
    ics = np.asarray(meta["ics"])
    An = anomalies(year)
    out = np.lib.format.open_memmap(f"{A.dir}/{year}_{A.tag}.npy", mode="w+", dtype=np.float16, shape=(len(ics), 10, C, H, W))
    with torch.no_grad():
        for i in range(0, len(ics), 32):
            k = ics[i:i + 32]
            p = net(inputs(An, k)).numpy()
            out[i:i + len(k)] = p[:, None]                                            # the same field at every lead
    out.flush()
    print(json.dumps(dict(year=year, ics=len(ics), sec=round(time.time() - t0))), flush=True)
json.dump(dict(kind="slow", target_days=A.target_days, hidden=A.hidden, blocks=A.blocks, iters=A.iters, batch=A.batch, lr=A.lr,
               val_mse_over_var=best, fit_years=[fit_y[0], fit_y[-1]], val_years=val_y), open(f"{A.dir}/{A.tag}_info.json", "w"))
print("done", round(time.time() - t0))
