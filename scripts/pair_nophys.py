"""Paired bootstrap (2000 draws over the ICs) of two scored arms on the 65 scored channels: nRMSE gain %, CRPS,
IGN and the predicted entropy H (test minus reference), plus the json summaries of both arms.

    python scripts/pair_nophys.py devk_band_sphW devk_band_sphNW docs/results/kernel_nophys.json
"""
import json, sys
import numpy as np

A, B = sys.argv[1], sys.argv[2]   # ref arm (physics), test arm (no physics)
R = "runs/fpk/"
ra, rb = np.load(R + f"score_{A}_raw.npz"), np.load(R + f"score_{B}_raw.npz")
ja, jb = json.load(open(R + f"score_{A}.json")), json.load(open(R + f"score_{B}.json"))
assert np.array_equal(ra["ics"], rb["ics"])
TISR_DYN = 62
sc = [i for i in range(66) if i != TISR_DYN]
def g(r, a, k): return r[f"{a}__{k}"].astype(np.float64)[:, :, sc]   # (IC, lead, 65)
clim = ra["climatology__mse"].astype(np.float64)[:, :, sc]
out = {}
rng = np.random.default_rng(0)
n = clim.shape[0]
draws = rng.integers(0, n, size=(2000, n))
def ci(stat):   # stat(idx) -> (lead,)
    v = np.array([stat(d) for d in draws])
    return np.percentile(v, [2.5, 97.5], axis=0)
Q = [0, 2, 4, 6, 9]
# nRMSE: sqrt(mean over IC and channels of mse / clim mse)? use the json's own nrmse to check, gain % = 1 - B/A
ma, mb = g(ra, A, "mse"), g(rb, B, "mse")
def nr(m, idx): return np.sqrt(m[idx].mean(0)).mean(-1)
full = np.arange(n)
print("json nrmse A", np.round(ja["scores"][A]["nrmse"], 4)[Q], "mine", np.round(nr(ma, full), 4)[Q])
gain = lambda idx: 100 * (1 - nr(mb, idx) / nr(ma, idx))
out["nrmse_gain_pct"] = [gain(full)[Q].round(2).tolist(), ci(gain)[:, Q].round(2).tolist()]
for k in ("crps", "IGN", "H"):
    xa, xb = g(ra, A, k), g(rb, B, k)
    d = lambda idx: (xb[idx] - xa[idx]).mean((0, 2))
    out[k] = dict(A=xa.mean((0, 2))[Q].round(4).tolist(), B=xb.mean((0, 2))[Q].round(4).tolist(),
                  diff=d(full)[Q].round(4).tolist(), ci=ci(d)[:, Q].round(4).tolist())
print("json crps A", np.round(ja["scores"][A]["crps"], 4)[Q], "json ign A", np.round(ja["scores"][A]["ign"], 4)[Q])
for k in ("spread_skill", "entropy_error_corr", "spread", "ign", "entropy", "crps", "nrmse"):
    out["json_" + k] = {A: np.round(ja["scores"][A][k], 4)[Q].tolist(), B: np.round(jb["scores"][B][k], 4)[Q].tolist()}
out["rmse_z500"] = {A: np.round(ja["scores"][A]["rmse_phys"]["z500"], 1)[Q].tolist(), B: np.round(jb["scores"][B]["rmse_phys"]["z500"], 1)[Q].tolist()}
for v in ("z500", "t850", "u850", "t2m", "msl"):
    out["crps_" + v] = {A: np.round(ja["scores"][A]["crps_vars"][v], 3)[Q].tolist(), B: np.round(jb["scores"][B]["crps_vars"][v], 3)[Q].tolist()}
    out["ss_" + v] = {A: np.round(ja["scores"][A]["spread_skill_vars"][v], 3)[Q].tolist(), B: np.round(jb["scores"][B]["spread_skill_vars"][v], 3)[Q].tolist()}
print(json.dumps(out, indent=1, ensure_ascii=False))
json.dump(out, open(sys.argv[3], "w"), indent=1, ensure_ascii=False)
