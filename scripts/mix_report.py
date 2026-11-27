"""A markdown table of scored arms at leads 1 / 3 / 5 / 7 / 10 (the mixed kernel and the condition, the + chain's
``mixrep`` step): z500, t850, t2m RMSE (physical), 65-channel nRMSE, CRPS, spread / skill and the IGN of the scorer;
then the paired comparisons of ``pair_nophys.py`` json files.

    FPK_OUT=runs/fpk_plus python scripts/mix_report.py runs/fpk_plus/mix_plus.md \\
        score_mixk_plus:devk_band_sph=physics score_mixk_plus:devk_band_sph_ng=no-physics ... \\
        --pairs "mixed + condition - no-physics=runs/fpk_plus/pair_mixkp_vs_ng.json"
"""
import argparse
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("md")
ap.add_argument("arms", nargs="+", help="FILE:ARM=label (FILE.json in $FPK_OUT, else runs/fpk)")
ap.add_argument("--pairs", nargs="*", default=[], help="label=pair_nophys.py json")
A = ap.parse_args()
R = os.environ.get("FPK_OUT", "runs/fpk").rstrip("/") + "/"
Q = [0, 2, 4, 6, 9]
f2 = lambda v, d: " / ".join(f"{x:.{d}f}" for x in np.asarray(v)[Q])  # noqa: E731
rows, cache = [], {}
for spec in A.arms:
    fa, label = spec.split("=", 1) if "=" in spec else (spec, spec)
    f, arm = fa.split(":", 1)
    if f not in cache:
        cache[f] = json.load(open(R + f"{f}.json"))["scores"]
    s = cache[f][arm]
    rp = s["rmse_phys"]
    rows.append(f"| {label} | {f2(rp['z500'], 1)} | {f2(rp['t850'], 3)} | {f2(rp['t2m'], 3)} | {f2(s['nrmse'], 4)} | "
                f"{f2(s['crps'], 4)} | {f2(s['spread_skill'], 3)} | {f2(s['ign'], 3)} |")
out = ["# The mixed kernel and the invariant-measure condition", "",
       "Leads 1 / 3 / 5 / 7 / 10 days. IGN: the scorer's (it depends on the member count, see docs/results/mix_kernel.md;",
       "the corrected values are in mix_ign_*.json).", "",
       "| arm | z500 RMSE | t850 RMSE | t2m RMSE | nRMSE | CRPS | spread/skill | IGN |",
       "| --- | --- | --- | --- | --- | --- | --- | --- |"] + rows
for spec in A.pairs:
    label, path = spec.split("=", 1)
    p = json.load(open(path))
    ci = np.asarray(p["crps"]["ci"])
    g, gci = np.asarray(p["nrmse_gain_pct"][0]), np.asarray(p["nrmse_gain_pct"][1])
    out += ["", f"**{label}** (paired over the ICs, 95 % bootstrap)", "",
            "| | 1 | 3 | 5 | 7 | 10 |", "| --- | --- | --- | --- | --- | --- |",
            "| CRPS diff | " + " | ".join(f"{d:+.4f} [{a:+.4f}, {b:+.4f}]" for d, a, b in zip(p["crps"]["diff"], ci[0], ci[1])) + " |",
            "| nRMSE gain % | " + " | ".join(f"{d:+.2f} [{a:+.2f}, {b:+.2f}]" for d, a, b in zip(g, gci[0], gci[1])) + " |"]
open(A.md, "w").write("\n".join(out) + "\n")
print("\n".join(out))
