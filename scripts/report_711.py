"""The table of the 711 stack (the + version, scripts/run_fpk_plus.sh steps cfit, kmix, rep711).

    python scripts/report_711.py runs/fpk_plus/plus711.md --two runs/fpk_plus/cond_two.json \\
        --four runs/fpk_plus/cond711.json --fixed runs/fpk_plus/mix_fixed711.log

--two / --four: scripts/cond_fit.py json (fit on the validation year, scored on the test year); --fixed: the output of
scripts/mix_fixed.py on the matrix of the kernels moved onto the conditioned mean (pasted as it is).
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--two", default="")
ap.add_argument("--four", required=True)
ap.add_argument("--fixed", default="")
ap.add_argument("--cores", default="")
A = ap.parse_args()

NAMES = {"bands": "bands of the cores' average (the invariant-measure condition per band)",
         "vert": "+ the neighbouring levels",
         "vert+math": "+ the measure continuation mc0, mc3 (the 711 stack, mean cm)"}
four = json.load(open(A.four))
two = json.load(open(A.two)) if A.two else None
days = sorted(four["days"], key=int)
L = []
fit, test = four["meta"]["fit"], four["meta"]["test"]
L.append(f"# The 711 stack: fitted on {', '.join(map(str, fit))}, scored on {test} ({four['meta']['test_ics']} ICs)\n")
L.append("5.625 deg (16-year cores, 2017, z500 day 10): two cores' bands 717.6 -> four cores 714.6 -> + 16-year climatology and the "
         "neighbouring levels 713.4 -> + the measure continuation 711.0. On this grid the climatology of stats.npz is already the "
         "training years' (+-15 days), so the climatology piece is in every row.\n")
if A.cores:
    L.append(f"Cores: {A.cores}\n")
for var, unit in (("z500", "m2/s2"), ("t850", "K"), ("t2m", "K")):
    L.append(f"\n## {var} RMSE of the conditioned mean ({unit}), day " + " / ".join(days) + "\n")
    L.append("| mean | " + " | ".join(f"day {d}" for d in days) + " |")
    L.append("| --- |" + " --- |" * len(days))
    rows = []
    if two is not None:
        rows.append(("two cores: " + NAMES["bands"], [two["days"][d]["bands"][var] if d in two["days"] else None for d in days]))
    nc = len(A.cores.split(",")) if A.cores else None
    for e in four["days"][days[0]]:
        lab = (f"{nc} cores: " if nc else "") + NAMES.get(e, e)
        rows.append((lab, [four["days"][d][e][var] for d in days]))
    fmt = (lambda v: f"{v:.1f}") if var == "z500" else (lambda v: f"{v:.3f}")
    for lab, vals in rows:
        L.append(f"| {lab} | " + " | ".join("–" if v is None else fmt(v) for v in vals) + " |")
L.append("\n## Increments of z500 on the test year (paired bootstrap, 95 %)\n")
for d in days:
    r = four["days"][d]
    parts = [f"{e} − bands {r[e]['z500_minus_bands']}" for e in r if "z500_minus_bands" in r[e]]
    if two is not None and d in two["days"]:
        parts.insert(0, f"four cores' bands − two cores' bands {round(r['bands']['z500'] - two['days'][d]['bands']['z500'], 2)} (unpaired)")
    L.append(f"- day {d}: " + "; ".join(parts))
if A.fixed:
    L.append("\n## CRPS: the kernels on the conditioned mean (scripts/mix_fixed.py; base = the + model's half and half)\n")
    L.append("```")
    L.append(open(A.fixed).read().rstrip())
    L.append("```")
open(A.out, "w").write("\n".join(L) + "\n")
print(A.out)
