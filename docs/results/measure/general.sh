#!/bin/bash
# the measure continuation at other leads and channels, on three cores, and one larger setting (l < 32, 400 patterns, one lag)
cd /home/claude/flow_p_kenel_model
C3=core_gngmz8W,core_ngmz8W,core_gngmz8W_s1
python scripts/measure_core.py --days 3 5 7 10 --chans z500 t850 t2m --taus 0 1 2 3 4 5 7 10 > runs/fpk/measure/core_hyb_general.log 2>&1
python scripts/measure_core.py --days 10 --taus 3 4 --lt 32 --npc 400 --lags 1 --tag _b > runs/fpk/measure/core_hyb_b.log 2>&1
python scripts/measure_core.py --days 10 --taus 0 3 4 --cores $C3 --tag _3 > runs/fpk/measure/core_hyb_3.log 2>&1
python scripts/measure_core.py --days 10 --taus 0 --lt 32 --npc 400 --cores $C3 --tag _3l32 > runs/fpk/measure/core_hyb_3l32.log 2>&1
python scripts/measure_core.py --days 10 --taus 3 4 --lt 32 --npc 400 --lags 1 --cores $C3 --tag _3b > runs/fpk/measure/core_hyb_3b.log 2>&1
for c in z500 t850 t2m; do
  python scripts/measure_cond.py --chan $c --day 3 --threads 4 --exps bands hyb3 hyb2 hyb1 hyb0 hyb3+hyb2 hyb3+hyb1 --json runs/fpk/measure/m5_${c}_d3.json > runs/fpk/measure/m5_${c}_d3.log 2>&1
  python scripts/measure_cond.py --chan $c --day 5 --threads 4 --exps bands hyb5 hyb4 hyb3 hyb2 hyb5+hyb3 hyb5+hyb2 --json runs/fpk/measure/m5_${c}_d5.json > runs/fpk/measure/m5_${c}_d5.log 2>&1
  python scripts/measure_cond.py --chan $c --day 7 --threads 4 --exps bands hyb7 hyb6 hyb5 hyb4 hyb3 hyb7+hyb5 hyb7+hyb4 --json runs/fpk/measure/m5_${c}_d7.json > runs/fpk/measure/m5_${c}_d7.log 2>&1
  python scripts/measure_cond.py --chan $c --day 10 --threads 4 --exps bands hyb10 hyb7 hyb6 hyb10+hyb7 hyb10+hyb6 --json runs/fpk/measure/m5_${c}_d10.json > runs/fpk/measure/m5_${c}_d10.log 2>&1
done
python scripts/measure_cond.py --day 10 --threads 4 --exps bands hyb10@_l32 hyb7@_b hyb6@_b hyb10@_l32+hyb7@_b hyb10@_l32+hyb6@_b bands+clim16+vert bands+clim16+vert+hyb10@_l32+hyb7@_b --json runs/fpk/measure/m6_b_d10.json > runs/fpk/measure/m6_b_d10.log 2>&1
python scripts/measure_cond.py --day 10 --threads 4 --cores $C3 --exps bands hyb10@_3 hyb7@_3 hyb6@_3 hyb10@_3+hyb7@_3 hyb10@_3l32+hyb7@_3b bands+clim16+vert bands+clim16+vert+hyb10@_3+hyb7@_3 bands+clim16+vert+hyb10@_3l32+hyb7@_3b --json runs/fpk/measure/m6_three_d10.json > runs/fpk/measure/m6_three_d10.log 2>&1
