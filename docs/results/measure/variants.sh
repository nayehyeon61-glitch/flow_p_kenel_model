#!/bin/bash
# where the measure continuation takes its information from: input subsets, number of patterns, lags
cd /home/claude/flow_p_kenel_model
for v in "chan _c" "var _v" "novar _nv" "lev _lv"; do set -- $v
  python scripts/measure_core.py --taus 0 3 4 --inp $1 --tag $2 > runs/fpk/measure/core_hyb$2.log 2>&1
done
python scripts/measure_core.py --taus 0 3 4 --npc 100 --tag _n100 > runs/fpk/measure/core_hyb_n100.log 2>&1
python scripts/measure_core.py --taus 0 3 4 --npc 400 --tag _n400 > runs/fpk/measure/core_hyb_n400.log 2>&1
python scripts/measure_core.py --taus 0 3 4 --npc 800 --tag _n800 > runs/fpk/measure/core_hyb_n800.log 2>&1
python scripts/measure_core.py --taus 3 4 --lags 1 --tag _g1 > runs/fpk/measure/core_hyb_g1.log 2>&1
python scripts/measure_core.py --taus 0 3 4 --lt 32 --npc 400 --tag _l32 > runs/fpk/measure/core_hyb_l32.log 2>&1
E="bands hyb10 hyb7 hyb6 hyb10+hyb7"
for t in _c _v _nv _lv _n100 _n400 _n800 _l32; do E="$E hyb10@$t hyb7@$t hyb6@$t hyb10@$t+hyb7@$t"; done
python scripts/measure_cond.py --day 10 --threads 4 --exps $E hyb7@_g1 hyb6@_g1 hyb10+hyb7@_g1 --json runs/fpk/measure/m4_variants_d10.json > runs/fpk/measure/m4_variants_d10.log 2>&1
