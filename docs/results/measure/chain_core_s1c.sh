#!/bin/bash
# After the second sister core (chain_core_s1b.sh): the measure continuation of the FOUR cores' average as two more
# "cores" (measure_core.py --dump) and the official scoring with it (cond_fit.py --math), 2026-11-25.
cd /home/claude/flow_p_kenel_model
until [ -f runs/fpk/chain_core_s1b_done ]; do sleep 60; done
C4=core_gngmz8W,core_ngmz8W,core_gngmz8W_s1,core_ngmz8W_s1
M="--days 3 5 7 10 --lt 32 --npc 400 --cores $C4"
[ -f runs/fpk/cond/2017_mc0q.npy ] || python scripts/measure_core.py --dump mc0q --taus 0 $M > runs/fpk/measure/dump_mc0q.log 2>&1 || exit 1
[ -f runs/fpk/cond/2017_mc3q.npy ] || python scripts/measure_core.py --dump mc3q --taus 3 --lags 1 $M > runs/fpk/measure/dump_mc3q.log 2>&1 || exit 1
python scripts/cond_fit.py --dir runs/fpk/cond --cores $C4 --days 3 5 7 10 --exps bands clim16+vert clim16+vert+math --math mc0q,mc3q --threads 4 \
  --json runs/fpk/core_mix/s1c_four_measure.json > runs/fpk/core_mix/s1c_four_measure.log 2>&1 || exit 1
touch runs/fpk/chain_core_s1c_done
