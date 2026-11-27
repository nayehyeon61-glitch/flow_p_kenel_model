#!/bin/bash
# the measure continuation of the two cores' forecast for every channel, as two more cores for cond_fit.py
cd /home/claude/flow_p_kenel_model
python scripts/measure_core.py --dump mc0 --taus 0 --days 1 3 5 7 10 --lt 32 --npc 400 > runs/fpk/measure/dump_mc0.log 2>&1 || exit 1
python scripts/measure_core.py --dump mc3 --taus 3 --lags 1 --days 1 3 5 7 10 --lt 32 --npc 400 > runs/fpk/measure/dump_mc3.log 2>&1 || exit 1
