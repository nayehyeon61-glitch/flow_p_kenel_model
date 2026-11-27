#!/bin/bash
# Frozen configurations on the 295 starts of 2017 that no measure experiment scored (the daily dump runs/fpk/cond_v without
# the 60 starts of every 6th day), against bands and against clim16+vert (the measure continuation's own increment), with a
# 15-day block bootstrap. tuned: l < 32, 400 patterns, tau 0 + tau 3 with one lag (the one of mc_two); default: l < 16,
# 200 patterns, tau 0 + tau 3, no lag (the settings before any tuning).
cd /home/claude/flow_p_kenel_model
V=runs/fpk/cond_v; O=runs/fpk/measure
python scripts/measure_core.py --dir $V --dump mc0 --taus 0 --days 5 10 --lt 32 --npc 400 > $O/u_dump_mc0.log 2>&1 || exit 1
python scripts/measure_core.py --dir $V --dump mc3 --taus 3 --lags 1 --days 5 10 --lt 32 --npc 400 > $O/u_dump_mc3.log 2>&1 || exit 1
python scripts/measure_core.py --dir $V --dump md0 --taus 0 --days 10 > $O/u_dump_md0.log 2>&1 || exit 1
python scripts/measure_core.py --dir $V --dump md3 --taus 3 --days 10 > $O/u_dump_md3.log 2>&1 || exit 1
python scripts/cond_fit.py --dir $V --days 10 5 --exps bands clim16+vert clim16+vert+math --math mc0,mc3 --ref clim16+vert --test-drop-mod 6 --block 15 \
  --threads 4 --json $O/u_tuned.json > $O/u_tuned.log 2>&1 || exit 1
python scripts/cond_fit.py --dir $V --days 10 --exps bands clim16+vert clim16+vert+math math --math md0,md3 --ref clim16+vert --test-drop-mod 6 --block 15 \
  --threads 4 --json $O/u_default.json > $O/u_default.log 2>&1 || exit 1
python scripts/cond_fit.py --dir $V --days 10 --exps bands clim16+vert clim16+vert+math --math mc0,mc3 --ref clim16+vert --block 15 \
  --threads 4 --json $O/u_tuned_all355.json > $O/u_tuned_all355.log 2>&1 || exit 1
touch $O/untouched_done
