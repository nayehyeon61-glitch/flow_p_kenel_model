set -x
# 감자 2026-11-25 09:22 KST card "core 하나 먼저": the first sister (core_gngmz8W_s1) left 2017 at 716.0 (-1.6) with the
# new conditions, short of 3, so the second sister: the data core with a new seed (the recipe of core_ngmz8W), its
# fields, and the four-core mean under the band condition with clim16 + vert. Resumes after a restart (--resume, markers).
cd /home/claude/flow_p_kenel_model
Y="2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015"
ARCH="--core-mem 8 --core-mz --core-hidden 72 --curriculum 1 2 3 4 6 8 10 --resume"
T=core_ngmz8W_s1
grep -q '"it": 3000' runs/fpk/$T.log 2>/dev/null || python scripts/run_fpk.py core --core-tag $T --no-gfeat --seed 1 --train-years $Y --iters 3000 --threads 4 -- $ARCH >> runs/fpk/$T.log 2>&1 || exit 1
touch runs/fpk/${T}_done
mkdir -p runs/fpk/cond_s1 runs/fpk/core_mix
for ys in "2017 6" "2016 1" "2020 1"; do
  set -- $ys
  [ -f runs/fpk/cond_s1/$1_${T}_done ] || { python scripts/diag_cond.py --cores $T --year $1 --ic-stride $2 --dir runs/fpk/cond_s1 > runs/fpk/cond_s1_$1_nw.log 2>&1 && touch runs/fpk/cond_s1/$1_${T}_done; } || exit 1
  for t in o x0; do                                   # the truth and the start are the same files as in cond/: links, not copies
    f=runs/fpk/cond_s1/$1_$t.npy
    [ -L $f ] || { cmp -s $f runs/fpk/cond/$1_$t.npy && rm $f && ln -s ../cond/$1_$t.npy $f; }
  done
  ln -sf ../cond_s1/$1_$T.npy runs/fpk/cond/$1_$T.npy
done
C4=core_gngmz8W,core_ngmz8W,core_gngmz8W_s1,$T
python scripts/cond_fit.py --dir runs/fpk/cond --cores $C4 --days 10 --exps bands free clim16+vert --threads 4 --json runs/fpk/core_mix/s1b_four_d10.json > runs/fpk/core_mix/s1b_four_d10.log 2>&1
python scripts/cond_fit.py --dir runs/fpk/cond --cores $C4 --days 3 5 7 --exps bands clim16+vert --threads 4 --json runs/fpk/core_mix/s1b_four_d357.json > runs/fpk/core_mix/s1b_four_d357.log 2>&1
python scripts/core_value.py --years 2016 2020 2017 --days 10 --cores $C4,core_gngmz8L,core_gngmz8W_c --cache runs/fpk/core_mix/gram_s1b --threads 4 --json runs/fpk/core_mix/core_value_s1b.json > runs/fpk/core_mix/core_value_s1b.log 2>&1
touch runs/fpk/chain_core_s1b_done
