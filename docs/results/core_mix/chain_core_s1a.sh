set -x
# 감자 2026-11-25 09:17 KST "z500 10일을 3 만 내려도": one more core of the same recipe with a new seed (the physics core
# first), its fields for the condition fits, and the three-core mean under the band condition with the two conditions
# that survived (clim16, vert). Resumes after a restart of the machine (--resume, the markers).
cd /home/claude/flow_p_kenel_model
Y="2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015"
ARCH="--core-mem 8 --core-mz --core-hidden 72 --curriculum 1 2 3 4 6 8 10 --resume"
T=core_gngmz8W_s1
grep -q '"it": 3000' runs/fpk/$T.log 2>/dev/null || python scripts/run_fpk.py core --core-tag $T --seed 1 --train-years $Y --iters 3000 --threads 4 -- $ARCH >> runs/fpk/$T.log 2>&1 || exit 1
touch runs/fpk/${T}_done
mkdir -p runs/fpk/cond_s1 runs/fpk/core_mix
for ys in "2017 6" "2016 1" "2020 1"; do
  set -- $ys
  [ -f runs/fpk/cond_s1/$1_${T}_done ] || { python scripts/diag_cond.py --cores $T --year $1 --ic-stride $2 --dir runs/fpk/cond_s1 > runs/fpk/cond_s1_$1.log 2>&1 && touch runs/fpk/cond_s1/$1_${T}_done; } || exit 1
  ln -sf ../cond_s1/$1_$T.npy runs/fpk/cond/$1_$T.npy
done
W=core_gngmz8W,core_ngmz8W
python scripts/cond_fit.py --dir runs/fpk/cond --cores $W --third $T --days 10 --exps bands free3 clim16+vert free3+clim16+vert --threads 4 --json runs/fpk/core_mix/s1a_third_d10.json > runs/fpk/core_mix/s1a_third_d10.log 2>&1
python scripts/cond_fit.py --dir runs/fpk/cond --cores $W,$T --days 10 --exps bands free clim16+vert --threads 4 --json runs/fpk/core_mix/s1a_three_d10.json > runs/fpk/core_mix/s1a_three_d10.log 2>&1
python scripts/cond_fit.py --dir runs/fpk/cond --cores $W,$T --days 3 5 7 --exps bands clim16+vert --threads 4 --json runs/fpk/core_mix/s1a_three_d357.json > runs/fpk/core_mix/s1a_three_d357.log 2>&1
touch runs/fpk/chain_core_s1a_done
