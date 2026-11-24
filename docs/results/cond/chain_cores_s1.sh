set -x
# 감자 2026-11-24 19:37 KST "712 까지": the linear conditions saturate near 717 and members through the core hurt, so the
# remaining lever is the core's own error. Two more cores with a new seed (one with physics, one without; the same recipe
# as core_gngmz8W / core_ngmz8W) for a 4-core mixture under the band condition. Then their fields for the condition fits.
cd /home/claude/flow_p_kenel_model
Y="2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015"
ARCH="--core-mem 8 --core-mz --core-hidden 72 --curriculum 1 2 3 4 6 8 10 --resume"
grep -q '"it": 3000' runs/fpk/core_gngmz8W_s1.log 2>/dev/null || python scripts/run_fpk.py core --core-tag core_gngmz8W_s1 --seed 1 --train-years $Y --iters 3000 --threads 4 -- $ARCH >> runs/fpk/core_gngmz8W_s1.log 2>&1 || exit 1
touch runs/fpk/core_gngmz8W_s1_done
grep -q '"it": 3000' runs/fpk/core_ngmz8W_s1.log 2>/dev/null || python scripts/run_fpk.py core --core-tag core_ngmz8W_s1 --no-gfeat --seed 1 --train-years $Y --iters 3000 --threads 4 -- $ARCH >> runs/fpk/core_ngmz8W_s1.log 2>&1 || exit 1
touch runs/fpk/core_ngmz8W_s1_done
python scripts/diag_cond.py --cores core_gngmz8W_s1,core_ngmz8W_s1 --year 2017 --ic-stride 6 --dir runs/fpk/cond_s1 > runs/fpk/cond_s1_2017.log 2>&1 || exit 1
python scripts/diag_cond.py --cores core_gngmz8W_s1,core_ngmz8W_s1 --year 2016 --ic-stride 1 --dir runs/fpk/cond_s1 > runs/fpk/cond_s1_2016.log 2>&1 || exit 1
python scripts/diag_cond.py --cores core_gngmz8W_s1,core_ngmz8W_s1 --year 2020 --ic-stride 1 --dir runs/fpk/cond_s1 > runs/fpk/cond_s1_2020.log 2>&1 || exit 1
touch runs/fpk/chain_cores_s1_done
