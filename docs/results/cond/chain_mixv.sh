#!/bin/bash
# Verification of the design found (2026-11-24 23:5x KST): W_p 0.3 + NW_p 0.3 + the season's error kernel 0.4, nothing fitted.
#  1. a third year: 2016 was never scored while the candidates, the window and the weights were chosen (it was a pool only).
#     Scored here with the errors of 2017 + 2020 (2017's core forecasts at every date go to runs/fpk/cond_v, next to links
#     to the dumps of 2016 and 2020). The condition g of the learned kernels was fitted on 2016: in favour of the base.
#  2. 2020 and 2017 again with other seeds (the learned kernels' noise and the error dates drawn): the size of the
#     member-sampling noise in the headline numbers.
# Both with the energy score's matrices (ea, eD) and the antithetic error pairs (asdress45).
set -x
cd /home/claude/flow_p_kenel_model
Y="--train-years 2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015"
W=core_gngmz8W:devk_band_sphW_p; N=core_ngmz8W:devk_band_sphNW_p; D=core_gngmz8W+core_ngmz8W
SYNTH="clim dress:$D sdress45:$D asdress45:$D gsdress45:$D bgsdress45:$D"
C=runs/fpk/cond; V=runs/fpk/cond_v
mkdir -p $V
for f in 2016_core_gngmz8W.npy 2016_core_ngmz8W.npy 2016_o.npy 2016_x0.npy 2016_meta.json 2020_core_gngmz8W.npy 2020_core_ngmz8W.npy 2020_o.npy 2020_x0.npy 2020_meta.json archive_anom.npy archive_meta.json; do
  [ -e $V/$f ] || ln -s ../cond/$f $V/$f
done
[ -f $V/2017_meta.json ] || python scripts/diag_cond.py --cores core_gngmz8W,core_ngmz8W --year 2017 --ic-stride 1 --dir $V > $C/dump_v2017.log 2>&1 || exit 1
[ -f $C/mixmat_v2016.npz ] || python scripts/mix_matrix.py --year 2016 --pool 2017 2020 --pool-stride 1 --dir $V --learned $W $N --synth $SYNTH --npz $C/mixmat_v2016.npz $Y > $C/mixmat_v2016.log 2>&1
touch $C/chain_mixv_a_done
[ -f $C/mixmat_s1_2020.npz ] || python scripts/mix_matrix.py --year 2020 --pool 2016 --pool-stride 1 --seed 1 --learned $W $N --synth $SYNTH --npz $C/mixmat_s1_2020.npz $Y > $C/mixmat_s1_2020.log 2>&1
[ -f $C/mixmat_s1_2017.npz ] || python scripts/mix_matrix.py --year 2017 --pool 2016 2020 --pool-stride 1 --seed 1 --learned $W $N --synth $SYNTH --npz $C/mixmat_s1_2017.npz $Y > $C/mixmat_s1_2017.log 2>&1
touch $C/chain_mixv_done
