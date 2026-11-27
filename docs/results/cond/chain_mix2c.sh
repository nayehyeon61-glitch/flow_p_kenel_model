#!/bin/bash
# The empirical-error kernel in detail (2026-11-24 23:05 KST): every date of the pool years (--pool-stride 1), the season's
# window (+-30, 45, 60, 90 days, all the year), the season's width map alone (gsdress), next to the two learned kernels
# (and their point-wise rescaled variants) and the invariant measure.
set -x
cd /home/claude/flow_p_kenel_model
Y="--train-years 2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015"
W=core_gngmz8W:devk_band_sphW_p; N=core_ngmz8W:devk_band_sphNW_p; D=core_gngmz8W+core_ngmz8W
SYNTH="clim dress:$D sdress30:$D sdress45:$D sdress60:$D sdress90:$D gsdress:$D gdress:$D"
C=runs/fpk/cond
for yp in "2020 2016" "2017 2016 2020"; do
  set -- $yp; y=$1; shift
  [ -f $C/mixmat2c_$y.npz ] || python scripts/mix_matrix.py --year $y --pool $@ --pool-stride 1 --learned $W $N --cal-maps $C/maps_2016.npz --cal pt --synth $SYNTH --npz $C/mixmat2c_$y.npz $Y > $C/mixmat2c_$y.log 2>&1
done
VAR="$W@cpt $N@cpt"
SETS="$W,$N,dress:$D $W,$N,sdress30:$D $W,$N,sdress45:$D $W,$N,sdress60:$D $W,$N,sdress90:$D $W,$N,gsdress:$D $W,$N,gdress:$D $W@cpt,$N@cpt,sdress45:$D $W@cpt,$N@cpt,sdress60:$D $W,$N,sdress45:$D,clim $W,$N,sdress60:$D,clim $W,$N,sdress45:$D,dress:$D sdress45:$D sdress60:$D"
python scripts/mix_design.py --fit $C/mixmat2c_2020.npz --test $C/mixmat2c_2017.npz --base $W $N --drop $VAR --sets $SETS --json $C/mix_design2c.json > $C/mix_design2c.log 2>&1
python scripts/mix_design.py --fit $C/mixmat2c_2017.npz --test $C/mixmat2c_2020.npz --base $W $N --drop $VAR --sets $SETS --json $C/mix_design2c_rev.json > $C/mix_design2c_rev.log 2>&1
touch $C/chain_mix2c_done
