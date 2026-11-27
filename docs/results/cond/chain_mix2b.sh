#!/bin/bash
# The deviation flows (the FM ensembles on the W and NW cores) as candidates of the mixture (2026-11-24 23:33 KST).
# Pass 1 rolls the two conditioned kernels and the two flows out once per year and keeps their members (--cache), with the
# flows also read on the conditioned mean (name@p) and the maps of their deviations (--save-maps). Pass 2 reads the cache and
# adds the flows rescaled by the maps of the OTHER year (name@pk, name@pklat). Then the designs, both directions.
set -x
cd /home/claude/flow_p_kenel_model
Y="--train-years 2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015"
W=core_gngmz8W:devk_band_sphW_p; N=core_ngmz8W:devk_band_sphNW_p; FW=core_gngmz8W:devflow_gW; FN=core_ngmz8W:devflow_gNW
D=core_gngmz8W+core_ngmz8W
E=${ERRK:-sdress45}
SYNTH="clim dress:$D $E:$D"
C=runs/fpk/cond
T=${THREADS:-4}
for yp in "2020 2016" "2017 2016 2020"; do
  set -- $yp; y=$1; shift
  [ -f $C/mixmat2b_$y.npz ] || python scripts/mix_matrix.py --year $y --pool $@ --pool-stride 1 --learned $W $N $FW $FN --flow-proj --cache $C/mcache \
    --save-maps $C/fmaps_$y.npz --synth $SYNTH --npz $C/mixmat2b_$y.npz --threads $T $Y > $C/mixmat2b_$y.log 2>&1
done
for yo in "2020 2017" "2017 2020"; do
  set -- $yo; y=$1; o=$2
  p="2016"; [ $y = 2017 ] && p="2016 2020"
  [ -f $C/mixmat2bk_$y.npz ] || python scripts/mix_matrix.py --year $y --pool $p --pool-stride 1 --learned $W $N $FW $FN --flow-proj --cache $C/mcache \
    --kappa-maps $C/fmaps_$o.npz --synth $SYNTH --npz $C/mixmat2bk_$y.npz --threads $T $Y > $C/mixmat2bk_$y.log 2>&1
done
VAR="$FW@p $FN@p $FW@pk $FN@pk $FW@pklat $FN@pklat"
SETS="$W,$N,$E:$D $FW,$FN $W,$N,$FW,$FN $FW@p,$FN@p $W,$N,$FW@p,$FN@p $FW@pk,$FN@pk $W,$N,$FW@pk,$FN@pk $FW@pklat,$FN@pklat $W,$N,$FW@pklat,$FN@pklat $W,$N,$E:$D,$FW@pk,$FN@pk $FW@pk,$FN@pk,$E:$D $W,$N,$E:$D,$FW,$FN $W,$N,$E:$D,$FW@p,$FN@p"
python scripts/mix_design.py --fit $C/mixmat2bk_2020.npz --test $C/mixmat2bk_2017.npz --base $W $N --drop $VAR --sets $SETS --json $C/mix_design2b.json > $C/mix_design2b.log 2>&1
python scripts/mix_design.py --fit $C/mixmat2bk_2017.npz --test $C/mixmat2bk_2020.npz --base $W $N --drop $VAR --sets $SETS --json $C/mix_design2b_rev.json > $C/mix_design2b_rev.log 2>&1
touch $C/chain_mix2b_done
