#!/bin/bash
# The measure condition read out on the kernels (감자 2026-11-25 16:49 KST): the mixture of now (0.3 K_W^p + 0.3 K_NW^p + 0.4 K_err)
# against the same kernels moved onto the conditioned mean without (cv) and with (cm) the measure continuation, and the
# error kernel built about those means (their own out-of-sample errors of the pool years).
cd /home/claude/flow_p_kenel_model
Y="--train-years 2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015"
W=core_gngmz8W:devk_band_sphW_p; N=core_ngmz8W:devk_band_sphNW_p
D=core_gngmz8W+core_ngmz8W
C=runs/fpk/cond; O=runs/fpk/measure
for yp in "2017 2016 2020" "2020 2016"; do
  set -- $yp; y=$1; shift
  [ -f $O/mixmat_mc_$y.npz ] || python scripts/mix_matrix.py --year $y --pool $@ --pool-stride 1 --learned $W $N --recenter cv cm --cache $C/mcache \
    --synth sdress45:$D sdress45:cv sdress45:cm --npz $O/mixmat_mc_$y.npz --threads 4 $Y > $O/mixmat_mc_$y.log 2>&1 || exit 1
done
B="sphW_p=0.3,sphNW_p=0.3,sdress45:core_gngmz8W=0.4"
python scripts/mix_fixed.py $O/mixmat_mc_2017.npz $O/mixmat_mc_2020.npz --base "$B" --mix \
  "sphW_p=0.3,sphNW_p=0.3,sdress45:cv=0.4" "sphW_p=0.3,sphNW_p=0.3,sdress45:cm=0.4" \
  "sphW_p@cv=0.3,sphNW_p@cv=0.3,sdress45:cv=0.4" "sphW_p@cm=0.3,sphNW_p@cm=0.3,sdress45:cm=0.4" \
  "sphW_p@cm=0.5,sphNW_p@cm=0.5" "sdress45:cm=1.0" "sphW_p@cm=0.2,sphNW_p@cm=0.2,sdress45:cm=0.6" > $O/mix_fixed_mc.log 2>&1
python scripts/mix_fixed.py $O/mixmat_mc_2017.npz $O/mixmat_mc_2020.npz --base "sphW_p=0.5,sphNW_p=0.5" --mix "$B" \
  "sphW_p@cm=0.3,sphNW_p@cm=0.3,sdress45:cm=0.4" > $O/mix_fixed_mc_vs_half.log 2>&1
touch $O/chain_kmix_done
