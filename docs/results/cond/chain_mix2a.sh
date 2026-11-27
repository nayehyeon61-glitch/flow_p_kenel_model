#!/bin/bash
# What does the empirical-error kernel have that the learned kernels lack (2026-11-24 22:25 KST)?
#   A  the learned kernels' variance map and their means' squared-error map on 2016 (178 ICs)
#   B, C  2020 and 2017: the learned kernels rescaled with those maps (the second-moment condition; one factor per lead and
#         channel, per latitude, per grid point), and the empirical-error kernel reduced to its width map (gdress) or to the
#         errors of the same season (sdress), next to the first candidates (same seeds: W_p, NW_p and the first six
#         synthetic kernels come out as in mixmat_{2020,2017}.npz)
set -x
cd /home/claude/flow_p_kenel_model
Y="--train-years 2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015"
W=core_gngmz8W:devk_band_sphW_p; N=core_ngmz8W:devk_band_sphNW_p; D=core_gngmz8W+core_ngmz8W
SYNTH="clim kaf dress:lim dress:$D dress:core_gngmz8W dress:core_ngmz8W gdress:$D sdress:$D"
C=runs/fpk/cond
[ -f $C/maps_2016.npz ] || python scripts/mix_matrix.py --year 2016 --ic-stride 2 --pool 2020 --learned $W $N --save-maps $C/maps_2016.npz --npz $C/mixmat_maps_2016.npz $Y > $C/maps_2016.log 2>&1
for yp in "2020 2016" "2017 2016 2020"; do
  set -- $yp; y=$1; shift
  [ -f $C/mixmat2a_$y.npz ] || python scripts/mix_matrix.py --year $y --pool $@ --learned $W $N --cal-maps $C/maps_2016.npz --synth $SYNTH --save-maps $C/maps_$y.npz --npz $C/mixmat2a_$y.npz $Y > $C/mixmat2a_$y.log 2>&1
done
VAR="$W@cglob $W@clat $W@cpt $N@cglob $N@clat $N@cpt"
G=gdress:$D; S=sdress:$D; DR=dress:$D
SETS="$W@cglob,$N@cglob $W@clat,$N@clat $W@cpt,$N@cpt $W,$N,$DR $W,$N,$G $W,$N,$S $W@clat,$N@clat,$DR $W@cpt,$N@cpt,$DR $W@cpt,$N@cpt,$G $W,$N,$DR,clim $W@cpt,$N@cpt,$DR,clim $W@cpt,$N@cpt,clim $G,$DR $W,$N,$G,$DR"
python scripts/mix_design.py --fit $C/mixmat2a_2020.npz --test $C/mixmat2a_2017.npz --base $W $N --drop $VAR --sets $SETS --json $C/mix_design2a.json > $C/mix_design2a.log 2>&1
python scripts/mix_design.py --fit $C/mixmat2a_2017.npz --test $C/mixmat2a_2020.npz --base $W $N --drop $VAR --sets $SETS --json $C/mix_design2a_rev.json > $C/mix_design2a_rev.log 2>&1
touch $C/chain_mix2a_done
