#!/bin/bash
# The second set of candidates (2026-11-24 22:1x KST): the two deviation flows (the FM ensembles on the W and NW kernels)
# and two variants of the empirical-error kernel (its width map alone, the errors of the same season) next to the first ten.
# The first ten kernels keep their seeds, so their rows of the matrix must come out as in mixmat_{2020,2017}.npz.
set -x
cd /home/claude/flow_p_kenel_model
Y="--train-years 2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015"
T=${THREADS:-3}
LEARNED="core_gngmz8W:devk_band_sphW_p core_ngmz8W:devk_band_sphNW_p core_ngmz8W:devk_band_sphEqNW_p core_gngmz8L:devk_band_sph_p core_gngmz8W:devflow_gW core_ngmz8W:devflow_gNW"
SYNTH="clim kaf dress:lim dress:core_gngmz8W+core_ngmz8W dress:core_gngmz8W dress:core_ngmz8W gdress:core_gngmz8W+core_ngmz8W sdress:core_gngmz8W+core_ngmz8W"
[ -f runs/fpk/cond/mixmat2_2020.npz ] || python scripts/mix_matrix.py --year 2020 --pool 2016 --learned $LEARNED --synth $SYNTH --npz runs/fpk/cond/mixmat2_2020.npz --threads $T $Y > runs/fpk/cond/mixmat2_2020.log 2>&1
[ -f runs/fpk/cond/mixmat2_2017.npz ] || python scripts/mix_matrix.py --year 2017 --pool 2016 2020 --learned $LEARNED --synth $SYNTH --npz runs/fpk/cond/mixmat2_2017.npz --threads $T $Y > runs/fpk/cond/mixmat2_2017.log 2>&1
B="core_gngmz8W:devk_band_sphW_p core_ngmz8W:devk_band_sphNW_p"
python scripts/mix_design.py --fit runs/fpk/cond/mixmat2_2020.npz --test runs/fpk/cond/mixmat2_2017.npz --base $B --json runs/fpk/cond/mix_design2.json > runs/fpk/cond/mix_design2.log 2>&1
python scripts/mix_design.py --fit runs/fpk/cond/mixmat2_2017.npz --test runs/fpk/cond/mixmat2_2020.npz --base $B --json runs/fpk/cond/mix_design2_rev.json > runs/fpk/cond/mix_design2_rev.log 2>&1
touch runs/fpk/cond/chain_mix2_done
