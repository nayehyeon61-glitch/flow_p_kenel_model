set -x
cd /home/claude/flow_p_kenel_model
Y="--train-years 2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015"
[ -f runs/fpk/proj_2016_L.npz ] || python scripts/diag_projection.py stats --cores core_gngmz8L --year 2016 --ic-stride 2 --npz runs/fpk/proj_2016_L.npz > runs/fpk/cond/proj_2016_L.log 2>&1 || exit 1
[ -f runs/fpk/devk_band_sph_p.pt ] || python scripts/run_fpk.py projk --core-tag core_gngmz8L --dev-tag devk_band_sph --tag devk_band_sph_p --proj-init runs/fpk/proj_2016_L.npz > runs/fpk/cond/projk_L.log 2>&1 || exit 1
LEARNED="core_gngmz8W:devk_band_sphW_p core_ngmz8W:devk_band_sphNW_p core_ngmz8W:devk_band_sphEqNW_p core_gngmz8L:devk_band_sph_p"
SYNTH="clim kaf dress:lim dress:core_gngmz8W+core_ngmz8W dress:core_gngmz8W dress:core_ngmz8W"
python scripts/mix_matrix.py --year 2020 --pool 2016 --learned $LEARNED --synth $SYNTH --npz runs/fpk/cond/mixmat_2020.npz $Y > runs/fpk/cond/mixmat_2020.log 2>&1 || exit 1
python scripts/mix_matrix.py --year 2017 --pool 2016 2020 --learned $LEARNED --synth $SYNTH --npz runs/fpk/cond/mixmat_2017.npz $Y > runs/fpk/cond/mixmat_2017.log 2>&1 || exit 1
python scripts/mix_design.py --fit runs/fpk/cond/mixmat_2020.npz --test runs/fpk/cond/mixmat_2017.npz --base core_gngmz8W:devk_band_sphW_p core_ngmz8W:devk_band_sphNW_p --json runs/fpk/cond/mix_design.json > runs/fpk/cond/mix_design.log 2>&1
touch runs/fpk/cond/chain_mix_done
