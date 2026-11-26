#!/usr/bin/env bash
# The + version of fpk on one GPU: the pipeline of the best 5.625 deg arm (core -> devk_m -> devk_band_sph ->
# devflow_line, docs/results) on the 1.5 deg grid (121 x 240 with the poles), ERA5 daily means, 32 training years
# (1984-2015), validation 2016, test 2017. The guide, with the commands from git clone to the scores:
# docs/plus_gpu_ko.md.
#
#   bash scripts/run_fpk_plus.sh                          # every step not done yet, in order
#   nohup bash scripts/gpu_free.sh bash scripts/run_fpk_plus.sh > runs/fpk_plus/chain.log 2>&1 &
#   STEPS="data stats" bash scripts/run_fpk_plus.sh       # only these steps (still in the chain's order)
#   DRYRUN=1 bash scripts/run_fpk_plus.sh                 # print the commands, run nothing
#   NOPHYS=1 bash scripts/run_fpk_plus.sh                 # the core without G(x): the same chain, tags *_ng
#
# The + model is the MIXED kernel under the invariant-measure condition (docs/results/mix_kernel.md; 5.625 deg: CRPS
# day 10 -5.4 %, z500 733 -> 718): the kernels of the two chains (core with G(x), core without) as one distribution,
# each kernel's forecast read out under the condition fitted on the validation year. So BOTH chains are run (two GPUs
# at once, or one after the other); the chain without NOPHYS ends with the mixture's steps once both kernels exist.
#
# A step that finished has the marker $OUT/done/<step> (<step>_ng for NOPHYS=1); it is skipped when the chain runs again. The training steps
# (core, devk1, devk2, flow) save their state at every log step and go on from it after a kill (--resume), so the
# chain can simply be started again. Every step's output: $OUT/logs/<step>.log. A failed step stops the chain.
#
# Steps:  data    ERA5 0.25 deg daily means (lustre) -> kaiday/<year>.npy on the 1.5 deg grid (only missing years)
#         stats   <out>/stats.npz of the training years (streaming pass) and the grid's check
#         smoke   every stage for 2-4 iterations at the full size (GPU memory and seconds per iteration in the log)
#         core    the tendency core (U-Net trunk, memory 8 + Mori-Zwanzig, G(x) inputs, 4 substeps a day)
#         devk1   the entropy deviation kernel on the frozen core (smooth noise)
#         spec    its 1-day error spectrum on THIS grid (the 5.625 deg one has 32 degrees, this one 119)
#         devk2   the chained kernel: correction, gradient through the mean's 10 days, entropy by band, sphere
#                 noise, initial deviations shaped like the spectrum of spec
#         flow    flow matching from the kernel's deviations (straight path)
#         score   2017, 60 ICs x 8 members, 10 days: persistence, core, both kernels, the flow
#         report  the per-variable table (RMSE, CRPS, spread / skill, paired bootstrap against the core)
#  then, in the chain without NOPHYS, when both chains' devk2 are done (it waits for a running NOPHYS chain):
#         proj    the strength of the condition: both cores' trajectories on the validation year (178 ICs), per lead,
#                 channel and band (scripts/diag_projection.py stats)
#         projk   each kernel with its forecast read out under the condition: devk_band_sph_p, devk_band_sph_ng_p
#         mix     2017: the mixed kernel and the mixed kernel under the condition (scripts/score_mix.py), each with
#                 both sides
#         mixrep  the table runs/fpk_plus/mix_plus.md, paired comparisons, the IGN without the member-count bias
#
# The 711 stack (5.625 deg: z500 day 10 717.6 -> 711.0 on 2017, docs/results/mix_kernel.md and project notes; 감자
# 2026-11-26 "그것까지 해야 711"): no new kernel, only cores and read-outs of their mean, so all of it but the two sister
# cores is post hoc (strengths fitted on the validation year).
#         core_s1 (each chain, after report; SISTERS=0 leaves it out) the sister core: the same recipe with seed 1
#  then, in the chain without NOPHYS, once the mixture's steps are done and the other chain's sister is there:
#         arch    the training record in the measure's coordinates (spherical harmonics l < MEASURE_LT, all 66 channels)
#         dump    the four cores' forecasts, the truth and the start on the validation year (every 2nd day) and the test
#                 year (the scorer's 60 ICs) (scripts/diag_cond.py)
#         mcore   the measure continuation (scripts/measure_core.py --dump): the four cores' average carried on by the
#                 record's transfer operator, mc0 (its projection on the measure's patterns, same day) and mc3 (the cores
#                 to day 7, the operator the last 3 days)
#         cfit    the conditioned mean (scripts/cond_fit.py, fitted on the validation year): bands of the four cores'
#                 average + the neighbouring levels + mc0, mc3 = the mean "cm" of the 711 stack (cond711.json; the
#                 reference: the two cores' bands, cond_two.json)
#         kmix    2017: the two conditioned kernels moved onto cm and the error kernel about cm (errors of the
#                 validation year, the same season) as one matrix (scripts/mix_matrix.py)
#         rep711  the table runs/fpk_plus/plus711.md: the mean (z500, t850, t2m) of every piece and the CRPS of the
#                 fixed mixture 0.3 / 0.3 / 0.4 against the + model (scripts/mix_fixed.py)
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
SRC=${SRC:-/lustre/home/mahmed/ERA5_0p25_DAILY}
DATA=${DATA:-data/era5_1p5}
OUT=${OUT:-runs/fpk_plus}
Y0=${Y0:-1984}; Y1=${Y1:-2015}; VAL=${VAL:-2016}; TEST=${TEST:-2017}
WORKERS=${WORKERS:-8}
DEV=${DEV:-cuda}
CORE_ITERS=${CORE_ITERS:-6000}
DEVK1_ITERS=${DEVK1_ITERS:-800}; DEVK2_ITERS=${DEVK2_ITERS:-400}; FLOW_ITERS=${FLOW_ITERS:-600}
DEVK_BATCH=${DEVK_BATCH:-4}; THROUGH=${THROUGH:-10}; FLOW_BATCH=${FLOW_BATCH:-4}
SFX=""; NG=""
if [ "${NOPHYS:-0}" = 1 ]; then SFX="_ng"; NG="--no-gfeat"; fi
CORE=core_u64$SFX; K1=devk_m$SFX; K2=devk_band_sph$SFX; F=devflow_line$SFX
SISTERS=${SISTERS:-1}                    # 1: one more core per chain with seed 1 (the four-core average of the 711 stack)
THREADS=${THREADS:-16}                   # CPU threads of the post-hoc steps (arch, mcore, cfit, kmix)
MEASURE_LT=${MEASURE_LT:-32}; MEASURE_NPC=${MEASURE_NPC:-400}

# the 1.5 deg architecture: the core's U-Net (levels 1.5 / 3 / 6 deg), the kernel's and the flow's U-Nets (down to
# 12 deg), the noise and the entropy's smoothing at the physical sizes of the 5.625 deg runs (17 deg and 28 deg)
CORE_SUBSTEPS=${CORE_SUBSTEPS:-4}
CORE_ARCH="--core-mem 8 --core-mz --core-trunk unet --core-unet-dims 64 128 192 --core-substeps $CORE_SUBSTEPS --batch ${CORE_BATCH:-8}"
DEVK_ARCH="--unet 32 64 128 192 --noise-cells 11 --smooth 19 --batch $DEVK_BATCH"
FLOW_ARCH="--unet 64 96 128 192 --batch $FLOW_BATCH"
GPU="--core-ckpt --amp --resume"            # core substeps recomputed in the backward pass; the nets in bf16
[ "${NET_CKPT:-0}" = 1 ] && GPU="$GPU --net-ckpt"
YRS="--train-range $Y0 $Y1 --val-year $VAL --test-year $TEST --device $DEV"
C="--data-root $DATA --out $OUT $YRS --core-tag $CORE"
K2_OPTS="--correct --through $THROUGH --seq-weight 0.5 --days 10 --lr 2e-4 --lr-correct 2e-4 --val-batches 4 --log-every 25 \
--band-scale --band-weight 0.5 --noise sphere"
FPK="$PY scripts/run_fpk.py"

mkdir -p "$OUT/done" "$OUT/logs"
step() {  # step NAME COMMAND...: run unless done (marker) or not in $STEPS
    local n=$1; shift
    if [ -n "${STEPS:-}" ] && [[ " $STEPS " != *" $n "* ]]; then return 0; fi
    local m=$n                                    # the control (NOPHYS=1) has markers and logs of its own, but
    case $n in data|stats) ;; *) m=$n$SFX ;; esac  # shares the data and the statistics
    if [ -f "$OUT/done/$m" ]; then echo "[skip] $m (done $(date -r "$OUT/done/$m" '+%m-%d %H:%M'))"; return 0; fi
    echo "[$(date '+%m-%d %H:%M')] $m: $*"
    [ "${DRYRUN:-0}" = 1 ] && return 0
    local t0=$SECONDS
    if "$@" >> "$OUT/logs/$m.log" 2>&1; then
        touch "$OUT/done/$m"
        echo "[$(date '+%m-%d %H:%M')] $m done in $(( (SECONDS - t0) / 60 )) min"
    else
        echo "[$(date '+%m-%d %H:%M')] $m FAILED, the end of $OUT/logs/$m.log:"; tail -25 "$OUT/logs/$m.log"; exit 1
    fi
}

check_gpu() {
    [ "$DEV" = cpu ] && return 0
    $PY -c "import torch, sys; ok = torch.cuda.is_available(); print('torch', torch.__version__, 'cuda', torch.version.cuda, \
torch.cuda.get_device_name(0) if ok else 'NO GPU', round(torch.cuda.get_device_properties(0).total_memory / 2**30) if ok else ''); sys.exit(0 if ok else 1)" \
        || { echo "no CUDA GPU for torch (see docs/plus_gpu_ko.md, the venv)"; exit 1; }
}

do_data() {
    local need=() y
    for y in $(seq "$Y0" "$Y1") "$VAL" "$TEST"; do [ -f "$DATA/kaiday/$y.npy" ] || need+=("$y"); done
    [ ${#need[@]} = 0 ] && { echo "all years are there"; return 0; }
    echo "converting ${#need[@]} years: ${need[*]}"
    $PY third_party/pgfm/scripts/local_era5_to_kai.py --src "$SRC" --src-daily --check --years "${need[0]}" --grid 1.5 --out "$DATA" || return 1
    $PY third_party/pgfm/scripts/local_era5_to_kai.py --src "$SRC" --src-daily --daily-mean --grid 1.5 --out "$DATA" \
        --years "${need[@]}" --workers "$WORKERS"
}

do_smoke() {  # every stage at the full grid and batch, a few iterations, in its own folder (the main run is untouched)
    local SM=$OUT/smoke$SFX
    local S="--data-root $DATA --out $SM $YRS --core-tag core_s"
    rm -rf "$SM" && mkdir -p "$SM" && cp "$OUT/stats.npz" "$SM/" || return 1
    $FPK core $S $NG --iters 4 --core-ckpt -- $CORE_ARCH --curriculum 10 --log-every 2 || return 1
    $FPK devk $S --tag devk_s1 --iters 2 --days 10 --val-batches 1 --log-every 1 $DEVK_ARCH $GPU || return 1
    $PY scripts/diag_spectra.py $S --dev-tag devk_s1 --ics 2 --batch 2 --json "$SM/spec.json" --amp || return 1
    $FPK devk $S --tag devk_s2 --init devk_s1 $K2_OPTS --iters 2 --val-batches 1 --log-every 1 \
        --e0-spec "$SM/spec.json" $DEVK_ARCH $GPU || return 1
    $FPK devflow $S --tag devflow_s --dev-tag devk_s2 --iters 2 --val-batches 1 --log-every 1 $FLOW_ARCH $GPU || return 1
    $FPK score $S --arms persistence core devk_s2 devflow_s --max-ics 2 --score-tag score_s || return 1
    # the mixture's steps, with the smoke's one core on both sides
    $PY scripts/diag_projection.py stats --cores core_s,core_s --year "$VAL" --ic-stride 90 --npz "$SM/proj.npz" \
        --data-root "$DATA" --out "$SM" $YRS || return 1
    $FPK projk $S --dev-tag devk_s2 --tag devk_s2_p --proj-init "$SM/proj.npz" || return 1
    $PY scripts/score_mix.py mix_s core_s:devk_s2 core_s:devk_s2_p --data-root "$DATA" --out "$SM" $YRS \
        --max-ics 2 --ic-batch 1 || return 1
    # the 711 stack, small: a record of three years, a dump of a few ICs, the smoke's core on both sides
    local SC="$SM/cond" SR="--data-root $DATA --out $SM $YRS"
    $PY scripts/plus_archive.py --dir "$SC" --lt 8 --years $((Y1 - 2)) $((Y1 - 1)) "$Y1" $SR || return 1
    $PY scripts/diag_cond.py --cores core_s --year "$VAL" --ic-stride 30 --dir "$SC" $SR || return 1
    $PY scripts/diag_cond.py --cores core_s --year "$TEST" --ic-stride 90 --dir "$SC" $SR || return 1
    local MM="--dir $SC --years $VAL $TEST --days 1 3 5 7 10 --lt 8 --npc 50 --cores core_s --threads $THREADS"
    $PY scripts/measure_core.py --dump mc0 --taus 0 $MM || return 1
    $PY scripts/measure_core.py --dump mc3 --taus 3 --lags 1 $MM || return 1
    $PY scripts/cond_fit.py --dir "$SC" --fit "$VAL" --test "$TEST" --cores core_s,core_s --days 1 3 5 7 10 \
        --exps bands vert vert+math --math mc0,mc3 --save-pred cm:vert+math --batch 1 --threads "$THREADS" \
        --json "$SM/cond711.json" || return 1
    $PY scripts/mix_matrix.py --year "$TEST" --ic-stride 90 --max-ics 2 --pool "$VAL" --pool-stride 1 --base-core core_s \
        --learned core_s:devk_s2 core_s:devk_s2_p --recenter cm --synth sdress400:cm asdress400:cm --dir "$SC" \
        --npz "$SM/mixmat711.npz" --threads "$THREADS" $SR || return 1
    $PY scripts/mix_fixed.py "$SM/mixmat711.npz" --base "devk_s2=0.5,devk_s2_p=0.5" \
        --mix "devk_s2@cm=0.3,devk_s2_p@cm=0.3,sdress400:cm=0.4" > "$SM/mix_fixed711.log" || return 1
    $PY scripts/report_711.py "$SM/plus711.md" --four "$SM/cond711.json" --fixed "$SM/mix_fixed711.log" || return 1
    echo "== smoke: GPU memory (GB) and seconds of the logged iterations"
    grep -ho '"gpu_gb": [0-9.]*' "$OUT/logs/smoke$SFX.log" | sort -t: -k2 -n | tail -1 || true   # none on the CPU
}

# the mixed kernel and the condition (both chains' kernels; the folder and the years of this chain)
CP=core_u64; CN=core_u64_ng; KP=devk_band_sph; KN=devk_band_sph_ng
RA="--data-root $DATA --out $OUT $YRS"
PROJ=$OUT/proj_$VAL.npz

do_projk() {
    local c k
    for ck in "$CP:$KP" "$CN:$KN"; do
        c=${ck%%:*}; k=${ck#*:}
        [ -f "$OUT/${k}_p.pt" ] && continue
        $FPK projk --data-root "$DATA" --out "$OUT" --core-tag "$c" --dev-tag "$k" --tag "${k}_p" --proj-init "$PROJ" || return 1
    done
}

do_mix() {   # one IC at a time: the float64 statistics of a 1.5 deg IC take about 10 GB of host memory
    $PY scripts/score_mix.py mixk_plus "$CP:$KP" "$CN:$KN" $RA --ic-batch 1 || return 1
    $PY scripts/score_mix.py mixkp_plus "$CP:${KP}_p" "$CN:${KN}_p" $RA --ic-batch 1
}

do_mixrep() {
    export FPK_OUT=$OUT
    local P="$PY scripts/pair_nophys.py"
    $P "score_mixk_plus:$KN" score_mixkp_plus:mixkp_plus "$OUT/pair_mixkp_vs_ng.json" > /dev/null || return 1
    $P "score_mixk_plus:$KP" score_mixkp_plus:mixkp_plus "$OUT/pair_mixkp_vs_phys.json" > /dev/null || return 1
    $P score_mixk_plus:mixk_plus score_mixkp_plus:mixkp_plus "$OUT/pair_mixkp_vs_mixk.json" > /dev/null || return 1
    $P "score_mixk_plus:$KN" score_mixk_plus:mixk_plus "$OUT/pair_mixk_vs_ng.json" > /dev/null || return 1
    $PY scripts/mix_ign.py mixk_plus "$KP" "$KN" "$OUT/mix_ign_mixk.json" > /dev/null || return 1
    $PY scripts/mix_ign.py mixkp_plus "${KP}_p" "${KN}_p" "$OUT/mix_ign_mixkp.json" > /dev/null || return 1
    $PY scripts/mix_report.py "$OUT/mix_plus.md" "score_mixk_plus:$KP=kernel, core with G(x)" \
        "score_mixk_plus:$KN=kernel, core without G(x)" "score_mixkp_plus:${KP}_p=kernel with G(x) + condition" \
        "score_mixkp_plus:${KN}_p=kernel without G(x) + condition" "score_mixk_plus:mixk_plus=mixed kernel" \
        "score_mixkp_plus:mixkp_plus=mixed kernel + condition (the + model)" \
        --pairs "the + model - the kernel without G(x)=$OUT/pair_mixkp_vs_ng.json" \
        "the + model - the kernel with G(x)=$OUT/pair_mixkp_vs_phys.json" \
        "the + model - the mixed kernel=$OUT/pair_mixkp_vs_mixk.json" \
        "the mixed kernel - the kernel without G(x)=$OUT/pair_mixk_vs_ng.json"
}

mix_ready() { [ -f "$OUT/done/devk2" ] && [ -f "$OUT/done/devk2_ng" ]; }

# ---- the 711 stack (the cores of both chains, the conditioned kernels of the mixture's steps)
CD=$OUT/cond
CORES711="$CP,$CN"; [ "$SISTERS" = 1 ] && CORES711="$CP,$CN,${CP}_s1,${CN}_s1"
s711_ready() { [ -f "$OUT/done/projk" ] && { [ "$SISTERS" != 1 ] || { [ -f "$OUT/done/core_s1" ] && [ -f "$OUT/done/core_s1_ng" ]; }; }; }

do_dump() {   # the validation year every 2nd day (the fit, the error kernel's pool), the test year at the scorer's ICs
    local ys y s have
    for ys in "$VAL 2" "$TEST 6"; do
        set -- $ys; y=$1; s=$2
        have=""                                   # meta.json is written last: a finished dump of the same cores
        [ -f "$CD/${y}_meta.json" ] && have=$($PY -c "import json; print(','.join(json.load(open('$CD/${y}_meta.json'))['cores']))")
        if [ "$have" = "$CORES711" ]; then echo "dump $y: there ($have)"; continue; fi
        $PY scripts/diag_cond.py --cores "$CORES711" --year "$y" --ic-stride "$s" --dir "$CD" $RA || return 1
    done
}

do_mcore() {
    local M="--dir $CD --years $VAL $TEST --days 1 3 5 7 10 --lt $MEASURE_LT --npc $MEASURE_NPC --cores $CORES711 --threads $THREADS"
    $PY scripts/measure_core.py --dump mc0 --taus 0 $M || return 1
    $PY scripts/measure_core.py --dump mc3 --taus 3 --lags 1 $M
}

do_cfit() {
    local F="--dir $CD --fit $VAL --test $TEST --days 1 3 5 7 10 --batch 2 --threads $THREADS"
    $PY scripts/cond_fit.py $F --cores "$CP,$CN" --exps bands --json "$OUT/cond_two.json" || return 1
    $PY scripts/cond_fit.py $F --cores "$CORES711" --exps bands vert vert+math --math mc0,mc3 \
        --save-pred cm:vert+math --json "$OUT/cond711.json"
}

do_kmix() {   # host memory: the pool's truth and errors (f16 / f32) and 2 ICs of 6 kernels' members, about 20 GB
    $PY scripts/mix_matrix.py --year "$TEST" --pool "$VAL" --pool-stride 1 --base-core "$CP" \
        --learned "$CP:${KP}_p" "$CN:${KN}_p" --recenter cm --synth sdress45:cm asdress45:cm --dir "$CD" \
        --npz "$OUT/mixmat711_$TEST.npz" --threads "$THREADS" $RA
}

do_rep711() {
    $PY scripts/mix_fixed.py "$OUT/mixmat711_$TEST.npz" --base "sph_p=0.5,sph_ng_p=0.5" --mix \
        "sph_p@cm=0.3,sph_ng_p@cm=0.3,sdress45:cm=0.4" "sph_p@cm=0.3,sph_ng_p@cm=0.3,asdress45:cm=0.4" \
        "sph_p@cm=0.5,sph_ng_p@cm=0.5" > "$OUT/mix_fixed711.log" || return 1
    $PY scripts/report_711.py "$OUT/plus711.md" --two "$OUT/cond_two.json" --four "$OUT/cond711.json" \
        --fixed "$OUT/mix_fixed711.log" --cores "$CORES711" && cat "$OUT/plus711.md"
}

wait_for() {  # wait_for MARKER...: while the chain without G(x) writes its logs (6 hours without a line ends the wait)
    local m ok
    while :; do
        ok=1; for m in "$@"; do [ -f "$OUT/done/$m" ] || ok=0; done
        [ $ok = 1 ] && return 0
        [ -n "$(find "$OUT/logs" -name '*_ng.log' -mmin -360 2>/dev/null)" ] || return 1
        sleep 600
    done
}

do_report() {
    $PY scripts/eval_fpk_vars.py "$OUT/score_plus$SFX" --ref core --arms $K1 $K2 $F --out "$OUT/vars_plus$SFX.md" \
        --json "$OUT/vars_plus$SFX.json" && cat "$OUT/vars_plus$SFX.md" | head -60
}

check_gpu
step data   do_data
step stats  $FPK stats $C
step smoke  do_smoke
step core   $FPK core $C $NG --iters "$CORE_ITERS" --core-ckpt --resume -- $CORE_ARCH --curriculum 1 2 3 4 6 8 10
step devk1  $FPK devk $C --tag $K1 --iters "$DEVK1_ITERS" $DEVK_ARCH $GPU
step spec   $PY scripts/diag_spectra.py $C --dev-tag $K1 --json "$OUT/spectra_$K1.json" --amp
step devk2  $FPK devk $C --tag $K2 --init $K1 $K2_OPTS --iters "$DEVK2_ITERS" --e0-spec "$OUT/spectra_$K1.json" $DEVK_ARCH $GPU
step flow   $FPK devflow $C --tag $F --dev-tag $K2 --iters "$FLOW_ITERS" $FLOW_ARCH $GPU
step score  $FPK score $C --arms persistence core $K1 $K2 $F --score-tag score_plus$SFX
step report do_report
echo "[$(date '+%m-%d %H:%M')] chain finished: $OUT/score_plus$SFX.json, $OUT/vars_plus$SFX.md"
# the sister core (the 711 stack's four-core average): the same recipe, seed 1, its own tag
[ "$SISTERS" = 1 ] && step core_s1 $FPK core --data-root "$DATA" --out "$OUT" $YRS --core-tag "${CORE}_s1" $NG --seed 1 \
    --iters "$CORE_ITERS" --core-ckpt --resume -- $CORE_ARCH --curriculum 1 2 3 4 6 8 10

# ---- the + model: the mixed kernel under the condition (needs the kernels of both chains)
[ "${NOPHYS:-0}" = 1 ] && { echo "the mixture's and the 711 stack's steps run in the chain without NOPHYS (it waits for this one)"; exit 0; }
if ! mix_ready && [ "${DRYRUN:-0}" != 1 ] && [ "${MIX_WAIT:-1}" = 1 ] && [ -f "$OUT/logs/core_ng.log" ]; then
    echo "[$(date '+%m-%d %H:%M')] waiting for the chain without G(x) (done/devk2_ng)"
    wait_for devk2_ng                         # while that chain writes its logs; 6 hours without a line: it stopped
fi
if mix_ready || [ "${DRYRUN:-0}" = 1 ]; then
    step proj   $PY scripts/diag_projection.py stats --cores "$CP,$CN" --year "$VAL" --ic-stride 2 --npz "$PROJ" $RA
    step projk  do_projk
    step mix    do_mix
    step mixrep do_mixrep
    echo "[$(date '+%m-%d %H:%M')] the + model: $OUT/mix_plus.md, $OUT/score_mixkp_plus.json"
    # ---- the 711 stack: four cores, the measure continuation, the conditioned mean, the kernels on it
    if [ "$SISTERS" = 1 ] && ! s711_ready && [ "${DRYRUN:-0}" != 1 ] && [ "${MIX_WAIT:-1}" = 1 ]; then
        echo "[$(date '+%m-%d %H:%M')] waiting for the sister core of the chain without G(x) (done/core_s1_ng)"
        wait_for core_s1_ng
    fi
    if s711_ready || [ "${DRYRUN:-0}" = 1 ]; then
        step arch   $PY scripts/plus_archive.py --dir "$CD" --lt "$MEASURE_LT" $RA
        step dump   do_dump
        step mcore  do_mcore
        step cfit   do_cfit
        step kmix   do_kmix
        step rep711 do_rep711
        echo "[$(date '+%m-%d %H:%M')] the 711 stack: $OUT/plus711.md"
    else
        echo "the 711 stack needs the sister core of the chain without G(x) (done/core_s1_ng): when it is done, start"
        echo "this chain again (the finished steps are skipped), or SISTERS=0 for the stack on the two cores"
    fi
else
    echo "the + model (the mixed kernel) needs the kernel of the chain without G(x) too:"
    echo "  NOPHYS=1 nohup bash scripts/gpu_free.sh bash scripts/run_fpk_plus.sh > $OUT/chain_ng.log 2>&1 &"
    echo "and, when it is done, this chain again (the finished steps are skipped)."
fi
