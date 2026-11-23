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
#   NOPHYS=1 bash scripts/run_fpk_plus.sh                 # the control without G(x): the same chain, tags *_ng
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
    echo "== smoke: GPU memory (GB) and seconds of the logged iterations"
    grep -ho '"gpu_gb": [0-9.]*' "$OUT/logs/smoke$SFX.log" | sort -t: -k2 -n | tail -1 || true   # none on the CPU
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
