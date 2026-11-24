# The kernelized equation in the kernel, no-physics core (2017, 2026-11-24)

감자 (11/24 11:27 KST): "tendency core 가 식을 넣고 학습한 것처럼 이번에는 kernel 화 한 식이 학습되도록". The core is the
one WITHOUT G(x) (`core_ngmz8W`); the kernel reads the operator G(x) of `core_gngmz8W` (operator and ERA5 scale only,
`--eq-core`) in its distributional form (`fpk/devkernel.py` EqOperator, `eq_kernel`): every day the operator's own day
of the deviation pairs p_G = [G(xbar + s e) - G(xbar - s e)] / 2s, its entropy production and growth rate as inputs of
the entropy and correction nets, and + A (.) p_G in the direction (A from zero, step 20 x lr). The equation is fixed;
the nets and A learn from the log score, as the core learned on G(x). Same two stages, options and iterations as the
kernel without it (`devk_meqNW` 800 -> `devk_band_sphEqNW` 400, as `devk_mNW` -> `devk_band_sphNW`); 2017, 60 IC x 8
members, IC-paired bootstrap 2000 (`scripts/pair_nophys.py`: `eqk_vs_NW.json`, `eqk_vs_W.json`).

## Against the kernel without the equation (`devk_band_sphNW`; leads 1 / 3 / 5 / 7 / 10; equation - without)

| | 1 | 3 | 5 | 7 | 10 |
| --- | --- | --- | --- | --- | --- |
| nRMSE gain % | 0.00 | -0.01 | 0.00 | +0.01 | +0.01 |
| CRPS diff | 0.0000 | **+0.0001** | +0.0001 | +0.0001 | **+0.0002** |
| IGN diff | +0.0008 | -0.0010 | **-0.0043** | **-0.0058** | -0.0049 [-0.0092, +0.0002] |
| predicted entropy H diff | **+0.002** | **+0.004** | **+0.010** | **+0.011** | **+0.008** |
| spread / skill | 1.019 / 1.019 | 0.993 / 0.997 | 0.977 / 0.982 | 0.973 / 0.977 | 0.987 / 0.986 |
| z500 RMSE (kernel mean) | 164.5 / 164.4 | 400.1 / 400.1 | 567.5 / 567.3 | 670.1 / 670.1 | 733.4 / 734.0 |

Bold: the interval excludes 0. The mean is the core's (unchanged). The equation kernel's members are a little wider
(H +0.01) where the kernel without it is a little under-dispersed (spread/skill 0.97-0.98), and its IGN is lower at
days 5-7 (significant) and 10 (borderline): better calibrated, not sharper. CRPS +0.0001-0.0002 (worse, tiny).

## Against the physics kernel (`devk_band_sphW`: G(x) in the core and the kernel)

| IGN diff (no-physics - physics) | 1 | 3 | 5 | 7 | 10 |
| --- | --- | --- | --- | --- | --- |
| kernel without the equation (kernel_nophys.md) | **+0.036** | **+0.022** | **+0.013** | **+0.013** | +0.001 |
| **equation kernel** | **+0.037** | **+0.021** | **+0.009** | +0.007 [-0.000, +0.014] | -0.004 [-0.011, +0.004] |
| H diff (equation kernel - physics) | **+0.040** | **+0.022** | **+0.025** | **+0.026** | **+0.024** |

The equation kernel closes about a third of the physics kernel's IGN lead at day 5 and half at day 7; days 1-3 do not
move (the core's: G(x) at every step of the integration, the day-1 nRMSE 3.5 %). The physics stack is sharper (H 0.02-0.04
lower); the equation kernel does not reproduce that, it gains through calibration.

## A and the equation's share

End of stage 1: A up to 0.35 in size, mixed signs (T -0.28 ... +0.29), q +0.26 ... +0.35 at 400-925 hPa. End of stage 2
(band scale, correction net, sphere noise): u, v, T back near 0 (|A| <= 0.08), q stays +0.15 ... +0.22 at 400-925 hPa;
the equation's entropy production 0.02-0.04 a day (stage 1: 0.09-0.20). The kernel keeps the equation's moisture
direction in the mid and lower troposphere and reads the rest through the nets' inputs.

## Next (감자 14:21 KST: "검사 끝나면 바로 kernel fm 함께 학습으로")

`runs/fpk/chain_eqk2.sh`: the kernel and the G-flow trained together (`devtune --kernel-lr 1e-5`, A at 20 x), starting
from `devflow_gNW` put on this kernel (`devflow_geqNW0`) -> `devflow_geqNW_j` / `devk_band_sphEqNW_j`; control: the same
on the kernel without the equation -> `devflow_gNW_j` / `devk_band_sphNW_j`. Both start from the same flow.
