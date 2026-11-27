# G-flow on the kernel WITHOUT physics (2017, 2026-11-24)

감자 질문(11/24 08:30 KST): "무물리 kernel 에 G-flow 를 더하면 z500 733 보다 줄 수 있지 않을까?" → "한번 수행해 보죠".

`devflow_gNW`: the no-physics core and kernel (`core_ngmz8W` → `devk_band_sphNW`, no G(x) anywhere) with the flow of
`devflow_gW` (`--gmember --gdirect`, G(x̄) in the conditioning), the operator G(x) read from the physics core
(`--g-core core_gngmz8W`: the operator and its ERA5 scale, none of that core's weights; `fpk/devflow.py OuterG`).
Same years (2008–2015), 600 iterations, options. 2017, the same 60 IC × 8 members, IC-paired bootstrap 2000
(`scripts/pair_nophys.py`: `gflowNW_vs_lineNW.json`, `gflowNW_vs_gW.json`, `gflowNW_vs_lineW.json`), per-variable table
`gflowNW_vars.md` (500 draws). The run was cut at 450 by a machine restart and run again from the start (same seed,
identical numbers).

## Against the no-physics FM (`devflow_lineNW`; leads 1 / 3 / 5 / 7 / 10)

| | 1 | 3 | 5 | 7 | 10 |
| --- | --- | --- | --- | --- | --- |
| nRMSE gain % (65 ch) | **+1.12** | **+0.62** | **+0.32** | +0.14 [−0.01, +0.30] | **+0.15** [+0.01, +0.29] |
| CRPS diff | **−0.0012** | **−0.0015** | **−0.0013** | **−0.0012** | **−0.0013** [−0.0016, −0.0009] |
| IGN diff | **−0.020** | **−0.020** | **−0.019** | **−0.019** | **−0.019** [−0.020, −0.018] |
| z500 RMSE (8-member mean) | 181 → 172 | 447 → 436 | 627 → 615 | 735 → 724 | 798 → 786 (+1.5 % [+1.1, +2.0]) |
| spread/skill | 1.14 / 1.14 | 1.15 / 1.15 | 1.14 / 1.14 | 1.12 / 1.12 | 1.11 / 1.11 |

Bold: the interval excludes 0. Every score improves at every lead.

## Below 733? The mean of infinitely many members

E[mse of the M-member mean] = MSE∞ + E[var]/M (independent members, `var` the unbiased member variance of the scorer),
so per IC MSE∞ ≈ mse − var/8. z500 RMSE∞ (m²/s²) against the arm's own kernel (= its core's trajectory), bootstrap over
the ICs:

| | 1 | 3 | 5 | 7 | 10 | − own kernel, day 10 [95 %] |
| --- | --- | --- | --- | --- | --- | --- |
| no-physics kernel | 164.5 | 400.1 | 567.5 | 670.1 | 733.4 | |
| no-physics FM | 165.3 | 404.2 | 571.6 | 675.3 | 734.6 | +1.2 [−9.8, +12.1] |
| **no-physics kernel + G-flow** | **157.2** | 395.5 | 562.3 | 666.6 | **725.1** | −8.3 [−19.4, +2.8] |
| physics G-flow | 157.3 | 410.5 | 572.1 | 672.8 | 734.1 | −2.1 [−13.5, +8.7] (vs 736.2) |
| current FM | 158.5 | 409.7 | 571.0 | 670.6 | 729.8 | −6.4 [−18.1, +5.0] (vs 736.2) |

The G-flow on the no-physics kernel moves the mean: day 1 −7.2 [−9.2, −5.3] (significant), day 10 725 against 733
(lower, not significant). The no-physics FM does not move it (+1.2). On the physics kernel the same G-flow does not move
it either (734 against 736): there the core already read G, here G is new information and the flow uses it for the mean.

## Against the G-flow on the physics kernel (`devflow_gW`)

| | 1 | 3 | 5 | 7 | 10 |
| --- | --- | --- | --- | --- | --- |
| nRMSE gain % | **−2.64** | **+0.85** | +0.40 | +0.42 | +0.77 [−0.01, +1.53] |
| CRPS diff (no-physics − physics) | **+0.0027** | −0.0006 | −0.0006 | −0.0008 | −0.0016 [−0.0033, +0.0001] |
| IGN diff | **+0.030** | **+0.011** | **+0.010** | **+0.008** | +0.002 [−0.003, +0.007] |
| CRPS | 0.0995 / 0.0967 | 0.1681 / 0.1687 | 0.2046 / 0.2052 | 0.2257 / 0.2266 | 0.2397 / 0.2413 |
| IGN | 0.426 / 0.397 | 0.981 / 0.970 | 1.169 / 1.159 | 1.267 / 1.259 | 1.335 / 1.334 |

## Reading

- G given only to the flow, member by member, on a stack that never saw G: every score improves at every lead, and at
  day 10 the IGN of the full physics stack is reached (1.335 against 1.334). The day-10 CRPS (0.2397) and z500 RMSE
  (786) are the lowest of all flows so far.
- Days 1–7: the physics core and kernel still add sharpness (IGN 0.008–0.030) and the day-1 accuracy (the core,
  nRMSE 2.6 %).
- So G(x) helps where it is NEW information. In the physics stack the core already read G and the flow's dG adds
  only sharpness (IGN 1/3 of the gain); in the no-physics stack the flow's dG also corrects the mean.
- My prediction before the run (the flow cannot move the 3–10 day mean) was wrong for the no-physics stack.
