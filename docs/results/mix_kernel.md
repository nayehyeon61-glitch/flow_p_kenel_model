# The mixed kernel and the invariant-measure condition (2017, 2026-11-24)

감자(11/24 17:35 KST): "지금 딱 한가지만 더하면 … 700대까지 갈것 같습니다. 물리적 그리고 함수적 조건에서 만족할만한 기술이
있을까요?" → 17:49 "이거 한번 먼저 진행해 보죠" (the mixed kernel) → 17:59 "마지막 한방으로 생각하는것이 있나요?" (the
invariant-measure condition) → 18:03 "conditional 한 flow 로 가자는 거군요 좋습니다."

Everything below is on the existing 8-year kernels and the 16-year cores, no new training: 2017, the scorer's 60 ICs,
IC-paired bootstrap 2000. Leads 1 / 3 / 5 / 7 / 10 days.

## 1. The mixed kernel  K = ½ K_W + ½ K_NW

A convex combination of two Markov kernels is a Markov kernel (the Bayesian average of two models). Here the kernel on
the physics core (`core_gngmz8W` → `devk_band_sphW`) and the one on the core without physics (`core_ngmz8W` →
`devk_band_sphNW`). Its mean is the average of the two means; its variance ½ (v_W + v_NW) + ¼ (m_W − m_NW)²: the
second term is the spread BETWEEN the cores, the uncertainty of the model itself, which neither kernel has.
`scripts/score_mix.py mixk_W_NW core_gngmz8W:devk_band_sphW core_ngmz8W:devk_band_sphNW` (`runs/fpk/score_mixk_W_NW*`,
pairs `mixk_W_NW_vs_W.json`, `mixk_W_NW_vs_NW.json`, IGN `mixk_W_NW_ign.json`).

| | 1 | 3 | 5 | 7 | 10 |
| --- | --- | --- | --- | --- | --- |
| z500 RMSE, physics kernel | 164.6 | 409.4 | 577.5 | 675.9 | 736.2 |
| z500 RMSE, no-physics kernel | 164.5 | 400.1 | 567.5 | 670.1 | 733.4 |
| **z500 RMSE, mixed kernel** | **155.1** | **390.7** | **557.3** | **659.0** | **722.5** |
| nRMSE gain % vs physics (65 ch) | **+1.97** | **+3.21** | **+2.89** | **+2.54** | **+2.42** [+2.15, +2.71] |
| nRMSE gain % vs no physics | **+5.29** | **+3.13** | **+2.40** | **+2.05** | **+1.94** [+1.64, +2.23] |
| CRPS physics / no physics | 0.0970 / 0.1004 | 0.1686 / 0.1693 | 0.2066 / 0.2061 | 0.2283 / 0.2277 | 0.2438 / 0.2429 |
| **CRPS mixed** | **0.0949** | **0.1631** | **0.2001** | **0.2218** | **0.2369** |
| CRPS diff vs physics | **−0.0021** | **−0.0055** | **−0.0065** | **−0.0065** | **−0.0069** [−0.0076, −0.0063] |
| CRPS diff vs no physics | **−0.0055** | **−0.0062** | **−0.0060** | **−0.0059** | **−0.0060** [−0.0066, −0.0053] |
| IGN diff vs physics (8 members) | −0.002 | **−0.011** | **−0.020** | **−0.024** | **−0.036** [−0.040, −0.031] |
| IGN diff vs physics (16 members) | **−0.022** | **−0.030** | **−0.037** | **−0.039** | **−0.049** [−0.054, −0.045] |
| spread/skill (single 1.01 … 0.99) | 1.095 | 1.059 | 1.033 | 1.022 | 1.033 |
| share of the between-core variance, z500 | 0.10 | 0.07 | 0.06 | 0.05 | 0.04 |

Bold: the interval excludes 0. The mean is exact (a kernel's mean is known), so z500 722.5 is the mixture's own mean, not
an 8-member one. Day-10 CRPS 0.2369 is the lowest of every kernel and flow so far (best flow: 0.2397).

Why it works and where it stops: the two cores' day-10 z500 errors correlate 0.93 (anomalies 0.79), so averaging two
removes ½ (1 − 0.93) of the error variance (735 → 722). More cores of the same kind: 4 ≈ 716, the floor ≈ 709. Lagged
averages of one core give nothing (735 → 736).

### Two traps of scoring a mixture (both measured, both fixed in the scripts)

- **The same seed on both sides.** The scorer's seed gives both kernels the same noise draws; their members are then
  correlated pair by pair, the cross term of the CRPS is too small and the count of the variance too large. Measured:
  the half mixture's CRPS 0.0014 above the full one's (the same distribution). `score_mix.py` rolls side B with seed + 1;
  full and half then agree to 0.0001.
- **IGN depends on the member count.** The scorer's 0.5 log v̂ is biased by ψ(P/2) − log(P/2): −0.135 at P = 4 pairs,
  −0.065 at 8; and the mixture's variance has an exact part (the between-core term) that cuts the tail of 1/v̂, so the
  scorer's (P − 2)/P over-corrects. `scripts/mix_ign.py` removes both by quadrature over χ²_k. The corrected IGN of the
  full (16 members) and half (8) mixtures still differ by 0.014–0.020, so the IGN gain is given as that range, not a
  number. (The corrected IGN of the single arms: physics 0.462 / 1.036 / 1.243 / 1.354 / 1.439, no physics
  0.498 / 1.058 / 1.256 / 1.366 / 1.441; the mixture 0.440–0.460 / 1.006–1.025 / 1.206–1.223 / 1.315–1.329 / 1.390–1.404.)

### The same mixture of the two G-flows (`devflow_gW`, `devflow_gNW`)

`scripts/score_mix.py mixg_W_NW core_gngmz8W:devflow_gW core_ngmz8W:devflow_gNW --members 4`: 4 + 4 independent members,
the scorer's 8 (`runs/fpk/score_mixg_W_NW*`). The 8-member mean carries the members' noise (z500 day 10: 776), so the
mean is given as the estimate for infinitely many members, mse − (the mean's sampling variance).

| | 1 | 3 | 5 | 7 | 10 |
| --- | --- | --- | --- | --- | --- |
| CRPS, G-flow physics / no physics (8 members) | 0.0967 / 0.0995 | 0.1687 / 0.1681 | 0.2052 / 0.2046 | 0.2266 / 0.2257 | 0.2413 / 0.2397 |
| **CRPS, mixed G-flow** | **0.0948** | **0.1634** | **0.2006** | **0.2218** | **0.2362** |
| CRPS diff vs the no-physics G-flow | **−0.0047** | **−0.0047** | **−0.0040** | **−0.0039** | **−0.0035** [−0.0052, −0.0019] |
| CRPS diff vs the mixed kernel | −0.0001 | +0.0003 | +0.0005 | 0.0000 | −0.0007 [−0.0017, +0.0003] |
| z500 RMSE∞, mixed G-flow | 150.6 | 387.0 | 557.9 | 656.4 | 715.0 [695, 734] |
| − the mixed kernel's mean | **−4.5** | −3.7 | +0.6 | −2.6 | −7.5 [−18.3, +2.7] |
| spread/skill (the no-physics G-flow, 8 members: 1.14 … 1.11) | 1.21 | 1.21 | 1.17 | 1.15 | 1.14 |

The mixture helps the flows as it helps the kernels (CRPS −0.004 to −0.005 against either flow, every lead), and the
mixed flow and the mixed kernel then have the same CRPS. The flows' mean of infinitely many members is 715 at day 10,
7.5 below the mixed kernel's, not significant (day 1: −4.5, significant: the G-flow's own gain). The flows were already
too wide and the mixture is wider (z500 spread/skill 1.18 at day 10): the between-core variance is added to a spread
that was fitted without it. IGN by the scorer's formula: −0.019 to −0.037 against the no-physics G-flow and −0.008 to
−0.017 against the physics one (not corrected for the member count as the kernels' above: a range, not a number).

## 2. The invariant-measure condition  μ K = μ

The climate pushed through the kernel must be the climate again (the atmosphere is statistically stationary; the
invariant measure of the Markov operator). In second moments this is the law of total variance,
(variance of the mean) + (spread²) = (variance of the climate). Together with the calibration the kernel already learns
(spread² = error²) it gives ⟨f (o − f)⟩ = 0, f = mean − climatology, o = truth − climatology: the error is orthogonal to
the forecast, i.e. the mean is the conditional expectation (the L² projection), scale by scale. In flow-matching terms:
the climate is the unconditional distribution, the kernel the conditional one, and the condition is the marginalization
identity; where the IC carries no information the mean must go back to the climate.

The mean with a strength g of the condition,  m = climatology + Σ_b g_b (x̄ − climatology)_b  (b: the Littlewood–Paley
bands of the degree, l = 0, 1, 2–3, 4–7, 8–15, 16–31; g per lead, channel and band), `scripts/diag_projection.py`: the
exact grid statistics of the two cores' trajectories (the kernels' means: z500 within 1 m²/s² of their cores'), g
fitted by least squares on **2016** (178 ICs) and scored on **2017** (`projection.json`; a 2-fold fit inside 2017 gives
the same numbers to 0.5).

| z500 RMSE | 1 | 3 | 5 | 7 | 10 | Δ day 10 vs the mean of the cores |
| --- | --- | --- | --- | --- | --- | --- |
| no-physics core | 164.5 | 399.6 | 568.2 | 670.7 | 733.7 | |
| the same + g per band | 163.8 | 397.3 | 564.3 | 663.9 | 723.5 | |
| mean of the two cores (the mixed kernel's mean) | 155.2 | 390.4 | 556.5 | 658.0 | 722.0 | |
| + one g per lead and channel | 155.2 | 389.9 | 556.5 | 657.3 | 718.4 | −3.6 [−5.7, −1.5] |
| **+ g per band** | **154.8** | **388.9** | **554.6** | **655.7** | **717.3** | **−4.7 [−6.6, −2.8]** |
| + a weight per core and band | 154.8 | 388.3 | 554.3 | 655.4 | 717.4 | −4.6 [−6.6, −2.7] |

| nRMSE (65 ch) | 1 | 3 | 5 | 7 | 10 |
| --- | --- | --- | --- | --- | --- |
| no-physics core | 0.1942 | 0.3288 | 0.4050 | 0.4484 | 0.4767 |
| mean of the two cores | 0.1840 | 0.3185 | 0.3946 | 0.4385 | 0.4669 |
| + g per band | 0.1827 | 0.3147 | 0.3890 | 0.4313 | 0.4566 |

The fitted strength (2016) of the mean of the cores, day 10: z500 0.86 (bands l = 1: 0.84, 2–3: 0.87, 4–7: 0.92,
8–15: 0.78, 16–31: 0.21), levels' mean z 0.85, q 0.71, u 0.70, T 0.66, v 0.62, msl 0.72, t2m 0.49. So z500 was close to
its projection and the temperatures were not: day 10, mean of the cores → + g per band, against the climatology:

| day 10 RMSE | no-physics core | mean of the cores | + g per band | climatology |
| --- | --- | --- | --- | --- |
| z500 (m²/s²) | 733.7 | 722.0 | 717.3 | 789.2 |
| t850 (K) | 3.221 | 3.140 | 2.972 | 3.151 |
| t2m (K) | 2.419 | 2.361 | 2.145 | 2.282 |
| u850 (m/s) | 4.841 | 4.763 | 4.653 | 4.924 |
| msl (Pa) | 656.6 | 646.7 | 637.0 | 677.9 |

The t2m and t850 means of day 10 were WORSE than the climatology; with the condition they are below it.

(My first reading, "the z500 mean moves 40 % more than is predictable", from activity 0.535 against ACC 0.381, was
wrong: those are means of per-IC ratios. The fitted strength is 0.86.)

## Together

| day 10 | z500 | nRMSE |
| --- | --- | --- |
| the current kernel (no physics) | 733.4 | 0.4766 |
| mixed kernel | 722.5 | 0.4673 |
| mixed kernel + the condition (mean; cores' trajectories) | 717.3 | 0.4566 |

710 is 7 below: the correlation term (95 % of the error) only falls with more independent information, i.e. more cores
in the mixture (estimate: 4 cores + the condition ≈ 711).
