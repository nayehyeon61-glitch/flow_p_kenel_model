# 방정식 → 분포 → 10일 ensemble: 물리 kernel 을 source 로 하는 flow matching (2026-11-20)

## 요청을 이렇게 읽었습니다

소유자 요청(2026-11-20): "physical_guide_flowmatching 에서 tendency core 를 확인하고, 원시식을 kernel 형태의 distribution 으로 만들고 flow matching 에 넣어서 ensemble 예측 구조를 만들어 10일 예측 구조를 만들고, 이를 통해서 물리 분포를 통한 예측 구조를 만든다." 프로젝트 설명: "G(x) → Σ K(x,y) 의 K 를 통해서 distribution 을 구성해서, flow matching 의 distribution 을 통해서 G(x) → K(x,y) → 예측(10일)에 대한 entropy 예측 구조." 메인 코드는 이 저장소입니다.

kernel 의 구체적인 식은 정해져 있지 않아서 아래 해석을 택했습니다. 다른 해석으로 바꾸는 데 필요한 자리는 4절에 적었습니다.

## 1. 출발점: tendency core

`physical_guide_flowmatching` 의 최신 작업 브랜치(`claude/project-thread-c97r6n`, commit 5d38e56)를 확인했습니다. 이 저장소의 `third_party/pgfm` 에 그 코드가 그대로 들어 있어 거기서 학습한 core 가 여기서 그대로 로드됩니다.

    dx/dt = f_θ(x, G(x), static, 시각)          x: 66개 동적 채널 (KAI 계약, 정규화)
    G(x)  = −R0(x)                               이산 원시방정식 연산자가 낸 u, v, T, q × 12층의 경향 = 48 물리항
    Φ(x)  = Heun 한 step = 하루                  (tisr 강제, PositiveMass)

`docs/kai_tendency.md` 5절의 core_gng 가 이 형태입니다(물리항이 1일 nRMSE 를 5 % 줄임). 기억 core(core_gngmz8)는 member 마다 기억 상태를 따로 가져야 해서 이번 첫 구현에서는 plain core 만 받습니다(4절).

## 2. 구조

### 2.1 방정식 → 분포: 물리 kernel

    K(x, y) = (1/N) Σ_k  N( y ; Φ(x + ε ⊙ s ⊙ δ_k) , diag (s ⊙ h(x))² )

| 기호 | 뜻 | 학습 |
| --- | --- | --- |
| Φ | tendency core 의 하루 (G(x) 포함) | 얼림 |
| δ_k | 3 × 3 격자로 매끄럽게 한 단위 noise, ±δ 쌍 (중심들의 평균이 Φ(x) 근처에 남음) | |
| ε | 채널별, 방정식을 x 에서 얼마나 떨어진 점들에서 출발시키는가 (분석 불확실성) | NLL |
| h(x) | 채널 × 격자점별 폭. 입력 [x, G(x), (Φ(x) − x)/s, 정적 장, 시각] | NLL |
| s | 채널별 하루 변화의 표준편차 (`core.t6`). 모든 분포의 단위 | |

- **Σ K(x,y) 의 뜻.** 방정식이 x 주변의 점 구름을 하루 적분하고, 그 도착점마다 kernel 을 놓아 더한 것(kernel density)이 다음 날의 분포입니다. 방정식이 구름을 늘리는 방향(불안정한 곳)으로 분포가 넓어지므로 폭이 흐름에 따라 달라집니다. 방정식이 정하지 못하는 나머지(closure 의 오차, 격자 아래의 운동)는 h(x) 가 맡습니다.
- **학습.** 관측된 다음 날 y 의 음의 로그 우도. 66 × H × W 차원의 혼합 우도는 가장 가까운 중심 하나로 쏠리므로, 채널 × 격자점별 주변 우도의 합(composite likelihood)을 씁니다. 표본은 장 전체에서 중심 하나를 고르고 매끄러운 noise 를 더해 뽑으므로, 주변 분포는 학습한 그대로입니다.
- **entropy.** 한 값당 Gaussian(모멘트) entropy, ½ log(2πe · (중심의 분산 + h²)), s 단위의 nats. kernel 합 자체의 주변 entropy 의 상한이고, 방정식이 하루 동안 만드는 entropy 생산입니다.

### 2.2 분포 → 분포: kernel 을 source 로 하는 flow matching

증분 단위 z = (y − x)/s 에서

    z0 ~ K(x, ·) 를 증분으로       (source: 물리 kernel 의 표본)
    z1 = 관측된 다음 날의 증분      (target)
    z_τ = (1 − τ) z0 + τ z1 + σ_min n,     u = z1 − z0
    loss = || v_θ(z_τ, τ | x, G(x), (Φ(x) − x)/s, log h(x), 정적 장, 시각) − u ||²   (면적 가중, tisr 제외)

- source 와 target 은 x 가 주어졌을 때 독립으로 뽑으므로, 학습된 흐름은 K(x, ·) 를 다음 날의 조건부 분포로 옮깁니다(conditional flow matching 의 주변 보존 성질).
- 속도 head 가 0 에서 시작하므로 **학습 전의 모델은 물리 kernel 그 자체**입니다. 흐름은 kernel 이 틀린 만큼(편향, 모양)만 배우고, 옮기는 거리가 짧아 적분 step 이 적어도 됩니다(Heun 6 step).
- 대조군 `--source gauss`: 같은 망, 같은 조건(G(x), Φ(x) 포함)에 source 만 x 주변의 매끄러운 N(0, I)(s 단위)로 바꾼 표준 flow matching. 물리가 "조건"으로만 들어갈 때와 "분포"로 들어갈 때를 가릅니다.

### 2.3 10일 ensemble 과 entropy 예측

member 마다 자기 궤적을 갑니다: 매일 K(x_t^m, ·) 에서 뽑고(새 중심과 새 폭 noise), 흐름으로 옮겨 다음 날 상태로 삼습니다. 퍼짐은 방정식(섭동된 점들의 적분)과 학습된 폭, 흐름을 지나 자랍니다.

| 양 | 식 (한 값당, s 단위 nats) | 뜻 |
| --- | --- | --- |
| H_ens(L) | ½ log(2πe · var_m(x_L^m)/s²) | 선행 L 일 예보 분포의 entropy (예측된 entropy) |
| H_K(L) | member 평균의 kernel entropy | L 일째에 방정식이 만드는 하루치 entropy 생산 |
| IGN(L) | ½ log(2π var/s²) + e²/(2 var), e = 앙상블 평균 − 참값 | 참값의 Gaussian log score |

- 보정이 맞으면 E[IGN] = H_ens 입니다(entropy = 참값이 주는 놀람의 기댓값). IGN − H_ens > 0 이면 ensemble 이 지나치게 확신합니다.
- IC 마다 H_ens 와 ½ log(e²/s²) 의 상관은 예측된 entropy 가 좋은 예보와 나쁜 예보를 가려내는지를 봅니다.
- 함께 내는 점수: 앙상블 평균 nRMSE(65채널, 정규화 RMSE 평균), fair CRPS, spread/skill, 주요 변수(z500, t850, t2m, u850, v850, q700, msl, u250)의 물리 단위 RMSE.

## 3. 명령

```bash
pip install -r requirements.txt
python third_party/pgfm/scripts/wb2_to_kai.py --out data/era5_5p625 --daily-mean --no-climode --years 2014 2015 2016 2017
python scripts/run_fpk.py core --iters 3000                       # core_gng: GroupNorm, G(x), rollout 1~4일
python scripts/run_fpk.py kernel --iters 600 --batch 4 --centres 4
python scripts/run_fpk.py flow --source kernel --tag flow
python scripts/run_fpk.py flow --source gauss --tag flow_gauss
python scripts/run_fpk.py score --arms persistence core kernel flow flow_gauss --kernel-entropy
python -m pytest -q tests
```

PG-CFM 에서 이미 학습한 core 를 쓰려면 `runs/kaitend/<tag>.pt` 와 같은 폴더의 `stats.npz` 를 `--out` 폴더에 두고 `--core-tag <tag>` 를 줍니다(plain core 만).

## 4. 정한 가정과 바꿀 수 있는 자리

- **kernel 의 형태.** 중심 = 섭동된 점들의 방정식 적분, 폭 = 학습. 다른 해석(예: Green 함수 형태의 적분 연산자 ∫K(x,y)φ(y)dy, 또는 G(x) 의 물리항별 불확실성을 직접 공분산으로 쓰는 형태)은 `fpk/kernel.py` 의 `centres` / `bandwidth` 만 바꾸면 됩니다.
- **core 는 얼림.** 방정식 부분은 PG-CFM 에서 학습한 그대로 두고 분포의 부분만 학습했습니다.
- **기억 core 미지원.** member 마다 ConvGRU 기억을 복제해 굴리는 것이 다음 단계입니다.
- **entropy 는 Gaussian 모멘트 근사.** 앙상블 8 member 로는 비모수 entropy 를 추정할 수 없습니다.
- **CPU 크기.** 5.625°, 학습 2년, 시험 2017년 IC 6일 간격. GPU 에서는 같은 명령에 `--device cuda` 와 긴 학습을 줍니다.
