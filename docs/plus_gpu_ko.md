# fpk + 버전: 1.5° · ERA5 32년 · GPU (본 실험)

5.625° 에서 가장 좋았던 구성을 1.5° 격자(121 × 240, 극 포함)와 32년 자료로 키운 버전입니다.
**+ 모델은 섞은 kernel 에 불변 측도 조건을 건 것입니다** (5.625° 결과: `docs/results/mix_kernel.md`).

    사슬 A (G(x) 물리장 입력)              사슬 B (G(x) 없음, NOPHYS=1)
    core_u64                              core_u64_ng
      → kernel 1단계 (devk_m)               → devk_m_ng
      → 1.5° 오차 스펙트럼                   → 스펙트럼
      → kernel 2단계 (devk_band_sph)        → devk_band_sph_ng
      → flow matching (devflow_line)        → devflow_line_ng
                     ↘                     ↙
       불변 측도 조건의 세기 (두 core 의 2016 궤적, 예측일·변수·파수 띠별)
       → 각 kernel 의 예측을 조건으로 읽기 (devk_band_sph_p, devk_band_sph_ng_p)
       → 두 kernel 을 반씩 섞은 분포 = + 모델 (2017, 60 IC 채점)

- **섞은 kernel**: K = ½ K_A + ½ K_B. 두 Markov kernel 의 볼록 결합도 Markov kernel 이고, 분산에 두 core 사이의 차이
  (모형 자체의 불확실성)가 더해집니다.
- **불변 측도 조건** (μK = μ): 기후 분포를 kernel 에 통과시키면 다시 기후 분포여야 합니다. 보정 조건과 합치면 평균이
  조건부 기대값이 되어야 한다는 조건이 되고, 예측 평균 = 기후값 + g ⊙ (평균 − 기후값) 으로 읽습니다. g 는 예측일·변수·
  파수 띠마다 하나이고, 학습에 쓰지 않은 검증 해(2016)에서 맞춥니다. 퍼짐도 남은 오차에 맞춰 줄입니다. 사슬 자체(core
  궤적, 편차)는 바뀌지 않습니다.
- 5.625° (16년 core, 2017): 10일 CRPS 0.2429 → 0.2298 (−5.4 %), z500 RMSE 733 → 718, nRMSE 0.477 → 0.458, 퍼짐/오차 1.00.

- 학습 1984–2015 (32년), 검증 2016, 시험 2017 (5.625° 실험과 같은 검증·시험 해)
- 대상: 서버 terra 의 RTX PRO 6000 Blackwell (96 GB). 사슬 두 개라 GPU 두 장이면 동시에, 한 장이면 차례로 돌립니다.
- 사슬 스크립트: `scripts/run_fpk_plus.sh`. 단계마다 완료 표시를 남기고, 학습 단계는 끊겨도 이어서 돕니다.

## 1. 코드 받기와 환경 (처음 한 번)

```bash
cd /lustre/home/yehyeon
git clone -b claude/project-thread-c39mah https://github.com/nayehyeon61-glitch/flow_p_kenel_model.git
cd flow_p_kenel_model
# 이미 받아 두었다면: git fetch origin && git checkout claude/project-thread-c39mah && git pull

python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128      # Blackwell 은 CUDA 12.8 빌드
pip install -r requirements.txt

# 확인: GPU 가 보이는지, 시험이 통과하는지
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m pytest -q tests/test_plus.py
GPU_LIST=1 bash scripts/gpu_free.sh          # GPU 별 사용 상태
```

## 2. 자료 (CPU, 약 5–6시간, 디스크 약 97 GB)

lustre 의 0.25° 일평균(`/lustre/home/mahmed/ERA5_0p25_DAILY`)을 1.5° 로 보존 재격자화해서
`data/era5_1p5/kaiday/<연도>.npy` 로 만듭니다. 이미 있는 연도는 건너뜁니다. 먼저 첫 날 파일을 검사하고, 문제가 없으면 변환합니다.

```bash
source .venv/bin/activate
mkdir -p runs/fpk_plus
# 자료와 통계만 먼저, GPU 없이 (GPU 를 잡지 않음)
STEPS="data stats" DEV=cpu WORKERS=8 nohup bash scripts/run_fpk_plus.sh > runs/fpk_plus/chain_data.log 2>&1 &
tail -f runs/fpk_plus/chain_data.log
```

- 이미 1.5° 로 변환한 연도가 다른 폴더에 있으면 `DATA=<그 폴더>` 로 지정합니다. 그 폴더에는 `kaiday/<연도>.npy` 와
  `era5_data/constants/constants_1.5deg.nc` 가 있어야 합니다. 빠진 연도만 그 폴더에 새로 만듭니다.
- 원 자료가 다른 곳에 있으면 `SRC=<경로>` 로 지정합니다.
- `stats` 단계는 32년을 메모리에 올리지 않고 훑어서 `runs/fpk_plus/stats.npz` 를 만듭니다. 시간은 약 30–60분입니다.
  이 파일은 이 격자 전용입니다. 5.625° 의 `runs/fpk` 와 섞이지 않게 `--out` 을 따로 쓰고, 다른 격자의 파일이면 거부합니다.

## 3. 학습과 평가 (GPU)

두 사슬을 띄웁니다. 자료와 `stats.npz` 는 함께 씁니다.

```bash
source .venv/bin/activate
# GPU 두 장: 동시에
NOPHYS=1 nohup bash scripts/gpu_free.sh bash scripts/run_fpk_plus.sh > runs/fpk_plus/chain_ng.log 2>&1 &
sleep 120   # 사슬 B 가 GPU 를 잡은 뒤에
nohup bash scripts/gpu_free.sh bash scripts/run_fpk_plus.sh > runs/fpk_plus/chain.log 2>&1 &
# 비어 있는 GPU 가 없을 때 기다리게 하려면 GPU_WAIT=1, 특정 GPU 는 GPU_ID=0
tail -f runs/fpk_plus/chain.log
```

GPU 가 한 장이면 사슬 B 를 먼저 끝까지 돌린 뒤 사슬 A 를 띄웁니다(같은 명령, `sleep` 없이 차례로).

```bash
NOPHYS=1 bash scripts/gpu_free.sh bash scripts/run_fpk_plus.sh > runs/fpk_plus/chain_ng.log 2>&1 && \
    bash scripts/gpu_free.sh bash scripts/run_fpk_plus.sh > runs/fpk_plus/chain.log 2>&1 &
```

`gpu_free.sh` 는 아무도 쓰지 않는 GPU 하나를 골라 `CUDA_VISIBLE_DEVICES` 로 고정합니다. 사슬 A 는 자기 단계를 마친 뒤
사슬 B 의 kernel(`done/devk2_ng`)을 기다렸다가(B 의 log 가 6시간 동안 멈추면 기다림을 끝냄) + 모델 단계를 돕니다.
B 가 아직 시작하지 않았으면 기다리지 않고, 띄울 명령을 출력한 뒤 끝납니다. 그때는 B 가 끝난 뒤 사슬 A 를 다시 띄우면
끝난 단계는 건너뛰고 + 모델 단계만 돕니다. 각 사슬은 아래 순서로 돕니다.

| 단계 | 하는 일 | 반복 | 예상 시간 (추정) |
| --- | --- | --- | --- |
| smoke | 모든 단계를 실제 크기로 2–4 반복씩 (`runs/fpk_plus/smoke`) | 2–4 | 20–40분 |
| core | U-Net core (1.5/3/6° 세 층, 약 290만 매개변수), G(x) 입력, 하루 4 substep, 곡선 학습 1→10일 | 6000 | 12–20시간 |
| devk1 | entropy 편차 kernel (smooth noise) | 800 | 2–4시간 |
| spec | devk1 의 1일 오차 스펙트럼 (1.5°, 119 차수) | – | 10–20분 |
| devk2 | 사슬 kernel: correction, 10일 gradient, 띠별 entropy, 구면조화 noise, 초기 편차 = spec | 400 | 2–4시간 |
| flow | flow matching (U-Net, 4층) | 600 | 2–4시간 |
| score | 2017 채점: persistence, core, devk1, devk2, flow | – | 1–2시간 |
| report | 변수별 표 (core 대비 이득, 짝 bootstrap) | – | 1분 |
| **사슬 A 끝에, 두 kernel 이 다 있으면** | | | |
| proj | 불변 측도 조건의 세기: 두 core 의 2016 궤적 (178 IC), 예측일·변수·파수 띠별 최소제곱 | – | 20–60분 |
| projk | 두 kernel 을 조건으로 읽는 체크포인트 (`devk_band_sph_p`, `devk_band_sph_ng_p`, 학습 없음) | – | 1분 |
| mix | 2017 채점: 섞은 kernel, 섞은 kernel + 조건 (+ 모델), 각 쪽 kernel 도 같이 | – | 1–3시간 |
| mixrep | 표 `mix_plus.md`, 짝 비교, member 수 편향을 뺀 IGN | – | 1분 |

시간은 5.625° CPU 실행과 이전 1.5° GPU 실행에서 추정한 값입니다. 실제 값은 smoke 로그의 `sec` 와 `gpu_gb` 로 확인합니다
(`runs/fpk_plus/logs/smoke.log`). 사슬 하나가 대략 1–2일이고, + 모델 단계가 2–4시간 더 걸립니다. smoke 는 + 모델 단계도
작은 크기로 한 번 돌립니다(core 하나를 양쪽에 써서).

### 끊겼을 때

같은 명령을 다시 실행하면 됩니다. 끝난 단계는 `runs/fpk_plus/done/<단계>` 표시가 있어 건너뜁니다.
core, devk1, devk2, flow 는 log 마다 `<tag>.part` 를 저장하고, 거기서 이어 갑니다.

### 진행 보기

```bash
tail -3 runs/fpk_plus/logs/core.log        # {"it": ..., "K": 곡선 일수, "loss", "val", "sec"}
tail -2 runs/fpk_plus/logs/devk2.log       # val_ign, val_rmse_s, gpu_gb, sec
ls runs/fpk_plus/done/
nvidia-smi
```

## 4. 결과

**+ 모델**

- `runs/fpk_plus/mix_plus.md`: 표 하나. 각 kernel, 조건 건 각 kernel, 섞은 kernel, + 모델의 z500·t850·t2m RMSE,
  nRMSE, CRPS, 퍼짐/오차, IGN (1/3/5/7/10일)과 + 모델의 짝 비교 (CRPS 차, nRMSE 이득 %, 95 % 구간).
- `runs/fpk_plus/score_mixkp_plus.json`, `score_mixkp_plus_raw.npz`: + 모델의 점수 (키는 아래 score_plus 와 같음),
  `between_var_share_z500` 은 분산 중 두 core 사이 몫. `score_mixk_plus.*`: 조건 없이 섞은 kernel.
- `runs/fpk_plus/pair_*.json`: 짝 bootstrap. `mix_ign_*.json`: member 수 편향을 뺀 IGN.
- `runs/fpk_plus/proj_2016.npz`: 조건의 세기를 맞춘 통계. `devk_band_sph_p.pt`, `devk_band_sph_ng_p.pt`: 조건으로 읽는
  kernel (`run_fpk.py score --arms devk_band_sph_p` 로 따로 채점할 수 있음).

**사슬마다**

- `runs/fpk_plus/score_plus.json`: 단계별 점수. 키는 nrmse, crps, spread_skill, acc, entropy, ign 이고,
  `rmse_phys` 는 z500·t850 등 물리 단위입니다. 10일까지 하루 단위로 들어 있습니다.
- `runs/fpk_plus/score_plus_raw.npz`: IC 별 값 (짝 비교용)
- `runs/fpk_plus/vars_plus.md`: 변수별 표. core 대비 RMSE 이득 %, 95% 짝 bootstrap 구간, CRPS, spread/skill, ACC 가 들어 있습니다.
- `runs/fpk_plus/spectra_devk_m.json`: 1.5° 오차 스펙트럼

표만 다시 만들려면:

```bash
python scripts/eval_fpk_vars.py runs/fpk_plus/score_plus --ref core --arms devk_m devk_band_sph devflow_line \
    --out runs/fpk_plus/vars_plus.md --json runs/fpk_plus/vars_plus.json
```

5.625° 결과와는 물리 단위(`rmse_phys`, `crps_vars`)로 비교합니다. z-score 단위 점수(nrmse, ign)는 격자마다 s 가 달라서 직접 비교할 수 없습니다.

## 5. 사슬 B (G(x) 없는 core)

+ 모델의 한쪽입니다(예전에는 선택이던 물리장 대조). 자료와 `stats.npz` 는 공유하고, tag 는 `*_ng` 입니다.
완료 표시와 log 는 `done/<단계>_ng`, `logs/<단계>_ng.log` 로 따로 남아 사슬 A 와 섞이지 않습니다. 사슬 B 만의 결과는
`score_plus_ng.json`, `vars_plus_ng.md` 이고, 사슬 A 와 B 의 짝 비교가 그대로 물리장 효과의 검정입니다.

## 6. 조절할 수 있는 것 (환경 변수)

| 변수 | 기본값 | 뜻 |
| --- | --- | --- |
| `Y0`, `Y1`, `VAL`, `TEST` | 1984, 2015, 2016, 2017 | 학습 연도 범위, 검증 해, 시험 해 |
| `DATA`, `OUT`, `SRC` | data/era5_1p5, runs/fpk_plus, lustre 일평균 | 자료 폴더, 결과 폴더, 원 자료 |
| `CORE_ITERS` | 6000 | core 반복. 5.625° 16년 3000 과 같은 epoch 수 |
| `CORE_SUBSTEPS` | 4 | core 의 하루 substep. 2로 하면 core·kernel·flow 가 모두 약 2배 빨라짐 (이전 1.5° 실행은 4) |
| `DEVK1_ITERS`, `DEVK2_ITERS`, `FLOW_ITERS` | 800, 400, 600 | 5.625° 와 같음 |
| `CORE_BATCH` | 8 | core 의 batch |
| `DEVK_BATCH`, `FLOW_BATCH` | 4, 4 | GPU 메모리가 모자라면 2 |
| `THROUGH` | 10 | devk2 의 평균 경로 gradient 일수. 메모리가 모자라면 5 |
| `NET_CKPT` | 0 | 1이면 kernel·flow 망의 activation 을 backward 때 다시 계산 (메모리 ↓, 시간 약 30% ↑) |
| `STEPS` | (전부) | 이 단계만 실행, 예: `STEPS="score report"` |
| `DRYRUN` | 0 | 1이면 명령만 출력 |
| `NOPHYS` | 0 | 1이면 사슬 B (G(x) 없는 core) |
| `MIX_WAIT` | 1 | 사슬 A 가 끝에 사슬 B 의 kernel 을 기다림. 0이면 기다리지 않고 명령만 출력 |

## 7. 5.625° 판과 같은 것, 다른 것

같은 것: 모형 구조(core → entropy 편차 kernel → flow matching), 손실 함수, kernel·flow 반복 수와 학습률,
devk2 의 선택(correction, 10일 gradient, 띠별 entropy, 구면조화 noise, 초기 편차 스펙트럼), 채점 방식.

해상도에 맞춰 바꾼 것:
- **구면조화 변환**: 극을 포함한 격자에서 경도 240, lmax 118 을 씁니다. 극 행은 가중치 0 입니다. 만들 때 차수마다 왕복 오차를 스스로 검사하고, 1e-6 을 넘으면 멈춥니다.
- **잡음과 평활의 크기**: 칸 수가 아니라 같은 물리적 크기로 맞췄습니다. smooth noise 11칸 = 17°, entropy 평활 19칸 = 28°.
- **신경망**: core 는 U-Net (64/128/192) 입니다. kernel 의 r-net·correction 망(32/64/128/192)과 flow 망(64/96/128/192)도 U-Net 이라서, 1.5° 에서도 5.625° 때만큼 멀리 봅니다.
- **초기 편차 스펙트럼**: 1.5° 에서 새로 만듭니다(spec 단계). 5.625° 파일(32 차수)은 거부합니다.
- **GPU**:
  - 자료는 연도별 memmap 에서 표본마다 한 번에 읽고 GPU 에서 정규화하며, 다음 batch 를 미리 읽습니다.
  - 난수는 GPU 에서 뽑습니다.
  - 신경망만 bf16 이고, 구면조화·entropy·점수·상태는 fp32 입니다.
  - core 는 substep 마다 backward 때 다시 계산합니다.
  - 채점은 IC 단위로 나눕니다.

## 8. 문제가 생기면

- **GPU 메모리 부족 (CUDA out of memory)**: `NET_CKPT=1` → `THROUGH=5` → `DEVK_BATCH=2` 순서로 시도합니다.
  smoke 로그의 `gpu_gb` 가 90 에 가까우면 미리 조정합니다.
  - 가장 큰 단계는 devk2 입니다. 1.5° 에서 CPU(fp32)로 잰 크기(2026-11-24, 가짜 1.5° 자료): batch 하나에 3 GB 이상,
    여기에 pair 하나마다 1.3 GB, 평균 경로 gradient 하루마다 0.65 GB 가 더 듭니다. 기본값(batch 4, pair 4, 10일)이면
    activation 만 약 50–80 GB 로 추정되어 96 GB 에 들어가지만 여유는 크지 않습니다. `THROUGH=5` 는 약 13 GB,
    `DEVK_BATCH=2` 는 약 절반을 줄입니다. pair 수는 모형의 일부라 바꾸지 않습니다.
  - 같은 가짜 자료로 stats, core, devk1, spec, flow, score 는 1.5° 격자에서 끝까지 돌았습니다. devk2 는 이 검사 기계
    (15 GB)에서 batch 1 로도 메모리를 넘어 크기만 쟀습니다. 그래서 devk2 가 실제로 도는지는 GPU smoke 가 처음 확인합니다.
- **자료 검사 실패 (`--check` 가 NOT OK)**: 원 자료의 변수나 파일 이름이 다른 것입니다. `logs/data.log` 의 출력을 보내 주세요.
- **"stats.npz is for a 32 x 64 grid"**: `OUT` 이 5.625° 결과 폴더입니다. 다른 폴더를 씁니다.
- 실패한 단계는 사슬이 멈추고, `runs/fpk_plus/logs/<단계>.log` 의 끝을 `chain.log` 에 보여 줍니다.
- **mix 단계의 CPU 메모리**: 1.5° 에서 IC 하나의 float64 통계가 약 10 GB 입니다(IC 하나씩 채점). 서버 메모리가 모자라면
  다른 작업이 끝난 뒤 `STEPS="mix mixrep" bash scripts/run_fpk_plus.sh` 로 다시 돌립니다.
- **+ 모델 단계만 다시**: `STEPS="proj projk mix mixrep" bash scripts/gpu_free.sh bash scripts/run_fpk_plus.sh`.
  한 단계를 처음부터 다시 하려면 `runs/fpk_plus/done/<단계>` 를 지웁니다(projk 는 `*_p.pt` 도 지움).
- **새 단계의 검사**: 5.625° 실제 결과(16년 core 둘, kernel 둘)로 proj → projk → mix → mixrep 를 CPU 에서 끝까지 돌렸고,
  조건 읽기는 극 격자 시험(`tests/test_proj.py`)을 통과했습니다. 1.5° GPU 에서는 smoke 가 처음 확인합니다.
