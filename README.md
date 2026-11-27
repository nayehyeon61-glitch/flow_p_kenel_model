# flow_p_kenel_model

방정식 → 분포 → 10일 ensemble: PG-CFM 의 tendency core(원시방정식 물리항 G(x) 포함)를 물리 kernel 분포
K(x, y) 로 바꾸고, 그 분포를 source 로 하는 flow matching 으로 10일 ensemble 과 그 entropy 를 예측합니다.
설계와 결과는 [`docs/design_ko.md`](docs/design_ko.md).

    G(x) ─► Φ(x) = core 의 하루 ─► K(x,y) = (1/N) Σ_k N(y; Φ(x+ε s δ_k), (s h(x))²) ─► flow v_θ: K(x,·) → p(x_{t+1}|x_t)
                                                                                   └► member 별 10일 rollout ─► H_ens(L), IGN(L)

| 경로 | 내용 |
| --- | --- |
| `fpk/kernel.py` | 물리 kernel K(x, y): 방정식(core) 중심의 kernel 합, 학습되는 폭, composite NLL, entropy, 표본 |
| `fpk/flow.py` | kernel 을 source 로 하는 conditional flow matching (대조: Gaussian source) |
| `fpk/ensemble.py` | member 별 10일 rollout, entropy / log score / CRPS / spread-skill 채점 |
| `fpk/data.py`, `fpk/env.py` | KAI 계약의 일평균 ERA5, vendored PG-CFM 접근 |
| `scripts/run_fpk.py` | `core` / `kernel` / `flow` / `score` |
| `third_party/pgfm/` | physical_guide_flowmatching (c97r6n, 5d38e56) 의 core, 연산자, 자료 코드 (변경 없음, `UPSTREAM.md`) |

```bash
pip install -r requirements.txt
python -m pytest -q tests
```
