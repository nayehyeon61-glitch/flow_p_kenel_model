"""One table per score over all the structures (65-channel means, leads 1 / 3 / 5 / 7 / 10) -> docs/results/summary.md.

    python scripts/summary_fpk.py --out docs/results/summary.md
"""

import argparse
import json
import os

ROWS = [("core (결정론)", "score_det", "core"), ("메모리 core (결정론)", "score_m_kernel", "core"),
        ("물리 kernel 단독", "score_kernel", "kernel"), ("FM, Gaussian source", "score_flowg", "flow_gauss"),
        ("FM, 물리 kernel source", "score_flow", "flow"), ("FM + rollout 미세조정", "score_flow_tt", "flow_tt"),
        ("메모리 kernel", "score_m_kernel", "kernel"), ("메모리 FM", "score_m_flow", "flow_m"),
        ("메모리 FM + rollout 미세조정", "score_m_flow", "flow_mt"),
        ("entropy 편차 kernel", "score_devk", "devk"), ("entropy 편차 kernel (메모리 core)", "score_devk_m", "devk_m"),
        ("사슬 kernel, 10일 sequence 학습 (메모리 core)", "score_devk_seq_m", "devk_seq_m"),
        ("편차 kernel source FM", "score_devflow", "devflow"),
        ("편차 FM, Gaussian source", "score_devflow_gauss", "devflow_gauss"),
        ("편차 kernel source FM (메모리 core)", "score_devflow_m", "devflow_m"),
        ("16년 메모리 core (결정론)", "score_devk_mL", "core"),
        ("entropy 편차 kernel (16년 core)", "score_devk_mL", "devk_mL"),
        ("사슬 kernel, sequence 학습 (16년 core)", "score_devk_seq_mL", "devk_seq_mL"),
        ("+ 띠별 entropy", "score_devk_band", "devk_band"),
        ("+ 띠별 entropy, 구면조화 noise, 초기 편차 스펙트럼", "score_devk_band_sph", "devk_band_sph"),
        ("+ 잔차 부호 제약, 전체 에너지 제약", "score_devk_band_sph_en", "devk_band_sph_en")]
KEYS = [("nrmse", "앙상블 평균 nRMSE (낮을수록 좋음)"), ("crps", "CRPS (낮을수록 좋음)"),
        ("spread_skill", "spread / skill (1 = 보정됨)"), ("entropy_error_corr", "entropy–오차 상관 (높을수록 좋음)")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs/fpk")
    p.add_argument("--out", default="docs/results/summary.md")
    a = p.parse_args()
    S = []
    for name, f, arm in ROWS:
        path = os.path.join(a.runs, f + ".json")
        if os.path.exists(path):
            S.append((name, json.load(open(path))["scores"][arm]))
    md = ["시험 2017년, IC 60개, member 8개, 65채널 평균(정규화 단위). 선행 1 / 3 / 5 / 7 / 10일.", ""]
    for k, t in KEYS:
        md += [f"### {t}", "", "| | 1일 | 3일 | 5일 | 7일 | 10일 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
        md += [f"| {n} | " + " | ".join(f"{s[k][i]:.3f}" for i in (0, 2, 4, 6, 9)) + " |" for n, s in S if k in s]
        md.append("")
    open(a.out, "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
