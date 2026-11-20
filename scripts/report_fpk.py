"""Merge the score JSONs of run_fpk.py score and draw the result figures (docs/results/).

    python scripts/report_fpk.py runs/fpk/score_det.json runs/fpk/score_kernel.json runs/fpk/score_flow.json \
        runs/fpk/score_flowg.json --out docs/results
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:  # NanumGothic for the Korean labels (pip install koreanize-matplotlib)
    import koreanize_matplotlib  # noqa: F401
except ImportError:
    pass

ORDER = ["climatology", "persistence", "core", "kernel", "flow_gauss", "flow"]
NAME = {"climatology": "기후값", "persistence": "지속성", "core": "core (결정론)", "kernel": "물리 kernel 단독",
        "flow_gauss": "FM, Gaussian source", "flow": "FM, 물리 kernel source"}
COLOR = {"climatology": "#8a8985", "persistence": "#b8b7b0", "core": "#2a78d6", "kernel": "#1baf7a",
         "flow_gauss": "#eb6834", "flow": "#6250d6"}
STYLE = {"climatology": ":", "persistence": "--"}
ENS = ["kernel", "flow_gauss", "flow"]

plt.rcParams.update({"font.family": ["NanumGothic", "DejaVu Sans"], "axes.unicode_minus": False, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#8a8985", "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})


def merge(paths):
    scores, meta = {}, {}
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        meta.update(d["meta"])
        for k, v in d["scores"].items():
            scores.setdefault(k, v)
    return meta, scores


def line(ax, S, key, arms, ylabel, ref=None, sub=None):
    for a in arms:
        if a not in S or key not in S[a]:
            continue
        y = S[a][key] if sub is None else S[a][key][sub]
        x = list(range(1, len(y) + 1))
        ax.plot(x, y, STYLE.get(a, "-"), color=COLOR[a], lw=2, marker="o" if a in ENS + ["core"] else None, ms=4,
                label=NAME[a])
    if ref is not None:
        ax.axhline(ref, color="#52514e", lw=1, ls="--")
    ax.set_xlabel("선행 (일)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(1, 11))


def fmt(v):
    return "" if v is None else f"{v:.3f}"


def table(S, key, arms, leads=(1, 3, 5, 7, 10), sub=None):
    rows = ["| | " + " | ".join(f"{l}일" for l in leads) + " |", "| --- |" + " ---: |" * len(leads)]
    for a in arms:
        if a not in S or key not in S[a]:
            continue
        y = S[a][key] if sub is None else S[a][key][sub]
        rows.append(f"| {NAME[a]} | " + " | ".join(fmt(y[l - 1]) for l in leads) + " |")
    return "\n".join(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scores", nargs="+")
    p.add_argument("--out", default="docs/results")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta, S = merge(a.scores)
    with open(os.path.join(a.out, "scores.json"), "w") as f:
        json.dump(dict(meta=meta, scores=S), f, indent=1)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    line(ax[0], S, "nrmse", ORDER, "앙상블 평균 nRMSE (65채널)")
    ax[0].set_title("평균 예보의 오차 (낮을수록 좋음)", loc="left")
    line(ax[1], S, "crps", ORDER, "CRPS (정규화, 65채널)")
    ax[1].set_title("분포 예보의 점수 (낮을수록 좋음)", loc="left")
    line(ax[2], S, "spread_skill", ENS, "spread / skill (유한 member 보정)", ref=1.0)
    ax[2].set_title("퍼짐의 보정 (1 = 맞음)", loc="left")
    ax[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig1_skill.png"), dpi=150)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for e in ENS:
        if e not in S or "entropy" not in S[e]:
            continue
        x = range(1, 11)
        ax[0].plot(x, S[e]["entropy"], "-", color=COLOR[e], lw=2, marker="o", ms=4, label=f"{NAME[e]}: H_ens")
        ax[0].plot(x, S[e]["ign"], "--", color=COLOR[e], lw=1.5, label=f"{NAME[e]}: IGN")
        if "kernel_entropy" in S[e]:
            ax[2].plot(x, S[e]["kernel_entropy"], "-", color=COLOR[e], lw=2, marker="o", ms=4, label=NAME[e])
    ax[0].set_title("예측 entropy (실선) 와 참값의 log score (점선)", loc="left")
    ax[0].set_xlabel("선행 (일)")
    ax[0].set_ylabel("nats / 값 (단위 s)")
    ax[0].legend(frameon=False, fontsize=8)
    line(ax[1], S, "entropy_error_corr", ENS, "IC 간 상관: H_ens 대 ½ log e²")
    ax[1].set_title("entropy 가 오차를 가려내는가 (높을수록 좋음)", loc="left")
    ax[2].set_title("궤적 위에서 kernel 의 하루 entropy 생산 H_K", loc="left")
    ax[2].set_xlabel("선행 (일)")
    ax[2].set_ylabel("nats / 값 (단위 s)")
    for x_ in ax[1:]:
        x_.set_xticks(range(1, 11))
    ax[0].set_xticks(range(1, 11))
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig2_entropy.png"), dpi=150)

    md = [f"시험 {meta.get('test_year')}년, IC {meta.get('ics')}개 ({meta.get('ic_stride')}일 간격), member "
          f"{meta.get('members')}개, flow Heun {meta.get('steps')} step.", "",
          "### 앙상블 평균 nRMSE (65채널 평균, 정규화 단위)", "", table(S, "nrmse", ORDER), "",
          "### CRPS (65채널 평균, 정규화 단위)", "", table(S, "crps", ORDER), "",
          "### spread / skill (1 = 보정됨)", "", table(S, "spread_skill", ENS), "",
          "### 예측 entropy H_ens 와 log score IGN (nats / 값, 단위 s; 보정되면 같음)", "",
          table(S, "entropy", ENS), "", table(S, "ign", ENS), "",
          "### IC 간 상관: 예측 entropy 대 실제 오차 크기", "", table(S, "entropy_error_corr", ENS), ""]
    for v in ("z500", "t850", "t2m", "u850"):
        md += [f"### {v} RMSE (물리 단위)", "", table(S, "rmse_phys", ORDER, sub=v), ""]
    with open(os.path.join(a.out, "tables.md"), "w") as f:
        f.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
