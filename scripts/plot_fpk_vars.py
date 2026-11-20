"""Per-variable curves over leads 1..10 with the exact numbers under them, as one HTML page.

    python scripts/plot_fpk_vars.py runs/fpk/score_det runs/fpk/score_kernel runs/fpk/score_flow runs/fpk/score_flowg \
        --out docs/results/vars.html

Reads the same score files as eval_fpk_vars.py. Per variable: RMSE of the ensemble mean, CRPS, ACC and spread / skill
for every lead day, drawn as inline SVG (no library), and one table with the values of every day.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from eval_fpk_vars import CHANS, PAIRS, UNIT, load, per_var  # noqa: E402
from eval_fpk_vars import NAME as NAME0  # noqa: E402

ARMS = ["core", "kernel", "flow_gauss", "flow"]
NAME = dict(NAME0, core="core (결정론)", kernel="물리 kernel 단독", flow_gauss="FM, Gaussian source",
            flow="FM, 물리 kernel source")
DET = {"core", "core_m"}  # deterministic arms: no spread
LONG = {"z500": "500 hPa 지위", "z850": "850 hPa 지위", "t850": "850 hPa 기온", "t2m": "2 m 기온", "msl": "해면기압",
        "u850": "850 hPa 동서바람", "v850": "850 hPa 남북바람", "wind850": "850 hPa 바람 벡터",
        "u250": "250 hPa 동서바람", "v250": "250 hPa 남북바람", "wind250": "250 hPa 바람 벡터"}
W, H, L, R, T, B = 340, 220, 46, 12, 12, 30


def nice(lo, hi, n=4):
    span = (hi - lo) or 1.0
    step = 10 ** np.floor(np.log10(span / n))
    for m in (1, 2, 2.5, 5, 10):
        if span / (step * m) <= n:
            step *= m
            break
    a, b = np.floor(lo / step) * step, np.ceil(hi / step) * step
    return np.arange(a, b + step / 2, step)


def num(v, d):
    return f"{v:,.{d}f}".replace("-", "−")


def chart(title, series, d, unit="", ref=None, zero=False):
    """series: [(arm, values(10))]"""
    vals = np.concatenate([np.asarray(v) for _, v in series] + ([[ref]] if ref is not None else []))
    lo, hi = (0.0 if zero else float(vals.min())), float(vals.max())
    tk = nice(lo, hi)
    lo, hi = tk[0], tk[-1]
    X = lambda k: L + (W - L - R) * k / 9  # noqa: E731
    Y = lambda v: T + (H - T - B) * (1 - (v - lo) / (hi - lo))  # noqa: E731
    td = d if (tk[1] - tk[0]) < 1 or d == 0 else max(0, d - 1)
    if any(abs(t - round(t, td)) > 1e-9 for t in tk):
        td += 1
    o = [f'<figure class="chart"><figcaption>{title}{f" <span>{unit}</span>" if unit else ""}</figcaption>',
         f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{title}">']
    for t in tk:
        o.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>'
                 f'<text class="tick" x="{L - 6}" y="{Y(t) + 3.5:.1f}" text-anchor="end">{num(t, td)}</text>')
    for k in range(10):
        o.append(f'<text class="tick" x="{X(k):.1f}" y="{H - 12}" text-anchor="middle">{k + 1}</text>')
    o.append(f'<text class="tick" x="{W - R}" y="{H - 1}" text-anchor="end">선행 (일)</text>')
    if ref is not None:
        o.append(f'<line class="ref" x1="{L}" x2="{W - R}" y1="{Y(ref):.1f}" y2="{Y(ref):.1f}"/>')
    for arm, v in series:
        pts = " ".join(f"{X(k):.1f},{Y(x):.1f}" for k, x in enumerate(v))
        o.append(f'<polyline class="ln s-{arm}" points="{pts}"/>')
        for k, x in enumerate(v):
            o.append(f'<circle class="pt s-{arm}" cx="{X(k):.1f}" cy="{Y(x):.1f}" r="3">'
                     f'<title>{NAME[arm]} · {k + 1}일 · {num(x, d)}{" " + unit if unit else ""}</title></circle>')
    o.append("</svg></figure>")
    return "".join(o)


def rows(label, series, d, best=None):
    """best: 'min' / 'max' / None -> the best value of every day in bold"""
    A = np.array([v for _, v in series])
    pick = None if best is None else (A.argmin(0) if best == "min" else A.argmax(0))
    out = []
    for r, (arm, v) in enumerate(series):
        head = f'<th class="grp" rowspan="{len(series)}" scope="rowgroup">{label}</th>' if r == 0 else ""
        cells = "".join(f'<td{" class=best" if pick is not None and pick[k] == r else ""}>{num(x, d)}</td>'
                        for k, x in enumerate(v))
        out.append(f'<tr{" class=first" if r == 0 else ""}>{head}<th scope="row"><i class="sw s-{arm}"></i>{NAME[arm]}</th>{cells}</tr>')
    return "".join(out)


CSS = """
:root{--bg:#fbfbf9;--panel:#ffffff;--fg:#1f2328;--mute:#5f6670;--line:#e3e4e0;--grid:#ebece8;
--c0:#2a78d6;--c1:#12976a;--c2:#e0602c;--c3:#6250d6;--c4:#b8860b;--c5:#c2377e;--c6:#3c8f9e;--c7:#7a7f87;--best:#f1f0ff;
--sans:"IBM Plex Sans KR","Apple SD Gothic Neo","Malgun Gothic",system-ui,sans-serif;--mono:"IBM Plex Mono",ui-monospace,Menlo,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#15171a;--panel:#1c1f23;--fg:#e8e9ea;--mute:#a0a6ae;--line:#30343a;--grid:#272b30;
--c0:#5b9cf0;--c1:#34c08f;--c2:#f08a5b;--c3:#9486f2;--c4:#d9a93a;--c5:#e56aa6;--c6:#62b7c6;--c7:#a3a8b0;--best:#2a2848;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#15171a;--panel:#1c1f23;--fg:#e8e9ea;--mute:#a0a6ae;--line:#30343a;--grid:#272b30;
--c0:#5b9cf0;--c1:#34c08f;--c2:#f08a5b;--c3:#9486f2;--c4:#d9a93a;--c5:#e56aa6;--c6:#62b7c6;--c7:#a3a8b0;--best:#2a2848;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font-family:var(--sans);font-size:15px;line-height:1.6}
.wrap{max-width:1120px;margin:0 auto;padding-block:28px 64px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));display:flex;flex-direction:column;gap:28px}
h1{font-size:26px;line-height:1.25;margin:0;text-wrap:balance}
h2{font-size:19px;margin:0;display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}
h2 small{font-size:13px;font-weight:400;color:var(--mute)}
p{margin:0;max-width:68ch}.mute{color:var(--mute);font-size:13.5px}
header{display:flex;flex-direction:column;gap:10px}
nav{display:flex;flex-wrap:wrap;gap:6px}
nav a{font-family:var(--mono);font-size:12.5px;color:var(--fg);text-decoration:none;border:1px solid var(--line);border-radius:6px;padding:3px 9px;background:var(--panel)}
nav a:hover,nav a:focus-visible{border-color:var(--c3);outline:none}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13.5px;position:sticky;top:env(safe-area-inset-top,0px);background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line);z-index:2}
.sw{display:inline-block;width:14px;height:3px;border-radius:2px;margin-right:7px;vertical-align:middle}
section{display:flex;flex-direction:column;gap:12px;scroll-margin-top:48px}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:12px}
.chart{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 10px 4px;min-width:0}
.chart figcaption{font-size:13px;font-weight:600}.chart figcaption span{font-weight:400;color:var(--mute);font-family:var(--mono);font-size:12px}
svg{display:block;width:100%;height:auto}
.grid{stroke:var(--grid);stroke-width:1}.ref{stroke:var(--mute);stroke-width:1;stroke-dasharray:4 3}
.tick{fill:var(--mute);font-size:10.5px;font-family:var(--mono)}
.ln{fill:none;stroke-width:2;stroke-linejoin:round}.pt{stroke:var(--panel);stroke-width:1.5}
.tw{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--panel)}
table{border-collapse:collapse;width:100%;font-size:12.5px}
td,thead th{font-family:var(--mono);font-variant-numeric:tabular-nums;text-align:right;padding:4px 9px;white-space:nowrap}
thead th{color:var(--mute);font-weight:500;border-bottom:1px solid var(--line)}
tbody th{text-align:left;font-weight:400;padding:4px 9px;white-space:nowrap}
th.grp{font-weight:600;vertical-align:top;border-right:1px solid var(--line)}
tr.first>*{border-top:1px solid var(--line)}
td.best{background:var(--best);font-weight:600}
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scores", nargs="+")
    p.add_argument("--members", type=int, default=8)
    p.add_argument("--arms", nargs="+", default=ARMS, help="at most 8, in the order of the colours")
    p.add_argument("--ref", default="core")
    p.add_argument("--note", default="")
    p.add_argument("--out", default="docs/results/vars.html")
    a = p.parse_args()
    S, Rw, names, std = load(a.scores)
    arms = [x for x in a.arms if x in Rw][:8]
    ens = [x for x in arms if x not in DET]
    det = [x for x in arms if x in DET]
    colour = "".join(f".ln.s-{x}{{stroke:var(--c{i})}}.pt.s-{x}{{fill:var(--c{i})}}.sw.s-{x}{{background:var(--c{i})}}"
                     for i, x in enumerate(arms))
    fin = np.sqrt((a.members + 1) / a.members)
    body = []
    for k, c in enumerate(CHANS):
        E = {x: per_var(Rw, names, std, x, a.ref, c, k) for x in arms}
        unit = UNIT.get(c[0], "")
        rm = [(x, E[x]["rmse"]) for x in arms]
        d = 1 if max(E[a.ref]["rmse"]) >= 100 else 2
        ch, tb = [chart("앙상블 평균 RMSE", rm, d, unit)], [rows(f"RMSE ({unit})", rm, d, "min")]
        tb.append(rows(f"{a.ref} 대비 RMSE 이득 %", [(x, E[x]["gain"]) for x in arms if x != a.ref], 1, "max"))
        if not c.startswith("wind"):
            i = names.index(c)
            cr = [(x, Rw[x]["crps"][..., i].mean(0) * std[i]) for x in det] + [(x, E[x]["crps"]) for x in ens]
            ac = [(x, E[x]["acc"]) for x in arms]
            ss = [(x, (1.0 if x in PAIRS else fin) * E[x]["spread"] / E[x]["rmse"]) for x in ens]
            ch += [chart("CRPS", cr, d, unit), chart("ACC", ac, 3, zero=True), chart("spread / skill", ss, 2, ref=1.0)]
            tb += [rows(f"CRPS ({unit})", cr, d, "min"), rows("ACC", ac, 3, "max"), rows("spread / skill", ss, 2),
                   rows("entropy–오차 상관", [(x, E[x]["corr"]) for x in ens], 2, "max")]
        head = "".join(f"<th>{j}일</th>" for j in range(1, 11))
        body.append(f'<section id="{c}"><h2>{c} <small>{LONG[c]}</small></h2><div class="charts">{"".join(ch)}</div>'
                    f'<div class="tw"><table><thead><tr><th></th><th></th>{head}</tr></thead><tbody>{"".join(tb)}'
                    f"</tbody></table></div></section>")
    meta = next(iter(S.values()))
    legend = "".join(f'<span><i class="sw s-{x}"></i>{NAME[x]}</span>' for x in arms)
    nav = "".join(f'<a href="#{c}">{c}</a>' for c in CHANS)
    html = (f'<title>변수별 10일 예보 점수</title><meta name="claude:viewport-fit-sides" content="cover">'
            f'<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+KR:wght@400;600&display=swap">'
            f"<style>{CSS}{colour}</style><div class=\"wrap\"><header><h1>변수별 10일 예보 점수</h1>"
            f"<p>ERA5 5.625° 일평균, 시험 2017년, 초기조건 60개(6일 간격), 앙상블 member {a.members}개. 변수마다 선행 1~10일의 곡선과 "
            f"그 아래 매일의 수치입니다.</p>"
            f'<p class="mute">RMSE·ACC 는 앙상블 평균의 값이고, 결정론 core 의 CRPS 는 절대오차의 평균입니다. spread / skill 은 유한 member 보정을 '
            f"했고 1 이면 퍼짐이 맞습니다(점선). 표의 굵은 칸은 그날 가장 좋은 값입니다. 바람 벡터(wind)는 u, v 의 MSE 를 더한 RMSE 만 있습니다.</p>"
            f"{f'<p class=mute>{a.note}</p>' if a.note else ''}<nav>{nav}</nav></header><div class=\"legend\">{legend}</div>{''.join(body)}</div>")
    open(a.out, "w").write(html)
    print(a.out, len(html))


if __name__ == "__main__":
    main()
