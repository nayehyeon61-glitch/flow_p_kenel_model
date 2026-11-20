"""Per-variable evaluation of the 10-day ensembles, in the form of PG-CFM's scripts/eval_kai_channel_gains.py.

    python scripts/eval_fpk_vars.py runs/fpk/score_det runs/fpk/score_kernel runs/fpk/score_flow runs/fpk/score_flowg \
        --ref core --out docs/results/vars.md --json docs/results/vars.json

Every score is read from <prefix>.json and <prefix>_raw.npz (run_fpk.py score). All arms are scored on the same ICs,
so the comparison with the reference arm is paired. Per variable and lead:

  RMSE (physical units) of the ensemble mean, and the RMSE gain % over the reference = 1 - RMSE_A / RMSE_ref
      (+ = A better), with the 95 % interval of a paired bootstrap over the ICs (500 draws)
  information / amplitude / bias: the split of the gain from the exact MSE split of every forecast,
      MSE = bias^2 + s_o^2 (1 - r^2) + s_o^2 (a - r)^2; the three add up to the gain
  ACC and activity (pattern amplitude of the forecast anomaly / the truth's) of the ensemble mean
  ensembles: CRPS (physical units), spread / skill (finite-ensemble corrected, 1 = calibrated), entropy H_ens
      (nats per value in units of the daily increment std) and the correlation over ICs of H_ens with the log error
windNNN is the wind vector at NNN hPa: RMSE = sqrt(MSE_u + MSE_v), the split terms of u and v add (no ACC / CRPS).
"""

import argparse
import json

import numpy as np

CHANS = ["z500", "z850", "t850", "t2m", "msl", "u850", "v850", "wind850", "u250", "v250", "wind250"]
Q = [0, 2, 4, 6, 9]  # leads 1 / 3 / 5 / 7 / 10
NAME = {"climatology": "기후값", "persistence": "지속성", "core": "core", "kernel": "kernel 단독",
        "flow_gauss": "FM Gaussian", "flow": "FM 물리 kernel"}
UNIT = {"z": "m²/s²", "t": "K", "u": "m/s", "v": "m/s", "w": "m/s", "m": "Pa"}


def load(prefixes):
    S, R, names, std = {}, {}, None, None
    for p in prefixes:
        d = json.load(open(p + ".json"))
        raw = np.load(p + "_raw.npz")
        names = d["meta"]["channels"]
        std = raw["std"].astype(np.float64)
        for arm, v in d["scores"].items():
            if arm in S:
                continue
            S[arm] = v
            R[arm] = {k.split("__", 1)[1]: raw[k].astype(np.float64) for k in raw.files if k.startswith(arm + "__")}
    return S, R, names, std


def idx(names, c):
    return [names.index("u" + c[4:]), names.index("v" + c[4:])] if c.startswith("wind") else [names.index(c)]


def boot(va, vb, seed):
    rng, N = np.random.default_rng(seed), va.shape[0]
    d = rng.integers(0, N, (500, N))
    g = 100 * (1 - np.sqrt(va[d].mean(1)) / np.sqrt(vb[d].mean(1)))  # (500, L)
    return np.percentile(g, [2.5, 97.5], axis=0)


def per_var(R, names, std, arm, ref, c, k):
    ix = idx(names, c)
    s2 = std[ix] ** 2
    ma, mb = (R[arm]["mse"][..., ix] * s2).sum(-1), (R[ref]["mse"][..., ix] * s2).sum(-1)  # (N, L) physical
    ra, rb = np.sqrt(ma.mean(0)), np.sqrt(mb.mean(0))
    e = dict(rmse=ra, gain=100 * (1 - ra / rb))
    e["lo"], e["hi"] = boot(ma, mb, k)
    if "terms" in R[arm] and "terms" in R[ref]:
        ta = (R[arm]["terms"][..., ix, :] * s2[:, None]).sum(-2).mean(0)  # (L, 3)
        tb = (R[ref]["terms"][..., ix, :] * s2[:, None]).sum(-2).mean(0)
        part = 100 * (tb - ta) / ((ra + rb) * rb)[:, None]
        e["bias"], e["info"], e["amp"] = part[:, 0], part[:, 1], part[:, 2]
    if len(ix) == 1:
        i = ix[0]
        if "acc" in R[arm]:
            e["acc"], e["act"] = R[arm]["acc"][..., i].mean(0), R[arm]["act"][..., i].mean(0)
        if "var" in R[arm]:
            e["crps"] = R[arm]["crps"][..., i].mean(0) * std[i]
            e["spread"] = np.sqrt(R[arm]["var"][..., i].mean(0)) * std[i]
            e["H"] = R[arm]["H"][..., i].mean(0)
            e["corr"] = np.array([np.corrcoef(R[arm]["H"][:, l, i], R[arm]["logerr"][:, l, i])[0, 1]
                                  for l in range(R[arm]["H"].shape[1])])
    return e


def f(v, d=1, sign=True):
    return " / ".join((f"{v[q]:+.{d}f}" if sign else f"{v[q]:.{d}f}").replace("-", "−") for q in Q)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scores", nargs="+", help="score prefixes (without .json / _raw.npz)")
    p.add_argument("--ref", default="core")
    p.add_argument("--arms", nargs="+", default=["kernel", "flow_gauss", "flow"])
    p.add_argument("--members", type=int, default=8)
    p.add_argument("--out")
    p.add_argument("--json")
    a = p.parse_args()
    S, R, names, std = load(a.scores)
    arms = [x for x in a.arms if x in R]
    E = {arm: {c: per_var(R, names, std, arm, a.ref, c, k) for k, c in enumerate(CHANS)} for arm in arms + [a.ref]}
    corr = np.sqrt((a.members + 1) / a.members)
    md = [f"기준: `{a.ref}` (결정론). 선행 1 / 3 / 5 / 7 / 10일. 이득 % 는 + 가 좋아짐. 95 % 구간은 IC 에 대한 paired "
          f"bootstrap(500회), 10일 값.", ""]
    md += ["### 한눈에: core 대비 RMSE 이득 % (앙상블 평균)", "", "| 변수 | " + " | ".join(NAME[x] for x in arms) + " |",
           "| --- |" + " --- |" * len(arms)]
    for c in CHANS:
        md.append(f"| {c} | " + " | ".join(f(E[x][c]["gain"]) for x in arms) + " |")
    md += ["", "### 한눈에: CRPS (물리 단위), 결정론 core 는 |오차| 의 평균", "",
           "| 변수 | core | " + " | ".join(NAME[x] for x in arms) + " |", "| --- | --- |" + " --- |" * len(arms)]
    for c in CHANS:
        if c.startswith("wind"):
            continue
        i = names.index(c)
        base = R[a.ref]["crps"][..., i].mean(0) * std[i]
        md.append(f"| {c} | {f(base, 2, False)} | " + " | ".join(f(E[x][c]["crps"], 2, False) for x in arms) + " |")
    for c in CHANS:
        unit = UNIT.get(c[0], "")
        eb = E[a.ref][c]
        head = f"\n#### {c} (RMSE 단위 {unit}; core RMSE {f(eb['rmse'], 2, False)}"
        head += f", core ACC 10일 {eb['acc'][9]:.3f}, 활동도 {eb['act'][9]:.2f})" if "acc" in eb else ")"
        md += [head, "", "| A | RMSE | 이득 % | 정보 항 | 진폭 항 | 95% 구간 (10일) | ACC | 활동도 10일 | spread/skill | entropy | entropy-오차 상관 |",
               "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for x in arms:
            e = E[x][c]
            ci = f"{e['lo'][9]:+.1f} ~ {e['hi'][9]:+.1f}".replace("-", "−")
            acc = f(e["acc"], 3, False) if "acc" in e else ""
            act = f"{e['act'][9]:.2f}" if "act" in e else ""
            ss = f(corr * e["spread"] / e["rmse"], 2, False) if "spread" in e else ""
            H = f(e["H"], 2, False) if "H" in e else ""
            co = f(e["corr"], 2, False) if "corr" in e else ""
            md.append(f"| {NAME[x]} | {f(e['rmse'], 2, False)} | {f(e['gain'])} | {f(e.get('info', np.zeros(10)))} | "
                      f"{f(e.get('amp', np.zeros(10)))} | {ci} | {acc} | {act} | {ss} | {H} | {co} |")
    text = "\n".join(md)
    print(text)
    if a.out:
        open(a.out, "w").write(text + "\n")
    if a.json:
        json.dump({x: {c: {k: np.round(np.asarray(v), 5).tolist() for k, v in e.items()} for c, e in E[x].items()}
                   for x in E}, open(a.json, "w"))


if __name__ == "__main__":
    main()
