"""IGN of a mixture (scripts/score_mix.py) against its single arms WITHOUT the finite-ensemble bias.

The scorer's IGN of P antithetic pairs, 0.5 log v^ + c e^2 / (2 v^) with c = (P - 2) / P, has an unbiased error term
but E[log v^] = log v + psi(P / 2) - log(P / 2): 0.135 too low at P = 4, 0.065 at P = 8. The mixture's variance is
v^ = W + B, W = 1/2 (v_A + v_B) (sigma_w^2 chi^2_k / k, k = P_A + P_B) and B = 1/4 (m_A - m_B)^2 EXACT (kernel means are
known): the constant B also cuts the tail of 1 / v^, so c over-corrects. With s = B / (sigma_w^2 + B):
    E[log v^] = log v + g1(k, s),   E[1 / v^] = g2(k, s) / v,   g1, g2 by quadrature over chi^2_k
so  IGN = 0.5 (log v^ - g1) + e^2 / (2 v^ g2)  is unbiased (s from the area means of B and W per IC, lead, channel).
Check: the full mixture (k = 8) and the half one (k = 4, TAG_h) are the SAME distribution and must then agree.

    python scripts/mix_ign.py mixk_W_NW devk_band_sphW devk_band_sphNW docs/results/mixk_W_NW_ign.json
"""
import json, sys
import numpy as np
from scipy import integrate, special, stats

T, A, B = sys.argv[1], sys.argv[2], sys.argv[3]
R = "runs/fpk/"
SC = [i for i in range(66) if i != 62]
Q = [0, 2, 4, 6, 9]


def g(k, s):
    """g1 = E log(s + (1 - s) X / k), g2 = E 1 / (s + (1 - s) X / k), X ~ chi^2_k."""
    f = lambda x, fn: fn(s + (1 - s) * x / k) * stats.chi2.pdf(x, k)  # noqa: E731
    g1 = integrate.quad(f, 0, np.inf, args=(np.log,), limit=200)[0]
    g2 = integrate.quad(f, 0, np.inf, args=(lambda u: 1 / u,), limit=200)[0]
    return g1, g2


def table(k):
    s = np.linspace(0, 0.6, 61)
    t = np.array([g(k, x) if x > 0 else (special.digamma(k / 2) - np.log(k / 2), k / (k - 2)) for x in s])
    return s, t[:, 0], t[:, 1]


def unbiased(raw, tag, k, mix):
    ign, H = raw[f"{tag}__IGN"].astype(np.float64), raw[f"{tag}__H"].astype(np.float64)
    logt = H - 0.5                              # area mean of 0.5 (log 2 pi + log v^)
    c = (k - 2) / k
    err = (ign - logt) / c                      # area mean of e^2 / (2 v^)
    if mix:
        ex = raw[f"{tag}__mix"].astype(np.float64)
        s = np.clip(ex[..., 1] / np.maximum(ex[..., 1] + ex[..., 2], 1e-30), 0, 0.6)
    else:
        s = np.zeros_like(ign)
    ss, t1, t2 = table(k)
    return logt - 0.5 * np.interp(s, ss, t1) + err / np.interp(s, ss, t2), s


out = {}
rt, rh = np.load(R + f"score_{T}_raw.npz"), np.load(R + f"score_{T}_h_raw.npz")
PT = json.load(open(R + f"score_{T}.json"))["scores"][T]["pairs"]
U = {T: unbiased(rt, T, PT, True)[0], T + "_h": unbiased(rh, T + "_h", PT // 2, True)[0]}
for arm in (A, B):
    r = np.load(R + f"score_{arm}_raw.npz")
    P = json.load(open(R + f"score_{arm}.json"))["scores"][arm]["members"] // 2
    U[arm] = unbiased(r, arm, P, False)[0]
    out[f"scorer_ign_{arm}"] = r[f"{arm}__IGN"][:, :, SC].mean((0, 2))[Q].round(4).tolist()
out["scorer_ign_mix_full_half"] = [rt[f"{T}__IGN"][:, :, SC].mean((0, 2))[Q].round(4).tolist(), rh[f"{T}_h__IGN"][:, :, SC].mean((0, 2))[Q].round(4).tolist()]
rng = np.random.default_rng(0)
n = U[T].shape[0]
draws = rng.integers(0, n, size=(2000, n))
for k, v in U.items():
    out[f"ign_{k}"] = v[:, :, SC].mean((0, 2))[Q].round(4).tolist()


def diff(x, y, name):
    d = (x - y)[:, :, SC].mean(2)
    bs = np.array([d[i].mean(0) for i in draws])
    out[name] = dict(diff=d.mean(0)[Q].round(4).tolist(), ci=np.percentile(bs, [2.5, 97.5], 0)[:, Q].round(4).tolist())


diff(U[T], U[T + "_h"], "full_minus_half")
for arm in (A, B):
    diff(U[T], U[arm], f"mix_minus_{arm}")
    diff(U[T + "_h"], U[arm], f"mixh_minus_{arm}")
Z = 6
for arm in (A, B):
    d = (U[T] - U[arm])[:, :, Z]
    out[f"z500_mix_minus_{arm}"] = d.mean(0)[Q].round(4).tolist()
print(json.dumps(out, indent=1))
json.dump(out, open(sys.argv[4], "w"), indent=1)
