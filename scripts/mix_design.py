"""Which kernels to mix, from the mixing matrices of mix_matrix.py: the weights are fitted on one year and scored on another.

    CRPS(w) = w.a - 1/2 w^T D w      (convex on the simplex: -D is positive definite on sum w = 0, the energy distance)
    mse of the mean = w^T C w ,  spread^2 = sum_i w_i (V_i + M_ii) - w^T M w

    value of kernel j for a mixture w (the slope of the CRPS when a little of K_j is mixed in):
        s_j = [a_j - (D w)_j] - [w.a - w^T D w]       < 0 : worth mixing in
      = (how far K_j is from the truth  -  how far K_j is from the mixture)  -  the same for the mixture itself

    python scripts/mix_design.py --fit runs/fpk/cond/mixmat_2020.npz --test runs/fpk/cond/mixmat_2017.npz \\
        --base core_gngmz8W:devk_band_sphW_p core_ngmz8W:devk_band_sphNW_p --json docs/results/cond/mix_design.json
"""
import argparse
import json

import numpy as np
from scipy.optimize import minimize

ap = argparse.ArgumentParser()
ap.add_argument("--fit", required=True)
ap.add_argument("--test", required=True)
ap.add_argument("--base", nargs="+", required=True, help="the mixture of now (equal weights)")
ap.add_argument("--drop", nargs="*", default=[], help="kernels left out of the search")
ap.add_argument("--sets", nargs="*", default=[], help="mixtures reported in full: kernel names joined by ','")
ap.add_argument("--json", default="")
A = ap.parse_args()
F, T = np.load(A.fit), np.load(A.test)
names = [str(x) for x in T["names"]]
assert names == [str(x) for x in F["names"]]
ch = [str(x) for x in T["channels"]]
std = T["std"]
SC = [i for i in range(len(ch)) if ch[i] != "tisr"]
Z = ch.index("z500")
leads = T["leads"].tolist()
K, NL = len(names), len(leads)
use = [i for i, n in enumerate(names) if n not in A.drop]


def crps_ic(d, w):
    """w (NL, K) -> (N, NL): mean over the 65 channels."""
    a = np.einsum("lk,nklc->nlc", w, d["a"])
    q = np.einsum("lp,npqlc,lq->nlc", w, d["D"], w)
    return (a - 0.5 * q)[:, :, SC].mean(-1)


def mse_ic(d, w):
    return np.einsum("lp,npqlc,lq->nlc", w, d["C"], w)                              # (N, NL, 66)


def spread2_ic(d, w):
    diag = np.einsum("nkklc->nklc", d["Mm"])
    return np.einsum("lk,nklc->nlc", w, d["V"] + diag) - np.einsum("lp,npqlc,lq->nlc", w, d["Mm"], w)


def summary(d, w):
    m = mse_ic(d, w).mean(0)
    sp = spread2_ic(d, w).mean(0)
    return dict(crps=crps_ic(d, w).mean(0).round(4).tolist(), z500=(np.sqrt(m[:, Z]) * std[Z]).round(1).tolist(),
                nrmse=np.sqrt(m)[:, SC].mean(-1).round(4).tolist(),
                spread_skill=np.sqrt(sp[:, SC] / np.maximum(m[:, SC], 1e-30)).mean(-1).round(3).tolist())


def fit_w(d, idx, lead):
    """The CRPS-optimal weights of the kernels idx at one lead (the simplex)."""
    a = d["a"][:, idx][:, :, lead][:, :, SC].mean((0, 2))
    D = d["D"][:, idx][:, :, idx][:, :, :, lead][..., SC].mean((0, 3))
    f = lambda w: float(w @ a - 0.5 * w @ D @ w)  # noqa: E731
    g = lambda w: a - D @ w  # noqa: E731
    n = len(idx)
    best = None
    for w0 in [np.full(n, 1.0 / n)] + [np.eye(n)[i] * 0.7 + 0.3 / n for i in range(n)]:
        r = minimize(f, w0, jac=g, method="SLSQP", bounds=[(0, 1)] * n, constraints=[dict(type="eq", fun=lambda w: w.sum() - 1, jac=lambda w: np.ones(n))],
                     options=dict(maxiter=500, ftol=1e-12))
        if best is None or r.fun < best.fun:
            best = r
    w = np.clip(best.x, 0, None)
    return w / w.sum()


def full(idx, d=None):
    w = np.zeros((NL, K))
    for l in range(NL):
        w[l, idx] = fit_w(F if d is None else d, idx, l)
    return w


def onehot(i):
    w = np.zeros((NL, K))
    w[:, i] = 1
    return w


rng = np.random.default_rng(0)
nT = T["a"].shape[0]
draws = rng.integers(0, nT, size=(2000, nT))


def diff_ci(w, w0):
    x = crps_ic(T, w) - crps_ic(T, w0)                                             # (N, NL)
    bs = x[draws].mean(1)
    return [[round(float(v), 4) for v in (x.mean(0)[l], *np.percentile(bs[:, l], [2.5, 97.5]))] for l in range(NL)]


base_i = [names.index(b) for b in A.base]
wb = np.zeros((NL, K))
wb[:, base_i] = 1.0 / len(base_i)
out = dict(meta=dict(fit=A.fit, test=A.test, names=names, leads=leads, base=A.base, fit_ics=int(F["a"].shape[0]), test_ics=int(nT)), singles={}, value={}, pairs={})
print("leads", leads)
print(f"{'base mixture':44s}", summary(T, wb))
out["base"] = summary(T, wb)
for i, n in enumerate(names):
    out["singles"][n] = summary(T, onehot(i))
    print(f"{n:44s}", out["singles"][n])
# the value of every kernel for the base mixture (fit year and test year), and the best two-part mixture (1 - e) base + e K_j
print("\nvalue for the base mixture: slope s_j (x 1e3; < 0 = worth mixing in), the fitted share e, CRPS change on the test year [95 %]")
for i, n in enumerate(names):
    if i in base_i:
        continue
    val = {}
    for key, d in (("fit", F), ("test", T)):
        a, D = d["a"][..., SC].mean((0, 3)), d["D"][..., SC].mean((0, 4))          # (K, NL), (K, K, NL)
        Dw = np.einsum("pql,lq->pl", D, wb)
        mixv = np.einsum("lk,kl->l", wb, a) - np.einsum("lp,pl->l", wb, Dw)
        val[key] = (a[i] - Dw[i] - mixv).round(5).tolist()
    # e fitted on the fit year per lead (a 1-d convex problem)
    e = np.zeros(NL)
    for l in range(NL):
        best = min(np.linspace(0, 1, 201), key=lambda t: crps_ic(F, (1 - t) * wb + t * onehot(i))[:, l].mean())
        e[l] = best
    w = (1 - e)[:, None] * wb + e[:, None] * onehot(i)
    out["value"][n] = dict(slope=val, share=e.round(3).tolist(), test=summary(T, w), crps_diff=diff_ci(w, wb))
    print(f"{n:44s} slope fit {np.round(np.array(val['fit']) * 1e3, 2)} test {np.round(np.array(val['test']) * 1e3, 2)}  share {e.round(2)}  "
          f"day10 {out['value'][n]['crps_diff'][-1]}  z500 {out['value'][n]['test']['z500'][-1]}")
# all kernels together, and the greedy path
wf = full(use)
out["optimum"] = dict(kernels=[names[i] for i in use], weights={names[i]: wf[:, i].round(3).tolist() for i in use}, test=summary(T, wf), fit=summary(F, wf),
                      crps_diff=diff_ci(wf, wb))
print("\noptimum over", [names[i] for i in use])
for i in use:
    print(f"  {names[i]:42s} w {wf[:, i].round(3)}")
print("  test", out["optimum"]["test"], "\n  diff vs base", out["optimum"]["crps_diff"])
sel, path = [], []
rest = list(use)
while rest:
    cand = {i: crps_ic(F, full(sel + [i])).mean() for i in rest}
    i = min(cand, key=cand.get)
    sel.append(i)
    rest.remove(i)
    w = full(sel)
    path.append(dict(added=names[i], fit_crps=round(float(cand[i]), 5), test=summary(T, w), crps_diff=diff_ci(w, wb)[-1]))
    print(f"greedy +{names[i]:40s} fit {cand[i]:.5f}  test day10 crps {path[-1]['test']['crps'][-1]}  z500 {path[-1]['test']['z500'][-1]}  ss {path[-1]['test']['spread_skill'][-1]}  diff {path[-1]['crps_diff']}")
out["greedy"] = path
# where every kernel sits (test year): its distance to the truth, its own width, its distance to the base mixture
GRP = {"z": [i for i in SC if ch[i][0] == "z"], "q": [i for i in SC if ch[i][0] == "q"], "t": [i for i in SC if ch[i][0] == "t" and ch[i][1:].isdigit()],
       "u": [i for i in SC if ch[i][0] == "u"], "v": [i for i in SC if ch[i][0] == "v"], "surface": [i for i in SC if ch[i] in ("t2m", "sp", "tcwv", "msl", "skt")]}
a, D = T["a"][..., SC].mean((0, 3)), T["D"][..., SC].mean((0, 4))
Dw = np.einsum("pql,lq->pl", D, wb)
wDw = np.einsum("lp,pl->l", wb, Dw)
out["geometry"] = {n: dict(a=a[i].round(4).tolist(), width=D[i, i].round(4).tolist(), to_base=Dw[i].round(4).tolist(),
                           crps=(a[i] - 0.5 * D[i, i]).round(4).tolist(), energy_to_base=(2 * Dw[i] - D[i, i] - wDw).round(4).tolist()) for i, n in enumerate(names)}
print("\ngeometry on the test year (day 10): a = E|X - y|, width = E|X - X'|, to_base = E|X - B|, energy distance to the base")
for n, g in out["geometry"].items():
    print(f"  {n:42s} a {g['a'][-1]:.4f}  width {g['width'][-1]:.4f}  to_base {g['to_base'][-1]:.4f}  crps {g['crps'][-1]:.4f}  energy {g['energy_to_base'][-1]:.4f}")


def crps_grp(d, w):
    aa = np.einsum("lk,nklc->nlc", w, d["a"])
    q = np.einsum("lp,npqlc,lq->nlc", w, d["D"], w)
    x = (aa - 0.5 * q).mean(0)
    return {g: x[:, ix].mean(-1).round(4).tolist() for g, ix in GRP.items()}


def const_w(idx):
    """One weight vector for all the leads (the mean CRPS over the leads, fit year)."""
    aa = F["a"][:, idx][..., SC].mean((0, 2, 3))
    DD = F["D"][:, idx][:, :, idx][..., SC].mean((0, 3, 4))
    n = len(idx)
    r = minimize(lambda w: float(w @ aa - 0.5 * w @ DD @ w), np.full(n, 1.0 / n), jac=lambda w: aa - DD @ w, method="SLSQP", bounds=[(0, 1)] * n,
                 constraints=[dict(type="eq", fun=lambda w: w.sum() - 1, jac=lambda w: np.ones(n))], options=dict(maxiter=500, ftol=1e-12))
    w1 = np.clip(r.x, 0, None)
    w = np.zeros((NL, K))
    w[:, idx] = w1 / w1.sum()
    return w


out["base"]["groups"] = crps_grp(T, wb)
out["sets"] = {}
for spec in A.sets:
    idx = [names.index(n) for n in spec.split(",")]
    we = np.zeros((NL, K))
    we[:, idx] = 1.0 / len(idx)
    res = {}
    for kind, w in (("fitted", full(idx)), ("const", const_w(idx)), ("equal", we)):
        res[kind] = dict(weights={names[i]: w[:, i].round(3).tolist() for i in idx}, test=summary(T, w), groups=crps_grp(T, w), crps_diff=diff_ci(w, wb))
        print(f"set {spec} [{kind}]  w(day10) {[round(float(w[-1, i]), 2) for i in idx]}  test {res[kind]['test']}\n    diff vs base {res[kind]['crps_diff']}")
    out["sets"][spec] = res
if A.json:
    json.dump(out, open(A.json, "w"), indent=1)
