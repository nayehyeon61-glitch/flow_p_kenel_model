"""Fixed-weight mixtures on the two held-out years (no fitting): CRPS change to the base per lead with a bootstrap interval,
z500 of the mean, spread / skill, the change by variable group, and a scan of the clim share on top.
  python scripts/mix_fixed.py A.npz B.npz --mix "W_p=0.3,NW_p=0.3,sdress45=0.4" [--clim 0,0,0,0.05,0.1] [--scan clim]
Names are matched by substring (the first kernel whose name contains the key and no '@' unless the key has one)."""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("npz", nargs="+")
ap.add_argument("--mix", nargs="+", default=[])
ap.add_argument("--base", default="sphW_p=0.5,sphNW_p=0.5")
ap.add_argument("--clim", default="", help="a clim share per lead put on top of every mixture (the rest is scaled by 1 - e)")
ap.add_argument("--scan", default="", help="a kernel whose share is scanned on top of the first mixture at every lead")
ap.add_argument("--infinite-pool", action="store_true")
A = ap.parse_args()


def load(f):
    v = np.load(f)
    d = {k: v[k] for k in v.files}
    names = [str(x) for x in d["names"]]
    K = len(names)
    if not A.infinite_pool:
        if "atoms" in d:
            cnt = d["atoms"].astype(float)
        else:
            pool, ics, pd = np.atleast_1d(d["pool"]), d["ics"], np.arange(0, 356, 2)
            cnt = np.zeros((len(ics), K))
            for i, n in enumerate(names):
                kind = n.split(":")[0]
                if kind == "dress":
                    cnt[:, i] = len(pd) * len(pool)
                elif kind == "sdress":
                    dd = np.abs(pd[None] - ics[:, None])
                    cnt[:, i] = (np.minimum(dd, 365 - dd) <= 45).sum(1) * len(pool)
                elif n == "clim":
                    cnt[:, i] = 16 * 31
        for i in range(K):
            if cnt[:, i].min() > 0:
                d["D"][:, i, i] *= (1 - 1 / cnt[:, i])[:, None, None]
        d["cnt"] = cnt
    return d, names


def find(names, key):
    c = [i for i, n in enumerate(names) if key in n and (("@" in key) or ("@" not in n))]
    ex = [i for i in c if names[i] == key or names[i].split(":")[0] == key]
    c = ex or c
    assert c, (key, names)
    return c[0]


def wvec(names, spec, NL):
    w = np.zeros((NL, len(names)))
    for part in spec.split(","):
        k, v = part.split("=")
        w[:, find(names, k)] = float(v)
    assert abs(w[0].sum() - 1) < 1e-6, (spec, w[0].sum())
    return w


for f in A.npz:
    d, names = load(f)
    ch = [str(x) for x in d["channels"]]
    std = d["std"]
    SC = [i for i in range(len(ch)) if ch[i] != "tisr"]
    Z = ch.index("z500")
    NL = len(d["leads"])
    GRP = {"z": [i for i in SC if ch[i][0] == "z"], "q": [i for i in SC if ch[i][0] == "q"], "t": [i for i in SC if ch[i][0] == "t" and ch[i][1:].isdigit()],
           "u": [i for i in SC if ch[i][0] == "u"], "v": [i for i in SC if ch[i][0] == "v"], "surface": [i for i in SC if ch[i] in ("t2m", "sp", "tcwv", "msl", "skt")]}
    N = d["a"].shape[0]
    rng = np.random.default_rng(0)
    draws = rng.integers(0, N, size=(2000, N))

    def crps_c(w):
        a = np.einsum("lk,nklc->nlc", w, d["a"])
        q = np.einsum("lp,npqlc,lq->nlc", w, d["D"], w, optimize=True)
        return a - 0.5 * q                                                          # (N, NL, C)

    def summ(w):
        m = np.einsum("lp,npqlc,lq->nlc", w, d["C"], w, optimize=True).mean(0)
        diag = np.einsum("nkklc->nklc", d["Mm"])
        sp = (np.einsum("lk,nklc->nlc", w, d["V"] + diag) - np.einsum("lp,npqlc,lq->nlc", w, d["Mm"], w, optimize=True)).mean(0)
        return (np.sqrt(m[:, Z]) * std[Z]).round(1), np.sqrt(sp[:, SC] / np.maximum(m[:, SC], 1e-30)).mean(-1).round(3), np.sqrt(m)[:, SC].mean(-1).round(4)

    wb = wvec(names, A.base, NL)
    cb = crps_c(wb)
    zb, sb, nb = summ(wb)
    print(f"== {f}  year {int(d['year'])}  ICs {N}  leads {d['leads'].tolist()}")
    if "cnt" in d:
        print("   atoms:", {n.split(':')[0]: round(float(d['cnt'][:, i].mean()), 1) for i, n in enumerate(names) if d['cnt'][:, i].min() > 0})
    print(f"   base crps {cb[:, :, SC].mean((0, 2)).round(4)}  z500 {zb}  ss {sb}  nrmse {nb}")
    print("   base by group, day 10:", {g: round(float(cb[:, -1, ix].mean()), 4) for g, ix in GRP.items()})
    ecl = np.array([float(x) for x in A.clim.split(",")]) if A.clim else None
    first = None
    for spec in A.mix:
        w = wvec(names, spec, NL)
        variants = [("", w)]
        if ecl is not None:
            wc = (1 - ecl)[:, None] * w
            wc[:, find(names, "clim")] += ecl
            variants.append((" +clim" + A.clim, wc))
        for tag, ww in variants:
            if first is None:
                first = ww
            c = crps_c(ww)
            x = (c - cb)[:, :, SC].mean(-1)                                         # (N, NL)
            bs = x[draws].mean(1)
            lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
            z, s, nr = summ(ww)
            t = x.mean(0) / (x.std(0, ddof=1) / np.sqrt(N))
            print(f"-- {spec}{tag}")
            print("   diff x1e4 " + "  ".join(f"{x.mean(0)[l] * 1e4:+.1f} [{lo[l] * 1e4:+.1f},{hi[l] * 1e4:+.1f}]" for l in range(NL)))
            print(f"   rel % {(100 * x.mean(0) / cb[:, :, SC].mean((0, 2))).round(2)}  t {t.round(1)}  dates improved % {(100 * (x < 0).mean(0)).round(0)}")
            print(f"   z500 {z}  ss {s}  nrmse {nr}")
            print("   by group % (day 5 | day 10): " + "  ".join(
                f"{g} {100 * (c - cb)[:, 2, ix].mean() / cb[:, 2, ix].mean():+.2f}|{100 * (c - cb)[:, -1, ix].mean() / cb[:, -1, ix].mean():+.2f}" for g, ix in GRP.items()))
    if A.scan and first is not None:
        j = find(names, A.scan)
        print(f"-- scan of the share of {names[j]} on top of the first mixture: diff to the base x1e4 (z500)")
        for e in (0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4):
            ww = (1 - e) * first
            ww[:, j] += e
            x = (crps_c(ww) - cb)[:, :, SC].mean((0, 2))
            z, s, nr = summ(ww)
            print(f"   e={e:.2f}  " + "  ".join(f"{x[l] * 1e4:+.1f} ({z[l]:.0f})" for l in range(NL)))
