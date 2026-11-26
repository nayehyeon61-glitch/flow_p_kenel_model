"""The 711 stack in the + version (scripts/run_fpk_plus.sh: arch, dump, mcore, cfit, kmix, rep711), on a small grid
with the poles (W = 2 (H - 1), as 121 x 240: the measure core wrote 2 H longitudes before).

    MeasureCoords     the coefficients l < lt and back: exact for a field of degree < lt, the grid's width
    measure + cond    measure_core.py --dump from the record's coefficients (archive_vec.npy), then cond_fit.py fitted on
                      ONE year (the + version's validation year) writing the conditioned mean of both years
    report            report_711.py on cond_fit's json
    chain             run_fpk_plus.sh DRYRUN: the sister cores in both chains, the 711 steps in the chain with G(x),
                      SISTERS=0 leaves the sisters out and runs the stack on the two cores
"""
import json
import os
import subprocess
import sys

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from fpk.env import DYN, contract  # noqa: E402
from fpk.measure import MeasureCoords  # noqa: E402
from fpk.sphere import grid_lat  # noqa: E402

HP, WP = 13, 24


def test_measure_coords_round_trip_on_the_pole_grid():
    lat = grid_lat(HP, WP)
    MC = MeasureCoords(lat, 5)
    assert (MC.H, MC.W) == (HP, WP) and MC.nv == 15 + 10        # l < 5: 15 real parts, 10 imaginary (m > 0)
    g = torch.Generator().manual_seed(0)
    c = torch.complex(torch.randn(3, MC.sht.L, MC.M, generator=g), torch.randn(3, MC.sht.L, MC.M, generator=g))
    c = c * torch.tril(torch.ones(MC.sht.L, MC.M))
    c[:, 5:] = 0                                                # degree < 5
    c[..., 0] = torch.complex(c[..., 0].real, torch.zeros_like(c[..., 0].real))
    f = MC.sht.synthesis(c)                                     # (3, H, W)
    v = MC.to_vec(f[:, None])                                   # (3, 1, NV)
    back = MC.to_field(v[:, 0])
    assert back.shape == (3, HP, WP) and torch.allclose(back, f, atol=1e-4)


def _setup(d, rng):
    lat = grid_lat(HP, WP)
    ch = [contract.CHANNEL_NAMES[c] for c in DYN]
    C = len(ch)
    MC = MeasureCoords(lat, 5)
    days = [60, 60, 60]
    arc = rng.standard_normal((sum(days), C, HP, WP)).astype(np.float32)
    for t in range(1, len(arc)):                                # a persistent record: something for the operator to find
        arc[t] = 0.8 * arc[t - 1] + 0.6 * arc[t]
    np.save(d / "archive_vec.npy", MC.to_vec(arc).numpy())
    json.dump(dict(years=[2000, 2001, 2002], days=days, lt=5, nv=MC.nv, grid=[HP, WP]), open(d / "archive_meta.json", "w"))
    for y, n in ((2016, 16), (2017, 8)):
        o = rng.standard_normal((n, 10, C, HP, WP)).astype(np.float32)
        for t, s in (("cA", 0.3), ("cB", 0.4)):
            np.save(d / f"{y}_{t}.npy", (0.8 * o + s * rng.standard_normal(o.shape)).astype(np.float16))
        np.save(d / f"{y}_o.npy", o.astype(np.float16))
        np.save(d / f"{y}_x0.npy", rng.standard_normal((n, C, HP, WP)).astype(np.float16))
        json.dump(dict(year=y, ics=list(range(0, 2 * n, 2)), cores=["cA", "cB"], leads=10, std=[1.0] * C, lat=lat.tolist(),
                       channels=ch, ic_stride=2), open(d / f"{y}_meta.json", "w"))
    return C


def _run(*args):
    r = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    return r.stdout


def test_measure_continuation_and_the_condition_fitted_on_one_year(tmp_path):
    C = _setup(tmp_path, np.random.default_rng(0))
    M = ["--dir", str(tmp_path), "--years", "2016", "2017", "--days", "1", "3", "10", "--lt", "5", "--npc", "20",
         "--cores", "cA,cB", "--hold", "1", "--threads", "2"]
    _run("scripts/measure_core.py", "--dump", "mc0", "--taus", "0", *M)
    _run("scripts/measure_core.py", "--dump", "mc3", "--taus", "3", "--lags", "1", *M)
    for t in ("mc0", "mc3"):
        x = np.load(tmp_path / f"2017_{t}.npy")
        assert x.shape == (8, 10, C, HP, WP) and np.isfinite(x).all()
        assert np.abs(x[:, [0, 2, 9]]).max() > 0 and np.abs(x[:, 1]).max() == 0     # only the leads asked for
    js = tmp_path / "c.json"
    _run("scripts/cond_fit.py", "--dir", str(tmp_path), "--fit", "2016", "--test", "2017", "--cores", "cA,cB", "--days", "1", "10",
         "--exps", "bands", "vert", "vert+math", "--math", "mc0,mc3", "--save-pred", "cm:vert+math", "--batch", "4", "--threads", "2",
         "--json", str(js))
    out = json.load(open(js))
    r = out["days"]["10"]
    assert set(r) == {"bands", "vert", "vert+math"} and "fit_z500" in r["bands"] and "cv_z500" not in r["bands"]
    o = np.load(tmp_path / "2017_o.npy").astype(np.float32)
    avg = (np.load(tmp_path / "2017_cA.npy").astype(np.float32) + np.load(tmp_path / "2017_cB.npy").astype(np.float32)) / 2
    w = np.cos(np.deg2rad(grid_lat(HP, WP)))[:, None]
    w = w / w.mean()
    z = [contract.CHANNEL_NAMES[c] for c in DYN].index("z500")
    raw = float(np.sqrt(((avg[:, 9, z] - o[:, 9, z]) ** 2 * w).mean()))
    assert r["bands"]["z500"] < raw                             # the strength < 1 shrinks the noisy average
    for y in (2016, 2017):                                       # the conditioned mean of both years (2016 in sample)
        cm = np.load(tmp_path / f"{y}_cm.npy")
        assert cm.shape[2:] == (C, HP, WP) and np.abs(cm[:, [0, 9]]).max() > 0 and np.isfinite(cm).all()
    cm = np.load(tmp_path / "2017_cm.npy").astype(np.float32)
    mine = float(np.sqrt(((cm[:, 9, z] - o[:, 9, z]) ** 2 * w).mean()))
    assert abs(mine - r["vert+math"]["z500"]) < 2e-2 * mine     # the dump is the scored mean (f16)
    md = tmp_path / "p.md"
    _run("scripts/report_711.py", str(md), "--two", str(js), "--four", str(js), "--cores", "cA,cB")
    text = md.read_text()
    assert "measure continuation" in text and "| day 1 | day 10 |" in text


def _dry(tmp_path, **env):
    e = dict(os.environ, DRYRUN="1", DEV="cpu", OUT=str(tmp_path / "o"), DATA=str(tmp_path / "d"), PY=sys.executable, **env)
    r = subprocess.run(["bash", "scripts/run_fpk_plus.sh"], cwd=ROOT, capture_output=True, text=True, env=e)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return r.stdout


def test_chain_has_the_sisters_and_the_711_steps(tmp_path):
    a = _dry(tmp_path)
    for s in ("core_s1:", "arch:", "dump:", "mcore:", "cfit:", "kmix:", "rep711:"):
        assert s in a, s
    assert "--core-tag core_u64_s1" in a and "--seed 1" in a
    assert a.index("] report:") < a.index("] core_s1:") < a.index("] mixrep:") < a.index("] arch:")
    b = _dry(tmp_path, NOPHYS="1")
    assert "core_s1_ng:" in b and "--core-tag core_u64_ng_s1" in b and "arch:" not in b
    c = _dry(tmp_path, SISTERS="0")
    assert "core_s1" not in c and "arch:" in c
