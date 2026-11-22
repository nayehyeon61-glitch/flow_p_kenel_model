"""Core and kernel learning along instead of frozen: ``devk --core-lr`` and ``devtune --kernel-lr`` (scripts/run_fpk.py).

What is checked here
    core     its trained weights get a gradient only with ``through`` >= 1 and only along the mean's path (the pairs of
             the deviations and the memory's warm-up are constants); through the trainer the tuned core is a checkpoint
             of its own in the format the core's loader reads, the kernel is recorded on it, and the original core's
             file is never written
    kernel   the flow-matching loss sends nothing to the kernel's parameters; the gradient ``devtune --kernel-lr``
             hands to the optimiser for them is, by hand in one graph, that of sum_L ES_L / (n_L D) + kernel_lam x
             the kernel's own loss / D (the two-stage chain rule of the trainer, with and without --chunk), and the
             anchor of the network (--lam) does not change it; the tuned kernel and the flow on it are both written
"""

import copy
import hashlib
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from test_fpk import H, W, _cli, _day_inputs, _flat_grad, _flow_with_a_head, _kernel, _logs, mcore  # noqa: E402,F401

LAT = np.linspace(-78.75, 78.75, H)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()  # noqa: E731
REAL = dict(correct=True, noise="sphere", band_scale=True)       # the options of the kernel of the runs (devk_band_sphW)


def _away_from_start(Dk, std=0.05, seed=5):
    """The kernel's own parameters away from their start values (zero heads, r = 0): every part has a gradient."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in Dk.named_parameters():
            if not n.startswith("core."):
                p.add_(std * torch.randn(p.shape, generator=g))
    return Dk


# ------------------------------------------------------------------------------------------------------------ core
def test_core_weights_get_the_gradient_of_the_mean_path_only(mcore):
    """``train_core``: the core's weights are the core module's parameters and no part of the kernel's own; with
    through = 0 the score has no gradient for them, with through = 1 it has; the memory's warm-up before the IC and
    the pairs of the deviations carry no graph; and switching it off freezes them again."""
    from fpk.devkernel import core_weights
    core = copy.deepcopy(mcore)                       # the module's fixture stays frozen
    Dk = _away_from_start(_kernel(core, **REAL))
    assert not any(p.requires_grad for p in core.parameters())
    own = Dk.own_parameters()
    assert own and not any(n.startswith("core.") for n in own) and all(p.requires_grad for p in own.values())
    cw = Dk.train_core()
    assert list(cw) == [n for n, _ in core.named_parameters()] == list(core_weights(core)) and len(cw) > 4
    assert all(p.requires_grad for p in cw.values()) and not set(map(id, cw.values())) & set(map(id, own.values()))
    xd, doy, tis, hist = _day_inputs()
    w = torch.ones(H, 1)
    truth = torch.randn(2, 66, H, W, generator=torch.Generator().manual_seed(3))

    def score(through, days=2):
        Dk.through = through
        e = Dk.start(xd, doy, hist, torch.Generator().manual_seed(1))
        assert not any(torch.is_tensor(getattr(core, k, None)) and getattr(core, k).requires_grad for k in ("m", "prev", "fc"))
        xb, tot = xd, 0.0
        for k in range(days):
            xb, e, Hn, _ = Dk.day(xb, e, doy + k, tis[:, k], k, torch.Generator().manual_seed(2 + k))
            assert not Dk.last["p"].requires_grad                 # the pairs: no graph, with or without the mean's
            tot = tot + Dk.lead_loss(Hn, xb, e, truth, w, 0.5)[0]
            e = e.detach()
            if not through:
                xb = xb.detach()
                Dk.detach_state()
        return tot

    g0 = torch.autograd.grad(score(0), list(cw.values()), allow_unused=True)
    assert all(q is None or float(q.abs().max()) == 0 for q in g0)
    g1 = torch.autograd.grad(score(2), list(cw.values()), allow_unused=True)
    got = {n: float(q.abs().max()) for n, q in zip(cw, g1) if q is not None}
    assert len(got) > 4 and all(np.isfinite(v) for v in got.values()) and max(got.values()) > 1e-6
    assert any(v > 0 for n, v in got.items() if n.startswith("net.")), sorted(got)       # the tendency network
    # the kernel's own parameters keep their gradient next to the core's
    gk = torch.autograd.grad(score(2), list(own.values()), allow_unused=True)
    assert sum(q is not None and float(q.abs().max()) > 0 for q in gk) > 4
    Dk.train_core(False)
    assert not any(p.requires_grad for p in core.parameters())
    assert not Dk.train().core.training                                                   # the core stays in eval mode


def _devk_cli(mcore, tmp_path, monkeypatch):
    """The trainer ``devk`` on a copy of the fixture's core, saved as the core checkpoint c.pt, with one fixed batch."""
    R, _, _ = _cli(mcore, tmp_path, monkeypatch, fixed=True)
    core = copy.deepcopy(mcore)
    torch.save(dict(state={k: v.clone() for k, v in core.state_dict().items()}, core_opts=dict(hidden=8), note="original"),
               tmp_path / "c.pt")
    monkeypatch.setattr(R, "load_core", lambda a: (None, SimpleNamespace(lat=LAT, std=np.ones(67)), core))
    base = ["devk", "--out", str(tmp_path), "--core-tag", "c", "--days", "2", "--batch", "2", "--pairs", "2",
            "--val-batches", "1", "--hidden", "8", "--smooth", "3", "--log-every", "1", "--correct"]
    return R, core, base


def test_devk_core_lr_writes_a_core_of_its_own_and_never_the_original(mcore, tmp_path, monkeypatch, capsys):
    """``devk --core-lr`` through the trainer: the weights move, the tuned core <core-tag>_<tag>.pt has the original
    checkpoint's format (its options and buffers, the trained weights replaced) and is the state of the logged
    core_rel; the kernel is recorded on the tuned core and refused on the original; c.pt is not written; the anchor
    keeps the weights nearer; and the refusals (no --through, the core's own tag, another run's file)."""
    R, core, base = _devk_cli(mcore, tmp_path, monkeypatch)
    th0 = {k: v.clone() for k, v in core.state_dict().items()}
    names = [n for n, _ in core.named_parameters()]
    h0 = sha(tmp_path / "c.pt")
    capsys.readouterr()
    R.main(base + ["--tag", "u", "--through", "2", "--core-lr", "1e-3", "--iters", "3"])
    logs = _logs(capsys)
    head = [q for q in logs if "core_tuned" in q][0]
    assert head["core_tuned"] == "c_u" and head["tuned_from"] == "c" and head["core_weights"] == len(names)
    assert head["core_params"] == sum(th0[n].numel() for n in names) and head["through"] == 2
    its = [q for q in logs if "it" in q]
    assert len(its) == 3 and all(q["core_rel"] > 0 and q["core_anchor"] == 0 == q["core_pull"] and q["core_grad"] > 0 for q in its)
    assert sha(tmp_path / "c.pt") == h0
    ck = torch.load(tmp_path / "c_u.pt", weights_only=False)
    assert ck["tuned_from"] == "c" and ck["devk"]["tag"] == "u" and ck["core_opts"] == dict(hidden=8) and ck["note"] == "original"
    assert list(ck["state"]) == list(th0)
    assert all(torch.equal(ck["state"][k], th0[k]) for k in th0 if k not in names)       # the buffers: the original's
    assert sum(not torch.equal(ck["state"][k], th0[k]) for k in names) > 4               # the trained weights moved
    # the saved core is the state of the best validation: its distance is that line's core_rel
    saved = [q for i, q in enumerate(its) if q["best"] < (its[i - 1]["best"] if i else float("inf"))][-1]
    rel = (sum(float((ck["state"][k] - th0[k]).pow(2).sum()) for k in names) / sum(float(th0[k].pow(2).sum()) for k in names)) ** 0.5
    assert abs(rel - saved["core_rel"]) < 2e-6, (rel, saved["core_rel"])
    fresh = copy.deepcopy(mcore)
    fresh.load_state_dict(ck["state"])                                                    # as the core's loader does
    kk = torch.load(tmp_path / "u.pt", weights_only=False)
    assert kk["core_tag"] == "c_u" and kk["core_from"] == "c" and kk["train"]["core_lr"] == 1e-3
    assert not any(k.startswith("core.") for k in kk["state"])
    with pytest.raises(SystemExit, match="fitted on core c_u"):
        R.load_devk(SimpleNamespace(out=str(tmp_path), core_tag="c", device="cpu"), fresh, "u")
    Dk = R.load_devk(SimpleNamespace(out=str(tmp_path), core_tag="c_u", device="cpu"), fresh, "u")
    xd, doy, tis, hist = _day_inputs()
    ens, ent = Dk.rollout(xd, doy, tis, 2, hist=hist, gen=torch.Generator().manual_seed(4))
    assert torch.isfinite(ens).all() and torch.isfinite(ent).all()

    def rel_of(extra, tag):
        core.load_state_dict(th0)                                                         # every run starts at c.pt
        capsys.readouterr()
        R.main(base + ["--tag", tag, "--through", "2", "--core-lr", "1e-3", "--iters", "3"] + extra)
        return [q for q in _logs(capsys) if "it" in q][-1]
    free, held = rel_of([], "u"), rel_of(["--core-anchor", "1e5"], "v")                   # (the same tag again: its own file)
    assert free["core_rel"] == its[-1]["core_rel"] and 0 < held["core_rel"] < 0.7 * free["core_rel"]
    assert abs(held["core_anchor"] - 1e5 * held["core_rel"] ** 2) < 0.02 * held["core_anchor"] + 1e-6
    # the pull is added after the loss's clip: the logged loss gradient is the same run's own (not the anchor's),
    # and the pull's norm is 2 W core_rel / |theta0|
    n0 = sum(float(th0[k].pow(2).sum()) for k in names) ** 0.5
    assert abs(held["core_pull"] - 2e5 * held["core_rel"] / n0) < 0.02 * held["core_pull"] and held["core_grad"] < 0.1 * held["core_pull"]
    assert sha(tmp_path / "c.pt") == h0 and os.path.exists(tmp_path / "c_v.pt")
    core.load_state_dict(th0)
    for bad, msg in ((["--tag", "w", "--core-lr", "1e-3"], "--through"),                           # no path to learn from
                     (["--tag", "w", "--through", "2", "--core-lr", "1e-3", "--core-out", "c"], "never overwritten"),
                     (["--tag", "w", "--through", "2", "--core-lr", "1e-3", "--core-out", "w"], "two files"),
                     (["--tag", "w", "--through", "2", "--core-anchor", "1"], "belong to --core-lr"),   # a switch that is off
                     (["--tag", "w", "--through", "2", "--core-out", "c_w"], "belong to --core-lr"),
                     (["--tag", "w", "--through", "2", "--core-lr=-1e-3"], "belong to --core-lr"),
                     (["--tag", "w", "--fresh-best"], "belongs to --init"),
                     (["--tag", "w", "--through", "2", "--core-lr", "1e-3", "--core-out", "c_u"], "not written by this run")):
        with pytest.raises(SystemExit, match=msg):
            R.main(base + ["--iters", "1"] + bad)
    assert not os.path.exists(tmp_path / "w.pt") and sha(tmp_path / "c.pt") == h0
    # frozen (the default): the core's weights are not touched and no core is written
    R.main(base + ["--tag", "z", "--through", "2", "--iters", "2"])
    assert all(torch.equal(v, th0[k]) for k, v in core.state_dict().items()) and not os.path.exists(tmp_path / "c_z.pt")
    assert torch.load(tmp_path / "z.pt", weights_only=False)["core_tag"] == "c"
    assert not any(p.requires_grad for p in mcore.parameters())


def test_devk_fresh_best_keeps_a_trained_state(mcore, tmp_path, monkeypatch, capsys):
    """``--init`` counts the start as a candidate: a training that only makes the validation worse (here every step
    also takes 3 off the entropy residual's bias: a spread e^3 too small) leaves the start in the checkpoint.
    ``--fresh-best`` takes the best of the trained states, so a run whose loss is not the validation's score (--fdt
    and its penalty) is not discarded as a whole."""
    R, core, base = _devk_cli(mcore, tmp_path, monkeypatch)
    R.main(base + ["--tag", "a", "--iters", "2"])
    a = torch.load(tmp_path / "a.pt", weights_only=False)["state"]
    made, real = [], R.EntropyDeviationKernel

    def make(*a_, **k_):
        made.append(real(*a_, **k_))
        return made[-1]

    class Worse(torch.optim.AdamW):
        def step(self, closure=None):
            out = super().step(closure)
            with torch.no_grad():
                made[-1].r_bias.sub_(3.0)
            return out
    monkeypatch.setattr(R, "EntropyDeviationKernel", make)
    monkeypatch.setattr(R.torch.optim, "AdamW", Worse)
    out = {}
    for tag, extra in (("b", []), ("f", ["--fresh-best"])):
        capsys.readouterr()
        R.main(base + ["--tag", tag, "--init", "a", "--iters", "2", "--lr", "1e-3"] + extra)
        logs = _logs(capsys)
        start = [q for q in logs if "devk_val_start" in q][0]["devk_val_start"]
        out[tag] = (sum(start["ign"]) / 2, [q for q in logs if "it" in q], torch.load(tmp_path / f"{tag}.pt", weights_only=False)["state"])
    (s_b, lb, b), (s_f, lf, f) = out["b"], out["f"]
    score = lambda q: sum(q["val_ign"]) / 2  # noqa: E731
    assert s_b == s_f and [score(q) for q in lb] == [score(q) for q in lf] and min(map(score, lb)) > s_b + 1e-3   # only worse
    assert all(abs(q["best"] - s_b) < 1e-3 for q in lb) and all(torch.equal(a[k], b[k]) for k in a)          # the start stayed
    assert abs(lf[-1]["best"] - min(map(score, lf))) < 1e-3 and any(not torch.equal(a[k], f[k]) for k in a)  # a trained state


# ---------------------------------------------------------------------------------------------------------- kernel
def test_flow_matching_loss_sends_nothing_to_the_kernel(mcore):
    """The flow-matching loss with the kernel's day in the graph: no gradient to any of the kernel's parameters
    (it would shrink the source), a gradient to the network; on the straight line and on the core's path."""
    for kw in (dict(steps=3), dict(path="core", core_steps=2, core_sub=2)):
        Dk = _away_from_start(_kernel(mcore, **REAL))
        Fm = _flow_with_a_head(Dk, 0.3, **kw)
        kp = list(Dk.own_parameters().values())
        for p in kp:
            p.requires_grad_(True)
        xd, doy, tis, hist = _day_inputs()
        with torch.no_grad():
            e = Dk.start(xd, doy, hist, torch.Generator().manual_seed(1))
        xb, e, Hn, _ = Dk.day(xd, e, doy, tis[:, 0], 0, torch.Generator().manual_seed(2))
        assert e.requires_grad and Hn.requires_grad and xb.requires_grad
        truth = torch.randn(2, 66, H, W, generator=torch.Generator().manual_seed(3))
        an = Fm.day_loss(xd, xb, e, Hn, truth, doy, tis[:, 0], 0, torch.ones(H, 1), torch.Generator().manual_seed(4))
        assert all(q is None for q in torch.autograd.grad(an, kp, allow_unused=True, retain_graph=True))
        assert float(_flat_grad(an, list(Fm.net.parameters())).norm()) > 0


def _devtune_k(mcore, tmp_path, monkeypatch, path):
    """``devtune`` on the fixture, one fixed batch, a kernel with the options of the runs away from its start, the
    flow devflow_t.pt with a head. Returns run_fpk, the arguments, the flows the trainer loaded, the gradients handed
    to the clip (one a call: the network's, then the kernel's) and the random states at the batches."""
    R, _, base = _cli(mcore, tmp_path, monkeypatch, fixed=True)
    Dk = _away_from_start(_kernel(mcore, **REAL))
    with torch.no_grad():
        Dk.r_bias.fill_(1.0)                                    # members of the error's size: the score sees them
    torch.save(dict(state={k: v for k, v in Dk.state_dict().items() if not k.startswith("core.")},
                    cfg=dict(Dk.cfg.__dict__), core_tag="c"), tmp_path / "devk.pt")
    kw = dict(line=dict(steps=3), core=dict(path="core", core_steps=2, core_sub=2))[path]
    Fm = _flow_with_a_head(Dk, 0.3, **kw)
    cfg = dict(Fm.cfg.__dict__)
    if path == "line":
        cfg = {k: cfg[k] for k in ("hidden", "blocks", "sigma_min", "steps", "source")}
    torch.save(dict(state=Fm.net.state_dict(), cfg=cfg, dev_tag="devk", core_tag="c"), tmp_path / "devflow_t.pt")
    live, grads, states = [], [], []
    load, clip, Fixed = R.load_devflow, torch.nn.utils.clip_grad_norm_, R.D.Pairs

    def loaded(*a_, **k_):
        live.append(load(*a_, **k_))
        return live[-1]

    def spy(params, mx):
        params = list(params)
        grads.append(torch.cat([(torch.zeros_like(q) if q.grad is None else q.grad).reshape(-1).clone() for q in params]))
        return clip(params, mx)

    class Rec(Fixed):
        def __call__(self, B):
            states.append(torch.get_rng_state())
            return Fixed.__call__(self, B)
    monkeypatch.setattr(R, "load_devflow", loaded)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
    monkeypatch.setattr(R.D, "Pairs", Rec)
    return R, ["devtune"] + base + ["--flow-tag", "devflow_t", "--batch", "2", "--val-batches", "1"], live, grads, states


@pytest.mark.parametrize("path", ["line", "core"])
def test_devtune_kernel_gradient_by_hand(mcore, tmp_path, monkeypatch, capsys, path):
    """The gradient ``devtune --kernel-lr`` hands to the optimiser for the kernel's parameters, two leads: by hand in
    ONE graph (the kernel's days with their graph, the sampler on them, the truth's deviation from the kernel's
    mean) it is that of sum_L [ES_L / (n_L D) + kernel_lam K_L / D], K_L the kernel's own loss; the network's
    gradient is the one of the frozen kernel; --chunk gives both again; and the anchor of the network (--lam)
    changes the network's gradient and not the kernel's."""
    from fpk.env import DYN
    from fpk.escore import energy_score, make_norm
    R, argv, live, grads, states = _devtune_k(mcore, tmp_path, monkeypatch, path)
    run = argv + ["--iters", "1", "--lr", "0", "--steps", "1", "--tag", "devflow_k"]
    capsys.readouterr()
    R.main(run + ["--days", "2", "--lam", "0", "--kernel-lr", "1e-9"])
    log0 = _logs(capsys)[0]
    scale = log0["scale"]
    assert log0["kernel"] == "devk" and log0["kernel_out"] == "devk_j" and log0["kernel_band_weight"] == 0.5
    Fm = live[0]
    Dk = Fm.dk
    kp = list(Dk.own_parameters().values())
    assert log0["kernel_params"] == sum(p.numel() for p in kp) and all(p.requires_grad for p in kp)
    assert not any(p.requires_grad for p in Dk.core.parameters())
    x, ys, doy, tns, hi = R.D.Pairs.__mro__[1](R.D.load(None, [0]), 0, hist=2, ahead=2)(2)     # the fixed batch
    nrm, w = make_norm("grid", Dk.keep, LAT), R.D.area_weights(LAT)
    torch.set_rng_state(states[-1])
    with torch.no_grad():
        e = Dk.start(x[:, DYN], doy, hi[:, :, DYN])
    xb, es, ka = x[:, DYN], 0.0, 0.0
    for k in range(2):
        x0 = xb
        xg, eg, Hg, _ = Dk.day(xb, e, doy + k, tns[:, k], k)                              # with the kernel's graph
        c = Fm.day_ctx(x0, xg, eg, Hg, doy, tns[:, k], k, grad=True)
        z1 = (ys[:, k][:, DYN] - xg) / Dk.s * Dk.keep
        es = es + energy_score(Fm.day_run(c, 1, grad=True), z1, nrm)[0].mean() / (scale[k] * 2)
        ka = ka + Dk.lead_loss(Hg, xg, eg, ys[:, k][:, DYN], w, 0.5)[0] / 2
        xb, e = xg.detach(), eg.detach()
    flat = lambda loss, ps: torch.cat([(torch.zeros_like(p) if q is None else q).reshape(-1) for p, q in zip(  # noqa: E731
        ps, torch.autograd.grad(loss, ps, allow_unused=True, retain_graph=True))])
    g_es, g_ka, g_net = flat(es, kp), flat(ka, kp), flat(es, list(Fm.net.parameters()))
    gn, gk = grads[0], grads[1]
    small = min(float(g_es.norm()), 0.5 * float(g_ka.norm()))
    assert small > 1e-4 and float((gk - g_es - 0.5 * g_ka).norm()) < 1e-3 * small, (
        float(g_es.norm()), float(g_ka.norm()), float((gk - g_es - 0.5 * g_ka).norm()))
    assert float(g_net.norm()) > 1e-4 and float((gn - g_net).norm()) < 1e-3 * float(g_net.norm())
    # the score alone (--kernel-lam 0), in pieces (--chunk), and the network's gradient without the kernel learning
    R.main(run + ["--days", "2", "--lam", "0", "--kernel-lr", "1e-9", "--kernel-lam", "0"])
    R.main(run + ["--days", "2", "--lam", "0", "--kernel-lr", "1e-9", "--chunk", "2"])
    R.main(run + ["--days", "2", "--lam", "0"])
    assert float((grads[3] - g_es).norm()) < 1e-3 * float(g_es.norm())
    assert float((grads[4] - gn).norm()) < 1e-4 * float(gn.norm()) and float((grads[5] - gk).norm()) < 1e-4 * float(gk.norm())
    assert len(grads) == 7 and float((grads[6] - gn).norm()) < 1e-5 * float(gn.norm())    # (one clip: the network's only)
    # one lead (the anchor's draws come after the lead's members): --lam moves the network's gradient, not the kernel's
    n = len(grads)
    R.main(run + ["--days", "1", "--lam", "0", "--kernel-lr", "1e-9"])
    R.main(run + ["--days", "1", "--lam", "1", "--kernel-lr", "1e-9"])
    (n0, k0), (n1, k1) = grads[n:n + 2], grads[n + 2:n + 4]
    assert float((n1 - n0).norm()) > 0.01 * float(n0.norm()) and float(k0.norm()) > 1e-4
    assert float((k1 - k0).norm()) < 1e-6 * float(k0.norm())


def test_devtune_kernel_lr_writes_the_tuned_kernel_and_the_flow_on_it(mcore, tmp_path, monkeypatch, capsys):
    """``devtune --kernel-lr`` through the trainer on one fixed batch: the score falls, the kernel's parameters move
    and the tuned kernel <kernel>_j.pt (the original's format plus tuned_from) and the flow recorded on it load
    together; the original kernel's file is not written; the log has the kernel's IGN and entropy per lead and its
    anchor; and the refusals (the kernel's own tag, another run's file)."""
    R, argv, live, grads, states = _devtune_k(mcore, tmp_path, monkeypatch, "line")
    src = torch.load(tmp_path / "devk.pt", weights_only=False)
    h0 = sha(tmp_path / "devk.pt")
    run = argv + ["--days", "2", "--lam", "0.5", "--tag", "devflow_j"]
    capsys.readouterr()
    R.main(run + ["--iters", "6", "--lr", "3e-3", "--kernel-lr", "3e-3", "--log-every", "3"])
    logs = _logs(capsys)
    v0, end = logs[0]["devtune_val_start"], logs[-1]
    assert len(v0["kernel_ign"]) == len(v0["kernel_H"]) == len(end["val_kernel_ign"]) == len(end["val_kernel_H"]) == 2
    assert all(np.isfinite(v) for v in v0["kernel_ign"] + end["val_kernel_ign"] + end["val_kernel_H"] + [end["kernel_anchor"]])
    assert end["best"] < sum(v0["es"]) / 2 - 1e-3                                         # the score fell: a tuned state
    assert sha(tmp_path / "devk.pt") == h0
    kj = torch.load(tmp_path / "devk_j.pt", weights_only=False)
    assert kj["tuned_from"] == "devk" and kj["tune"]["tag"] == "devflow_j" and kj["core_tag"] == "c" and kj["cfg"] == src["cfg"]
    assert kj["tune"]["band_weight"] == 0.5 and sorted(kj["state"]) == sorted(src["state"])
    own = [n for n in live[0].dk.own_parameters()]
    assert sum(not torch.equal(kj["state"][n], src["state"][n]) for n in own) > 4         # the kernel moved
    assert all(torch.equal(kj["state"][k], src["state"][k]) for k in src["state"] if k not in own)   # its buffers did not
    fj = torch.load(tmp_path / "devflow_j.pt", weights_only=False)
    assert fj["dev_tag"] == "devk_j" and fj["tuned_from"] == "devflow_t" and fj["core_tag"] == "c"
    monkeypatch.undo()                                                                    # the plain loader
    F2 = R.load_devflow(SimpleNamespace(out=str(tmp_path), core_tag="c", device="cpu"), mcore, "devflow_j", pairs=2)
    assert all(torch.equal(v, kj["state"][k]) for k, v in F2.dk.state_dict().items() if not k.startswith("core."))
    xd, doy, tis, hist = _day_inputs()
    ens, ent = F2.rollout(xd, doy, tis, 2, hist=hist, gen=torch.Generator().manual_seed(4), steps=2)
    assert ens.shape == (2, 2, 2, 66, H, W) and torch.isfinite(ens).all() and torch.isfinite(ent).all()
    assert not any(p.requires_grad for p in mcore.parameters())


def test_devtune_kernel_lr_refusals(mcore, tmp_path, monkeypatch, capsys):
    R, argv, live, grads, states = _devtune_k(mcore, tmp_path, monkeypatch, "line")
    run = argv + ["--days", "1", "--iters", "1", "--kernel-lr", "1e-4"]
    with pytest.raises(SystemExit, match="never overwritten"):
        R.main(run + ["--tag", "devflow_a", "--kernel-out", "devk"])
    for out in ("devflow_a", "devflow_t"):
        with pytest.raises(SystemExit, match="two files"):
            R.main(run + ["--tag", "devflow_a", "--kernel-out", out])
    for off in (["--kernel-out", "devk_x"], ["--kernel-band-weight", "0.2"]):                     # a switch that is off
        with pytest.raises(SystemExit, match="belong to --kernel-lr"):
            R.main(argv + ["--days", "1", "--iters", "1", "--tag", "devflow_a"] + off)
    torch.save(torch.load(tmp_path / "devk.pt", weights_only=False), tmp_path / "devk_o.pt")     # an original kernel
    with pytest.raises(SystemExit, match="not written by this run"):
        R.main(run + ["--tag", "devflow_a", "--kernel-out", "devk_o"])
    assert not os.path.exists(tmp_path / "devflow_a.pt")
    R.main(run + ["--tag", "devflow_a"])
    R.main(run + ["--tag", "devflow_a"])                                                  # its own output: again
    with pytest.raises(SystemExit, match="not written by this run"):
        R.main(run + ["--tag", "devflow_b"])                                              # devk_j.pt is devflow_a's
    R.main(run + ["--tag", "devflow_b", "--kernel-out", "devk_b", "--kernel-band-weight", "0.2"])
    assert torch.load(tmp_path / "devflow_b.pt", weights_only=False)["dev_tag"] == "devk_b"
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]
    # a tuned kernel tuned again: the anchor's band weight is the one stored with it
    capsys.readouterr()
    R.main(argv + ["--days", "1", "--iters", "1", "--kernel-lr", "1e-4", "--flow-tag", "devflow_b", "--tag", "devflow_c"])
    assert [q for q in _logs(capsys) if "devtune_val_start" in q][0]["kernel_band_weight"] == 0.2
