"""The convolutional nets of the kernel and the flow, and how they run on a GPU (the + version, 1.5 deg).

    make_net(c_in, c_out, hidden, blocks)            ClimODE's residual trunk on the data's grid (as before)
    make_net(c_in, c_out, ..., unet=(64, 128, 192))  the multi-scale trunk of the tendency core + (pgcfm.kai_tend
                                                     .UNetNet): the data's grid and its 2x and 4x coarsening, so a
                                                     net on the 1.5 deg grid sees about as far as the plain trunk
                                                     sees on the 5.625 deg grid (its coarsest level has that cell)
    run_net(net, x, owner)                           the forward pass with the owner's ``net_dtype`` (bf16 autocast
                                                     of the net alone, the output cast back to fp32) and
                                                     ``net_ckpt`` (activations recomputed in the backward pass)

Both trunks end in a zero-initialized head and have GroupNorm in place of BatchNorm (``stateless_norm``), so a new
net starts as the kernel / flow without it. Only the nets run in bf16: the spherical harmonics, the entropy, the log
scores and the states stay fp32, because they are never inside the autocast region.
"""

from typing import Sequence

import torch
from torch.utils.checkpoint import checkpoint

from .env import PGFM  # noqa: F401  (sys.path)
from pgcfm.core import Net  # noqa: E402
from pgcfm.kai_tend import UNetNet, stateless_norm  # noqa: E402
from pgcfm.seqdct import _NetCfg  # noqa: E402


def make_net(c_in: int, c_out: int, hidden: int, blocks: Sequence[int], unet: Sequence[int] = (), reps: int = 2):
    if unet:
        return stateless_norm(UNetNet(c_in, c_out, tuple(int(d) for d in unet), int(reps)))
    return stateless_norm(Net(c_in, c_out, _NetCfg(hidden, tuple(blocks))))


def set_run(module: torch.nn.Module, dtype=None, ckpt: bool = False):
    """Set how the nets of ``module`` (and of every fpk module inside it) run: see ``run_net``."""
    for m in module.modules():
        if hasattr(m, "net_dtype"):
            m.net_dtype, m.net_ckpt = dtype, bool(ckpt)
    return module


def run_net(net: torch.nn.Module, x: torch.Tensor, owner=None) -> torch.Tensor:
    """net(x) in fp32 out. ``owner.net_dtype`` (None or torch.bfloat16): autocast of this forward pass only.
    ``owner.net_ckpt``: with a gradient, the activations are not kept but recomputed in the backward pass (a
    pure function of x and the weights, so recomputing gives the same values)."""
    dtype, ck = getattr(owner, "net_dtype", None), getattr(owner, "net_ckpt", False)

    def f(t):
        if dtype is None:
            return net(t)
        with torch.autocast(t.device.type, dtype=dtype):
            return net(t).float()

    if ck and torch.is_grad_enabled() and (x.requires_grad or any(p.requires_grad for p in net.parameters())):
        return checkpoint(f, x, use_reentrant=False)
    return f(x)


__all__ = ["make_net", "run_net", "set_run"]
