"""10-day ensembles from the physical kernel and the kernel flow, their entropy, and the scores.

Every member is its own trajectory: each day it draws the next day from K(x_t^m, .) (fresh centre and bandwidth
noise), transported by the flow (or not, ``kernel`` ensembles), and goes on from there. The forecast distribution's
spread therefore grows through the equation (the centres follow Phi from perturbed states) and through the learned
bandwidth and flow.

Entropy (all in nats per value, in units of the daily increment std s of each channel, Gaussian moment form):

    H_ens(L)   = 0.5 log(2 pi e var_m(x_L^m) / s^2)       the forecast distribution at lead L (the prediction)
    H_K(L)     = mean over members of the kernel's entropy  the equation's one-day entropy production on day L
    IGN(L)     = 0.5 log(2 pi var / s^2) + e^2 / (2 var)   the Gaussian log score of the truth (e = mean - truth)

A calibrated forecast has E[IGN] = H_ens (the entropy is then the expected surprise of the truth); IGN - H_ens > 0
means the ensemble is too sure. With M members, the error of the ensemble mean has variance sigma^2 (1 + 1/M) and
E[1/var] = (M - 1) / ((M - 3) sigma^2), so the error term of IGN is scaled by c = M (M - 3) / ((M + 1)(M - 1))
(M > 3), and spread / skill by sqrt((M + 1) / M): both are then exactly 0 / 1 for a calibrated ensemble. Per initial condition, the correlation of H_ens with 0.5 log(e^2 / s^2) says whether
the predicted entropy tells good forecasts from bad ones.
"""

import math
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from .env import DYN, TISR, TISR_DYN, contract

LOG2PI = math.log(2 * math.pi)
SCORED = [i for i, c in enumerate(DYN) if i != TISR_DYN]  # the 65 scored dynamic channels (index into DYN)
KEY_VARS = ["z500", "t850", "t2m", "u850", "v850", "q700", "msl", "u250"]


def dyn_index(name: str) -> int:
    return DYN.index(contract.CHANNEL_INDEX[name])


@torch.no_grad()
def rollout(step: Callable, xd0: torch.Tensor, doy0: torch.Tensor, tisr: torch.Tensor, members: int,
            leads: int, kernel=None, gen=None):
    """``step(xd, doy, tisr_next) -> xd_next`` for a batch of member states. xd0 (B, 66, H, W), tisr (B, L, H, W).
    Returns states (B, M, L, 66, H, W) and, with ``kernel``, its entropy per member and lead (B, M, L, 66, H, W)
    averaged later (float32, kept on the grid for the maps)."""
    B = xd0.shape[0]
    x = xd0.repeat_interleave(members, 0)
    doy = doy0.repeat_interleave(members, 0)
    out, hk = [], []
    for L in range(leads):
        tn = tisr[:, L].repeat_interleave(members, 0)
        if kernel is not None:
            c, h, _, _ = kernel(x, doy + L, tn, n=2, gen=gen)
            hk.append(kernel.entropy(c, h).view(B, members, *x.shape[1:]))
        x = step(x, doy + L, tn)
        out.append(x.view(B, members, *x.shape[1:]))
    y = torch.stack(out, 2)
    return y, (torch.stack(hk, 2) if hk else None)


def area_mean(f: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Mean over the last two axes (H, W) with latitude weights w (H,) normalized to mean 1."""
    return (f * w[:, None]).mean((-2, -1))


def fair_crps(ens: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Fair CRPS of an ensemble (B, M, ...) vs obs (B, ...) per value."""
    M = ens.shape[1]
    e1 = np.abs(ens - obs[:, None]).mean(1)
    s = np.sort(ens, axis=1)
    i = np.arange(1, M + 1).reshape((1, M) + (1,) * (ens.ndim - 2))
    e2 = ((2 * i - M - 1) * s).sum(1) / (M * (M - 1)) if M > 1 else 0.0
    return e1 - e2


class Scorer:
    """Accumulates per-lead scores over batches of ICs. All fields in normalized units (z-score), entropy in units
    of s; physical RMSE for KEY_VARS."""

    def __init__(self, leads: int, w: np.ndarray, inc: np.ndarray, std: np.ndarray):
        self.L, self.w = leads, w
        self.s = inc[DYN].astype(np.float64)        # (66,)
        self.std = std[DYN].astype(np.float64)
        self.acc: Dict[str, List[np.ndarray]] = {}

    def add(self, name: str, v: np.ndarray):
        self.acc.setdefault(name, []).append(v)

    def update(self, ens: np.ndarray, truth: np.ndarray, hk: Optional[np.ndarray] = None):
        """ens (B, M, L, 66, H, W), truth (B, L, 66, H, W), normalized; hk (B, M, L, 66, H, W) kernel entropy."""
        w = self.w
        M = ens.shape[1]
        self.M = M
        mu = ens.mean(1)
        err2 = area_mean((mu - truth) ** 2, w)                                          # (B, L, 66)
        self.add("mse", err2)
        if M > 1:
            var = ens.var(1, ddof=1)                                                    # (B, L, 66, H, W)
            self.add("var", area_mean(var, w))
            v = np.maximum(var / self.s[None, None, :, None, None] ** 2, 1e-4)
            e2 = (mu - truth) ** 2 / self.s[None, None, :, None, None] ** 2
            H = 0.5 * (LOG2PI + 1 + np.log(v))
            self.add("H", area_mean(H, w))                                              # (B, L, 66)
            c = M * (M - 3) / ((M + 1) * (M - 1)) if M > 3 else 1.0
            self.add("IGN", area_mean(0.5 * (LOG2PI + np.log(v)) + c * e2 / (2 * v), w))
            self.add("logerr", area_mean(0.5 * np.log(np.maximum(e2, 1e-8)), w))
            crps = np.stack([fair_crps(ens[:, :, l], truth[:, l]) for l in range(ens.shape[2])], 1)  # (B, L, 66, H, W)
            self.add("crps", area_mean(crps, w))
        else:
            self.add("crps", area_mean(np.abs(mu - truth), w))
        if hk is not None:
            self.add("HK", area_mean(hk.mean(1), w))

    def summary(self) -> Dict:
        cat = {k: np.concatenate(v, 0) for k, v in self.acc.items()}                    # (N, L, 66)
        sc = SCORED
        out: Dict = {"n_ic": int(cat["mse"].shape[0])}
        rmse = np.sqrt(cat["mse"].mean(0))                                             # (L, 66)
        out["nrmse"] = rmse[:, sc].mean(1).tolist()                                    # z-score RMSE, 65-channel mean
        out["crps"] = cat["crps"].mean(0)[:, sc].mean(1).tolist()
        out["rmse_phys"] = {v: (rmse[:, dyn_index(v)] * self.std[dyn_index(v)]).tolist() for v in KEY_VARS}
        if "var" in cat:
            spread = np.sqrt(cat["var"].mean(0))
            out["spread"] = spread[:, sc].mean(1).tolist()
            corr = np.sqrt((self.M + 1) / self.M)  # 1 = calibrated
            out["spread_skill"] = (corr * spread[:, sc] / rmse[:, sc]).mean(1).tolist()
            out["spread_skill_vars"] = {v: (corr * spread[:, dyn_index(v)] / rmse[:, dyn_index(v)]).tolist()
                                        for v in KEY_VARS}
            out["crps_vars"] = {v: (cat["crps"].mean(0)[:, dyn_index(v)] * self.std[dyn_index(v)]).tolist()
                                for v in KEY_VARS}
            out["entropy"] = cat["H"][:, :, sc].mean((0, 2)).tolist()
            out["ign"] = cat["IGN"][:, :, sc].mean((0, 2)).tolist()
            out["entropy_vars"] = {v: cat["H"][:, :, dyn_index(v)].mean(0).tolist() for v in KEY_VARS}
            # does the predicted entropy of an IC tell its error? corr over ICs, per lead (65-channel means)
            Hi, Ei = cat["H"][:, :, sc].mean(2), cat["logerr"][:, :, sc].mean(2)
            out["entropy_error_corr"] = [float(np.corrcoef(Hi[:, l], Ei[:, l])[0, 1]) if Hi.shape[0] > 2 else None
                                         for l in range(Hi.shape[1])]
            out["entropy_error_corr_vars"] = {
                v: [float(np.corrcoef(cat["H"][:, l, dyn_index(v)], cat["logerr"][:, l, dyn_index(v)])[0, 1])
                    for l in range(Hi.shape[1])] for v in KEY_VARS}
        if "HK" in cat:
            out["kernel_entropy"] = cat["HK"][:, :, sc].mean((0, 2)).tolist()
        return out
