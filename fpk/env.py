"""Access to the vendored physical_guide_flowmatching code (third_party/pgfm).

The tendency core, the primitive-equation operator G(x) and the KAI data contract live in the PG-CFM repository
(branch ``claude/project-thread-c97r6n``, commit 5d38e56). They are vendored unchanged under ``third_party/pgfm``
so that cores trained there load here as they are (same options, same checkpoints, same ``stats.npz``).
"""

import importlib.util
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PGFM = os.path.join(ROOT, "third_party", "pgfm")
if PGFM not in sys.path:
    sys.path.insert(0, PGFM)

from pgcfm import contract  # noqa: E402,F401
from pgcfm.core import time_features  # noqa: E402,F401
from pgcfm.kai_tend import DYN, OROG, TISR, TISR_DYN  # noqa: E402,F401

_RKT = None


def rkt():
    """The vendored ``scripts/run_kai_tendency.py`` as a module (data loading, statistics, ``build_core``)."""
    global _RKT
    if _RKT is None:
        path = os.path.join(PGFM, "scripts", "run_kai_tendency.py")
        spec = importlib.util.spec_from_file_location("pgfm_run_kai_tendency", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _RKT = mod
    return _RKT


def pgfm_args(data_root: str, out: str, train_years, val_year: int, test_year: int, device: str = "cpu",
              core_tag: str = "core_gng", extra=None):
    """The argument namespace the vendored script's helpers read (paths, years, device, core tag), with every other
    option at its default."""
    argv = ["--data", os.path.join(data_root, "kai6h"), "--era5", os.path.join(data_root, "era5_data"),
            "--cache", os.path.join(data_root, "kaiday"), "--out", out,
            "--train-years", *map(str, train_years), "--val-year", str(val_year), "--test-year", str(test_year),
            "--device", device, "--core-tag", core_tag] + list(extra or [])
    return rkt().parser().parse_args(argv)
