"""
src/utils/seed.py

Seed-setting utility for full reproducibility (M0 — Project Bootstrap).

Purpose
-------
DS-03 requires that all five random seeds ({42, 123, 456, 789, 1024})
be applied "identically to every source of randomness (Python, NumPy,
PyTorch, CUDA)" (Table 3.11) and that execution be deterministic given
a fixed seed (INV-007 / V-INV-007). This module is the single place
where that seeding happens, so every module that needs reproducibility
calls `set_all_seeds` instead of seeding libraries individually.

Notes on determinism
---------------------
`torch.use_deterministic_algorithms(True)` is opt-in and can raise a
RuntimeError for operations that have no deterministic implementation.
Per IMP-01 Risk R-06, any such non-deterministic operation encountered
during real training must be documented as an exception rather than
silently disabling determinism. This module exposes `strict` to control
that behavior explicitly.
"""

from __future__ import annotations

import os
import random

import numpy as np

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in torch-less envs
    _TORCH_AVAILABLE = False


def set_all_seeds(seed: int, strict: bool = False) -> None:
    """
    Set the random seed for Python, NumPy, PyTorch, and CUDA identically.

    Parameters
    ----------
    seed : int
        The seed value. Per DS-03, must be one of
        `src.utils.paths.RANDOM_SEEDS` ({42, 123, 456, 789, 1024}) during
        actual experiment runs, though this function does not enforce
        that restriction (it is reused by tests with arbitrary seeds).
    strict : bool, optional
        If True, additionally call
        `torch.use_deterministic_algorithms(True)`, which raises a
        RuntimeError at call time for any operation lacking a
        deterministic implementation. Default False, since strict mode
        is only required during real training runs (M8), not for every
        module that merely calls this function for test reproducibility.

    Notes
    -----
    Sets, in order: `PYTHONHASHSEED` (env var, affects hash-based
    iteration order in this and future subprocesses), `random.seed`,
    `numpy.random.seed`, and, if PyTorch is available,
    `torch.manual_seed` plus both CUDA seed functions (safe to call even
    without a GPU present).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if strict:
            torch.use_deterministic_algorithms(True)
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def get_torch_rng_state() -> "dict[str, object]":
    """
    Capture the current RNG state of every seeded library.

    Used by tests to verify that `set_all_seeds` actually changes state
    (V-INV-007 precondition, per IMP-01 M0 Definition of Done).

    Returns
    -------
    dict
        Keys: "python", "numpy", and, if available, "torch" and "cuda".
        Values are the corresponding RNG state objects.
    """
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    if _TORCH_AVAILABLE:
        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
    return state
