"""
tests/test_seed.py

Unit tests for src/utils/seed.py (M0 — Project Bootstrap).

Covers IMP-01 M0 Definition of Done:
- "set_all_seeds verifiably affects NumPy and PyTorch random state
  (confirmed by a test that checks reproducibility)."
This is also the precondition test for V-INV-007 (deterministic
execution), which is fully verified later at the pipeline level (M13).
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from src.utils.seed import get_torch_rng_state, set_all_seeds


class TestSetAllSeeds:
    def test_same_seed_produces_same_numpy_output(self) -> None:
        set_all_seeds(42)
        a = np.random.rand(10)
        set_all_seeds(42)
        b = np.random.rand(10)
        np.testing.assert_array_equal(a, b)

    def test_same_seed_produces_same_python_random_output(self) -> None:
        set_all_seeds(123)
        a = [random.random() for _ in range(10)]
        set_all_seeds(123)
        b = [random.random() for _ in range(10)]
        assert a == b

    def test_same_seed_produces_same_torch_output(self) -> None:
        set_all_seeds(456)
        a = torch.rand(10)
        set_all_seeds(456)
        b = torch.rand(10)
        assert torch.equal(a, b)

    def test_different_seeds_produce_different_output(self) -> None:
        set_all_seeds(42)
        a = np.random.rand(10)
        set_all_seeds(789)
        b = np.random.rand(10)
        assert not np.array_equal(a, b)

    def test_rng_state_actually_changes(self) -> None:
        """
        DoD wording: 'verifiably affects NumPy and PyTorch random
        state'. This directly inspects the RNG state object rather
        than just downstream samples.
        """
        set_all_seeds(42)
        state_a = get_torch_rng_state()
        set_all_seeds(1024)
        state_b = get_torch_rng_state()

        assert state_a["numpy"][1].tolist() != state_b["numpy"][1].tolist()
        assert not torch.equal(state_a["torch"], state_b["torch"])

    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 1024])
    def test_all_five_ds03_seeds_are_usable(self, seed: int) -> None:
        """
        DS-03 Table 3.11 specifies exactly these five seeds. Confirm
        each one is a valid, reproducible seed value.
        """
        set_all_seeds(seed)
        a = np.random.rand(5)
        set_all_seeds(seed)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_strict_mode_does_not_error_on_cpu(self) -> None:
        """
        strict=True enables torch.use_deterministic_algorithms(True).
        This must not raise merely from being enabled on a CPU-only
        environment (it only raises when a non-deterministic op is
        actually invoked, which this test does not do).
        """
        set_all_seeds(42, strict=True)
        # Reset to non-strict so subsequent tests in the suite aren't
        # affected by a global deterministic-algorithms flag.
        torch.use_deterministic_algorithms(False)
