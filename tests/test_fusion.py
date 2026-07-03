"""
tests/test_fusion.py

Unit tests for src/models/fusion.py (M9 — Fusion).

Covers IMP-01 v1.3 M9 Definition of Done and DS-04 v1.1:
- V-MODEL-002 / V-INV-002 (fused output [N, 256] for all 7 conditions)
- V-MODEL-003 / V-INV-003 (0 trainable params; projection unchanged by a
  gradient step)
- V-MODEL-004 (deterministic projection given the same seed + input dim)
- ADR-013 fixed concat order (15m,1h,4h,1d), enforced as an ordered list
- 4TF (concat dim already 256) still passes through the projection (not
  identity)

The deterministic fusion core is tested directly with synthetic branch
embeddings; no trained checkpoints are required.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.fusion import (
    DEFAULT_CONCAT_ORDER,
    EmbeddingPipeline,
    FusionError,
    FusionModule,
    build_projection_matrix,
    order_active_timeframes,
)

# The 7 TS2Vec conditions and their active timeframes (DS-03 Table 3.10).
CONDITIONS: dict[str, list[str]] = {
    "1TF": ["1h"],
    "2TF": ["15m", "1h"],
    "3TF": ["15m", "1h", "4h"],
    "4TF": ["15m", "1h", "4h", "1d"],
    "BL-15m": ["15m"],
    "BL-4h": ["4h"],
    "BL-1d": ["1d"],
}
EXPECTED_CONCAT_DIM = {"1TF": 64, "2TF": 128, "3TF": 192, "4TF": 256,
                       "BL-15m": 64, "BL-4h": 64, "BL-1d": 64}
N = 20


def _branch_embeddings(n: int = N, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.RandomState(seed)
    return {tf: rng.randn(n, 64).astype(np.float32) for tf in DEFAULT_CONCAT_ORDER}


# --- concat ordering --------------------------------------------------------


class TestOrdering:
    def test_reorders_to_fixed_concat_order(self) -> None:
        # supplied out of order -> canonical fine-to-coarse
        assert order_active_timeframes(["1d", "15m", "4h", "1h"]) == [
            "15m", "1h", "4h", "1d"
        ]

    def test_subset_keeps_order(self) -> None:
        assert order_active_timeframes(["4h", "15m"]) == ["15m", "4h"]

    def test_empty_raises(self) -> None:
        with pytest.raises(FusionError):
            order_active_timeframes([])

    def test_unknown_tf_raises(self) -> None:
        with pytest.raises(FusionError):
            order_active_timeframes(["30m"])


# --- projection matrix (ADR-003) -------------------------------------------


class TestProjectionMatrix:
    def test_shape_and_frozen(self) -> None:
        p = build_projection_matrix(128, 256, projection_seed=42)
        assert p.shape == (256, 128)
        assert p.requires_grad is False

    def test_rows_unit_l2_norm(self) -> None:
        p = build_projection_matrix(192, 256, projection_seed=42)
        norms = torch.linalg.norm(p, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def test_deterministic_given_seed(self) -> None:
        a = build_projection_matrix(128, 256, projection_seed=42)
        b = build_projection_matrix(128, 256, projection_seed=42)
        assert torch.equal(a, b)

    def test_different_seed_differs(self) -> None:
        a = build_projection_matrix(128, 256, projection_seed=42)
        b = build_projection_matrix(128, 256, projection_seed=123)
        assert not torch.equal(a, b)


# --- V-MODEL-002 / V-INV-002: output shape for all conditions --------------


class TestFusedOutputShape:
    @pytest.mark.parametrize("condition", list(CONDITIONS))
    def test_fused_is_n_by_256(self, condition) -> None:
        module = FusionModule(condition, CONDITIONS[condition])
        fused = module.fuse(_branch_embeddings())
        assert fused.shape == (N, 256)
        assert fused.dtype == np.float32

    @pytest.mark.parametrize("condition", list(CONDITIONS))
    def test_concat_dim_matches_expected(self, condition) -> None:
        module = FusionModule(condition, CONDITIONS[condition])
        assert module.input_dim == EXPECTED_CONCAT_DIM[condition]


# --- ADR-013 concat order ---------------------------------------------------


class TestConcatOrder:
    def test_concatenation_follows_fixed_order(self) -> None:
        module = FusionModule("4TF", ["1d", "4h", "1h", "15m"])  # scrambled input
        assert module.ordered_timeframes == ["15m", "1h", "4h", "1d"]
        # concatenated block layout is 15m|1h|4h|1d regardless of input order
        emb = _branch_embeddings()
        concat = module.concatenate(emb)
        assert concat.shape == (N, 256)
        np.testing.assert_array_equal(concat[:, 0:64], emb["15m"])
        np.testing.assert_array_equal(concat[:, 64:128], emb["1h"])
        np.testing.assert_array_equal(concat[:, 128:192], emb["4h"])
        np.testing.assert_array_equal(concat[:, 192:256], emb["1d"])

    def test_missing_active_branch_raises(self) -> None:
        module = FusionModule("2TF", ["15m", "1h"])
        with pytest.raises(FusionError):
            module.fuse({"15m": np.zeros((N, 64), dtype=np.float32)})  # missing 1h

    def test_mismatched_n_raises(self) -> None:
        module = FusionModule("2TF", ["15m", "1h"])
        with pytest.raises(FusionError):
            module.fuse({
                "15m": np.zeros((N, 64), dtype=np.float32),
                "1h": np.zeros((N + 1, 64), dtype=np.float32),
            })


# --- V-MODEL-004: determinism ----------------------------------------------


class TestDeterminism:
    def test_same_seed_same_fused(self) -> None:
        emb = _branch_embeddings()
        a = FusionModule("3TF", CONDITIONS["3TF"], projection_seed=42).fuse(emb)
        b = FusionModule("3TF", CONDITIONS["3TF"], projection_seed=42).fuse(emb)
        np.testing.assert_array_equal(a, b)

    def test_different_seed_changes_fused(self) -> None:
        emb = _branch_embeddings()
        a = FusionModule("3TF", CONDITIONS["3TF"], projection_seed=42).fuse(emb)
        b = FusionModule("3TF", CONDITIONS["3TF"], projection_seed=7).fuse(emb)
        assert not np.allclose(a, b)


# --- V-MODEL-003 / V-INV-003: zero trainable params, projection frozen ------


class TestNoTrainableParams:
    @pytest.mark.parametrize("condition", list(CONDITIONS))
    def test_zero_trainable_params(self, condition) -> None:
        module = FusionModule(condition, CONDITIONS[condition])
        assert module.n_trainable_parameters() == 0
        assert module.projection_matrix.requires_grad is False

    def test_projection_unchanged_after_gradient_step(self) -> None:
        # V-MODEL-003: even if a downstream tensor is optimized, P must not
        # move (it is frozen and detached from any graph).
        module = FusionModule("4TF", CONDITIONS["4TF"])
        p_before = module.projection_matrix.clone()
        emb = _branch_embeddings()
        # Simulate a gradient step on a separate learnable tensor that
        # consumes the fused output; P itself has requires_grad=False.
        fused = torch.from_numpy(module.fuse(emb))
        w = torch.zeros(256, 1, requires_grad=True)
        opt = torch.optim.SGD([w], lr=0.1)
        loss = (fused @ w).pow(2).mean()
        loss.backward()
        opt.step()
        assert torch.equal(module.projection_matrix, p_before)


# --- 4TF still projected (not identity) ------------------------------------


class Test4TFProjection:
    def test_4tf_is_projected_not_identity(self) -> None:
        emb = _branch_embeddings()
        module = FusionModule("4TF", CONDITIONS["4TF"])
        concat = module.concatenate(emb)  # [N, 256]
        fused = module.fuse(emb)          # [N, 256]
        # Same shape, but the projection must transform the values.
        assert concat.shape == fused.shape == (N, 256)
        assert not np.allclose(concat, fused)


# --- EmbeddingPipeline (deterministic core with a stub encoder) -------------


class _StubBranch:
    """Deterministic stand-in for TS2VecBranch: encode -> fixed [N,64]."""

    def __init__(self, timeframe: str, seed: int) -> None:
        self.timeframe = timeframe
        self.seed = seed

    def encode(self, windows: np.ndarray) -> np.ndarray:
        rng = np.random.RandomState(hash((self.timeframe, self.seed)) % (2**32))
        return rng.randn(windows.shape[0], 64).astype(np.float32)


class TestEmbeddingPipeline:
    def _config(self) -> dict:
        return {"fusion": {"concat_order": list(DEFAULT_CONCAT_ORDER),
                           "projection_seed": 42, "output_dim": 256}}

    def test_encode_all_branches_shapes(self) -> None:
        pipe = EmbeddingPipeline(self._config(), _StubBranch)
        windows = {tf: np.zeros((N, 48, 7), dtype=np.float32) for tf in DEFAULT_CONCAT_ORDER}
        branch_emb = pipe.encode_all_branches(seed=42, windows_by_tf=windows)
        assert set(branch_emb) == set(DEFAULT_CONCAT_ORDER)
        for tf, arr in branch_emb.items():
            assert arr.shape == (N, 64)

    def test_fuse_condition_produces_256(self) -> None:
        pipe = EmbeddingPipeline(self._config(), _StubBranch)
        emb = _branch_embeddings()
        for cond, tfs in CONDITIONS.items():
            fused = pipe.fuse_condition(cond, tfs, emb)
            assert fused.shape == (N, 256)

    def test_branch_reuse_same_across_conditions(self) -> None:
        # The 1h branch embedding is identical regardless of which
        # condition consumes it (ADR-002 reuse).
        pipe = EmbeddingPipeline(self._config(), _StubBranch)
        windows = {tf: np.zeros((N, 48, 7), dtype=np.float32) for tf in DEFAULT_CONCAT_ORDER}
        e1 = pipe.encode_all_branches(seed=42, windows_by_tf=windows)
        e2 = pipe.encode_all_branches(seed=42, windows_by_tf=windows)
        np.testing.assert_array_equal(e1["1h"], e2["1h"])
