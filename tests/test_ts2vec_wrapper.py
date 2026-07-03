"""
tests/test_ts2vec_wrapper.py

Unit tests for src/models/ts2vec_wrapper.py (M7 — TS2Vec Wrapper).

Covers IMP-01 v1.3 M7 Definition of Done and DS-04 v1.1:
- V-MODEL-001 (branch encoder output shape [N, 64])
- V-INV-004 (each branch owns an independent optimizer/model; no shared
  loss or weights across branches)
- ADR-010 checkpoint bundle completeness + GPU->CPU-safe load
  (map_location) + weight restoration reproduces encode output
- encode determinism given fixed weights
- early-stopping (patience) mechanism

Training runs use a tiny max_epochs and a small synthetic dataset so the
suite stays fast; correctness of the encoder itself is TS2Vec's concern
(pinned upstream), not this wrapper's.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.ts2vec_wrapper import (
    TS2VecBranch,
    TrainingHistory,
    _EarlyStop,
)
from src.utils.config import load_config
from src.utils.paths import BASE_CONFIG_PATH

CPU = torch.device("cpu")
N, T, F = 24, 48, 7


@pytest.fixture(scope="module")
def base_config() -> dict:
    """Real base.yaml, with epochs shrunk for fast tests (values-only)."""
    cfg = load_config(BASE_CONFIG_PATH)
    cfg["training"]["max_epochs"] = 2
    cfg["training"]["early_stopping_patience"] = 10
    cfg["training"]["batch_size"] = 8
    return cfg


@pytest.fixture
def windows() -> np.ndarray:
    return np.random.RandomState(0).randn(N, T, F).astype(np.float32)


def _make_branch(cfg: dict, timeframe: str = "1h") -> TS2VecBranch:
    return TS2VecBranch(config=cfg, timeframe=timeframe, device=CPU)


# --- construction -----------------------------------------------------------


class TestConstruction:
    def test_rejects_unknown_timeframe(self, base_config: dict) -> None:
        with pytest.raises(ValueError):
            TS2VecBranch(config=base_config, timeframe="30m", device=CPU)

    def test_accepts_each_valid_timeframe(self, base_config: dict) -> None:
        for tf in ("15m", "1h", "4h", "1d"):
            branch = _make_branch(base_config, tf)
            assert branch.timeframe == tf


# --- V-MODEL-001: encode output shape --------------------------------------


class TestEncodeShape:
    def test_encode_returns_n_by_64(self, base_config, windows) -> None:
        branch = _make_branch(base_config)
        out = branch.encode(windows)
        assert out.shape == (N, 64)
        assert out.dtype == np.float32

    def test_encode_shape_holds_after_training(self, base_config, windows) -> None:
        branch = _make_branch(base_config)
        branch.train(windows, seed=42)
        out = branch.encode(windows)
        assert out.shape == (N, 64)

    def test_encode_rejects_non_3d(self, base_config) -> None:
        branch = _make_branch(base_config)
        with pytest.raises(ValueError):
            branch.encode(np.zeros((N, T)))  # 2D

    def test_encode_arbitrary_batch_size(self, base_config) -> None:
        branch = _make_branch(base_config)
        small = np.random.RandomState(1).randn(3, T, F).astype(np.float32)
        assert branch.encode(small).shape == (3, 64)


# --- encode determinism -----------------------------------------------------


class TestEncodeDeterminism:
    def test_same_input_same_weights_identical(self, base_config, windows) -> None:
        branch = _make_branch(base_config)
        branch.train(windows, seed=42)
        a = branch.encode(windows)
        b = branch.encode(windows)
        np.testing.assert_array_equal(a, b)


# --- training history -------------------------------------------------------


class TestTraining:
    def test_returns_history_with_expected_length(self, base_config, windows) -> None:
        branch = _make_branch(base_config)
        hist = branch.train(windows, seed=42)
        assert isinstance(hist, TrainingHistory)
        # max_epochs=2 and patience=10 -> exactly 2 epochs, no early stop
        assert len(hist.train_loss_history) == 2
        assert len(hist.epoch_times) == 2
        assert not hist.stopped_early

    def test_best_epoch_matches_min_loss(self, base_config, windows) -> None:
        branch = _make_branch(base_config)
        hist = branch.train(windows, seed=42)
        expected_best = int(np.argmin(hist.train_loss_history))
        assert hist.best_epoch == expected_best
        assert hist.best_loss == pytest.approx(min(hist.train_loss_history))


# --- early stopping mechanism (patience) -----------------------------------


class TestEarlyStopping:
    def test_callback_raises_after_patience_without_improvement(
        self, base_config
    ) -> None:
        cfg = {**base_config, "training": {**base_config["training"]}}
        cfg["training"]["early_stopping_patience"] = 2
        branch = _make_branch(cfg)
        m = branch._model

        # improving then flat: 1.0, 0.5, then no improvement x2 == patience
        branch._on_epoch_end(m, 1.0)  # best=1.0
        branch._on_epoch_end(m, 0.5)  # best=0.5, no-improve=0
        branch._on_epoch_end(m, 0.6)  # no-improve=1
        with pytest.raises(_EarlyStop):
            branch._on_epoch_end(m, 0.6)  # no-improve=2 -> stop
        assert branch.history.stopped_early
        assert branch.history.best_loss == pytest.approx(0.5)
        assert branch.history.best_epoch == 1


# --- V-INV-004: independent branches ---------------------------------------


class TestIndependentBranches:
    def test_distinct_models_and_nets(self, base_config) -> None:
        a = _make_branch(base_config, "1h")
        b = _make_branch(base_config, "4h")
        assert a._model is not b._model
        assert a._model._net is not b._model._net

    def test_training_one_branch_does_not_change_another(
        self, base_config, windows
    ) -> None:
        a = _make_branch(base_config, "1h")
        b = _make_branch(base_config, "4h")
        before = b.encode(windows)
        a.train(windows, seed=42)
        after = b.encode(windows)
        # b's weights are untouched by training a
        np.testing.assert_array_equal(before, after)


# --- ADR-010 checkpoint bundle ---------------------------------------------

_ADR010_REQUIRED_KEYS = {
    "model_state_dict",
    "branch_timeframe",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "epoch",
    "global_step",
    "best_metric",
    "best_epoch",
    "seed",
    "condition",
    "config_snapshot",
    "timestamp",
    "git_commit_hash",
    "python_version",
    "torch_version",
    "numpy_version",
    "ts2vec_commit",
    "projection_matrix",
    "projection_seed",
    "train_loss_history",
    "epoch_times",
}


class TestCheckpoint:
    def test_bundle_has_all_adr010_keys(self, base_config, windows, tmp_path) -> None:
        branch = _make_branch(base_config)
        branch.train(windows, seed=123)
        ckpt = tmp_path / "best_model.pt"
        branch.save_checkpoint(ckpt)
        bundle = torch.load(ckpt, map_location="cpu", weights_only=False)
        assert _ADR010_REQUIRED_KEYS.issubset(bundle.keys())

    def test_metadata_values(self, base_config, windows, tmp_path) -> None:
        branch = _make_branch(base_config)
        branch.train(windows, seed=123)
        ckpt = tmp_path / "m.pt"
        branch.save_checkpoint(ckpt)
        bundle = torch.load(ckpt, map_location="cpu", weights_only=False)
        assert bundle["branch_timeframe"] == "1h"
        assert bundle["seed"] == 123
        assert bundle["ts2vec_commit"] == base_config["ts2vec"]["pinned_commit"]
        assert bundle["projection_seed"] == base_config["fusion"]["projection_seed"]
        assert bundle["torch_version"] == torch.__version__
        assert bundle["numpy_version"] == np.__version__
        assert len(bundle["train_loss_history"]) == 2

    def test_extra_metadata_overrides(self, base_config, windows, tmp_path) -> None:
        branch = _make_branch(base_config)
        branch.train(windows, seed=42)
        ckpt = tmp_path / "m.pt"
        proj = torch.randn(256, 64)
        branch.save_checkpoint(
            ckpt, extra_metadata={"condition": "2TF", "projection_matrix": proj}
        )
        bundle = torch.load(ckpt, map_location="cpu", weights_only=False)
        assert bundle["condition"] == "2TF"
        assert torch.equal(bundle["projection_matrix"], proj)

    def test_load_restores_weights_and_reproduces_encode(
        self, base_config, windows, tmp_path
    ) -> None:
        branch = _make_branch(base_config)
        branch.train(windows, seed=42)
        expected = branch.encode(windows)
        ckpt = tmp_path / "m.pt"
        branch.save_checkpoint(ckpt)

        # Fresh branch (untrained) loads the checkpoint on CPU.
        fresh = _make_branch(base_config)
        fresh.load_checkpoint(ckpt)
        got = fresh.encode(windows)
        np.testing.assert_array_equal(expected, got)

    def test_load_uses_map_location_to_cpu(
        self, base_config, windows, tmp_path
    ) -> None:
        # Simulates the "GPU checkpoint loaded on CPU" requirement: the
        # loading branch's device is CPU and load must not error.
        branch = _make_branch(base_config)
        branch.train(windows, seed=42)
        ckpt = tmp_path / "m.pt"
        branch.save_checkpoint(ckpt)
        fresh = TS2VecBranch(config=base_config, timeframe="1h", device=CPU)
        bundle = fresh.load_checkpoint(ckpt)  # must not raise
        assert bundle["branch_timeframe"] == "1h"

    def test_best_snapshot_taken_during_training(
        self, base_config, windows
    ) -> None:
        branch = _make_branch(base_config)
        assert branch._best_state_dict is None
        branch.train(windows, seed=42)
        assert branch._best_state_dict is not None

    def test_save_best_and_latest_kinds(
        self, base_config, windows, tmp_path
    ) -> None:
        branch = _make_branch(base_config)
        branch.train(windows, seed=42)
        best = tmp_path / "best_model.pt"
        latest = tmp_path / "latest_model.pt"
        branch.save_checkpoint(best, which="best")
        branch.save_checkpoint(latest, which="latest")
        b = torch.load(best, map_location="cpu", weights_only=False)
        l = torch.load(latest, map_location="cpu", weights_only=False)
        assert b["checkpoint_kind"] == "best"
        assert l["checkpoint_kind"] == "latest"
        # best checkpoint loads and encodes to the right shape
        fresh = _make_branch(base_config)
        fresh.load_checkpoint(best)
        assert fresh.encode(windows).shape == (N, 64)

    def test_save_best_matches_snapshot(self, base_config, windows, tmp_path) -> None:
        branch = _make_branch(base_config)
        branch.train(windows, seed=42)
        ckpt = tmp_path / "best.pt"
        branch.save_checkpoint(ckpt, which="best")
        bundle = torch.load(ckpt, map_location="cpu", weights_only=False)
        for k, v in branch._best_state_dict.items():
            assert torch.equal(bundle["model_state_dict"][k], v)

    def test_save_invalid_which_raises(self, base_config, tmp_path) -> None:
        branch = _make_branch(base_config)
        with pytest.raises(ValueError):
            branch.save_checkpoint(tmp_path / "x.pt", which="middle")

    def test_save_best_without_training_falls_back(
        self, base_config, tmp_path
    ) -> None:
        branch = _make_branch(base_config)  # never trained
        ckpt = tmp_path / "best.pt"
        branch.save_checkpoint(ckpt, which="best")  # must not raise
        bundle = torch.load(ckpt, map_location="cpu", weights_only=False)
        assert bundle["checkpoint_kind"] == "best"

    def test_load_warns_on_missing_fields(self, base_config, tmp_path) -> None:
        import logging

        branch = _make_branch(base_config)
        # Minimal bundle missing most ADR-010 fields.
        ckpt = tmp_path / "old.pt"
        torch.save({"model_state_dict": branch._model.net.state_dict()}, ckpt)

        # The project's get_logger sets propagate=False, so pytest's caplog
        # (which listens on the root logger) can't see these records. Attach
        # our own capturing handler to the module logger instead.
        records: list[logging.LogRecord] = []
        handler = logging.Handler()
        handler.emit = records.append  # type: ignore[method-assign]
        module_logger = logging.getLogger("src.models.ts2vec_wrapper")
        module_logger.addHandler(handler)
        try:
            branch.load_checkpoint(ckpt)
        finally:
            module_logger.removeHandler(handler)

        assert any("missing ADR-010 fields" in r.getMessage() for r in records)
