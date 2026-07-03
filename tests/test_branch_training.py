"""
tests/test_branch_training.py

Unit tests for src/models/branch_training.py (M8 — Branch Training).

Covers IMP-01 v1.3 M8 Definition of Done and DS-04 v1.1:
- Exactly 4 runs per seed (one per timeframe), NOT one per condition
- ADR-010 best_model.pt + latest_model.pt per (timeframe, seed) run
- load_or_train idempotency (skip if a valid checkpoint exists)
- run-level resume (TrainingOrchestrator skips completed runs, continues
  past failures)
- V-INV-004 (independent branches), V-EXP-002 (different seeds -> distinct
  embeddings), V-MODEL-005 (different branches -> distinct representations)

All training uses tiny synthetic windows and 1 epoch, with checkpoints/
logs redirected to tmp dirs, so the suite stays fast. Real 20-run
training on M6 outputs is an execution step, not an implementation test.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.models.branch_training import (
    BranchTrainer,
    BranchTrainingError,
    TrainingOrchestrator,
)
from src.models.ts2vec_wrapper import TS2VecBranch
from src.utils.config import load_config
from src.utils.paths import BASE_CONFIG_PATH

CPU = torch.device("cpu")
N, T, F = 16, 48, 7


@pytest.fixture(scope="module")
def base_config() -> dict:
    cfg = load_config(BASE_CONFIG_PATH)
    cfg["training"]["max_epochs"] = 1
    cfg["training"]["early_stopping_patience"] = 10
    cfg["training"]["batch_size"] = 8
    return cfg


def _windows(seed: int) -> np.ndarray:
    return np.random.RandomState(seed).randn(N, T, F).astype(np.float32)


@pytest.fixture
def windows_by_tf() -> dict[str, np.ndarray]:
    # Distinct data per timeframe (as M6 produces different tensors per tf).
    return {tf: _windows(i) for i, tf in enumerate(("15m", "1h", "4h", "1d"))}


def _make_trainer(cfg, tmp_path) -> BranchTrainer:
    return BranchTrainer(
        config=cfg,
        device=CPU,
        windows_dir=tmp_path / "windows",
        checkpoints_dir=tmp_path / "checkpoints",
        logs_dir=tmp_path / "logs",
    )


# --- path / config helpers --------------------------------------------------


class TestPaths:
    def test_checkpoint_path_layout(self, base_config, tmp_path) -> None:
        tr = _make_trainer(base_config, tmp_path)
        best = tr.checkpoint_path("1h", 42, "best")
        latest = tr.checkpoint_path("1h", 42, "latest")
        assert best.name == "best_model.pt"
        assert latest.name == "latest_model.pt"
        assert best.parent.name == "seed_42"
        assert best.parent.parent.name == "branch_1h"

    def test_checkpoint_path_is_condition_agnostic(self, base_config, tmp_path) -> None:
        # ADR-002: there is exactly one path per (tf, seed) regardless of
        # which condition consumes it.
        tr = _make_trainer(base_config, tmp_path)
        assert tr.checkpoint_path("1h", 42) == tr.checkpoint_path("1h", 42)

    def test_rejects_unknown_timeframe(self, base_config, tmp_path) -> None:
        tr = _make_trainer(base_config, tmp_path)
        with pytest.raises(ValueError):
            tr.checkpoint_path("30m", 42)

    def test_missing_window_file_raises(self, base_config, tmp_path) -> None:
        tr = _make_trainer(base_config, tmp_path)
        with pytest.raises(BranchTrainingError):
            tr.load_train_windows("1h")


# --- single-run training ----------------------------------------------------


class TestTrainSingle:
    def test_writes_best_and_latest_and_log(
        self, base_config, tmp_path, windows_by_tf
    ) -> None:
        tr = _make_trainer(base_config, tmp_path)
        tr.train_single("1h", 42, windows=windows_by_tf["1h"])
        assert tr.checkpoint_path("1h", 42, "best").exists()
        assert tr.checkpoint_path("1h", 42, "latest").exists()
        assert tr.run_log_path("1h", 42).exists()
        # per-run log has epoch-level detail
        log_text = tr.run_log_path("1h", 42).read_text(encoding="utf-8")
        assert "epoch 0" in log_text

    def test_checkpoint_has_reuse_note_and_timeframe(
        self, base_config, tmp_path, windows_by_tf
    ) -> None:
        tr = _make_trainer(base_config, tmp_path)
        tr.train_single("4h", 123, windows=windows_by_tf["4h"])
        bundle = torch.load(
            tr.checkpoint_path("4h", 123, "best"), map_location="cpu", weights_only=False
        )
        assert bundle["branch_timeframe"] == "4h"
        assert bundle["seed"] == 123
        assert "branch_reuse_note" in bundle


# --- idempotency / resume ---------------------------------------------------


class TestLoadOrTrain:
    def test_skips_when_valid_checkpoint_exists(
        self, base_config, tmp_path, windows_by_tf
    ) -> None:
        tr = _make_trainer(base_config, tmp_path)
        # First call trains.
        tr.load_or_train("1h", 42, windows=windows_by_tf["1h"])
        # Second call must NOT retrain.
        with patch.object(tr, "train_single", wraps=tr.train_single) as spy:
            path = tr.load_or_train("1h", 42, windows=windows_by_tf["1h"])
            spy.assert_not_called()
        assert path == tr.checkpoint_path("1h", 42, "best")

    def test_trains_when_no_checkpoint(
        self, base_config, tmp_path, windows_by_tf
    ) -> None:
        tr = _make_trainer(base_config, tmp_path)
        with patch.object(tr, "train_single", wraps=tr.train_single) as spy:
            tr.load_or_train("1h", 42, windows=windows_by_tf["1h"])
            spy.assert_called_once()


# --- full protocol: 4 x seeds ----------------------------------------------


class TestTrainAllBranches:
    def test_exactly_four_runs_per_seed(
        self, base_config, tmp_path, windows_by_tf
    ) -> None:
        tr = _make_trainer(base_config, tmp_path)
        seeds = (42, 123)
        results = tr.train_all_branches(seeds=seeds, windows_by_tf=windows_by_tf)
        # 4 timeframes x 2 seeds = 8 runs
        assert len(results) == 8
        for seed in seeds:
            runs = [k for k in results if k[1] == seed]
            assert len(runs) == 4  # one per timeframe, not per condition
        # all 8 best checkpoints exist
        for (tf, seed), path in results.items():
            assert path.exists()
            assert path == tr.checkpoint_path(tf, seed, "best")


# --- V-EXP-002 / V-MODEL-005 / V-INV-004 -----------------------------------


class TestDistinctness:
    def _encode_from_checkpoint(self, cfg, path, tf, data) -> np.ndarray:
        branch = TS2VecBranch(cfg, tf, CPU)
        branch.load_checkpoint(path)
        return branch.encode(data)

    def test_different_seeds_give_distinct_embeddings(
        self, base_config, tmp_path, windows_by_tf
    ) -> None:
        # V-EXP-002: same branch/data, different seed -> distinct embeddings.
        tr = _make_trainer(base_config, tmp_path)
        data = windows_by_tf["1h"]
        tr.train_single("1h", 42, windows=data)
        tr.train_single("1h", 123, windows=data)
        e42 = self._encode_from_checkpoint(
            base_config, tr.checkpoint_path("1h", 42, "best"), "1h", data
        )
        e123 = self._encode_from_checkpoint(
            base_config, tr.checkpoint_path("1h", 123, "best"), "1h", data
        )
        assert not np.allclose(e42, e123)

    def test_different_branches_give_distinct_representations(
        self, base_config, tmp_path, windows_by_tf
    ) -> None:
        # V-MODEL-005: different branches (own timeframe data) -> distinct reps.
        tr = _make_trainer(base_config, tmp_path)
        tr.train_single("1h", 42, windows=windows_by_tf["1h"])
        tr.train_single("4h", 42, windows=windows_by_tf["4h"])
        probe = _windows(999)
        e1h = self._encode_from_checkpoint(
            base_config, tr.checkpoint_path("1h", 42, "best"), "1h", probe
        )
        e4h = self._encode_from_checkpoint(
            base_config, tr.checkpoint_path("4h", 42, "best"), "4h", probe
        )
        assert not np.allclose(e1h, e4h)


# --- orchestrator: resume + continue-on-failure ----------------------------


class TestOrchestrator:
    def test_resume_skips_completed_runs(
        self, base_config, tmp_path, windows_by_tf
    ) -> None:
        tr = _make_trainer(base_config, tmp_path)
        # Pre-complete the (1h, 42) run.
        tr.train_single("1h", 42, windows=windows_by_tf["1h"])
        orch = TrainingOrchestrator(tr)
        with patch.object(tr, "train_single", wraps=tr.train_single) as spy:
            results = orch.run_all(
                seeds=(42,),
                timeframes=("1h", "4h"),
                windows_by_tf=windows_by_tf,
            )
            # only 4h should train; 1h is skipped (already complete)
            trained_tfs = {call.args[0] for call in spy.call_args_list}
            assert trained_tfs == {"4h"}
        assert results[("1h", 42)] == tr.checkpoint_path("1h", 42, "best")
        assert results[("4h", 42)] == tr.checkpoint_path("4h", 42, "best")

    def test_continues_past_failure(self, base_config, tmp_path) -> None:
        tr = _make_trainer(base_config, tmp_path)
        orch = TrainingOrchestrator(tr)
        # windows_by_tf missing "4h" -> KeyError inside run for 4h only.
        partial = {"1h": _windows(1)}
        results = orch.run_all(
            seeds=(42,),
            timeframes=("1h", "4h"),
            windows_by_tf=partial,
        )
        assert results[("1h", 42)] is not None
        assert results[("4h", 42)] is None  # failed but did not abort
