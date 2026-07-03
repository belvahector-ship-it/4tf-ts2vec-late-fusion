"""
src/models/branch_training.py

Branch training orchestration (M8 — Branch Training, ADR-002, ADR-010,
ADR-019).

Purpose
-------
Execute the FOUR independent TS2Vec branch encoders (one per timeframe:
15m, 1h, 4h, 1d), each trained once per seed, under the five-seed
protocol (ADR-019). This is **4 timeframes × 5 seeds = 20 training runs**
— NOT one run per experimental condition. The trained branch weights are
condition-agnostic (ADR-002): every condition that includes a given
timeframe reuses the identical checkpoint for that (timeframe, seed).

Each run writes an ADR-010 reproducibility bundle in two variants:
``best_model.pt`` (lowest-loss epoch) and ``latest_model.pt``
(final epoch, for weight-level resume), under
``checkpoints/branch_{tf}/seed_{seed}/``, plus a per-run log file
``logs/training_branch_{tf}_seed_{seed}.log`` with epoch-level detail.

Resume / idempotency
--------------------
``load_or_train`` skips a run whose ``best_model.pt`` is already a valid
checkpoint, so re-invoking the orchestrator continues where it left off
(run-level resume). Note (per ADR-021): TS2Vec does not expose its
optimizer, so there is no true optimizer-state (epoch-level) resume —
``latest_model.pt`` enables weight-level continuation only. Resume here
therefore means "skip already-completed runs", which is the granularity
the 20-run protocol needs.

This module never trains on windowed tensors it did not receive from M6
(``data/processed/{split}_windows_{tf}.npy``); it is subject to the
V-LEAK gate having passed first (IMP-01 §5 Coding Order).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from src.models.ts2vec_wrapper import TrainingHistory, TS2VecBranch
from src.utils.logging_utils import get_logger
from src.utils.paths import (
    CHECKPOINTS_DIR,
    DATA_PROCESSED_DIR,
    LOGS_DIR,
    RANDOM_SEEDS,
    TIMEFRAMES,
)

logger = get_logger(__name__)

# Loggers whose records should also be captured into a per-run log file.
_RUN_LOG_SOURCES = ("src.models.branch_training", "src.models.ts2vec_wrapper")
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Keys a checkpoint must contain to count as "valid" for skip/resume.
_MIN_CHECKPOINT_KEYS = ("model_state_dict", "branch_timeframe")


class BranchTrainingError(RuntimeError):
    """Raised when a required window file is missing or a run cannot complete."""


@contextmanager
def _run_log_file(path: Path) -> Iterator[None]:
    """
    Attach a FileHandler capturing epoch-level detail for one run, then
    detach it on exit (so each run's log is self-contained).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))
    attached = [logging.getLogger(name) for name in _RUN_LOG_SOURCES]
    for lg in attached:
        lg.addHandler(handler)
    try:
        yield
    finally:
        for lg in attached:
            lg.removeHandler(handler)
        handler.close()


class BranchTrainer:
    """
    Trains individual TS2Vec branches and manages their checkpoints.

    Parameters
    ----------
    config : dict
        Loaded configuration (base.yaml merged with any override).
    device : torch.device
        Training/inference device.
    windows_dir : Path, optional
        Directory holding M6 outputs ``{split}_windows_{tf}.npy``.
        Default ``data/processed/``.
    checkpoints_dir : Path, optional
        Root for branch checkpoints. Default ``checkpoints/``.
    logs_dir : Path, optional
        Directory for per-run log files. Default ``logs/``.
    """

    def __init__(
        self,
        config: dict[str, Any],
        device: torch.device,
        windows_dir: Path = DATA_PROCESSED_DIR,
        checkpoints_dir: Path = CHECKPOINTS_DIR,
        logs_dir: Path = LOGS_DIR,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.windows_dir = Path(windows_dir)
        self.checkpoints_dir = Path(checkpoints_dir)
        self.logs_dir = Path(logs_dir)

    # -- path helpers ------------------------------------------------------

    def branch_run_dir(self, timeframe: str, seed: int) -> Path:
        """Return ``checkpoints/branch_{tf}/seed_{seed}/`` (not created)."""
        if timeframe not in TIMEFRAMES:
            raise ValueError(
                f"Unknown timeframe '{timeframe}'. Expected one of {TIMEFRAMES}."
            )
        return self.checkpoints_dir / f"branch_{timeframe}" / f"seed_{seed}"

    def checkpoint_path(
        self, timeframe: str, seed: int, which: str = "best"
    ) -> Path:
        """
        Return the checkpoint path for one (timeframe, seed) run.

        This is condition-agnostic (ADR-002): every condition that
        includes ``timeframe`` reuses exactly this path — there is a
        single trained branch per (timeframe, seed).
        """
        if which not in ("best", "latest"):
            raise ValueError(f"which must be 'best' or 'latest', got {which!r}.")
        fname = "best_model.pt" if which == "best" else "latest_model.pt"
        return self.branch_run_dir(timeframe, seed) / fname

    def run_log_path(self, timeframe: str, seed: int) -> Path:
        return self.logs_dir / f"training_branch_{timeframe}_seed_{seed}.log"

    # -- data --------------------------------------------------------------

    def load_train_windows(self, timeframe: str) -> np.ndarray:
        """Load M6 train windows for a timeframe: ``train_windows_{tf}.npy``."""
        path = self.windows_dir / f"train_windows_{timeframe}.npy"
        if not path.exists():
            raise BranchTrainingError(
                f"Missing M6 window file {path}. Run the data pipeline "
                f"(M1-M6) first; M8 trains on M6 output, not raw data."
            )
        return np.load(path)

    # -- checkpoint validity ----------------------------------------------

    def is_valid_checkpoint(self, path: Path) -> bool:
        """True if `path` exists and contains the minimum ADR-010 keys."""
        if not Path(path).exists():
            return False
        try:
            bundle = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001 - corrupt/partial file
            logger.warning("Checkpoint %s is unreadable (%s); treating as invalid.", path, exc)
            return False
        return all(k in bundle for k in _MIN_CHECKPOINT_KEYS)

    # -- training ----------------------------------------------------------

    def train_single(
        self,
        timeframe: str,
        seed: int,
        windows: np.ndarray | None = None,
    ) -> TrainingHistory:
        """
        Train one branch for one seed and write best/latest checkpoints.

        Parameters
        ----------
        timeframe : str
            One of the four timeframes.
        seed : int
            Seed applied to all RNGs (ADR-019 five-seed protocol).
        windows : np.ndarray, optional
            Training windows ``[N, 48, 7]``. If omitted, loaded from the
            M6 ``train_windows_{tf}.npy`` file.

        Returns
        -------
        TrainingHistory
        """
        if windows is None:
            windows = self.load_train_windows(timeframe)

        best_path = self.checkpoint_path(timeframe, seed, "best")
        latest_path = self.checkpoint_path(timeframe, seed, "latest")

        with _run_log_file(self.run_log_path(timeframe, seed)):
            logger.info(
                "=== M8 branch run START: timeframe=%s seed=%d (device=%s) ===",
                timeframe,
                seed,
                self.device,
            )
            branch = TS2VecBranch(self.config, timeframe, self.device)
            history = branch.train(windows, seed=seed)

            reuse_note = {
                "branch_reuse_note": (
                    f"Branch '{timeframe}' seed {seed} is shared UNCHANGED by "
                    f"every condition that includes the {timeframe} timeframe "
                    f"(ADR-002)."
                )
            }
            branch.save_checkpoint(best_path, extra_metadata=reuse_note, which="best")
            branch.save_checkpoint(
                latest_path, extra_metadata=reuse_note, which="latest"
            )
            logger.info(
                "=== M8 branch run DONE: timeframe=%s seed=%d "
                "(epochs=%d, best_loss=%.6f) ===",
                timeframe,
                seed,
                len(history.train_loss_history),
                history.best_loss,
            )
        return history

    def load_or_train(
        self,
        timeframe: str,
        seed: int,
        windows: np.ndarray | None = None,
    ) -> Path:
        """
        Return the best-checkpoint path for (timeframe, seed), training
        only if a valid checkpoint does not already exist (idempotency).

        Returns
        -------
        Path
            Path to ``best_model.pt`` for this run.
        """
        best_path = self.checkpoint_path(timeframe, seed, "best")
        if self.is_valid_checkpoint(best_path):
            logger.info(
                "Skipping branch %s seed %d: valid checkpoint exists at %s.",
                timeframe,
                seed,
                best_path,
            )
            return best_path
        self.train_single(timeframe, seed, windows=windows)
        return best_path

    def train_all_branches(
        self,
        seeds: tuple[int, ...] = RANDOM_SEEDS,
        timeframes: tuple[str, ...] = TIMEFRAMES,
        windows_by_tf: dict[str, np.ndarray] | None = None,
    ) -> dict[tuple[str, int], Path]:
        """
        Train (or skip) every (timeframe, seed) branch.

        This performs exactly ``len(timeframes) * len(seeds)`` runs —
        four per seed by default (one per timeframe), never one per
        condition (ADR-002).

        Parameters
        ----------
        seeds, timeframes : tuple
            The seeds and timeframes to cover (defaults: 5 seeds × 4 TF).
        windows_by_tf : dict, optional
            In-memory ``{tf: windows}`` to train on (used by tests to
            avoid disk I/O). If omitted, windows are loaded per timeframe
            from the M6 files.

        Returns
        -------
        dict[(timeframe, seed) -> Path]
            Best-checkpoint path for each run.
        """
        results: dict[tuple[str, int], Path] = {}
        for seed in seeds:
            runs_this_seed = 0
            for tf in timeframes:
                windows = None if windows_by_tf is None else windows_by_tf[tf]
                results[(tf, seed)] = self.load_or_train(tf, seed, windows=windows)
                runs_this_seed += 1
            logger.info("Seed %d complete: %d branch runs.", seed, runs_this_seed)
        logger.info(
            "train_all_branches complete: %d checkpoints (%d timeframes × %d seeds).",
            len(results),
            len(timeframes),
            len(seeds),
        )
        return results


class TrainingOrchestrator:
    """
    Coordinates the full 20-run branch-training protocol with run-level
    resume.

    Thin coordinator over :class:`BranchTrainer`. ``run_all`` drives all
    four timeframes across all five seeds; already-completed runs are
    skipped (resume), and a failing run is logged and does not abort the
    remaining runs (so a long protocol makes forward progress).
    """

    def __init__(self, trainer: BranchTrainer) -> None:
        self.trainer = trainer

    def run_all(
        self,
        seeds: tuple[int, ...] = RANDOM_SEEDS,
        timeframes: tuple[str, ...] = TIMEFRAMES,
        windows_by_tf: dict[str, np.ndarray] | None = None,
    ) -> dict[tuple[str, int], Path | None]:
        """
        Run every (timeframe, seed); continue past individual failures.

        Returns
        -------
        dict[(timeframe, seed) -> Path | None]
            Best-checkpoint path per completed run, or ``None`` if that
            run failed (the failure is logged).
        """
        results: dict[tuple[str, int], Path | None] = {}
        n_ok = n_fail = 0
        for seed in seeds:
            for tf in timeframes:
                try:
                    # Lookup is inside the try so a per-run data-access
                    # error is logged-and-continued like any other failure.
                    windows = None if windows_by_tf is None else windows_by_tf[tf]
                    results[(tf, seed)] = self.trainer.load_or_train(
                        tf, seed, windows=windows
                    )
                    n_ok += 1
                except Exception as exc:  # noqa: BLE001 - keep going per M8 DoD
                    logger.error(
                        "Branch run FAILED (timeframe=%s seed=%d): %s — continuing.",
                        tf,
                        seed,
                        exc,
                    )
                    results[(tf, seed)] = None
                    n_fail += 1
        logger.info("run_all finished: %d ok, %d failed.", n_ok, n_fail)
        return results
