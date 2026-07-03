"""
src/models/ts2vec_wrapper.py

TS2Vec branch-encoder wrapper (M7 — TS2Vec Wrapper, ADR-001, ADR-002,
ADR-010).

Purpose
-------
Provide a clean, reproducible interface to the pinned TS2Vec encoder for
a SINGLE timeframe branch (ADR-002: each of the four branches — 15m, 1h,
4h, 1d — is trained independently, and its weights are reused across
every condition that includes that timeframe). This module:

  - wraps TS2Vec training (`train`), inference (`encode`), and full
    reproducibility-bundle checkpointing (`save_checkpoint` /
    `load_checkpoint`, ADR-010);
  - NEVER modifies TS2Vec source (ADR-001). It imports the pinned
    `TS2Vec` class exactly as installed and only orchestrates around it.

`encode` returns one 64-dim vector per input window via TS2Vec's
`encoding_window='full_series'` max-pool over the time axis (DS-03 §3.6;
V-MODEL-001 requires output shape `[N, 64]`).

Provenance of the `TS2Vec` import (deviation, session 9 — 2026-07-03)
--------------------------------------------------------------------
ADR-001 originally specified installing TS2Vec via
`pip install git+...@<pinned_commit>`. That commit's repository has no
`setup.py`/`pyproject.toml`, so it is not pip-installable. Per an
approved session-9 decision, the pinned source is instead imported from
the vendored copy `third_party_reference/ts2vec/` (wired onto `sys.path`
via a `.pth` file in the venv). `from ts2vec import TS2Vec` therefore
resolves to that vendored, byte-identical-to-pinned copy. No TS2Vec
source is copied into `src/`. See docs/CHECKPOINT_LATEST.md (Sesi 9
Keputusan/Deviasi) for the full rationale.

Known frictions with the pinned TS2Vec (documented, NOT worked around by
editing upstream source)
------------------------------------------------------------------------
1. TS2Vec constructs its `AdamW` optimizer *inside* `fit()` as a local
   variable, using the library default `weight_decay` (0.01). Our
   `base.yaml` `training.weight_decay` (1e-4) is therefore NOT applied by
   the upstream code, and the optimizer object is not exposed. The
   effective weight decay actually used is recorded in the checkpoint
   metadata (`effective_weight_decay`) for transparency. Per ADR-021
   (approved, session 9), this is accepted deliberately: the project is
   neutral toward upstream TS2Vec and does NOT monkey-patch/subclass it
   to honor the config value. `base.yaml training.weight_decay` is kept
   as documented design intent; the paper will note this as a limitation.
   See docs/ADR-021_ts2vec_optimizer_defaults.md.
2. Because the optimizer is internal to `fit()`, no `optimizer_state_dict`
   is available for a true optimizer-state resume; ADR-010's
   `optimizer_state_dict` field is therefore written as `None`. Weight-
   level resume (re-`fit` from saved encoder weights) is still possible.
   Also accepted per ADR-021 (same neutrality-toward-upstream rationale).
"""

from __future__ import annotations

import platform
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ts2vec import TS2Vec  # vendored pinned copy via .pth (see module docstring)

from src.utils.logging_utils import get_logger
from src.utils.paths import TIMEFRAMES
from src.utils.seed import set_all_seeds

logger = get_logger(__name__)

# Expected per-window feature/branch-embedding dimensions (DS-03 §3.6).
_ENCODE_OUTPUT_DIM = 64


@dataclass
class TrainingHistory:
    """
    Per-run training telemetry for one branch (ADR-010 metrics fields).

    Attributes
    ----------
    train_loss_history : list[float]
        Mean self-supervised contrastive loss per completed epoch.
    epoch_times : list[float]
        Wall-clock seconds for each completed epoch (same length as
        `train_loss_history`).
    best_epoch : int
        Zero-based index of the epoch with the lowest training loss.
    best_loss : float
        The lowest training loss observed (``float('inf')`` if no epoch
        completed).
    stopped_early : bool
        True if early stopping (patience) halted training before
        `max_epochs`.
    """

    train_loss_history: list[float] = field(default_factory=list)
    epoch_times: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_loss: float = float("inf")
    stopped_early: bool = False


class _EarlyStop(Exception):
    """Internal signal to break out of TS2Vec.fit()'s epoch loop cleanly."""


def _get_git_commit_hash() -> str | None:
    """Return the current repo git commit hash, or None if unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or None
    except (subprocess.SubprocessError, FileNotFoundError):  # pragma: no cover
        return None


class TS2VecBranch:
    """
    A single-timeframe TS2Vec branch encoder.

    One instance corresponds to one branch (e.g. the "1h" branch). Per
    ADR-002 the four branches are trained independently — this class owns
    exactly one TS2Vec model and one (internal) optimizer, guaranteeing
    no shared loss or optimizer across branches (V-INV-004).

    Parameters
    ----------
    config : dict
        Loaded configuration (``base.yaml`` merged with any condition
        override). Reads ``encoder.*`` and ``training.*`` hyperparameters
        plus ``ts2vec.pinned_commit`` and ``fusion.projection_seed`` for
        checkpoint metadata. No hyperparameter is hardcoded here (M0
        philosophy).
    timeframe : str
        One of ``("15m", "1h", "4h", "1d")``.
    device : torch.device
        Device for training/inference (from ``src.utils.device.get_device``).
    """

    def __init__(
        self,
        config: dict[str, Any],
        timeframe: str,
        device: torch.device,
    ) -> None:
        if timeframe not in TIMEFRAMES:
            raise ValueError(
                f"Unknown timeframe '{timeframe}'. Expected one of {TIMEFRAMES}."
            )

        self.config = config
        self.timeframe = timeframe
        self.device = torch.device(device)

        enc = config["encoder"]
        train_cfg = config["training"]

        self._input_dims = int(enc["input_dim"])
        self._output_dims = int(enc["output_dim"])
        self._hidden_dims = int(enc["hidden_dim"])
        self._depth = int(enc["depth"])
        self._lr = float(train_cfg["learning_rate"])
        self._batch_size = int(train_cfg["batch_size"])
        self._max_epochs = int(train_cfg["max_epochs"])
        self._patience = int(train_cfg["early_stopping_patience"])

        # NOTE (friction #1, see module docstring): TS2Vec's fit() builds
        # AdamW internally with the library default weight_decay, ignoring
        # config training.weight_decay. We record what config *asked for*
        # and what is *actually* used, but do not edit upstream source.
        self._config_weight_decay = float(train_cfg.get("weight_decay", 0.0))
        self._effective_weight_decay = _default_adamw_weight_decay()

        self._model = TS2Vec(
            input_dims=self._input_dims,
            output_dims=self._output_dims,
            hidden_dims=self._hidden_dims,
            depth=self._depth,
            device=str(self.device),
            lr=self._lr,
            batch_size=self._batch_size,
            after_epoch_callback=self._on_epoch_end,
        )

        self._history = TrainingHistory()
        self._seed: int | None = None
        # CPU snapshot of net.state_dict() at the lowest-loss epoch, so
        # ADR-010's `best_model.pt` can be written distinctly from the
        # final-epoch `latest_model.pt`. None until the first epoch runs.
        self._best_state_dict: dict[str, torch.Tensor] | None = None
        # Transient state used by the early-stopping callback.
        self._epochs_without_improvement = 0
        self._last_epoch_start: float | None = None

        logger.info(
            "Initialized TS2VecBranch(timeframe=%s, input_dims=%d, output_dims=%d, "
            "hidden_dims=%d, depth=%d, device=%s)",
            self.timeframe,
            self._input_dims,
            self._output_dims,
            self._hidden_dims,
            self._depth,
            self.device,
        )
        if self._config_weight_decay != self._effective_weight_decay:
            logger.warning(
                "config training.weight_decay=%s is NOT applied: TS2Vec.fit() "
                "uses AdamW default weight_decay=%s (optimizer is internal to "
                "upstream fit(); see ts2vec_wrapper module docstring, friction #1).",
                self._config_weight_decay,
                self._effective_weight_decay,
            )

    # -- Training ----------------------------------------------------------

    def _on_epoch_end(self, model: TS2Vec, cum_loss: float) -> None:
        """
        after_epoch_callback for TS2Vec.fit(): records history and applies
        early stopping (patience on training loss).

        TS2Vec increments ``model.n_epochs`` *before* invoking this
        callback, so ``model.n_epochs - 1`` is the just-completed epoch
        index.
        """
        now = time.perf_counter()
        epoch_time = (now - self._last_epoch_start) if self._last_epoch_start else 0.0
        self._last_epoch_start = now

        self._history.train_loss_history.append(float(cum_loss))
        self._history.epoch_times.append(float(epoch_time))
        completed_epoch = len(self._history.train_loss_history) - 1

        if cum_loss < self._history.best_loss:
            self._history.best_loss = float(cum_loss)
            self._history.best_epoch = completed_epoch
            self._epochs_without_improvement = 0
            # Snapshot best-epoch weights (CPU copy) for best_model.pt.
            self._best_state_dict = {
                k: v.detach().cpu().clone()
                for k, v in self._model.net.state_dict().items()
            }
        else:
            self._epochs_without_improvement += 1

        logger.info(
            "[%s] epoch %d: loss=%.6f (best=%.6f @ epoch %d, no-improve=%d/%d)",
            self.timeframe,
            completed_epoch,
            cum_loss,
            self._history.best_loss,
            self._history.best_epoch,
            self._epochs_without_improvement,
            self._patience,
        )

        if self._epochs_without_improvement >= self._patience:
            self._history.stopped_early = True
            logger.info(
                "[%s] early stopping at epoch %d (patience=%d reached).",
                self.timeframe,
                completed_epoch,
                self._patience,
            )
            raise _EarlyStop

    def train(self, windows: np.ndarray, seed: int) -> TrainingHistory:
        """
        Train this branch's TS2Vec encoder on its timeframe windows.

        Parameters
        ----------
        windows : np.ndarray
            Shape ``[N, 48, 7]`` (windows, timesteps, features), the M6
            output for this timeframe. Cast to float32 internally.
        seed : int
            Seed applied to every RNG source before training (INV-007).

        Returns
        -------
        TrainingHistory
            Loss/time history plus best epoch/loss, from at most
            ``training.max_epochs`` epochs (fewer if early-stopped).
        """
        windows = _as_3d_float32(windows)

        set_all_seeds(seed)
        self._seed = seed
        # Reset history/state so re-training an instance is well defined.
        self._history = TrainingHistory()
        self._best_state_dict = None
        self._epochs_without_improvement = 0
        self._last_epoch_start = time.perf_counter()

        logger.info(
            "[%s] training on %d windows, max_epochs=%d, patience=%d, seed=%d",
            self.timeframe,
            windows.shape[0],
            self._max_epochs,
            self._patience,
            seed,
        )

        try:
            self._model.fit(windows, n_epochs=self._max_epochs, verbose=False)
        except _EarlyStop:
            pass  # history already captured in the callback

        logger.info(
            "[%s] training complete: %d epochs, best_loss=%.6f @ epoch %d%s",
            self.timeframe,
            len(self._history.train_loss_history),
            self._history.best_loss,
            self._history.best_epoch,
            " (early-stopped)" if self._history.stopped_early else "",
        )
        return self._history

    # -- Inference ---------------------------------------------------------

    def encode(self, windows: np.ndarray) -> np.ndarray:
        """
        Encode windows into one 64-dim vector each (V-MODEL-001).

        Uses TS2Vec ``encoding_window='full_series'`` (max-pool over the
        time axis). Deterministic given the same input and loaded weights.

        Parameters
        ----------
        windows : np.ndarray
            Shape ``[N, 48, 7]``.

        Returns
        -------
        np.ndarray
            Shape ``[N, 64]``, dtype float32.
        """
        windows = _as_3d_float32(windows)
        reprs = self._model.encode(windows, encoding_window="full_series")
        reprs = np.asarray(reprs, dtype=np.float32)
        if reprs.ndim != 2 or reprs.shape[1] != _ENCODE_OUTPUT_DIM:
            raise RuntimeError(
                f"encode produced shape {reprs.shape}, expected [N, {_ENCODE_OUTPUT_DIM}]. "
                "Check encoder.output_dim in config (must be 64) and that "
                "encoding_window='full_series'."
            )
        return reprs

    # -- Checkpointing (ADR-010) ------------------------------------------

    def save_checkpoint(
        self,
        path: Path,
        extra_metadata: dict[str, Any] | None = None,
        which: str = "latest",
    ) -> None:
        """
        Write a full reproducibility-bundle checkpoint (ADR-010).

        Parameters
        ----------
        path : Path
            Destination ``.pt`` file (parent dirs created if needed).
        extra_metadata : dict, optional
            Caller-supplied fields merged into (and overriding) the
            bundle — e.g. ``condition`` and ``projection_matrix`` /
            ``projection_seed``, which are condition-level (M9) concerns a
            per-timeframe branch does not itself own (ADR-002/ADR-003).
        which : {"latest", "best"}, optional
            Which encoder weights to store in ``model_state_dict``.
            ``"latest"`` (default) uses the current (final-epoch)
            weights; ``"best"`` uses the snapshot from the lowest-loss
            epoch (ADR-010 ``best_model.pt``). If ``"best"`` is requested
            but no epoch has completed (no snapshot), the current weights
            are used and a warning is logged.

        Notes
        -----
        ``optimizer_state_dict`` is ``None`` (friction #2, see module
        docstring). ``model_state_dict`` is TS2Vec's ``net.state_dict()``
        (the AveragedModel/EMA weights used for inference).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if which not in ("latest", "best"):
            raise ValueError(f"which must be 'latest' or 'best', got {which!r}.")
        if which == "best":
            if self._best_state_dict is not None:
                model_state = self._best_state_dict
            else:
                logger.warning(
                    "[%s] save_checkpoint(which='best') requested but no best "
                    "snapshot exists (no epoch completed); saving current weights.",
                    self.timeframe,
                )
                model_state = self._model.net.state_dict()
        else:
            model_state = self._model.net.state_dict()

        bundle: dict[str, Any] = {
            # Model state
            "model_state_dict": model_state,
            "checkpoint_kind": which,
            "branch_timeframe": self.timeframe,
            # Training state (for resume)
            "optimizer_state_dict": None,  # not exposed by upstream fit()
            "scheduler_state_dict": None,
            "epoch": int(self._model.n_epochs),
            "global_step": int(self._model.n_iters),
            "best_metric": float(self._history.best_loss),
            "best_epoch": int(self._history.best_epoch),
            # Experiment identity
            "seed": self._seed,
            "condition": None,  # branch is condition-agnostic; override via extra_metadata
            "config_snapshot": self.config,
            # Reproducibility metadata
            "timestamp": _utc_now_iso(),
            "git_commit_hash": _get_git_commit_hash(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "ts2vec_commit": self.config.get("ts2vec", {}).get("pinned_commit"),
            # Projection matrix (ADR-003) — condition-level, supplied by M9
            "projection_matrix": None,
            "projection_seed": self.config.get("fusion", {}).get("projection_seed"),
            # Training metrics history
            "train_loss_history": list(self._history.train_loss_history),
            "epoch_times": list(self._history.epoch_times),
            # Transparency about the weight-decay friction (see docstring)
            "config_weight_decay": self._config_weight_decay,
            "effective_weight_decay": self._effective_weight_decay,
        }

        if extra_metadata:
            bundle.update(extra_metadata)

        torch.save(bundle, path)
        logger.info(
            "[%s] saved checkpoint -> %s (epoch=%d, best_loss=%.6f)",
            self.timeframe,
            path,
            bundle["epoch"],
            bundle["best_metric"],
        )

    def load_checkpoint(self, path: Path) -> dict[str, Any]:
        """
        Load encoder weights (and history) from an ADR-010 checkpoint.

        Uses ``map_location=self.device`` so a checkpoint saved on GPU
        loads correctly on a CPU-only machine (and vice versa).

        Parameters
        ----------
        path : Path
            Checkpoint ``.pt`` file.

        Returns
        -------
        dict
            The full loaded bundle (so callers can inspect metadata).
        """
        path = Path(path)
        bundle = torch.load(path, map_location=self.device, weights_only=False)

        self._warn_on_missing_adr010_fields(bundle)

        self._model.net.load_state_dict(bundle["model_state_dict"])
        self._model.n_epochs = int(bundle.get("epoch", 0))
        self._model.n_iters = int(bundle.get("global_step", 0))

        self._seed = bundle.get("seed")
        self._history = TrainingHistory(
            train_loss_history=list(bundle.get("train_loss_history", [])),
            epoch_times=list(bundle.get("epoch_times", [])),
            best_epoch=int(bundle.get("best_epoch", -1)),
            best_loss=float(bundle.get("best_metric", float("inf"))),
        )

        logger.info(
            "[%s] loaded checkpoint <- %s (epoch=%d, ts2vec_commit=%s)",
            self.timeframe,
            path,
            self._model.n_epochs,
            bundle.get("ts2vec_commit"),
        )
        return bundle

    @staticmethod
    def _warn_on_missing_adr010_fields(bundle: dict[str, Any]) -> None:
        """Warn (do not fail) if an older checkpoint lacks ADR-010 fields."""
        required = (
            "model_state_dict",
            "branch_timeframe",
            "config_snapshot",
            "ts2vec_commit",
            "projection_seed",
            "train_loss_history",
        )
        missing = [k for k in required if k not in bundle]
        if missing:
            logger.warning(
                "Checkpoint is missing ADR-010 fields %s — loading anyway "
                "(backward compatibility).",
                missing,
            )

    @property
    def history(self) -> TrainingHistory:
        """The most recent training history for this branch."""
        return self._history


# --- module-level helpers ---------------------------------------------------


def _as_3d_float32(windows: np.ndarray) -> np.ndarray:
    """Validate and coerce a window tensor to contiguous float32 [N, T, F]."""
    arr = np.asarray(windows)
    if arr.ndim != 3:
        raise ValueError(
            f"windows must be 3D [N, timesteps, features]; got shape {arr.shape}."
        )
    return np.ascontiguousarray(arr, dtype=np.float32)


def _default_adamw_weight_decay() -> float:
    """
    The weight_decay TS2Vec.fit() actually uses (torch AdamW default).

    Read from the live torch default rather than hardcoded, so this stays
    correct if the pinned torch version's default ever changes.
    """
    return float(torch.optim.AdamW([torch.zeros(1, requires_grad=True)]).defaults["weight_decay"])


def _utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string (ADR-010 `timestamp`)."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
