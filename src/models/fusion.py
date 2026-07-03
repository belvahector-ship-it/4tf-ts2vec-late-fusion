"""
src/models/fusion.py

Late fusion module (M9 — Fusion, ADR-003, ADR-013).

Purpose
-------
Combine the independently-trained branch embeddings (M8, each `[N, 64]`)
into a single fixed-size representation per experimental condition, using
**deterministic late fusion with ZERO learnable parameters**:

  1. Concatenate the active branches' 64-dim embeddings in the fixed
     fine-to-coarse order 15m, 1h, 4h, 1d (ADR-013). This yields
     64/128/192/256 dims for 1TF/2TF/3TF/4TF (and 64 for the single-TF
     secondary baselines BL-15m/BL-4h/BL-1d).
  2. Apply a fixed, seeded random projection P (ADR-003) mapping the
     concatenated vector to a common **256-dim** space, so every
     condition is compared in the same dimensionality (avoids the
     dimensionality confound in geometric metrics). P is generated once
     from `projection_seed`, its rows are L2-normalized, and it is frozen
     (`requires_grad=False`). Even 4TF (concat dim already 256) passes
     through P — it is a projection, not an identity.

This module never trains anything: the number of trainable parameters is
exactly 0 (V-INV-003), and P is identical given the same seed and input
dimension (V-MODEL-004). External baselines (HMM, KM-PCA) do NOT use this
module (see M10.5).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Fixed fine-to-coarse concatenation order (ADR-013). Enforced as an
# ordered tuple, never a set/dict, so branch order can never vary.
DEFAULT_CONCAT_ORDER: tuple[str, ...] = ("15m", "1h", "4h", "1d")

BRANCH_EMBED_DIM: int = 64  # per-branch embedding dim (DS-03 §3.6)
FUSED_OUTPUT_DIM: int = 256  # common projected dim (ADR-003)


class FusionError(RuntimeError):
    """Raised when fusion inputs are inconsistent with the condition."""


def order_active_timeframes(
    active_timeframes: Sequence[str],
    concat_order: Sequence[str] = DEFAULT_CONCAT_ORDER,
) -> list[str]:
    """
    Return `active_timeframes` sorted into the fixed concat order (ADR-013).

    Parameters
    ----------
    active_timeframes : sequence of str
        The condition's active timeframes (any order).
    concat_order : sequence of str, optional
        The canonical fine-to-coarse order, default 15m,1h,4h,1d.

    Returns
    -------
    list[str]
        The active timeframes in canonical order.

    Raises
    ------
    FusionError
        If `active_timeframes` is empty or contains an unknown timeframe.
    """
    active = list(active_timeframes)
    if not active:
        raise FusionError("A condition must have at least one active timeframe.")
    unknown = [tf for tf in active if tf not in concat_order]
    if unknown:
        raise FusionError(
            f"Unknown timeframe(s) {unknown}; expected subset of {tuple(concat_order)}."
        )
    return [tf for tf in concat_order if tf in active]


def build_projection_matrix(
    input_dim: int,
    output_dim: int = FUSED_OUTPUT_DIM,
    projection_seed: int = 42,
) -> torch.Tensor:
    """
    Build the fixed random projection P ∈ R^{output_dim × input_dim} (ADR-003).

    Deterministic given `(input_dim, output_dim, projection_seed)`: a
    fresh, seeded `torch.Generator` draws a Gaussian matrix; each ROW is
    then L2-normalized (orthogonal-like, Johnson-Lindenstrauss). The
    matrix is frozen (`requires_grad=False`) — it is never trained.

    Returns
    -------
    torch.Tensor
        Shape `[output_dim, input_dim]`, dtype float32, requires_grad=False.
    """
    if input_dim <= 0:
        raise FusionError(f"input_dim must be positive, got {input_dim}.")
    generator = torch.Generator().manual_seed(int(projection_seed))
    p = torch.randn(output_dim, input_dim, generator=generator, dtype=torch.float32)
    # Row-normalize to unit L2 norm (ADR-003 "orthogonal-like behavior").
    row_norms = torch.linalg.norm(p, dim=1, keepdim=True)
    p = p / row_norms
    p.requires_grad_(False)
    return p


class FusionModule:
    """
    Deterministic late-fusion for one experimental condition (ADR-003/013).

    Parameters
    ----------
    condition : str
        Condition label (e.g. "2TF", "BL-4h"). Used for logging/metadata.
    active_timeframes : sequence of str
        The condition's active branches (any order; reordered to the
        fixed concat order internally).
    projection_seed : int, optional
        Seed for the fixed random projection, default 42.
    output_dim : int, optional
        Projected dimensionality, default 256.
    branch_dim : int, optional
        Per-branch embedding dim, default 64.
    concat_order : sequence of str, optional
        Canonical concat order, default 15m,1h,4h,1d.
    """

    def __init__(
        self,
        condition: str,
        active_timeframes: Sequence[str],
        projection_seed: int = 42,
        output_dim: int = FUSED_OUTPUT_DIM,
        branch_dim: int = BRANCH_EMBED_DIM,
        concat_order: Sequence[str] = DEFAULT_CONCAT_ORDER,
    ) -> None:
        self.condition = condition
        self.projection_seed = int(projection_seed)
        self.output_dim = int(output_dim)
        self.branch_dim = int(branch_dim)
        self.concat_order = tuple(concat_order)
        self.ordered_timeframes = order_active_timeframes(active_timeframes, concat_order)
        self.input_dim = self.branch_dim * len(self.ordered_timeframes)
        # Built once, deterministic, frozen.
        self._projection = build_projection_matrix(
            self.input_dim, self.output_dim, self.projection_seed
        )
        logger.info(
            "FusionModule(condition=%s): concat order %s -> input_dim=%d -> "
            "projected output_dim=%d (projection_seed=%d, trainable_params=0)",
            self.condition,
            self.ordered_timeframes,
            self.input_dim,
            self.output_dim,
            self.projection_seed,
        )

    @property
    def projection_matrix(self) -> torch.Tensor:
        """The fixed projection matrix P (`[output_dim, input_dim]`, frozen)."""
        return self._projection

    def n_trainable_parameters(self) -> int:
        """Number of trainable parameters — always 0 (V-INV-003)."""
        return 0

    def concatenate(self, branch_embeddings: dict[str, np.ndarray]) -> np.ndarray:
        """
        Concatenate the active branch embeddings in the fixed order.

        Parameters
        ----------
        branch_embeddings : dict[str, np.ndarray]
            Maps timeframe -> `[N, branch_dim]` array. Must contain (at
            least) every active timeframe of this condition, all with the
            same N.

        Returns
        -------
        np.ndarray
            Shape `[N, input_dim]`, dtype float32.
        """
        missing = [tf for tf in self.ordered_timeframes if tf not in branch_embeddings]
        if missing:
            raise FusionError(
                f"Condition '{self.condition}' needs branch embeddings for "
                f"{missing}, which were not provided."
            )
        parts = []
        n_ref = None
        for tf in self.ordered_timeframes:
            arr = np.asarray(branch_embeddings[tf], dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != self.branch_dim:
                raise FusionError(
                    f"branch '{tf}' embedding must be [N, {self.branch_dim}], "
                    f"got {arr.shape}."
                )
            if n_ref is None:
                n_ref = arr.shape[0]
            elif arr.shape[0] != n_ref:
                raise FusionError(
                    f"branch '{tf}' has N={arr.shape[0]} but expected N={n_ref} "
                    "(all active branches must share the same window count)."
                )
            parts.append(arr)
        return np.concatenate(parts, axis=1)

    def fuse(self, branch_embeddings: dict[str, np.ndarray]) -> np.ndarray:
        """
        Concatenate active branches and apply the fixed projection.

        Parameters
        ----------
        branch_embeddings : dict[str, np.ndarray]
            Maps timeframe -> `[N, branch_dim]`.

        Returns
        -------
        np.ndarray
            Fused embeddings `[N, output_dim]` (256), dtype float32
            (V-INV-002 / V-MODEL-002).
        """
        concat = self.concatenate(branch_embeddings)  # [N, input_dim]
        with torch.no_grad():
            x = torch.from_numpy(concat)  # [N, input_dim]
            fused = x @ self._projection.T  # [N, output_dim]
        result = fused.numpy().astype(np.float32)
        if result.shape[1] != self.output_dim:
            raise FusionError(
                f"fused output has dim {result.shape[1]}, expected {self.output_dim}."
            )
        return result


class EmbeddingPipeline:
    """
    Produce branch and fused embeddings for a seed across conditions.

    Loads each trained branch checkpoint (M8), encodes the train/test
    windows to `[N, 64]` per branch (M7), then fuses per condition. The
    branch encoding for a timeframe is computed ONCE per seed and reused
    by every condition that includes it (consistent with ADR-002 branch
    reuse).

    Parameters
    ----------
    config : dict
        Loaded base config (encoder/fusion/training params).
    branch_loader : callable
        `branch_loader(timeframe, seed) -> object` returning something
        with `.encode(windows) -> np.ndarray [N, 64]`. In production this
        loads a `TS2VecBranch` from its checkpoint; tests may inject a
        stub. Kept as an injected dependency so the deterministic fusion
        logic is testable without real checkpoints on disk.
    """

    def __init__(self, config: dict[str, Any], branch_loader) -> None:
        self.config = config
        self.branch_loader = branch_loader
        self.concat_order = tuple(
            config.get("fusion", {}).get("concat_order", DEFAULT_CONCAT_ORDER)
        )
        self.projection_seed = int(
            config.get("fusion", {}).get("projection_seed", 42)
        )
        self.output_dim = int(config.get("fusion", {}).get("output_dim", FUSED_OUTPUT_DIM))

    def encode_all_branches(
        self,
        seed: int,
        windows_by_tf: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """
        Encode every provided timeframe's windows to `[N, 64]` for `seed`.

        Parameters
        ----------
        seed : int
            The seed whose branch checkpoints to load.
        windows_by_tf : dict[str, np.ndarray]
            Maps timeframe -> windows `[N, 48, 7]` (one split).

        Returns
        -------
        dict[str, np.ndarray]
            Maps timeframe -> `[N, 64]` branch embeddings.
        """
        out: dict[str, np.ndarray] = {}
        for tf, windows in windows_by_tf.items():
            branch = self.branch_loader(tf, seed)
            out[tf] = np.asarray(branch.encode(windows), dtype=np.float32)
        return out

    def fuse_condition(
        self,
        condition: str,
        active_timeframes: Sequence[str],
        branch_embeddings: dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Fuse the branch embeddings for one condition to `[N, 256]`.

        Parameters
        ----------
        condition : str
            Condition label.
        active_timeframes : sequence of str
            The condition's active timeframes.
        branch_embeddings : dict[str, np.ndarray]
            Branch embeddings keyed by timeframe (superset allowed).

        Returns
        -------
        np.ndarray
            Fused `[N, 256]`.
        """
        module = FusionModule(
            condition,
            active_timeframes,
            projection_seed=self.projection_seed,
            output_dim=self.output_dim,
            concat_order=self.concat_order,
        )
        return module.fuse(branch_embeddings)
