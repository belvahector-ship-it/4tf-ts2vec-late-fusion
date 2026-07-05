"""
src/models/hdbscan_clustering.py

HDBSCAN clustering (M10, ADR-004, ADR-018, DS-02 Stage 8, DS-03 §5).

Implements the two-stage protocol's **Stage 1 (primary)**: grid-search
HDBSCAN on the 1TF fused embeddings only, lock the configuration that
maximizes Silhouette Score with 2 ≤ k ≤ 8 clusters, then apply those
locked parameters UNCHANGED to every TS2Vec condition (V-EXP-001). Test
embeddings are labeled via ``hdbscan.approximate_predict`` (not re-fit).

Key rules:
- ``k`` (non-noise cluster count) must be within [min_clusters=2,
  max_clusters=8] (ADR-018).
- Ties in Silhouette are broken by lower ``min_cluster_size`` first, then
  lower ``min_samples`` (Risk R-08 tiebreaker).
- Fallback (ADR-004): if the locked params yield < 2 non-noise clusters
  for any condition, expand ``min_cluster_size`` grid to [20,50,100,200]
  and re-lock from 1TF; logged as WARNING.
- Noise points (label ``-1``) are RETAINED in all artifacts; they are
  excluded only from Silhouette during selection (and from metrics in
  M11).

Computational note (documented engineering choice):
Full ``silhouette_score`` on ~26k×256 embeddings is O(n²) (a ~5.5 GB
distance matrix) — infeasible to run 9× per seed per condition. For the
grid-SELECTION Silhouette here we therefore use a deterministic random
subsample (``silhouette_sample_size``, fixed ``random_state``). This
affects only which HDBSCAN params are *selected*; the final reported
Stage-9 geometric metrics (M11) are computed separately per DS-02.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

import hdbscan

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

METRIC = "euclidean"  # fixed (ADR-004)
# Subsample size for grid-selection silhouette (see module docstring).
SILHOUETTE_SAMPLE_SIZE = 5000
# Fallback grid (ADR-004) if Stage-1 params yield < 2 non-noise clusters.
FALLBACK_MIN_CLUSTER_SIZE_GRID = [20, 50, 100, 200]


class ClusteringError(RuntimeError):
    """Raised when no grid configuration yields a usable clustering."""


@dataclass
class GridSearchResult:
    """Outcome of an HDBSCAN grid search."""

    min_cluster_size: int
    min_samples: int
    silhouette: float
    n_clusters: int
    noise_fraction: float
    table: list[dict[str, Any]] = field(default_factory=list)  # every config tried


def count_clusters(labels: np.ndarray) -> int:
    """Number of non-noise clusters (unique labels excluding -1)."""
    uniq = set(np.unique(labels).tolist())
    uniq.discard(-1)
    return len(uniq)


def noise_fraction(labels: np.ndarray) -> float:
    """Fraction of points labeled noise (-1)."""
    return float(np.mean(labels == -1)) if len(labels) else 0.0


def silhouette_noise_excluded(
    embeddings: np.ndarray,
    labels: np.ndarray,
    sample_size: int | None = SILHOUETTE_SAMPLE_SIZE,
    random_state: int = 0,
) -> float:
    """
    Silhouette Score with noise points removed (DS-02 Stage 8 / V-EXP-005).

    Returns ``float('-inf')`` when the score is undefined (fewer than 2
    non-noise clusters, or every non-noise point in one cluster), so such
    configurations are never selected.
    """
    mask = labels != -1
    y = labels[mask]
    if count_clusters(labels) < 2:
        return float("-inf")
    x = embeddings[mask]
    n = x.shape[0]
    kwargs: dict[str, Any] = {"metric": METRIC}
    if sample_size is not None and n > sample_size:
        kwargs.update(sample_size=sample_size, random_state=random_state)
    try:
        return float(silhouette_score(x, y, **kwargs))
    except ValueError:
        return float("-inf")


class HDBSCANClusterer:
    """
    Thin wrapper over ``hdbscan.HDBSCAN`` with grid search, fit, and
    ``approximate_predict``-based test labeling.

    Parameters
    ----------
    min_cluster_size_grid, min_samples_grid : sequence of int
        Grid axes (DS-03 §5: [50,100,200] × [5,10,20]).
    min_clusters, max_clusters : int
        Allowed non-noise cluster count (2 and 8, ADR-018).
    silhouette_sample_size : int
        Subsample for grid-selection silhouette (see module docstring).
    """

    def __init__(
        self,
        min_cluster_size_grid: Sequence[int] = (50, 100, 200),
        min_samples_grid: Sequence[int] = (5, 10, 20),
        min_clusters: int = 2,
        max_clusters: int = 8,
        silhouette_sample_size: int = SILHOUETTE_SAMPLE_SIZE,
    ) -> None:
        self.min_cluster_size_grid = list(min_cluster_size_grid)
        self.min_samples_grid = list(min_samples_grid)
        self.min_clusters = int(min_clusters)
        self.max_clusters = int(max_clusters)
        self.silhouette_sample_size = int(silhouette_sample_size)
        self._clusterer: hdbscan.HDBSCAN | None = None

    def _fit_labels(
        self, embeddings: np.ndarray, mcs: int, ms: int
    ) -> tuple[hdbscan.HDBSCAN, np.ndarray]:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=int(mcs),
            min_samples=int(ms),
            metric=METRIC,
            prediction_data=True,  # required for approximate_predict on test
            core_dist_n_jobs=-1,   # parallelize core-distance calc (speed only;
                                   # does not change clustering results)
        )
        labels = clusterer.fit_predict(np.ascontiguousarray(embeddings, dtype=np.float64))
        return clusterer, labels

    def grid_search(
        self,
        embeddings: np.ndarray,
        seed: int = 0,
        min_cluster_size_grid: Sequence[int] | None = None,
    ) -> GridSearchResult:
        """
        Grid-search HDBSCAN; select max-Silhouette config with 2 ≤ k ≤ 8.

        Tie-break: lower ``min_cluster_size`` first, then lower
        ``min_samples`` (R-08).

        Returns
        -------
        GridSearchResult

        Raises
        ------
        ClusteringError
            If no configuration in the grid produces 2 ≤ k ≤ 8 clusters.
        """
        mcs_grid = list(min_cluster_size_grid or self.min_cluster_size_grid)
        table: list[dict[str, Any]] = []
        # candidates sorted by tie-break order so the first max is chosen
        for mcs in sorted(mcs_grid):
            for ms in sorted(self.min_samples_grid):
                _, labels = self._fit_labels(embeddings, mcs, ms)
                k = count_clusters(labels)
                eligible = self.min_clusters <= k <= self.max_clusters
                sil = (
                    silhouette_noise_excluded(
                        embeddings, labels, self.silhouette_sample_size, seed
                    )
                    if eligible
                    else float("-inf")
                )
                table.append({
                    "min_cluster_size": mcs, "min_samples": ms, "n_clusters": k,
                    "noise_fraction": round(noise_fraction(labels), 4),
                    "silhouette": (None if sil == float("-inf") else round(sil, 6)),
                    "eligible": eligible,
                })
                logger.info(
                    "[grid] mcs=%d ms=%d -> k=%d noise=%.3f sil=%s%s",
                    mcs, ms, k, noise_fraction(labels),
                    "n/a" if sil == float("-inf") else f"{sil:.4f}",
                    "" if eligible else " (k out of [2,8])",
                )

        eligible_rows = [r for r in table if r["eligible"] and r["silhouette"] is not None]
        if not eligible_rows:
            raise ClusteringError(
                f"No config produced {self.min_clusters}<=k<={self.max_clusters} clusters."
            )
        # max silhouette; ties -> lowest mcs, then lowest ms (already sorted,
        # so a strict '>' keeps the first/best tie-break winner)
        best = eligible_rows[0]
        for r in eligible_rows[1:]:
            if r["silhouette"] > best["silhouette"]:
                best = r
        return GridSearchResult(
            min_cluster_size=best["min_cluster_size"],
            min_samples=best["min_samples"],
            silhouette=best["silhouette"],
            n_clusters=best["n_clusters"],
            noise_fraction=best["noise_fraction"],
            table=table,
        )

    def fit(self, embeddings: np.ndarray, params: dict[str, int]) -> np.ndarray:
        """Fit HDBSCAN with fixed params; return train labels (int, -1=noise)."""
        self._clusterer, labels = self._fit_labels(
            embeddings, params["min_cluster_size"], params["min_samples"]
        )
        return labels.astype(np.int32)

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Label new (test) points via ``approximate_predict`` (no re-fit)."""
        if self._clusterer is None:
            raise RuntimeError("HDBSCANClusterer: call fit before predict.")
        labels, _ = hdbscan.approximate_predict(
            self._clusterer, np.ascontiguousarray(embeddings, dtype=np.float64)
        )
        return labels.astype(np.int32)


def build_cluster_parquet(
    labels: np.ndarray,
    timestamps_ns: np.ndarray,
    ohlcv_df: pd.DataFrame,
    condition: str,
    seed: int,
) -> pd.DataFrame:
    """
    Build the DS-02 Stage 8 cluster-artifact DataFrame.

    Parameters
    ----------
    labels : np.ndarray
        Cluster labels ``[N]`` (int, -1 = noise).
    timestamps_ns : np.ndarray
        Anchor timestamps per window ``[N]`` (int64 Unix ns UTC, from M6).
    ohlcv_df : pd.DataFrame
        Indexed/keyed by ``timestamp`` (datetime64[ns, UTC]) with columns
        ``open_1h, high_1h, low_1h, close_1h, volume_1h, close_return_1h``.
    condition, seed : str, int
        Run identity columns.

    Returns
    -------
    pd.DataFrame
        Columns per DS-02 Stage 8 schema (timestamp, OHLCV, close_return,
        cluster_label int32, is_noise bool, condition str, seed int32).
    """
    ts = pd.to_datetime(np.asarray(timestamps_ns, dtype="int64"), utc=True)
    df = pd.DataFrame({
        "timestamp": ts,
        "cluster_label": np.asarray(labels, dtype=np.int32),
    })
    df["is_noise"] = df["cluster_label"] == -1
    join_cols = ["open_1h", "high_1h", "low_1h", "close_1h", "volume_1h", "close_return_1h"]
    df = df.merge(
        ohlcv_df[["timestamp", *join_cols]], on="timestamp", how="left"
    )
    df["condition"] = str(condition)
    df["seed"] = np.int32(seed)
    # column order per DS-02 Stage 8
    return df[[
        "timestamp", "open_1h", "high_1h", "low_1h", "close_1h", "volume_1h",
        "close_return_1h", "cluster_label", "is_noise", "condition", "seed",
    ]]


class ClusteringPipeline:
    """
    Stage-1 primary clustering across all TS2Vec conditions for one seed.

    Locks HDBSCAN params from the 1TF condition and applies them unchanged
    to every condition (V-EXP-001); test labels via approximate_predict.
    """

    def __init__(self, clusterer: HDBSCANClusterer, grid_search_condition: str = "1TF") -> None:
        self.clusterer = clusterer
        self.grid_search_condition = grid_search_condition

    def run_stage1(
        self,
        seed: int,
        train_by_condition: dict[str, np.ndarray],
        test_by_condition: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        """
        Grid-search on 1TF, lock, cluster every condition.

        Parameters
        ----------
        seed : int
        train_by_condition, test_by_condition : dict[str, np.ndarray]
            Fused train/test embeddings per condition (`[N, 256]`).

        Returns
        -------
        dict
            ``{"locked_params", "grid_table", "labels": {cond: {"train","test"}}}``
        """
        gsc = self.grid_search_condition
        if gsc not in train_by_condition:
            raise ClusteringError(f"Grid-search condition '{gsc}' embeddings missing.")

        gs = self.clusterer.grid_search(train_by_condition[gsc], seed=seed)
        locked = {"min_cluster_size": gs.min_cluster_size, "min_samples": gs.min_samples}
        logger.info(
            "[seed=%d] locked from %s: %s (sil=%.4f, k=%d)",
            seed, gsc, locked, gs.silhouette, gs.n_clusters,
        )

        labels, needs_fallback = self._cluster_all(seed, train_by_condition, test_by_condition, locked)

        if needs_fallback:
            logger.warning(
                "[seed=%d] locked params gave <2 clusters for some condition -> "
                "ADR-004 fallback: re-lock from %s with expanded mcs grid %s.",
                seed, gsc, FALLBACK_MIN_CLUSTER_SIZE_GRID,
            )
            gs = self.clusterer.grid_search(
                train_by_condition[gsc], seed=seed,
                min_cluster_size_grid=FALLBACK_MIN_CLUSTER_SIZE_GRID,
            )
            locked = {"min_cluster_size": gs.min_cluster_size, "min_samples": gs.min_samples}
            labels, _ = self._cluster_all(seed, train_by_condition, test_by_condition, locked)

        return {"locked_params": locked, "grid_table": gs.table, "labels": labels}

    def _cluster_all(self, seed, train_by_condition, test_by_condition, locked):
        labels: dict[str, dict[str, np.ndarray]] = {}
        needs_fallback = False
        for cond in train_by_condition:
            tr = self.clusterer.fit(train_by_condition[cond], locked)
            te = self.clusterer.predict(test_by_condition[cond])
            labels[cond] = {"train": tr, "test": te}
            if count_clusters(tr) < 2:
                needs_fallback = True
            logger.info(
                "[seed=%d] %-6s: train k=%d (noise %.1f%%), test k=%d",
                seed, cond, count_clusters(tr), 100 * noise_fraction(tr), count_clusters(te),
            )
        return labels, needs_fallback
