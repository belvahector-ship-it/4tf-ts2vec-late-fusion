"""
src/models/external_baselines.py

External comparison baselines (M10.5 — HMM + KM-PCA, DS-03 §4, ADR-024).

Purpose
-------
Two classical, non-TS2Vec baselines run under the identical five-seed
protocol (ADR-019) so they can be compared against the TS2Vec conditions
on equal footing (V-EXP-004):

  - **HMM** — a Gaussian Hidden Markov Model over the 1h feature
    sequence; the number of hidden states `n_components ∈ {2,3,4}` is
    selected by lowest BIC.
  - **KM-PCA** — PCA followed by K-Means; the number of clusters
    `k ∈ {2,3,4,5,6}` is selected by highest Silhouette Score. Per
    ADR-024, PCA `n_components` is clamped to
    `min(configured, n_features)` — for the 7-feature/1h input this is
    **7**, a full-rank whitening (NOT dimensionality reduction), which
    keeps KM-PCA's information access identical to the 1TF/BL-1h
    condition it is benchmarked against.

Both operate ONLY on the 7 OHLCV-derived features at 1h resolution
(the same input as 1TF/BL-1h). This module is fully independent of
TS2Vec: it never imports `TS2VecBranch`, `FusionModule`, or
`HDBSCANClusterer`, and depends only on M5's feature split (it does NOT
need M6 windows, M7/M8 checkpoints, or M9 fusion).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from src.data.feature_engineering import FEATURE_NAMES
from src.utils.logging_utils import get_logger
from src.utils.paths import RANDOM_SEEDS
from src.utils.seed import set_all_seeds

logger = get_logger(__name__)

# hmmlearn GaussianHMM emission covariance. "diag" is the library default:
# robust and standard for financial regime detection (fewer parameters ->
# more stable EM on real data); the spec does not constrain this, so the
# conservative default is used and documented here.
HMM_COVARIANCE_TYPE: str = "diag"
HMM_MAX_ITER: int = 100
KMEANS_N_INIT: int = 10


def feature_columns_for_timeframe(timeframe: str = "1h") -> list[str]:
    """
    Return the 7 OHLCV-derived feature column names for a timeframe.

    e.g. for "1h": open_return_1h, high_return_1h, ..., body_ratio_1h.
    """
    return [f"{name}_{timeframe}" for name in FEATURE_NAMES]


def extract_features(
    df: pd.DataFrame, timeframe: str = "1h"
) -> np.ndarray:
    """
    Extract the timeframe's 7-feature matrix from an M5 features frame.

    Parameters
    ----------
    df : pd.DataFrame
        `train_features.parquet` / `test_features.parquet` (M5 output).
    timeframe : str, optional
        Feature resolution to use, default "1h" (DS-03 §4).

    Returns
    -------
    np.ndarray
        Shape `[N, 7]`, dtype float64, rows in the DataFrame's (temporal)
        order — HMM relies on this ordering being chronological.
    """
    cols = feature_columns_for_timeframe(timeframe)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"features frame is missing columns: {missing}")
    return df[cols].to_numpy(dtype=np.float64)


# --- HMM --------------------------------------------------------------------


@dataclass
class HMMSelection:
    """Result of HMM model selection over `n_components`."""

    n_components_selected: int
    best_bic: float
    bic_table: dict[int, float]  # n_components -> BIC


class HMMBaseline:
    """
    Gaussian HMM baseline with BIC-based state-count selection.

    Parameters
    ----------
    n_components_grid : sequence of int, optional
        Candidate hidden-state counts, default [2, 3, 4] (DS-03 §4).
    covariance_type, n_iter : see module constants.
    """

    def __init__(
        self,
        n_components_grid: Sequence[int] = (2, 3, 4),
        covariance_type: str = HMM_COVARIANCE_TYPE,
        n_iter: int = HMM_MAX_ITER,
    ) -> None:
        self.n_components_grid = list(n_components_grid)
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.model: GaussianHMM | None = None
        self.selection: HMMSelection | None = None

    def fit_select(self, features: np.ndarray, seed: int) -> HMMSelection:
        """
        Fit one HMM per `n_components` and keep the lowest-BIC model.

        Parameters
        ----------
        features : np.ndarray
            `[N, 7]` chronologically-ordered 1h features (single sequence).
        seed : int
            Seed applied to all RNGs and to `GaussianHMM.random_state`.

        Returns
        -------
        HMMSelection
        """
        set_all_seeds(seed)
        bic_table: dict[int, float] = {}
        best_model: GaussianHMM | None = None
        best_bic = np.inf
        best_n = self.n_components_grid[0]

        for n in self.n_components_grid:
            model = GaussianHMM(
                n_components=n,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=seed,
            )
            model.fit(features)  # single sequence -> no `lengths` needed
            bic = float(model.bic(features))
            bic_table[n] = bic
            logger.info("[HMM seed=%d] n_components=%d -> BIC=%.2f", seed, n, bic)
            if bic < best_bic:
                best_bic, best_model, best_n = bic, model, n

        self.model = best_model
        self.selection = HMMSelection(
            n_components_selected=best_n, best_bic=best_bic, bic_table=bic_table
        )
        logger.info(
            "[HMM seed=%d] selected n_components=%d (BIC=%.2f)", seed, best_n, best_bic
        )
        return self.selection

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return the Viterbi state label per row (`[N]`, int)."""
        if self.model is None:
            raise RuntimeError("HMMBaseline: call fit_select before predict.")
        return self.model.predict(features).astype(np.int64)


# --- KM-PCA -----------------------------------------------------------------


@dataclass
class KMPCASelection:
    """Result of KM-PCA model selection over `k`."""

    k_selected: int
    best_silhouette: float
    silhouette_table: dict[int, float]  # k -> silhouette
    effective_pca_components: int
    pca_explained_variance_ratio: list[float] = field(default_factory=list)


class KMeansPCABaseline:
    """
    PCA + K-Means baseline with Silhouette-based cluster-count selection.

    Per ADR-024, PCA `n_components` is clamped to
    `min(pca_components, n_features)`. For the 7-feature/1h input this is
    7 — full-rank whitening, not dimensionality reduction.

    Parameters
    ----------
    k_grid : sequence of int, optional
        Candidate cluster counts, default [2, 3, 4, 5, 6] (DS-03 §4).
    pca_components : int, optional
        Ceiling on PCA components (default 10, per base.yaml). Clamped to
        the available feature dimension at fit time (ADR-024).
    """

    def __init__(
        self,
        k_grid: Sequence[int] = (2, 3, 4, 5, 6),
        pca_components: int = 10,
    ) -> None:
        self.k_grid = list(k_grid)
        self.pca_components_ceiling = int(pca_components)
        self.pca: PCA | None = None
        self.kmeans: KMeans | None = None
        self.selection: KMPCASelection | None = None

    def fit_select(self, features: np.ndarray, seed: int) -> KMPCASelection:
        """
        Fit PCA (clamped), then K-Means per `k`; keep the best Silhouette.

        Parameters
        ----------
        features : np.ndarray
            `[N, 7]` 1h features.
        seed : int
            Seed applied to all RNGs and to PCA/KMeans `random_state`.

        Returns
        -------
        KMPCASelection
        """
        set_all_seeds(seed)
        n_features = features.shape[1]
        # ADR-024: clamp the configured ceiling to the available dimension.
        effective = min(self.pca_components_ceiling, n_features)
        self.pca = PCA(n_components=effective, random_state=seed)
        z = self.pca.fit_transform(features)

        sil_table: dict[int, float] = {}
        best_km: KMeans | None = None
        best_sil = -np.inf
        best_k = self.k_grid[0]

        for k in self.k_grid:
            km = KMeans(n_clusters=k, random_state=seed, n_init=KMEANS_N_INIT)
            labels = km.fit_predict(z)
            sil = float(silhouette_score(z, labels))
            sil_table[k] = sil
            logger.info("[KM-PCA seed=%d] k=%d -> silhouette=%.4f", seed, k, sil)
            if sil > best_sil:
                best_sil, best_km, best_k = sil, km, k

        self.kmeans = best_km
        self.selection = KMPCASelection(
            k_selected=best_k,
            best_silhouette=best_sil,
            silhouette_table=sil_table,
            effective_pca_components=effective,
            pca_explained_variance_ratio=self.pca.explained_variance_ratio_.tolist(),
        )
        logger.info(
            "[KM-PCA seed=%d] selected k=%d (silhouette=%.4f); PCA components=%d "
            "(ceiling=%d, clamped per ADR-024)",
            seed, best_k, best_sil, effective, self.pca_components_ceiling,
        )
        return self.selection

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return the cluster label per row (`[N]`, int)."""
        if self.pca is None or self.kmeans is None:
            raise RuntimeError("KMeansPCABaseline: call fit_select before predict.")
        return self.kmeans.predict(self.pca.transform(features)).astype(np.int64)


# --- Orchestration ----------------------------------------------------------


class ExternalBaselineRunner:
    """
    Run HMM and KM-PCA across all seeds and write M11-consumable outputs.

    Writes, per method and seed, under
    ``{output_dir}/{hmm,kmpca}/``:
      - ``labels_train_seed{seed}.npy`` / ``labels_test_seed{seed}.npy``
      - ``model_seed{seed}.pkl``
      - ``selection_seed{seed}.json`` (BIC table / silhouette table +
        chosen hyperparameter — audit trail)

    Parameters
    ----------
    config : dict
        Base config; reads `external_baselines.*` grids and
        `input_timeframe`.
    output_dir : Path
        Root for the ``hmm/`` and ``kmpca/`` output folders.
    """

    def __init__(self, config: dict[str, Any], output_dir: Path) -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        ext = config.get("external_baselines", {})
        self.timeframe = ext.get("hmm", {}).get("input_timeframe", "1h")
        self.hmm_grid = list(ext.get("hmm", {}).get("n_components_grid", [2, 3, 4]))
        self.k_grid = list(ext.get("km_pca", {}).get("k_grid", [2, 3, 4, 5, 6]))
        self.pca_ceiling = int(ext.get("km_pca", {}).get("pca_components", 10))

    def run_seed(
        self,
        seed: int,
        train_features: np.ndarray,
        test_features: np.ndarray,
    ) -> dict[str, Any]:
        """Run both methods for one seed; write outputs; return a summary."""
        import json

        summary: dict[str, Any] = {"seed": seed}

        # HMM
        hmm = HMMBaseline(n_components_grid=self.hmm_grid)
        hmm_sel = hmm.fit_select(train_features, seed)
        hmm_dir = self.output_dir / "hmm"
        hmm_dir.mkdir(parents=True, exist_ok=True)
        np.save(hmm_dir / f"labels_train_seed{seed}.npy", hmm.predict(train_features))
        np.save(hmm_dir / f"labels_test_seed{seed}.npy", hmm.predict(test_features))
        with open(hmm_dir / f"model_seed{seed}.pkl", "wb") as f:
            pickle.dump(hmm.model, f)
        with open(hmm_dir / f"selection_seed{seed}.json", "w", encoding="utf-8") as f:
            json.dump(
                {"n_components_selected": hmm_sel.n_components_selected,
                 "best_bic": hmm_sel.best_bic, "bic_table": hmm_sel.bic_table}, f, indent=2)
        summary["hmm"] = {"n_components": hmm_sel.n_components_selected}

        # KM-PCA
        kmpca = KMeansPCABaseline(k_grid=self.k_grid, pca_components=self.pca_ceiling)
        km_sel = kmpca.fit_select(train_features, seed)
        km_dir = self.output_dir / "kmpca"
        km_dir.mkdir(parents=True, exist_ok=True)
        np.save(km_dir / f"labels_train_seed{seed}.npy", kmpca.predict(train_features))
        np.save(km_dir / f"labels_test_seed{seed}.npy", kmpca.predict(test_features))
        with open(km_dir / f"model_seed{seed}.pkl", "wb") as f:
            pickle.dump({"pca": kmpca.pca, "kmeans": kmpca.kmeans}, f)
        with open(km_dir / f"selection_seed{seed}.json", "w", encoding="utf-8") as f:
            json.dump(
                {"k_selected": km_sel.k_selected, "best_silhouette": km_sel.best_silhouette,
                 "silhouette_table": km_sel.silhouette_table,
                 "effective_pca_components": km_sel.effective_pca_components,
                 "pca_explained_variance_ratio": km_sel.pca_explained_variance_ratio}, f, indent=2)
        summary["kmpca"] = {"k": km_sel.k_selected,
                            "pca_components": km_sel.effective_pca_components}

        logger.info("[seed=%d] external baselines done: %s", seed, summary)
        return summary

    def run_all(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        seeds: Sequence[int] = RANDOM_SEEDS,
    ) -> dict[int, dict[str, Any]]:
        """
        Run both baselines for every seed (5 HMM + 5 KM-PCA = 10 runs).

        Parameters
        ----------
        train_df, test_df : pd.DataFrame
            M5 `train_features` / `test_features` frames.
        seeds : sequence of int, optional
            Default the five-seed protocol.

        Returns
        -------
        dict[int, dict]
            seed -> per-method summary.
        """
        train_features = extract_features(train_df, self.timeframe)
        test_features = extract_features(test_df, self.timeframe)
        results: dict[int, dict[str, Any]] = {}
        for seed in seeds:
            results[seed] = self.run_seed(seed, train_features, test_features)
        logger.info(
            "ExternalBaselineRunner complete: %d seeds x 2 methods = %d runs.",
            len(seeds), len(seeds) * 2,
        )
        return results
