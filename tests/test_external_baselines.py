"""
tests/test_external_baselines.py

Unit tests for src/models/external_baselines.py (M10.5 — HMM + KM-PCA).

Uses REAL M5 feature data (data/processed/{train,test}_features.parquet,
1h columns) — NOT synthetic — per the M10.5 plan. A modest row slice is
used to keep the suite fast while exercising genuine feature values.

Covers IMP-01 v1.3 M10.5 Definition of Done and DS-04 v1.1:
- V-EXP-004 (HMM and KM-PCA under the five-seed protocol; 10 runs)
- ADR-024: PCA n_components is clamped to min(10, n_features)=7 for the
  7-feature/1h input (no ValueError, not 10)
- HMM selects n_components in {2,3,4} by BIC; KM-PCA selects k in
  {2,3,4,5,6} by Silhouette
- Label arrays are int and correctly sized (M11-consumable)
- Determinism given a fixed seed
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.external_baselines import (
    ExternalBaselineRunner,
    HMMBaseline,
    KMeansPCABaseline,
    extract_features,
    feature_columns_for_timeframe,
)
from src.utils.config import load_config
from src.utils.paths import BASE_CONFIG_PATH, DATA_PROCESSED_DIR

pd = pytest.importorskip("pandas")

_TRAIN_PATH = DATA_PROCESSED_DIR / "train_features.parquet"
_TEST_PATH = DATA_PROCESSED_DIR / "test_features.parquet"

# Skip cleanly if the real M5 data hasn't been generated in this env.
pytestmark = pytest.mark.skipif(
    not (_TRAIN_PATH.exists() and _TEST_PATH.exists()),
    reason="real M5 feature parquets not present (run M1-M5 first)",
)

# Row slices: real feature values, small enough for a fast suite.
N_TRAIN, N_TEST = 3000, 1000


@pytest.fixture(scope="module")
def real_features():
    train = pd.read_parquet(_TRAIN_PATH).iloc[:N_TRAIN].reset_index(drop=True)
    test = pd.read_parquet(_TEST_PATH).iloc[:N_TEST].reset_index(drop=True)
    return train, test


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config(BASE_CONFIG_PATH)


# --- feature extraction -----------------------------------------------------


class TestFeatureExtraction:
    def test_1h_columns(self) -> None:
        cols = feature_columns_for_timeframe("1h")
        assert cols == [
            "open_return_1h", "high_return_1h", "low_return_1h",
            "close_return_1h", "volume_zscore_1h", "hl_range_1h", "body_ratio_1h",
        ]

    def test_extract_shape(self, real_features) -> None:
        train, _ = real_features
        X = extract_features(train, "1h")
        assert X.shape == (N_TRAIN, 7)
        assert np.isfinite(X).all()


# --- HMM --------------------------------------------------------------------


class TestHMMBaseline:
    def test_bic_selection_in_grid(self, real_features) -> None:
        train, _ = real_features
        X = extract_features(train, "1h")
        hmm = HMMBaseline(n_components_grid=(2, 3, 4))
        sel = hmm.fit_select(X, seed=42)
        assert sel.n_components_selected in (2, 3, 4)
        assert set(sel.bic_table) == {2, 3, 4}
        # selected model is the lowest-BIC one
        assert sel.best_bic == min(sel.bic_table.values())

    def test_predict_labels(self, real_features) -> None:
        train, test = real_features
        Xtr, Xte = extract_features(train, "1h"), extract_features(test, "1h")
        hmm = HMMBaseline()
        sel = hmm.fit_select(Xtr, seed=42)
        labels = hmm.predict(Xte)
        assert labels.shape == (N_TEST,)
        assert labels.dtype == np.int64
        assert labels.min() >= 0 and labels.max() < sel.n_components_selected

    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            HMMBaseline().predict(np.zeros((5, 7)))


# --- KM-PCA + ADR-024 clamp -------------------------------------------------


class TestKMeansPCABaseline:
    def test_pca_components_clamped_to_7_not_10(self, real_features) -> None:
        """ADR-024: PCA(min(10,7)) = 7 — must NOT raise and must be 7."""
        train, _ = real_features
        X = extract_features(train, "1h")
        kmpca = KMeansPCABaseline(k_grid=(2, 3, 4, 5, 6), pca_components=10)
        sel = kmpca.fit_select(X, seed=42)  # must not raise ValueError
        assert sel.effective_pca_components == 7
        assert kmpca.pca.n_components_ == 7
        # full-rank whitening: explained variance sums to ~1.0
        assert abs(sum(sel.pca_explained_variance_ratio) - 1.0) < 1e-6

    def test_pca10_on_7_features_would_error_without_clamp(self, real_features) -> None:
        """Documents WHY the clamp is needed: raw PCA(10) on 7-dim errors."""
        from sklearn.decomposition import PCA

        train, _ = real_features
        X = extract_features(train, "1h")
        with pytest.raises(ValueError):
            PCA(n_components=10).fit(X)

    def test_silhouette_selection_in_grid(self, real_features) -> None:
        train, _ = real_features
        X = extract_features(train, "1h")
        kmpca = KMeansPCABaseline(k_grid=(2, 3, 4, 5, 6))
        sel = kmpca.fit_select(X, seed=42)
        assert sel.k_selected in (2, 3, 4, 5, 6)
        assert sel.best_silhouette == max(sel.silhouette_table.values())

    def test_predict_labels(self, real_features) -> None:
        train, test = real_features
        Xtr, Xte = extract_features(train, "1h"), extract_features(test, "1h")
        kmpca = KMeansPCABaseline()
        sel = kmpca.fit_select(Xtr, seed=42)
        labels = kmpca.predict(Xte)
        assert labels.shape == (N_TEST,)
        assert labels.dtype == np.int64
        assert set(np.unique(labels)).issubset(set(range(sel.k_selected)))


# --- determinism ------------------------------------------------------------


class TestDeterminism:
    def test_hmm_same_seed_same_labels(self, real_features) -> None:
        train, _ = real_features
        X = extract_features(train, "1h")
        a = HMMBaseline(); a.fit_select(X, seed=123)
        b = HMMBaseline(); b.fit_select(X, seed=123)
        np.testing.assert_array_equal(a.predict(X), b.predict(X))

    def test_kmpca_same_seed_same_labels(self, real_features) -> None:
        train, _ = real_features
        X = extract_features(train, "1h")
        a = KMeansPCABaseline(); a.fit_select(X, seed=123)
        b = KMeansPCABaseline(); b.fit_select(X, seed=123)
        np.testing.assert_array_equal(a.predict(X), b.predict(X))


# --- V-EXP-004: five-seed protocol via the runner --------------------------


class TestExternalBaselineRunner:
    def test_ten_runs_five_seeds(self, real_features, config, tmp_path) -> None:
        train, test = real_features
        runner = ExternalBaselineRunner(config, output_dir=tmp_path / "external_baselines")
        seeds = (42, 123, 456, 789, 1024)
        results = runner.run_all(train, test, seeds=seeds)

        assert set(results) == set(seeds)  # 5 seeds
        for seed in seeds:
            assert results[seed]["hmm"]["n_components"] in (2, 3, 4)
            assert results[seed]["kmpca"]["k"] in (2, 3, 4, 5, 6)
            assert results[seed]["kmpca"]["pca_components"] == 7
            # outputs written for both methods, both splits (M11-consumable)
            for method in ("hmm", "kmpca"):
                base = tmp_path / "external_baselines" / method
                for split, n in (("train", N_TRAIN), ("test", N_TEST)):
                    arr = np.load(base / f"labels_{split}_seed{seed}.npy")
                    assert arr.shape == (n,)
                assert (base / f"model_seed{seed}.pkl").exists()
                assert (base / f"selection_seed{seed}.json").exists()
