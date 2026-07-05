"""
tests/test_hdbscan_clustering.py

Unit tests for src/models/hdbscan_clustering.py (M10 — HDBSCAN, ADR-004).

Uses small synthetic Gaussian blobs so HDBSCAN runs fast while still
exercising: grid search + 2<=k<=8 selection, fit/approximate_predict,
noise (-1) retention, the 1TF-locked-params protocol (V-EXP-001), the
DS-02 Stage 8 parquet schema, and the no-eligible-config error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from src.models.hdbscan_clustering import (
    ClusteringError,
    ClusteringPipeline,
    HDBSCANClusterer,
    build_cluster_parquet,
    count_clusters,
    noise_fraction,
    silhouette_noise_excluded,
)


@pytest.fixture(scope="module")
def blobs():
    X, _ = make_blobs(n_samples=600, centers=4, n_features=8,
                      cluster_std=0.40, random_state=0)
    return X.astype(np.float64)


@pytest.fixture(scope="module")
def blobs_test():
    X, _ = make_blobs(n_samples=150, centers=4, n_features=8,
                      cluster_std=0.40, random_state=1)
    return X.astype(np.float64)


def _clusterer():
    return HDBSCANClusterer(min_cluster_size_grid=(15, 30),
                            min_samples_grid=(5, 10), max_clusters=8)


# --- helpers ----------------------------------------------------------------


class TestHelpers:
    def test_count_clusters_excludes_noise(self) -> None:
        assert count_clusters(np.array([-1, 0, 0, 1, 2, -1])) == 3

    def test_noise_fraction(self) -> None:
        assert noise_fraction(np.array([-1, -1, 0, 0])) == 0.5

    def test_silhouette_neg_inf_when_single_cluster(self, blobs) -> None:
        labels = np.zeros(len(blobs), dtype=int)  # 1 cluster -> undefined
        assert silhouette_noise_excluded(blobs, labels) == float("-inf")


# --- grid search ------------------------------------------------------------


class TestGridSearch:
    def test_selects_config_with_valid_k(self, blobs) -> None:
        res = _clusterer().grid_search(blobs, seed=0)
        assert 2 <= res.n_clusters <= 8
        assert res.min_cluster_size in (15, 30)
        assert res.min_samples in (5, 10)
        assert len(res.table) == 4  # 2 x 2 grid, all recorded
        # selected silhouette is the max among eligible configs
        elig = [r["silhouette"] for r in res.table if r["eligible"] and r["silhouette"] is not None]
        assert res.silhouette == max(elig)

    def test_raises_when_no_eligible_config(self, blobs) -> None:
        # min_cluster_size far too large -> 0/1 clusters everywhere
        clu = HDBSCANClusterer(min_cluster_size_grid=(500,), min_samples_grid=(5,))
        with pytest.raises(ClusteringError):
            clu.grid_search(blobs, seed=0)

    def test_tiebreak_prefers_lower_params(self) -> None:
        # Force a silhouette tie by monkeypatching: two eligible configs,
        # equal silhouette -> lower mcs (then lower ms) must win.
        clu = HDBSCANClusterer(min_cluster_size_grid=(15, 30), min_samples_grid=(5,))
        import src.models.hdbscan_clustering as m
        orig = m.silhouette_noise_excluded
        m.silhouette_noise_excluded = lambda *a, **k: 0.5  # constant tie
        try:
            X, _ = make_blobs(n_samples=400, centers=3, n_features=6,
                              cluster_std=0.5, random_state=2)
            res = clu.grid_search(X.astype(np.float64), seed=0)
            assert res.min_cluster_size == 15  # lower mcs wins the tie
        finally:
            m.silhouette_noise_excluded = orig


# --- fit / predict ----------------------------------------------------------


class TestFitPredict:
    def test_fit_returns_labels_with_noise_retained(self, blobs) -> None:
        clu = _clusterer()
        params = {"min_cluster_size": 15, "min_samples": 5}
        labels = clu.fit(blobs, params)
        assert labels.shape == (len(blobs),)
        assert labels.dtype == np.int32
        assert set(np.unique(labels)) - {-1}  # at least one real cluster
        # -1 is a legal, retained label (not remapped)
        assert labels.min() >= -1

    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            _clusterer().predict(np.zeros((5, 8)))

    def test_approximate_predict_shapes(self, blobs, blobs_test) -> None:
        clu = _clusterer()
        clu.fit(blobs, {"min_cluster_size": 15, "min_samples": 5})
        te = clu.predict(blobs_test)
        assert te.shape == (len(blobs_test),)
        assert te.dtype == np.int32


# --- Stage-1 pipeline (V-EXP-001: identical params across conditions) --------


class TestClusteringPipeline:
    def test_locked_params_applied_to_all_conditions(self, blobs, blobs_test) -> None:
        conds = ["1TF", "2TF", "3TF", "4TF"]
        train = {c: blobs for c in conds}
        test = {c: blobs_test for c in conds}
        pipe = ClusteringPipeline(_clusterer(), grid_search_condition="1TF")
        out = pipe.run_stage1(seed=42, train_by_condition=train, test_by_condition=test)
        assert set(out["locked_params"]) == {"min_cluster_size", "min_samples"}
        assert set(out["labels"]) == set(conds)
        for c in conds:
            assert out["labels"][c]["train"].shape == (len(blobs),)
            assert out["labels"][c]["test"].shape == (len(blobs_test),)

    def test_missing_grid_condition_raises(self, blobs) -> None:
        pipe = ClusteringPipeline(_clusterer(), grid_search_condition="1TF")
        with pytest.raises(ClusteringError):
            pipe.run_stage1(seed=0, train_by_condition={"2TF": blobs}, test_by_condition={"2TF": blobs})


# --- DS-02 Stage 8 parquet schema -------------------------------------------


class TestClusterParquet:
    def test_schema_and_dtypes(self) -> None:
        n = 5
        ts = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
        ts_ns = ts.asi8  # int64 Unix-ns (as M6 stores anchor timestamps)
        ohlcv = pd.DataFrame({
            "timestamp": ts,
            "open_1h": np.arange(n, dtype="float64"),
            "high_1h": np.arange(n, dtype="float64") + 1,
            "low_1h": np.arange(n, dtype="float64") - 1,
            "close_1h": np.arange(n, dtype="float64") + 0.5,
            "volume_1h": np.arange(n, dtype="float64") * 10,
            "close_return_1h": np.linspace(-0.01, 0.01, n),
        })
        labels = np.array([0, 1, -1, 0, 2], dtype=np.int32)
        df = build_cluster_parquet(labels, ts_ns, ohlcv, condition="2TF", seed=42)

        assert list(df.columns) == [
            "timestamp", "open_1h", "high_1h", "low_1h", "close_1h", "volume_1h",
            "close_return_1h", "cluster_label", "is_noise", "condition", "seed",
        ]
        assert df["cluster_label"].dtype == np.int32
        assert df["is_noise"].tolist() == [False, False, True, False, False]
        assert (df["condition"] == "2TF").all()
        assert df["seed"].dtype == np.int32
        # OHLCV joined correctly by timestamp
        assert df["open_1h"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
