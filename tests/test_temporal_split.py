"""
tests/test_temporal_split.py

Unit tests for src/data/temporal_split.py (M5 — Temporal Split).

Covers DS-04 v1.1:
- V-DATA-005 (split boundary exact, no overlap)
- V-LEAK-002 (no future price information crosses the split boundary,
  documented per DS-02 v1.1 LC-3)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.temporal_split import (
    EXPECTED_TEST_ROWS_APPROX,
    EXPECTED_TRAIN_ROWS_APPROX,
    TEST_START,
    TRAIN_END,
    SplitResult,
    TemporalSplitError,
    TemporalSplitter,
    check_no_overlap,
    check_split_boundary,
    check_split_sizes,
    run_temporal_split,
    verify_lc3,
)


# --- Fixtures ----------------------------------------------------------------

def _make_features_df(start: str, periods: int) -> pd.DataFrame:
    """Build a minimal feature-matrix-shaped DataFrame for split testing."""
    timestamps = pd.date_range(start, periods=periods, freq="1h", tz="UTC")
    df = pd.DataFrame({"timestamp": timestamps})
    df["volume_zscore_1h"] = np.random.RandomState(0).randn(periods)
    df["close_return_1h"] = np.random.RandomState(1).randn(periods) * 0.01
    return df


@pytest.fixture
def splitter() -> TemporalSplitter:
    return TemporalSplitter()


@pytest.fixture
def boundary_straddling_df() -> pd.DataFrame:
    """
    A small DataFrame whose timestamps straddle the exact ADR-014
    boundary (2022-12-31 23:00 -> 2023-01-01 00:00), for precise
    boundary testing.
    """
    timestamps = pd.date_range("2022-12-31 20:00", periods=8, freq="1h", tz="UTC")
    # This gives: 2022-12-31 20,21,22,23:00, 2023-01-01 00,01,02,03:00
    df = pd.DataFrame({"timestamp": timestamps})
    df["volume_zscore_1h"] = np.arange(8, dtype="float64")
    df["close_return_1h"] = np.arange(8, dtype="float64") * 0.01
    return df


# --- TemporalSplitter.split -----------------------------------------------------

class TestSplit:
    def test_train_contains_only_timestamps_at_or_before_boundary(
        self, splitter, boundary_straddling_df
    ) -> None:
        result = splitter.split(boundary_straddling_df)
        assert (result.train["timestamp"] <= TRAIN_END).all()

    def test_test_contains_only_timestamps_at_or_after_boundary(
        self, splitter, boundary_straddling_df
    ) -> None:
        result = splitter.split(boundary_straddling_df)
        assert (result.test["timestamp"] >= TEST_START).all()

    def test_every_row_assigned_to_exactly_one_split(
        self, splitter, boundary_straddling_df
    ) -> None:
        result = splitter.split(boundary_straddling_df)
        assert len(result.train) + len(result.test) == len(boundary_straddling_df)

    def test_exact_boundary_row_goes_to_train(self, splitter, boundary_straddling_df) -> None:
        result = splitter.split(boundary_straddling_df)
        assert TRAIN_END in set(result.train["timestamp"])
        assert TRAIN_END not in set(result.test["timestamp"])

    def test_exact_test_start_row_goes_to_test(self, splitter, boundary_straddling_df) -> None:
        result = splitter.split(boundary_straddling_df)
        assert TEST_START in set(result.test["timestamp"])
        assert TEST_START not in set(result.train["timestamp"])

    def test_returns_split_result_instance(self, splitter, boundary_straddling_df) -> None:
        result = splitter.split(boundary_straddling_df)
        assert isinstance(result, SplitResult)


# --- check_split_boundary (V-DATA-005) -----------------------------------------

class TestCheckSplitBoundary:
    def test_correct_boundary_passes(self, splitter, boundary_straddling_df) -> None:
        result = splitter.split(boundary_straddling_df)
        passed, detail = check_split_boundary(result)
        assert passed

    def test_train_max_matches_exactly(self, splitter, boundary_straddling_df) -> None:
        result = splitter.split(boundary_straddling_df)
        assert result.train["timestamp"].max() == TRAIN_END

    def test_test_min_matches_exactly(self, splitter, boundary_straddling_df) -> None:
        result = splitter.split(boundary_straddling_df)
        assert result.test["timestamp"].min() == TEST_START

    def test_empty_train_fails(self) -> None:
        empty_train = pd.DataFrame({"timestamp": pd.Series([], dtype="datetime64[ns, UTC]")})
        test_only = _make_features_df("2023-01-01", 10)
        result = SplitResult(train=empty_train, test=test_only)
        passed, detail = check_split_boundary(result)
        assert not passed
        assert "empty" in detail

    def test_empty_test_fails(self) -> None:
        train_only = _make_features_df("2020-01-01", 10)
        empty_test = pd.DataFrame({"timestamp": pd.Series([], dtype="datetime64[ns, UTC]")})
        result = SplitResult(train=train_only, test=empty_test)
        passed, detail = check_split_boundary(result)
        assert not passed
        assert "empty" in detail

    def test_wrong_train_max_fails(self) -> None:
        # Train that stops early (doesn't reach the boundary).
        train = _make_features_df("2022-12-01", 24)  # ends well before boundary
        test = _make_features_df("2023-01-01", 10)
        result = SplitResult(train=train, test=test)
        passed, detail = check_split_boundary(result)
        assert not passed
        assert "train max timestamp" in detail


# --- check_no_overlap (V-DATA-005) ---------------------------------------------

class TestCheckNoOverlap:
    def test_no_overlap_passes(self, splitter, boundary_straddling_df) -> None:
        result = splitter.split(boundary_straddling_df)
        passed, detail = check_no_overlap(result)
        assert passed

    def test_overlap_detected(self) -> None:
        shared_ts = pd.date_range("2022-12-31 22:00", periods=3, freq="1h", tz="UTC")
        train = pd.DataFrame({"timestamp": shared_ts})
        train["volume_zscore_1h"] = 0.0
        train["close_return_1h"] = 0.0
        test = pd.DataFrame({"timestamp": shared_ts})  # deliberately duplicated
        test["volume_zscore_1h"] = 0.0
        test["close_return_1h"] = 0.0
        result = SplitResult(train=train, test=test)
        passed, detail = check_no_overlap(result)
        assert not passed
        assert "duplicate" in detail


# --- check_split_sizes ----------------------------------------------------------

class TestCheckSplitSizes:
    def test_sizes_within_tolerance_pass(self) -> None:
        train = _make_features_df("2020-01-01", EXPECTED_TRAIN_ROWS_APPROX)
        test = _make_features_df("2023-01-01", EXPECTED_TEST_ROWS_APPROX)
        result = SplitResult(train=train, test=test)
        passed, detail = check_split_sizes(result)
        assert passed

    def test_sizes_far_off_fail(self) -> None:
        train = _make_features_df("2020-01-01", 100)  # way too small
        test = _make_features_df("2023-01-01", 100)
        result = SplitResult(train=train, test=test)
        passed, detail = check_split_sizes(result)
        assert not passed

    def test_none_skips_check(self) -> None:
        train = _make_features_df("2020-01-01", 100)
        test = _make_features_df("2023-01-01", 100)
        result = SplitResult(train=train, test=test)
        passed, detail = check_split_sizes(
            result, expected_train_rows=None, expected_test_rows=None
        )
        assert passed


# --- verify_lc3 (V-LEAK-002) -----------------------------------------------------

class TestVerifyLc3:
    def test_correct_split_passes(self, splitter, boundary_straddling_df) -> None:
        result = splitter.split(boundary_straddling_df)
        passed, detail = verify_lc3(result.train, result.test)
        assert passed

    def test_includes_informational_note_about_rolling_window(
        self, splitter, boundary_straddling_df
    ) -> None:
        result = splitter.split(boundary_straddling_df)
        passed, detail = verify_lc3(result.train, result.test)
        assert passed
        assert "Informational" in detail or "informational" in detail.lower()

    def test_detects_train_row_after_boundary(self) -> None:
        # Manually construct a "corrupted" split where a row leaked
        # into train past the boundary.
        bad_train = pd.DataFrame(
            {
                "timestamp": [TRAIN_END, TEST_START],  # TEST_START should not be here
                "volume_zscore_1h": [0.0, 0.0],
                "close_return_1h": [0.0, 0.0],
            }
        )
        good_test = pd.DataFrame(
            {
                "timestamp": [TEST_START],
                "volume_zscore_1h": [0.0],
                "close_return_1h": [0.0],
            }
        )
        passed, detail = verify_lc3(bad_train, good_test)
        assert not passed
        assert "after TRAIN_END" in detail

    def test_detects_test_row_before_boundary(self) -> None:
        good_train = pd.DataFrame(
            {
                "timestamp": [TRAIN_END],
                "volume_zscore_1h": [0.0],
                "close_return_1h": [0.0],
            }
        )
        bad_test = pd.DataFrame(
            {
                "timestamp": [TRAIN_END, TEST_START],  # TRAIN_END should not be here
                "volume_zscore_1h": [0.0, 0.0],
                "close_return_1h": [0.0, 0.0],
            }
        )
        passed, detail = verify_lc3(good_train, bad_test)
        assert not passed
        assert "before TEST_START" in detail

    def test_missing_volume_zscore_column_does_not_crash(self) -> None:
        train = pd.DataFrame({"timestamp": [TRAIN_END]})
        test = pd.DataFrame({"timestamp": [TEST_START]})
        # Should not raise even without volume_zscore_1h present.
        passed, detail = verify_lc3(train, test)
        assert passed


# --- run_temporal_split (end-to-end M5 orchestration) --------------------------

class TestRunTemporalSplit:
    def test_full_size_dataset_matches_ds02_expectations(self) -> None:
        """
        Build a feature matrix matching M4's real output shape
        (35,045 rows starting 2020-01-01 19:00:00 UTC, per the
        corrected DS-02 v1.1 / DS-04 v1.1 design) and confirm the
        split produces train/test sizes matching DS-02 v1.1 Stage 4.
        """
        df = _make_features_df("2020-01-01 19:00:00", 35_045)
        result = run_temporal_split(df, raise_on_failure=True)

        assert result.train["timestamp"].max() == TRAIN_END
        assert result.test["timestamp"].min() == TEST_START
        assert abs(len(result.train) - EXPECTED_TRAIN_ROWS_APPROX) < 50
        assert abs(len(result.test) - EXPECTED_TEST_ROWS_APPROX) < 50

        # Cross-check: train + test should account for all rows in
        # range (no rows silently dropped).
        assert len(result.train) + len(result.test) == len(df)

    def test_raises_on_empty_input(self) -> None:
        empty_df = pd.DataFrame({"timestamp": pd.Series([], dtype="datetime64[ns, UTC]")})
        with pytest.raises(TemporalSplitError):
            run_temporal_split(empty_df, raise_on_failure=True)

    def test_does_not_raise_when_size_check_disabled(self) -> None:
        # Small dataset, but disable size checks so only
        # boundary/overlap/lc3 matter.
        df = _make_features_df("2022-12-31 20:00", 8)
        result = run_temporal_split(
            df,
            raise_on_failure=True,
            expected_train_rows=None,
            expected_test_rows=None,
        )
        assert len(result.train) > 0
        assert len(result.test) > 0

    def test_no_shuffling_train_is_sorted(self) -> None:
        """DS-02 v1.1: 'No shuffling. No stratification. No overlap.'"""
        df = _make_features_df("2020-01-01 19:00:00", 1000)
        result = run_temporal_split(
            df, raise_on_failure=True, expected_train_rows=None, expected_test_rows=None
        )
        assert result.train["timestamp"].is_monotonic_increasing
