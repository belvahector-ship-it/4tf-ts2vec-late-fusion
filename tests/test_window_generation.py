"""
tests/test_window_generation.py

Unit tests for src/data/window_generation.py (M6 — Window Generation).

Design: DS-02 v1.2 "window-then-split-by-anchor" (Scenario C, adopted
after AUDIT_LC4_ADDENDUM.md resolved an internal contradiction in
DS-02 v1.1). Windows are generated over the FULL feature matrix and
categorized as train/test by anchor timestamp, not generated separately
from pre-split files.

This is the final gate before branch training (M8). Covers DS-04 v1.1:
- V-LEAK-003 (train windows stay within train period; documented ≤47
  test-window boundary overlap matches expectation, not zero)
- V-LEAK-004 (per-window z-score normalization uses only that window's
  own values, no global statistics)

Like tests/test_alignment.py's verify_no_lookahead tests, the most
important tests here are the ones that DELIBERATELY INJECT a leakage
bug and confirm the detector catches it — not just that clean data passes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.window_generation import (
    EXPECTED_MAX_BOUNDARY_OVERLAP_WINDOWS,
    EXPECTED_N_TEST_WINDOWS,
    NORMALIZATION_EPSILON,
    STRIDE,
    TEST_START,
    TRAIN_END,
    WINDOW_SIZE,
    WindowGenerationError,
    WindowGenerator,
    WindowSet,
    check_window_count,
    check_window_shape,
    count_boundary_overlap_windows,
    run_window_generation,
    verify_per_window_normalization,
    verify_test_windows_anchored_in_test_period,
    verify_train_windows_stay_in_train_period,
)


# --- Fixtures ----------------------------------------------------------------

def _make_full_feature_df(
    start: str, periods: int, timeframes: tuple[str, ...] = ("15m", "1h", "4h", "1d"), seed: int = 42
) -> pd.DataFrame:
    """Build a full (unsplit) feature matrix spanning `periods` hours."""
    timestamps = pd.date_range(start, periods=periods, freq="1h", tz="UTC")
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({"timestamp": timestamps})
    for tf in timeframes:
        for feat in (
            "open_return", "high_return", "low_return", "close_return",
            "volume_zscore", "hl_range", "body_ratio",
        ):
            df[f"{feat}_{tf}"] = rng.randn(periods)
    return df


@pytest.fixture
def generator() -> WindowGenerator:
    return WindowGenerator()


@pytest.fixture
def boundary_straddling_features() -> pd.DataFrame:
    """
    A feature matrix straddling the ADR-014 boundary, large enough
    (before + after) to produce several boundary-overlap test windows
    for precise LC-4 testing.
    """
    return _make_full_feature_df("2022-12-27 00:00:00", 200, ("1h",))


# --- extract_windows (over the full matrix, no pre-split) ------------------------

class TestExtractWindows:
    def test_correct_number_of_windows(self, generator) -> None:
        df = _make_full_feature_df("2020-01-01 19:00:00", 100, ("1h",))
        result = generator.extract_windows(df, "1h")
        expected_n = (100 - WINDOW_SIZE) // STRIDE + 1
        assert len(result.windows) == expected_n

    def test_correct_shape(self, generator) -> None:
        df = _make_full_feature_df("2020-01-01 19:00:00", 100, ("1h",))
        result = generator.extract_windows(df, "1h")
        assert result.windows.shape[1:] == (WINDOW_SIZE, 7)

    def test_first_window_matches_first_48_rows(self, generator) -> None:
        df = _make_full_feature_df("2020-01-01 19:00:00", 100, ("1h",))
        result = generator.extract_windows(df, "1h")
        cols = [
            f"{f}_1h"
            for f in (
                "open_return", "high_return", "low_return", "close_return",
                "volume_zscore", "hl_range", "body_ratio",
            )
        ]
        expected_first_window = df[cols].iloc[0:48].to_numpy(dtype="float32")
        np.testing.assert_array_equal(result.windows[0], expected_first_window)

    def test_anchor_timestamp_is_last_row_of_window(self, generator) -> None:
        df = _make_full_feature_df("2020-01-01 19:00:00", 100, ("1h",))
        result = generator.extract_windows(df, "1h")
        expected_anchor_ts = pd.Timestamp(df["timestamp"].iloc[47])
        actual_anchor_ts = pd.Timestamp(result.anchor_timestamps[0], tz="UTC")
        assert actual_anchor_ts == expected_anchor_ts

    def test_earliest_timestamp_is_first_row_of_window(self, generator) -> None:
        df = _make_full_feature_df("2020-01-01 19:00:00", 100, ("1h",))
        result = generator.extract_windows(df, "1h")
        expected_earliest_ts = pd.Timestamp(df["timestamp"].iloc[0])
        actual_earliest_ts = pd.Timestamp(result.earliest_timestamps[0], tz="UTC")
        assert actual_earliest_ts == expected_earliest_ts

    def test_too_few_rows_raises(self, generator) -> None:
        tiny_df = _make_full_feature_df("2020-01-01", 10, ("1h",))
        with pytest.raises(ValueError, match="Cannot extract windows"):
            generator.extract_windows(tiny_df, "1h")


# --- normalization (V-LEAK-004) -------------------------------------------------

class TestNormalization:
    def test_matches_manual_computation(self, generator) -> None:
        rng = np.random.RandomState(1)
        window = rng.randn(48, 7)
        normalized = generator.normalize_window(window)
        manual_mu = window.mean(axis=0)
        manual_sig = window.std(axis=0)
        manual_normalized = (window - manual_mu) / (manual_sig + NORMALIZATION_EPSILON)
        np.testing.assert_allclose(normalized, manual_normalized, rtol=1e-6)

    def test_zero_variance_window_does_not_produce_nan_or_inf(self, generator) -> None:
        flat_window = np.full((48, 7), 3.0, dtype="float32")
        normalized = generator.normalize_window(flat_window)
        assert np.isfinite(normalized).all()

    def test_vectorized_matches_per_window_loop(self, generator) -> None:
        rng = np.random.RandomState(2)
        windows = rng.randn(10, 48, 7).astype("float32")
        vectorized = generator.normalize_all_windows(windows)
        looped = np.stack([generator.normalize_window(w) for w in windows]).astype("float32")
        np.testing.assert_allclose(vectorized, looped, rtol=1e-5, atol=1e-6)

    def test_epsilon_matches_adr_016(self) -> None:
        assert NORMALIZATION_EPSILON == 1e-8


class TestVerifyPerWindowNormalization:
    def test_correct_normalization_passes(self, generator) -> None:
        rng = np.random.RandomState(3)
        raw = rng.randn(48, 7)
        normalized = generator.normalize_window(raw)
        passed, detail = verify_per_window_normalization(raw, normalized)
        assert passed

    def test_detects_global_statistics_used_instead_of_window_local(self, generator) -> None:
        """
        CRITICAL: simulate the exact leakage bug V-LEAK-004 exists to
        catch — normalizing with a DIFFERENT (e.g. global) mean/std
        instead of this window's own. The detector must catch this.
        """
        rng = np.random.RandomState(4)
        raw_window = rng.randn(48, 7)
        wrong_source = rng.randn(1000, 7)
        wrong_mu = wrong_source.mean(axis=0)
        wrong_sig = wrong_source.std(axis=0)
        bugged_normalized = (raw_window - wrong_mu) / (wrong_sig + NORMALIZATION_EPSILON)

        passed, detail = verify_per_window_normalization(raw_window, bugged_normalized)
        assert not passed, (
            "verify_per_window_normalization FAILED TO DETECT normalization "
            "using external/global statistics instead of window-local ones."
        )

    def test_normalization_correct_even_for_boundary_overlapping_window(self, generator) -> None:
        """
        V-LEAK-004 must hold identically for a window whose rows span
        the train/test boundary (one of the ~47 LC-4 overlap windows)
        — normalization never uses anything outside the window's own
        48 rows, regardless of which calendar period those rows
        belong to.
        """
        df = _make_full_feature_df("2022-12-27 00:00:00", 200, ("1h",))
        gen = WindowGenerator()
        raw = gen.extract_windows(df, "1h")
        test_start_ns = TEST_START.value
        overlap_indices = np.where(
            (raw.earliest_timestamps < test_start_ns) & (raw.anchor_timestamps >= test_start_ns)
        )[0]
        assert len(overlap_indices) > 0, "fixture should contain at least one LC-4 overlap window"
        idx = overlap_indices[0]
        raw_window = raw.windows[idx]
        normalized_window = gen.normalize_window(raw_window)
        passed, detail = verify_per_window_normalization(raw_window, normalized_window)
        assert passed


# --- check_window_shape / check_window_count --------------------------------------

class TestCheckWindowShape:
    def test_correct_shape_passes(self, generator) -> None:
        df = _make_full_feature_df("2020-01-01 19:00:00", 100, ("1h",))
        train_ws, test_ws = generator.generate(df, "1h")
        for ws in (train_ws, test_ws):
            if len(ws.windows) > 0:
                passed, detail = check_window_shape(ws)
                assert passed

    def test_wrong_feature_count_fails(self) -> None:
        ws = WindowSet(
            windows=np.zeros((5, 48, 5), dtype="float32"),
            anchor_timestamps=np.zeros(5, dtype="int64"),
            earliest_timestamps=np.zeros(5, dtype="int64"),
        )
        passed, detail = check_window_shape(ws)
        assert not passed
        assert "features" in detail

    def test_wrong_window_size_fails(self) -> None:
        ws = WindowSet(
            windows=np.zeros((5, 40, 7), dtype="float32"),
            anchor_timestamps=np.zeros(5, dtype="int64"),
            earliest_timestamps=np.zeros(5, dtype="int64"),
        )
        passed, detail = check_window_shape(ws)
        assert not passed
        assert "window dim" in detail


class TestCheckWindowCount:
    def test_none_skips_check(self) -> None:
        ws = WindowSet(
            windows=np.zeros((5, 48, 7), dtype="float32"),
            anchor_timestamps=np.zeros(5, dtype="int64"),
            earliest_timestamps=np.zeros(5, dtype="int64"),
        )
        passed, detail = check_window_count(ws, expected_count=None)
        assert passed

    def test_far_off_fails(self) -> None:
        ws = WindowSet(
            windows=np.zeros((5, 48, 7), dtype="float32"),
            anchor_timestamps=np.zeros(5, dtype="int64"),
            earliest_timestamps=np.zeros(5, dtype="int64"),
        )
        passed, detail = check_window_count(ws, expected_count=99999)
        assert not passed


# --- categorize_by_anchor / verify_train_windows_stay_in_train_period ---------------

class TestCategorizeByAnchor:
    def test_train_windows_all_anchored_at_or_before_train_end(
        self, generator, boundary_straddling_features
    ) -> None:
        train_ws, test_ws = generator.generate(boundary_straddling_features, "1h")
        assert (train_ws.anchor_timestamps <= TRAIN_END.value).all()

    def test_test_windows_all_anchored_at_or_after_test_start(
        self, generator, boundary_straddling_features
    ) -> None:
        train_ws, test_ws = generator.generate(boundary_straddling_features, "1h")
        assert (test_ws.anchor_timestamps >= TEST_START.value).all()

    def test_every_window_categorized_exactly_once(
        self, generator, boundary_straddling_features
    ) -> None:
        raw = generator.extract_windows(boundary_straddling_features, "1h")
        train_ws, test_ws = generator.categorize_by_anchor(raw)
        assert len(train_ws.windows) + len(test_ws.windows) == len(raw.windows)

    def test_verify_train_windows_stay_in_train_period_passes(
        self, generator, boundary_straddling_features
    ) -> None:
        train_ws, test_ws = generator.generate(boundary_straddling_features, "1h")
        passed, detail = verify_train_windows_stay_in_train_period(train_ws)
        assert passed

    def test_verify_train_windows_detects_injected_leak(self) -> None:
        """
        CRITICAL: manually inject a train window whose anchor is AFTER
        TRAIN_END, simulating a categorization bug, and confirm the
        detector catches it.
        """
        bad_anchor = TEST_START.value  # deliberately wrong: a test-period anchor
        bugged_train_ws = WindowSet(
            windows=np.zeros((1, 48, 7), dtype="float32"),
            anchor_timestamps=np.array([bad_anchor], dtype="int64"),
            earliest_timestamps=np.array([bad_anchor - 47 * 3_600_000_000_000], dtype="int64"),
        )
        passed, detail = verify_train_windows_stay_in_train_period(bugged_train_ws)
        assert not passed, (
            "verify_train_windows_stay_in_train_period FAILED TO DETECT a "
            "train window anchored past TRAIN_END."
        )
        assert "TRAIN_END" in detail

    def test_verify_test_windows_anchored_in_test_period_passes(
        self, generator, boundary_straddling_features
    ) -> None:
        train_ws, test_ws = generator.generate(boundary_straddling_features, "1h")
        passed, detail = verify_test_windows_anchored_in_test_period(test_ws)
        assert passed

    def test_verify_test_windows_detects_injected_leak(self) -> None:
        bad_anchor = TRAIN_END.value  # deliberately wrong: a train-period anchor
        bugged_test_ws = WindowSet(
            windows=np.zeros((1, 48, 7), dtype="float32"),
            anchor_timestamps=np.array([bad_anchor], dtype="int64"),
            earliest_timestamps=np.array([bad_anchor - 47 * 3_600_000_000_000], dtype="int64"),
        )
        passed, detail = verify_test_windows_anchored_in_test_period(bugged_test_ws)
        assert not passed
        assert "TEST_START" in detail


# --- count_boundary_overlap_windows (LC-4, EXPECTED overlap, not a bug) -------------

class TestCountBoundaryOverlapWindows:
    def test_boundary_straddling_data_shows_exactly_47_overlap_windows(
        self, generator, boundary_straddling_features
    ) -> None:
        """
        THE key regression test for the corrected LC-4 design: with
        data straddling the boundary, EXACTLY 47 test windows must
        show an earliest timestamp before TEST_START — not 0 (the
        old, incorrect "split-then-window" design) and not some other
        number.
        """
        train_ws, test_ws = generator.generate(boundary_straddling_features, "1h")
        passed, detail, count = count_boundary_overlap_windows(test_ws)
        assert count == EXPECTED_MAX_BOUNDARY_OVERLAP_WINDOWS
        assert passed

    def test_test_windows_far_from_boundary_have_zero_overlap(self, generator) -> None:
        df = _make_full_feature_df("2023-03-01 00:00:00", 200, ("1h",))
        train_ws, test_ws = generator.generate(df, "1h")
        passed, detail, count = count_boundary_overlap_windows(test_ws)
        assert count == 0
        assert passed

    def test_expected_max_matches_ds02_lc4(self) -> None:
        assert EXPECTED_MAX_BOUNDARY_OVERLAP_WINDOWS == 47

    def test_exceeding_expected_max_fails(self, generator, boundary_straddling_features) -> None:
        train_ws, test_ws = generator.generate(boundary_straddling_features, "1h")
        passed, detail, count = count_boundary_overlap_windows(test_ws, expected_max=10)
        assert not passed


# --- run_window_generation (end-to-end M6 orchestration, full-size dataset) --------

class TestRunWindowGeneration:
    def test_all_four_timeframes_processed(self) -> None:
        df = _make_full_feature_df("2022-12-27 00:00:00", 200)
        result = run_window_generation(
            df, raise_on_failure=True, expected_n_train_windows=None, expected_n_test_windows=None
        )
        assert set(result["train"].keys()) == {"15m", "1h", "4h", "1d"}
        assert set(result["test"].keys()) == {"15m", "1h", "4h", "1d"}

    def test_full_size_dataset_matches_ds02_v1_2_exact_numbers(self) -> None:
        """
        THE critical DS-02 v1.2 / DS-04 v1.1 assertion on the real
        35,045-row feature matrix: N_train_windows ≈ 26,222,
        N_test_windows = EXACTLY 8,760, and exactly 47 test windows
        show the documented boundary overlap.
        """
        df = _make_full_feature_df("2020-01-01 19:00:00", 35_045, ("1h",))
        result = run_window_generation(df, raise_on_failure=True, timeframes=("1h",))

        train_ws = result["train"]["1h"]
        test_ws = result["test"]["1h"]

        assert abs(len(train_ws.windows) - 26_222) < 50
        assert len(test_ws.windows) == EXPECTED_N_TEST_WINDOWS
        assert len(test_ws.windows) == 8_760

        _, _, overlap_count = count_boundary_overlap_windows(test_ws)
        assert overlap_count == 47

    def test_raises_on_shape_or_boundary_failure(self) -> None:
        """
        Deliberately corrupt window generation via an impossibly tight
        window-count expectation to confirm run_window_generation
        surfaces failures via WindowGenerationError.
        """
        df = _make_full_feature_df("2022-12-27 00:00:00", 200)
        with pytest.raises(WindowGenerationError):
            run_window_generation(
                df,
                raise_on_failure=True,
                expected_n_train_windows=1,
                expected_n_test_windows=1,
            )

    def test_does_not_raise_when_disabled(self) -> None:
        df = _make_full_feature_df("2022-12-27 00:00:00", 200)
        run_window_generation(
            df,
            raise_on_failure=False,
            expected_n_train_windows=1,
            expected_n_test_windows=1,
        )
