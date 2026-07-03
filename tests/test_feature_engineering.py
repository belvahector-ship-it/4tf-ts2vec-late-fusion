"""
tests/test_feature_engineering.py

Unit tests for src/data/feature_engineering.py (M4 — Feature Engineering).

Covers IMP-01 v1.2 M4 Definition of Done:
- All 7 features are computed identically for all 4 timeframe suffixes
- No NaN or Inf values remain after row dropping
- First timestamp after processing is 2020-01-01 19:00:00 UTC
  (CORRECTED per DS-02 v1.1 / DS-04 v1.1 — see
  AUDIT_REPORT_DS01-DS04_IMP01.md. Previously misstated as
  "2020-01-19" in v1.0/v1.1 documents; that was 19 DAYS instead of
  19 HOURS after the 2020-01-01 00:00 UTC dataset start.)
- body_ratio uses +1e-8 epsilon — no division-by-zero on doji candles
- volume_zscore rolling window uses window=20
- Unit tests verify each feature formula against manually computed
  expected values
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.feature_engineering import (
    BODY_RATIO_EPSILON,
    EXPECTED_FEATURE_COLUMNS,
    EXPECTED_FEATURE_ROWS,
    EXPECTED_FIRST_TIMESTAMP,
    VOLUME_ZSCORE_WINDOW,
    FeatureEngineer,
    FeatureEngineeringError,
    check_feature_matrix_integrity,
    check_feature_matrix_schema,
    run_feature_engineering,
)


# --- Fixtures ----------------------------------------------------------------

def _make_master_row(open_, high, low, close, volume) -> dict:
    """Build a single row's worth of {field}_1h columns for hand-crafted tests."""
    return {"open_1h": open_, "high_1h": high, "low_1h": low, "close_1h": close, "volume_1h": volume}


@pytest.fixture
def engineer() -> FeatureEngineer:
    return FeatureEngineer()


@pytest.fixture
def small_master_df() -> pd.DataFrame:
    """
    A small master DataFrame with all 4 timeframe suffixes, values
    chosen for easy hand-calculation of expected feature values.
    """
    n = 30
    timestamps = pd.date_range("2020-01-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame({"timestamp": timestamps})
    for suffix, base in (("15m", 100.0), ("1h", 1000.0), ("4h", 5000.0), ("1d", 20000.0)):
        open_ = base + np.arange(n, dtype="float64")
        close = open_ + 1.0
        high = np.maximum(open_, close) + 2.0
        low = np.minimum(open_, close) - 2.0
        volume = 10.0 + np.arange(n, dtype="float64")
        df[f"open_{suffix}"] = open_
        df[f"high_{suffix}"] = high
        df[f"low_{suffix}"] = low
        df[f"close_{suffix}"] = close
        df[f"volume_{suffix}"] = volume
    return df


# --- Individual feature formulas (hand-computed expected values) --------------

class TestOpenReturn:
    def test_matches_hand_computed_value(self, engineer) -> None:
        df = pd.DataFrame(
            {
                "open_1h": [100.0, 102.0, 99.0],
                "close_1h": [101.0, 103.0, 98.0],
            }
        )
        result = engineer.compute_open_return(df, "1h")
        # row 1: (open[1] - close[0]) / close[0] = (102 - 101) / 101
        expected_row1 = (102.0 - 101.0) / 101.0
        assert np.isclose(result.iloc[1], expected_row1)
        # row 2: (99 - 103) / 103
        expected_row2 = (99.0 - 103.0) / 103.0
        assert np.isclose(result.iloc[2], expected_row2)

    def test_first_row_is_nan(self, engineer) -> None:
        df = pd.DataFrame({"open_1h": [100.0, 102.0], "close_1h": [101.0, 103.0]})
        result = engineer.compute_open_return(df, "1h")
        assert pd.isna(result.iloc[0])


class TestHighReturn:
    def test_matches_hand_computed_value(self, engineer) -> None:
        df = pd.DataFrame({"open_1h": [100.0], "high_1h": [105.0]})
        result = engineer.compute_high_return(df, "1h")
        expected = (105.0 - 100.0) / 100.0
        assert np.isclose(result.iloc[0], expected)
        assert np.isclose(result.iloc[0], 0.05)


class TestLowReturn:
    def test_matches_hand_computed_value(self, engineer) -> None:
        df = pd.DataFrame({"open_1h": [100.0], "low_1h": [95.0]})
        result = engineer.compute_low_return(df, "1h")
        expected = (95.0 - 100.0) / 100.0
        assert np.isclose(result.iloc[0], expected)
        assert np.isclose(result.iloc[0], -0.05)


class TestCloseReturn:
    def test_matches_hand_computed_value(self, engineer) -> None:
        df = pd.DataFrame({"open_1h": [100.0], "close_1h": [103.0]})
        result = engineer.compute_close_return(df, "1h")
        expected = (103.0 - 100.0) / 100.0
        assert np.isclose(result.iloc[0], expected)
        assert np.isclose(result.iloc[0], 0.03)


class TestHlRange:
    def test_matches_hand_computed_value(self, engineer) -> None:
        df = pd.DataFrame({"open_1h": [100.0], "high_1h": [110.0], "low_1h": [90.0]})
        result = engineer.compute_hl_range(df, "1h")
        expected = (110.0 - 90.0) / 100.0
        assert np.isclose(result.iloc[0], expected)
        assert np.isclose(result.iloc[0], 0.20)


class TestBodyRatio:
    def test_matches_hand_computed_value(self, engineer) -> None:
        df = pd.DataFrame(
            {"open_1h": [100.0], "close_1h": [103.0], "high_1h": [105.0], "low_1h": [98.0]}
        )
        result = engineer.compute_body_ratio(df, "1h")
        expected = abs(103.0 - 100.0) / (105.0 - 98.0 + BODY_RATIO_EPSILON)
        assert np.isclose(result.iloc[0], expected)

    def test_doji_candle_no_division_by_zero(self, engineer) -> None:
        """
        A doji candle has high == low (or very close), which would
        cause ZeroDivisionError / Inf without the epsilon. ADR-015
        explicitly requires +1e-8 to prevent this.
        """
        df = pd.DataFrame(
            {"open_1h": [100.0], "close_1h": [100.0], "high_1h": [100.0], "low_1h": [100.0]}
        )
        result = engineer.compute_body_ratio(df, "1h")
        assert np.isfinite(result.iloc[0])
        assert not np.isnan(result.iloc[0])
        assert not np.isinf(result.iloc[0])
        # body=0, hl=0 -> 0 / (0 + 1e-8) = 0
        assert np.isclose(result.iloc[0], 0.0)

    def test_epsilon_value_matches_adr_015(self) -> None:
        assert BODY_RATIO_EPSILON == 1e-8


class TestVolumeZscore:
    def test_matches_hand_computed_value(self, engineer) -> None:
        # 25 rows so we have 5 valid z-score rows after the 20-row warmup.
        volumes = list(range(1, 26))  # 1..25
        df = pd.DataFrame({"volume_1h": [float(v) for v in volumes]})
        result = engineer.compute_volume_zscore(df, "1h", window=20)

        # Row index 19 (the 20th value) is the first with a full window (rows 0-19).
        window_values = np.array(volumes[0:20], dtype="float64")
        expected_mean = window_values.mean()
        expected_std = window_values.std(ddof=1)  # pandas default ddof=1
        expected_z = (volumes[19] - expected_mean) / expected_std
        assert np.isclose(result.iloc[19], expected_z)

    def test_first_19_rows_are_nan_with_default_window(self, engineer) -> None:
        volumes = [float(v) for v in range(1, 26)]
        df = pd.DataFrame({"volume_1h": volumes})
        result = engineer.compute_volume_zscore(df, "1h", window=VOLUME_ZSCORE_WINDOW)
        assert result.iloc[0:19].isna().all()
        assert not pd.isna(result.iloc[19])

    def test_default_window_is_20(self) -> None:
        assert VOLUME_ZSCORE_WINDOW == 20


# --- compute_features_for_timeframe / compute_all_features -------------------

class TestComputeFeaturesForTimeframe:
    def test_returns_7_columns(self, engineer, small_master_df) -> None:
        result = engineer.compute_features_for_timeframe(small_master_df, "1h")
        assert len(result.columns) == 7

    def test_column_names_have_correct_suffix(self, engineer, small_master_df) -> None:
        result = engineer.compute_features_for_timeframe(small_master_df, "4h")
        for col in result.columns:
            assert col.endswith("_4h")

    def test_same_row_count_as_input(self, engineer, small_master_df) -> None:
        result = engineer.compute_features_for_timeframe(small_master_df, "1h")
        assert len(result) == len(small_master_df)


class TestComputeAllFeatures:
    def test_returns_29_columns(self, engineer, small_master_df) -> None:
        result = engineer.compute_all_features(small_master_df)
        assert len(result.columns) == EXPECTED_FEATURE_COLUMNS

    def test_includes_timestamp(self, engineer, small_master_df) -> None:
        result = engineer.compute_all_features(small_master_df)
        assert "timestamp" in result.columns

    def test_identical_formula_across_all_4_timeframes(self, engineer) -> None:
        """
        DoD: 'All 7 features are computed identically for all 4
        timeframe suffixes.' Verify this by constructing a master
        DataFrame where all 4 timeframes have IDENTICAL underlying
        OHLCV values, then confirming all 4 timeframes' computed
        features are also identical.
        """
        n = 25
        timestamps = pd.date_range("2020-01-01", periods=n, freq="1h", tz="UTC")
        df = pd.DataFrame({"timestamp": timestamps})
        open_ = 100.0 + np.arange(n, dtype="float64")
        close = open_ + 1.0
        high = np.maximum(open_, close) + 2.0
        low = np.minimum(open_, close) - 2.0
        volume = 10.0 + np.arange(n, dtype="float64")
        for suffix in ("15m", "1h", "4h", "1d"):
            df[f"open_{suffix}"] = open_
            df[f"high_{suffix}"] = high
            df[f"low_{suffix}"] = low
            df[f"close_{suffix}"] = close
            df[f"volume_{suffix}"] = volume

        result = engineer.compute_all_features(df)
        for feature in (
            "open_return",
            "high_return",
            "low_return",
            "close_return",
            "volume_zscore",
            "hl_range",
            "body_ratio",
        ):
            col_15m = result[f"{feature}_15m"].to_numpy()
            col_1h = result[f"{feature}_1h"].to_numpy()
            col_4h = result[f"{feature}_4h"].to_numpy()
            col_1d = result[f"{feature}_1d"].to_numpy()
            np.testing.assert_array_equal(col_15m, col_1h)
            np.testing.assert_array_equal(col_1h, col_4h)
            np.testing.assert_array_equal(col_4h, col_1d)


# --- drop_nan_rows --------------------------------------------------------------

class TestDropNanRows:
    def test_drops_leading_nan_rows(self, engineer, small_master_df) -> None:
        raw = engineer.compute_all_features(small_master_df)
        cleaned = engineer.drop_nan_rows(raw)
        # 30-row fixture, window=20 -> expect 19 rows dropped, 11 remain.
        assert len(cleaned) == len(small_master_df) - 19

    def test_no_nan_remains_after_drop(self, engineer, small_master_df) -> None:
        raw = engineer.compute_all_features(small_master_df)
        cleaned = engineer.drop_nan_rows(raw)
        feature_cols = [c for c in cleaned.columns if c != "timestamp"]
        assert not cleaned[feature_cols].isna().any().any()

    def test_index_is_reset(self, engineer, small_master_df) -> None:
        raw = engineer.compute_all_features(small_master_df)
        cleaned = engineer.drop_nan_rows(raw)
        assert list(cleaned.index) == list(range(len(cleaned)))

    def test_first_remaining_timestamp_is_19_hours_after_start(
        self, engineer, small_master_df
    ) -> None:
        """
        Direct regression test for the corrected bug: dropping 19 rows
        from a 2020-01-01 00:00 UTC hourly start must land on
        2020-01-01 19:00 UTC (19 hours later), NOT 2020-01-19 00:00 UTC
        (19 days later). See module docstring and
        AUDIT_REPORT_DS01-DS04_IMP01.md.
        """
        raw = engineer.compute_all_features(small_master_df)
        cleaned = engineer.drop_nan_rows(raw)
        expected = pd.Timestamp("2020-01-01 19:00:00", tz="UTC")
        assert cleaned["timestamp"].iloc[0] == expected


# --- check_feature_matrix_integrity ------------------------------------------------

class TestCheckFeatureMatrixIntegrity:
    def test_clean_matrix_passes(self, engineer, small_master_df) -> None:
        raw = engineer.compute_all_features(small_master_df)
        cleaned = engineer.drop_nan_rows(raw)
        passed, detail = check_feature_matrix_integrity(cleaned)
        assert passed

    def test_nan_detected(self, engineer, small_master_df) -> None:
        raw = engineer.compute_all_features(small_master_df)
        cleaned = engineer.drop_nan_rows(raw)
        broken = cleaned.copy()
        broken.loc[0, "hl_range_1h"] = np.nan
        passed, detail = check_feature_matrix_integrity(broken)
        assert not passed
        assert "hl_range_1h" in detail

    def test_inf_detected(self, engineer, small_master_df) -> None:
        raw = engineer.compute_all_features(small_master_df)
        cleaned = engineer.drop_nan_rows(raw)
        broken = cleaned.copy()
        broken.loc[0, "close_return_1h"] = np.inf
        passed, detail = check_feature_matrix_integrity(broken)
        assert not passed
        assert "close_return_1h" in detail


# --- check_feature_matrix_schema ----------------------------------------------------

class TestCheckFeatureMatrixSchema:
    def test_wrong_column_count_fails(self, engineer, small_master_df) -> None:
        raw = engineer.compute_all_features(small_master_df)
        cleaned = engineer.drop_nan_rows(raw)
        broken = cleaned.drop(columns=["body_ratio_1d"])
        passed, detail = check_feature_matrix_schema(broken, expected_rows=None, expected_first_timestamp=None)
        assert not passed
        assert "columns" in detail

    def test_wrong_first_timestamp_fails(self, engineer, small_master_df) -> None:
        """
        Confirms the schema check would have caught the old bug: if a
        feature matrix's first timestamp were "2020-01-19" instead of
        the correct "2020-01-01 19:00", the check must fail.
        """
        raw = engineer.compute_all_features(small_master_df)
        cleaned = engineer.drop_nan_rows(raw)
        wrong_timestamp = pd.Timestamp("2020-01-19 00:00:00", tz="UTC")
        passed, detail = check_feature_matrix_schema(
            cleaned, expected_rows=None, expected_first_timestamp=wrong_timestamp
        )
        assert not passed
        assert "first timestamp" in detail

    def test_full_size_dataset_matches_exact_ds04_numbers(self, engineer) -> None:
        """
        The critical DS-04 v1.1 V-DATA-004 assertion: with the real
        35,064-row aligned master, the feature matrix must end up with
        EXACTLY 35,045 rows starting at 2020-01-01 19:00:00 UTC
        (CORRECTED — see module docstring).
        """
        n = 35_064
        timestamps = pd.date_range("2020-01-01", periods=n, freq="1h", tz="UTC")
        df = pd.DataFrame({"timestamp": timestamps})
        for suffix, base in (("15m", 100.0), ("1h", 1000.0), ("4h", 5000.0), ("1d", 20000.0)):
            open_ = base + np.arange(n, dtype="float64") * 0.01
            close = open_ + 0.5
            high = np.maximum(open_, close) + 1.0
            low = np.minimum(open_, close) - 1.0
            volume = 10.0 + np.arange(n, dtype="float64") * 0.001
            df[f"open_{suffix}"] = open_
            df[f"high_{suffix}"] = high
            df[f"low_{suffix}"] = low
            df[f"close_{suffix}"] = close
            df[f"volume_{suffix}"] = volume

        features = run_feature_engineering(df, raise_on_failure=True)
        assert len(features) == EXPECTED_FEATURE_ROWS
        assert len(features.columns) == EXPECTED_FEATURE_COLUMNS
        assert features["timestamp"].iloc[0] == EXPECTED_FIRST_TIMESTAMP
        assert features["timestamp"].iloc[0] == pd.Timestamp("2020-01-01 19:00:00", tz="UTC")


# --- run_feature_engineering (end-to-end M4 orchestration) --------------------

class TestRunFeatureEngineering:
    def test_raises_on_schema_mismatch(self, small_master_df) -> None:
        # small_master_df is only 30 rows; with expected_rows=35045
        # (the production default), this must raise.
        with pytest.raises(FeatureEngineeringError):
            run_feature_engineering(small_master_df, raise_on_failure=True)

    def test_does_not_raise_when_expected_rows_none(self, small_master_df) -> None:
        result = run_feature_engineering(
            small_master_df,
            raise_on_failure=True,
            expected_rows=None,
            expected_first_timestamp=None,
        )
        assert len(result) == len(small_master_df) - 19

    def test_returns_clean_dataframe(self, small_master_df) -> None:
        result = run_feature_engineering(
            small_master_df,
            raise_on_failure=False,
            expected_rows=None,
            expected_first_timestamp=None,
        )
        feature_cols = [c for c in result.columns if c != "timestamp"]
        assert not result[feature_cols].isna().any().any()
