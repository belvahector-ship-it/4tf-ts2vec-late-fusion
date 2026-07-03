"""
tests/test_validation.py

Unit tests for src/data/validation.py (M2 — Data Validation).

Covers IMP-01 M2 Definition of Done:
- "All checks from DS-02 Stage 1 are implemented"
- "Each failed check produces an error message identifying which rows
  violate the condition"
- "The module aborts (raises DataValidationError) if any check fails"
- "A clean dataset passes all checks without warnings"
- "All checks are covered by unit tests with both passing and failing
  synthetic inputs"

Design note
-----------
Unlike M1, this module has zero dependency on ccxt/pyarrow/network —
it operates purely on in-memory pandas DataFrames. Every test in this
file is runnable in any environment with pandas and numpy installed,
including this repository's sandbox.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.validation import (
    GAP_RATIO_TOLERANCE,
    DataValidationError,
    DataValidator,
    ValidationReport,
    append_validation_to_manifest,
    validate_all_timeframes,
)


# --- Fixtures ----------------------------------------------------------------

def _make_clean_df(
    n: int = 100, timeframe: str = "1h", start: str = "2020-01-01"
) -> pd.DataFrame:
    """Build a fully valid synthetic OHLCV DataFrame with no gaps or violations."""
    freq_map = {"15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}
    timestamps = pd.date_range(start, periods=n, freq=freq_map[timeframe], tz="UTC")
    open_ = np.linspace(100.0, 200.0, n)
    close = open_ + 0.5
    high = np.maximum(open_, close) + 1.0
    low = np.minimum(open_, close) - 1.0
    volume = np.linspace(10.0, 20.0, n)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


@pytest.fixture
def clean_1h_df() -> pd.DataFrame:
    return _make_clean_df(n=200, timeframe="1h")


@pytest.fixture
def validator() -> DataValidator:
    return DataValidator()


# --- check_columns_present -----------------------------------------------------

class TestCheckColumnsPresent:
    def test_all_columns_present_passes(self, validator, clean_1h_df) -> None:
        result = validator.check_columns_present(clean_1h_df)
        assert result.passed

    def test_missing_column_fails(self, validator, clean_1h_df) -> None:
        broken = clean_1h_df.drop(columns=["volume"])
        result = validator.check_columns_present(broken)
        assert not result.passed
        assert "volume" in result.detail


# --- check_monotonicity ----------------------------------------------------------

class TestCheckMonotonicity:
    def test_sorted_timestamps_pass(self, validator, clean_1h_df) -> None:
        assert validator.check_monotonicity(clean_1h_df)

    def test_shuffled_timestamps_fail(self, validator, clean_1h_df) -> None:
        shuffled = clean_1h_df.sample(frac=1, random_state=42).reset_index(drop=True)
        assert not validator.check_monotonicity(shuffled)

    def test_single_swapped_row_fails(self, validator, clean_1h_df) -> None:
        df = clean_1h_df.copy()
        df.loc[[5, 6], "timestamp"] = df.loc[[6, 5], "timestamp"].to_numpy()
        assert not validator.check_monotonicity(df)


# --- check_duplicates -------------------------------------------------------------

class TestCheckDuplicates:
    def test_unique_timestamps_pass(self, validator, clean_1h_df) -> None:
        assert validator.check_duplicates(clean_1h_df)

    def test_duplicate_timestamp_fails(self, validator, clean_1h_df) -> None:
        df = clean_1h_df.copy()
        df.loc[1, "timestamp"] = df.loc[0, "timestamp"]
        assert not validator.check_duplicates(df)


# --- check_ohlc_integrity ----------------------------------------------------------

class TestCheckOhlcIntegrity:
    def test_valid_ohlc_passes(self, validator, clean_1h_df) -> None:
        assert validator.check_ohlc_integrity(clean_1h_df)

    def test_high_less_than_open_fails(self, validator, clean_1h_df) -> None:
        df = clean_1h_df.copy()
        df.loc[0, "high"] = df.loc[0, "open"] - 10.0
        assert not validator.check_ohlc_integrity(df)

    def test_low_greater_than_close_fails(self, validator, clean_1h_df) -> None:
        df = clean_1h_df.copy()
        df.loc[0, "low"] = df.loc[0, "close"] + 10.0
        assert not validator.check_ohlc_integrity(df)

    def test_high_less_than_low_fails(self, validator, clean_1h_df) -> None:
        df = clean_1h_df.copy()
        df.loc[0, "high"] = 1.0
        df.loc[0, "low"] = 100.0
        assert not validator.check_ohlc_integrity(df)


# --- check_positive_prices / check_non_negative_volume ---------------------------

class TestPricesAndVolume:
    def test_positive_prices_pass(self, validator, clean_1h_df) -> None:
        assert validator.check_positive_prices(clean_1h_df)

    def test_zero_price_fails(self, validator, clean_1h_df) -> None:
        df = clean_1h_df.copy()
        df.loc[0, "open"] = 0.0
        assert not validator.check_positive_prices(df)

    def test_negative_price_fails(self, validator, clean_1h_df) -> None:
        df = clean_1h_df.copy()
        df.loc[0, "close"] = -5.0
        assert not validator.check_positive_prices(df)

    def test_non_negative_volume_passes(self, validator, clean_1h_df) -> None:
        assert validator.check_non_negative_volume(clean_1h_df)

    def test_negative_volume_fails(self, validator, clean_1h_df) -> None:
        df = clean_1h_df.copy()
        df.loc[0, "volume"] = -1.0
        assert not validator.check_non_negative_volume(df)

    def test_zero_volume_passes(self, validator, clean_1h_df) -> None:
        df = clean_1h_df.copy()
        df.loc[0, "volume"] = 0.0
        assert validator.check_non_negative_volume(df)


# --- check_nan_inf ----------------------------------------------------------------

class TestCheckNanInf:
    def test_clean_data_passes(self, validator, clean_1h_df) -> None:
        passed, counts = validator.check_nan_inf(clean_1h_df)
        assert passed
        assert counts == {}

    def test_nan_value_fails_with_column_identified(self, validator, clean_1h_df) -> None:
        df = clean_1h_df.copy()
        df.loc[3, "close"] = np.nan
        passed, counts = validator.check_nan_inf(df)
        assert not passed
        assert "close" in counts
        assert counts["close"] == 1

    def test_inf_value_fails_with_column_identified(self, validator, clean_1h_df) -> None:
        df = clean_1h_df.copy()
        df.loc[7, "volume"] = np.inf
        passed, counts = validator.check_nan_inf(df)
        assert not passed
        assert "volume" in counts

    def test_multiple_nan_in_same_column_counted(self, validator, clean_1h_df) -> None:
        df = clean_1h_df.copy()
        df.loc[[1, 2, 3], "open"] = np.nan
        passed, counts = validator.check_nan_inf(df)
        assert not passed
        assert counts["open"] == 3


# --- check_timezone -----------------------------------------------------------------

class TestCheckTimezone:
    def test_utc_timestamps_pass(self, validator, clean_1h_df) -> None:
        assert validator.check_timezone(clean_1h_df)

    def test_naive_timestamps_fail(self, validator, clean_1h_df) -> None:
        df = clean_1h_df.copy()
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
        assert not validator.check_timezone(df)

    def test_non_utc_timezone_fails(self, validator, clean_1h_df) -> None:
        df = clean_1h_df.copy()
        df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")
        assert not validator.check_timezone(df)


# --- check_date_coverage --------------------------------------------------------------

class TestCheckDateCoverage:
    def test_full_coverage_passes(self, validator) -> None:
        df = _make_clean_df(n=35_064, timeframe="1h", start="2020-01-01")
        assert validator.check_date_coverage(df, "2020-01-01", "2023-12-31")

    def test_late_start_fails(self, validator) -> None:
        df = _make_clean_df(n=100, timeframe="1h", start="2020-06-01")
        assert not validator.check_date_coverage(df, "2020-01-01", "2023-12-31")

    def test_early_end_fails(self, validator) -> None:
        df = _make_clean_df(n=100, timeframe="1h", start="2020-01-01")
        assert not validator.check_date_coverage(df, "2020-01-01", "2023-12-31")

    @pytest.mark.parametrize(
        "timeframe,periods",
        [("15m", 96 * 3), ("1h", 24 * 3), ("4h", 6 * 3), ("1d", 3)],
    )
    def test_timeframe_aware_last_candle_passes(
        self, validator, timeframe, periods
    ) -> None:
        """
        Regression (ADR-022 / DS-04 amendment): a timeframe whose last
        candle is NOT at 23:00 (4h ends 20:00, 1d ends 00:00) must still
        count as full coverage of the end date. Found on first real run:
        a fixed 23:00 boundary wrongly rejected 4h/1d.
        """
        # 3 full days starting 2020-01-01; end date is the 3rd day.
        df = _make_clean_df(n=periods, timeframe=timeframe, start="2020-01-01")
        assert validator.check_date_coverage(
            df, "2020-01-01", "2020-01-03", timeframe
        )

    def test_coverage_stops_one_candle_short_fails(self, validator) -> None:
        """Dropping the final candle of the end day must fail coverage."""
        df = _make_clean_df(n=6 * 3, timeframe="4h", start="2020-01-01")
        df_short = df.iloc[:-1]  # drop the 2020-01-03 20:00 candle
        assert not validator.check_date_coverage(
            df_short, "2020-01-01", "2020-01-03", "4h"
        )


# --- check_gap_ratio -----------------------------------------------------------------

class TestCheckGapRatio:
    def test_no_gaps_returns_zero(self, validator, clean_1h_df) -> None:
        assert validator.check_gap_ratio(clean_1h_df, "1h") == 0.0

    def test_single_missing_candle_detected(self, validator) -> None:
        df = _make_clean_df(n=100, timeframe="1h")
        # Remove one row to create a 2-unit gap (1 missing candle).
        df_with_gap = df.drop(index=50).reset_index(drop=True)
        gap_ratio = validator.check_gap_ratio(df_with_gap, "1h")
        assert gap_ratio > 0.0
        # 1 missing out of ~100 expected -> ~1%
        assert gap_ratio < 0.05

    def test_large_gap_exceeds_tolerance(self, validator) -> None:
        df = _make_clean_df(n=100, timeframe="1h")
        # Remove 10 consecutive rows -> large gap ratio.
        df_with_large_gap = df.drop(index=range(40, 50)).reset_index(drop=True)
        gap_ratio = validator.check_gap_ratio(df_with_large_gap, "1h")
        assert gap_ratio > GAP_RATIO_TOLERANCE

    def test_unknown_timeframe_raises(self, validator, clean_1h_df) -> None:
        with pytest.raises(ValueError, match="Unknown timeframe"):
            validator.check_gap_ratio(clean_1h_df, "30m")

    def test_single_row_returns_zero(self, validator) -> None:
        df = _make_clean_df(n=1, timeframe="1h")
        assert validator.check_gap_ratio(df, "1h") == 0.0


# --- check_max_single_gap -------------------------------------------------------------

class TestCheckMaxSingleGap:
    def test_no_gap_passes(self, validator, clean_1h_df) -> None:
        assert validator.check_max_single_gap(clean_1h_df, "1h")

    def test_one_missing_candle_still_passes(self, validator) -> None:
        # DS-04 V-DATA-001: "no single gap exceeds one candle duration
        # by more than one unit" -> a 2x gap (1 missing candle) is OK.
        df = _make_clean_df(n=100, timeframe="1h")
        df_with_gap = df.drop(index=50).reset_index(drop=True)
        assert validator.check_max_single_gap(df_with_gap, "1h")

    def test_large_single_gap_fails(self, validator) -> None:
        df = _make_clean_df(n=100, timeframe="1h")
        df_with_large_gap = df.drop(index=range(40, 60)).reset_index(drop=True)
        assert not validator.check_max_single_gap(df_with_large_gap, "1h")


# --- validate_timeframe (integration of all checks) -----------------------------------

class TestValidateTimeframe:
    def test_clean_dataset_passes_without_warnings(self, validator) -> None:
        df = _make_clean_df(n=35_064, timeframe="1h", start="2020-01-01")
        report = validator.validate_timeframe(
            df, "1h", start_date="2020-01-01", end_date="2023-12-31"
        )
        assert report.passed
        assert len(report.failed_checks) == 0

    def test_report_is_validation_report_instance(self, validator, clean_1h_df) -> None:
        report = validator.validate_timeframe(
            clean_1h_df,
            "1h",
            start_date="2020-01-01",
            end_date="2020-01-09",  # matches the small fixture's coverage
            raise_on_failure=False,
        )
        assert isinstance(report, ValidationReport)
        assert report.timeframe == "1h"

    def test_failing_dataset_raises_by_default(self, validator) -> None:
        df = _make_clean_df(n=50, timeframe="1h")
        df.loc[5, "high"] = -1.0  # violates OHLC integrity AND positive price
        with pytest.raises(DataValidationError):
            validator.validate_timeframe(df, "1h", raise_on_failure=True)

    def test_failing_dataset_does_not_raise_when_disabled(self, validator) -> None:
        df = _make_clean_df(n=50, timeframe="1h")
        df.loc[5, "high"] = -1.0
        report = validator.validate_timeframe(df, "1h", raise_on_failure=False)
        assert not report.passed
        assert len(report.failed_checks) > 0

    def test_error_message_identifies_failing_check(self, validator) -> None:
        """
        DoD: 'Each failed check produces an error message identifying
        which rows violate the condition' — verified here by checking
        the raised exception's message names the failing check.
        """
        df = _make_clean_df(n=50, timeframe="1h")
        df.loc[10, "volume"] = -5.0
        with pytest.raises(DataValidationError) as exc_info:
            validator.validate_timeframe(df, "1h", raise_on_failure=True)
        message = str(exc_info.value)
        assert "non_negative_volume" in message
        assert "FAIL" in message

    def test_missing_columns_short_circuits_remaining_checks(self, validator) -> None:
        df = _make_clean_df(n=50, timeframe="1h").drop(columns=["close"])
        with pytest.raises(DataValidationError) as exc_info:
            validator.validate_timeframe(df, "1h", raise_on_failure=True)
        assert "columns_present" in str(exc_info.value)

    def test_large_single_gap_is_warning_not_failure(self, validator) -> None:
        """
        ADR-022 regression: a single large gap (real-exchange-outage
        analogue) must NOT abort validation as long as the aggregate
        gap ratio stays under 5% — it is recorded/logged as a WARNING.
        Guards the 2020-02-19-style Binance-outage case found on the
        first real M1 run (2026-07-03).
        """
        # 1000 clean hourly candles; end_date at 23:00 must be <= last ts.
        df = _make_clean_df(n=1000, timeframe="1h", start="2020-01-01")
        # Remove 10 consecutive candles -> one 11-unit gap; ~1% aggregate.
        df_gap = df.drop(index=range(400, 410)).reset_index(drop=True)

        report = validator.validate_timeframe(
            df_gap,
            "1h",
            start_date="2020-01-01",
            end_date="2020-02-10",  # last ts is 2020-02-11 15:00 >= this@23:00
            raise_on_failure=True,  # must NOT raise despite the big gap
        )
        assert report.passed  # not aborted
        assert report.gap_ratio < 0.05  # aggregate still under the hard gate
        # single-gap recorded as a WARNING, not a hard failure
        assert "no_excessive_single_gap" in {w.name for w in report.warnings}
        assert "no_excessive_single_gap" not in {c.name for c in report.failed_checks}
        assert "[WARN] no_excessive_single_gap" in report.summary()

    def test_excessive_aggregate_gap_still_hard_fails(self, validator) -> None:
        """The 5% aggregate gap_ratio remains a HARD gate (unchanged)."""
        df = _make_clean_df(n=100, timeframe="1h", start="2020-01-01")
        # Drop 20% of candles -> aggregate gap ratio > 5% -> hard fail.
        df_gap = df.drop(index=range(40, 60)).reset_index(drop=True)
        report = validator.validate_timeframe(
            df_gap, "1h", start_date="2020-01-01", end_date="2020-01-01",
            raise_on_failure=False,
        )
        assert not report.passed
        assert "gap_ratio_within_tolerance" in {c.name for c in report.failed_checks}


# --- ValidationReport ------------------------------------------------------------------

class TestValidationReport:
    def test_passed_property_true_when_all_pass(self) -> None:
        from src.data.validation import CheckResult

        report = ValidationReport(
            timeframe="1h",
            checks=[CheckResult("a", True), CheckResult("b", True)],
        )
        assert report.passed

    def test_passed_property_false_when_any_fails(self) -> None:
        from src.data.validation import CheckResult

        report = ValidationReport(
            timeframe="1h",
            checks=[CheckResult("a", True), CheckResult("b", False)],
        )
        assert not report.passed
        assert len(report.failed_checks) == 1
        assert report.failed_checks[0].name == "b"

    def test_summary_includes_all_checks(self) -> None:
        from src.data.validation import CheckResult

        report = ValidationReport(
            timeframe="1h",
            checks=[CheckResult("check_a", True), CheckResult("check_b", False, "bad rows")],
            gap_ratio=0.01,
        )
        summary = report.summary()
        assert "check_a" in summary
        assert "check_b" in summary
        assert "bad rows" in summary
        assert "1h" in summary


# --- validate_all_timeframes -----------------------------------------------------------

class TestValidateAllTimeframes:
    def test_all_clean_timeframes_pass(self) -> None:
        dataframes = {
            "15m": _make_clean_df(n=500, timeframe="15m"),
            "1h": _make_clean_df(n=500, timeframe="1h"),
            "4h": _make_clean_df(n=500, timeframe="4h"),
            "1d": _make_clean_df(n=500, timeframe="1d"),
        }
        reports = validate_all_timeframes(
            dataframes,
            start_date="2020-01-01",
            end_date="2020-01-09",  # small range matching fixture coverage
            raise_on_failure=False,
        )
        assert set(reports.keys()) == {"15m", "1h", "4h", "1d"}
        assert reports["1h"].timeframe == "1h"

    def test_one_bad_timeframe_raises(self) -> None:
        bad_df = _make_clean_df(n=100, timeframe="1h")
        bad_df.loc[5, "open"] = -1.0
        dataframes = {
            "1h": bad_df,
            "4h": _make_clean_df(n=100, timeframe="4h"),
        }
        with pytest.raises(DataValidationError):
            validate_all_timeframes(dataframes, raise_on_failure=True)


# --- append_validation_to_manifest ------------------------------------------------------

class TestAppendValidationToManifest:
    def test_appends_validation_key(self, validator, clean_1h_df) -> None:
        report = validator.validate_timeframe(
            clean_1h_df,
            "1h",
            start_date="2020-01-01",
            end_date="2020-01-09",
            raise_on_failure=False,
        )
        manifest = {"symbol": "BTC/USDT", "exchange": "binance", "timeframes": {}}
        result = append_validation_to_manifest(manifest, {"1h": report})

        assert "validation" in result
        assert "1h" in result["validation"]
        assert "passed" in result["validation"]["1h"]
        assert "gap_ratio" in result["validation"]["1h"]
        assert "checks" in result["validation"]["1h"]

    def test_mutates_manifest_in_place(self, validator, clean_1h_df) -> None:
        report = validator.validate_timeframe(
            clean_1h_df,
            "1h",
            start_date="2020-01-01",
            end_date="2020-01-09",
            raise_on_failure=False,
        )
        manifest = {"symbol": "BTC/USDT"}
        result = append_validation_to_manifest(manifest, {"1h": report})
        assert result is manifest
