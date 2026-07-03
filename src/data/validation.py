"""
src/data/validation.py

Data validation module (M2 — Data Validation, DS-02 Stage 1, DS-04 §3.1).

Purpose
-------
Verifies all data integrity conditions defined in DS-02 Stage 1 before
any data is allowed to propagate downstream to M3 (Temporal Alignment).
Detecting a problem here — rather than after alignment, feature
engineering, or (worst case) after 20 branch training runs — is the
entire point of this module (IMP-01 "Coding Order" rationale, Risk
R-04).

Per DS-02 Leakage Checkpoint LC-1: this module performs NO imputation,
interpolation, or forward-fill. Gaps are logged only. Filling gaps is
the exclusive responsibility of Stage 2 (M3, Temporal Alignment), which
keeps leakage auditing unambiguous — there is exactly one place in the
pipeline where upsampling occurs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Candle duration per timeframe, used for gap detection.
TIMEFRAME_DURATIONS: dict[str, pd.Timedelta] = {
    "15m": pd.Timedelta(minutes=15),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
}

# Missing-candle tolerance per DS-02 Stage 1: "abort if total missing > 5%".
GAP_RATIO_TOLERANCE: float = 0.05

REQUIRED_COLUMNS: tuple[str, ...] = ("timestamp", "open", "high", "low", "close", "volume")


class DataValidationError(RuntimeError):
    """
    Raised when a dataset fails one or more DS-02 Stage 1 validation checks.

    The exception message contains a full report of every failed check,
    per IMP-01 M2 Definition of Done: "Each failed check produces an
    error message identifying which rows violate the condition."
    """


@dataclass
class CheckResult:
    """Result of a single validation check."""

    name: str
    passed: bool
    detail: str = ""


@dataclass
class ValidationReport:
    """
    Aggregated validation results for one timeframe.

    Attributes
    ----------
    timeframe : str
        The timeframe validated (e.g. "1h").
    checks : list[CheckResult]
        One entry per check performed, in the order run.
    gap_ratio : float
        Fraction of expected candles that are missing (0.0 = no gaps).
    """

    timeframe: str
    checks: list[CheckResult] = field(default_factory=list)
    gap_ratio: float = 0.0

    @property
    def passed(self) -> bool:
        """True if every check in this report passed."""
        return all(c.passed for c in self.checks)

    @property
    def failed_checks(self) -> list[CheckResult]:
        """The subset of checks that failed."""
        return [c for c in self.checks if not c.passed]

    def summary(self) -> str:
        """
        Build a human-readable multi-line summary of this report.

        Returns
        -------
        str
            One line per check, plus the gap ratio, suitable for
            logging or inclusion in a raised DataValidationError.
        """
        lines = [f"Validation report for timeframe='{self.timeframe}':"]
        for check in self.checks:
            status = "PASS" if check.passed else "FAIL"
            line = f"  [{status}] {check.name}"
            if check.detail:
                line += f" — {check.detail}"
            lines.append(line)
        lines.append(f"  gap_ratio = {self.gap_ratio:.4%}")
        return "\n".join(lines)


class DataValidator:
    """
    Runs all DS-02 Stage 1 validation checks against raw OHLCV data.

    Each `check_*` method returns a bool (or float for `check_gap_ratio`)
    and never raises on its own — `validate_timeframe` is the single
    place that aggregates results and decides whether to raise
    `DataValidationError`. This keeps individual checks composable and
    independently unit-testable (IMP-01 M2 DoD: "covered by unit tests
    with both passing and failing synthetic inputs").
    """

    def check_columns_present(self, df: pd.DataFrame) -> CheckResult:
        """Verify all required columns exist."""
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            return CheckResult(
                "columns_present", False, f"missing columns: {missing}"
            )
        return CheckResult("columns_present", True)

    def check_monotonicity(self, df: pd.DataFrame) -> bool:
        """
        Verify `timestamp` is strictly ascending.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain a `timestamp` column.

        Returns
        -------
        bool
            True if `df['timestamp']` equals its own sorted version.
        """
        return df["timestamp"].equals(df["timestamp"].sort_values().reset_index(drop=True))

    def check_duplicates(self, df: pd.DataFrame) -> bool:
        """
        Verify no duplicate timestamps exist.

        Returns
        -------
        bool
            True if `nunique(timestamp) == len(df)`.
        """
        return df["timestamp"].nunique() == len(df)

    def check_ohlc_integrity(self, df: pd.DataFrame) -> bool:
        """
        Verify OHLC price relationships hold for every row.

        Checks: high >= open, high >= close, low <= open, low <= close,
        high >= low (DS-02 Stage 1 table).

        Returns
        -------
        bool
            True if all rows satisfy every OHLC relationship.
        """
        violations = (
            (df["high"] < df["open"])
            | (df["high"] < df["close"])
            | (df["low"] > df["open"])
            | (df["low"] > df["close"])
            | (df["high"] < df["low"])
        )
        return not violations.any()

    def check_positive_prices(self, df: pd.DataFrame) -> bool:
        """Verify open, high, low, close are all strictly positive."""
        return bool((df[["open", "high", "low", "close"]] > 0).all().all())

    def check_non_negative_volume(self, df: pd.DataFrame) -> bool:
        """Verify volume is non-negative for every row."""
        return bool((df["volume"] >= 0).all())

    def check_nan_inf(self, df: pd.DataFrame) -> tuple[bool, dict[str, int]]:
        """
        Verify no NaN or Inf values exist in any numeric column.

        Returns
        -------
        tuple[bool, dict[str, int]]
            (passed, per-column NaN+Inf counts). `passed` is True only
            if every column has zero NaN and zero Inf values.
        """
        numeric_cols = ["open", "high", "low", "close", "volume"]
        counts: dict[str, int] = {}
        for col in numeric_cols:
            nan_count = df[col].isna().sum()
            inf_count = np.isinf(df[col].to_numpy(dtype="float64")).sum()
            total = int(nan_count + inf_count)
            if total > 0:
                counts[col] = total
        return (len(counts) == 0, counts)

    def check_timezone(self, df: pd.DataFrame) -> bool:
        """
        Verify `timestamp` is UTC-localized (not naive, not another zone).

        Returns
        -------
        bool
            True if the timestamp dtype has tz info equal to UTC.
        """
        dtype = df["timestamp"].dtype
        if not isinstance(dtype, pd.DatetimeTZDtype):
            return False
        return str(dtype.tz) == "UTC"

    def check_date_coverage(
        self, df: pd.DataFrame, start: str, end: str
    ) -> bool:
        """
        Verify the data covers the full required study period.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain a `timestamp` column, assumed sorted.
        start : str
            Study start date "YYYY-MM-DD" — first timestamp must be
            ≤ this date at 00:00 UTC.
        end : str
            Study end date "YYYY-MM-DD" — last timestamp must be
            ≥ this date at 23:00 UTC (per DS-02 Stage 1 table).

        Returns
        -------
        bool
            True if first timestamp ≤ start and last timestamp ≥ end
            (with end interpreted as 23:00 UTC on that date).
        """
        start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(end, "%Y-%m-%d").replace(
            tzinfo=timezone.utc, hour=23, minute=0, second=0
        )
        first_ts = df["timestamp"].iloc[0].to_pydatetime()
        last_ts = df["timestamp"].iloc[-1].to_pydatetime()
        return first_ts <= start_dt and last_ts >= end_dt

    def check_gap_ratio(self, df: pd.DataFrame, timeframe: str) -> float:
        """
        Compute the fraction of expected candles that are missing.

        Per DS-02 Stage 1: "Missing candle detection: gap between
        consecutive timestamps equals exactly 1 timeframe unit; log
        warning for each gap; abort if total missing > 5%."

        Parameters
        ----------
        df : pd.DataFrame
            Must contain a sorted `timestamp` column.
        timeframe : str
            One of "15m", "1h", "4h", "1d".

        Returns
        -------
        float
            Ratio of missing candles to total expected candles in the
            observed date range. 0.0 means no gaps at all.

        Raises
        ------
        ValueError
            If `timeframe` is not recognized.
        """
        if timeframe not in TIMEFRAME_DURATIONS:
            raise ValueError(
                f"Unknown timeframe '{timeframe}'. "
                f"Expected one of {list(TIMEFRAME_DURATIONS)}."
            )

        duration = TIMEFRAME_DURATIONS[timeframe]
        deltas = df["timestamp"].diff().dropna()

        if len(deltas) == 0:
            return 0.0

        # Number of "extra" candle-durations beyond the expected single
        # step, summed across all gaps. E.g. a gap of 3x duration means
        # 2 missing candles.
        missing_candles = ((deltas / duration).round().astype("int64") - 1).clip(lower=0).sum()

        expected_total = len(df) + missing_candles
        if expected_total == 0:
            return 0.0

        gap_ratio = missing_candles / expected_total
        if missing_candles > 0:
            logger.warning(
                "Detected %d missing candle(s) for timeframe=%s (gap_ratio=%.4f)",
                missing_candles,
                timeframe,
                gap_ratio,
            )
        return float(gap_ratio)

    def check_max_single_gap(self, df: pd.DataFrame, timeframe: str) -> bool:
        """
        Verify no single gap exceeds one candle duration by more than one unit.

        Per DS-04 V-DATA-001 pass criteria: "No single gap exceeds one
        candle duration by more than one unit." This is stricter than
        the aggregate `check_gap_ratio` — it catches one large gap even
        if the overall gap ratio is within the 5% tolerance.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain a sorted `timestamp` column.
        timeframe : str
            One of "15m", "1h", "4h", "1d".

        Returns
        -------
        bool
            True if the largest single gap is at most 2x the expected
            candle duration (i.e., at most one missing candle between
            any two consecutive observed timestamps).
        """
        duration = TIMEFRAME_DURATIONS[timeframe]
        deltas = df["timestamp"].diff().dropna()
        if len(deltas) == 0:
            return True
        max_gap_in_units = (deltas / duration).max()
        return bool(max_gap_in_units <= 2.0)

    def validate_timeframe(
        self,
        df: pd.DataFrame,
        timeframe: str,
        start_date: str = "2020-01-01",
        end_date: str = "2023-12-31",
        raise_on_failure: bool = True,
    ) -> ValidationReport:
        """
        Run the complete DS-02 Stage 1 validation suite for one timeframe.

        Parameters
        ----------
        df : pd.DataFrame
            Raw OHLCV data for this timeframe (as produced by M1).
        timeframe : str
            One of "15m", "1h", "4h", "1d".
        start_date : str, optional
            Study start date, default "2020-01-01".
        end_date : str, optional
            Study end date, default "2023-12-31".
        raise_on_failure : bool, optional
            If True (default), raise `DataValidationError` when any
            check fails. If False, return the report without raising
            (used by tests that want to inspect failures directly).

        Returns
        -------
        ValidationReport
            Full report of every check performed.

        Raises
        ------
        DataValidationError
            If `raise_on_failure=True` and any check fails.
        """
        report = ValidationReport(timeframe=timeframe)

        columns_check = self.check_columns_present(df)
        report.checks.append(columns_check)

        if not columns_check.passed:
            # Cannot run remaining checks without required columns.
            if raise_on_failure:
                raise DataValidationError(report.summary())
            return report

        report.checks.append(
            CheckResult("monotonicity", self.check_monotonicity(df))
        )
        report.checks.append(
            CheckResult("no_duplicate_timestamps", self.check_duplicates(df))
        )
        report.checks.append(
            CheckResult("ohlc_integrity", self.check_ohlc_integrity(df))
        )
        report.checks.append(
            CheckResult("positive_prices", self.check_positive_prices(df))
        )
        report.checks.append(
            CheckResult("non_negative_volume", self.check_non_negative_volume(df))
        )

        nan_inf_passed, nan_inf_counts = self.check_nan_inf(df)
        report.checks.append(
            CheckResult(
                "no_nan_or_inf",
                nan_inf_passed,
                detail=f"counts by column: {nan_inf_counts}" if nan_inf_counts else "",
            )
        )

        report.checks.append(
            CheckResult("timezone_utc", self.check_timezone(df))
        )
        report.checks.append(
            CheckResult(
                "date_range_coverage",
                self.check_date_coverage(df, start_date, end_date),
                detail=f"expected [{start_date}, {end_date}], "
                f"got [{df['timestamp'].iloc[0]}, {df['timestamp'].iloc[-1]}]",
            )
        )

        gap_ratio = self.check_gap_ratio(df, timeframe)
        report.gap_ratio = gap_ratio
        report.checks.append(
            CheckResult(
                "gap_ratio_within_tolerance",
                gap_ratio <= GAP_RATIO_TOLERANCE,
                detail=f"gap_ratio={gap_ratio:.4%}, tolerance={GAP_RATIO_TOLERANCE:.0%}",
            )
        )
        report.checks.append(
            CheckResult(
                "no_excessive_single_gap",
                self.check_max_single_gap(df, timeframe),
            )
        )

        if report.passed:
            logger.info("Validation PASSED for timeframe=%s", timeframe)
        else:
            logger.error(
                "Validation FAILED for timeframe=%s: %d/%d checks failed",
                timeframe,
                len(report.failed_checks),
                len(report.checks),
            )
            if raise_on_failure:
                raise DataValidationError(report.summary())

        return report


def validate_all_timeframes(
    dataframes: dict[str, pd.DataFrame],
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
    raise_on_failure: bool = True,
) -> dict[str, ValidationReport]:
    """
    Run `DataValidator.validate_timeframe` across all provided timeframes.

    Parameters
    ----------
    dataframes : dict[str, pd.DataFrame]
        Mapping of timeframe -> raw OHLCV DataFrame (as loaded from M1
        Parquet outputs).
    start_date : str, optional
        Study start date, default "2020-01-01".
    end_date : str, optional
        Study end date, default "2023-12-31".
    raise_on_failure : bool, optional
        If True (default), raise `DataValidationError` on the first
        timeframe that fails validation.

    Returns
    -------
    dict[str, ValidationReport]
        One report per timeframe. If `raise_on_failure=True`, this
        return value is only reached if every timeframe passed.

    Raises
    ------
    DataValidationError
        If `raise_on_failure=True` and any timeframe fails.
    """
    validator = DataValidator()
    reports: dict[str, ValidationReport] = {}

    for tf, df in dataframes.items():
        reports[tf] = validator.validate_timeframe(
            df,
            tf,
            start_date=start_date,
            end_date=end_date,
            raise_on_failure=raise_on_failure,
        )

    return reports


def append_validation_to_manifest(
    manifest: dict, reports: dict[str, ValidationReport]
) -> dict:
    """
    Append a validation report section to an existing manifest dict.

    Per IMP-01 M2 Outputs: "Validation report appended to
    `data/raw/manifest.json`."

    Parameters
    ----------
    manifest : dict
        The manifest dict loaded/built by M1 (`src.data.acquisition`).
    reports : dict[str, ValidationReport]
        Validation reports produced by `validate_all_timeframes`.

    Returns
    -------
    dict
        The same manifest dict, mutated in place, with a new
        `"validation"` key containing per-timeframe pass/fail status
        and gap ratios.
    """
    manifest["validation"] = {
        tf: {
            "passed": report.passed,
            "gap_ratio": report.gap_ratio,
            "checks": {c.name: c.passed for c in report.checks},
        }
        for tf, report in reports.items()
    }
    return manifest
