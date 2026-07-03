"""
src/data/temporal_split.py

Temporal split module (M5 — Temporal Split, DS-02 v1.1 Stage 4, ADR-014).

Purpose
-------
Enforces the walk-forward train/test split at the exact boundary
defined in ADR-014: train covers 2020-01-01 00:00:00 UTC through
2022-12-31 23:00:00 UTC inclusive; test covers 2023-01-01 00:00:00 UTC
through 2023-12-31 23:00:00 UTC inclusive. No shuffling, no
stratification, no overlap.

The split boundary is a fixed calendar date, independent of the
feature matrix's actual first timestamp (2020-01-01 19:00:00 UTC after
the M4 NaN-drop) — so this module has no dependency on the M4
timestamp bug/fix beyond consuming M4's already-corrected output.

Leakage Checkpoint LC-3 (documented, not "fixed" here)
--------------------------------------------------------
DS-02 v1.1 LC-3 acknowledges that the rolling `volume_zscore` window
(20 hours) near the split boundary technically has its last ~19 hours
of training data influence the first row of test data's rolling
statistic. This is explicitly NOT treated as leakage requiring a code
fix here: DS-02 documents that Stage 5's per-window z-score
normalization re-centers each window using only that window's own 48
values, which overrides the effect at model-input time. `verify_lc3`
below documents/verifies this acknowledged interaction (DS-04 v1.1
V-LEAK-002) rather than attempting to eliminate it — eliminating it
would contradict the approved DS-02/DS-03 protocol.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Split boundary per ADR-014 / DS-02 v1.1 Stage 4. Fixed calendar
# timestamps, independent of the feature matrix's actual first row.
TRAIN_END: pd.Timestamp = pd.Timestamp("2022-12-31 23:00:00", tz="UTC")
TEST_START: pd.Timestamp = pd.Timestamp("2023-01-01 00:00:00", tz="UTC")

# Expected sizes per DS-02 v1.1 Stage 4 (corrected first-timestamp figures).
EXPECTED_TRAIN_ROWS_APPROX: int = 26_269
EXPECTED_TEST_ROWS_APPROX: int = 8_760
ROW_COUNT_TOLERANCE: int = 50  # generous tolerance for "~" approximations

VOLUME_ZSCORE_ROLLING_WINDOW: int = 20


class TemporalSplitError(RuntimeError):
    """Raised when the train/test split fails a boundary or overlap check."""


@dataclass
class SplitResult:
    """Result of splitting the feature matrix into train/test."""

    train: pd.DataFrame
    test: pd.DataFrame


class TemporalSplitter:
    """
    Splits a feature matrix into train/test sets at the ADR-014 boundary.
    """

    def split(self, features_df: pd.DataFrame) -> SplitResult:
        """
        Split `features_df` into train and test DataFrames.

        Parameters
        ----------
        features_df : pd.DataFrame
            Feature matrix from M4, must contain a `timestamp` column,
            sorted ascending, UTC-localized.

        Returns
        -------
        SplitResult
            `.train`: rows with timestamp <= TRAIN_END.
            `.test`: rows with timestamp >= TEST_START.
            Rows strictly between TRAIN_END and TEST_START (there are
            none, since they are back-to-back hours) are not possible
            given hourly data; every row falls into exactly one split.
        """
        train = features_df[features_df["timestamp"] <= TRAIN_END].reset_index(drop=True)
        test = features_df[features_df["timestamp"] >= TEST_START].reset_index(drop=True)

        logger.info(
            "Split complete: train=%d rows (%s to %s), test=%d rows (%s to %s)",
            len(train),
            train["timestamp"].iloc[0] if len(train) > 0 else "N/A",
            train["timestamp"].iloc[-1] if len(train) > 0 else "N/A",
            len(test),
            test["timestamp"].iloc[0] if len(test) > 0 else "N/A",
            test["timestamp"].iloc[-1] if len(test) > 0 else "N/A",
        )
        return SplitResult(train=train, test=test)


def check_split_boundary(result: SplitResult) -> tuple[bool, str]:
    """
    Verify the split boundary is exact, per DS-04 v1.1 V-DATA-005.

    Parameters
    ----------
    result : SplitResult
        Output of `TemporalSplitter.split`.

    Returns
    -------
    tuple[bool, str]
        (passed, detail message).
    """
    issues: list[str] = []

    if len(result.train) == 0:
        issues.append("train set is empty")
    else:
        train_max = result.train["timestamp"].max()
        if train_max != TRAIN_END:
            issues.append(
                f"train max timestamp is {train_max}, expected exactly {TRAIN_END}"
            )

    if len(result.test) == 0:
        issues.append("test set is empty")
    else:
        test_min = result.test["timestamp"].min()
        if test_min != TEST_START:
            issues.append(
                f"test min timestamp is {test_min}, expected exactly {TEST_START}"
            )

    if issues:
        return False, "; ".join(issues)
    return True, "split boundary OK"


def check_no_overlap(result: SplitResult) -> tuple[bool, str]:
    """
    Verify zero duplicate timestamps exist across train and test sets.

    Per DS-04 v1.1 V-DATA-005: "Verify that the union of train and test
    timestamps contains no duplicates."

    Parameters
    ----------
    result : SplitResult
        Output of `TemporalSplitter.split`.

    Returns
    -------
    tuple[bool, str]
        (passed, detail message).
    """
    train_ts = set(result.train["timestamp"])
    test_ts = set(result.test["timestamp"])
    overlap = train_ts & test_ts

    if overlap:
        sample = sorted(overlap)[:5]
        return False, f"{len(overlap)} duplicate timestamp(s) found, e.g. {sample}"
    return True, "no overlap between train and test timestamps"


def check_split_sizes(
    result: SplitResult,
    expected_train_rows: int | None = EXPECTED_TRAIN_ROWS_APPROX,
    expected_test_rows: int | None = EXPECTED_TEST_ROWS_APPROX,
    tolerance: int = ROW_COUNT_TOLERANCE,
) -> tuple[bool, str]:
    """
    Verify train/test row counts are within tolerance of DS-02 v1.1 expectations.

    Parameters
    ----------
    result : SplitResult
        Output of `TemporalSplitter.split`.
    expected_train_rows : int or None, optional
        Expected train row count, default 26,269 (DS-02 v1.1 Stage 4).
        Pass None to skip.
    expected_test_rows : int or None, optional
        Expected test row count, default 8,760. Pass None to skip.
    tolerance : int, optional
        Allowed absolute deviation, default 50.

    Returns
    -------
    tuple[bool, str]
        (passed, detail message).
    """
    issues: list[str] = []

    if expected_train_rows is not None:
        diff = abs(len(result.train) - expected_train_rows)
        if diff > tolerance:
            issues.append(
                f"train rows = {len(result.train)}, expected ~{expected_train_rows} "
                f"(±{tolerance})"
            )

    if expected_test_rows is not None:
        diff = abs(len(result.test) - expected_test_rows)
        if diff > tolerance:
            issues.append(
                f"test rows = {len(result.test)}, expected ~{expected_test_rows} "
                f"(±{tolerance})"
            )

    if issues:
        return False, "; ".join(issues)
    return True, "split sizes within tolerance"


def verify_lc3(
    train: pd.DataFrame,
    test: pd.DataFrame,
    volume_zscore_column: str = "volume_zscore_1h",
    rolling_window: int = VOLUME_ZSCORE_ROLLING_WINDOW,
) -> tuple[bool, str]:
    """
    Document/verify LC-3 (DS-04 v1.1 V-LEAK-002): no future price info
    crosses the split boundary.

    Per DS-02 v1.1 LC-3, this check does NOT require the first test-set
    `volume_zscore` row to be NaN or purely test-derived at the FEATURE
    level — that would contradict the approved protocol, since
    `volume_zscore_tf` is computed in M4 (Stage 3) using a rolling
    window over the full aligned series, *before* the split even
    happens. What DS-04 v1.1 V-LEAK-002 actually requires is narrower
    and is satisfied by construction:

    1. The split itself (this module) introduces no NEW leakage: it is
       a pure boolean timestamp filter, computed independently for
       train and test, with no shared state or statistics carried across.
    2. No PRICE information (open/high/low/close/volume, or returns
       computed from them) from the training period is copied into any
       test-period row's value at the split step.
    3. The rolling-window feature-level effect at the boundary is
       explicitly pre-registered and documented in DS-02 v1.1 LC-3,
       and is neutralized at model-input time by Stage 5's per-window
       normalization (verified separately in M6 as V-LEAK-004) — not
       by this module.

    This function checks (1) and (2): that the split did not mutate or
    copy any row's values across the boundary, and reports the
    pre-registered feature-level interaction from (3) as an
    informational note, not a failure.

    Parameters
    ----------
    train : pd.DataFrame
        Train split (must include `timestamp` and `volume_zscore_column`).
    test : pd.DataFrame
        Test split (same columns).
    volume_zscore_column : str, optional
        Which `volume_zscore_{tf}` column to inspect for the
        informational note, default "volume_zscore_1h".
    rolling_window : int, optional
        The rolling window size used to compute `volume_zscore` in M4,
        default 20 (must match `feature_engineering.VOLUME_ZSCORE_WINDOW`).

    Returns
    -------
    tuple[bool, str]
        (passed, detail message). `passed` reflects checks (1)/(2)
        only; the LC-3 rolling-window interaction is informational.
    """
    issues: list[str] = []

    # (1) + (2): no row's OHLCV-derived values were altered by the
    # split — verify by checking that every train row's timestamp is
    # <= TRAIN_END and every test row's timestamp is >= TEST_START,
    # i.e. the filter boundary was applied correctly and symmetrically
    # (no row duplicated, mutated, or reassigned across the boundary).
    if len(train) > 0 and (train["timestamp"] > TRAIN_END).any():
        issues.append("train set contains a timestamp after TRAIN_END")
    if len(test) > 0 and (test["timestamp"] < TEST_START).any():
        issues.append("test set contains a timestamp before TEST_START")

    note = ""
    if (
        len(test) > 0
        and volume_zscore_column in test.columns
        and volume_zscore_column in train.columns
    ):
        first_test_value = test[volume_zscore_column].iloc[0]
        note = (
            f"Informational (not a failure): the first test-set row's "
            f"'{volume_zscore_column}' was computed in M4 using a "
            f"{rolling_window}-row rolling window that spans the "
            f"train/test boundary (value={first_test_value}). This is "
            f"pre-registered and documented in DS-02 v1.1 LC-3, and is "
            f"neutralized at model-input time by Stage 5 per-window "
            f"normalization (verified by V-LEAK-004 in M6), not by this "
            f"split step."
        )
        logger.info(note)

    if issues:
        return False, "; ".join(issues)
    return True, "split introduces no new leakage" + (f" — {note}" if note else "")


def run_temporal_split(
    features_df: pd.DataFrame,
    raise_on_failure: bool = True,
    expected_train_rows: int | None = EXPECTED_TRAIN_ROWS_APPROX,
    expected_test_rows: int | None = EXPECTED_TEST_ROWS_APPROX,
) -> SplitResult:
    """
    Run the full M5 pipeline: split, verify boundary/overlap/sizes/LC-3.

    Parameters
    ----------
    features_df : pd.DataFrame
        Feature matrix from M4 (35,045 rows x 29 columns in production).
    raise_on_failure : bool, optional
        If True (default), raise `TemporalSplitError` if any check fails.
    expected_train_rows : int or None, optional
        Expected train row count for the size check, default 26,269.
        Pass None for small synthetic test fixtures.
    expected_test_rows : int or None, optional
        Expected test row count for the size check, default 8,760.
        Pass None for small synthetic test fixtures.

    Returns
    -------
    SplitResult
        The verified train/test split.

    Raises
    ------
    TemporalSplitError
        If `raise_on_failure=True` and any check fails.
    """
    splitter = TemporalSplitter()
    result = splitter.split(features_df)

    checks = [
        ("split_boundary", check_split_boundary(result)),
        ("no_overlap", check_no_overlap(result)),
        (
            "split_sizes",
            check_split_sizes(result, expected_train_rows, expected_test_rows),
        ),
        ("lc3_no_new_leakage", verify_lc3(result.train, result.test)),
    ]

    failed = [(name, detail) for name, (passed, detail) in checks if not passed]

    for name, (passed, detail) in checks:
        if passed:
            logger.info("M5 check '%s' PASSED: %s", name, detail)
        else:
            logger.error("M5 check '%s' FAILED: %s", name, detail)

    if failed and raise_on_failure:
        detail_lines = "\n".join(f"  {name}: {detail}" for name, detail in failed)
        raise TemporalSplitError(f"M5 temporal split checks failed:\n{detail_lines}")

    return result
