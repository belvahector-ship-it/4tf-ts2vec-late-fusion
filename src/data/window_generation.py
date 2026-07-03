"""
src/data/window_generation.py

Window generation module (M6 — Window Generation, DS-02 v1.2 Stage 5,
ADR-006, ADR-016).

Purpose
-------
Converts the FULL feature matrix (M4 output, `btc_features_all.parquet`,
35,045 rows) into 3D NumPy tensors of sliding windows, with per-window
z-score normalization, ready for TS2Vec DataLoader consumption. This is
the last module in the data pipeline before branch training (M8) — and
the last opportunity to catch a leakage bug before 20 branch-training
runs would otherwise need to be discarded (IMP-01 Coding Order rationale).

DESIGN (DS-02 v1.2, corrected from v1.1 — see AUDIT_LC4_ADDENDUM.md)
----------------------------------------------------------------------
Windows are generated ONCE over the full 35,045-row feature matrix, and
are THEN categorized as train/test based on each window's **anchor
timestamp** (the timestamp of its LAST row — the same anchor used to
join embeddings back to OHLCV in Stage 9), using the identical ADR-014
boundary as M5:

    anchor <= 2022-12-31 23:00:00 UTC  -> train
    anchor >= 2023-01-01 00:00:00 UTC  -> test

This is NOT the same as generating windows separately from M5's
already-split `train_features.parquet` / `test_features.parquet` files
(that alternative design was shown to be internally inconsistent with
DS-02 v1.1's own LC-4 paragraph and with DS-03/DS-04's explicit "up to
47 overlap windows" pass criteria — see AUDIT_LC4_ADDENDUM.md for the
full arithmetic proof). `train_features.parquet` / `test_features.parquet`
(M5 output) remain valid, useful artifacts — for audit purposes and for
Stage 9 economic-validity joins — but are NOT the input to this module.

Why this does not leak future information (INV-006)
------------------------------------------------------
A window only ever looks BACKWARD from its anchor. Therefore:
  - A TRAIN window (anchor <= TRAIN_END) can never contain a row after
    TRAIN_END, since every row in the window is chronologically at or
    before the anchor. Leakage into train windows from the test period
    is structurally impossible.
  - A TEST window's EARLIEST row can carry a timestamp from the
    training period (this happens for the first ~47 test windows,
    whose anchors are close to TEST_START) — but this is *past*
    information relative to the window's anchor, which is exactly what
    self-supervised sliding-window learning is designed to consume. It
    is not *future* information relative to any row inside the window.
    This is DS-02 v1.2 LC-4, corroborated by DS-03 v1.2 §3 CF-002 and
    §5 Protocol B, and DS-04 v1.1 V-LEAK-003 (which explicitly expects
    and requires documenting up to 47 such windows as a PASS criterion).
  - Per-window normalization (V-LEAK-004) uses ONLY that window's own
    48 rows regardless of which calendar period those rows belong to
    — no external or global statistic is ever used, so the LC-4
    overlap does not introduce any new normalization leakage either.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Feature columns extracted per timeframe, in fixed order (ADR-006 / DS-02 v1.2 Stage 5).
FEATURE_COLUMN_ORDER: tuple[str, ...] = (
    "open_return",
    "high_return",
    "low_return",
    "close_return",
    "volume_zscore",
    "hl_range",
    "body_ratio",
)

TIMEFRAME_SUFFIXES: tuple[str, ...] = ("15m", "1h", "4h", "1d")

WINDOW_SIZE: int = 48  # ADR-006, W=48
STRIDE: int = 1  # ADR-006, stride=1
NORMALIZATION_EPSILON: float = 1e-8  # ADR-016

# Split boundary, identical to M5 (ADR-014).
TRAIN_END: pd.Timestamp = pd.Timestamp("2022-12-31 23:00:00", tz="UTC")
TEST_START: pd.Timestamp = pd.Timestamp("2023-01-01 00:00:00", tz="UTC")

# Expected window counts per DS-02 v1.2 Stage 5 (corrected).
EXPECTED_N_TRAIN_WINDOWS_APPROX: int = 26_222
EXPECTED_N_TEST_WINDOWS: int = 8_760  # exact, not approximate — see module docstring
WINDOW_COUNT_TOLERANCE: int = 50

# LC-4: up to this many test windows may have an earliest timestamp
# before TEST_START, due to the backward-looking window. This is
# EXPECTED and REQUIRED behavior under DS-02 v1.2 / DS-03 v1.2 / DS-04
# v1.1 — not a leakage bug.
EXPECTED_MAX_BOUNDARY_OVERLAP_WINDOWS: int = 47


class WindowGenerationError(RuntimeError):
    """Raised when window generation produces an invalid tensor or fails a leakage check."""


@dataclass
class WindowSet:
    """
    Windowed tensors and timestamps for one category (train or test),
    for one timeframe.

    Attributes
    ----------
    windows : np.ndarray
        Shape [N_windows, 48, 7], dtype float32, already per-window
        z-score normalized.
    anchor_timestamps : np.ndarray
        Shape [N_windows], dtype int64 (Unix nanoseconds UTC). The
        timestamp of the LAST row in each window — used both to
        categorize the window as train/test and to join embeddings
        back to OHLCV in Stage 9.
    earliest_timestamps : np.ndarray
        Shape [N_windows], dtype int64 (Unix nanoseconds UTC). The
        timestamp of the FIRST row in each window. Not saved to disk,
        but used internally by `count_boundary_overlap_windows` for
        LC-4 verification.
    """

    windows: np.ndarray
    anchor_timestamps: np.ndarray
    earliest_timestamps: np.ndarray


class WindowGenerator:
    """
    Generates sliding windows (over the FULL feature matrix) with
    per-window z-score normalization, and categorizes them by anchor
    timestamp into train/test.
    """

    def _to_int64_ns(self, timestamp_series: pd.Series) -> np.ndarray:
        """
        Convert a tz-aware timestamp Series to an int64 Unix-nanoseconds array.

        Explicitly pins the datetime unit to nanoseconds before
        extracting the integer representation. This is required for
        correctness across pandas versions: tz-aware columns may be
        stored internally as `datetime64[us]` or `datetime64[ns]`
        depending on the pandas version, and naively casting straight
        to int64 without pinning the unit first silently produces
        values in the wrong unit (a bug caught during real verification
        of this module — see repository checkpoint history).

        Parameters
        ----------
        timestamp_series : pd.Series
            A tz-aware datetime Series.

        Returns
        -------
        np.ndarray
            dtype int64, Unix nanoseconds UTC — directly comparable to
            `pd.Timestamp(...).value`.
        """
        return timestamp_series.astype("datetime64[ns, UTC]").astype("int64").to_numpy()

    def extract_windows(
        self, features_df: pd.DataFrame, tf_suffix: str
    ) -> WindowSet:
        """
        Extract raw (unnormalized) sliding windows over the FULL feature matrix.

        Parameters
        ----------
        features_df : pd.DataFrame
            The FULL feature matrix (M4 output, all 35,045 rows in
            production — NOT pre-split into train/test). Must contain
            `timestamp` and all 7 `{feature}_{tf_suffix}` columns.
        tf_suffix : str
            One of "15m", "1h", "4h", "1d".

        Returns
        -------
        WindowSet
            `.windows` shape [N, 48, 7], NOT yet normalized (raw
            feature values), where N spans the ENTIRE matrix (both
            train- and test-period windows together). Categorization
            into train/test happens in `categorize_by_anchor`, not here.

        Raises
        ------
        ValueError
            If `features_df` has fewer than WINDOW_SIZE rows.
        """
        cols = [f"{feat}_{tf_suffix}" for feat in FEATURE_COLUMN_ORDER]
        values = features_df[cols].to_numpy(dtype="float32")
        timestamps = self._to_int64_ns(features_df["timestamp"])

        n_rows = len(features_df)
        if n_rows < WINDOW_SIZE:
            raise ValueError(
                f"Cannot extract windows: only {n_rows} rows available, "
                f"need at least {WINDOW_SIZE} (W={WINDOW_SIZE})."
            )

        n_windows = (n_rows - WINDOW_SIZE) // STRIDE + 1
        windows = np.empty((n_windows, WINDOW_SIZE, len(FEATURE_COLUMN_ORDER)), dtype="float32")
        anchor_ts = np.empty(n_windows, dtype="int64")
        earliest_ts = np.empty(n_windows, dtype="int64")

        for i in range(n_windows):
            start = i * STRIDE
            end = start + WINDOW_SIZE
            windows[i] = values[start:end]
            anchor_ts[i] = timestamps[end - 1]
            earliest_ts[i] = timestamps[start]

        return WindowSet(windows=windows, anchor_timestamps=anchor_ts, earliest_timestamps=earliest_ts)

    def normalize_window(self, window: np.ndarray, epsilon: float = NORMALIZATION_EPSILON) -> np.ndarray:
        """
        Apply per-window z-score normalization to a single window.

        Parameters
        ----------
        window : np.ndarray
            Shape [48, 7].
        epsilon : float, optional
            Denominator epsilon, default 1e-8 per ADR-016.

        Returns
        -------
        np.ndarray
            Shape [48, 7], normalized: (window - mean) / (std + epsilon),
            computed column-wise (per feature) using ONLY this window's
            48 rows — regardless of which calendar period those rows
            belong to (see module docstring re: LC-4).
        """
        mu = window.mean(axis=0)
        sig = window.std(axis=0)
        return (window - mu) / (sig + epsilon)

    def normalize_all_windows(
        self, windows: np.ndarray, epsilon: float = NORMALIZATION_EPSILON
    ) -> np.ndarray:
        """
        Apply per-window z-score normalization to a full window tensor.

        Parameters
        ----------
        windows : np.ndarray
            Shape [N, 48, 7], raw (unnormalized) windows.
        epsilon : float, optional
            Denominator epsilon, default 1e-8.

        Returns
        -------
        np.ndarray
            Shape [N, 48, 7], float32, each window independently
            normalized using only its own 48 rows (vectorized
            equivalent of calling `normalize_window` per window).
        """
        mu = windows.mean(axis=1, keepdims=True)  # [N, 1, 7]
        sig = windows.std(axis=1, keepdims=True)  # [N, 1, 7]
        return ((windows - mu) / (sig + epsilon)).astype("float32")

    def categorize_by_anchor(self, window_set: WindowSet) -> tuple[WindowSet, WindowSet]:
        """
        Split a WindowSet into train/test based on each window's anchor timestamp.

        Per DS-02 v1.2 Stage 5: anchor <= TRAIN_END -> train;
        anchor >= TEST_START -> test. Since the data is hourly and
        TRAIN_END/TEST_START are exactly 1 hour apart, every window's
        anchor falls into exactly one category (no window is dropped
        or double-counted).

        Parameters
        ----------
        window_set : WindowSet
            Output of `extract_windows` (or its normalized version),
            spanning the full feature matrix.

        Returns
        -------
        tuple[WindowSet, WindowSet]
            (train_window_set, test_window_set).
        """
        train_end_ns = TRAIN_END.value
        test_start_ns = TEST_START.value

        train_mask = window_set.anchor_timestamps <= train_end_ns
        test_mask = window_set.anchor_timestamps >= test_start_ns

        train_ws = WindowSet(
            windows=window_set.windows[train_mask],
            anchor_timestamps=window_set.anchor_timestamps[train_mask],
            earliest_timestamps=window_set.earliest_timestamps[train_mask],
        )
        test_ws = WindowSet(
            windows=window_set.windows[test_mask],
            anchor_timestamps=window_set.anchor_timestamps[test_mask],
            earliest_timestamps=window_set.earliest_timestamps[test_mask],
        )
        return train_ws, test_ws

    def generate(self, features_df: pd.DataFrame, tf_suffix: str) -> tuple[WindowSet, WindowSet]:
        """
        Full M6 pipeline for one timeframe: extract, normalize, categorize.

        Parameters
        ----------
        features_df : pd.DataFrame
            The FULL feature matrix (M4 output).
        tf_suffix : str
            One of "15m", "1h", "4h", "1d".

        Returns
        -------
        tuple[WindowSet, WindowSet]
            (train_window_set, test_window_set), both normalized.
        """
        raw = self.extract_windows(features_df, tf_suffix)
        normalized_windows = self.normalize_all_windows(raw.windows)
        normalized_set = WindowSet(
            windows=normalized_windows,
            anchor_timestamps=raw.anchor_timestamps,
            earliest_timestamps=raw.earliest_timestamps,
        )
        return self.categorize_by_anchor(normalized_set)


def check_window_shape(window_set: WindowSet, n_features: int = 7) -> tuple[bool, str]:
    """
    Verify a WindowSet has the expected tensor shape.

    Parameters
    ----------
    window_set : WindowSet
        A train or test WindowSet.
    n_features : int, optional
        Expected feature count, default 7.

    Returns
    -------
    tuple[bool, str]
        (passed, detail message).
    """
    shape = window_set.windows.shape
    if len(shape) != 3:
        return False, f"expected 3D tensor, got shape {shape}"
    if shape[1] != WINDOW_SIZE:
        return False, f"expected window dim {WINDOW_SIZE}, got {shape[1]}"
    if shape[2] != n_features:
        return False, f"expected {n_features} features, got {shape[2]}"
    if window_set.windows.dtype != np.dtype("float32"):
        return False, f"expected dtype float32, got {window_set.windows.dtype}"
    return True, f"shape OK: {shape}, dtype {window_set.windows.dtype}"


def check_window_count(
    window_set: WindowSet,
    expected_count: int | None,
    tolerance: int = WINDOW_COUNT_TOLERANCE,
) -> tuple[bool, str]:
    """
    Verify the number of windows is within tolerance of the expected count.

    Parameters
    ----------
    window_set : WindowSet
        A train or test WindowSet.
    expected_count : int or None
        Expected N_windows. Pass None to skip.
    tolerance : int, optional
        Allowed absolute deviation, default 50.

    Returns
    -------
    tuple[bool, str]
        (passed, detail message).
    """
    if expected_count is None:
        return True, "window count check skipped"
    actual = len(window_set.windows)
    diff = abs(actual - expected_count)
    if diff > tolerance:
        return False, f"expected ~{expected_count} windows (±{tolerance}), got {actual}"
    return True, f"window count OK: {actual} (~{expected_count} expected)"


def verify_train_windows_stay_in_train_period(train_window_set: WindowSet) -> tuple[bool, str]:
    """
    Verify V-LEAK-003: no train window's anchor exceeds TRAIN_END.

    Per DS-04 v1.1 V-LEAK-003 Verification Method: "the latest
    timestamp in any train window does not exceed 2022-12-31 23:00
    UTC." Since a window only looks backward from its anchor, an
    anchor <= TRAIN_END guarantees EVERY row in that window is also
    <= TRAIN_END — so checking the anchor alone is sufficient (and
    equivalent to checking every individual row).

    Parameters
    ----------
    train_window_set : WindowSet
        Output of `WindowGenerator.categorize_by_anchor` (the train half).

    Returns
    -------
    tuple[bool, str]
        (passed, detail message).
    """
    if len(train_window_set.anchor_timestamps) == 0:
        return False, "train window set is empty"

    train_end_ns = TRAIN_END.value
    max_anchor = train_window_set.anchor_timestamps.max()
    if max_anchor > train_end_ns:
        return False, (
            f"train windows contain an anchor timestamp "
            f"({pd.Timestamp(max_anchor, tz='UTC')}) after TRAIN_END ({TRAIN_END})"
        )
    return True, "all train window anchors are within the train period"


def verify_test_windows_anchored_in_test_period(test_window_set: WindowSet) -> tuple[bool, str]:
    """
    Verify every test window's anchor is within the test period.

    This is the complementary check to
    `verify_train_windows_stay_in_train_period`: while a test window's
    EARLIEST row may legitimately fall in the training period (LC-4,
    see `count_boundary_overlap_windows`), its ANCHOR (the row used
    for evaluation and OHLCV joins) must always be within the test
    period — otherwise the window would have been mis-categorized.

    Parameters
    ----------
    test_window_set : WindowSet
        Output of `WindowGenerator.categorize_by_anchor` (the test half).

    Returns
    -------
    tuple[bool, str]
        (passed, detail message).
    """
    if len(test_window_set.anchor_timestamps) == 0:
        return False, "test window set is empty"

    test_start_ns = TEST_START.value
    min_anchor = test_window_set.anchor_timestamps.min()
    if min_anchor < test_start_ns:
        return False, (
            f"test windows contain an anchor timestamp "
            f"({pd.Timestamp(min_anchor, tz='UTC')}) before TEST_START ({TEST_START})"
        )
    return True, "all test window anchors are within the test period"


def count_boundary_overlap_windows(
    test_window_set: WindowSet, expected_max: int = EXPECTED_MAX_BOUNDARY_OVERLAP_WINDOWS
) -> tuple[bool, str, int]:
    """
    Count and verify test windows whose earliest timestamp precedes
    TEST_START (DS-02 v1.2 LC-4, DS-04 v1.1 V-LEAK-003).

    This is NOT a leakage bug — it is the documented, intentional,
    REQUIRED consequence of a backward-looking sliding window near the
    split boundary (see module docstring). Every such window's ANCHOR
    remains strictly within the test period (verified separately by
    `verify_test_windows_anchored_in_test_period`); only the window's
    earliest (oldest) row reaches back into the training period, which
    is legitimate past information relative to that window's anchor.

    Parameters
    ----------
    test_window_set : WindowSet
        Output of `WindowGenerator.categorize_by_anchor` (the test half).
    expected_max : int, optional
        Maximum expected count, default 47 (DS-02 v1.2 LC-4 / DS-04
        v1.1 V-LEAK-003).

    Returns
    -------
    tuple[bool, str, int]
        (passed, detail message, actual count). `passed` is True if
        the count is <= expected_max.
    """
    test_start_ns = TEST_START.value
    overlap_count = int((test_window_set.earliest_timestamps < test_start_ns).sum())

    passed = overlap_count <= expected_max
    detail = (
        f"{overlap_count} test window(s) have an earliest timestamp before "
        f"{TEST_START} (expected: up to {expected_max}, per DS-02 v1.2 LC-4 — "
        f"this is REQUIRED/expected behavior, not a leakage bug)"
    )
    return passed, detail, overlap_count


def verify_per_window_normalization(
    raw_window: np.ndarray, normalized_window: np.ndarray, epsilon: float = NORMALIZATION_EPSILON
) -> tuple[bool, str]:
    """
    Verify V-LEAK-004: per-window normalization uses only that window's
    own values, no external/global statistics.

    Per DS-04 v1.1 V-LEAK-004: "manually compute the mean and standard
    deviation from the raw window values and verify they match the
    normalization statistics used during processing." This holds
    identically for boundary-overlapping and non-overlapping windows —
    normalization never looks outside the window's own 48 rows.

    Parameters
    ----------
    raw_window : np.ndarray
        Shape [48, 7], unnormalized feature values for one window.
    normalized_window : np.ndarray
        Shape [48, 7], the normalized version to verify.
    epsilon : float, optional
        Expected denominator epsilon, default 1e-8.

    Returns
    -------
    tuple[bool, str]
        (passed, detail message).
    """
    expected_mu = raw_window.mean(axis=0)
    expected_sig = raw_window.std(axis=0)
    expected_normalized = (raw_window - expected_mu) / (expected_sig + epsilon)

    matches = np.allclose(normalized_window, expected_normalized, rtol=1e-5, atol=1e-6)
    if not matches:
        max_diff = np.abs(normalized_window - expected_normalized).max()
        return False, f"normalized window does not match window-local statistics (max abs diff={max_diff})"
    return True, "normalization uses only this window's own 48 rows"


def run_window_generation(
    features_df: pd.DataFrame,
    timeframes: tuple[str, ...] = TIMEFRAME_SUFFIXES,
    raise_on_failure: bool = True,
    expected_n_train_windows: int | None = EXPECTED_N_TRAIN_WINDOWS_APPROX,
    expected_n_test_windows: int | None = EXPECTED_N_TEST_WINDOWS,
) -> dict[str, dict[str, WindowSet]]:
    """
    Run the full M6 pipeline for all 4 timeframes: generate windows
    over the full feature matrix, normalize, categorize by anchor,
    verify shapes/counts/leakage checks.

    Parameters
    ----------
    features_df : pd.DataFrame
        The FULL feature matrix (M4 output, `btc_features_all.parquet`
        in production — NOT the M5 split files).
    timeframes : tuple[str, ...], optional
        Timeframe suffixes to process, default all 4.
    raise_on_failure : bool, optional
        If True (default), raise `WindowGenerationError` if any check fails.
    expected_n_train_windows : int or None, optional
        Expected train window count, default 26,222. Pass None for
        small synthetic test fixtures.
    expected_n_test_windows : int or None, optional
        Expected test window count, default 8,760 (exact). Pass None
        for small synthetic test fixtures.

    Returns
    -------
    dict[str, dict[str, WindowSet]]
        `{"train": {tf: WindowSet, ...}, "test": {tf: WindowSet, ...}}`.

    Raises
    ------
    WindowGenerationError
        If `raise_on_failure=True` and any check fails.
    """
    generator = WindowGenerator()
    result: dict[str, dict[str, WindowSet]] = {"train": {}, "test": {}}
    all_issues: list[str] = []

    for tf in timeframes:
        train_ws, test_ws = generator.generate(features_df, tf)
        result["train"][tf] = train_ws
        result["test"][tf] = test_ws

        for label, ws in (("train", train_ws), ("test", test_ws)):
            shape_passed, shape_detail = check_window_shape(ws)
            if not shape_passed:
                all_issues.append(f"[{tf}] {label} shape check: {shape_detail}")

        count_passed, count_detail = check_window_count(train_ws, expected_n_train_windows)
        if not count_passed:
            all_issues.append(f"[{tf}] train count check: {count_detail}")
        count_passed_test, count_detail_test = check_window_count(test_ws, expected_n_test_windows)
        if not count_passed_test:
            all_issues.append(f"[{tf}] test count check: {count_detail_test}")

        train_boundary_passed, train_boundary_detail = verify_train_windows_stay_in_train_period(train_ws)
        if not train_boundary_passed:
            all_issues.append(f"[{tf}] V-LEAK-003 train boundary check: {train_boundary_detail}")

        test_boundary_passed, test_boundary_detail = verify_test_windows_anchored_in_test_period(test_ws)
        if not test_boundary_passed:
            all_issues.append(f"[{tf}] V-LEAK-003 test anchor check: {test_boundary_detail}")

        overlap_passed, overlap_detail, overlap_count = count_boundary_overlap_windows(test_ws)
        if not overlap_passed:
            all_issues.append(f"[{tf}] V-LEAK-003 LC-4 overlap count: {overlap_detail}")
        else:
            logger.info("[%s] LC-4 check: %s", tf, overlap_detail)

        logger.info(
            "[%s] Generated train=%d windows, test=%d windows (shape %s)",
            tf,
            len(train_ws.windows),
            len(test_ws.windows),
            train_ws.windows.shape[1:],
        )

    if all_issues:
        detail_lines = "\n".join(f"  {issue}" for issue in all_issues)
        message = f"M6 window generation checks failed:\n{detail_lines}"
        if raise_on_failure:
            raise WindowGenerationError(message)
        logger.error(message)
    else:
        logger.info("M6 window generation: all checks PASSED for all timeframes.")

    return result
