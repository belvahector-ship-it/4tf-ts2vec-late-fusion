"""
src/data/feature_engineering.py

Feature engineering module (M4 — Feature Engineering, DS-02 v1.1 Stage 3,
ADR-015).

Purpose
-------
Computes the 7 OHLCV-derived features (ADR-015) for each of the 4
timeframe column groups in the aligned master DataFrame, independently
and identically. No technical indicators are used — every feature is a
direct, closed-form transformation of open/high/low/close/volume,
chosen specifically to avoid embedding period parameters (e.g. RSI-14)
that would inject implicit temporal smoothing that differs by
timeframe and confounds the experiment (ADR-015 Rationale).

The 7 features (per timeframe suffix `_tf`), per DS-02 Stage 3:

    1. open_return_tf  = (open_tf - prev_close_tf) / prev_close_tf
    2. high_return_tf  = (high_tf - open_tf) / open_tf
    3. low_return_tf   = (low_tf - open_tf) / open_tf
    4. close_return_tf = (close_tf - open_tf) / open_tf
    5. volume_zscore_tf = (volume_tf - rolling_mean20_tf) / rolling_std20_tf
    6. hl_range_tf     = (high_tf - low_tf) / open_tf
    7. body_ratio_tf   = abs(close_tf - open_tf) / (high_tf - low_tf + 1e-8)

NaN handling (per DS-02 v1.1 Stage 3): row 0 has NaN open_return (no
previous close); rows 0-18 have NaN volume_zscore (rolling window of
20 needs 20 observations, so the first 19 are insufficient). Net
effect: rows 0-18 are dropped, leaving 35,064 - 19 = 35,045 rows.

IMPORTANT (DS-02 v1.1 / DS-04 v1.1 correction): the first remaining
timestamp is **2020-01-01 19:00:00 UTC** — i.e. 19 HOURS after the
2020-01-01 00:00 UTC start of the 1h-resolution dataset — NOT
"2020-01-19 00:00 UTC" (19 days). The pipeline is entirely 1-hour
resolution, so dropping 19 rows means dropping 19 hours. The "19 days"
figure that appeared in DS-02 v1.0 / DS-04 v1.0 / IMP-01 v1.1 was an
arithmetic/transcription error, corrected during the 2026-07 DS-01
through DS-04 + IMP-01 logic audit (see AUDIT_REPORT_DS01-DS04_IMP01.md
and each document's v1.1 changelog). This was confirmed by three
independent cross-checks: Train rows ≈ 26,269, N_train_windows ≈
26,222, and the arithmetic 35,064 − 19 = 35,045 all match the "19
hours" interpretation and are off by ~397 under the "19 days"
interpretation. See EXPECTED_FIRST_TIMESTAMP below.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# The 4 timeframe suffixes, matching M3's master DataFrame column naming.
TIMEFRAME_SUFFIXES: tuple[str, ...] = ("15m", "1h", "4h", "1d")

# The 7 feature names (without timeframe suffix), per ADR-015.
FEATURE_NAMES: tuple[str, ...] = (
    "open_return",
    "high_return",
    "low_return",
    "close_return",
    "volume_zscore",
    "hl_range",
    "body_ratio",
)

VOLUME_ZSCORE_WINDOW: int = 20
BODY_RATIO_EPSILON: float = 1e-8

# Expected output schema per DS-02 v1.1 Stage 3 / DS-04 v1.1 V-DATA-004.
EXPECTED_FEATURE_COLUMNS: int = 1 + len(FEATURE_NAMES) * len(TIMEFRAME_SUFFIXES)  # 29
EXPECTED_FEATURE_ROWS: int = 35_045
EXPECTED_NAN_ROWS_DROPPED: int = 19

# CORRECTED per DS-02 v1.1 / DS-04 v1.1: 2020-01-01 00:00 UTC + 19 hours
# = 2020-01-01 19:00:00 UTC (NOT "2020-01-19 00:00 UTC" — that was a
# 19-hours-vs-19-days transcription error in the v1.0 documents).
EXPECTED_FIRST_TIMESTAMP: pd.Timestamp = pd.Timestamp("2020-01-01 19:00:00", tz="UTC")


class FeatureEngineeringError(RuntimeError):
    """Raised when feature engineering produces an invalid feature matrix."""


class FeatureEngineer:
    """
    Computes the 7 ADR-015 features for one or all timeframe suffixes.

    Each `compute_*` method operates on a single timeframe's OHLCV
    columns (e.g. `open_1h`, `close_1h`, ...) from the master DataFrame,
    so the exact same formula is applied identically to all 4
    timeframes — this identical-computation property is itself part of
    what keeps temporal resolution the sole independent variable
    (DS-03 INV-001).
    """

    def compute_open_return(self, df: pd.DataFrame, suffix: str) -> pd.Series:
        """
        open_return = (open_t - prev_close_t) / prev_close_t.

        Parameters
        ----------
        df : pd.DataFrame
            Master DataFrame with columns `open_{suffix}`, `close_{suffix}`.
        suffix : str
            Timeframe suffix, e.g. "1h".

        Returns
        -------
        pd.Series
            Length matches `df`. First value is NaN (no previous close).
        """
        prev_close = df[f"close_{suffix}"].shift(1)
        return (df[f"open_{suffix}"] - prev_close) / prev_close

    def compute_high_return(self, df: pd.DataFrame, suffix: str) -> pd.Series:
        """high_return = (high_t - open_t) / open_t."""
        return (df[f"high_{suffix}"] - df[f"open_{suffix}"]) / df[f"open_{suffix}"]

    def compute_low_return(self, df: pd.DataFrame, suffix: str) -> pd.Series:
        """low_return = (low_t - open_t) / open_t."""
        return (df[f"low_{suffix}"] - df[f"open_{suffix}"]) / df[f"open_{suffix}"]

    def compute_close_return(self, df: pd.DataFrame, suffix: str) -> pd.Series:
        """close_return = (close_t - open_t) / open_t."""
        return (df[f"close_{suffix}"] - df[f"open_{suffix}"]) / df[f"open_{suffix}"]

    def compute_volume_zscore(
        self, df: pd.DataFrame, suffix: str, window: int = VOLUME_ZSCORE_WINDOW
    ) -> pd.Series:
        """
        volume_zscore = (volume_t - rolling_mean_w) / rolling_std_w.

        Per DS-02 Stage 3 "Rolling Window Clarification": the rolling
        window is applied over `window` consecutive rows of the
        ALIGNED 1h-anchor DataFrame (not over each timeframe's native
        candle count). This is intentional and documented — for 4h/1d
        columns (which are forward-filled), the effective look-back in
        real time is larger than for 15m/1h, but this behavior is
        identical across all experimental conditions and is therefore
        not a confound between conditions (only a within-timeframe
        smoothing characteristic).

        Parameters
        ----------
        df : pd.DataFrame
            Master DataFrame with column `volume_{suffix}`.
        suffix : str
            Timeframe suffix.
        window : int, optional
            Rolling window size in aligned-1h rows, default 20.

        Returns
        -------
        pd.Series
            Length matches `df`. First `window - 1` values are NaN
            (insufficient rolling history).
        """
        volume = df[f"volume_{suffix}"]
        rolling_mean = volume.rolling(window=window, min_periods=window).mean()
        rolling_std = volume.rolling(window=window, min_periods=window).std()
        return (volume - rolling_mean) / rolling_std

    def compute_hl_range(self, df: pd.DataFrame, suffix: str) -> pd.Series:
        """hl_range = (high_t - low_t) / open_t."""
        return (df[f"high_{suffix}"] - df[f"low_{suffix}"]) / df[f"open_{suffix}"]

    def compute_body_ratio(
        self, df: pd.DataFrame, suffix: str, epsilon: float = BODY_RATIO_EPSILON
    ) -> pd.Series:
        """
        body_ratio = abs(close_t - open_t) / (high_t - low_t + epsilon).

        The epsilon in the denominator prevents division by zero on
        doji candles (where high == low), per ADR-015.

        Parameters
        ----------
        df : pd.DataFrame
            Master DataFrame with `open_{suffix}`, `close_{suffix}`,
            `high_{suffix}`, `low_{suffix}` columns.
        suffix : str
            Timeframe suffix.
        epsilon : float, optional
            Denominator epsilon, default 1e-8 per ADR-015.

        Returns
        -------
        pd.Series
            Length matches `df`. No NaN introduced by this formula
            itself (epsilon prevents Inf/NaN from division by zero).
        """
        body = (df[f"close_{suffix}"] - df[f"open_{suffix}"]).abs()
        hl = df[f"high_{suffix}"] - df[f"low_{suffix}"]
        return body / (hl + epsilon)

    def compute_features_for_timeframe(
        self, df: pd.DataFrame, tf_suffix: str
    ) -> pd.DataFrame:
        """
        Compute all 7 ADR-015 features for one timeframe.

        Parameters
        ----------
        df : pd.DataFrame
            Master DataFrame (M3 output) containing
            `{open,high,low,close,volume}_{tf_suffix}` columns.
        tf_suffix : str
            One of "15m", "1h", "4h", "1d".

        Returns
        -------
        pd.DataFrame
            7 columns named `{feature}_{tf_suffix}`, same row count and
            index as `df`.
        """
        return pd.DataFrame(
            {
                f"open_return_{tf_suffix}": self.compute_open_return(df, tf_suffix),
                f"high_return_{tf_suffix}": self.compute_high_return(df, tf_suffix),
                f"low_return_{tf_suffix}": self.compute_low_return(df, tf_suffix),
                f"close_return_{tf_suffix}": self.compute_close_return(df, tf_suffix),
                f"volume_zscore_{tf_suffix}": self.compute_volume_zscore(df, tf_suffix),
                f"hl_range_{tf_suffix}": self.compute_hl_range(df, tf_suffix),
                f"body_ratio_{tf_suffix}": self.compute_body_ratio(df, tf_suffix),
            }
        )

    def compute_all_features(
        self, master_df: pd.DataFrame, timeframes: tuple[str, ...] = TIMEFRAME_SUFFIXES
    ) -> pd.DataFrame:
        """
        Compute all 7 features for all 4 timeframes and assemble the full matrix.

        Parameters
        ----------
        master_df : pd.DataFrame
            Aligned master DataFrame from M3 (21 columns: timestamp +
            5 OHLCV fields x 4 timeframes).
        timeframes : tuple[str, ...], optional
            Timeframe suffixes to compute features for, default all
            four DS-03 timeframes.

        Returns
        -------
        pd.DataFrame
            `timestamp` column plus 7 x len(timeframes) feature
            columns, same row count as `master_df` (NaN rows are NOT
            yet dropped — see `drop_nan_rows`).
        """
        result = pd.DataFrame({"timestamp": master_df["timestamp"]})
        for tf in timeframes:
            tf_features = self.compute_features_for_timeframe(master_df, tf)
            result = pd.concat([result, tf_features], axis=1)
        return result

    def drop_nan_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop leading rows containing any NaN value.

        Per DS-02 Stage 3: row 0 has NaN `open_return` (no previous
        close) and rows 0-18 have NaN `volume_zscore` (rolling window
        of 20 needs 20 observations). This drops any row containing at
        least one NaN across any feature column — which in practice
        means exactly the leading rows, since NaN only ever occurs at
        the start of each series for these particular formulas.

        Parameters
        ----------
        df : pd.DataFrame
            Feature matrix with a `timestamp` column and feature columns.

        Returns
        -------
        pd.DataFrame
            Rows with any NaN dropped, index reset. Per DS-02 v1.1,
            this is expected to drop exactly the first 19 rows for the
            full 35,064-row dataset, leaving 35,045 rows starting at
            2020-01-01 19:00:00 UTC (19 hours after the dataset start
            — see module docstring and EXPECTED_FIRST_TIMESTAMP).
        """
        feature_cols = [c for c in df.columns if c != "timestamp"]
        n_before = len(df)
        cleaned = df.dropna(subset=feature_cols).reset_index(drop=True)
        n_dropped = n_before - len(cleaned)
        logger.info(
            "drop_nan_rows: dropped %d row(s) (%d -> %d rows). "
            "First remaining timestamp: %s",
            n_dropped,
            n_before,
            len(cleaned),
            cleaned["timestamp"].iloc[0] if len(cleaned) > 0 else "N/A",
        )
        return cleaned


def check_feature_matrix_integrity(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Verify the feature matrix has no NaN or Inf values remaining.

    Per IMP-01 v1.2 M4 Definition of Done: "No NaN or Inf values remain
    after row dropping."

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix after `drop_nan_rows`.

    Returns
    -------
    tuple[bool, str]
        (passed, detail message naming any offending columns).
    """
    feature_cols = [c for c in df.columns if c != "timestamp"]
    issues: list[str] = []

    nan_counts = df[feature_cols].isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        issues.append(f"NaN found in columns: {nan_cols.to_dict()}")

    inf_mask = np.isinf(df[feature_cols].to_numpy(dtype="float64"))
    if inf_mask.any():
        inf_col_indices = np.where(inf_mask.any(axis=0))[0]
        inf_col_names = [feature_cols[i] for i in inf_col_indices]
        issues.append(f"Inf found in columns: {inf_col_names}")

    if issues:
        return False, "; ".join(issues)
    return True, "no NaN or Inf values found"


def check_feature_matrix_schema(
    df: pd.DataFrame,
    expected_rows: int | None = EXPECTED_FEATURE_ROWS,
    expected_first_timestamp: pd.Timestamp | None = EXPECTED_FIRST_TIMESTAMP,
) -> tuple[bool, str]:
    """
    Verify the feature matrix matches DS-02 v1.1 Stage 3 / DS-04 v1.1
    V-DATA-004 schema.

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix after `drop_nan_rows`.
    expected_rows : int or None, optional
        Expected row count, default `EXPECTED_FEATURE_ROWS` (35,045).
        Pass None to skip this check (for small synthetic test fixtures).
    expected_first_timestamp : pd.Timestamp or None, optional
        Expected first timestamp after NaN-drop, default
        **2020-01-01 19:00:00 UTC** (corrected per DS-02 v1.1 / DS-04
        v1.1 — see module docstring). Pass None to skip this check.

    Returns
    -------
    tuple[bool, str]
        (passed, detail message).
    """
    issues: list[str] = []

    if len(df.columns) != EXPECTED_FEATURE_COLUMNS:
        issues.append(
            f"expected {EXPECTED_FEATURE_COLUMNS} columns, got {len(df.columns)}"
        )

    expected_cols = {"timestamp"} | {
        f"{feat}_{tf}" for tf in TIMEFRAME_SUFFIXES for feat in FEATURE_NAMES
    }
    actual_cols = set(df.columns)
    if expected_cols != actual_cols:
        missing = expected_cols - actual_cols
        extra = actual_cols - expected_cols
        if missing:
            issues.append(f"missing columns: {sorted(missing)}")
        if extra:
            issues.append(f"unexpected columns: {sorted(extra)}")

    if expected_rows is not None and len(df) != expected_rows:
        issues.append(f"expected exactly {expected_rows} rows, got {len(df)}")

    if expected_first_timestamp is not None and len(df) > 0:
        actual_first = df["timestamp"].iloc[0]
        if actual_first != expected_first_timestamp:
            issues.append(
                f"expected first timestamp {expected_first_timestamp}, "
                f"got {actual_first}"
            )

    nan_inf_passed, nan_inf_detail = check_feature_matrix_integrity(df)
    if not nan_inf_passed:
        issues.append(nan_inf_detail)

    if issues:
        return False, "; ".join(issues)
    return True, "feature matrix schema OK"


def run_feature_engineering(
    master_df: pd.DataFrame,
    raise_on_failure: bool = True,
    expected_rows: int | None = EXPECTED_FEATURE_ROWS,
    expected_first_timestamp: pd.Timestamp | None = EXPECTED_FIRST_TIMESTAMP,
) -> pd.DataFrame:
    """
    Run the full M4 pipeline: compute features, drop NaN rows, verify schema.

    Parameters
    ----------
    master_df : pd.DataFrame
        Aligned master DataFrame from M3.
    raise_on_failure : bool, optional
        If True (default), raise `FeatureEngineeringError` if the
        resulting feature matrix fails schema/integrity checks.
    expected_rows : int or None, optional
        Expected row count for the schema check, default 35,045. Pass
        None for tests using small synthetic datasets.
    expected_first_timestamp : pd.Timestamp or None, optional
        Expected first timestamp for the schema check, default
        **2020-01-01 19:00:00 UTC** (corrected, see module docstring).
        Pass None for small synthetic datasets.

    Returns
    -------
    pd.DataFrame
        The final feature matrix (NaN rows dropped, schema-verified).

    Raises
    ------
    FeatureEngineeringError
        If `raise_on_failure=True` and the resulting matrix fails
        schema or NaN/Inf checks.
    """
    engineer = FeatureEngineer()
    raw_features = engineer.compute_all_features(master_df)
    clean_features = engineer.drop_nan_rows(raw_features)

    passed, detail = check_feature_matrix_schema(
        clean_features,
        expected_rows=expected_rows,
        expected_first_timestamp=expected_first_timestamp,
    )
    if passed:
        logger.info("M4 feature matrix schema check PASSED: %s", detail)
    else:
        logger.error("M4 feature matrix schema check FAILED: %s", detail)
        if raise_on_failure:
            raise FeatureEngineeringError(
                f"Feature matrix schema check failed: {detail}"
            )

    return clean_features
