"""
src/data/alignment.py

Temporal alignment module (M3 — Temporal Alignment, DS-02 Stage 2).

Purpose
-------
Aligns all four timeframes (15m, 1h, 4h, 1d) to the 1h anchor, producing
a single master DataFrame where every 1h timestamp from 2020-01-01
00:00 UTC to 2023-12-31 23:00 UTC has exactly one row containing OHLCV
data from all four timeframes (21 columns total).

This is the ONLY pipeline stage where upsampling occurs (per DS-02
Leakage Checkpoint LC-1, enforced in M2 by never forward-filling
there). That makes leakage auditing unambiguous — if look-ahead bias
exists anywhere in the alignment logic, it can only be here.

Leakage Checkpoint LC-2 (the most scientifically critical check in this
module)
--------------------------------------------------------------------
Forward-fill must use ONLY the current candle's values, never a future
candle's values. At timestamp T, the 4h/1d value assigned must reflect
the candle that opened at or before T — never one that opens after T.
`verify_no_lookahead()` below is the programmatic check for this
(DS-04 V-LEAK-001), and it is designed to be called on real sampled
timestamps from every year in the study period, not just at the
boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Master DataFrame schema per DS-02 Stage 2: 1 (timestamp) + 5 x 4
# (OHLCV x 4 timeframes) = 21 columns.
OHLCV_FIELDS: tuple[str, ...] = ("open", "high", "low", "close", "volume")
TIMEFRAME_SUFFIXES: tuple[str, ...] = ("1h", "15m", "4h", "1d")
EXPECTED_MASTER_COLUMNS: int = 1 + len(OHLCV_FIELDS) * len(TIMEFRAME_SUFFIXES)  # 21
EXPECTED_MASTER_ROWS: int = 35_064
ROW_COUNT_EDGE_TOLERANCE: int = 5  # per IMP-01 M3 DoD: "~35,064 rows (±5)"


class AlignmentError(RuntimeError):
    """Raised when temporal alignment fails a structural or leakage check."""


@dataclass
class LookaheadCheckResult:
    """Result of one look-ahead verification at a single sampled timestamp."""

    timestamp: pd.Timestamp
    source_timeframe: str
    assigned_open_time: pd.Timestamp | None
    passed: bool
    detail: str = ""


class TemporalAligner:
    """
    Aligns 15m, 1h, 4h, and 1d OHLCV DataFrames to a single 1h-anchor master.

    All methods are pure functions of their inputs (no hidden state),
    so each alignment operation can be independently unit-tested against
    small synthetic DataFrames before being run on the full dataset.
    """

    def aggregate_15m_to_1h(self, df_15m: pd.DataFrame) -> pd.DataFrame:
        """
        Downsample 15m candles to 1h via standard OHLCV aggregation.

        Per DS-02 Stage 2a: groups every 4 consecutive 15m candles into
        one 1h candle using open=first, high=max, low=min, close=last,
        volume=sum, with the 1h timestamp floored to the hour.

        Parameters
        ----------
        df_15m : pd.DataFrame
            Raw 15m OHLCV data, columns: timestamp, open, high, low,
            close, volume. Must be sorted ascending by timestamp.

        Returns
        -------
        pd.DataFrame
            Aggregated 1h-resolution OHLCV, same column names, one row
            per hour that had at least one contributing 15m candle.
        """
        df = df_15m.copy()
        df["_hour"] = df["timestamp"].dt.floor("1h")

        aggregated = (
            df.groupby("_hour", sort=True)
            .agg(
                open=("open", "first"),
                high=("high", "max"),
                low=("low", "min"),
                close=("close", "last"),
                volume=("volume", "sum"),
            )
            .reset_index()
            .rename(columns={"_hour": "timestamp"})
        )
        return aggregated

    def forward_fill_to_1h(
        self, df: pd.DataFrame, source_tf: str, target_index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Broadcast a lower-frequency timeframe's candles onto an hourly index.

        Per DS-02 Stage 2c/2d: each source candle at open-time T is
        broadcast forward to every 1h timestamp in [T, next_source_open),
        i.e. standard "last observation carried forward" — never a value
        from a candle that opens AFTER the target timestamp (LC-2).

        Parameters
        ----------
        df : pd.DataFrame
            Source OHLCV data (4h or 1d), columns: timestamp, open,
            high, low, close, volume. Must be sorted ascending.
        source_tf : str
            Label for the source timeframe (e.g. "4h", "1d") — used
            only for error messages and logging.
        target_index : pd.DatetimeIndex
            The full 1h timestamp index to broadcast onto (UTC).

        Returns
        -------
        pd.DataFrame
            One row per timestamp in `target_index`, columns: timestamp,
            open, high, low, close, volume — each hour's values are the
            most recent source candle with open-time ≤ that hour.
            Hours before the first source candle are left as NaN (no
            look-ahead is possible for them; documented boundary case).
        """
        df = df.sort_values("timestamp").set_index("timestamp")
        # reindex + ffill: for each target timestamp, pandas assigns the
        # value from the most recent PRIOR (or equal) index entry. This
        # is exactly "last observation carried forward" and by
        # construction cannot pull a value from a future source row.
        combined_index = df.index.union(target_index).sort_values()
        filled = df.reindex(combined_index).ffill()
        result = filled.reindex(target_index)
        result = result.reset_index().rename(columns={"index": "timestamp"})
        n_leading_nan = result["open"].isna().sum()
        if n_leading_nan > 0:
            logger.info(
                "forward_fill_to_1h(%s): %d leading timestamp(s) have no "
                "prior source candle (expected only at dataset start).",
                source_tf,
                n_leading_nan,
            )
        return result

    def _reindex_highfreq_to_grid(
        self,
        df: pd.DataFrame,
        target_index: pd.DatetimeIndex,
        present_index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """
        Reindex a high-frequency source (1h anchor or 15m-aggregated) onto
        the complete hourly grid, filling exchange-outage holes (ADR-023).

        For every hour in `target_index` that had no source candle
        (`present_index`): O/H/L/C are last-observation-carried-forward
        (no look-ahead — only a prior candle is ever used), and volume is
        set to 0.0 (no trading occurred during the outage). Hours that DID
        have a candle keep their real values unchanged.

        Parameters
        ----------
        df : pd.DataFrame
            Source OHLCV with a `timestamp` column.
        target_index : pd.DatetimeIndex
            The complete hourly grid to reindex onto.
        present_index : pd.DatetimeIndex
            The timestamps that genuinely had a source candle.

        Returns
        -------
        pd.DataFrame
            One row per `target_index` hour, columns: timestamp, open,
            high, low, close, volume.
        """
        s = df.set_index("timestamp").reindex(target_index)
        missing_mask = ~s.index.isin(present_index)
        for col in ("open", "high", "low", "close"):
            s[col] = s[col].ffill()
        s.loc[missing_mask, "volume"] = 0.0
        s = s.reset_index().rename(columns={"index": "timestamp"})
        if "timestamp" not in s.columns:  # defensive (index had no name)
            s = s.rename(columns={s.columns[0]: "timestamp"})
        return s

    def build_master(self, dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Build the full 21-column master DataFrame from all four timeframes.

        Parameters
        ----------
        dfs : dict[str, pd.DataFrame]
            Mapping "15m" -> df, "1h" -> df, "4h" -> df, "1d" -> df.
            Each df has raw columns: timestamp, open, high, low, close,
            volume (as produced by M1/validated by M2).

        Returns
        -------
        pd.DataFrame
            Master DataFrame per DS-02 Stage 2 schema: `timestamp` plus
            `{open,high,low,close,volume}_{1h,15m,4h,1d}` (21 columns
            total), one row per 1h timestamp, sorted ascending.

        Raises
        ------
        ValueError
            If `dfs` is missing any of the four required timeframes.
        """
        required = {"15m", "1h", "4h", "1d"}
        missing = required - set(dfs.keys())
        if missing:
            raise ValueError(f"build_master is missing timeframes: {missing}")

        df_1h = dfs["1h"].sort_values("timestamp").reset_index(drop=True)

        # Complete hourly grid over the observed span (ADR-023). Real
        # exchange outages leave holes in the 1h anchor; anchoring on the
        # actual 1h timestamps would shorten the master and shift every
        # LC-4-audited downstream count. Instead we reindex to a gap-free
        # hourly grid: missing hours receive last-observation-carried-
        # forward PRICES (O/H/L/C — no look-ahead) and volume=0 (no
        # trading occurred during the outage). This preserves the full
        # 35,064-hour temporal structure the whole pipeline assumes.
        full_index = pd.date_range(
            start=df_1h["timestamp"].min(),
            end=df_1h["timestamp"].max(),
            freq="1h",
        )
        target_index = pd.DatetimeIndex(full_index, name="timestamp")

        # 2a: 15m -> 1h aggregation (then gap-fill onto the full grid)
        agg_15m_raw = self.aggregate_15m_to_1h(dfs["15m"])
        agg_15m = self._reindex_highfreq_to_grid(
            agg_15m_raw, target_index, pd.DatetimeIndex(agg_15m_raw["timestamp"])
        )

        # 2b: 1h identity (gap-filled onto the full grid)
        anchor_1h = self._reindex_highfreq_to_grid(
            df_1h, target_index, pd.DatetimeIndex(df_1h["timestamp"])
        )

        # 2c, 2d: 4h and 1d forward-fill
        ff_4h = self.forward_fill_to_1h(dfs["4h"], "4h", target_index)
        ff_1d = self.forward_fill_to_1h(dfs["1d"], "1d", target_index)

        master = pd.DataFrame({"timestamp": target_index})
        for suffix, source in (
            ("1h", anchor_1h),
            ("15m", agg_15m),
            ("4h", ff_4h),
            ("1d", ff_1d),
        ):
            source_indexed = source.set_index("timestamp").reindex(target_index)
            for field in OHLCV_FIELDS:
                master[f"{field}_{suffix}"] = source_indexed[field].to_numpy()

        master = master.sort_values("timestamp").reset_index(drop=True)

        logger.info(
            "Built master DataFrame: %d rows x %d columns",
            len(master),
            len(master.columns),
        )
        return master


def verify_no_lookahead(
    master_df: pd.DataFrame,
    source_df: pd.DataFrame,
    tf: str,
    sample_timestamps: list[pd.Timestamp] | None = None,
) -> list[LookaheadCheckResult]:
    """
    Programmatically verify LC-2 / V-LEAK-001: no look-ahead in forward-fill.

    For each sampled timestamp T in `master_df`, checks that the
    `open_{tf}` value assigned at T equals the `open` value of the
    source candle whose timestamp is the largest value ≤ T (i.e. the
    most recent candle at or before T — never a candle opening after T).

    Parameters
    ----------
    master_df : pd.DataFrame
        The aligned master DataFrame (output of `build_master`).
    source_df : pd.DataFrame
        The original (pre-alignment) OHLCV data for timeframe `tf`.
    tf : str
        One of "4h" or "1d" (the two forward-filled timeframes; 1h is
        the identity anchor and 15m is aggregated, not forward-filled).
    sample_timestamps : list[pd.Timestamp], optional
        Specific timestamps to check. If None, one timestamp is sampled
        near the start of each year present in `master_df` (per IMP-01
        M3 DoD: "unit test covers at least one timestamp from each year
        (2020-2023)").

    Returns
    -------
    list[LookaheadCheckResult]
        One result per timestamp checked. `V-LEAK-001` passes only if
        every result has `passed=True`.
    """
    if tf not in ("4h", "1d"):
        raise ValueError(f"verify_no_lookahead only applies to forward-filled timeframes (4h, 1d), got '{tf}'.")

    source_sorted = source_df.sort_values("timestamp").reset_index(drop=True)
    source_timestamps = pd.DatetimeIndex(source_sorted["timestamp"])

    if sample_timestamps is None:
        years = sorted(master_df["timestamp"].dt.year.unique())
        sample_timestamps = []
        for year in years:
            year_rows = master_df[master_df["timestamp"].dt.year == year]
            if len(year_rows) > 0:
                # Sample a timestamp a few hours into the year, not the
                # exact boundary, to catch off-by-one errors in ffill
                # logic that boundary-only testing could miss.
                idx = min(5, len(year_rows) - 1)
                sample_timestamps.append(year_rows["timestamp"].iloc[idx])

    results: list[LookaheadCheckResult] = []
    open_col = f"open_{tf}"

    for T in sample_timestamps:
        # Compare using pandas Timestamps/DatetimeIndex directly (not
        # np.datetime64, which silently drops timezone info and causes
        # incorrect or erroring comparisons between tz-aware values).
        T_ts = pd.Timestamp(T)
        # The correct source candle is the one with the largest
        # open-time <= T (last observation carried forward).
        valid_source_mask = source_timestamps <= T_ts
        if not valid_source_mask.any():
            # No prior candle exists yet (dataset-start boundary case).
            expected_open_time = None
            expected_open_value = np.nan
        else:
            candidate_positions = np.where(valid_source_mask)[0]
            candidate_idx = candidate_positions[-1]
            expected_open_time = source_timestamps[candidate_idx]
            expected_open_value = source_sorted["open"].iloc[candidate_idx]

            # Explicitly confirm no candle opening AFTER T was used —
            # this is the actual LC-2 condition, not just "some prior
            # candle exists".
            future_candles = source_timestamps > T_ts
            if future_candles.any():
                first_future_positions = np.where(future_candles)[0]
                first_future_idx = first_future_positions[0]
                first_future_time = source_timestamps[first_future_idx]
                if first_future_time <= expected_open_time:
                    # Defensive check: should never trigger given sorted
                    # data, but protects against silent sort corruption.
                    results.append(
                        LookaheadCheckResult(
                            timestamp=T,
                            source_timeframe=tf,
                            assigned_open_time=expected_open_time,
                            passed=False,
                            detail="Internal consistency error: candidate "
                            "candle is not strictly before the next future candle.",
                        )
                    )
                    continue

        master_row = master_df.loc[master_df["timestamp"] == T]
        if master_row.empty:
            results.append(
                LookaheadCheckResult(
                    timestamp=T,
                    source_timeframe=tf,
                    assigned_open_time=None,
                    passed=False,
                    detail=f"Timestamp {T} not found in master_df.",
                )
            )
            continue

        actual_open_value = master_row[open_col].iloc[0]

        if expected_open_time is None:
            passed = pd.isna(actual_open_value)
            detail = "no prior source candle exists (dataset-start boundary)"
        else:
            passed = bool(
                np.isclose(actual_open_value, expected_open_value, equal_nan=True)
            )
            detail = (
                f"expected candle open_time={expected_open_time} "
                f"(open={expected_open_value}), "
                f"master assigned open={actual_open_value}"
            )

        results.append(
            LookaheadCheckResult(
                timestamp=T,
                source_timeframe=tf,
                assigned_open_time=expected_open_time,
                passed=passed,
                detail=detail,
            )
        )

    return results


def verify_grid_completeness(master_df: pd.DataFrame) -> tuple[bool, str]:
    """
    Verify the master spans a gap-free hourly grid (ADR-023).

    After the M3 reindex, every consecutive pair of timestamps must be
    exactly 1 hour apart — no exchange-outage holes remain. This is the
    structural guarantee that all downstream LC-4-audited counts
    (35,045 features; N_test_windows=8,760) rest on.

    Returns
    -------
    tuple[bool, str]
        (passed, detail).
    """
    ts = master_df["timestamp"]
    if len(ts) < 2:
        return True, "grid completeness trivially OK (<2 rows)"
    deltas = ts.diff().dropna()
    one_hour = pd.Timedelta(hours=1)
    bad = int((deltas != one_hour).sum())
    if bad > 0:
        first_bad = ts.iloc[1:][deltas.to_numpy() != one_hour].iloc[0]
        return False, (
            f"{bad} non-1h step(s) in the hourly grid (first near {first_bad}); "
            "reindex did not produce a gap-free grid."
        )
    return True, f"hourly grid complete: {len(ts)} rows, all steps = 1h"


def check_master_schema(
    master_df: pd.DataFrame,
    expected_rows: int | None = EXPECTED_MASTER_ROWS,
    row_tolerance: int = ROW_COUNT_EDGE_TOLERANCE,
) -> tuple[bool, str]:
    """
    Verify the master DataFrame matches the DS-02 Stage 2 / V-DATA-003 schema.

    Parameters
    ----------
    master_df : pd.DataFrame
        Output of `TemporalAligner.build_master`.
    expected_rows : int or None, optional
        Expected row count to check against, default
        `EXPECTED_MASTER_ROWS` (35,064, the real DS-04 study-period
        value). Pass `None` to skip the row-count check entirely —
        used by unit tests that build a small synthetic master
        DataFrame and only care about column/monotonicity/timezone
        correctness, not the real dataset's row count.
    row_tolerance : int, optional
        Allowed absolute deviation from `expected_rows`, default
        `ROW_COUNT_EDGE_TOLERANCE` (5, per IMP-01 M3 DoD).

    Returns
    -------
    tuple[bool, str]
        (passed, detail message).
    """
    issues: list[str] = []

    if len(master_df.columns) != EXPECTED_MASTER_COLUMNS:
        issues.append(
            f"expected {EXPECTED_MASTER_COLUMNS} columns, got {len(master_df.columns)}"
        )

    expected_cols = {"timestamp"} | {
        f"{field}_{suffix}" for suffix in TIMEFRAME_SUFFIXES for field in OHLCV_FIELDS
    }
    actual_cols = set(master_df.columns)
    if expected_cols != actual_cols:
        missing = expected_cols - actual_cols
        extra = actual_cols - expected_cols
        if missing:
            issues.append(f"missing columns: {sorted(missing)}")
        if extra:
            issues.append(f"unexpected columns: {sorted(extra)}")

    row_diff = abs(len(master_df) - expected_rows) if expected_rows is not None else 0
    if expected_rows is not None and row_diff > row_tolerance:
        issues.append(
            f"expected ~{expected_rows} rows (±{row_tolerance}), "
            f"got {len(master_df)}"
        )

    if not master_df["timestamp"].is_monotonic_increasing:
        issues.append("timestamp column is not strictly monotonic increasing")

    if not isinstance(master_df["timestamp"].dtype, pd.DatetimeTZDtype) or str(
        master_df["timestamp"].dtype.tz
    ) != "UTC":
        issues.append("timestamp column is not UTC-localized")

    if issues:
        return False, "; ".join(issues)
    return True, "master DataFrame schema OK"


def build_and_verify_master(
    dfs: dict[str, pd.DataFrame],
    raise_on_failure: bool = True,
    expected_rows: int | None = EXPECTED_MASTER_ROWS,
) -> tuple[pd.DataFrame, list[LookaheadCheckResult]]:
    """
    Run the full M3 pipeline: build the master DataFrame and verify LC-2.

    Parameters
    ----------
    dfs : dict[str, pd.DataFrame]
        Mapping of timeframe -> raw OHLCV DataFrame (M1/M2 output).
    raise_on_failure : bool, optional
        If True (default), raise `AlignmentError` if the schema check
        or any look-ahead check fails.
    expected_rows : int or None, optional
        Expected master row count for the schema check, default
        `EXPECTED_MASTER_ROWS` (35,064, the real study-period value).
        Pass `None` to skip the row-count portion of the schema check
        — used by tests that build small synthetic datasets and want
        to isolate the leakage check from the row-count check.

    Returns
    -------
    tuple[pd.DataFrame, list[LookaheadCheckResult]]
        The master DataFrame and the full list of look-ahead check
        results (for 4h and 1d combined).

    Raises
    ------
    AlignmentError
        If `raise_on_failure=True` and the schema is wrong or any
        look-ahead check fails.
    """
    aligner = TemporalAligner()
    master = aligner.build_master(dfs)

    schema_passed, schema_detail = check_master_schema(master, expected_rows=expected_rows)
    if not schema_passed:
        logger.error("M3 schema check FAILED: %s", schema_detail)
        if raise_on_failure:
            raise AlignmentError(f"Master DataFrame schema check failed: {schema_detail}")
    else:
        logger.info("M3 schema check PASSED: %s", schema_detail)

    # ADR-023: the reindexed master must be a gap-free hourly grid.
    grid_passed, grid_detail = verify_grid_completeness(master)
    if not grid_passed:
        logger.error("M3 grid-completeness check FAILED: %s", grid_detail)
        if raise_on_failure:
            raise AlignmentError(f"Master grid completeness failed: {grid_detail}")
    else:
        logger.info("M3 grid-completeness check PASSED: %s", grid_detail)

    all_lookahead_results: list[LookaheadCheckResult] = []
    for tf in ("4h", "1d"):
        results = verify_no_lookahead(master, dfs[tf], tf)
        all_lookahead_results.extend(results)
        failed = [r for r in results if not r.passed]
        if failed:
            logger.error(
                "V-LEAK-001 FAILED for timeframe=%s: %d/%d sampled timestamps "
                "show look-ahead or mismatch.",
                tf,
                len(failed),
                len(results),
            )
            for r in failed:
                logger.error("  %s @ %s: %s", tf, r.timestamp, r.detail)
        else:
            logger.info(
                "V-LEAK-001 PASSED for timeframe=%s: all %d sampled timestamps OK.",
                tf,
                len(results),
            )

    if raise_on_failure:
        all_failed = [r for r in all_lookahead_results if not r.passed]
        if all_failed:
            detail_lines = "\n".join(
                f"  {r.source_timeframe} @ {r.timestamp}: {r.detail}" for r in all_failed
            )
            raise AlignmentError(
                f"V-LEAK-001 (no look-ahead) FAILED for {len(all_failed)} "
                f"sampled timestamp(s):\n{detail_lines}"
            )

    return master, all_lookahead_results
