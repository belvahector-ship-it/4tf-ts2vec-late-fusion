"""
tests/test_alignment.py

Unit tests for src/data/alignment.py (M3 — Temporal Alignment).

This is the most scientifically critical test file in the pipeline so
far: it verifies Leakage Checkpoint LC-2 / DS-04 V-LEAK-001, which
protects against look-ahead bias entering the forward-fill alignment
step. A failure here would silently invalidate the entire experimental
comparison (DS-04 §1.1: "Temporal Integrity Validation... the most
scientifically critical validations in this document").

Covers IMP-01 M3 Definition of Done:
- Output has exactly 21 columns and ~35,064 rows (±5 for edge handling)
- Timestamp column is strictly monotonic and UTC-localized
- Forward-fill produces no look-ahead (verify_no_lookahead)
- The first partial period at dataset start is handled explicitly
- Unit test covers at least one timestamp from each year (2020-2023)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.alignment import (
    EXPECTED_MASTER_COLUMNS,
    EXPECTED_MASTER_ROWS,
    AlignmentError,
    TemporalAligner,
    build_and_verify_master,
    check_master_schema,
    verify_no_lookahead,
)


# --- Fixtures ----------------------------------------------------------------

def _make_ohlcv(timestamps: pd.DatetimeIndex, base_price: float = 100.0) -> pd.DataFrame:
    """Build a simple, internally-consistent OHLCV DataFrame."""
    n = len(timestamps)
    open_ = base_price + np.arange(n, dtype="float64")
    close = open_ + 0.5
    high = np.maximum(open_, close) + 1.0
    low = np.minimum(open_, close) - 1.0
    volume = np.full(n, 10.0)
    return pd.DataFrame(
        {"timestamp": timestamps, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


@pytest.fixture
def small_multi_tf_dfs() -> dict[str, pd.DataFrame]:
    """
    A small, hand-constructed multi-timeframe dataset spanning exactly
    2 days (48 hours), so 1d has 2 candles, 4h has 12 candles, 1h has
    48 candles, and 15m has 192 candles — small enough to reason about
    by hand while still exercising every alignment rule.
    """
    ts_1h = pd.date_range("2020-01-01 00:00", periods=48, freq="1h", tz="UTC")
    ts_15m = pd.date_range("2020-01-01 00:00", periods=192, freq="15min", tz="UTC")
    ts_4h = pd.date_range("2020-01-01 00:00", periods=12, freq="4h", tz="UTC")
    ts_1d = pd.date_range("2020-01-01 00:00", periods=2, freq="1D", tz="UTC")

    return {
        "1h": _make_ohlcv(ts_1h, base_price=1000.0),
        "15m": _make_ohlcv(ts_15m, base_price=100.0),
        "4h": _make_ohlcv(ts_4h, base_price=5000.0),
        "1d": _make_ohlcv(ts_1d, base_price=20000.0),
    }


@pytest.fixture
def aligner() -> TemporalAligner:
    return TemporalAligner()


# --- aggregate_15m_to_1h -------------------------------------------------------

class TestAggregate15mTo1h:
    def test_output_row_count_matches_hours(self, aligner) -> None:
        ts_15m = pd.date_range("2020-01-01 00:00", periods=192, freq="15min", tz="UTC")
        df_15m = _make_ohlcv(ts_15m)
        result = aligner.aggregate_15m_to_1h(df_15m)
        assert len(result) == 48  # 192 / 4 = 48 hours

    def test_open_is_first_of_four(self, aligner) -> None:
        ts_15m = pd.date_range("2020-01-01 00:00", periods=4, freq="15min", tz="UTC")
        df_15m = pd.DataFrame(
            {
                "timestamp": ts_15m,
                "open": [1.0, 2.0, 3.0, 4.0],
                "high": [1.5, 2.5, 3.5, 4.5],
                "low": [0.5, 1.5, 2.5, 3.5],
                "close": [1.2, 2.2, 3.2, 4.2],
                "volume": [10.0, 10.0, 10.0, 10.0],
            }
        )
        result = aligner.aggregate_15m_to_1h(df_15m)
        assert result["open"].iloc[0] == 1.0  # first candle's open

    def test_close_is_last_of_four(self, aligner) -> None:
        ts_15m = pd.date_range("2020-01-01 00:00", periods=4, freq="15min", tz="UTC")
        df_15m = pd.DataFrame(
            {
                "timestamp": ts_15m,
                "open": [1.0, 2.0, 3.0, 4.0],
                "high": [1.5, 2.5, 3.5, 4.5],
                "low": [0.5, 1.5, 2.5, 3.5],
                "close": [1.2, 2.2, 3.2, 4.2],
                "volume": [10.0, 10.0, 10.0, 10.0],
            }
        )
        result = aligner.aggregate_15m_to_1h(df_15m)
        assert result["close"].iloc[0] == 4.2  # last candle's close

    def test_high_is_max_low_is_min(self, aligner) -> None:
        ts_15m = pd.date_range("2020-01-01 00:00", periods=4, freq="15min", tz="UTC")
        df_15m = pd.DataFrame(
            {
                "timestamp": ts_15m,
                "open": [1.0, 2.0, 3.0, 4.0],
                "high": [1.5, 9.0, 3.5, 4.5],   # max = 9.0
                "low": [0.5, 1.5, -2.0, 3.5],   # min = -2.0
                "close": [1.2, 2.2, 3.2, 4.2],
                "volume": [10.0, 10.0, 10.0, 10.0],
            }
        )
        result = aligner.aggregate_15m_to_1h(df_15m)
        assert result["high"].iloc[0] == 9.0
        assert result["low"].iloc[0] == -2.0

    def test_volume_is_sum(self, aligner) -> None:
        ts_15m = pd.date_range("2020-01-01 00:00", periods=4, freq="15min", tz="UTC")
        df_15m = pd.DataFrame(
            {
                "timestamp": ts_15m,
                "open": [1.0] * 4,
                "high": [1.5] * 4,
                "low": [0.5] * 4,
                "close": [1.2] * 4,
                "volume": [1.0, 2.0, 3.0, 4.0],
            }
        )
        result = aligner.aggregate_15m_to_1h(df_15m)
        assert result["volume"].iloc[0] == 10.0

    def test_timestamp_is_floored_to_hour(self, aligner) -> None:
        ts_15m = pd.date_range("2020-01-01 00:00", periods=4, freq="15min", tz="UTC")
        df_15m = _make_ohlcv(ts_15m)
        result = aligner.aggregate_15m_to_1h(df_15m)
        assert result["timestamp"].iloc[0] == pd.Timestamp("2020-01-01 00:00", tz="UTC")


# --- forward_fill_to_1h ------------------------------------------------------------

class TestForwardFillTo1h:
    def test_4h_candle_broadcasts_to_4_hours(self, aligner) -> None:
        ts_4h = pd.date_range("2020-01-01 00:00", periods=2, freq="4h", tz="UTC")
        df_4h = _make_ohlcv(ts_4h, base_price=5000.0)
        target_index = pd.date_range("2020-01-01 00:00", periods=8, freq="1h", tz="UTC")

        result = aligner.forward_fill_to_1h(df_4h, "4h", target_index)

        # Hours 0,1,2,3 should all get the first 4h candle's open value.
        first_open = df_4h["open"].iloc[0]
        assert (result["open"].iloc[0:4] == first_open).all()
        # Hours 4,5,6,7 should get the second 4h candle's open value.
        second_open = df_4h["open"].iloc[1]
        assert (result["open"].iloc[4:8] == second_open).all()

    def test_1d_candle_broadcasts_to_24_hours(self, aligner) -> None:
        ts_1d = pd.date_range("2020-01-01", periods=2, freq="1D", tz="UTC")
        df_1d = _make_ohlcv(ts_1d, base_price=20000.0)
        target_index = pd.date_range("2020-01-01 00:00", periods=48, freq="1h", tz="UTC")

        result = aligner.forward_fill_to_1h(df_1d, "1d", target_index)

        first_open = df_1d["open"].iloc[0]
        assert (result["open"].iloc[0:24] == first_open).all()
        second_open = df_1d["open"].iloc[1]
        assert (result["open"].iloc[24:48] == second_open).all()

    def test_no_future_candle_used_at_exact_boundary(self, aligner) -> None:
        """
        At the exact moment a new source candle opens, that NEW candle's
        value should be used (it is "at or before" that timestamp), not
        held over from the previous one. This is the precise boundary
        condition LC-2 cares about.
        """
        ts_4h = pd.date_range("2020-01-01 00:00", periods=2, freq="4h", tz="UTC")
        df_4h = _make_ohlcv(ts_4h, base_price=5000.0)
        target_index = pd.date_range("2020-01-01 00:00", periods=8, freq="1h", tz="UTC")

        result = aligner.forward_fill_to_1h(df_4h, "4h", target_index)

        # At hour 4 (exactly when the second 4h candle opens), the
        # second candle's value must already be in effect.
        assert result["open"].iloc[4] == df_4h["open"].iloc[1]
        # At hour 3 (just before), the first candle's value must still
        # be in effect — NOT the second candle's (which would be
        # look-ahead).
        assert result["open"].iloc[3] == df_4h["open"].iloc[0]

    def test_leading_nan_before_first_candle(self, aligner) -> None:
        """Dataset-start boundary: hours before the first source candle
        have no valid prior value and must be NaN, not silently
        borrowed from the future."""
        ts_4h = pd.date_range("2020-01-01 04:00", periods=1, freq="4h", tz="UTC")
        df_4h = _make_ohlcv(ts_4h, base_price=5000.0)
        target_index = pd.date_range("2020-01-01 00:00", periods=8, freq="1h", tz="UTC")

        result = aligner.forward_fill_to_1h(df_4h, "4h", target_index)

        assert result["open"].iloc[0:4].isna().all()
        assert not result["open"].iloc[4:8].isna().any()


# --- build_master ---------------------------------------------------------------------

class TestBuildMaster:
    def test_output_has_21_columns(self, aligner, small_multi_tf_dfs) -> None:
        master = aligner.build_master(small_multi_tf_dfs)
        assert len(master.columns) == EXPECTED_MASTER_COLUMNS

    def test_output_row_count_matches_1h_anchor(self, aligner, small_multi_tf_dfs) -> None:
        master = aligner.build_master(small_multi_tf_dfs)
        assert len(master) == 48  # matches the 1h fixture length

    def test_missing_timeframe_raises(self, aligner, small_multi_tf_dfs) -> None:
        incomplete = {k: v for k, v in small_multi_tf_dfs.items() if k != "4h"}
        with pytest.raises(ValueError, match="missing timeframes"):
            aligner.build_master(incomplete)

    def test_timestamp_is_monotonic(self, aligner, small_multi_tf_dfs) -> None:
        master = aligner.build_master(small_multi_tf_dfs)
        assert master["timestamp"].is_monotonic_increasing

    def test_all_expected_columns_present(self, aligner, small_multi_tf_dfs) -> None:
        master = aligner.build_master(small_multi_tf_dfs)
        for suffix in ("1h", "15m", "4h", "1d"):
            for field in ("open", "high", "low", "close", "volume"):
                assert f"{field}_{suffix}" in master.columns

    def test_1h_columns_match_anchor_exactly(self, aligner, small_multi_tf_dfs) -> None:
        master = aligner.build_master(small_multi_tf_dfs)
        np.testing.assert_array_equal(
            master["open_1h"].to_numpy(), small_multi_tf_dfs["1h"]["open"].to_numpy()
        )

    def test_4h_columns_show_forward_fill_pattern(self, aligner, small_multi_tf_dfs) -> None:
        master = aligner.build_master(small_multi_tf_dfs)
        # First 4 hours share the same 4h open value.
        assert master["open_4h"].iloc[0:4].nunique() == 1
        # Next 4 hours (a new 4h candle) differ from the first block.
        assert master["open_4h"].iloc[4] != master["open_4h"].iloc[0]


# --- check_master_schema (V-DATA-003) --------------------------------------------------

class TestCheckMasterSchema:
    def test_correct_schema_passes(self, aligner, small_multi_tf_dfs) -> None:
        master = aligner.build_master(small_multi_tf_dfs)
        passed, detail = check_master_schema(master)
        # Row count will fail the ±5 tolerance check against the full
        # 35,064 expectation since this is a 48-row fixture — that's
        # expected and fine; we only assert schema/column/monotonic
        # aspects here by checking the detail message content.
        assert "columns" not in detail or passed  # column count is correct regardless
        assert master["timestamp"].is_monotonic_increasing

    def test_wrong_column_count_fails(self, aligner, small_multi_tf_dfs) -> None:
        master = aligner.build_master(small_multi_tf_dfs)
        broken = master.drop(columns=["open_1d"])
        passed, detail = check_master_schema(broken)
        assert not passed
        assert "columns" in detail

    def test_non_monotonic_timestamp_fails(self, aligner, small_multi_tf_dfs) -> None:
        master = aligner.build_master(small_multi_tf_dfs)
        shuffled = master.sample(frac=1, random_state=1).reset_index(drop=True)
        passed, detail = check_master_schema(shuffled)
        assert not passed
        assert "monotonic" in detail

    def test_full_size_dataset_passes_row_count(self, aligner) -> None:
        """Construct a full ~35,064-row dataset to confirm the exact
        row-count expectation from DS-02/V-DATA-003 is met."""
        ts_1h = pd.date_range("2020-01-01", periods=EXPECTED_MASTER_ROWS, freq="1h", tz="UTC")
        ts_15m = pd.date_range(
            "2020-01-01", periods=EXPECTED_MASTER_ROWS * 4, freq="15min", tz="UTC"
        )
        ts_4h = pd.date_range(
            "2020-01-01", periods=EXPECTED_MASTER_ROWS // 4 + 1, freq="4h", tz="UTC"
        )
        ts_1d = pd.date_range(
            "2020-01-01", periods=EXPECTED_MASTER_ROWS // 24 + 1, freq="1D", tz="UTC"
        )
        dfs = {
            "1h": _make_ohlcv(ts_1h, 1000.0),
            "15m": _make_ohlcv(ts_15m, 100.0),
            "4h": _make_ohlcv(ts_4h, 5000.0),
            "1d": _make_ohlcv(ts_1d, 20000.0),
        }
        master = aligner.build_master(dfs)
        passed, detail = check_master_schema(master)
        assert passed, detail
        assert len(master) == EXPECTED_MASTER_ROWS
        assert len(master.columns) == 21


# --- verify_no_lookahead (V-LEAK-001, the most critical test in this file) ------------

class TestVerifyNoLookahead:
    def test_all_sampled_timestamps_pass_on_correct_alignment(
        self, aligner, small_multi_tf_dfs
    ) -> None:
        master = aligner.build_master(small_multi_tf_dfs)
        results = verify_no_lookahead(
            master, small_multi_tf_dfs["4h"], "4h",
            sample_timestamps=list(
                pd.date_range("2020-01-01 00:00", periods=48, freq="1h", tz="UTC")
            ),
        )
        assert all(r.passed for r in results), [r.detail for r in results if not r.passed]

    def test_detects_injected_lookahead_bug(self, aligner, small_multi_tf_dfs) -> None:
        """
        Deliberately corrupt the master DataFrame to simulate a
        look-ahead bug (using a FUTURE 4h candle's value instead of the
        correct one), and confirm verify_no_lookahead catches it. This
        is the single most important test in the repository: it proves
        the leakage detector actually detects leakage, not just that it
        passes on already-correct data.
        """
        master = aligner.build_master(small_multi_tf_dfs)
        corrupted = master.copy()

        # At hour 0 (should use 4h candle #0's open), inject candle #1's
        # (a FUTURE candle's) open value instead.
        future_open_value = small_multi_tf_dfs["4h"]["open"].iloc[1]
        corrupted.loc[0, "open_4h"] = future_open_value

        results = verify_no_lookahead(
            corrupted,
            small_multi_tf_dfs["4h"],
            "4h",
            sample_timestamps=[corrupted["timestamp"].iloc[0]],
        )
        assert len(results) == 1
        assert not results[0].passed, (
            "verify_no_lookahead FAILED TO DETECT an injected look-ahead "
            "bug. This is a critical failure of the leakage detector itself."
        )

    def test_boundary_timestamp_with_no_prior_candle_is_nan(
        self, aligner
    ) -> None:
        """At the very start of the dataset, before any 4h candle has
        opened, the master value must be NaN — and verify_no_lookahead
        must recognize NaN as the CORRECT (not failing) state here."""
        ts_1h = pd.date_range("2020-01-01 00:00", periods=8, freq="1h", tz="UTC")
        ts_4h = pd.date_range("2020-01-01 02:00", periods=1, freq="4h", tz="UTC")  # starts late
        df_1h = _make_ohlcv(ts_1h, 1000.0)
        df_4h = _make_ohlcv(ts_4h, 5000.0)

        result = aligner.forward_fill_to_1h(df_4h, "4h", pd.DatetimeIndex(ts_1h))
        master = pd.DataFrame({"timestamp": ts_1h})
        for field in ("open", "high", "low", "close", "volume"):
            master[f"{field}_4h"] = result[field].to_numpy()

        results = verify_no_lookahead(
            master, df_4h, "4h", sample_timestamps=[ts_1h[0]]
        )
        assert results[0].passed
        assert "boundary" in results[0].detail

    def test_covers_at_least_one_timestamp_per_year_when_unspecified(
        self, aligner
    ) -> None:
        """IMP-01 M3 DoD: 'Unit test covers at least one timestamp from
        each year (2020-2023)'. Confirm the default sampling behavior
        (sample_timestamps=None) actually produces one per year."""
        ts_1h = pd.date_range("2020-01-01", "2023-12-31 23:00", freq="1h", tz="UTC")
        ts_4h = pd.date_range("2020-01-01", "2023-12-31", freq="4h", tz="UTC")
        df_1h = _make_ohlcv(ts_1h, 1000.0)
        df_4h = _make_ohlcv(ts_4h, 5000.0)

        aligner_local = TemporalAligner()
        master = aligner_local.build_master(
            {
                "1h": df_1h,
                "15m": _make_ohlcv(
                    pd.date_range("2020-01-01", "2023-12-31 23:45", freq="15min", tz="UTC"), 100.0
                ),
                "4h": df_4h,
                "1d": _make_ohlcv(
                    pd.date_range("2020-01-01", "2023-12-31", freq="1D", tz="UTC"), 20000.0
                ),
            }
        )
        results = verify_no_lookahead(master, df_4h, "4h", sample_timestamps=None)
        years_covered = {r.timestamp.year for r in results}
        assert years_covered == {2020, 2021, 2022, 2023}

    def test_rejects_invalid_timeframe(self, aligner, small_multi_tf_dfs) -> None:
        master = aligner.build_master(small_multi_tf_dfs)
        with pytest.raises(ValueError, match="only applies to"):
            verify_no_lookahead(master, small_multi_tf_dfs["1h"], "1h")


# --- ADR-023: outage-gap reindex to a complete hourly grid ---------------------------

class TestOutageGapReindex:
    """
    ADR-023 regression: real exchange outages leave holes in the 1h/15m
    anchor. M3 must reindex to a complete hourly grid, forward-fill prices
    (no look-ahead), and set volume=0 for the outage hours — preserving
    the full temporal structure the downstream LC-4 counts depend on.
    """

    def _dfs_with_hole(self, hole: pd.Timestamp) -> dict[str, pd.DataFrame]:
        ts_1h = pd.date_range("2020-01-01 00:00", periods=48, freq="1h", tz="UTC")
        ts_15m = pd.date_range("2020-01-01 00:00", periods=192, freq="15min", tz="UTC")
        ts_4h = pd.date_range("2020-01-01 00:00", periods=12, freq="4h", tz="UTC")
        ts_1d = pd.date_range("2020-01-01 00:00", periods=2, freq="1D", tz="UTC")
        df_1h = _make_ohlcv(ts_1h, base_price=1000.0)
        df_15m = _make_ohlcv(ts_15m, base_price=100.0)
        df_4h = _make_ohlcv(ts_4h, base_price=5000.0)
        df_1d = _make_ohlcv(ts_1d, base_price=20000.0)
        # Punch a 1-hour hole in the 1h anchor and its four 15m candles.
        df_1h = df_1h[df_1h["timestamp"] != hole].reset_index(drop=True)
        df_15m = df_15m[df_15m["timestamp"].dt.floor("1h") != hole].reset_index(
            drop=True
        )
        return {"1h": df_1h, "15m": df_15m, "4h": df_4h, "1d": df_1d}

    def test_hole_is_reindexed_and_grid_complete(self, aligner) -> None:
        from src.data.alignment import verify_grid_completeness

        hole = pd.Timestamp("2020-01-01 10:00", tz="UTC")
        master = aligner.build_master(self._dfs_with_hole(hole))
        assert len(master) == 48  # hole reindexed back into the grid
        ok, detail = verify_grid_completeness(master)
        assert ok, detail

    def test_price_locf_and_volume_zero_at_outage_hour(self, aligner) -> None:
        hole = pd.Timestamp("2020-01-01 10:00", tz="UTC")
        prev = pd.Timestamp("2020-01-01 09:00", tz="UTC")
        master = aligner.build_master(self._dfs_with_hole(hole))
        row = master.loc[master["timestamp"] == hole]
        prev_row = master.loc[master["timestamp"] == prev]
        assert not row.empty
        # price O/H/L/C are last-observation-carried-forward from hour 9
        for field in ("open", "high", "low", "close"):
            assert row[f"{field}_1h"].iloc[0] == prev_row[f"{field}_1h"].iloc[0]
        # volume is 0 for the outage hour (1h anchor AND 15m aggregate)
        assert row["volume_1h"].iloc[0] == 0.0
        assert row["volume_15m"].iloc[0] == 0.0

    def test_end_to_end_passes_with_hole(self) -> None:
        hole = pd.Timestamp("2020-01-01 10:00", tz="UTC")
        # expected_rows=None: this synthetic set is 48h, not the real 35,064.
        master, results = build_and_verify_master(
            self._dfs_with_hole(hole), raise_on_failure=True, expected_rows=None
        )
        assert all(r.passed for r in results)  # V-LEAK-001 still holds


# --- build_and_verify_master (end-to-end M3 orchestration) ---------------------------

class TestBuildAndVerifyMaster:
    def test_clean_data_passes_end_to_end(self, small_multi_tf_dfs) -> None:
        master, results = build_and_verify_master(
            small_multi_tf_dfs, raise_on_failure=False, expected_rows=None
        )
        assert len(master) == 48
        assert len(results) > 0
        assert all(r.passed for r in results)

    def test_raises_on_lookahead_failure(self, small_multi_tf_dfs) -> None:
        """
        Simulate a genuinely broken TemporalAligner (via monkeypatching
        forward_fill_to_1h to inject a bug) and confirm
        build_and_verify_master raises AlignmentError rather than
        silently producing a leaky master DataFrame.
        """
        import src.data.alignment as alignment_module

        original_method = alignment_module.TemporalAligner.forward_fill_to_1h

        def broken_forward_fill(self, df, source_tf, target_index):
            result = original_method(self, df, source_tf, target_index)
            if source_tf == "4h" and len(result) > 4:
                # Shift open values backward by one row to simulate
                # look-ahead: hour 0 gets hour 4's (future) value.
                result = result.copy()
                result["open"] = result["open"].shift(-4)
            return result

        try:
            alignment_module.TemporalAligner.forward_fill_to_1h = broken_forward_fill
            with pytest.raises(AlignmentError, match="V-LEAK-001"):
                build_and_verify_master(
                    small_multi_tf_dfs, raise_on_failure=True, expected_rows=None
                )
        finally:
            alignment_module.TemporalAligner.forward_fill_to_1h = original_method

    def test_does_not_raise_when_disabled(self, small_multi_tf_dfs) -> None:
        # Should not raise even if we don't check the result explicitly.
        build_and_verify_master(
            small_multi_tf_dfs, raise_on_failure=False, expected_rows=None
        )
