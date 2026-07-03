"""
tests/test_acquisition.py

Unit tests for src/data/acquisition.py (M1 — Data Acquisition).

Design note
-----------
These tests deliberately avoid any real network call to Binance. Pure
logic (checksum computation, row-count tolerance, manifest building,
Parquet round-trip, idempotency detection) is tested directly against
synthetic DataFrames. The network-dependent `BinanceDownloader.download`
and `_fetch_with_retry` methods are tested using a mocked ccxt exchange
object, so retry/backoff behavior is verified without requiring
internet access — consistent with this repository's sandbox
constraints (see docs/CHECKPOINT_LATEST.md).

Covers IMP-01 M1 Definition of Done:
- Parquet files exist and are readable
- Row counts within 5% of expected values (V-DATA-001)
- manifest.json contains all required fields
- Re-run is idempotent (skips download if checksum matches)
- Download failures raise informative exceptions
- No print() calls (logging only) — verified by code inspection, see
  test_no_print_statements below
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.acquisition import (
    EXPECTED_ROW_COUNTS,
    RAW_COLUMNS,
    AcquisitionError,
    BinanceDownloader,
    DownloadResult,
    build_manifest,
    check_row_count_tolerance,
    is_already_downloaded,
    load_manifest,
    save_manifest,
)


# --- Fixtures ----------------------------------------------------------------

@pytest.fixture
def synthetic_1h_df() -> pd.DataFrame:
    """A small but schema-correct synthetic 1h OHLCV DataFrame."""
    timestamps = pd.date_range(
        "2020-01-01", periods=100, freq="1h", tz="UTC"
    )
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0 + i for i in range(100)],
            "high": [101.0 + i for i in range(100)],
            "low": [99.0 + i for i in range(100)],
            "close": [100.5 + i for i in range(100)],
            "volume": [10.0 + i * 0.1 for i in range(100)],
        }
    )


@pytest.fixture
def downloader() -> BinanceDownloader:
    return BinanceDownloader(symbol="BTC/USDT", exchange_id="binance")


# --- check_row_count_tolerance ------------------------------------------------

class TestRowCountTolerance:
    def test_exact_expected_count_passes(self) -> None:
        assert check_row_count_tolerance("1h", EXPECTED_ROW_COUNTS["1h"])

    def test_within_5_percent_passes(self) -> None:
        expected = EXPECTED_ROW_COUNTS["1h"]
        assert check_row_count_tolerance("1h", int(expected * 1.04))
        assert check_row_count_tolerance("1h", int(expected * 0.96))

    def test_outside_5_percent_fails(self) -> None:
        expected = EXPECTED_ROW_COUNTS["1h"]
        assert not check_row_count_tolerance("1h", int(expected * 1.10))
        assert not check_row_count_tolerance("1h", int(expected * 0.80))

    @pytest.mark.parametrize("timeframe", ["15m", "1h", "4h", "1d"])
    def test_all_four_timeframes_have_expected_counts(self, timeframe: str) -> None:
        assert timeframe in EXPECTED_ROW_COUNTS
        assert check_row_count_tolerance(timeframe, EXPECTED_ROW_COUNTS[timeframe])


# --- BinanceDownloader.compute_checksum ---------------------------------------

class TestComputeChecksum:
    def test_checksum_is_deterministic(self, tmp_path: Path) -> None:
        file_path = tmp_path / "test.parquet"
        file_path.write_bytes(b"some content")
        checksum_a = BinanceDownloader.compute_checksum(file_path)
        checksum_b = BinanceDownloader.compute_checksum(file_path)
        assert checksum_a == checksum_b

    def test_checksum_changes_with_content(self, tmp_path: Path) -> None:
        file_a = tmp_path / "a.parquet"
        file_b = tmp_path / "b.parquet"
        file_a.write_bytes(b"content A")
        file_b.write_bytes(b"content B")
        assert BinanceDownloader.compute_checksum(
            file_a
        ) != BinanceDownloader.compute_checksum(file_b)

    def test_checksum_is_sha256_hex_length(self, tmp_path: Path) -> None:
        file_path = tmp_path / "test.parquet"
        file_path.write_bytes(b"some content")
        checksum = BinanceDownloader.compute_checksum(file_path)
        assert len(checksum) == 64  # SHA-256 hex digest length
        int(checksum, 16)  # must be valid hex


# --- save_parquet / round-trip -------------------------------------------------

class TestSaveParquet:
    def test_saved_parquet_is_readable(
        self, downloader: BinanceDownloader, synthetic_1h_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        path = tmp_path / "btc_1h_raw.parquet"
        downloader.save_parquet(synthetic_1h_df, path)
        assert path.exists()
        loaded = pd.read_parquet(path)
        assert len(loaded) == len(synthetic_1h_df)
        assert list(loaded.columns) == list(RAW_COLUMNS)

    def test_creates_parent_directories(
        self, downloader: BinanceDownloader, synthetic_1h_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        path = tmp_path / "nested" / "dir" / "btc_1h_raw.parquet"
        downloader.save_parquet(synthetic_1h_df, path)
        assert path.exists()


# --- build_manifest / save_manifest / load_manifest ---------------------------

class TestManifest:
    def _make_result(self, df: pd.DataFrame, path: Path) -> DownloadResult:
        return DownloadResult(
            timeframe="1h",
            dataframe=df,
            path=path,
            sha256="a" * 64,
            download_timestamp="2026-07-03T00:00:00+00:00",
            ccxt_version="4.3.41",
        )

    def test_build_manifest_contains_required_fields(
        self, synthetic_1h_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        result = self._make_result(synthetic_1h_df, tmp_path / "btc_1h_raw.parquet")
        manifest = build_manifest({"1h": result})

        assert manifest["symbol"]
        assert manifest["exchange"] == "binance"
        assert "download_timestamp" in manifest
        assert manifest["ccxt_version"] == "4.3.41"
        assert "1h" in manifest["timeframes"]

        tf_entry = manifest["timeframes"]["1h"]
        assert tf_entry["rows"] == len(synthetic_1h_df)
        assert tf_entry["file"] == "btc_1h_raw.parquet"
        assert tf_entry["sha256"] == "a" * 64
        assert "start" in tf_entry
        assert "end" in tf_entry
        assert "within_tolerance" in tf_entry

    def test_build_manifest_empty_results_raises(self) -> None:
        with pytest.raises(ValueError):
            build_manifest({})

    def test_save_and_load_manifest_roundtrip(
        self, synthetic_1h_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        result = self._make_result(synthetic_1h_df, tmp_path / "btc_1h_raw.parquet")
        manifest = build_manifest({"1h": result})

        manifest_path = tmp_path / "manifest.json"
        save_manifest(manifest, manifest_path)
        assert manifest_path.exists()

        loaded = load_manifest(manifest_path)
        assert loaded == manifest

    def test_load_manifest_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert load_manifest(tmp_path / "does_not_exist.json") is None

    def test_manifest_is_valid_json(
        self, synthetic_1h_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        result = self._make_result(synthetic_1h_df, tmp_path / "btc_1h_raw.parquet")
        manifest = build_manifest({"1h": result})
        manifest_path = tmp_path / "manifest.json"
        save_manifest(manifest, manifest_path)

        with open(manifest_path) as f:
            reloaded = json.load(f)  # must not raise
        assert reloaded["exchange"] == "binance"


# --- is_already_downloaded (idempotency) ---------------------------------------

class TestIdempotency:
    def test_no_manifest_means_not_downloaded(self, tmp_path: Path) -> None:
        path = tmp_path / "btc_1h_raw.parquet"
        path.write_bytes(b"content")
        assert not is_already_downloaded("1h", path, manifest=None)

    def test_missing_file_means_not_downloaded(self, tmp_path: Path) -> None:
        path = tmp_path / "does_not_exist.parquet"
        manifest = {"timeframes": {"1h": {"sha256": "irrelevant"}}}
        assert not is_already_downloaded("1h", path, manifest=manifest)

    def test_matching_checksum_means_already_downloaded(self, tmp_path: Path) -> None:
        path = tmp_path / "btc_1h_raw.parquet"
        path.write_bytes(b"deterministic content")
        actual_checksum = BinanceDownloader.compute_checksum(path)
        manifest = {"timeframes": {"1h": {"sha256": actual_checksum}}}
        assert is_already_downloaded("1h", path, manifest=manifest)

    def test_mismatched_checksum_means_not_already_downloaded(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "btc_1h_raw.parquet"
        path.write_bytes(b"content that changed since manifest was written")
        manifest = {"timeframes": {"1h": {"sha256": "0" * 64}}}
        assert not is_already_downloaded("1h", path, manifest=manifest)

    def test_timeframe_not_in_manifest_means_not_downloaded(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "btc_4h_raw.parquet"
        path.write_bytes(b"content")
        manifest = {"timeframes": {"1h": {"sha256": "irrelevant"}}}
        assert not is_already_downloaded("4h", path, manifest=manifest)


# --- BinanceDownloader.download (mocked ccxt, no real network) ----------------

class TestDownloadWithMockedExchange:
    """
    These tests verify retry/backoff and error-handling logic by
    mocking the ccxt exchange object entirely. No real network request
    is made, consistent with this sandbox's lack of internet access.
    Anyone running these tests with real ccxt installed will exercise
    the actual retry code paths against a fake exchange.
    """

    def test_unknown_timeframe_raises_value_error(
        self, downloader: BinanceDownloader
    ) -> None:
        with pytest.raises(ValueError, match="Unknown timeframe"):
            downloader.download("30m", "2020-01-01", "2020-01-02")

    def test_empty_response_raises_acquisition_error(
        self, downloader: BinanceDownloader
    ) -> None:
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = []

        with patch.object(downloader, "_get_exchange", return_value=mock_exchange):
            with pytest.raises(AcquisitionError, match="No candles returned"):
                downloader.download("1h", "2020-01-01", "2020-01-02")

    def test_successful_single_page_download(
        self, downloader: BinanceDownloader
    ) -> None:
        # Fewer than _MAX_CANDLES_PER_REQUEST rows -> loop stops after one page.
        mock_candles = [
            [1577836800000 + i * 3_600_000, 100.0, 101.0, 99.0, 100.5, 10.0]
            for i in range(24)
        ]
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = mock_candles

        with patch.object(downloader, "_get_exchange", return_value=mock_exchange):
            df = downloader.download("1h", "2020-01-01", "2020-01-02")

        assert len(df) == 24
        assert list(df.columns) == list(RAW_COLUMNS)
        assert df["timestamp"].is_monotonic_increasing

    def test_retries_on_network_error_then_succeeds(
        self, downloader: BinanceDownloader
    ) -> None:
        import ccxt

        mock_candles = [
            [1577836800000, 100.0, 101.0, 99.0, 100.5, 10.0],
        ]
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.side_effect = [
            ccxt.NetworkError("transient failure"),
            mock_candles,
        ]

        with patch.object(downloader, "_get_exchange", return_value=mock_exchange):
            with patch("time.sleep"):  # skip real backoff delay in test
                df = downloader.download("1h", "2020-01-01", "2020-01-02")

        assert len(df) == 1
        assert mock_exchange.fetch_ohlcv.call_count == 2

    def test_exchange_error_fails_immediately_without_exhausting_retries(
        self, downloader: BinanceDownloader
    ) -> None:
        import ccxt

        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.side_effect = ccxt.ExchangeError("invalid symbol")

        with patch.object(downloader, "_get_exchange", return_value=mock_exchange):
            with pytest.raises(AcquisitionError, match="Exchange API error"):
                downloader.download("1h", "2020-01-01", "2020-01-02")

        # ExchangeError is not transient — must fail on first attempt,
        # not retry 5 times.
        assert mock_exchange.fetch_ohlcv.call_count == 1

    def test_exhausted_retries_raises_acquisition_error(
        self, downloader: BinanceDownloader
    ) -> None:
        import ccxt

        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.side_effect = ccxt.NetworkError("always fails")

        with patch.object(downloader, "_get_exchange", return_value=mock_exchange):
            with patch("time.sleep"):
                with pytest.raises(AcquisitionError, match="Failed to fetch"):
                    downloader.download("1h", "2020-01-01", "2020-01-02")


# --- Code-quality check: no print() calls (IMP-01 M1 DoD) ---------------------

class TestNoPrintStatements:
    def test_acquisition_module_uses_no_print(self) -> None:
        """
        IMP-01 M1 Definition of Done: 'No print() calls — all output
        via logging'. This is a static check on the source file itself.
        """
        source_path = Path(__file__).resolve().parents[1] / "src" / "data" / "acquisition.py"
        source = source_path.read_text(encoding="utf-8")

        for lineno, line in enumerate(source.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert "print(" not in line, (
                f"Found print() call at acquisition.py:{lineno}: {line!r}. "
                "Use src.utils.logging_utils.get_logger instead."
            )
