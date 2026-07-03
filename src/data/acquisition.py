"""
src/data/acquisition.py

Data acquisition module (M1 — Data Acquisition, DS-02 Stage 0).

Purpose
-------
Downloads BTC/USDT OHLCV klines from Binance for all four timeframes
(15m, 1h, 4h, 1d) covering 2020-01-01 to 2023-12-31 UTC, and produces
raw Parquet files with full download provenance metadata in
`data/raw/manifest.json`.

Per DS-02 Stage 0 / Leakage Checkpoint LC-1: this module performs NO
imputation, interpolation, or forward-fill. Gaps are logged only, not
filled — that happens exclusively at Stage 2 (M3, Temporal Alignment),
so leakage auditing stays unambiguous (one place upsampling can occur).

Idempotency
-----------
Per IMP-01 M1 Definition of Done, re-running this module is safe: if a
target Parquet file already exists and its checksum matches the
manifest, the download is skipped rather than repeated.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logging_utils import get_logger
from src.utils.paths import DATA_RAW_DIR, RAW_MANIFEST_PATH

logger = get_logger(__name__)

# Raw OHLCV column schema per DS-02 Stage 0.
RAW_COLUMNS: tuple[str, ...] = ("timestamp", "open", "high", "low", "close", "volume")

# Expected row counts per DS-04 V-DATA-001 / DS-02 Stage 0 table.
# Tolerance is 5% per V-DATA-001 pass criteria.
EXPECTED_ROW_COUNTS: dict[str, int] = {
    "15m": 140_256,
    "1h": 35_064,
    "4h": 8_766,
    "1d": 1_461,
}
ROW_COUNT_TOLERANCE: float = 0.05

# ccxt timeframe string mapping (Binance uses these directly).
CCXT_TIMEFRAME_MAP: dict[str, str] = {"15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}

# Binance's public klines endpoint caps each request at 1000 candles.
_MAX_CANDLES_PER_REQUEST = 1000

# Exponential backoff parameters for rate-limit / transient network errors
# (IMP-01 Risk R-07).
_MAX_RETRIES = 5
_INITIAL_BACKOFF_SECONDS = 2.0
_BACKOFF_MULTIPLIER = 2.0


class AcquisitionError(RuntimeError):
    """Raised when data acquisition fails after exhausting retries."""


@dataclass
class DownloadResult:
    """Result of downloading and saving one timeframe."""

    timeframe: str
    dataframe: pd.DataFrame
    path: Path
    sha256: str
    download_timestamp: str
    ccxt_version: str
    row_count: int = field(init=False)
    start: str = field(init=False)
    end: str = field(init=False)

    def __post_init__(self) -> None:
        self.row_count = len(self.dataframe)
        self.start = self.dataframe["timestamp"].iloc[0].isoformat()
        self.end = self.dataframe["timestamp"].iloc[-1].isoformat()


class BinanceDownloader:
    """
    Downloads BTC/USDT OHLCV klines from Binance via ccxt.

    Parameters
    ----------
    symbol : str
        Trading pair symbol, e.g. "BTC/USDT".
    exchange_id : str
        ccxt exchange identifier, e.g. "binance".
    """

    def __init__(self, symbol: str = "BTC/USDT", exchange_id: str = "binance") -> None:
        self.symbol = symbol
        self.exchange_id = exchange_id
        self._exchange = None  # lazily constructed; see _get_exchange

    def _get_exchange(self):
        """
        Lazily construct and return the ccxt exchange instance.

        Lazy construction means this module can be imported (and its
        pure functions like `compute_checksum` unit-tested) in
        environments where `ccxt` is not installed or network access
        is unavailable, as long as `download()` itself is never called.

        Raises
        ------
        ImportError
            If the `ccxt` package is not installed.
        """
        if self._exchange is None:
            try:
                import ccxt
            except ImportError as e:
                raise ImportError(
                    "The 'ccxt' package is required for data acquisition "
                    "but is not installed. Install it with "
                    "`pip install ccxt` or via requirements.txt."
                ) from e
            self._exchange = getattr(ccxt, self.exchange_id)()
        return self._exchange

    def download(self, timeframe: str, start: str, end: str) -> pd.DataFrame:
        """
        Download OHLCV candles for one timeframe over [start, end].

        Parameters
        ----------
        timeframe : str
            One of "15m", "1h", "4h", "1d".
        start : str
            Start date, "YYYY-MM-DD" (UTC, inclusive).
        end : str
            End date, "YYYY-MM-DD" (UTC, inclusive of the full day).

        Returns
        -------
        pd.DataFrame
            Columns: timestamp (datetime64[ns, UTC]), open, high, low,
            close, volume (all float64). Sorted ascending by timestamp,
            with duplicate timestamps dropped (ccxt pagination can
            occasionally return an overlapping boundary candle).

        Raises
        ------
        ValueError
            If `timeframe` is not one of the four recognized values.
        AcquisitionError
            If the download fails after exhausting all retries.
        """
        if timeframe not in CCXT_TIMEFRAME_MAP:
            raise ValueError(
                f"Unknown timeframe '{timeframe}'. "
                f"Expected one of {list(CCXT_TIMEFRAME_MAP)}."
            )

        exchange = self._get_exchange()
        ccxt_tf = CCXT_TIMEFRAME_MAP[timeframe]

        since = int(
            datetime.strptime(start, "%Y-%m-%d")
            .replace(tzinfo=timezone.utc)
            .timestamp()
            * 1000
        )
        end_ms = int(
            datetime.strptime(end, "%Y-%m-%d")
            .replace(tzinfo=timezone.utc, hour=23, minute=59, second=59)
            .timestamp()
            * 1000
        )

        all_candles: list[list[Any]] = []
        cursor = since
        logger.info(
            "Starting download: symbol=%s timeframe=%s start=%s end=%s",
            self.symbol,
            timeframe,
            start,
            end,
        )

        while cursor < end_ms:
            batch = self._fetch_with_retry(exchange, ccxt_tf, cursor)
            if not batch:
                break

            all_candles.extend(batch)
            last_candle_ts = batch[-1][0]

            # Advance cursor past the last received candle to avoid
            # re-fetching it. If the exchange returns no forward
            # progress (last timestamp == cursor), stop to avoid an
            # infinite loop.
            if last_candle_ts <= cursor:
                break
            cursor = last_candle_ts + 1

            if len(batch) < _MAX_CANDLES_PER_REQUEST:
                # Fewer candles than the page size means we've reached
                # the end of available data.
                break

        if not all_candles:
            raise AcquisitionError(
                f"No candles returned for timeframe={timeframe}, "
                f"symbol={self.symbol}, range=[{start}, {end}]. "
                "Check date range, symbol spelling, and exchange status."
            )

        df = pd.DataFrame(all_candles, columns=list(RAW_COLUMNS))
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
        df = df.reset_index(drop=True)

        for col in ("open", "high", "low", "close", "volume"):
            df[col] = df[col].astype("float64")

        logger.info(
            "Completed download: timeframe=%s rows=%d first=%s last=%s",
            timeframe,
            len(df),
            df["timestamp"].iloc[0],
            df["timestamp"].iloc[-1],
        )
        return df

    def _fetch_with_retry(
        self, exchange, ccxt_tf: str, since_ms: int
    ) -> list[list[Any]]:
        """
        Fetch one page of OHLCV candles with exponential backoff retry.

        Parameters
        ----------
        exchange : ccxt.Exchange
            An instantiated ccxt exchange object.
        ccxt_tf : str
            ccxt-formatted timeframe string.
        since_ms : int
            Start timestamp in Unix milliseconds.

        Returns
        -------
        list[list]
            Raw OHLCV rows as returned by ccxt:
            [timestamp_ms, open, high, low, close, volume].

        Raises
        ------
        AcquisitionError
            If all retries are exhausted without success.
        """
        import ccxt

        backoff = _INITIAL_BACKOFF_SECONDS
        last_exception: Exception | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                return exchange.fetch_ohlcv(
                    self.symbol,
                    timeframe=ccxt_tf,
                    since=since_ms,
                    limit=_MAX_CANDLES_PER_REQUEST,
                )
            except ccxt.RateLimitExceeded as e:
                last_exception = e
                logger.warning(
                    "Rate limit hit (attempt %d/%d). Backing off %.1fs.",
                    attempt,
                    _MAX_RETRIES,
                    backoff,
                )
            except ccxt.NetworkError as e:
                last_exception = e
                logger.warning(
                    "Network error (attempt %d/%d): %s. Backing off %.1fs.",
                    attempt,
                    _MAX_RETRIES,
                    e,
                    backoff,
                )
            except ccxt.ExchangeError as e:
                # Exchange-level errors (bad symbol, invalid params) are
                # not transient — fail immediately with an informative
                # message rather than retrying uselessly.
                raise AcquisitionError(
                    f"Exchange API error while fetching {self.symbol} "
                    f"{ccxt_tf} candles: {e}"
                ) from e

            time.sleep(backoff)
            backoff *= _BACKOFF_MULTIPLIER

        raise AcquisitionError(
            f"Failed to fetch OHLCV data for {self.symbol} {ccxt_tf} after "
            f"{_MAX_RETRIES} attempts. Last error: {last_exception}"
        )

    def save_parquet(self, df: pd.DataFrame, path: Path) -> None:
        """
        Save a validated OHLCV DataFrame to Parquet.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame with RAW_COLUMNS.
        path : Path
            Destination Parquet file path. Parent directories are
            created if missing.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, engine="pyarrow", index=False)
        logger.info("Saved %d rows to %s", len(df), path)

    @staticmethod
    def compute_checksum(path: Path) -> str:
        """
        Compute the SHA-256 checksum of a file.

        Parameters
        ----------
        path : Path
            File to checksum.

        Returns
        -------
        str
            Hex-encoded SHA-256 digest.
        """
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()


def check_row_count_tolerance(timeframe: str, actual_rows: int) -> bool:
    """
    Check whether `actual_rows` is within tolerance of the expected count.

    Parameters
    ----------
    timeframe : str
        One of "15m", "1h", "4h", "1d".
    actual_rows : int
        The row count actually downloaded.

    Returns
    -------
    bool
        True if `actual_rows` is within ROW_COUNT_TOLERANCE (5%, per
        DS-04 V-DATA-001) of EXPECTED_ROW_COUNTS[timeframe].
    """
    expected = EXPECTED_ROW_COUNTS[timeframe]
    lower = expected * (1 - ROW_COUNT_TOLERANCE)
    upper = expected * (1 + ROW_COUNT_TOLERANCE)
    return lower <= actual_rows <= upper


def build_manifest(results: dict[str, DownloadResult]) -> dict[str, Any]:
    """
    Build the manifest.json structure from download results.

    Parameters
    ----------
    results : dict[str, DownloadResult]
        Mapping of timeframe -> DownloadResult, one entry per
        timeframe successfully downloaded.

    Returns
    -------
    dict
        Manifest matching the schema in DS-02 Stage 0: symbol,
        exchange, download_timestamp, ccxt_version, and a
        `timeframes` dict with rows/start/end/file/sha256 per
        timeframe.
    """
    if not results:
        raise ValueError("Cannot build manifest from empty results.")

    any_result = next(iter(results.values()))
    manifest: dict[str, Any] = {
        "symbol": any_result.dataframe.attrs.get("symbol", "BTC/USDT"),
        "exchange": "binance",
        "download_timestamp": datetime.now(timezone.utc).isoformat(),
        "ccxt_version": any_result.ccxt_version,
        "timeframes": {},
    }

    for tf, result in results.items():
        manifest["timeframes"][tf] = {
            "rows": result.row_count,
            "start": result.start,
            "end": result.end,
            "file": result.path.name,
            "sha256": result.sha256,
            "within_tolerance": check_row_count_tolerance(tf, result.row_count),
        }

    return manifest


def save_manifest(manifest: dict[str, Any], path: Path = RAW_MANIFEST_PATH) -> None:
    """
    Write the manifest dict to disk as pretty-printed JSON.

    Parameters
    ----------
    manifest : dict
        Manifest structure from `build_manifest`.
    path : Path, optional
        Destination path, defaults to `data/raw/manifest.json`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=False)
    logger.info("Manifest written to %s", path)


def load_manifest(path: Path = RAW_MANIFEST_PATH) -> dict[str, Any] | None:
    """
    Load an existing manifest.json if present.

    Parameters
    ----------
    path : Path, optional
        Manifest path, defaults to `data/raw/manifest.json`.

    Returns
    -------
    dict or None
        Parsed manifest, or None if the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_already_downloaded(
    timeframe: str, path: Path, manifest: dict[str, Any] | None
) -> bool:
    """
    Check whether a timeframe's Parquet file is already downloaded and valid.

    Used to implement the idempotent re-run behavior required by
    IMP-01 M1 Definition of Done: "if files already exist and
    checksums match, skip re-download."

    Parameters
    ----------
    timeframe : str
        Timeframe to check.
    path : Path
        Expected path of the Parquet file.
    manifest : dict or None
        Previously loaded manifest, or None if no manifest exists yet.

    Returns
    -------
    bool
        True if `path` exists and its current SHA-256 checksum matches
        the checksum recorded in `manifest` for this timeframe.
    """
    if manifest is None or not path.exists():
        return False

    recorded = manifest.get("timeframes", {}).get(timeframe)
    if recorded is None:
        return False

    actual_checksum = BinanceDownloader.compute_checksum(path)
    return actual_checksum == recorded.get("sha256")


def run_acquisition(
    symbol: str = "BTC/USDT",
    exchange_id: str = "binance",
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
    timeframes: tuple[str, ...] = ("15m", "1h", "4h", "1d"),
    output_dir: Path = DATA_RAW_DIR,
    force: bool = False,
) -> dict[str, Any]:
    """
    Run the full M1 acquisition pipeline for all timeframes.

    Parameters
    ----------
    symbol : str, optional
        Trading pair, default "BTC/USDT".
    exchange_id : str, optional
        ccxt exchange id, default "binance".
    start_date : str, optional
        Study start date "YYYY-MM-DD", default "2020-01-01".
    end_date : str, optional
        Study end date "YYYY-MM-DD", default "2023-12-31".
    timeframes : tuple[str, ...], optional
        Timeframes to download, default all four DS-03 timeframes.
    output_dir : Path, optional
        Directory to write Parquet files and manifest.json into,
        default `data/raw/`.
    force : bool, optional
        If True, re-download even if a valid cached file exists.
        Default False (idempotent behavior).

    Returns
    -------
    dict
        The final manifest written to `{output_dir}/manifest.json`.

    Raises
    ------
    AcquisitionError
        If any timeframe's download ultimately fails.
    """
    import ccxt

    downloader = BinanceDownloader(symbol=symbol, exchange_id=exchange_id)
    manifest_path = output_dir / "manifest.json"
    existing_manifest = load_manifest(manifest_path)

    results: dict[str, DownloadResult] = {}

    for tf in timeframes:
        target_path = output_dir / f"btc_{tf}_raw.parquet"

        if not force and is_already_downloaded(tf, target_path, existing_manifest):
            logger.info(
                "Skipping download for timeframe=%s: cached file at %s "
                "matches manifest checksum.",
                tf,
                target_path,
            )
            df = pd.read_parquet(target_path)
            checksum = existing_manifest["timeframes"][tf]["sha256"]
            download_ts = existing_manifest["download_timestamp"]
            ccxt_version = existing_manifest["ccxt_version"]
        else:
            df = downloader.download(tf, start_date, end_date)

            if not check_row_count_tolerance(tf, len(df)):
                logger.warning(
                    "Row count for timeframe=%s (%d) is outside the 5%% "
                    "tolerance of expected value (%d). Proceeding, but "
                    "this will fail V-DATA-001 in M2.",
                    tf,
                    len(df),
                    EXPECTED_ROW_COUNTS[tf],
                )

            downloader.save_parquet(df, target_path)
            checksum = BinanceDownloader.compute_checksum(target_path)
            download_ts = datetime.now(timezone.utc).isoformat()
            ccxt_version = ccxt.__version__

        results[tf] = DownloadResult(
            timeframe=tf,
            dataframe=df,
            path=target_path,
            sha256=checksum,
            download_timestamp=download_ts,
            ccxt_version=ccxt_version,
        )

    manifest = build_manifest(results)
    save_manifest(manifest, manifest_path)

    failed = [tf for tf, r in manifest["timeframes"].items() if not r["within_tolerance"]]
    if failed:
        logger.warning(
            "The following timeframes are outside the DS-04 V-DATA-001 "
            "5%% row-count tolerance: %s. Acquisition completed, but "
            "M2 (Data Validation) will likely reject this data.",
            failed,
        )

    return manifest
