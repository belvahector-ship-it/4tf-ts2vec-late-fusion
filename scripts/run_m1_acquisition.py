#!/usr/bin/env python3
"""
scripts/run_m1_acquisition.py

Thin CLI entrypoint for M1 — Data Acquisition.

Usage
-----
    python scripts/run_m1_acquisition.py --config configs/base.yaml
    python scripts/run_m1_acquisition.py --config configs/base.yaml --force

Reads dataset parameters (symbol, exchange, date range, timeframes)
from the given config file and runs the full acquisition pipeline,
writing Parquet files and manifest.json to `data/raw/`.

Requires network access and the `ccxt` package. If run in an
environment without internet access, this script will raise an
informative `AcquisitionError` or `ImportError` rather than silently
producing partial or fabricated data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/run_m1_acquisition.py` from repo root
# without requiring the package to be installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.acquisition import AcquisitionError, run_acquisition
from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.utils.paths import BASE_CONFIG_PATH, DATA_RAW_DIR

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="M1 — Download BTC/USDT OHLCV data from Binance."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=BASE_CONFIG_PATH,
        help="Path to base config YAML (default: configs/base.yaml).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if a valid cached file already exists.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_RAW_DIR,
        help="Directory to write Parquet files and manifest.json into "
        "(default: data/raw/).",
    )
    return parser.parse_args()


def main() -> int:
    """
    Run M1 acquisition and return a process exit code.

    Returns
    -------
    int
        0 on success, 1 on failure (network error, missing ccxt, etc.).
    """
    args = parse_args()

    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error("Failed to load config from %s: %s", args.config, e)
        return 1

    dataset_cfg = config["dataset"]

    try:
        manifest = run_acquisition(
            symbol=dataset_cfg["symbol"],
            exchange_id=dataset_cfg["exchange"],
            start_date=dataset_cfg["start_date"],
            end_date=dataset_cfg["end_date"],
            timeframes=tuple(dataset_cfg["timeframes"]),
            output_dir=args.output_dir,
            force=args.force,
        )
    except ImportError as e:
        logger.error(
            "Missing dependency: %s. Install requirements with "
            "`pip install -r requirements.txt`.",
            e,
        )
        return 1
    except AcquisitionError as e:
        logger.error("Acquisition failed: %s", e)
        return 1

    logger.info("M1 acquisition complete. Manifest summary:")
    for tf, info in manifest["timeframes"].items():
        status = "OK" if info["within_tolerance"] else "OUT OF TOLERANCE"
        logger.info(
            "  %-4s | rows=%-8d | %s -> %s | %s",
            tf,
            info["rows"],
            info["start"],
            info["end"],
            status,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
