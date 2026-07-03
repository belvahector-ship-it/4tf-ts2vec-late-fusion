#!/usr/bin/env python3
"""
scripts/run_m3_alignment.py

Thin CLI entrypoint for M3 — Temporal Alignment.

Usage
-----
    python scripts/run_m3_alignment.py --config configs/base.yaml

Loads the four validated raw Parquet files from `data/raw/`, builds the
aligned 1h-anchor master DataFrame, verifies V-LEAK-001 (no look-ahead
in forward-fill), and writes `data/interim/btc_aligned_1h.parquet`.

This is the mandatory leakage gate referenced in IMP-01 Coding Order:
no downstream module (M4 onward) should consume the master Parquet
until this script exits 0.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.alignment import AlignmentError, build_and_verify_master
from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.utils.paths import BASE_CONFIG_PATH, DATA_INTERIM_DIR, DATA_RAW_DIR

logger = get_logger(__name__)


def main() -> int:
    """
    Run M3 alignment and return a process exit code.

    Returns
    -------
    int
        0 if alignment succeeds and V-LEAK-001 passes, 1 otherwise.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="M3 — Align 15m/1h/4h/1d OHLCV data to a 1h-anchor master."
    )
    parser.add_argument("--config", type=Path, default=BASE_CONFIG_PATH)
    parser.add_argument("--raw-dir", type=Path, default=DATA_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DATA_INTERIM_DIR)
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error("Failed to load config from %s: %s", args.config, e)
        return 1

    timeframes = tuple(config["dataset"]["timeframes"])

    dataframes: dict[str, pd.DataFrame] = {}
    for tf in timeframes:
        path = args.raw_dir / f"btc_{tf}_raw.parquet"
        if not path.exists():
            logger.error(
                "Missing raw file for timeframe=%s at %s. Run M1 and M2 first.",
                tf,
                path,
            )
            return 1
        dataframes[tf] = pd.read_parquet(path)

    try:
        master, lookahead_results = build_and_verify_master(
            dataframes, raise_on_failure=True
        )
    except AlignmentError as e:
        logger.error("M3 alignment FAILED:\n%s", e)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "btc_aligned_1h.parquet"
    master.to_parquet(output_path, engine="pyarrow", index=False)

    logger.info(
        "M3 alignment complete. Master DataFrame: %d rows x %d columns -> %s",
        len(master),
        len(master.columns),
        output_path,
    )
    logger.info(
        "V-LEAK-001: %d/%d sampled timestamps passed look-ahead check.",
        sum(r.passed for r in lookahead_results),
        len(lookahead_results),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
