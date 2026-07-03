#!/usr/bin/env python3
"""
scripts/run_m2_validation.py

Thin CLI entrypoint for M2 — Data Validation.

Usage
-----
    python scripts/run_m2_validation.py --config configs/base.yaml

Loads the four raw Parquet files produced by M1 from `data/raw/`, runs
the full DS-02 Stage 1 validation suite on each, appends the results to
`data/raw/manifest.json`, and exits non-zero if any timeframe fails
validation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.acquisition import load_manifest, save_manifest
from src.data.validation import (
    DataValidationError,
    append_validation_to_manifest,
    validate_all_timeframes,
)
from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.utils.paths import BASE_CONFIG_PATH, DATA_RAW_DIR, RAW_MANIFEST_PATH

logger = get_logger(__name__)


def main() -> int:
    """
    Run M2 validation and return a process exit code.

    Returns
    -------
    int
        0 if all timeframes pass validation, 1 otherwise (missing
        files, failed checks, or config errors).
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="M2 — Validate raw OHLCV Parquet files from M1."
    )
    parser.add_argument("--config", type=Path, default=BASE_CONFIG_PATH)
    parser.add_argument("--raw-dir", type=Path, default=DATA_RAW_DIR)
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error("Failed to load config from %s: %s", args.config, e)
        return 1

    dataset_cfg = config["dataset"]
    timeframes = tuple(dataset_cfg["timeframes"])

    dataframes: dict[str, pd.DataFrame] = {}
    for tf in timeframes:
        path = args.raw_dir / f"btc_{tf}_raw.parquet"
        if not path.exists():
            logger.error(
                "Missing raw file for timeframe=%s at %s. Run M1 "
                "(scripts/run_m1_acquisition.py) first.",
                tf,
                path,
            )
            return 1
        dataframes[tf] = pd.read_parquet(path)

    try:
        reports = validate_all_timeframes(
            dataframes,
            start_date=dataset_cfg["start_date"],
            end_date=dataset_cfg["end_date"],
            raise_on_failure=True,
        )
    except DataValidationError as e:
        logger.error("M2 validation FAILED:\n%s", e)
        return 1

    manifest = load_manifest(args.raw_dir / "manifest.json") or {}
    manifest = append_validation_to_manifest(manifest, reports)
    save_manifest(manifest, args.raw_dir / "manifest.json")

    logger.info("M2 validation PASSED for all %d timeframes.", len(reports))
    return 0


if __name__ == "__main__":
    sys.exit(main())
