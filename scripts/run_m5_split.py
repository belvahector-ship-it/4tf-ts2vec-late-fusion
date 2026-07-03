#!/usr/bin/env python3
"""
scripts/run_m5_split.py

Thin CLI entrypoint for M5 — Temporal Split.

Usage
-----
    python scripts/run_m5_split.py

Loads `data/processed/btc_features_all.parquet` (M4 output), splits it
at the ADR-014 boundary, verifies V-DATA-005 and V-LEAK-002, and writes
`data/processed/train_features.parquet` and `test_features.parquet`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.temporal_split import TemporalSplitError, run_temporal_split
from src.utils.logging_utils import get_logger
from src.utils.paths import DATA_PROCESSED_DIR

logger = get_logger(__name__)


def main() -> int:
    """
    Run M5 temporal split and return a process exit code.

    Returns
    -------
    int
        0 on success, 1 on failure (missing input, boundary/overlap error).
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="M5 — Split feature matrix into train/test at ADR-014 boundary."
    )
    parser.add_argument(
        "--input", type=Path, default=DATA_PROCESSED_DIR / "btc_features_all.parquet"
    )
    parser.add_argument("--output-dir", type=Path, default=DATA_PROCESSED_DIR)
    args = parser.parse_args()

    if not args.input.exists():
        logger.error(
            "Missing M4 output at %s. Run M4 (scripts/run_m4_features.py) first.",
            args.input,
        )
        return 1

    features_df = pd.read_parquet(args.input)

    try:
        result = run_temporal_split(features_df, raise_on_failure=True)
    except TemporalSplitError as e:
        logger.error("M5 temporal split FAILED:\n%s", e)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train_features.parquet"
    test_path = args.output_dir / "test_features.parquet"
    result.train.to_parquet(train_path, engine="pyarrow", index=False)
    result.test.to_parquet(test_path, engine="pyarrow", index=False)

    logger.info(
        "M5 split complete: train=%d rows -> %s, test=%d rows -> %s",
        len(result.train),
        train_path,
        len(result.test),
        test_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
