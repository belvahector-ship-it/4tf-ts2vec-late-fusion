#!/usr/bin/env python3
"""
scripts/run_m4_features.py

Thin CLI entrypoint for M4 — Feature Engineering.

Usage
-----
    python scripts/run_m4_features.py --config configs/base.yaml

Loads `data/interim/btc_aligned_1h.parquet` (M3 output), computes the 7
ADR-015 features for all 4 timeframes, drops the leading NaN rows, and
writes `data/processed/btc_features_all.parquet`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.feature_engineering import FeatureEngineeringError, run_feature_engineering
from src.utils.logging_utils import get_logger
from src.utils.paths import DATA_INTERIM_DIR, DATA_PROCESSED_DIR

logger = get_logger(__name__)


def main() -> int:
    """
    Run M4 feature engineering and return a process exit code.

    Returns
    -------
    int
        0 on success, 1 on failure (missing input, schema mismatch).
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="M4 — Compute ADR-015 OHLCV-derived features."
    )
    parser.add_argument(
        "--input", type=Path, default=DATA_INTERIM_DIR / "btc_aligned_1h.parquet"
    )
    parser.add_argument("--output-dir", type=Path, default=DATA_PROCESSED_DIR)
    args = parser.parse_args()

    if not args.input.exists():
        logger.error(
            "Missing M3 output at %s. Run M3 (scripts/run_m3_alignment.py) first.",
            args.input,
        )
        return 1

    master_df = pd.read_parquet(args.input)

    try:
        features = run_feature_engineering(master_df, raise_on_failure=True)
    except FeatureEngineeringError as e:
        logger.error("M4 feature engineering FAILED:\n%s", e)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "btc_features_all.parquet"
    features.to_parquet(output_path, engine="pyarrow", index=False)

    logger.info(
        "M4 feature engineering complete: %d rows x %d columns -> %s",
        len(features),
        len(features.columns),
        output_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
