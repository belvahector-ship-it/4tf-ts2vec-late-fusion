#!/usr/bin/env python3
"""
scripts/run_m6_windows.py

Thin CLI entrypoint for M6 — Window Generation.

Usage
-----
    python scripts/run_m6_windows.py

Loads `data/processed/btc_features_all.parquet` (M4 output — the FULL
feature matrix, NOT the M5 split files), generates sliding-window
tensors for all 4 timeframes, normalizes per-window, categorizes each
window as train/test by its anchor timestamp, verifies V-LEAK-003 and
V-LEAK-004, and writes the 8 window `.npy` files plus
`train_timestamps.npy` / `test_timestamps.npy`.

This is the mandatory V-LEAK gate referenced in IMP-01: no training
(M8) should begin until this script exits 0.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data.window_generation import (
    TIMEFRAME_SUFFIXES,
    WindowGenerationError,
    run_window_generation,
)
from src.utils.logging_utils import get_logger
from src.utils.paths import DATA_PROCESSED_DIR

logger = get_logger(__name__)


def main() -> int:
    """
    Run M6 window generation and return a process exit code.

    Returns
    -------
    int
        0 if generation succeeds and all leakage checks pass, 1 otherwise.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="M6 — Generate normalized sliding windows for all timeframes."
    )
    parser.add_argument(
        "--input", type=Path, default=DATA_PROCESSED_DIR / "btc_features_all.parquet"
    )
    parser.add_argument("--output-dir", type=Path, default=DATA_PROCESSED_DIR)
    args = parser.parse_args()

    if not args.input.exists():
        logger.error(
            "Missing M4 output at %s. Run M4 (scripts/run_m4_features.py) first. "
            "Note: M6 reads the FULL feature matrix from M4, not the M5 split files.",
            args.input,
        )
        return 1

    features_df = pd.read_parquet(args.input)

    try:
        result = run_window_generation(features_df, raise_on_failure=True)
    except WindowGenerationError as e:
        logger.error("M6 window generation FAILED:\n%s", e)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "test"):
        for tf in TIMEFRAME_SUFFIXES:
            window_set = result[split][tf]
            out_path = args.output_dir / f"{split}_windows_{tf}.npy"
            np.save(out_path, window_set.windows)
            logger.info("Saved %s -> shape %s", out_path, window_set.windows.shape)

    # Anchor timestamps are identical across timeframes (all branches
    # share the same 1h anchor), so save once per split.
    train_timestamps = result["train"][TIMEFRAME_SUFFIXES[0]].anchor_timestamps
    test_timestamps = result["test"][TIMEFRAME_SUFFIXES[0]].anchor_timestamps
    np.save(args.output_dir / "train_timestamps.npy", train_timestamps)
    np.save(args.output_dir / "test_timestamps.npy", test_timestamps)

    logger.info("M6 window generation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
