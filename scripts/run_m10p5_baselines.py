#!/usr/bin/env python3
"""
scripts/run_m10p5_baselines.py

Thin CLI entrypoint for M10.5 — External Baselines (HMM + KM-PCA).

Usage
-----
    python scripts/run_m10p5_baselines.py
    python scripts/run_m10p5_baselines.py --exp-id my_run --seeds 42 123

Runs both external baselines across all five seeds (5 HMM + 5 KM-PCA =
10 runs) on the M5 1h feature split, writing labels/models/selection
tables under experiments/{exp_id}/external_baselines/{hmm,kmpca}/.

PREREQUISITE: M5 outputs data/processed/{train,test}_features.parquet
(run M1-M5 first). Independent of M6/M7/M8/M9 (no TS2Vec).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.models.external_baselines import ExternalBaselineRunner  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402
from src.utils.paths import (  # noqa: E402
    BASE_CONFIG_PATH,
    DATA_PROCESSED_DIR,
    RANDOM_SEEDS,
    get_experiment_dir,
)

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="M10.5 — External baselines (HMM + KM-PCA) five-seed protocol."
    )
    parser.add_argument("--config", type=Path, default=BASE_CONFIG_PATH)
    parser.add_argument("--exp-id", type=str, default="m10p5_external_baselines")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(RANDOM_SEEDS))
    args = parser.parse_args()

    train_path = DATA_PROCESSED_DIR / "train_features.parquet"
    test_path = DATA_PROCESSED_DIR / "test_features.parquet"
    for p in (train_path, test_path):
        if not p.exists():
            logger.error("Missing M5 output %s. Run M1-M5 first.", p)
            return 1

    config = load_config(args.config)
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    logger.info("Loaded M5 features: train=%d, test=%d rows", len(train_df), len(test_df))

    out_dir = get_experiment_dir(args.exp_id) / "external_baselines"
    runner = ExternalBaselineRunner(config, output_dir=out_dir)
    results = runner.run_all(train_df, test_df, seeds=tuple(args.seeds))

    logger.info("M10.5 complete. Summary per seed:")
    for seed, summ in results.items():
        logger.info(
            "  seed=%-4d | HMM n_components=%d | KM-PCA k=%d, PCA=%d",
            seed, summ["hmm"]["n_components"],
            summ["kmpca"]["k"], summ["kmpca"]["pca_components"],
        )
    logger.info("Outputs written under %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
