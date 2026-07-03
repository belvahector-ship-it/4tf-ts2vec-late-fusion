#!/usr/bin/env python3
"""
scripts/run_m8_training.py

Thin CLI entrypoint for M8 — Branch Training.

Usage
-----
    python scripts/run_m8_training.py                 # all 4 TF x 5 seeds
    python scripts/run_m8_training.py --seeds 42      # a single seed
    python scripts/run_m8_training.py --timeframes 1h # a single timeframe

Trains the four independent TS2Vec branch encoders (ADR-002) under the
five-seed protocol (ADR-019), writing best/latest ADR-010 checkpoints
under ``checkpoints/branch_{tf}/seed_{seed}/`` and a per-run log under
``logs/``. Idempotent: already-completed runs are skipped (resume).

PREREQUISITE: the M6 window files ``data/processed/train_windows_{tf}.npy``
must exist (run the M1->M6 pipeline first, and ensure the V-LEAK gate
passed). This script trains on M6 output, never on raw data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.branch_training import (  # noqa: E402
    BranchTrainer,
    BranchTrainingError,
    TrainingOrchestrator,
)
from src.utils.config import load_config  # noqa: E402
from src.utils.device import get_device, get_device_info  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402
from src.utils.paths import BASE_CONFIG_PATH, RANDOM_SEEDS, TIMEFRAMES  # noqa: E402

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="M8 — Train the 4 TS2Vec branch encoders across all seeds."
    )
    parser.add_argument("--config", type=Path, default=BASE_CONFIG_PATH)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(RANDOM_SEEDS),
        help="Seeds to run (default: the five-seed protocol).",
    )
    parser.add_argument(
        "--timeframes", type=str, nargs="+", default=list(TIMEFRAMES),
        help="Timeframes to run (default: all four).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()
    logger.info("M8 device: %s", get_device_info())

    trainer = BranchTrainer(config=config, device=device)
    orchestrator = TrainingOrchestrator(trainer)

    try:
        results = orchestrator.run_all(
            seeds=tuple(args.seeds),
            timeframes=tuple(args.timeframes),
        )
    except BranchTrainingError as exc:
        logger.error("M8 branch training could not start: %s", exc)
        return 1

    n_fail = sum(1 for v in results.values() if v is None)
    if n_fail:
        logger.error("M8 finished with %d failed run(s) — see logs above.", n_fail)
        return 1
    logger.info("M8 complete: %d branch checkpoints written/verified.", len(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
