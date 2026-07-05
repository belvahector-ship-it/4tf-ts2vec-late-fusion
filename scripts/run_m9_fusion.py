#!/usr/bin/env python3
"""
scripts/run_m9_fusion.py

M9 — Fusion runner on REAL M8 branch checkpoints.

For every seed, loads the four trained branch encoders (M8), encodes the
train/test windows (M6) to `[N, 64]` per branch, then produces fused
`[N, 256]` embeddings for all 7 TS2Vec conditions via the deterministic
concat + fixed random projection (ADR-003/ADR-013, M9 `FusionModule`).

Outputs (per seed):
- experiments/{exp_id}/embeddings/branch/embeddings_{split}_{tf}_seed{seed}.npy   [N, 64]
- experiments/{exp_id}/embeddings/fused/embeddings_{split}_{cond}_seed{seed}.npy  [N, 256]

PREREQUISITES:
- M8 checkpoints at checkpoints/branch_{tf}/seed_{seed}/best_model.pt (20)
- M6 windows at data/processed/{split}_windows_{tf}.npy (8)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from src.models.fusion import EmbeddingPipeline  # noqa: E402
from src.models.ts2vec_wrapper import TS2VecBranch  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.device import get_device  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402
from src.utils.paths import (  # noqa: E402
    BASE_CONFIG_PATH,
    CHECKPOINTS_DIR,
    CONFIGS_DIR,
    DATA_PROCESSED_DIR,
    RANDOM_SEEDS,
    TIMEFRAMES,
    TS2VEC_CONDITIONS,
    get_experiment_dir,
)

logger = get_logger(__name__)

# Condition label -> per-condition experiment config file (active_timeframes).
_CONDITION_CONFIG = {
    "1TF": "experiment_1tf.yaml", "2TF": "experiment_2tf.yaml",
    "3TF": "experiment_3tf.yaml", "4TF": "experiment_4tf.yaml",
    "BL-15m": "experiment_bl_15m.yaml", "BL-4h": "experiment_bl_4h.yaml",
    "BL-1d": "experiment_bl_1d.yaml",
}


def _active_timeframes(condition: str) -> list[str]:
    cfg = yaml.safe_load((CONFIGS_DIR / _CONDITION_CONFIG[condition]).read_text())
    return list(cfg["condition"]["active_timeframes"])


def main() -> int:
    parser = argparse.ArgumentParser(description="M9 — fuse real M8 checkpoints.")
    parser.add_argument("--config", type=Path, default=BASE_CONFIG_PATH)
    parser.add_argument("--exp-id", type=str, default="m9_real")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(RANDOM_SEEDS))
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()
    logger.info("M9 fusion on device=%s, exp_id=%s", device, args.exp_id)

    conditions = {c: _active_timeframes(c) for c in TS2VEC_CONDITIONS}

    def branch_loader(timeframe: str, seed: int) -> TS2VecBranch:
        ckpt = CHECKPOINTS_DIR / f"branch_{timeframe}" / f"seed_{seed}" / "best_model.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing M8 checkpoint: {ckpt}")
        branch = TS2VecBranch(config, timeframe, device)
        branch.load_checkpoint(ckpt)
        return branch

    pipe = EmbeddingPipeline(config, branch_loader)
    exp = get_experiment_dir(args.exp_id)
    branch_dir = exp / "embeddings" / "branch"
    fused_dir = exp / "embeddings" / "fused"
    branch_dir.mkdir(parents=True, exist_ok=True)
    fused_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        for split in ("train", "test"):
            windows = {
                tf: np.load(DATA_PROCESSED_DIR / f"{split}_windows_{tf}.npy")
                for tf in TIMEFRAMES
            }
            n = next(iter(windows.values())).shape[0]
            branch_emb = pipe.encode_all_branches(seed, windows)  # {tf: [N,64]}
            for tf, arr in branch_emb.items():
                np.save(branch_dir / f"embeddings_{split}_{tf}_seed{seed}.npy", arr)
            for cond, tfs in conditions.items():
                fused = pipe.fuse_condition(cond, tfs, branch_emb)  # [N,256]
                np.save(fused_dir / f"embeddings_{split}_{cond}_seed{seed}.npy", fused)
            logger.info(
                "seed=%d split=%s: %d branch [%d,64] + %d fused [%d,256] saved",
                seed, split, len(branch_emb), n, len(conditions), n,
            )

    n_branch = len(list(branch_dir.glob("*.npy")))
    n_fused = len(list(fused_dir.glob("*.npy")))
    logger.info(
        "M9 complete: %d branch files + %d fused files under %s",
        n_branch, n_fused, exp,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
