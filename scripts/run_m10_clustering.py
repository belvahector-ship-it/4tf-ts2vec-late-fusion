#!/usr/bin/env python3
"""
scripts/run_m10_clustering.py

M10 — HDBSCAN clustering (Stage 1 primary) on the real M9 fused embeddings.

Per seed: grid-search HDBSCAN on the 1TF fused train embeddings, lock the
params, and apply them to all 7 TS2Vec conditions (V-EXP-001); test labels
via approximate_predict. Writes locked params + cluster-label .npy/.parquet
under experiments/{exp_id}/clustering/.

PREREQUISITES:
- M9 fused embeddings: experiments/{exp_id}/embeddings/fused/embeddings_{split}_{cond}_seed{seed}.npy
- M6 anchor timestamps: data/processed/{split}_timestamps.npy
- aligned master (OHLCV) + feature matrix (close_return_1h)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.models.hdbscan_clustering import (  # noqa: E402
    ClusteringPipeline,
    HDBSCANClusterer,
    build_cluster_parquet,
)
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402
from src.utils.paths import (  # noqa: E402
    BASE_CONFIG_PATH,
    DATA_INTERIM_DIR,
    DATA_PROCESSED_DIR,
    RANDOM_SEEDS,
    TS2VEC_CONDITIONS,
    get_experiment_dir,
)

logger = get_logger(__name__)


def _build_ohlcv_df() -> pd.DataFrame:
    """timestamp + OHLCV_1h (aligned master) + close_return_1h (feature matrix)."""
    aligned = pd.read_parquet(DATA_INTERIM_DIR / "btc_aligned_1h.parquet")
    feats = pd.read_parquet(DATA_PROCESSED_DIR / "btc_features_all.parquet")
    ohlcv = aligned[["timestamp", "open_1h", "high_1h", "low_1h", "close_1h", "volume_1h"]]
    return ohlcv.merge(feats[["timestamp", "close_return_1h"]], on="timestamp", how="inner")


def main() -> int:
    parser = argparse.ArgumentParser(description="M10 — HDBSCAN clustering (Stage 1).")
    parser.add_argument("--config", type=Path, default=BASE_CONFIG_PATH)
    parser.add_argument("--exp-id", type=str, default="m9_real")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(RANDOM_SEEDS))
    args = parser.parse_args()

    config = load_config(args.config)
    cc = config["clustering"]
    exp = get_experiment_dir(args.exp_id)
    fused_dir = exp / "embeddings" / "fused"
    out_dir = exp / "clustering"
    out_dir.mkdir(parents=True, exist_ok=True)

    ohlcv_df = _build_ohlcv_df()
    ts = {
        "train": np.load(DATA_PROCESSED_DIR / "train_timestamps.npy"),
        "test": np.load(DATA_PROCESSED_DIR / "test_timestamps.npy"),
    }

    def load_emb(split: str, cond: str, seed: int) -> np.ndarray:
        return np.load(fused_dir / f"embeddings_{split}_{cond}_seed{seed}.npy")

    locked_all: dict[str, dict] = {}
    for seed in args.seeds:
        clusterer = HDBSCANClusterer(
            min_cluster_size_grid=cc["min_cluster_size_grid"],
            min_samples_grid=cc["min_samples_grid"],
            min_clusters=cc["min_clusters"],
            max_clusters=cc["max_clusters"],
        )
        pipe = ClusteringPipeline(clusterer, grid_search_condition=cc["grid_search_condition"])
        train_by = {c: load_emb("train", c, seed) for c in TS2VEC_CONDITIONS}
        test_by = {c: load_emb("test", c, seed) for c in TS2VEC_CONDITIONS}

        result = pipe.run_stage1(seed, train_by, test_by)
        locked_all[str(seed)] = {
            "locked_params": result["locked_params"],
            "grid_table": result["grid_table"],
        }

        for cond, lab in result["labels"].items():
            for split in ("train", "test"):
                labels = lab[split]
                np.save(out_dir / f"cluster_labels_{split}_{cond}_seed{seed}.npy", labels)
                parq = build_cluster_parquet(labels, ts[split], ohlcv_df, cond, seed)
                parq.to_parquet(out_dir / f"cluster_labels_{split}_{cond}_seed{seed}.parquet",
                                index=False)
        logger.info("[seed=%d] locked=%s written", seed, result["locked_params"])

    # single locked-params file (all seeds) — written after Stage 1 completes
    (out_dir / "hdbscan_params_locked.json").write_text(json.dumps(locked_all, indent=2))
    n_npy = len(list(out_dir.glob("cluster_labels_*.npy")))
    n_parq = len(list(out_dir.glob("cluster_labels_*.parquet")))
    logger.info("M10 complete: %d label .npy + %d .parquet under %s", n_npy, n_parq, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
