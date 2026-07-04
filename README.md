# Multi-Resolution Temporal Representation Learning for Self-Supervised Cryptocurrency Market State Discovery

> **Status:** Implementation in progress — **Milestone 2 (Data Pipeline Complete) achieved**: M1–M6 code complete (Data Acquisition, Data Validation, Temporal Alignment, Feature Engineering, Temporal Split, Window Generation). All V-LEAK-001 through V-LEAK-004 checkpoints implemented and verified. Next: M7 (TS2Vec Wrapper), blocked on TS2Vec commit hash (status: Pending).
> **Design phase:** Complete. See `docs/` for PROPOSAL, DS-01 v1.2, DS-02 v1.2, DS-03 v1.2, DS-04 v1.1, and IMP-01 v1.3.

## Project Summary

A controlled empirical study evaluating whether static multi-resolution
temporal input (1–4 timeframes: 15m, 1h, 4h, 1d) improves the quality
of self-supervised latent representations for cryptocurrency market
state discovery, compared to a single-resolution baseline.

- **Encoder:** TS2Vec (unmodified, pinned upstream commit — see below), one independently trained branch per active timeframe.
- **Fusion:** deterministic late concatenation + fixed random projection to 256 dimensions (zero learnable parameters).
- **Clustering:** HDBSCAN, parameters locked from a grid search on the 1TF condition only, reused unchanged across all other conditions.
- **Conditions:** 7 unique TS2Vec conditions (1TF, 2TF, 3TF, 4TF, BL-15m, BL-4h, BL-1d) + 2 external baselines (HMM, K-Means+PCA), each run under 5 random seeds → **45 total runs**.
- **Statistics:** pre-registered Wilcoxon signed-rank tests (one-sided) with Holm-Bonferroni correction across 3 primary comparisons (2TF/3TF/4TF vs. 1TF).

Full research rationale, hypotheses, and design decisions are in
`docs/PROPOSAL_SEMI_FINAL.md`, `docs/DS-01_v1.1.md`,
`docs/DS-02_v1.2.md`, `docs/DS-03_v1.2.md`, and `docs/DS-04_v1.1.md`.
This README covers only what is needed to install, run, and reproduce results.

## Installation

### Option A — pip / venv

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Option B — conda

```bash
conda env create -f environment.yml
conda activate market-state-discovery
```

**TS2Vec dependency:** this repository pins the upstream TS2Vec
implementation to a specific commit for reproducibility (ADR-001).

**Dependency Pinning Status:**

| Field | Value |
|---|---|
| Repository | https://github.com/zhihanyue/ts2vec |
| Branch | `main` |
| Commit Hash | `b0088e14a99706c05451316dc6db8d3da9351163` |
| Status | **Pinned** |

*(NOTE: the author's GitHub username changed from "yuezhihan", as originally cited in DS-01/ADR-001, to "zhihanyue" — this URL was confirmed live and matching the AAAI-22 paper as of 2026-07-03. The pinned commit does not match the current `main` HEAD as of the same date — source code inspection confirmed the vendored copy is textually identical to current `main`'s `ts2vec.py`, so no functional discrepancy is expected, but this is noted for full transparency.)*

`pip install -r requirements.txt` / `conda env create -f environment.yml` will install from this exact commit.

Fallback fork (if upstream becomes unavailable): to be created **after Milestone 1** per author's decision (2026-07-03) — see IMP-01 Risk R-01. Once created, its URL replaces `REPLACE_WITH_FALLBACK_FORK_URL` in `configs/base.yaml`.

## Dataset

BTC/USDT OHLCV klines from Binance, 2020-01-01 to 2023-12-31 UTC, at
four resolutions (15m, 1h, 4h, 1d). Data is **not** committed to this
repository (see `.gitignore`). To acquire it:

```bash
python scripts/run_m1_acquisition.py --config configs/base.yaml
```

*(M1-M6 code complete — see docs/CHECKPOINT_LATEST.md. Script scaffolding for later stages continues per docs/IMP-01_v1.3.md.)*

## Reproduction Steps

Once implementation reaches Milestone 4 (Experiment Complete), the full
pipeline will be runnable end-to-end via:

```bash
python scripts/run_experiment.py --run-all
```

This will execute all 45 runs (35 TS2Vec: 7 conditions × 5 seeds; 10
external baseline: HMM + KM-PCA × 5 seeds) and write results to
`experiments/`, `outputs/final/figures/`, and `outputs/final/tables/`.

A detailed `docs/reproduction_guide.md` will be added at Milestone 5.

## Expected Outputs

- `outputs/final/figures/` — 4 publication-quality figures (UMAP, Silhouette-vs-N_TF curve, market state timeline, architecture diagram), each in PNG/PDF/SVG.
- `outputs/final/tables/` — ablation table, statistical test results, economic validity summary, sensitivity analysis (CSV).
- `experiments/{exp_id}/` — per-run embeddings, cluster labels, and metrics for full auditability.

## Repository Structure

```
configs/          # base.yaml + per-condition YAML configs (9 total: 7 TS2Vec + 2 external)
src/
  utils/          # config, seed, device, logging, paths (M0)
  data/           # acquisition, validation, alignment, features, split, windowing (M1-M6)
  models/         # TS2Vec wrapper, fusion module (M7, M9)
  experiments/    # HDBSCAN clustering, external baselines, experiment runner (M10, M10.5, M13)
  evaluation/     # geometric/economic metrics, statistical tests (M11, M14)
  visualization/  # figure generation (M12)
scripts/          # thin CLI entrypoints calling into src/
tests/            # unit tests, one file per module, mirrors DS-04 validations
docs/             # design documents (PROPOSAL, DS-01–DS-04, IMP-01) + session checkpoints
```

## Development Status

This repository follows the module build order defined in
`docs/IMP-01_v1.3.md` §5 (Coding Order). Current progress is tracked in
`docs/CHECKPOINT_LATEST.md`.

**Continuing this project in a new environment (e.g. Claude Code, local
machine)?** Read `MIGRATION_TO_CLAUDE_CODE.md` first, then run
`bash setup_and_verify.sh`. See `PUSH_TO_GITHUB.md` if you need to push
this repository to GitHub first.

## License

*(To be determined by the author before public release.)*
