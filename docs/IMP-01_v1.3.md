# IMP-01 — Implementation Roadmap
## Multi-Resolution Temporal Encoding for Self-Supervised Cryptocurrency Market State Discovery

> **Document version:** 1.3
> **Status:** Draft — Pending Approval
> **Date:** 2026-07
> **Author:** Research Software Engineer (RSE)
> **Depends on:** DS-01 v1.2, DS-02 v1.2, DS-03 v1.2, DS-04 v1.1
> **Does NOT modify:** Any research decision in DS-01 through DS-04

> **Changelog v1.2 → v1.3 (structural design correction, no research decision changed):**
> - **M6 — Window Generation** rebuilt on the corrected DS-02 v1.2 Stage 5 design ("window-then-split-by-anchor" instead of "split-then-window"): **Inputs** changed from `train_features.parquet` + `test_features.parquet` (M5 output) to `btc_features_all.parquet` (M4 output, the full 35,045-row feature matrix). Windows are generated once over the full matrix, then categorized as train/test based on each window's **anchor timestamp** (its last row) against the ADR-014 boundary — not by pre-filtering the source rows before windowing. This resolves an internal contradiction present in v1.2 (M6 DoD simultaneously claimed "train windows generated exclusively from train_features.parquet" AND "expected ≤47 test windows contain training-period timestamps" — these cannot both be true if windows are sourced from an already-split file). See DS-02 v1.2 changelog and `AUDIT_LC4_ADDENDUM.md` for the full arithmetic proof.
> - **M6 Definition of Done**: `N_test_windows ≈ 8,713` → **`N_test_windows = 8,760` exactly**; "no window spans the train/test boundary (train windows generated exclusively from train_features.parquet)" → replaced with a correct statement that train windows structurally cannot contain any test-period row (since a window only looks backward from its anchor), while up to 47 leading test windows are expected and required to show training-period rows in their earliest positions (V-LEAK-003 pass criterion, not a failure).
> - **Module Dependency Graph** updated: **M6 now depends only on M4**, not on M5. M5 (Temporal Split) is no longer a prerequisite for M6 — it becomes a parallel branch off M4, producing `train_features.parquet`/`test_features.parquet` as audit and economic-validity artifacts (used later in M11/Stage 9 to join cluster labels back to OHLCV and compute returns), not as window-generation input. M5 and M6 may now be built in either order or in parallel.
> - **Coding Order** updated to reflect M5 ∥ M6 (both depend only on M4; previously listed as strictly sequential M4→M5→M6).
> - **M10.5 — External Baselines "Dependencies"** corrected from "M6" to "M5" to match its own already-correct "Inputs" line (`train_features.parquet`/`test_features.parquet`) — this was a pre-existing label/dependency-graph mismatch unrelated to the LC-4 issue itself, caught and fixed while updating the M5/M6 dependency structure in this pass.
> - No change to M0–M4, M7–M15, or any DS-01/DS-03/DS-04 research decision.

> **Changelog v1.1 → v1.2 (internal consistency fix, no research decision changed):**
> - **M4 Definition of Done** corrected: "First timestamp after processing is 2020-01-19 00:00:00 UTC (reflecting 19-row NaN drop)" → **"First timestamp after processing is 2020-01-01 19:00:00 UTC (reflecting 19-row NaN drop)"**. This mirrors the identical correction made in DS-02 v1.1 and DS-04 v1.1 (see those documents' changelogs for full arithmetic justification): dropping 19 hourly rows from a 2020-01-01 00:00 UTC start lands on 2020-01-01 19:00 UTC, not 2020-01-19 00:00 UTC. The 35,045-row and 29-column figures were already correct and are unchanged.
> - Depends-on versions updated to reflect DS-01 v1.1, DS-02 v1.1, DS-04 v1.1 (all corrected in the same 2026-07 audit pass as this document).

> **Changelog v1.0 → v1.1 (internal consistency fix, no research decision changed):**
> - Corrected condition count: **7 unique TS2Vec conditions** (4 primary cumulative: 1TF/2TF/3TF/4TF + 3 secondary baselines: BL-15m/BL-4h/BL-1d), not 8. This document previously miscounted by treating the 7 conditions as 8, contradicting PROPOSAL §3.6 and DS-03 §4/§6, which are authoritative and were already correct.
> - Corrected total run count: **45 runs total** = 35 TS2Vec runs (7 conditions × 5 seeds) + 10 external baseline runs (HMM + KM-PCA × 5 seeds), not "40 runs (8×5)". This document previously omitted HMM and KM-PCA entirely from its run accounting despite DS-03 §4 and §6 requiring them under the identical five-seed protocol (V-EXP-004).
> - Added explicit **M10.5 — External Baselines** module (HMM + KM-PCA), which was referenced only indirectly via V-EXP-004 in M14 but had no corresponding build step, dependency edge, or coding-order position anywhere in v1.0.
> - Corrected M9 fused-embeddings file count per seed: **14 files (7 conditions × 2 splits)**, not 16.
> - Updated Module Dependency Graph, Coding Order, Milestone 3/4 completion criteria, Repository Completion Checklist, and `ExperimentRunner`/`ExperimentRegistry` scope to reflect 7 TS2Vec conditions + 2 external baselines = 9 unique methods, 45 total runs.
> - Branch checkpoint count (**20 = 4 timeframes × 5 seeds**) was already correct in v1.0 and is unchanged — branch training count is independent of condition count, since secondary baselines reuse the same 4 branch encoders, only recombined differently at fusion.

---

## 1. Objectives

This document converts the approved design specifications (DS-01 through DS-04) into a concrete, executable implementation sequence for a solo researcher.

It answers one question: **in what order, and at what level of detail, should each software component be built?**

IMP-01 is not a specification. Every scientific decision is already locked in DS-01 through DS-04. This document only concerns *how to build what has been decided*. If any implementation difficulty arises, the solution is engineering — not changing the protocol.

This roadmap is designed to:

- Eliminate ambiguity about what to build next
- Ensure each module is testable before the next depends on it
- Surface integration risks early, before they invalidate experiment runs
- Produce a repository that is auditable by a reviewer who has never seen this codebase

---

## 2. Development Philosophy

**Build from infrastructure upward.** Configuration, logging, and path management come before any scientific code. A broken config system breaks everything downstream silently.

**One module at a time.** No module's code is written until all modules it depends on are complete and passing their DS-04 validations. Parallel implementation of dependent modules is prohibited.

**No module may advance past its Definition of Done.** Each module has a specific DoD (see Section 4). A module is not complete until every item in its DoD is satisfied.

**DS-04 validations gate progress.** Every module maps to one or more validation items in DS-04. No milestone is complete until all mapped validations pass.

**No research decisions change during implementation.** If an implementation reveals a genuine impossibility (not just inconvenience), escalate to a documented deviation request — do not silently modify the protocol.

**Test as you build.** Tests for each module live in `tests/` and are written alongside the module, not after the experiment runs.

**The experiment is not run until M0–M12 are complete.** Running experiments on partially implemented infrastructure produces untrustworthy results. Resist the temptation to run early.

---

## 3. Module Dependency Graph

```
M0  Project Bootstrap
 │
 ├─→ M1  Data Acquisition
 │    │
 │    └─→ M2  Data Validation
 │          │
 │          └─→ M3  Temporal Alignment
 │                │
 │                └─→ M4  Feature Engineering
 │                      │
 │                      ├─→ M5  Temporal Split (audit/economic-validity artifacts;
 │                      │        NOT a prerequisite for M6 — see below)
 │                      │
 │                      └─→ M6  Window Generation (windows the FULL feature matrix,
 │                            │    then categorizes by anchor timestamp — DS-02 v1.2)
 │                            │
 │                            ├──────────────────────────────┐
 │                            │                               │
 ├─→ M7  TS2Vec Wrapper  ←── (depends on M0 only)             │
 │                            │                               │
 │   [M6 + M7 complete]       │                               │
 │         │                  │                               │
 │         └─→ M8  Branch Training                             │
 │                │                                             │
 │                └─→ M9  Fusion                                │
 │                      │                                       │
 │                      │           M10.5  External Baselines ──┤
 │                      │           (HMM + KM-PCA, depends on M5│
 │                      │            train/test features only — │
 │                      │            independent of M6/M7/M8/M9)│
 │                      │                                       │
 │                      └─→ M10  HDBSCAN Clustering              │
 │                             │       (TS2Vec conditions only)  │
 │                             │                                 │
 │                             └─→ M11  Evaluation  ←─────────────┘
 │                                   │   (merges TS2Vec + external
 │                                   │    baseline results)
 │                                   ├─→ M12  Visualization
 │                                   │
 │                                   └─→ M13  Experiment Runner
 │                                         │
 │                                         └─→ M14  Statistical Analysis
 │                                               │
 │                                               └─→ M15  Paper Artifact Generator
```

**Parallel tracks permitted:**

- M7 (TS2Vec Wrapper) may be built in parallel with M1–M6, since it depends only on M0.
- **M5 (Temporal Split) and M6 (Window Generation) may now be built in either order or in parallel** — both depend only on M4, not on each other. *(Corrected in v1.3 — M6 previously depended on M5's output; DS-02 v1.2 corrected Stage 5 to read directly from M4's full feature matrix. See document changelog and DS-02 v1.2 changelog.)*
- M10.5 (External Baselines) depends on M5 (`train_features.parquet`/`test_features.parquet`, 1h resolution) and may be built in parallel with M6, M7, M8, M9, and M10, since HMM and KM-PCA do not use TS2Vec embeddings, windowed tensors, or the fusion module at all. *(Corrected in v1.3 — the M10.5 module spec's "Dependencies" line previously said "M6," which mismatched its own "Inputs" line; both now correctly say M5.)*
- M12 (Visualization) and M14 (Statistical Analysis) may be built in parallel after M11 is complete.

---

## 4. Module Specifications

---

### M0 — Project Bootstrap

**Purpose:** Establish the repository skeleton, configuration system, logging infrastructure, path management, seed control utilities, and device detection. Every subsequent module depends on these foundations.

**Inputs:** None (empty directory)

**Outputs:**
- Complete directory structure matching the repository layout
- `configs/base.yaml` with all shared hyperparameters
- `configs/experiment_*.yaml` stubs for each of the 7 TS2Vec conditions (1TF, 2TF, 3TF, 4TF, BL-15m, BL-4h, BL-1d), plus `configs/baseline_hmm.yaml` and `configs/baseline_kmpca.yaml` for the 2 external baselines
- `src/utils/config.py` — YAML loader with schema validation
- `src/utils/logging_utils.py` — standardized logging setup
- `src/utils/seed.py` — seed-setting utility for Python, NumPy, PyTorch, CUDA
- `src/utils/device.py` — device detection (ADR-007)
- `src/utils/paths.py` — centralized path constants
- `requirements.txt` and `environment.yml`
- `README.md` skeleton
- `.gitignore` excluding data, checkpoints, and outputs

**Dependencies:** None

**Main Classes/Functions:**
- `load_config(path: Path) -> dict` — loads and validates YAML
- `set_all_seeds(seed: int) -> None` — sets Python/NumPy/PyTorch/CUDA seeds
- `get_device() -> torch.device` — returns best available device
- `get_project_root() -> Path` — returns absolute repository root

**Definition of Done:**
- All directories exist
- `base.yaml` contains every controlled variable from DS-03 Section 3 — no magic numbers exist anywhere in source
- Config loader raises `ValueError` with an informative message for any missing required field
- `set_all_seeds` verifiably affects NumPy and PyTorch random state (confirmed by a test that checks reproducibility)
- `get_device()` returns `cuda` when CUDA is available and `cpu` otherwise, without error on either

**DS-04 Tests That Must Pass:**
- V-INV-001 (same config fields across all conditions) — precondition established here
- V-INV-007 (deterministic execution) — seed utility verified here

**Estimated Complexity:** Low

**Implementation Order:** 1

---

### M1 — Data Acquisition

**Purpose:** Download BTC/USDT OHLCV klines from Binance for all four timeframes (15m, 1h, 4h, 1d) covering 2020-01-01 to 2023-12-31 UTC. Produce raw Parquet files with full download provenance metadata.

**Inputs:**
- `configs/base.yaml` — symbol, exchange, timeframes, date range

**Outputs:**
- `data/raw/btc_15m_raw.parquet`
- `data/raw/btc_1h_raw.parquet`
- `data/raw/btc_4h_raw.parquet`
- `data/raw/btc_1d_raw.parquet`
- `data/raw/manifest.json` — row counts, date ranges, SHA-256 checksums, ccxt version, download timestamp

**Dependencies:** M0

**Main Classes/Functions:**
- `BinanceDownloader` class
  - `download(timeframe, start, end) -> pd.DataFrame`
  - `save_parquet(df, path) -> None`
  - `compute_checksum(path) -> str`
- `build_manifest(results: dict) -> dict`

**Definition of Done:**
- All four Parquet files exist and are readable
- Row counts are within 5% of expected values per DS-04 V-DATA-001
- `manifest.json` is present and contains all required fields per DS-02 Stage 0
- Download is re-runnable: if files already exist and checksums match, skip re-download
- Download failures raise informative exceptions (network, rate limit, API error)
- No `print()` calls — all output via `logging`

**DS-04 Tests That Must Pass:**
- V-DATA-001 (row counts, date range coverage)

**Estimated Complexity:** Low

**Implementation Order:** 2

---

### M2 — Data Validation

**Purpose:** Verify all data integrity conditions defined in DS-02 Stage 1 and DS-04 Section 3.1. Abort execution with an informative error report if any check fails.

**Inputs:**
- `data/raw/btc_{tf}_raw.parquet` (4 files)

**Outputs:**
- Validation report appended to `data/raw/manifest.json`
- Log entries for all checks performed
- Raised exception with detailed report if any check fails

**Dependencies:** M1

**Main Classes/Functions:**
- `DataValidator` class
  - `validate_timeframe(df: pd.DataFrame, timeframe: str) -> ValidationReport`
  - `check_monotonicity(df) -> bool`
  - `check_duplicates(df) -> bool`
  - `check_ohlc_integrity(df) -> bool`
  - `check_nan_inf(df) -> bool`
  - `check_timezone(df) -> bool`
  - `check_date_coverage(df, start, end) -> bool`
  - `check_gap_ratio(df, timeframe) -> float`
- `ValidationReport` dataclass

**Definition of Done:**
- All checks from DS-02 Stage 1 are implemented
- Each failed check produces an error message identifying which rows violate the condition
- The module aborts (raises `DataValidationError`) if any check fails beyond the documented tolerance
- A clean dataset passes all checks without warnings
- All checks are covered by unit tests with both passing and failing synthetic inputs

**DS-04 Tests That Must Pass:**
- V-DATA-001 (row count, date range)
- V-DATA-002 (OHLCV integrity constraints)

**Estimated Complexity:** Low

**Implementation Order:** 3

---

### M3 — Temporal Alignment

**Purpose:** Align all four timeframes to the 1h anchor, producing a single master DataFrame as specified in DS-02 Stage 2. Implements 15m→1h aggregation, 4h→1h forward-fill, and 1d→1h forward-fill.

**Inputs:**
- `data/raw/btc_{tf}_raw.parquet` (4 validated files)

**Outputs:**
- `data/interim/btc_aligned_1h.parquet` — 35,064 rows × 21 columns per DS-02 Stage 2 schema

**Dependencies:** M2

**Main Classes/Functions:**
- `TemporalAligner` class
  - `aggregate_15m_to_1h(df_15m: pd.DataFrame) -> pd.DataFrame`
  - `forward_fill_to_1h(df: pd.DataFrame, source_tf: str) -> pd.DataFrame`
  - `build_master(dfs: dict) -> pd.DataFrame`
- Leakage verification utility: `verify_no_lookahead(master_df, source_df, tf) -> bool`

**Definition of Done:**
- Output has exactly 21 columns and ~35,064 rows (±5 for edge handling)
- Timestamp column is strictly monotonic and UTC-localized
- Forward-fill produces no look-ahead: at timestamp T, 4h/1d values reflect only candles with open_time ≤ T (verified programmatically by `verify_no_lookahead`)
- The first partial period at dataset start is handled explicitly
- Unit test covers at least one timestamp from each year (2020–2023) to confirm forward-fill correctness

**DS-04 Tests That Must Pass:**
- V-DATA-003 (aligned master schema and row count)
- V-LEAK-001 (forward-fill produces no look-ahead)

**Estimated Complexity:** Medium

**Implementation Order:** 4

---

### M4 — Feature Engineering

**Purpose:** Compute the 7 OHLCV-derived features per timeframe (ADR-015, DS-02 Stage 3). Drop NaN rows introduced by `open_return` and `volume_zscore` rolling window. Produce the feature matrix Parquet.

**Inputs:**
- `data/interim/btc_aligned_1h.parquet`

**Outputs:**
- `data/processed/btc_features_all.parquet` — 35,045 rows × 29 columns

**Dependencies:** M3

**Main Classes/Functions:**
- `FeatureEngineer` class
  - `compute_features_for_timeframe(df: pd.DataFrame, tf_suffix: str) -> pd.DataFrame`
  - `compute_open_return(df, suffix) -> pd.Series`
  - `compute_volume_zscore(df, suffix, window=20) -> pd.Series`
  - `compute_hl_range(df, suffix) -> pd.Series`
  - `compute_body_ratio(df, suffix) -> pd.Series`
  - `drop_nan_rows(df) -> pd.DataFrame`

**Definition of Done:**
- All 7 features are computed identically for all 4 timeframe suffixes
- No NaN or Inf values remain after row dropping
- First timestamp after processing is 2020-01-01 19:00:00 UTC (reflecting 19-row NaN drop) *(corrected in v1.2 — previously misstated as "2020-01-19"; see document changelog)*
- `body_ratio` uses `+ 1e-8` epsilon in denominator — no division-by-zero on doji candles
- `volume_zscore` rolling window uses window=20 over aligned 1h rows, with documented behavior per DS-02 Stage 3 rolling window clarification
- Unit tests verify each feature formula against manually computed expected values

**DS-04 Tests That Must Pass:**
- V-DATA-004 (feature matrix schema and row count)

**Estimated Complexity:** Low

**Implementation Order:** 5

---

### M5 — Temporal Split

**Purpose:** Enforce the walk-forward train/test split at the exact boundary defined in ADR-014 and DS-02 Stage 4. No shuffling, no stratification, no overlap.

**Inputs:**
- `data/processed/btc_features_all.parquet`

**Outputs:**
- `data/processed/train_features.parquet` — rows with timestamp ≤ 2022-12-31 23:00 UTC
- `data/processed/test_features.parquet` — rows with timestamp ≥ 2023-01-01 00:00 UTC

**Dependencies:** M4

**Main Classes/Functions:**
- `TemporalSplitter` class
  - `split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]`
  - `verify_split(train: pd.DataFrame, test: pd.DataFrame) -> None`

**Definition of Done:**
- Train max timestamp = 2022-12-31 23:00:00 UTC exactly (verified by assertion)
- Test min timestamp = 2023-01-01 00:00:00 UTC exactly (verified by assertion)
- Zero overlap: union of train and test timestamps contains no duplicates
- Row counts match expected values (train ~26,269, test ~8,760)
- `verify_split` raises `SplitIntegrityError` if any boundary condition fails

**DS-04 Tests That Must Pass:**
- V-DATA-005 (split boundary and no overlap)
- V-LEAK-002 (no future price information crosses the split boundary — documented behavior of volume_zscore rolling window)

**Estimated Complexity:** Low

**Implementation Order:** 6

---

### M6 — Window Generation

**Purpose:** Convert the full feature matrix into 3D sliding-window tensors with per-window z-score normalization (ADR-006, ADR-016, DS-02 v1.2 Stage 5). Categorize windows as train/test by anchor timestamp AFTER windowing (not before). Produce `.npy` files ready for DataLoader consumption.

**Inputs:**
- `data/processed/btc_features_all.parquet` (M4 output — the full 35,045-row feature matrix, NOT the M5 split files) *(corrected in v1.3 — see document changelog)*

**Outputs:**
- 8 windowed `.npy` files (`train_windows_{tf}.npy`, `test_windows_{tf}.npy`) — shape `[N_windows, 48, 7]`, dtype float32
- `train_timestamps.npy` and `test_timestamps.npy` — anchor timestamps per window

**Dependencies:** M4 *(corrected in v1.3 — previously M5; M6 no longer depends on M5's split output, see document changelog)*

**Main Classes/Functions:**
- `WindowGenerator` class
  - `extract_windows(features_df: pd.DataFrame, tf_suffix: str) -> WindowSet` — slides W=48/stride=1 over the FULL feature matrix, records both anchor (last row) and earliest (first row) timestamps per window
  - `normalize_window(window: np.ndarray) -> np.ndarray` / `normalize_all_windows(windows: np.ndarray) -> np.ndarray` — per-window z-score, vectorized and single-window variants
  - `generate(features_df: pd.DataFrame, tf_suffix: str) -> WindowSet` — extract + normalize in one call
- Categorization: windows with anchor timestamp ≤ 2022-12-31 23:00 UTC → train; anchor ≥ 2023-01-01 00:00 UTC → test (same ADR-014 boundary as M5, applied to the anchor, not the whole window)
- `WindowDataset(torch.utils.data.Dataset)` — wraps `.npy` for DataLoader
  - `__len__`, `__getitem__`

**Definition of Done:**
- Output shapes are `[N_windows, 48, 7]` for all 8 files, dtype float32
- N_train_windows ≈ 26,222; **N_test_windows = 8,760 exactly** *(corrected in v1.3 — previously "≈8,713," which was only consistent with the incorrect split-then-window design; see document changelog)*
- Per-window z-score uses only values within each window (no external statistics) — verified identically regardless of whether a window's rows span the train/test boundary
- **Train windows structurally cannot contain any test-period row**, since a window only looks backward from its anchor (verified: max anchor timestamp among train windows ≤ 2022-12-31 23:00 UTC, and therefore every row in every train window is also ≤ that boundary)
- **Up to 47 leading test windows are expected to have an earliest-row timestamp before 2023-01-01 00:00 UTC** — this is the correct, required behavior (LC-4), not a failure; the count is computed and logged, and must be ≤ 47 *(corrected in v1.3 — previously incorrectly stated "no window spans the train/test boundary," which contradicted the very next bullet about the ≤47 expected overlap; see document changelog)*
- `WindowDataset.__getitem__` returns a `torch.Tensor` of shape `[48, 7]`
- Unit tests verify normalization uses only within-window statistics, AND include at least one test that deliberately injects a boundary-crossing bug (e.g. a train window whose anchor exceeds TRAIN_END) and confirms the check detects it — mirroring the M3 `verify_no_lookahead` bug-injection test pattern

**DS-04 Tests That Must Pass:**
- V-LEAK-003 (train windows stay within train period; documented ≤47 test-window overlap matches expectation)
- V-LEAK-004 (per-window z-score uses only within-window statistics)

**Estimated Complexity:** Medium

**Implementation Order:** 7 (may be built in parallel with M5, or after it — both depend only on M4)

---

### M7 — TS2Vec Wrapper

**Purpose:** Provide a clean, reproducible interface to the pinned TS2Vec installation. Wraps TS2Vec training, inference, and max-pooling. Never modifies TS2Vec source. Implements ADR-001 and ADR-002.

**Inputs:**
- Pinned TS2Vec installation (commit hash in `requirements.txt`)
- Config: `depth`, `hidden_dim`, `output_dim`, `lr`, `batch_size`, `max_epochs`, `patience`

**Outputs:**
- `src/models/ts2vec_wrapper.py` — wrapper module
- `checkpoints/branch_{tf}/best_model.pt` — checkpoint per branch per seed (ADR-010 format)
- `checkpoints/branch_{tf}/latest_model.pt`

**Dependencies:** M0 (TS2Vec pinned installation from `requirements.txt`)

**Main Classes/Functions:**
- `TS2VecBranch` class
  - `__init__(config: dict, timeframe: str, device: torch.device)`
  - `train(windows: np.ndarray, seed: int) -> TrainingHistory`
  - `encode(windows: np.ndarray) -> np.ndarray` — returns `[N, 64]` via max-pool over time axis
  - `save_checkpoint(path: Path, extra_metadata: dict) -> None` — full reproducibility bundle per ADR-010
  - `load_checkpoint(path: Path) -> None`
- `TrainingHistory` dataclass: `train_loss_history`, `epoch_times`, `best_epoch`, `best_loss`

**Definition of Done:**
- `TS2VecBranch.encode` returns shape `[N, 64]` for any valid input batch
- Checkpoint format matches ADR-010 exactly: contains all required keys including `ts2vec_commit`, `projection_seed`, `config_snapshot`
- `load_checkpoint` is tested on CPU for checkpoints saved on GPU (map_location='cpu')
- The TS2Vec source is never modified: wrapper imports from the installed package
- No TS2Vec source files appear in `src/`
- `encode` is deterministic given same input and loaded weights

**DS-04 Tests That Must Pass:**
- V-MODEL-001 (branch encoder output shape `[B, 64]`)
- V-INV-004 (independent optimizers, no shared loss)

**Estimated Complexity:** Medium

**Implementation Order:** 8 (can be built in parallel with M1–M6)

---

### M8 — Branch Training

**Purpose:** Execute the 4 independent branch training runs (one per timeframe) per seed. Manage checkpoint reuse across conditions. Implement the five-seed protocol (ADR-019).

**Inputs:**
- `data/processed/train_windows_{tf}.npy` (4 files from M6)
- `TS2VecBranch` wrapper (M7)
- `configs/base.yaml` — training hyperparameters

**Outputs:**
- `checkpoints/branch_{tf}/seed_{seed}/best_model.pt` (4 timeframes × 5 seeds = 20 checkpoints)
- `checkpoints/branch_{tf}/seed_{seed}/latest_model.pt` (20 checkpoints)
- `logs/training_branch_{tf}_seed_{seed}.log` per run

**Dependencies:** M6, M7

**Main Classes/Functions:**
- `BranchTrainer` class
  - `train_all_branches(seeds: list[int]) -> dict`
  - `train_single(timeframe: str, seed: int) -> TrainingHistory`
  - `load_or_train(timeframe: str, seed: int) -> Path` — returns checkpoint path; skips if valid checkpoint exists
- `TrainingOrchestrator` — coordinates 4 × 5 = 20 training runs, handles resume

**Definition of Done:**
- Exactly 4 training runs per seed (one per timeframe), not one per condition
- Checkpoint reuse is verified: the 1h branch checkpoint for seed 42 is identical across all conditions that include 1h
- Resume from checkpoint works: interrupted training can be continued from `latest_model.pt`
- Training metrics (loss history, epoch times) are logged and saved in checkpoints
- Each training run produces a log file with epoch-level detail
- The `load_or_train` method skips training if a valid checkpoint already exists (idempotency)

**DS-04 Tests That Must Pass:**
- V-INV-004 (each branch has independent optimizer, no shared loss)
- V-EXP-002 (five seeds applied, embeddings from different seeds are distinct)
- V-MODEL-005 (different branches produce distinct representations)

**Estimated Complexity:** Medium

**Implementation Order:** 9

---

### M9 — Fusion

**Purpose:** Implement deterministic late fusion via concatenation followed by fixed random projection (ADR-003, ADR-013). Produce 256-dim fused embeddings for all 7 TS2Vec conditions (1TF, 2TF, 3TF, 4TF, BL-15m, BL-4h, BL-1d). External baselines (HMM, KM-PCA) do not use this module — see M10.5.

**Inputs:**
- `checkpoints/branch_{tf}/seed_{seed}/best_model.pt` (loaded via M7)
- `data/processed/train_windows_{tf}.npy` and `test_windows_{tf}.npy` (M6)

**Outputs:**
- `experiments/{exp_id}/embeddings/branch/embeddings_{split}_{tf}.npy` — shape `[N, 64]` (8 files per seed)
- `experiments/{exp_id}/embeddings/fused/embeddings_{split}_{condition}.npy` — shape `[N, 256]` (14 files per seed: 7 conditions × 2 splits)

**Dependencies:** M8

**Main Classes/Functions:**
- `FusionModule` class
  - `__init__(condition: str, projection_seed: int = 42)`
  - `build_projection_matrix(input_dim: int) -> torch.Tensor` — deterministic, `requires_grad=False`
  - `fuse(branch_embeddings: dict[str, np.ndarray]) -> np.ndarray` — returns `[N, 256]`
- `EmbeddingPipeline` class
  - `encode_all_branches(seed: int) -> dict` — runs inference for all 4 branches
  - `fuse_condition(condition: str, branch_embeddings: dict) -> np.ndarray`

**Definition of Done:**
- Output shape is `[N, 256]` for all 7 TS2Vec conditions (V-INV-002)
- `FusionModule` trainable parameter count = 0 (verified by `sum(p.numel() for p in module.parameters() if p.requires_grad)`)
- Projection matrix is identical given the same `PROJECTION_SEED` and condition (V-MODEL-004)
- Projection matrix is saved in every checkpoint bundle (ADR-010)
- Concatenation order is fixed: 15m, 1h, 4h, 1d — enforced by ordered list, not set or dict
- 4TF condition (D_in = 256) still passes through the random projection (not identity)

**DS-04 Tests That Must Pass:**
- V-MODEL-002 (fusion output shape `[B, 256]` for all conditions)
- V-MODEL-003 (fusion trainable parameters = 0; projection unchanged after gradient step)
- V-MODEL-004 (deterministic projection given same seed)
- V-INV-002 (all conditions produce 256-dim embeddings)
- V-INV-003 (zero learnable fusion parameters)

**Estimated Complexity:** Medium

**Implementation Order:** 10

---

### M10 — HDBSCAN Clustering

**Purpose:** Implement the two-stage HDBSCAN protocol (ADR-004, DS-02 Stage 8, DS-03 Section 5). Stage 1 locks parameters from 1TF grid search; Stage 2 is optional per-condition sensitivity analysis.

**Inputs:**
- `experiments/{exp_id}/embeddings/fused/embeddings_train_{condition}.npy` (from M9)
- `experiments/{exp_id}/embeddings/fused/embeddings_test_{condition}.npy` (from M9)
- `data/processed/train_timestamps.npy` and `test_timestamps.npy` (from M6)
- `data/interim/btc_aligned_1h.parquet` (for joining OHLCV to cluster labels)

**Outputs:**
- `experiments/{exp_id}/clustering/hdbscan_params_locked.json`
- `experiments/{exp_id}/clustering/cluster_labels_{split}_{condition}.npy`
- `experiments/{exp_id}/clustering/cluster_labels_{split}_{condition}.parquet` — full schema per DS-02 Stage 8

**Dependencies:** M9

**Main Classes/Functions:**
- `HDBSCANClusterer` class
  - `grid_search(embeddings: np.ndarray, grid: dict) -> dict` — returns best params
  - `fit(embeddings: np.ndarray, params: dict) -> np.ndarray` — train labels
  - `predict(embeddings: np.ndarray) -> np.ndarray` — test labels via `approximate_predict`
- `ClusteringPipeline` class
  - `run_stage1(conditions: list[str], embeddings: dict) -> dict`
  - `run_stage2(conditions: list[str], embeddings: dict) -> dict` — optional
- `build_cluster_parquet(labels, timestamps, ohlcv_df, condition, seed) -> pd.DataFrame`

**Definition of Done:**
- Stage 1 grid search runs on 1TF only, and locked parameters are applied unchanged to 2TF, 3TF, 4TF
- `hdbscan_params_locked.json` is written before any multi-TF condition is clustered
- Noise points (label = -1) are preserved in all artifacts
- Test labels are assigned via `approximate_predict`, not by re-fitting
- Parquet output matches schema defined in DS-02 Stage 8
- If no grid configuration produces 2 ≤ k ≤ 8 clusters, the fallback procedure (ADR-004) is triggered and logged as WARNING

**DS-04 Tests That Must Pass:**
- V-EXP-001 (identical HDBSCAN params across all primary conditions)
- V-EXP-005 (noise-excluded embeddings in all metric computations — precondition established here)

**Estimated Complexity:** Medium

**Implementation Order:** 11

---

### M10.5 — External Baselines (HMM + KM-PCA)

**Purpose:** Implement and run the two external comparison methods — Hidden Markov Model (HMM) and K-Means + PCA (KM-PCA) — under the identical five-seed protocol required by DS-03 §4 and §6, so that they can be statistically compared against the TS2Vec conditions on equal footing. This module was missing from IMP-01 v1.0; it is added here to close that gap.

**Inputs:**
- `data/processed/train_features.parquet` and `test_features.parquet` (from M4/M5) — 7 features, 1h resolution only, per DS-03 §4 ("External Baselines" table)

**Outputs:**
- `experiments/{exp_id}/external_baselines/hmm/labels_{split}_seed{seed}.npy`
- `experiments/{exp_id}/external_baselines/hmm/model_seed{seed}.pkl` (with `n_components` selected via BIC, and the BIC table saved alongside for auditability)
- `experiments/{exp_id}/external_baselines/kmpca/labels_{split}_seed{seed}.npy`
- `experiments/{exp_id}/external_baselines/kmpca/model_seed{seed}.pkl` (with `k` selected via Silhouette Score, and the grid-search table saved alongside)

**Dependencies:** M5 (feature split data only — this module does NOT depend on M6, M7, M8, or M9; it never touches TS2Vec, windowed tensors, branch encoders, or the fusion module) *(corrected in v1.3 — previously said "M6," which mismatched the Inputs line above; see document changelog)*

**Main Classes/Functions:**
- `HMMBaseline` class
  - `__init__(n_components_grid: list[int] = [2, 3, 4])`
  - `fit_select(features: np.ndarray, seed: int) -> dict` — fits one HMM per `n_components`, selects by lowest BIC
  - `predict(features: np.ndarray) -> np.ndarray` — returns state labels
- `KMeansPCABaseline` class
  - `__init__(k_grid: list[int] = [2, 3, 4, 5, 6], pca_components: int = 10)`
  - `fit_select(features: np.ndarray, seed: int) -> dict` — fits PCA(10) then K-Means per `k`, selects by highest Silhouette Score
  - `predict(features: np.ndarray) -> np.ndarray` — returns cluster labels
- `ExternalBaselineRunner` — orchestrates both methods across all 5 seeds, writes outputs in the same directory convention consumed by M11

**Definition of Done:**
- Both methods run on the same 1h-resolution, 7-feature input used by the 1TF/BL-1h condition (no TS2Vec embedding involved)
- Both methods are run once per seed for all 5 seeds ({42, 123, 456, 789, 1024}) — 5 HMM runs + 5 KM-PCA runs = 10 runs total
- HMM model selection uses BIC across `n_components ∈ {2,3,4}`; KM-PCA model selection uses Silhouette Score across `k ∈ {2,3,4,5,6}` with `PCA(n_components=10)`
- Output label arrays are directly consumable by `GeometricEvaluator` and `EconomicEvaluator` in M11 without modification
- No shared code path with `TS2VecBranch`, `FusionModule`, or `HDBSCANClusterer` — these baselines are fully independent implementations

**DS-04 Tests That Must Pass:**
- V-EXP-004 (external baselines HMM and KM-PCA under five-seed protocol)

**Estimated Complexity:** Low–Medium

**Implementation Order:** 8.5 (may be built any time after M6; independent of M7–M10, does not block or get blocked by the TS2Vec branch/fusion/clustering chain)

---

### M11 — Evaluation

**Purpose:** Compute all geometric metrics (Silhouette, DBI, CH Index), economic metrics (return distributions per cluster), and the Wilcoxon signed-rank statistical test, for both the 7 TS2Vec conditions and the 2 external baselines. Produce all CSV artifacts defined in DS-02 Stage 9.

**Inputs:**
- `experiments/{exp_id}/clustering/cluster_labels_train_{condition}.npy` (from M10, 7 TS2Vec conditions)
- `experiments/{exp_id}/embeddings/fused/embeddings_train_{condition}.npy` (from M9, 7 TS2Vec conditions)
- `experiments/{exp_id}/clustering/cluster_labels_test_{condition}.parquet` (from M10, 7 TS2Vec conditions)
- `experiments/{exp_id}/external_baselines/{hmm,kmpca}/labels_{split}_seed{seed}.npy` (from M10.5, 2 external baselines)

**Outputs:**
- `experiments/{exp_id}/evaluation/metrics_per_run.csv`
- `experiments/{exp_id}/evaluation/metrics_aggregated.csv`
- `experiments/{exp_id}/evaluation/wilcoxon_results.csv`
- `experiments/{exp_id}/evaluation/economic_validity.csv`
- `experiments/{exp_id}/evaluation/kruskal_wallis_results.csv`

**Dependencies:** M10, M10.5

**Main Classes/Functions:**
- `GeometricEvaluator` class
  - `compute_silhouette(embeddings, labels) -> float`
  - `compute_dbi(embeddings, labels) -> float`
  - `compute_ch(embeddings, labels) -> float`
  - `exclude_noise(embeddings, labels) -> tuple` — removes label=-1 rows
- `EconomicEvaluator` class
  - `compute_return_stats(parquet_df) -> pd.DataFrame`
  - `run_kruskal_wallis(parquet_df) -> dict`
- `StatisticalTester` class
  - `run_wilcoxon_tests(metrics_df: pd.DataFrame) -> pd.DataFrame`
  - `apply_holm_bonferroni(p_values: list[float]) -> list[float]`
- `MetricsAggregator` — computes mean ± std across seeds; writes all CSV outputs

**Definition of Done:**
- All metrics match schemas defined in DS-02 Stage 9
- Noise points (label=-1) are explicitly excluded before all geometric metric computations
- Wilcoxon tests use paired seed results (seed index i compared to seed index i)
- Three primary comparisons are tested: 2TF vs 1TF, 3TF vs 1TF, 4TF vs 1TF (Holm-Bonferroni corrected)
- `metrics_per_run.csv` contains all 45 runs: 35 TS2Vec runs (7 conditions × 5 seeds) + 10 external baseline runs (HMM + KM-PCA × 5 seeds)
- HMM and KM-PCA results are evaluated with the same geometric and economic metrics as TS2Vec conditions, reported in the same schema but a separate table section (external baselines are descriptive comparators, not part of the pre-registered Wilcoxon hypothesis tests)
- Holm-Bonferroni correction is applied only to the three pre-registered primary comparisons and documented in output CSV
- Effect size (rank-biserial correlation r) is reported alongside each primary comparison p-value

**DS-04 Tests That Must Pass:**
- V-EXP-003 (paired Wilcoxon with Holm-Bonferroni correction)
- V-EXP-004 (external baselines HMM and KM-PCA under five-seed protocol — verified here that results are present and correctly evaluated)
- V-EXP-005 (noise-excluded metric computation)

**Estimated Complexity:** Medium

**Implementation Order:** 12

---

### M12 — Visualization

**Purpose:** Generate all four publication-quality figures defined in the blueprint (Section 17) in PNG, PDF, and SVG formats (ADR-012).

**Inputs:**
- `experiments/{exp_id}/embeddings/fused/embeddings_train_{condition}.npy` (for UMAP)
- `experiments/{exp_id}/clustering/cluster_labels_train_{condition}.npy` (for UMAP + timeline)
- `experiments/{exp_id}/evaluation/metrics_aggregated.csv` (for Silhouette vs N_TF)
- `experiments/{exp_id}/clustering/cluster_labels_test_{condition}.parquet` (for timeline)

**Outputs (per figure):**
- `outputs/final/figures/{figure_name}/{figure_name}.png` — 300 DPI
- `outputs/final/figures/{figure_name}/{figure_name}.pdf` — vector
- `outputs/final/figures/{figure_name}/{figure_name}.svg` — editable

**Dependencies:** M11

**Main Classes/Functions:**
- `UMAPFigure` — Figure 1: UMAP 3-panel (best condition, 1TF, 4TF)
- `SilhouetteCurveFigure` — Figure 2: Silhouette vs number of TFs, with error bars
- `MarketStateTimelineFigure` — Figure 3: BTC close price + cluster color bands (test 2023)
- `ArchitectureDiagramFigure` — Figure 4: model architecture
- `FigureExporter` — saves all three formats in a single rendering pass; sets `rcParams['pdf.fonttype'] = 42`

**Definition of Done:**
- All four figures are generated without error
- All three formats (PNG at 300 DPI, PDF, SVG) are produced in a single rendering pass (no re-computation per format)
- PDF font embedding verified (`pdf.fonttype = 42`)
- Figures use only standard fonts available in LaTeX environments
- UMAP computation uses `umap-learn` with a fixed seed for reproducibility

**DS-04 Tests That Must Pass:** None specific (visual outputs); review against blueprint Figure descriptions.

**Estimated Complexity:** Medium

**Implementation Order:** 13 (parallel with M14 after M11 is complete)

---

### M13 — Experiment Runner

**Purpose:** Orchestrate the complete 45-run experiment (35 TS2Vec runs: 7 conditions × 5 seeds, plus 10 external baseline runs: HMM + KM-PCA × 5 seeds) as a single, resumable, logged pipeline. Manages experiment IDs and prevents overwriting prior results.

**Inputs:**
- All M6, M8, M9, M10, M10.5, M11 modules
- `configs/experiment_{condition}.yaml` files (7 TS2Vec conditions)
- `configs/baseline_hmm.yaml`, `configs/baseline_kmpca.yaml` (2 external baselines)

**Outputs:**
- `experiments/{exp_id}/` — complete experiment directory per run
- `experiments/registry.json` — maps experiment IDs to conditions/methods, seeds, timestamps, completion status

**Dependencies:** M6, M8, M9, M10, M10.5, M11

**Main Classes/Functions:**
- `ExperimentRunner` class
  - `run(condition: str, seed: int) -> ExperimentResult` — for the 7 TS2Vec conditions (runs M9 → M10 → M11 for that condition/seed)
  - `run_external_baseline(method: str, seed: int) -> ExperimentResult` — for HMM/KM-PCA (runs M10.5 → M11 for that method/seed)
  - `run_all(conditions: list, seeds: list, include_external: bool = True) -> None` — runs all 7 conditions × 5 seeds, and, if `include_external`, both external baselines × 5 seeds
  - `resume(exp_id: str) -> None` — resumes incomplete run
- `ExperimentRegistry` — tracks which runs are complete; prevents re-running
- `ExperimentID` utility — generates unique, human-readable experiment IDs (e.g., `20260601_1TF_seed42`, `20260601_HMM_seed42`)

**Definition of Done:**
- A single `python scripts/run_experiment.py --config configs/experiment_1tf.yaml --seed 42` command executes the full TS2Vec pipeline for that condition and seed
- A single `python scripts/run_experiment.py --config configs/baseline_hmm.yaml --seed 42` command executes the HMM baseline for that seed
- A `--run-all` flag executes all 45 runs sequentially (35 TS2Vec + 10 external)
- Completed runs are skipped on re-execution (idempotent)
- Failed runs log the failure and continue to the next run
- No previous experiment directory is overwritten
- `experiments/registry.json` correctly distinguishes TS2Vec conditions from external baseline methods

**DS-04 Tests That Must Pass:**
- V-INV-007 (re-running same seed/config produces identical outputs)

**Estimated Complexity:** Medium

**Implementation Order:** 14

---

### M14 — Statistical Analysis

**Purpose:** Produce the complete statistical results table and all test outputs required for the paper. Wraps M11's statistical outputs into publication-ready summaries.

**Inputs:**
- `experiments/{exp_id}/evaluation/metrics_per_run.csv` (across all runs)
- `experiments/{exp_id}/evaluation/wilcoxon_results.csv`

**Outputs:**
- `outputs/final/tables/ablation_table.csv` — primary ablation table (mean ± std per condition)
- `outputs/final/tables/statistical_tests.csv` — Wilcoxon p-values with Holm-Bonferroni correction
- `outputs/final/tables/economic_validity_summary.csv`
- `outputs/final/tables/sensitivity_analysis.csv` — Stage 2 results (if conducted)

**Dependencies:** M11

**Main Classes/Functions:**
- `AblationTableBuilder` — formats metrics_aggregated into paper-ready table
- `StatisticalSummaryBuilder` — formats Wilcoxon results with significance markers
- `SensitivityAnalysisReporter` — compares Stage 1 and Stage 2 HDBSCAN results

**Definition of Done:**
- All tables are self-contained CSVs that can be directly imported into a LaTeX table
- Statistical significance markers (*, **, ***) applied at corrected α thresholds
- Sensitivity analysis table clearly labeled; includes note if Stage 1 and Stage 2 conclusions differ
- All p-values reported to three decimal places

**DS-04 Tests That Must Pass:**
- V-EXP-003 (Wilcoxon paired test with correction — final output verified here)
- V-EXP-004 (external baselines HMM and KM-PCA under five-seed protocol)

**Estimated Complexity:** Low

**Implementation Order:** 15 (parallel with M12 after M11)

---

### M15 — Paper Artifact Generator

**Purpose:** Collect all final outputs into a clean, self-contained paper artifact bundle suitable for archival (Zenodo / Hugging Face Hub) and reviewer inspection.

**Inputs:**
- All outputs from M12 and M14
- `checkpoints/` directory
- `configs/` directory
- `data/raw/manifest.json`

**Outputs:**
- `outputs/final/paper_artifacts/` — reproducibility bundle
  - `figures/` — all 4 figures in 3 formats
  - `tables/` — all final tables
  - `checkpoints/` — all 20 branch checkpoints (symlinked or copied)
  - `configs/` — all experiment config snapshots
  - `data_manifest.json` — download provenance
  - `reproduction_guide.md` — step-by-step instructions for independent reproduction

**Dependencies:** M12, M14

**Main Classes/Functions:**
- `ArtifactBundler` class
  - `collect_figures() -> None`
  - `collect_tables() -> None`
  - `collect_checkpoints() -> None`
  - `write_reproduction_guide() -> None`
  - `generate_bundle() -> None`

**Definition of Done:**
- Bundle contains all required files with no broken symlinks
- `reproduction_guide.md` explains every step from `git clone` to final figures
- Bundle is verifiable: a checksum manifest lists SHA-256 for every file
- The bundle directory can be uploaded as-is to Zenodo or Hugging Face Hub

**DS-04 Tests That Must Pass:**
- V-INV-007 (reproducibility — bundle enables independent reproduction)

**Estimated Complexity:** Low

**Implementation Order:** 16

---

## 5. Coding Order

The following sequence minimizes debugging by ensuring every dependency is complete and tested before it is consumed.

```
Step 1:  M0    — Project Bootstrap + configuration + utilities
Step 2:  M1    — Data Acquisition
Step 3:  M2    — Data Validation
Step 4:  M7    — TS2Vec Wrapper    ← can start here in parallel with M3–M6
Step 5:  M3    — Temporal Alignment
Step 6:  M4    — Feature Engineering
Step 7:  M5    — Temporal Split           ← M5 and M6 may be done in either
Step 7:  M6    — Window Generation        ← order or in parallel; both depend only on M4
         [Pause: run V-LEAK tests now, before any training begins]
Step 9:  M8    — Branch Training
Step 10: M9    — Fusion
Step 11: M10.5 — External Baselines (HMM + KM-PCA)  ← can start any time after Step 7 (M5); independent of Steps 9–10
Step 12: M10   — HDBSCAN Clustering
Step 13: M11   — Evaluation
Step 14: M12   — Visualization        ← parallel with Step 15
Step 15: M14   — Statistical Analysis ← parallel with Step 14
Step 16: M13   — Experiment Runner
Step 17: M15   — Paper Artifact Generator
```

**Why this order minimizes debugging:**

- Leakage bugs discovered in Steps 2–7 do not corrupt any trained models or experiment results. If leakage were discovered after Step 9, all 20 branch training runs would need to be discarded.
- The TS2Vec wrapper (M7) is built early because it depends only on M0 and can be verified in isolation against synthetic data before any real windows are available.
- **M5 (Temporal Split) and M6 (Window Generation) are both listed at Step 7 because they no longer depend on each other** — DS-02 v1.2 corrected M6 to window the full M4 feature matrix directly and categorize by anchor timestamp, rather than depending on M5's pre-split files (see DS-02 v1.2 changelog, IMP-01 v1.3 changelog, `AUDIT_LC4_ADDENDUM.md`). M5's `train_features.parquet`/`test_features.parquet` remain required later, as M10.5's input and for M11/Stage 9 economic-validity joins — just not as a prerequisite for M6.
- The temporal integrity tests (V-LEAK-001 through V-LEAK-004) are run as a mandatory gate between Step 7 and Step 9. No training begins until all four leakage checks pass. This gate applies to M8 (TS2Vec branch training); M10.5 (external baselines) does not train on windowed tensors and is not subject to this specific gate, but still consumes the same leakage-checked feature split data.
- M10.5 (External Baselines) is placed after M9 in this sequence purely for narrative continuity with the TS2Vec chain; it has no dependency on M6, M7, M8, or M9 and, in practice, can be implemented as early as Step 7 if convenient, since its only dependency is M5.
- Evaluation (M11) is built before the experiment runner (M13), because the runner orchestrates M11 internally. Building M11 first ensures its interfaces are stable. M11 requires both M10 and M10.5 to be complete, since it evaluates all 9 unique methods (7 TS2Vec conditions + 2 external baselines) together.

---

## 6. Milestones

### Milestone 1 — Repository Ready

**Modules:** M0

**Completion Criteria:**
- Directory structure exists
- `configs/base.yaml` contains all controlled variables from DS-03 §3
- Config loader validates schema and fails informatively on missing fields
- Seed control, device detection, and logging utilities are tested
- `.gitignore` excludes data/, checkpoints/, and outputs/
- `requirements.txt` and `environment.yml` are complete, including pinned TS2Vec commit

**Gating question:** Can another researcher clone this repository and install all dependencies in one command?

---

### Milestone 2 — Data Pipeline Complete

**Modules:** M1, M2, M3, M4, M5, M6

**Completion Criteria:**
- All four raw Parquet files downloaded with provenance metadata
- All DS-04 data validations (V-DATA-001 through V-DATA-005) pass
- All DS-04 temporal leakage validations (V-LEAK-001 through V-LEAK-004) pass
- Feature matrix contains 35,045 rows × 29 columns with zero NaN/Inf
- Train/test split boundary is exact and verified by assertion
- All 8 window `.npy` files generated with correct shapes
- `tests/` contains passing unit tests for M1–M6

**Gating question:** Can the data pipeline be re-run from scratch and produce byte-identical outputs?

---

### Milestone 3 — Representation Learning Complete

**Modules:** M7, M8, M9, M10.5

**Completion Criteria:**
- TS2Vec wrapper tested with synthetic windows; output shape `[N, 64]` confirmed
- All 20 branch training runs complete (4 timeframes × 5 seeds)
- Checkpoint format matches ADR-010; GPU checkpoints load on CPU
- All 7 TS2Vec conditions × 5 seeds produce fused embeddings of shape `[N, 256]`
- Fusion module trainable parameter count verified as 0
- Projection matrix determinism confirmed (V-MODEL-004)
- HMM and KM-PCA baselines complete for all 5 seeds each (10 external baseline runs)
- `tests/` contains passing tests for M7, M8, M9, M10.5

**Gating question:** Given a fixed seed and config, do two independent runs produce identical embeddings (TS2Vec conditions) and identical labels (external baselines)?

---

### Milestone 4 — Experiment Complete

**Modules:** M10, M11, M13

**Completion Criteria:**
- Stage 1 HDBSCAN grid search complete; `hdbscan_params_locked.json` written
- Cluster labels produced for all 7 TS2Vec conditions × 5 seeds (35 clustering runs)
- All geometric and economic metrics computed for all 9 unique methods (7 TS2Vec conditions + HMM + KM-PCA) and saved to CSV
- Wilcoxon tests performed with Holm-Bonferroni correction on the 3 pre-registered primary comparisons; results in `wilcoxon_results.csv`
- Experiment runner executes all 45 runs sequentially from a single command (35 TS2Vec + 10 external baseline)
- All V-EXP tests (V-EXP-001 through V-EXP-005) pass, including V-EXP-004 (external baselines)
- All V-INV tests (V-INV-001 through V-INV-008) pass

**Gating question:** Can the complete experiment (all 45 runs) be re-run and produce the same conclusions?

---

### Milestone 5 — Paper-Ready Outputs Complete

**Modules:** M12, M14, M15

**Completion Criteria:**
- All 4 figures generated in PNG, PDF, SVG formats
- PDF font embedding verified
- All final tables formatted and self-contained as CSVs
- Ablation table contains primary results for all 4 cumulative conditions
- Statistical tests table contains corrected p-values and effect sizes
- Paper artifact bundle is complete and self-contained
- `reproduction_guide.md` enables end-to-end reproduction from a clean environment

**Gating question:** Could a reviewer reproduce every figure and table from the public artifact bundle without contacting the authors?

---

## 7. Risk Register

| ID | Risk | Cause | Impact | Mitigation | Priority |
|----|------|--------|--------|------------|----------|
| R-01 | TS2Vec repository unavailable at pinned commit | Upstream deletion or force-push | Blocks M7 entirely | Fork the repository immediately at project start and record fork URL in README as fallback (ADR-001) | High |
| R-02 | HDBSCAN produces fewer than 2 clusters for all grid configurations | Embedding space too diffuse; `min_cluster_size` too large | Stage 1 grid search fails; experiment stalls | Implement fallback grid expansion per ADR-004 before the experiment runner is called; log as WARNING | High |
| R-03 | GPU memory insufficient for TS2Vec training on full window tensor | Large N_windows × 48 × 7 tensors; batch accumulation | Training crashes mid-run | Implement gradient accumulation in `TS2VecBranch.train`; batch size is config-driven (ADR-020) | Medium |
| R-04 | Leakage discovered after training runs are complete | Silent bug in forward-fill or window generation | All 20 training runs invalid; significant rework | Mandatory leakage gate (Steps 2–8 in §5) before any training begins; V-LEAK tests must pass first | High |
| R-05 | Experiment runner interrupted mid-run (power, timeout, OOM) | Long-running training on limited hardware | Partial results; inconsistent state | Implement run registry and `--resume` flag in M13; `load_or_train` idempotency in M8 | Medium |
| R-06 | Non-determinism between runs on GPU | CUDA non-deterministic operations | V-INV-007 fails; reproducibility claim weakened | Document known non-deterministic ops; use `torch.use_deterministic_algorithms(True)` where possible; document exceptions | Medium |
| R-07 | `ccxt` API rate limiting or Binance API changes | Binance modifies endpoint or enforces new limits | M1 download fails or produces incomplete data | Implement exponential backoff retry in M1; verify checksums on re-download; document expected download time | Low |
| R-08 | Silhouette score tie across multiple HDBSCAN configurations in Stage 1 | Grid search produces equal scores | Non-deterministic parameter locking | Break ties by lower `min_cluster_size` first, then lower `min_samples`; document tiebreaker in M10 | Low |
| R-09 | `umap-learn` produces non-deterministic results | UMAP has known randomness even with fixed seed | Figure 1 (UMAP) not reproducible | Pin `umap-learn` version; set `random_state` in UMAP constructor; document in M12 | Low |
| R-10 | Parquet files grow too large for Git LFS limits | 8 window `.npy` files × float32 ≈ large; Parquet artifacts | Git history bloated | Enforce `.gitignore` for all data/ and outputs/ in M0; provide download instructions in README | Low |

---

## 8. Repository Completion Checklist

### Repository Structure
- [ ] All directories from the defined structure exist
- [ ] No source code files exist outside `src/`, `scripts/`, `tests/`, `configs/`
- [ ] `.gitignore` excludes: `data/`, `checkpoints/`, `outputs/`, `experiments/`, `logs/`, `__pycache__/`, `.env`

### Configuration
- [ ] `configs/base.yaml` contains all hyperparameters from DS-03 §3 controlled variables table
- [ ] `configs/experiment_{condition}.yaml` exists for all 7 TS2Vec conditions (1TF, 2TF, 3TF, 4TF, BL-15m, BL-4h, BL-1d)
- [ ] `configs/baseline_hmm.yaml` and `configs/baseline_kmpca.yaml` exist for the 2 external baselines
- [ ] No hardcoded numeric hyperparameters exist anywhere in source files
- [ ] Config schema validation is implemented and tested

### Data
- [ ] All 4 raw Parquet files exist with provenance metadata
- [ ] `data/raw/manifest.json` complete with checksums
- [ ] Feature matrix: 35,045 rows, 29 columns, zero NaN/Inf
- [ ] Train/test split boundary verified by assertion
- [ ] 8 window `.npy` files with correct shapes

### Models
- [ ] TS2Vec installed at pinned commit; commit hash in `requirements.txt`
- [ ] 20 branch checkpoints exist (4 TF × 5 seeds)
- [ ] All checkpoints contain ADR-010 required fields
- [ ] Fusion module verified: 0 trainable parameters, deterministic projection
- [ ] HMM baseline models exist for all 5 seeds, with BIC selection table saved per seed
- [ ] KM-PCA baseline models exist for all 5 seeds, with Silhouette grid-search table saved per seed

### Experiments
- [ ] 45 experiment runs complete (35 TS2Vec: 7 conditions × 5 seeds; 10 external: HMM + KM-PCA × 5 seeds)
- [ ] `hdbscan_params_locked.json` exists and was written before multi-TF runs
- [ ] All 45 `metrics_per_run.csv` rows present (35 TS2Vec + 10 external baseline)
- [ ] Wilcoxon results with Holm-Bonferroni correction in `wilcoxon_results.csv` (3 pre-registered primary comparisons: 2TF/3TF/4TF vs. 1TF)

### Evaluation
- [ ] All DS-04 V-DATA tests pass
- [ ] All DS-04 V-LEAK tests pass
- [ ] All DS-04 V-MODEL tests pass
- [ ] All DS-04 V-EXP tests pass
- [ ] All DS-04 V-INV tests pass

### Figures
- [ ] Figure 1 — UMAP (PNG, PDF, SVG)
- [ ] Figure 2 — Silhouette vs N_TF curve (PNG, PDF, SVG)
- [ ] Figure 3 — Market state timeline 2023 (PNG, PDF, SVG)
- [ ] Figure 4 — Architecture diagram (PNG, PDF, SVG)
- [ ] PDF font type verified (`pdf.fonttype = 42`)

### Documentation
- [ ] `README.md` contains: project summary, installation, reproduction steps, expected outputs, dataset download instructions, TS2Vec pinned commit with fallback fork URL
- [ ] Every `src/` module has a module-level docstring
- [ ] Every public class and function has a docstring with purpose, inputs, outputs
- [ ] `docs/` contains IMP-01 and links to DS-01 through DS-04

### Reproducibility
- [ ] End-to-end pipeline runs from a single entrypoint script
- [ ] Re-running the pipeline with the same seeds produces identical metrics
- [ ] `environment.yml` and `requirements.txt` are consistent and tested on a clean install
- [ ] Paper artifact bundle is self-contained and uploadable
- [ ] `reproduction_guide.md` verified by a dry-run

---

*End of IMP-01 — Implementation Roadmap*
*Design phase complete. Implementation begins at M0.*
