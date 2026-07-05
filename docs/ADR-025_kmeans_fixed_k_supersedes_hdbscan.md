# ADR-025 — Fixed-k K-Means Replaces HDBSCAN for Market-State Clustering (Supersedes the Algorithm Choice of ADR-004)

> **Status:** Approved (final)
> **Date:** 2026-07-06 (session 13, Claude Code)
> **Author:** RSE, decision by project author (Belva Fahrozi Chiangmaitri, P31202502702)
> **Supersedes:** the **clustering-algorithm choice** of ADR-004 (HDBSCAN two-stage protocol) and the **k-selection mechanism** of ADR-018. It does NOT supersede the *intent* of ADR-018 (interpretable, bounded number of states) — the chosen k=4 sits within the original k≤8 bound.
> **Relates to:** ADR-002 (per-branch training), ADR-003 (fixed projection), ADR-016 (per-window z-score), DS-02 Stage 8, DS-03 §5.
> **Placement:** standalone addendum (same pattern as ADR-021–024). Must be folded into the DS-01 ADR Index at the next DS-01 version bump, at which point ADR-004's status becomes **"Superseded (algorithm choice) by ADR-025"** and ADR-018's mechanism note is updated. Justification evidence: `docs/M10_CLUSTERING_EXPLORATION.md`, `docs/PIPELINE_SCIENTIFIC_DISSECTION.md`, `docs/M10_FINAL_METHODOLOGY.md`.

---

## Context

ADR-004 prescribed HDBSCAN on the 256-dim fused embeddings, grid-searched on 1TF (`min_cluster_size∈{50,100,200}×min_samples∈{5,10,20}`), selecting the configuration with **2≤k≤8** clusters (ADR-018) by Silhouette, locking those parameters, and applying them to every condition. On the **real** M9 fused embeddings this protocol is **structurally unsatisfiable**:

- HDBSCAN on the raw 256-dim embeddings yields **k=38 at 70% noise** (mcs=100) or **k=0 / 100% noise** (larger mcs).
- UMAP(5/10/15)→HDBSCAN across the full grid (216 runs, 2 seeds × 4 conditions): 1TF/2TF can reach 2≤k≤8, but **3TF/4TF never fall below k≈33** at any parameter or reduced dimension; eligible silhouettes are weak (0.10–0.31, some negative).

A systematic, literature-grounded exploration established **why** (not a bug): stride-1 windowing produces continuous delay-embedding **trajectories/filaments** (Takens; Perea & Harer), and multi-timeframe concatenation adds **near-orthogonal subspaces** (CCA 15m↔1h≈0.99 vs 4h↔1d≤0.72), raising intrinsic dimensionality (27→72 PCs) and **distance concentration** (CV 0.172→0.111). Density-based clustering has progressively less density contrast to exploit, while the data remain strongly structured (Hopkins 0.91) and partitionable. K-Means/GMM produce valid partitions for **all** conditions; k=4 yields interpretable, economically-separated states (bull / bear-high-vol / calm / choppy; Kruskal-Wallis p<1e-13), consistent across seeds; the project's own HMM baseline (M10.5) independently selected 4 states by BIC, matching the market-regime literature (2–5 states).

## Decision

**Cluster the fused embeddings with K-Means using a FIXED number of clusters k, IDENTICAL across all seven TS2Vec conditions.**

- **Primary k = 4.** Reported as the main clustering result.
- **Sensitivity analysis over k ∈ {3, 5, 6, 8}** (replaces ADR-004's Stage-2 per-condition grid search), reported as robustness.
- **Metric:** Euclidean (unchanged).
- **Test-window labels:** assigned by **nearest trained centroid** (the deterministic out-of-sample analog of HDBSCAN's `approximate_predict`; no re-fit on test).
- **Reproducibility:** `random_state` = the run's seed; `n_init=10`.
- **No noise label.** Every hour belongs to a market state; the HDBSCAN `-1` noise concept is dropped (M11's noise-exclusion step becomes a harmless no-op for these labels).

## Rationale

1. **Matches the data geometry and the literature.** Partitioning fits a strongly-structured continuum with no density islands (Hopkins 0.91; no valleys); density-based clustering does not. Regime detection literature is dominated by k-means/GMM/HMM with small k.
2. **Fixed k is the cleaner controlled experiment.** ADR-004 controlled *HDBSCAN parameters* across conditions but let k float — which the real data pushed to k=3 vs k=40 across conditions, making cross-condition comparison meaningless. Fixing k pins the **granularity** of the state space so the sole varying factor stays `active_timeframes` (INV-001 / V-EXP-001 in spirit).
3. **k=4 is quadruply supported:** interpretability (four canonical states, consistent across all conditions and both KMeans & GMM), the project's HMM-BIC=4 concordance, the 2–5-state regime literature, and proximity to the silhouette optimum (2–3) while retaining enough granularity for strong economic separation.
4. **k≤8 intent preserved** (4 ≤ 8); ADR-018's goal (a small, interpretable state count) is honored — only its HDBSCAN *mechanism* changes.
5. **K-Means over GMM:** numerically indistinguishable here (silhouette differs at the 3rd decimal), but simpler, faster, with a natural nearest-centroid out-of-sample rule and no reliance on GMM's BIC selector, which was **empirically shown unusable** on this data (BIC argmin pinned to the grid edge, k=12). GMM soft assignments remain available as a supplementary output if later desired.
6. **Downstream compatibility:** outputs remain `[N] int` labels per condition/seed/split exactly as M11's `GeometricEvaluator`/`EconomicEvaluator` expect; geometric metrics (Silhouette/DBI/CH) remain the comparison metrics; economic validity already shows strong signal.

## Consequences

- The HDBSCAN implementation (`src/models/hdbscan_clustering.py` + tests) is **retained in-repo as a documented negative result**, not deleted — it is part of the honest methodological narrative.
- A new/updated M10 module implements the `KMeansStateClusterer` path (fixed k, nearest-centroid predict, Stage-8 parquet unchanged in schema); tests updated accordingly.
- **DS-02 Stage 8** must be rewritten (KMeans fixed-k protocol + nearest-centroid test assignment) at the next DS-02 version bump; **DS-01** ADR-004 status → "Superseded (algorithm choice) by ADR-025" and ADR-025 added to the index at the next DS-01 bump.
- `configs/base.yaml` `clustering` block gains `n_clusters_primary: 4` and `n_clusters_sensitivity: [3,5,6,8]` (the HDBSCAN grid keys retained for provenance/negative-result reproducibility).
- The confirmatory **35 real clustering runs** (7 conditions × 5 seeds) proceed on the existing M9 embeddings; no retraining of M1–M9 is required or performed.
- No research question in DS-03 changes; the pre-registered M11 comparisons (2TF/3TF/4TF vs 1TF) are unaffected in structure — only the label-producing algorithm changes, uniformly across all conditions.

## Status

**Approved and locked.** Confirmatory execution (module rework + 35-run) is the next phase; this ADR fixes the method so that phase is purely mechanical.
