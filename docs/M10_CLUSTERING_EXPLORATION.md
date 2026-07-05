# M10 Clustering Exploration — Why HDBSCAN Fails Here, What the Embeddings Actually Contain, and the Recommended Path

> **Status:** Exploration COMPLETE — recommendation awaiting author decision (this is NOT a unilateral change; M1–M9 untouched)
> **Date:** 2026-07-05 (session 11, Claude Code)
> **Trigger:** ADR-004/ADR-018 protocol (HDBSCAN, 2 ≤ k ≤ 8) proved unsatisfiable on the real M9 fused embeddings
> **Author of decision-to-explore:** Belva F. Chiangmaitri — "the 2≤k≤8-via-HDBSCAN constraint was a means, not the goal; understand the data structure first, then recommend"
> **Data used:** REAL artifacts only — 20 M8 checkpoints (validated), M9 fused embeddings (7 conditions × 5 seeds, train split; primarily seeds 42/123 for exploration)

---

## 1. Problem Statement

DS-02 Stage 8 / ADR-004 prescribe: HDBSCAN on the 256-dim fused embeddings, grid `min_cluster_size ∈ {50,100,200} × min_samples ∈ {5,10,20}`, select by Silhouette with **2 ≤ k ≤ 8** (ADR-018), lock from 1TF, apply to all conditions.

On the real embeddings this fails structurally:

| Attempt | Result |
|---|---|
| HDBSCAN raw 256-dim, full 26,238 train windows, `mcs=100, ms=10` | **k = 38, noise = 70%**, ~6 min/fit |
| HDBSCAN raw 256-dim, 5000-sample, `mcs ∈ {100…1500}` | **k = 0** (100% noise) |
| UMAP(5/10/15) → HDBSCAN, full ADR-004 grid, 2 seeds × 4 conditions (216 runs) | 1TF/2TF: reachable (k=2–5, noise 2–16%); **3TF/4TF: k ≥ 33 always, never ≤ 8**; eligible silhouettes weak (0.10–0.31, some negative) |

So the prescribed method cannot produce comparable 2–8-state partitions across conditions. This document records the systematic exploration of **why**, and what captures the market-state structure instead.

---

## 2. Literature Grounding (similar work)

- **Market-regime detection literature uses partitioning/model-based clustering with small k, essentially never density-based clustering.** Wasserstein k-means for market regimes (Horvath et al., arXiv:2110.11848); sliced-Wasserstein k-means regimes (arXiv:2310.01285); non-parametric online regime detection (arXiv:2306.15835); HMM regime models. Typical number of regimes reported: **2–5** (bull/bear; bull/bear/neutral; up to 5-state hidden semi-Markov models — Springer Risk Management 2022).
- **Density-based clustering degrades in high dimension** due to distance concentration (arXiv:2401.00422): as dimensionality grows, pairwise distances concentrate and density contrast fades — precisely HDBSCAN's operating assumption breaking.
- **Deep-representation clustering practice** (DEC, arXiv:1511.06335; autoencoder-clustering surveys) partitions embeddings with k-means-family objectives, typically after dimensionality reduction; cluster-validity indices on deep embeddings must be cross-checked from multiple angles (arXiv:2403.14830).
- **Internal cross-check from this project itself:** the M10.5 HMM external baseline, selected purely by BIC on the same 1h features, chose **4 states** — squarely in the literature's 2–5 range.

Implication: choosing a partitioning method with small fixed k is the *norm* for this problem class; the original HDBSCAN choice was the unusual one.

---

## 3. Structural EDA of the Real Fused Embeddings

Script: standalone probes (scratchpad; nothing in `src/` changed). Data: train fused embeddings, seed 42 (PCA cross-checked on seed 123).

### 3.1 Intrinsic dimensionality grows with the number of timeframes

| Condition | PCs for 50% var | 80% | 90% | 95% | (seed 123: 90%) |
|---|---|---|---|---|---|
| 1TF | 4 | 15 | **27** | 38 | 27 |
| 2TF | 4 | 18 | **39** | 61 | 39 |
| 3TF | 9 | 36 | **61** | 87 | 60 |
| 4TF | 12 | 43 | **72** | 101 | 72 |

Multi-TF fusion genuinely adds information (higher effective rank), reproducibly across seeds. This supports the research hypothesis that more resolutions produce richer state representations — and simultaneously explains why a density method has progressively less contrast to work with.

### 3.2 Distance concentration worsens with #TFs

Coefficient of variation of pairwise distances (3000-sample): 1TF **0.172** → 2TF 0.162 → 3TF 0.125 → 4TF **0.111**. Lower CV = distances more uniform = density valleys vanish (the curse-of-dimensionality signature from arXiv:2401.00422). This is the quantitative *why* behind HDBSCAN's 3TF/4TF failure.

### 3.3 The embeddings are strongly clusterable — just not density-separated

Hopkins statistic (0.5 = uniform/no structure; →1 = strongly clusterable): **0.91–0.92 for every condition**, both in raw 256-dim and PCA-15. There *is* strong structure; it is simply not organized as isolated density islands.

### 3.4 Visual structure: temporal micro-patterns, not blobs

2D UMAP of all four conditions ([figure](figures/m10_exploration_umap2d_seed42.png)): every condition forms hundreds of short **filaments**. Colored by time, each filament is a contiguous time segment — the direct consequence of stride-1 windows overlapping 47/48 rows (adjacent windows are near-duplicates tracing continuous trajectories). 1TF/2TF filaments still cohere into larger masses; 3TF/4TF filaments disperse. Colored by volatility, high-vol windows concentrate on specific filaments — the embedding encodes market conditions.

**Filament-hypothesis test (honest revision):** if HDBSCAN clusters were pure filaments, each cluster would be 1–2 contiguous time-runs. Measured (1TF, UMAP-5, `mcs=50`, k=195): median **4 runs/cluster** (mean 5.7); only 45/195 clusters are ≤2 runs. So HDBSCAN's micro-clusters are **recurring day-scale micro-patterns** (similar segments revisited ~4–6 times), not single filaments. Refined conclusion: HDBSCAN *does* capture something real — micro-pattern recurrence — but at a granularity (~200 clusters) that is irreconcilable with macro market states (2–8), and the macro structure is not density-separated at any parameter setting.

---

## 4. Method Comparison (the main experiment)

Grid: conditions {1TF,2TF,3TF,4TF} × seeds {42,123} × k ∈ 2…12 × methods {KMeans (raw 256-d), GMM-diag (raw 256-d), Agglomerative-Ward (6000-sample)}; metrics: Silhouette (5000-sample, fixed rs), Davies-Bouldin, Calinski-Harabasz, temporal persistence (mean run length of the time-ordered label sequence, in hours), GMM BIC; plus cross-seed label stability (ARI seed42-vs-123). Full tables: `experiments/m10_exploration/method_comparison.csv`, `stability.csv`.

### 4.1 Headline results

- **KMeans and GMM produce valid partitions for every condition and every k** — including 3TF/4TF where HDBSCAN failed entirely. KMeans ≈ GMM numerically (silhouette differs at the 3rd decimal); Ward is consistently worse on every metric.
- **Best-k by silhouette is small (2–3) for all conditions**, echoing the regime literature. Silhouette at best-k: 1TF 0.16 → 4TF 0.10 (weaker geometric separation in richer spaces — consistent with §3.2, and an honest property to report, not hide).
- **k=4 snapshot** (all conditions): silhouette 0.08–0.14, DBI 2.1–3.3, CH 1.9k–3.9k, persistence 12.5–14.3 h.
- **Temporal persistence is regime-like, not flicker:** 17–33 h mean run length at best-k; ~13 h at k=4 (on 1-hour windows). Market states last half-a-day to a day+.
- **Cross-seed stability (mean ARI over k=2..8):** 1TF 0.62–0.64, 2TF 0.71–0.74, 3TF/4TF 0.51–0.55 (KMeans ≈ GMM). Far above chance (~0) — the states are substantially reproducible across TS2Vec training seeds, more so for fewer TFs.

### 4.2 Honest negative findings

- **GMM BIC is not a usable k-selector here:** BIC decreases monotonically to the grid edge (argmin at k=12 for every condition/seed) — the classic large-n, non-Gaussian-manifold failure mode. Any k chosen "by BIC" would be an artifact of the grid boundary.
- **Ward linkage** underperforms on all metrics and needs subsampling (O(n²) memory) — no advantage to justify it.
- Absolute silhouettes are modest everywhere (≤0.16). These embeddings do not contain textbook well-separated clusters; they contain a structured continuum that partitioning slices into meaningful regions (cf. §5). Any paper claim should be phrased accordingly.

---

## 5. Economic Validity (do the states MEAN anything?)

For seed 42, k ∈ {3,4,5,6,8}, KMeans & GMM, all conditions: Kruskal-Wallis tests of `close_return_1h` and `hl_range_1h` (volatility) across cluster labels. Full table: `experiments/m10_exploration/economic_probe.csv`.

- **Every configuration separates BOTH returns and volatility with overwhelming significance** (return: p < 1e-13 everywhere; volatility: down to p ≈ 1e-146).
- **The k=4 states are directly interpretable as canonical market states** — consistently across all four conditions and both methods (per-cluster means, seed 42):
  - a **bull state**: +8…+10 bp/h mean return, low vol
  - a **bear/high-vol state**: −8…−10 bp/h, the highest volatility (~0.0120 hl_range)
  - a **calm/neutral state**: ~+2 bp/h, the lowest volatility (~0.0100)
  - a **choppy/neutral state**: ~+1 bp/h, elevated volatility (~0.0111)
- **Return separation strengthens with more timeframes** (KW H at k=4: 1TF 67.6 → 3TF 83.0 → 4TF 80.2; at k=8: 1TF 79.9 → 4TF 103.0; bull/bear means widen from ±8 bp at 1TF to ±10 bp at 3TF/4TF). The multi-TF conditions do not merely survive the method change — they discriminate states *better*, which is evidence in favor of the project's core hypothesis and feeds directly into M11's pre-registered comparisons.

---

## 6. Recommendation (for author decision — not yet implemented)

**Replace HDBSCAN with K-Means at a fixed k applied identically to every condition; primary k = 4; sensitivity analysis over k ∈ {3,5,6,8} (replacing ADR-004 Stage 2).**

Justification, mapped to evidence:

1. **Method family:** partitioning matches both the data (Hopkins 0.91 structured continuum, no density islands — §3) and the regime literature (§2). KMeans over GMM: numerically indistinguishable here (§4.1), but simpler, faster, with a natural deterministic out-of-sample rule (nearest centroid — the `approximate_predict` analog for test windows), and no reliance on the BIC selector that demonstrably fails (§4.2). GMM soft-probabilities can be a supplementary output later if wanted.
2. **Fixed k = the cleaner controlled experiment.** The original design controlled *HDBSCAN params* across conditions but let k float — which real data pushed to absurdity (k=3 vs k=40 across conditions, incomparable). Fixing k pins the *granularity* of the state space so the only varying factor remains `active_timeframes` (the project's single-independent-variable invariant, INV-001/V-EXP-001 in spirit).
3. **k=4 (primary)** is supported by four independent lines: interpretability (bull/bear/calm/choppy — §5), the project's own HMM baseline choosing 4 by BIC (M10.5), the regime literature's 2–5 norm (§2), and proximity to the silhouette optimum (2–3) while retaining enough granularity for economic contrast. ADR-018's k ≤ 8 spirit is preserved (4 ≤ 8); the sensitivity sweep documents robustness to that choice.
4. **No noise label.** Every hour is in *some* market state (finance has no "no-state" hours); dropping HDBSCAN's noise concept removes an awkward artifact (70% noise) rather than a feature. M11's noise-exclusion logic simply becomes a no-op for these labels.
5. **Everything downstream keeps working:** labels are `[N] int` per condition/seed/split exactly as M11 expects; geometric metrics (Silhouette/DBI/CH) remain the comparison metrics; economic validity (Stage 9) already shows strong signal (§5).

**What this changes formally if approved:** a new ADR (supersedes ADR-004's algorithm choice and amends ADR-018's mechanism; k ≤ 8 outcome preserved), DS-02 Stage 8 rewrite (KMeans fixed-k protocol + nearest-centroid test assignment), M10 module rework (`KMeansStateClusterer` replacing the HDBSCAN path; the HDBSCAN implementation is kept in-repo as the documented negative result), matching test updates, then the real 35-run clustering (7 conditions × 5 seeds).

**What this does NOT change:** M1–M9 artifacts and code (untouched, per instruction); M10.5 external baselines; the five-seed protocol; the pre-registered M11 comparisons.

---

## 7. Reproducibility of this exploration

All probes are standalone scripts run against committed code + gitignored real artifacts; result CSVs under `experiments/m10_exploration/` (`eda_summary.csv`, `method_comparison.csv`, `stability.csv`, `economic_probe.csv`, `economic_probe_k4_snapshot.csv`), UMAP figure committed at `docs/figures/m10_exploration_umap2d_seed42.png`. Every number in this document traces to one of those files or to the session transcript tables.

*End of exploration report.*
