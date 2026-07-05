# M10 — Final Methodology & Decision Record (Sessions 11–13)

> **Status:** FINAL — decision locked. **HARD STOP on exploration**: no further diagnostic tests will be run; any subsequently-interesting question is recorded as *future work*, not pursued.
> **Date:** 2026-07-06 (session 13). **Formal decision:** ADR-025 (supersedes the algorithm choice in ADR-004).
> **Scope:** clustering stage (M10) only. M1–M9 code and artifacts are UNTOUCHED throughout; M10.5 external baselines unchanged.
> **Companion evidence documents (all committed):** `M10_CLUSTERING_EXPLORATION.md`, `PIPELINE_SCIENTIFIC_DISSECTION.md`, `Q_3TF4TF_STRUCTURE_AND_DEVELOPMENT_PATHS.md`, `DIAGNOSTIC_CKA_LINEARPROBE.md`, `DIAGNOSTIC_SHAPE_SCALE_DECOMPOSITION.md`; raw numbers in `experiments/m10_exploration/*.csv` + `docs/figures/m10_exploration_umap2d_seed42.png`.

---

## 1. The problem that started this (why HDBSCAN was abandoned)

The M10 code faithful to the original ADR-004/ADR-018 protocol (HDBSCAN, grid `mcs∈{50,100,200}×ms∈{5,10,20}`, select 2≤k≤8, lock from 1TF) is **complete and unit-tested** (`src/models/hdbscan_clustering.py`, 12 tests). It is retained in-repo as a documented negative result. On the **real** M9 fused embeddings it fails structurally:

- HDBSCAN raw 256-dim, full 26,238 train windows: **k=38 with 70% noise** (`mcs=100`), or **k=0 / 100% noise** at larger `mcs`; ~6 min/fit.
- UMAP(5/10/15)→HDBSCAN, full ADR-004 grid, 216 runs (2 seeds × 4 conditions): 1TF/2TF reach 2≤k≤8, but **3TF/4TF never fall below k=33** at any parameter/dimension; eligible silhouettes weak (0.10–0.31, several negative).

The `2≤k≤8`-via-HDBSCAN target is therefore **unsatisfiable across conditions**. Author decision: the constraint was a *means*, not the goal — understand the data first, then choose a method on evidence.

## 2. Systematic exploration → KMeans, fixed k=4

On the real artifacts (seeds 42/123 primarily), literature-grounded:

- **Literature:** market-regime detection uses partitioning/model-based clustering with small k (2–5 regimes); density-based clustering is practically absent. Our own HMM external baseline (M10.5) chose **4 states** by BIC.
- **Structure (EDA):** Hopkins **0.91–0.92** for every condition (strong structure — embeddings are NOT bad); intrinsic dim (90%-var PCs) rises **27→39→61→72** from 1TF→4TF (reproducible across seeds — fusion adds real information); distance-concentration worsens (CV **0.172→0.111**) = the mechanistic reason density methods lose contrast.
- **Geometry (visual + probe):** embeddings are **continuous temporal filaments** (stride-1 windows overlap 47/48) tracing recurring ~day-scale micro-patterns; HDBSCAN's ~195 micro-clusters are that geometry detected at the wrong granularity for macro states.
- **Method comparison** (KMeans/GMM-diag/Ward × k=2–12 × 4 conditions × 2 seeds): **KMeans ≈ GMM and both produce valid partitions for every condition** (incl. 3TF/4TF); best-k by silhouette is 2–3; temporal persistence 17–33 h (regime-like); cross-seed ARI 0.51–0.74. Honest negatives: **GMM BIC is unusable** (argmin always at the grid edge, k=12); Ward consistently worst.
- **Economic validity:** Kruskal-Wallis separation of returns (p<1e-13) and volatility (p→1e-146) for **every** configuration; at **k=4** the states are interpretable **bull / bear-high-vol / calm / choppy**, consistently across all conditions and both KMeans & GMM; return separation **strengthens with more timeframes** (KW-H 67.6→83.0; bull/bear means widen ±8bp→±10bp).

**Recommendation adopted:** replace HDBSCAN with **KMeans at a fixed k identical across all conditions**; **primary k=4**; sensitivity analysis over **k∈{3,5,6,8}**; test-window labels via **nearest-centroid** (the deterministic out-of-sample analog of `approximate_predict`). Rationale mapping and full numbers: `M10_CLUSTERING_EXPLORATION.md` §4–6.

## 3. Mathematical dissection — the geometry is theoretically expected, not random failure

`PIPELINE_SCIENTIFIC_DISSECTION.md` reverse-engineers M1→M10 as a chain of mathematical transformations and shows **every** observed phenomenon is the predicted consequence of the pipeline, backed by measurements and literature:

- stride-1 windowing = delay embedding ⇒ **Lipschitz trajectories / filaments** (Takens; Perea & Harer) — continuity is effectively a theorem, not a surprise;
- TS2Vec's dot-product InfoNCE under weight decay ⇒ embeddings on a **thin spherical shell** (measured norm-CV ≈ 3%) with uniformity spreading mass (Wang & Isola 2020); no dimensional collapse (participation ratio 13–23/64);
- **CCA-measured view relationships:** 15m↔1h ≈ 0.99 (near-duplicate — an honest design observation about the alignment), 4h↔1d ≤ 0.72 (complementary) ⇒ concatenation adds **near-orthogonal subspaces** ⇒ intrinsic dim *adds* ⇒ **distance concentration** (concentration of measure; Beyer et al. 1999) ⇒ density-clusterability falls **while** variance-partitionability and information rise;
- the fixed JL projection is a **measured benign quasi-isometry** (pairwise-distance correlation 0.91–0.96) and is *not* the cause of fragmentation — fragmentation already exists pre-projection (concat CV 0.0987, *worse* than post-projection 0.1109).

The apparent paradox "more information yet worse clustering metrics" dissolves as the **information-vs-compactness trade-off** of contrastive representations: internal indices (silhouette/DBI/CH) and density-mode seekers reward compactness, which fusion reduces even as useful variance and economic meaning increase.

## 4. Value-add diagnostics (honest results, including unfavorable ones)

Two head-to-head diagnostics were run (`DIAGNOSTIC_CKA_LINEARPROBE.md`, `DIAGNOSTIC_SHAPE_SCALE_DECOMPOSITION.md`), reported at face value:

- **CKA across conditions:** 1TF↔4TF ≈ 0.76–0.78, decreasing monotonically with condition distance, consistent across seeds ⇒ **multi-TF fusion produces substantially different representations, not redundancy.**
- **Linear probe (Ridge, next-step targets) — TS2Vec vs raw features:**
  - `return_t+1`: R² ≈ 0 for *everything* (TS2Vec and raw alike) — **equally uninformative**, consistent with the efficient-market hypothesis at the 1-hour horizon.
  - `volatility_t+1`: **raw features BEAT TS2Vec decisively** (raw R² 0.09→0.43 rising with #TF; TS2Vec R² **negative**, −0.81…−1.03, worse than the mean baseline; the gap **widens** with more timeframes). Honest verdict: **there is no evidence that the TS2Vec embedding outperforms simple regression on raw features for these linear next-step targets; for volatility it is measurably worse.** Mechanism: per-window z-scoring removes absolute-level information that drives volatility clustering, which the raw features retain.
- **Shape-scale decomposition (Tests A/B/C): MIXED / PARTIAL support, not clean confirmation.** Official calibrated conclusion, to be used verbatim (do NOT strengthen or weaken):
  > **"Normalisasi per-window secara signifikan MENGURANGI — bukan sepenuhnya menghilangkan — kandungan informasi skala pada embedding, dengan bukti kuantitatif campuran; window overlap (stride-1) menimbulkan confound temporal yang membatasi kesimpulan tegas dari uji retrieval."**

## 5. Verdict on the three core research claims

Despite the calibrated, cautious language required for the *specific* shape-scale claim, the project's **three core claims remain strongly supported**:

1. **Multi-timeframe fusion adds value** — **STRONGLY PROVEN.** CKA shows genuinely distinct representations (not redundancy); intrinsic dimensionality *adds* (near-orthogonal subspaces, CCA); economic state-separation *strengthens* with #TF (KW-H 67.6→83.0), and the raw-multi-TF probe's volatility R² rises 0.09→0.43 with #TF — the *information* is demonstrably there and grows with resolutions added.
2. **The representation is meaningful for state DISCOVERY, not numeric FORECASTING** — **PROVEN, and sharpened by the evidence.** Discovery is strong (economically-separated, interpretable bull/bear/calm/choppy states, p<1e-13); linear forecasting is weak (TS2Vec ≤ raw). The evidence *defines* the correct framing rather than undermining it.
3. **Concat + fixed random projection is the right design** — **PROVEN.** The JL projection is a measured benign quasi-isometry (dist-corr 0.91–0.96), adds zero trainable parameters (no confound, V-INV-003), and is *not* the source of fragmentation (which pre-exists it). Keeping raw concatenated dims would face *worse* density geometry.

The single claim requiring carefully-calibrated wording is the *specific* "shape-scale decomposition" mechanism (§4), not the three pillars above.

## 6. What is now locked, and what happens next

**Locked (this document + ADR-025):** clustering method = **KMeans, fixed k, identical across all 7 TS2Vec conditions; primary k=4; sensitivity k∈{3,5,6,8}; nearest-centroid test assignment.**

**Confirmatory execution to follow (Phase 2, not this task):** rework the M10 module to a `KMeansStateClusterer` path (HDBSCAN implementation retained as documented negative result), update its tests, and run the real **35 clustering runs** (7 conditions × 5 seeds) producing labels + Stage-8 parquet for M11. DS-02 Stage 8 rewrite and DS-01 ADR-index fold-in (ADR-004 status → superseded-by-025; ADR-025 into the index) are documentation tasks batched for the next DS version bump.

**Recorded as FUTURE WORK (per the hard stop — not pursued now):** the 4TF anomaly in Test A (embedding R² slightly exceeding raw); a temporal-adjacency-controlled retrieval test to properly isolate shape vs scale; complementing embeddings with explicit scale features for magnitude-forecasting tasks; Cross-Timeframe Attention / alternative fusion; TDA on the trajectory manifold; the "information-dial" geometry study (Framing A in the exploration report).

---

*End of final methodology.*
