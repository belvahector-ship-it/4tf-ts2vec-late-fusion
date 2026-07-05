# Scientific Dissection of the Multi-Timeframe TS2Vec Pipeline
## A Mathematical Reverse-Engineering of Stages M1→M10, and Why the Observed Geometry Is Theoretically Expected

> **Role:** senior-research-scientist analysis — *understanding*, not engineering.
> **Date:** 2026-07-05 (session 12). **Code untouched**; every number below is measured from the REAL artifacts (20 validated M8 checkpoints, M9 embeddings, M10 exploration) or derived from the actual vendored source (`third_party_reference/ts2vec/`).
> **Companion documents:** `docs/M10_CLUSTERING_EXPLORATION.md` (empirics), `experiments/m10_exploration/*.csv` (raw tables).
> **New evidence produced for this dissection** (seed 42, train split): per-branch PCA/participation-ratio, cross-branch CCA, concat-vs-projected geometry, embedding-norm statistics — cited inline as **[Probe]**.

---

## Executive summary (the one-paragraph answer)

The pipeline maps 4 years of OHLCV into a **shape-space of 48-hour market episodes**. Per-window z-scoring makes every window a scale-free *shape*; stride-1 overlap makes consecutive shapes near-identical, so the embedding is a **continuous trajectory (a filamented curve), not a cloud of separated blobs** — this is delay-embedding geometry (Takens; Perea & Harer), and it is why HDBSCAN sees either ~200 recurring micro-patterns or pure noise. TS2Vec's contrastive objective, operating with raw dot products under weight decay, pushes embeddings onto a **thin spherical shell** (measured norm CV ≈ 3%) and spreads them (uniformity pressure, Wang & Isola 2020). Adding timeframes adds **nearly-orthogonal subspaces** for coarse TFs (CCA: 15m↔1h ≈ 0.99 = near-duplicate; 4h↔1d ≤ 0.72 = complementary), so intrinsic dimensionality **adds** rather than mixes (90%-variance dim: 27 → 39 → 61 → 72), which mechanically produces distance concentration (CV 0.172 → 0.111) and kills density contrast — while simultaneously adding *real* information (economic separation strengthens: KW-H 67.6 → 83.0). The fixed Johnson–Lindenstrauss projection is measured to be a **benign quasi-isometry** (pairwise-distance correlation 0.91–0.96) and is *not* the cause of fragmentation — the fragmentation already exists pre-projection (concat distance-CV 0.0987, *worse* than post-projection 0.1109). Every observed phenomenon — HDBSCAN's graded failure, KMeans' adequacy, rising intrinsic dimension, rising concentration, improving economic separation, falling silhouette — is the **expected consequence of this chain**, and the apparent paradox "more information, worse clustering metrics" dissolves once one distinguishes *geometric compactness* from *information content*.

---

## STAGE 1 — Raw features: what the 7 numbers are

**Object entering:** the aligned hourly lattice `X_raw ∈ R^{35,064×21}` (timestamp + OHLCV × 4 TFs, M3). **Object leaving (M4):** `F ∈ R^{35,045×28}`, 7 features × 4 TF suffixes.

**Mathematical content of the 7 features (per TF suffix):**

| feature | formula (essence) | mathematical role |
|---|---|---|
| `open_return, high_return, low_return, close_return` | log/relative differentials of consecutive bars | first differences of log-price ⇒ (approximately) **stationarized increments** of the price process; the four differ only in which point of the bar is sampled |
| `volume_zscore` | (v − rolling μ₂₀)/(rolling σ₂₀ + ε) | **locally standardized activity** — a dimensionless innovation of volume relative to its recent regime |
| `hl_range` | (high−low)/open | normalized intrabar dispersion — a **realized-volatility proxy** (per-bar range estimator, cf. Parkinson-type estimators) |
| `body_ratio` | |close−open|/(high−low+ε) ∈ [0,1] | **bar geometry**: directional efficiency of the bar (trend-bar vs doji), scale-free by construction |

Three properties matter downstream: **(i) stationarity by construction** — every feature is a *relative* quantity; absolute price level is deliberately annihilated (a translation/scale invariance imposed at the data level); **(ii) heavy-tailedness** — return-type features inherit the leptokurtosis of crypto returns; volume z-scores are bounded-ish but skewed; **(iii) internal correlation** — the four return variants are strongly mutually correlated (they sample the same bar), hl_range correlates with |returns| (vol–range relation) and with volume_zscore (volume–volatility relation); body_ratio is the most independent axis. Effective feature rank is therefore ≈ 4–5, not 7 — mild, *intentional* redundancy that gives the encoder multiple noisy views of the same latent bar-state.

**What these features cannot contain:** microstructure (order-book, trades), cross-asset context, absolute price/scale (removed on purpose), calendar semantics (only implicit through periodic patterns), and any information destroyed by bar aggregation (intra-bar path).

**Why exclude RSI/SMA/EMA/MACD (ADR-015):** every classical indicator is a *deterministic causal filter* of OHLCV — RSI is a rectified-ratio of smoothed increments, SMA/EMA are fixed low-pass FIR/IIR kernels, MACD a difference of two EMAs. By the **data-processing inequality**, they add zero information beyond OHLCV; they only pre-commit to particular kernel shapes and time constants. A dilated-convolution encoder *is* a bank of learnable causal filters whose kernels are optimized by the contrastive objective — i.e., the encoder's first layers can synthesize indicator-like filters if and only if they help the objective. This is the standard representation-learning argument (learned features supplant hand-crafted ones; Bengio, Courville & Vincent, 2013, *Representation Learning: A Review*, IEEE TPAMI), instantiated for technical indicators.

**Can TS2Vec learn higher-order temporal structure from these primitives?** Yes, in principle and per its benchmark record: TS2Vec (Yue et al., AAAI 2022; arXiv:2106.10466) learns from raw/primitive channels across 125 UCR/UEA datasets without feature engineering; dilated CNNs are universal approximators of causal filters with exponentially growing receptive fields (van den Oord et al., WaveNet, 2016). The restriction here is not the encoder but the **information ceiling of stationarized OHLCV at 1h**: whatever is not a function of the last 48 hourly bars' shape cannot be learned (see Stage 3's normalization caveat).

---

## STAGE 2 — Multi-timeframe input: what a "timeframe" actually adds *in this pipeline*

**Critical implementation fact** (from M3, `src/data/alignment.py`): all four branches live on the **same 1-hour lattice**. The `_15m` columns are 15m bars *aggregated up* to 1h (OHLC first/max/min/last, volume-sum); `_1h` is native; `_4h`/`_1d` are **forward-filled step functions** (each value constant for 4 or 24 consecutive lattice points). So the four "timeframes" are four **scale-space views of the same process sampled on a common grid**:

- fine→anchor (15m→1h): aggregation ≈ identity at the lattice resolution — *almost no new information relative to the 1h view*;
- coarse→anchor (4h,1d): smoothing + hold ⇒ **longer effective memory** (a 48-step window of `_1d` features spans 48 *days* of underlying price history via its plateaus, vs 48 hours for `_1h`).

**[Probe] evidence — this is exactly what the trained branches encode.** Top-5 CCA correlations between branch embedding spaces (seed 42, 8k sample):
`15m↔1h: 0.993/0.985/0.973/0.958/0.934` (near-identical subspaces); `1h↔4h: 0.852/0.713/…`; `4h↔1d: 0.716/0.568/…`; `15m↔1d: 0.618/0.582/…`.
So **2TF is (in this design) close to information duplication, while 3TF/4TF add genuinely new, largely complementary temporal context** — new *scales*, in the scale-space sense (Lindeberg, *Scale-Space Theory in Computer Vision*, 1994; Mallat's multiresolution analysis, 1989). This single measurement explains a swath of the empirics: 2TF behaves like a slightly-enriched 1TF (its cross-seed stability is even the highest, ARI ≈ 0.71–0.74), while the big geometric transitions happen at 3TF and 4TF.

The correct theoretical frame is **multi-resolution / hierarchical temporal context** (wavelet decompositions; temporal pyramids; multi-scale sequence models): each added coarse branch appends a *slower* state descriptor — analogous to appending low-frequency wavelet coefficients to a high-frequency description. It is *not* mere duplication (except, ironically, for the 15m branch, which the alignment design makes nearly redundant — an honest design observation worth stating in the paper).

---

## STAGE 3 — Windowing: delay embedding, deliberate redundancy, and where the filaments come from

**Objects:** `F ∈ R^{35,045×28}` → per TF: sliding windows `W=48, stride=1` → tensor `X_tf ∈ R^{N×48×7}` (train N=26,238; test N=8,760), then **per-window, per-channel z-score** (ADR-016).

**What windowing *is*, mathematically:** a **delay-coordinate embedding** of a multivariate process: the map `t ↦ (f_{t−47},…,f_t) ∈ R^{336}` lifts each time point to a point carrying its own 48-step history. Takens' theorem (1981) is the classical statement that such windows of a (deterministic) dynamical system trace a diffeomorphic copy of the attractor; Perea & Harer (*Sliding Windows and Persistence*, Found. Comput. Math., 2015) prove and exploit that **sliding-window point clouds of a signal form continuous curves/manifolds** whose topology reflects the signal's dynamics. Windowing does *not* merely "prepare batches" — it already **constructs a temporal representation**: the identity of a point is its whole recent trajectory.

**Overlap and redundancy.** With stride 1, consecutive windows share 47/48 rows: `‖x_{i+1} − x_i‖ ≤ (‖new row‖ + ‖dropped row‖)` — the sequence `x_1, x_2, …` is a **Lipschitz-continuous path** in R^336. Redundancy factor ≈ 48 (each row is reused by up to 48 windows). This is intentional: it (a) multiplies training pairs for contrastive learning, (b) yields one label per *hour* downstream, and (c) **guarantees manifold continuity** — the "filaments" seen in the UMAP figure (`docs/figures/m10_exploration_umap2d_seed42.png`) are precisely these paths, and any continuous encoder must map a continuous path to a continuous path. **Continuous latent trajectories are therefore a theorem, not a surprise.** The measured HDBSCAN behaviour (k≈195 micro-clusters that are recurring ~day-scale segments, median 4 time-runs each) is this geometry being faithfully detected at the wrong granularity for our question.

**Per-window z-score: the strongest assumption in the pipeline.** For each window and channel: `x ← (x − mean_window)/(std_window + ε)`. Information **discarded:** the window's own level and *volatility magnitude* per channel (a violently volatile 48h and a calm 48h with the same normalized shape become identical inputs). Information **kept:** the *temporal shape* — ordering, relative amplitudes, cross-channel configuration. The embedding space is thus a **shape space** of market episodes; scale re-enters only indirectly (e.g., shape correlates of volatility, cross-channel ratios — which is why clusters still separate `hl_range` distributions strongly, KW-p ≈ 1e−146, without access to raw scale). Assumptions imposed: 48h suffices to characterize a state (ADR-006); states are functions of shape, not scale; stationarity within the window is irrelevant (no detrending beyond the mean shift).

**Relation to TS2Vec/contrastive learning:** overlapping windows also serve as *natural positives* in spirit — TS2Vec's own augmentation (random crops + masking; Stage 4) generates overlapping sub-views *within* a window, the same principle at a finer scale (cf. temporal-neighborhood coding, Tonekaboni et al., 2021: nearby-in-time ⇒ similar representation).

---

## STAGE 4 — The TS2Vec encoder: what is actually computed

**Architecture as implemented** (read from `third_party_reference/ts2vec/`):

```
input  x ∈ R^{B×48×7}
  → Linear 7→64  (pointwise "input_fc")
  → timestep mask (training only): binomial p=0.5, applied AFTER input_fc
  → DilatedConvEncoder: 11 residual ConvBlocks, channels [64×10, 64out]
       block i: GELU→SamePadConv(d=2^i,k=3)→GELU→SamePadConv(d=2^i,k=3) + residual (1×1 proj if needed)
  → dropout 0.1
output per-timestep representation z ∈ R^{B×48×64}
window vector: max-pool over the 48 timesteps (encoding_window='full_series') → R^{B×64}
```

**Receptive field.** Each block contributes two convolutions of dilation 2^i (i = 0…10), each spanning (k−1)·2^i = 2^{i+1} extra steps; total RF = 1 + Σᵢ 2·2^{i+1} = 1 + 4(2^{11}−1) = **8189 timesteps** — 170× the window length. Consequence: from roughly the 6th block onward, **every timestep's representation is a function of the entire window**; the per-timestep outputs are 48 *contextual* readouts of one episode, and the max-pool selects, per channel, the strongest activation of that channel's learned pattern-detector anywhere in the window.

**Training objective** (`losses.py`, exactly): two stochastic views of the same batch are produced by (a) random overlapping crops (in `fit()`) and (b) independent binomial timestep-masking p=0.5 (in the encoder); the **hierarchical contrastive loss** then averages, over a max-pool pyramid (T=48→24→…→1), the sum of
- **instance-wise contrast** (at each timestamp: same window's two views = positive; the other 2B−2 windows in the batch = negatives) — softmax-CE over **raw dot products**;
- **temporal contrast** (within a window: same timestamp across views = positive; the other 2T−2 timestamps = negatives).

This is TS2Vec's *contextual consistency*: a timestamp's representation must be recoverable under masking/cropping (context prediction), distinct from other timestamps (temporal), and window-identity-preserving against the batch (instance). The pyramid enforces this at every temporal scale up to the whole window — i.e., **multi-scale representation consistency inside a single branch**, before any multi-TF fusion.

**Two theoretically loaded details:**
1. **Raw dot-product similarity (no cosine, no temperature).** The loss can be gamed by norm inflation; the counterweight is AdamW's weight decay (0.01 — the effective value per ADR-021). Equilibrium: embeddings settle on a **thin shell** — measured **[Probe]**: mean norm ≈ 8.0–8.9 with norm-CV ≈ 2–3% across all four branches. On a shell, dot-product contrast ≈ cosine contrast, and the **alignment/uniformity** analysis of Wang & Isola (ICML 2020, arXiv:2005.10242) applies: positives are pulled together (alignment) while the batch-softmax spreads everything else toward a uniform distribution on the sphere (uniformity). Uniformity pressure is the single most important cause of the observed *high-entropy, blob-free* geometry.
2. **No dimensional collapse.** Contrastive encoders can collapse to a low-rank subspace (Jing et al., ICLR 2022, arXiv:2110.09348). Measured **[Probe]**: participation ratios 12.7–23.2 out of 64, 90%-variance dims 24–39 — the branches use a healthy fraction of their capacity; collapse is *not* our failure mode (rather the opposite: richness).

**What "64 dimensions" means.** Formally: the final conv layer has 64 output **channels**, so each dimension is the max-pooled response of one learned nonlinear **functional of the whole window** — 64 basis functionals spanning a representation space; they are *not* disentangled semantic factors, *not* orthogonal, and individually meaningless (only the geometry of the vector is meaningful). 64 is a **capacity knob**: TS2Vec's paper uses 320 for benchmarks; DS-03 chose 64 per branch so that four branches concatenate to 256. Theoretically: 32 would bind capacity (higher alignment, lower effective rank — likely *more* clusterable, less informative), 128/256 would raise the information ceiling and, under uniformity pressure, further raise intrinsic dimension and distance concentration (worse for density clustering, better for linear probes) — the same trade-off we observe **across conditions**, relocated **within a branch**.

---

## STAGE 5 — Concatenation: where the geometry actually changes

**Operation:** `z_cond = [z_15m; z_1h; z_4h; z_1d] ∈ R^{64·m}` (ADR-013 order, ADR-002 independence: branches never saw each other's gradients).

This is **late multi-view fusion by direct sum** (multi-view representation learning; Baltrušaitis et al., *Multimodal Machine Learning*, TPAMI 2019 — "early/late fusion" taxonomy): the joint representation is the Cartesian product of per-view representations. Two limit cases bracket what can happen:
- **fully redundant views** ⇒ concat ≈ a rotation/duplication of one view: intrinsic dim unchanged, densities unchanged;
- **independent views** ⇒ the support becomes a **product manifold** M₁×…×M_m: intrinsic dimensions **add**, volume grows multiplicatively, and *fixed* N points must cover it ⇒ density decays exponentially in the added dimensions.

**[Probe] verdict: our pipeline sits between, in a very structured way.** 15m↔1h are near-duplicates (CCA ≈ 0.99) — so 1TF→2TF barely moves intrinsic dim (27→39 @90%). 4h and 1d contribute near-orthogonal subspaces (CCA ≤ 0.72/0.57) — so 2TF→3TF→4TF adds dimensions almost additively (39→61→72 fused; the raw concat of 4 branches reaches **104** @90%, PR 32.3). Simultaneously the distance-CV falls to **0.0987** (concat) — the flattest geometry anywhere in the pipeline.

Answering the stage's questions precisely: concatenation is **not mere dimension expansion** — it is *information* fusion exactly to the degree the views are complementary (here: coarse TFs yes, fine TF no); it **does** increase intrinsic dimensionality (measured), it **does** reduce density and density-contrast (measured), and the "fragmentation" is the natural appearance of a higher-dimensional product-like manifold sampled by the same N points. The implicit assumptions: views are worth equal weight (64 dims each, no scaling), and the downstream consumer can exploit a direct-sum geometry — true for centroid/variance partitioners and linear probes, false for density-mode seekers.

---

## STAGE 6 — Fixed random projection: theoretically and measurably benign

**Operation (ADR-003):** `y = P z`, `P ∈ R^{256×64m}`, Gaussian rows L2-normalized, generated once from seed 42, frozen. For 4TF this is 256→256 mixing; for 1TF it is 64→256 (an *expansion* — a random isometric-in-expectation lift, no information added or lost: rank stays 64, measured 90%-dim stays 27).

**Johnson–Lindenstrauss** (Johnson & Lindenstrauss 1984; Dasgupta & Gupta 2003 proof): random projections to k = O(ε⁻² log N) dimensions preserve all pairwise distances within (1±ε). k=256, N≈26k ⇒ comfortable regime. JL preserves (approximately): pairwise Euclidean distances ⇒ **global geometry, neighborhoods, cluster separations in the metric sense**, and — because a bi-Lipschitz map is a homeomorphism onto its image — approximate **manifold topology**. JL does **not** preserve: axis semantics (channels are fully mixed — after projection no coordinate "belongs" to a timeframe), exact densities (volumes distort by ~(1±ε)^d), and any structure that was axis-aligned (sparsity, per-view subspace identity).

**[Probe] measurements vs theory:** pairwise-distance correlation concat↔fused = **0.9098** (4TF) and 0.9645 (1TF); intrinsic dim mildly compressed (104→72 @90%; PR 32→27); distance-CV mildly *improved* (0.0987→0.1109). Interpretation: the projection behaves as a slightly-contractive quasi-isometry — **it is not the cause of the fragmentation/concentration; it marginally alleviates it.** Therefore: (a) keeping the raw concatenated space would face *worse* density geometry, so removing the projection would **not** rescue HDBSCAN; (b) the projection's real function is the one ADR-003 claims — placing all 7 conditions in a **common 256-d metric space** so that cross-condition geometric comparisons are dimension-fair. Why *fixed*: any learned projection would add trainable parameters and a second objective (confound, V-INV-003); any per-run random draw would break reproducibility; a single seeded draw is the minimal neutral choice.

---

## STAGE 7 — The final embedding: what one vector *is*

One row `y_t ∈ R^{256}` answers: **"what is the multi-scale *shape* of the last 48 hours of the market, as of hour t?"** Concretely: 4 (or m) sets of 64 max-pooled pattern-detector responses over the window's z-scored features, linearly mixed by a fixed random basis. It represents the **entire window (episode), keyed by its anchor hour** — not a single timestamp (the RF sees everything), and not a "regime label" (no discrete structure was ever imposed; a *regime* can only exist as a **region** of this space).

Measured geometry of that space (train, seed 42): a **thin spherical shell** (norm CV ≈ 3%) of radius ≈ 8; effective dimension 27 (1TF) → 72 (4TF); Hopkins 0.91 (strong non-uniform structure); organized as **Lipschitz trajectories** (stride-1) that revisit similar micro-patterns on a ~daily scale; distances mean "dissimilarity of recent-history shape across the active scales"; neighborhoods mean "hours whose preceding 48h episodes looked alike (at all active TFs)". Correct interpretation for downstream use: a **continuous state manifold** to be *partitioned* (regions = market states), not a mixture of separated modes to be *detected*. Representation quality should accordingly be judged by information content (probing/economic separation: KW p<1e−13 everywhere) *and* stability (cross-seed ARI 0.51–0.74), not by blob-ness (silhouette ≤ 0.16).

---

## STAGE 8 — The causal chain, arrow by arrow

```
(1) OHLCV → 7 stationarized features      [scale/level invariance imposed; info ceiling set]
(2) 4 TF views on one 1h lattice          [scale-space views; 15m≈1h duplicate, 4h/1d new slow context — CCA 0.99 vs ≤0.72]
(3) stride-1 windows + per-window z-score [delay embedding ⇒ Lipschitz trajectory; shape space (scale discarded); 48× redundancy]
(4) TS2Vec (dilated CNN, RF≫48; hierarchical InfoNCE; dot-product+wd) 
                                          [contextual episode functionals; shell geometry (norm CV 3%); uniformity spreads mass;
                                           healthy rank per branch (PR 13–23) — no collapse]
(5) concatenation                         [direct-sum fusion; complementary coarse branches ⇒ intrinsic dim ~adds: 27→39→61→72(→104 raw);
                                           fixed N on growing support ⇒ density contrast decays: dist-CV 0.172→0.111(→0.0987)]
(6) fixed JL projection                   [quasi-isometry (dist-corr 0.91–0.96); common 256-d arena; geometry passed through, mildly compressed]
(7) final embedding                       [continuous multi-scale shape-manifold on a shell]
(8) ⇒ intrinsic dimension ↑ with m        [(2)+(5): new orthogonal subspaces per coarse TF — measured]
(9) ⇒ distance concentration ↑ with m     [concentration of measure in higher effective dim (Beyer et al. 1999; arXiv:2401.00422) — measured]
(10) ⇒ density clusterability ↓ with m    [HDBSCAN needs density valleys; valleys flatten as (9) proceeds; trajectories additionally
                                           chain neighborhoods ⇒ micro-clusters (k≈195) or all-noise — measured, incl. UMAP-assisted]
(11) ⇒ variance partitioning stays valid  [KMeans/GMM need only anisotropic second-moment structure, which (5) *increases*;
                                           states = regions of the manifold — measured: valid partitions at all k, all conditions]
(12) ⇒ market-state discovery             [regions carry meaning: KW return p<1e-13, vol p→1e-146; bull/bear/calm/choppy at k=4;
                                           separation *strengthens* with m (KW-H 67.6→83.0) because (5) added real information]
```

Every arrow above is backed by a measurement in this repo or a cited theorem; none is speculative.

---

## STAGE 9 — Each empirical observation: expected or not?

| Observation | Verdict | Mechanism (stage) |
|---|---|---|
| HDBSCAN gradually fails 1TF→4TF | **Expected** | (9)+(10): density contrast decays smoothly as complementary subspaces accumulate; failure is *graded* precisely because 2TF adds little (CCA 0.99) and 3TF/4TF add much |
| KMeans becomes appropriate | **Expected** | (11): centroid objectives consume variance structure, which fusion increases; no density assumption to violate |
| Intrinsic dimensionality increases | **Expected** (and *desired*) | (5): near-orthogonal view fusion; it is the signature that fusion added information rather than duplicating it |
| Distance concentration increases | **Expected** | (9): concentration of measure; corollary of the previous row at fixed N |
| Economic separation improves | **Expected under the information view** | (12): added slow-scale subspaces carry regime-relevant signal; partition tests read information, not geometry |
| Silhouette changes (drops with m) | **Expected** | uniformity + higher effective dim ⇒ smaller between/within contrast; silhouette measures compactness, not content |
| Manifold appears continuous | **Expected — a theorem, effectively** | (3): stride-1 delay embedding ⇒ Lipschitz trajectories (Takens; Perea & Harer) |

**The deeper phenomenon the instruction asks about:** the *joint* pattern — "geometric clusterability falls **while** downstream meaning rises" — is a clean empirical instance of the **alignment–uniformity / information-vs-compactness trade-off** in contrastive representation learning (Wang & Isola 2020; validation-index caveats for deep embeddings, arXiv:2403.14830). Richer representations spread variance over more useful axes; classical *internal* cluster indices (silhouette/DBI/CH) and density-mode seekers reward the opposite. Nothing in the pipeline malfunctioned; two families of tools disagree because they measure different things. That disagreement, measured systematically along a controlled 1→4-scale dial, is itself the most publishable object in this project.

---

## STAGE 10 — What is this research *really* about? Three framings

**Framing A — "Temporal resolution as an information dial: how multi-scale fusion reshapes self-supervised representation geometry."**
*Contribution:* the first controlled study (single independent variable = #timeframes; seeds, encoder, data, protocol fixed) tracing representation geometry — intrinsic dimension, participation ratio, distance concentration, cross-view CCA, cluster-tendency — as scales are added, with the causal chain of Stage 8. *Theory support:* alignment/uniformity, multi-view fusion, concentration of measure, delay embeddings. *Novelty:* geometry-evolution-vs-#views is essentially uncharted for time series SSL. *Publication:* strong workshop/proceedings material (representation-learning or time-series tracks, ICONIP/ACML/PAKDD level) and solid for a SINTA-1/2 informatics journal; the measurements are cheap to reproduce and the story is self-contained.

**Framing B — "Market states as regions of a learned regime manifold: multi-timeframe contrastive representations for state discovery."**
*Contribution:* the applied result — multi-TF fusion yields states with *stronger economic identity* (KW-H ↑, bull/bear ±8→±10 bp) and canonical k=4 semantics; plus the honest negative result that density clustering is structurally inappropriate for stride-1 contrastive embeddings, with the diagnosis and the partition-based alternative. *Theory support:* regime literature (HMM/GMM/Wasserstein-k-means, 2–5 states), our HMM-BIC=4 concordance. *Novelty:* moderate on method, good on evidence quality and negative-result honesty. *Publication:* financial-informatics / expert-systems venues; the most natural fit for a SINTA-1 journal targeting applications; also the framing closest to the original proposal (least re-writing).

**Framing C — "When clustering assumptions meet learned embeddings: a theory-guided diagnosis of density vs. partitioning methods on contrastive time-series representations."**
*Contribution:* a methodological case study + diagnostic toolkit (Hopkins, PR/PCA-dim, distance-CV, cross-view CCA, temporal-purity of clusters) that predicts *which* clustering family a given embedding admits, validated on a real pipeline where the prescribed method failed for measurable reasons. *Theory support:* Beyer et al. 1999; JL; density-clustering theory; deep-clustering validation literature. *Novelty:* practical guidance papers of this type are rare for SSL time series; reviewers like actionable diagnostics. *Publication:* proceedings-friendly (methods/benchmark tracks); medium journal ceiling unless expanded with more datasets/encoders.

**Recommendation between framings** (not a decision): A is the highest-upside *scientific* framing and subsumes the Stage-8 chain as its core figure; B is the safest path to the originally planned paper with the exploration as a strengthened methodology section; C is extractable later as a companion piece. A and B are compatible: B as the main paper with A's geometry analysis as its distinctive analytical backbone.

---

## Appendix — measurement provenance

| Quantity | Where |
|---|---|
| Architecture/loss facts | `third_party_reference/ts2vec/{ts2vec.py, models/encoder.py, models/dilated_conv.py, models/losses.py}` (read verbatim) |
| PCA dims, Hopkins, distance-CV per condition | `experiments/m10_exploration/eda_summary.csv` |
| Method×k×metrics, stability ARI | `experiments/m10_exploration/{method_comparison.csv, stability.csv}` |
| Economic separation (KW), k=4 state table | `experiments/m10_exploration/{economic_probe.csv, economic_probe_k4_snapshot.csv}` |
| Per-branch PCA/PR/norms, cross-branch CCA, concat-vs-fused geometry, JL distance correlations | dissection probe, session 12 (numbers reproduced in Stages 2, 4, 5, 6) |
| UMAP 2D figure | `docs/figures/m10_exploration_umap2d_seed42.png` |

**Key literature:** Yue et al. 2022 (TS2Vec, arXiv:2106.10466) · Wang & Isola 2020 (arXiv:2005.10242) · Jing et al. 2022 (arXiv:2110.09348) · Johnson & Lindenstrauss 1984; Dasgupta & Gupta 2003 · Beyer et al. 1999 (*When is "nearest neighbor" meaningful?*) · Yang et al. 2023 (distance concentration & manifold effect, arXiv:2401.00422) · Takens 1981; Perea & Harer 2015 (sliding windows & persistence) · Lindeberg 1994 (scale-space); Mallat 1989 (multiresolution) · Bengio et al. 2013 (representation learning survey) · Baltrušaitis et al. 2019 (multimodal fusion) · Tonekaboni et al. 2021 (TNC) · van den Oord et al. 2016 (WaveNet dilated convolutions) · regime clustering: arXiv:2110.11848, 2310.01285, 2306.15835 · deep-clustering validation: arXiv:2403.14830; DEC arXiv:1511.06335.

*End of dissection.*
