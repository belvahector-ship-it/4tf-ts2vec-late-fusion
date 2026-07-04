# ADR-024 — KM-PCA `n_components` Clamped to Available Feature Dimension

> **Status:** Approved
> **Date:** 2026-07-04 (session 10, Claude Code — M10.5 implementation)
> **Author:** RSE, decision by project author (Belva Fahrozi Chiangmaitri, P31202502702)
> **Relates to:** DS-03 §4 (External Baselines), IMP-01 M10.5, base.yaml `external_baselines.km_pca`
> **Supersedes:** none

> **Placement:** Standalone addendum (same pattern as ADR-021/022/023). Must be
> folded into the DS-01 ADR Index at the next DS-01 version bump (together with
> ADR-021/022/023, which are also still pending fold-in).

---

## Context

The KM-PCA external baseline specification contained an internal contradiction
present since the earliest design draft, surfaced when M10.5 implementation
began:

| Source | Claim |
|---|---|
| DS-03 §4 (line 149) | "KM-PCA: K-Means + PCA(10) … Input: **7 features, 1h resolution**" |
| base.yaml `km_pca.input_timeframe` | `1h` (→ 7 features) |
| base.yaml `km_pca.pca_components` | `10` |
| IMP-01 §M10.5 | repeats both claims without resolution |

**Mathematical impossibility:** PCA cannot produce more components than
`min(n_samples, n_features)`. With 7-feature input, `PCA(n_components=10)`
raises a `ValueError` in scikit-learn. The value `10` is almost certainly a
residue from an early draft that imagined a higher-dimensional feature space
(e.g. a multi-timeframe concatenation), never reconciled after the final
"7 features, 1 timeframe" decision was locked for all baselines (including the
1TF/BL-1h condition KM-PCA is compared against). The multi-timeframe source
reference (Sobreiro et al., 2026) does not specify any PCA configuration — it
uses manual feature engineering for *forecasting*, not clustering — so it is
not the origin of `10`. No document was found that justifies `10` explicitly.

## Decision

**Option 1 — clamp:** `n_components = min(configured_pca_components, n_features_available)`.
For the 7-feature / 1h input this yields **`PCA(n_components=7)`** — a full-rank
orthogonal projection (whitening / decorrelation), NOT dimensionality reduction.
Implemented as an **explicit clamp** (computed at fit time from the data), not a
silent override.

Two rejected alternatives: Option 2 (`PCA=5`) — an arbitrary number with no
document basis; Option 3 (use all 28 features so `PCA(10)` is literal) — breaks
the documented "7 features, 1h resolution" and gives KM-PCA an unfair
information advantage over the 1TF/BL-1h condition it is benchmarked against.

## Rationale

1. **Documentation consistency:** "7 features, 1h resolution" is locked in
   DS-03 §4, base.yaml, IMP-01 M10.5, and PROPOSAL §3.1/§3.6. No document ever
   specified PCA components exceeding 7.
2. **Fair comparison (INV-001, single independent variable):** KM-PCA must have
   information access **identical** to the 1TF/BL-1h condition (same 7 features,
   1h) — not more.
3. **No invented parameters:** clamping introduces no new undocumented number.
4. **Methodologically valid:** `PCA(n_components = n_features)` is a legitimate
   boundary case — PCA here functions as covariance-structure standardization
   (whitening/decorrelation) before clustering, not compression. No information
   is discarded; the data is only rotated to its principal-component basis.

## Consequences

- `KMeansPCABaseline` computes `effective_pca_components = min(pca_components,
  n_features)` at fit time (7 for the 1h/7-feature input) and exposes it for
  audit; the PCA `explained_variance_ratio_` is saved with each run.
- For this input PCA is full-rank whitening, **not** dimensionality reduction.
  This is documented as a **limitation of the external baseline** (to appear in
  the paper's Limitations section), not a bug.
- `base.yaml`'s `km_pca.pca_components: 10` is reinterpreted as an **upper
  bound / ceiling**, with a clarifying comment; the effective value is computed
  at runtime via the clamp.
- No research decision changes; DS-03 §4 gains a footnote at its next revision.

## Status

**Approved (final).** Implemented in `src/models/external_baselines.py`
(M10.5); regression test asserts effective components = 7 (not 10) and that no
`ValueError` is raised on the 7-feature input.
