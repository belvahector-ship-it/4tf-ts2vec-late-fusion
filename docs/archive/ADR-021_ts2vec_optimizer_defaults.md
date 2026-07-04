# ADR-021 — Accept TS2Vec's Internal Optimizer Defaults (No Monkey-Patch/Subclass)

> **Status:** Approved
> **Date:** 2026-07-03 (session 9, Claude Code)
> **Author:** RSE, decision by project author (Belva Fahrozi Chiangmaitri, P31202502702)
> **Relates to:** ADR-001 (TS2Vec as External Pinned Dependency), ADR-002 (Independent Per-Branch TS2Vec Training), ADR-010 (Full Reproducibility Bundle Checkpoint Policy)
> **Supersedes:** none (refines/extends ADR-001)

> **Note on placement:** This ADR is recorded as a standalone addendum (same
> pattern as `AUDIT_LC4_ADDENDUM.md`) because `DS-01_v1.1.md` is currently the
> version pinned as source-of-truth by `MIGRATION_TO_CLAUDE_CODE.md`, and DS-01
> ADRs are immutable once approved. **This ADR (ADR-021) must be folded into the
> DS-01 ADR Index at the next DS-01 version bump.**

---

## Context

M7 (TS2Vec Wrapper) implementation revealed that the pinned TS2Vec
(`third_party_reference/ts2vec/`, commit
`b0088e14a99706c05451316dc6db8d3da9351163`) constructs its `torch.optim.AdamW`
optimizer **inside** `TS2Vec.fit()` as a local variable:

```python
optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)
```

Two consequences follow, both structural to the upstream code (not to our wrapper):

1. **`weight_decay` is not configurable.** Upstream passes only `lr`, so AdamW
   uses the torch library default `weight_decay = 0.01`. Our
   `configs/base.yaml training.weight_decay: 0.0001` is therefore **never read**
   by the encoder's optimizer.
2. **The optimizer object is not exposed**, so no `optimizer_state_dict` can be
   captured for a true optimizer-state resume (ADR-010 lists this field).

Two options were considered:

- **(a)** Monkey-patch or subclass TS2Vec to inject `weight_decay = 1e-4` and to
  expose the optimizer for state-dict capture.
- **(b)** Accept TS2Vec's internal defaults (`weight_decay = 0.01`, weight-level
  resume only), leaving the third-party code untouched.

## Decision

**Option (b): accept TS2Vec's internal optimizer defaults. Do not monkey-patch
or subclass the TS2Vec library. Use TS2Vec exactly as pinned.**

## Rationale

1. `weight_decay` is an **internal parameter of TS2Vec** (a third-party,
   vendored/pinned dependency), not part of this research's contribution. The
   novelty of this project is the Cross-Timeframe Attention and Late Fusion
   architecture built *on top of* TS2Vec — not the internal tuning of TS2Vec
   itself.
2. The project is deliberately **neutral toward upstream TS2Vec**: TS2Vec is
   used as-is at the pinned commit, without editing or re-wrapping its
   optimizer. This is consistent with ADR-001 (pinned dependency; vendoring is
   read-reference only). Monkey-patching the optimizer would violate the spirit
   of ADR-001 and add new complexity and risk to already-validated third-party
   code (269 tests passing).
3. `base.yaml training.weight_decay: 0.0001` is **not removed** from config — it
   remains as documentation of the design-intended value. The wrapper explicitly
   records (already implemented) that this value is not actually used by
   `TS2Vec.fit()`, and is superseded by the library default `0.01`.
4. This will be documented honestly as a **limitation** in the paper's
   methodology: TS2Vec's internal optimizer uses the library default
   (`weight_decay = 0.01`) and is not overridden, in order to preserve full
   compatibility with the validated upstream implementation.
5. Related consequence: `optimizer_state_dict` remains `None` in checkpoints
   (same root cause — the optimizer is not exposed by TS2Vec). **Weight-level
   resume** (re-`fit` from saved encoder weights) — not true optimizer-state
   resume — is accepted as the same class of limitation, for the same reason
   (neutrality toward upstream).

## Consequences

- `src/models/ts2vec_wrapper.py` records `config_weight_decay` (0.0001) and
  `effective_weight_decay` (0.01, read live from the torch AdamW default) in
  every checkpoint bundle, and logs a WARNING at construction when they differ.
- ADR-010's `optimizer_state_dict` field is written as `None` for all branch
  checkpoints. M8's `load_or_train` resume is therefore weight-level, not
  optimizer-state-level.
- No change to `configs/base.yaml` (the `weight_decay: 0.0001` line stays as
  documented intent).
- The paper's Limitations/Methodology section must state this explicitly
  (see Rationale #4).
- If a future finding requires honoring `weight_decay = 1e-4`, that would be a
  new ADR superseding this one, and would require the monkey-patch/subclass
  path (with its ADR-001 implications) — it is out of scope now.

## Status

**Approved (final).** M8 (Branch Training) is built on this assumption.
