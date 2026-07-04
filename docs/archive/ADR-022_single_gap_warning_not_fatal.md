# ADR-022 — `no_excessive_single_gap` Is a Warning, Not a Hard Gate

> **Status:** Approved
> **Date:** 2026-07-03 (session 9, Claude Code — first real-data pipeline run)
> **Author:** RSE, decision by project author (Belva Fahrozi Chiangmaitri, P31202502702)
> **Relates to / amends:** DS-04 V-DATA-001 (M2 Data Validation pass criteria), DS-02 Stage 1
> **Supersedes:** none

> **Note on placement:** Standalone addendum (same pattern as `AUDIT_LC4_ADDENDUM.md`
> and `ADR-021`), because DS-01 v1.1 / DS-04 v1.1 are the version-pinned
> snapshots. **This ADR must be folded into the DS-01 ADR Index and the DS-04
> V-DATA-001 wording at the next version bump of those documents.**

---

## Context

DS-04 V-DATA-001 lists, among the M2 data-validation pass criteria:
*"No single gap exceeds one candle duration by more than one unit"* — i.e. no
gap between consecutive candles may exceed 2× the candle duration (at most one
missing candle in a row). This is implemented as `check_max_single_gap` /
`no_excessive_single_gap`.

On the **first run of M1 with real Binance data** (2026-07-03), this criterion
failed for the 15m and 1h timeframes:

| TF | single gaps > 2 candles | largest gap |
|----|-------------------------|-------------|
| 15m | 15 | 24 candles (6h) @ 2020-02-19 |
| 1h | 8 | 6 candles (6h) @ 2020-02-19 |
| 4h | 0 | 2 candles |
| 1d | 0 | 1 candle |

The largest gap (2020-02-19, ~11:30→17:30 UTC) is the well-known **Binance
exchange outage** that day; the others (2021-04-25, 2021-08-13, 2020-12-21,
2020-06-28) are likewise real exchange downtime. There are no duplicates, and
the **aggregate** gap ratio is only ~0.11% (far within the 5% tolerance).

The strict single-gap criterion was written against clean assumptions and is
**unsatisfiable for real 15m/1h Binance data**, because the exchange itself
went down for multiple hours during the study period. It only holds for 4h/1d.

## Decision

**`no_excessive_single_gap` is downgraded from a hard failure to a logged
WARNING.** It is still computed, recorded in the `ValidationReport`, and logged
at WARNING level (auditable, not removed). It no longer aborts M2. The
**aggregate `gap_ratio` (5%) remains the hard data-quality gate — unchanged.**

## Rationale

1. **Real market condition, not a data defect.** Multi-hour exchange downtime
   (e.g. 2020-02-19, 6h) is a genuine property of the market being studied.
   Hard-failing on legitimate downtime is counter-productive for research that
   aims to represent real market conditions.
2. **The aggregate gap ratio is the statistically meaningful guard.** 0.11% ≪
   5% demonstrates overall dataset completeness far better than forbidding any
   single large gap.
3. **Downstream handling exists.** M3 (Temporal Alignment) forward-fills /
   reindexes to a complete 1h grid by design (DS-02 Stage 2), so these gaps are
   technically resolved before any modeling; the strict raw-level single-gap
   check adds no protection M3 does not already provide.
4. **Auditability preserved.** The gap is not silently ignored — it is logged
   as a WARNING and appears in the report summary as `[WARN]`, so a reviewer can
   still see exactly where and how large every outage was.

## Consequences

- `CheckResult` gains a `severity` field (`"error"` default, `"warning"`).
  `ValidationReport.passed` considers only error-severity checks; a new
  `warnings` property lists failed warning-severity checks. `summary()` renders
  `[WARN]` for them.
- `no_excessive_single_gap` is appended with `severity="warning"` and, on
  failure, emits an explicit WARNING naming the timeframe, gap size, and the
  aggregate gap ratio.
- **DS-04 V-DATA-001 must be reworded** at its next version bump: the single-gap
  item becomes "logged as a warning; does not abort" and the aggregate 5%
  gap-ratio remains the pass/fail gate.
- Regression tests (`tests/test_validation.py`): a large single gap with
  aggregate ratio < 5% passes validation but is recorded as a warning; an
  aggregate ratio > 5% still hard-fails.
- No change to M3–M6 or any research decision. The set of rows entering M3 is
  unchanged (M2 never dropped rows); only M2's pass/fail verdict for legitimate
  outages changes.

## Status

**Approved.** M2 now passes on real BTC/USDT data; M3 forward-fill handles the
outage gaps (verified by `tests/test_alignment.py`).
