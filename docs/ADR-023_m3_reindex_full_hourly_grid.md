# ADR-023 — M3 Reindexes to a Complete Hourly Grid (Outage-Hole Fill)

> **Status:** Approved
> **Date:** 2026-07-03 (session 9, Claude Code — first real-data pipeline run)
> **Author:** RSE, decision by project author (Belva Fahrozi Chiangmaitri, P31202502702)
> **Relates to:** DS-02 Stage 2 (Temporal Alignment), DS-04 V-DATA-003 / V-LEAK-001, ADR-022 (single-gap warning)
> **Supersedes:** none

> **Placement:** Standalone addendum (same pattern as ADR-021/ADR-022). Must be
> folded into the DS-01 ADR Index and DS-02 Stage 2 at the next version bump.

---

## Context

M3 aligns all four timeframes to the 1h anchor. `build_master` originally set
`target_index = pd.DatetimeIndex(df_1h["timestamp"])` — the **actual** 1h
candle timestamps. With clean/mock data that is the full 35,064-hour grid, but
real Binance data is missing **32 hours** (exchange outages, e.g. 2020-02-19),
so the master came out to **35,032 rows** and failed the schema check.

Anchoring on actual candles would shorten the master and shift every
downstream count the pipeline depends on — including the LC-4-audited
**N_test_windows = 8,760 (exact)**, M4's 35,045-row feature matrix, and the
train/test split sizes. The M3 docstring itself states the intent: "every 1h
timestamp … has exactly one row" — i.e. a complete grid, which only held by
accident when the data had no gaps.

## Decision

**M3 reindexes the 1h anchor (and the 15m-aggregated series) to a complete,
gap-free hourly grid** spanning `[min, max]` of the observed 1h timestamps.
For every hour that had no source candle (an outage hole):

- **Prices (O/H/L/C) are last-observation-carried-forward (LOCF)** — filled
  from the most recent prior candle only; never a future candle (no
  look-ahead).
- **Volume is set to 0.0** — no trading occurred during the outage.

The 4h and 1d columns are unchanged (they are already forward-filled onto the
target index; the real 4h/1d candle covering an outage hour retains its real
volume).

## Rationale

1. A complete hourly grid is the **factual representation of elapsed time**,
   independent of exchange up/down status. Deleting outage hours would distort
   the temporal structure every downstream module is built on.
2. **LOCF prices** are the standard "no trading" assumption — price sits at the
   last known value, with **no look-ahead bias**.
3. **volume = 0** (not forward-filled) is chosen because it is *factually
   correct*: no volume traded during the outage. Forward-filling volume (the
   rejected alternative) would fabricate trading that did not happen.
4. Consistent with ADR-022: large single gaps are accepted as legitimate
   market conditions and handled technically here, exactly as anticipated.
5. **Preserves the LC-4-audited counts.** Because the base grid is the full
   35,064 hours the LC-4 audit assumed, N_test_windows=8,760 and all downstream
   exact figures remain valid — *and this was proven by re-running M4/M5/M6 on
   the reindexed real data, not assumed* (see CHECKPOINT_LATEST.md Sesi 9).

## Consequences

- `TemporalAligner._reindex_highfreq_to_grid` performs the LOCF-price /
  volume-0 fill; `build_master` builds `target_index` from a complete
  `pd.date_range(min, max, freq="1h")`.
- New `verify_grid_completeness(master)` asserts every step is exactly 1h; it
  is checked in `build_and_verify_master` alongside the schema and V-LEAK-001
  look-ahead checks.
- V-LEAK-001 (no look-ahead) continues to pass on real data (8/8 sampled
  timestamps); the LOCF fill uses only prior candles.
- Regression tests (`tests/test_alignment.py::TestOutageGapReindex`): a punched
  1h/15m hole is reindexed back, prices are LOCF, volume is 0, the grid is
  complete, and the end-to-end leakage check still passes.
- Downstream (M4–M6) values in outage windows differ slightly from the
  clean-data assumption (flat price, zero volume for those hours), but **row
  counts and all LC-4 exact figures are unchanged** (verified on real data).
- No research decision changes; DS-02 Stage 2 gains this gap-fill rule.

## Status

**Approved and verified on real BTC/USDT data.** M3 → 35,064 rows, gap-free;
M4 → 35,045; M5 test → 8,760; M6 N_test_windows → 8,760 exact, overlap 47.
