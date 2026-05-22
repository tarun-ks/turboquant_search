# Multi-seed streaming protocol

This document explains how to (re)produce the multi-seed evidence for
**Tables 2, 12, 13** of the PVLDB paper, plus the new capacity-corrected
PQ-retrain rows (m=96 on Deep-10M; m=128 on SIFT-10M).

Two scripts:

| Script | What it does |
|---|---|
| `streaming_multiseed.py` | Runs one experiment at N seeds and writes raw per-cell CSVs to `experiments/results/`. |
| `tables_from_multiseed.py` | Reads CSVs and writes LaTeX-ready table fragments + a human-readable `multiseed_summary.txt`. |

## Run order

Five experiments. Standard seed set: `42 123 7777`. Total wall-clock estimate
on a 16 GB MacBook Pro M-series: **~13 hours**. Recommend running the 10M
experiments overnight.

```bash
# Set thread count if not already exported
export TQS_THREADS=$(sysctl -n hw.ncpu)

# 1. SIFT-1M ingestion controls (Table 2)
python experiments/streaming_multiseed.py \
    --experiment sift1m --seeds 42 123 7777
# wall-clock: ~1.5 hours

# 2. Deep-10M streaming (Table 12)
python experiments/streaming_multiseed.py \
    --experiment deep10m --seeds 42 123 7777
# wall-clock: ~2.5 hours

# 3. SIFT-10M streaming (Table 13)
python experiments/streaming_multiseed.py \
    --experiment sift10m --seeds 42 123 7777
# wall-clock: ~4 hours

# 4. Capacity-corrected PQ baseline on Deep-10M (m=96, 8-bit)
python experiments/streaming_multiseed.py \
    --experiment deep10m_pqhigh --seeds 42 123 7777
# wall-clock: ~2.5 hours

# 5. Capacity-corrected PQ baseline on SIFT-10M (m=128, 8-bit)
python experiments/streaming_multiseed.py \
    --experiment sift10m_pqhigh --seeds 42 123 7777
# wall-clock: ~2.5 hours

# 6. Regenerate LaTeX tables from all CSVs
python experiments/tables_from_multiseed.py
```

## What the orchestrator does per seed

`set_all_seeds(seed)` is called at the start of every per-seed run. It seeds:

- `numpy.random.seed(seed)`
- `random.seed(seed)` (Python's stdlib)
- `faiss.seed_global(seed)` when available (FAISS k-means uses this)
- The IVF-TQ rotation matrix `Π` (via the `seed=...` argument to `IVFTurboQuantIndex`)

Per-experiment, additional seeded sources:

- **Shuffled ingestion (SIFT-1M):** `np.random.RandomState(seed).permutation(800_000)`
- **Mean-shift ingestion (SIFT-1M):** `np.random.RandomState(seed).standard_normal(d)` for the shift direction
- **k-means init** for both IVF-TQ coarse partition and IVF-PQ codebook (handled inside the library and FAISS via the global seed)

A determinism check runs at the start of each per-seed pass:

```
[14:01:33]  determinism check seed=42: -0.138264, +0.647689, +1.523030, -0.234153, -0.234137
```

If two runs at the same seed print different prefixes, a randomness source
is leaking — investigate before reporting numbers.

## Output schemas

All CSVs land under `experiments/results/`:

### `streaming_sift1m_multiseed.csv`

| seed | condition | index | state | recall10 |
|---|---|---|---|---|
| 42 | original | ivf_tq | 200K | 89.6 |
| 42 | original | ivf_tq | 1M | 92.1 |
| 42 | original | ivf_pq | 200K | 73.6 |
| 42 | original | ivf_pq | 1M | 69.4 |
| 42 | shuffled | ivf_tq | 200K | 89.3 |
| … |

3 conditions × 2 indexes × 2 states × 3 seeds = **36 rows**.

### `streaming_deep10m_multiseed.csv` and `streaming_sift10m_multiseed.csv`

| seed | dataset | m_pq | bits_per_sub | index | vectors_indexed | recall10 | retrain_cum_seconds |
|---|---|---|---|---|---|---|---|
| 42 | deep10m | 48 | 8 | ivf_tq | 1000000 | 87.54 | 0.0 |
| 42 | deep10m | 48 | 8 | ivf_pq_stale | 1000000 | 82.16 | 0.0 |
| 42 | deep10m | 48 | 8 | ivf_pq_retrain | 1000000 | 82.16 | 0.0 |
| 42 | deep10m | 48 | 8 | ivf_tq | 2000000 | 87.31 | 0.0 |
| … |

3 indexes × 10 batch states × 3 seeds = **90 rows per dataset**.

The `pqhigh` CSVs follow the same schema with `m_pq=96` (Deep) or `m_pq=128`
(SIFT).

## Statistical tests in `tables_from_multiseed.py`

For each cell (one index, one dataset, one state):

- **Mean ± 95% CI**: t-distribution with df = n_seeds − 1. With n=3, df=2,
  the critical value is `t.ppf(0.975, df=2) ≈ 4.303` — so the CI half-width
  is `4.303 × sample_std / sqrt(3)`. CI widths around 0.1–0.5 pp are
  expected; wider CIs (≥1 pp) suggest a high-variance cell that may need
  more seeds.

- **Paired t-test on within-seed differences**: for changes over time
  (e.g., IVF-TQ 1M vs 10M) and cross-index comparisons (e.g., IVF-TQ vs
  IVF-PQ at 10M). `scipy.stats.ttest_rel` is the canonical scipy call.
  P-values reported to 3 decimals; `p < 0.001` displayed as such.

## Validation checks (run automatically at the end of each orchestrator run)

The orchestrator asserts:

1. **No NaN/missing values** anywhere in the CSV.
2. **Correct seeds-per-cell**: every cell (grouped by `condition × index ×
   state` for SIFT-1M, or `index × vectors_indexed` for 10M) has exactly
   the expected number of seeds.

`tables_from_multiseed.py` additionally reports a human-readable summary at
`experiments/results/multiseed_summary.txt`:

```
=== Deep-10M (streaming_deep10m_multiseed.csv) ===
seeds: [42, 123, 7777]
  IVF-TQ: 1M=87.52±0.18 | 10M=86.71±0.24 | Δ=-0.81pp (p=0.072)
  IVF-PQ stale: 1M=82.18±0.21 | 10M=78.83±0.27 | Δ=-3.35pp (p=0.008)
  IVF-PQ retrain: 1M=82.18±0.21 | 10M=78.96±0.26 | Δ=-3.22pp (p=0.010)
  IVF-TQ vs IVF-PQ stale at 10M: Δ=+7.88±0.29pp (p=0.003)
  IVF-PQ stale vs retrain at 10M: Δ=-0.13±0.18pp (p=0.314)
```

Sanity-check this output before regenerating tables. Particularly:

- **Direction of change** — IVF-PQ stale should be negative on both 10M
  datasets; IVF-TQ should be small-negative on Deep-10M and small-positive
  on SIFT-10M (per the v1 numbers). If any sign flips, there's a seeding
  bug somewhere.
- **CI widths** — should be 0.1–0.5 pp for most cells. A CI of 5 pp means
  within-seed runs are noisier than expected; investigate before reporting.

## Recovery from interruptions

The orchestrator writes the CSV incrementally after each seed completes,
so if you Ctrl-C or the laptop sleeps mid-run, you can resume by re-running
with the remaining seeds:

```bash
# If seed 42 already completed, just run the rest
python experiments/streaming_multiseed.py \
    --experiment deep10m --seeds 123 7777
# Then manually append to the existing CSV, or re-run all 3 seeds for a clean overwrite.
```

For the simplest workflow: complete each experiment in a single run; if
interrupted, restart with all 3 seeds (the orchestrator overwrites the
CSV on each invocation).

## Paper integration after CSVs exist

Once the CSVs are populated, regenerate the LaTeX tables and paste them
into the PVLDB paper:

1. `python experiments/tables_from_multiseed.py`
2. Inspect `experiments/results/multiseed_summary.txt` for sanity.
3. Replace **Table 2** (`tab:streaming_controls_main`) in
   `paper/tex/pvldb_main.tex` with the contents of
   `experiments/results/table2_streaming_sift1m_controls.tex`.
4. Replace **Tables 12 and 13** (in the appendix) with the contents of
   `table12_streaming_deep10m.tex` and `table13_streaming_sift10m.tex`.
5. Add the capacity-corrected PQ tables as new appendix items, or fold
   them into the main paper if space permits.
6. Update §4.1 prose: "within statistical noise" becomes "paired-t-test
   $p={...}$"; "the gap widens from 5.4pp at 1M to 7.9pp at 10M" becomes
   "paired difference of $+7.88 \pm 0.29$ pp at 10M, $p=0.003$."
7. Remove "Multi-seed streaming experiments" from the §7 Limitations
   future-work list.

## What to do if multi-seed numbers differ from v1 single-seed

Three scenarios, in order of likelihood:

- **Within 1 pp of single-seed**: expected. Update tables; declare the
  experiment done.
- **>1 pp difference on any cell**: informative. The previous "within 1 pp"
  statement in v1 was wrong but in a way you can characterize. Investigate
  which seed was the outlier and report the corrected number. Add a
  sentence to §4.2 saying the v1 single-seed result was within (or outside)
  the 95% CI of the multi-seed mean.
- **IVF-TQ vs IVF-PQ gap shrinks materially (paired-difference $p > 0.05$
  at 10M)**: the streaming-operational story weakens. Expand to 5 seeds
  (≈10 hours more compute) before reframing §1 and §4. Do not suppress
  this result — knowing it before submission is the entire point of
  multi-seed evidence.
