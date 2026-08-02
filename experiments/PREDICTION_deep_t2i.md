# Pre-registered prediction: corpus-growth mechanism on Deep-10M & T2I-10M

**Written: 2026-08-01 11:55:10 EDT — BEFORE running any Deep/T2I oracle or
rank-margin experiment. Git HEAD at write time: fbb4479.**

This file records the mechanism's quantitative prediction for Deep-10M and
T2I-10M *before* the data exists, so the prediction is on record as preceding
the result. Do not edit after the runs complete.

## What we already know (SIFT-10M, measured, 3 seeds / 2 seeds)

Corpus growth 1M→10M, nprobe=20, rr=0, bit-matched (SIFT: PQ m64/b10, TQ 4+sign):
- PQ recall −2.31pp (DEGRADES); TQ +0.65pp; exact/uncompressed +3.90pp.
- Oracle codebook (trained on full 10M) −1.73pp ≈ stale −2.31pp → staleness irrelevant.
- Coverage RISES 88.8→92.7%; margins SHRINK (median 0.0008→0.0006, −25%).
- Score-err RMS: PQ 0.0042, TQ 0.0021 (PQ ≈ 2× TQ), ~N-independent.
- frac(err>margin @rank10): PQ 75.5→81.5%, TQ 62→69%.
- SIFT margin/err ratio at 10M = 0.0006/0.0042 = **0.14**.

## The puzzle to explain (from the original paper, bit-matched, ~5 bits/dim)

| dataset | dim | PQ (m,b) | PQ Δ 1M→10M | behaviour |
|---|---|---|---|---|
| SIFT-10M | 128 | 64, 10 | −2.31pp | DEGRADES |
| Deep-10M | 96 | 48, 10 | −0.84pp | stable |
| T2I-10M | 200 | 100, 10 | −0.89pp | stable |

All three: 2 dims per PQ subvector, b=10 (1024 levels/subvector), ~5 bits/dim.

## PREDICTIONS (falsifiable)

**Universal (must hold on BOTH Deep and T2I, else mechanism is wrong):**
- P1. Oracle ≈ stale (within ~0.6pp). Staleness is NOT the driver anywhere.
      FALSIFIER: oracle beats stale by >1pp (→ staleness matters there).
- P2. Coverage (true top-10 in probed cells) RISES from 1M to 10M.
- P3. Coherence holds: lossy IVF-TQ never beats exact uncompressed IVF
      (TQ − exact < 0 at every checkpoint). STOP if violated.

**Discriminator (explains why Deep/T2I are stable but SIFT degrades):**
- P4. Degradation magnitude tracks frac(err>margin)_PQ at 10M and its rise
      with N. Since Deep/T2I degrade ~2.7× LESS than SIFT (−0.85 vs −2.31pp),
      the mechanism REQUIRES their frac(err>margin)_PQ at 10M to be materially
      LOWER than SIFT's 81.5% — predict **Deep/T2I frac(err>margin)_PQ ≈
      60–73%** (below SIFT), and/or a smaller rise from 1M→10M.
- P5. The margin/error ratio at 10M is the single best predictor. SIFT = 0.14
      (degrades). Predict **Deep and T2I margin/err > 0.20** (stable side).

**Which lever — margins or distortion? (the specific call requested):**
- P6. Because bit budget AND subvector dimensionality are matched (2 dims/sub,
      b=10 for all three), PQ's score-error RMS should be COMPARABLE across
      datasets — predict Deep/T2I PQ err_RMS ∈ **[0.003, 0.006]** (within ~1.5×
      of SIFT's 0.0042), NOT dramatically lower.
- P7. Therefore the mechanism predicts the discriminator is **MARGINS, not
      distortion**: Deep and T2I have LARGER margins (relative to PQ error)
      and/or LESS margin shrinkage than SIFT. This is the primary falsifiable
      call for BOTH datasets.
      (Sub-case still mechanism-consistent: if PQ err_RMS turns out markedly
      lower on Deep/T2I, distortion contributes too — that is fine. The
      mechanism is only FALSIFIED if neither margins nor error explain the
      stability, i.e. Deep/T2I look like SIFT on margin AND error yet stay
      stable.)

**TQ prediction:** IVF-TQ err_RMS will scale with dimension differently than
PQ; predict TQ stays at or below PQ's error on both (TQ ≤ PQ err_RMS), keeping
TQ flat/stable on both datasets (TQ Δ ≥ −0.5pp).

## Verdict rule
- CONFIRMED-GENERAL if: P1–P3 hold, and Deep/T2I stability is explained by
  P4/P5/P7 (lower frac(err>margin), higher margin/err ratio, margins as the
  lever). Then we have explained the paper's dataset-dependence via one
  mechanism.
- INCOMPLETE (STOP, report) if: Deep/T2I are stable but frac(err>margin)_PQ ≈
  SIFT's high value and margin/err ≈ SIFT's low value — i.e. margin/distortion
  structure does NOT distinguish them. Something beyond distortion-vs-margin
  is at play; do not build the paper on it.
