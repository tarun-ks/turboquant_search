# Streaming results at 10M scale: Deep-10M (single-seed per-batch trajectory)

Referenced from main paper §4.2. The 3-seed headline values are in
**Table 5 of the main paper**; this file reports the single-seed per-batch
view for context.

## Setup

- Dataset: Deep-10M (10M vectors, 96-dim ResNet image features).
- Training: first 1M vectors used for IVF coarse partition and PQ codebook.
- Streaming: 9 batches of 1M each added incrementally.
- Recall@10 recomputed every batch against ground truth on the cumulative database (10K queries).
- IVF-PQ "retrain" variant: codebook re-trained from scratch on full cumulative data after every batch.
- Seed: 42.

## Per-batch trajectory (sub-matched regime: $m_{\text{PQ}}=48$, $b=8$, 384 bits/vec; IVF-TQ at $b=4 +$ sign-bit, 512 bits/vec)

| Vectors indexed | IVF-TQ | IVF-PQ (stale) | IVF-PQ (retrain/1M) | Retrain cum. |
|---|---|---|---|---|
| 1M (trained)    | 87.54% | 82.16% | 82.16% |   0 s |
| 2M              | 87.31% | 81.18% | 81.11% |  19 s |
| 3M              | 87.24% | 80.70% | 80.67% |  43 s |
| 4M              | 86.96% | 80.32% | 80.29% |  70 s |
| 5M              | 86.89% | 79.87% | 80.01% | 100 s |
| 6M              | 86.84% | 79.61% | 79.81% | 135 s |
| 7M              | 86.70% | 79.32% | 79.60% | 173 s |
| 8M              | 86.74% | 79.20% | 79.15% | 214 s |
| 9M              | 86.71% | 79.04% | 79.10% | 260 s |
| 10M             | **86.65%** | 78.79% | 78.94% | **309 s** |
| **Change 1M→10M** | **−0.89 pp** | −3.37 pp | −3.22 pp | — |

## Reading

- The IVF-TQ vs. IVF-PQ recall gap widens from 5.38 pp at 1M to 7.86 pp at 10M.
- Re-training every 1M new vectors (the most aggressive practical schedule)
  costs 309 s of cumulative compute over the run and recovers only +0.15 pp,
  confirming that the gap is not a codebook-staleness issue alone but
  a capacity issue at sub-matched bit budgets (main paper §4.1).
- For the multi-seed (3-seed paired-$t$) summary across sub-/bit-/super-matched
  PQ memory regimes, see main paper **Table 5**.
