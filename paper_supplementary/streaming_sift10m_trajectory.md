# Streaming results at 10M scale: SIFT-10M (single-seed per-batch trajectory)

Referenced from main paper §4.2. The 3-seed headline values are in
**Table 6 of the main paper**; this file reports the single-seed per-batch
view for context.

## Setup

- Dataset: SIFT-10M (first 10M base vectors of SIFT-1B; 128-dim).
- Queries: 10K BIGANN query set, ground truth from the matching 10M
  `bigann_gnd.tar.gz` file.
- Training: first 1M vectors.
- Streaming: 9 batches of 1M each.
- IVF-TQ: $b=4 +$ sign-bit (effective 5 bits/coord), $L = 3162$ partitions, $n_p = 20$ (672 bits/vec total).
- IVF-PQ (sub-matched): $m = 64$ subspaces × 8 bits/sub = 512 bits/vec.
- Seed: 42.

## Per-batch trajectory (sub-matched regime)

| Vectors indexed | IVF-TQ | IVF-PQ (stale) | IVF-PQ (retrain/1M) | Retrain cum. |
|---|---|---|---|---|
| 1M (trained)    | 83.93% | 72.43% | 72.43% |   0 s |
| 2M              | 84.40% | 70.86% | 71.23% |  21 s |
| 3M              | 84.58% | 69.84% | 69.69% |  47 s |
| 4M              | 84.57% | 69.19% | 69.36% |  78 s |
| 5M              | 84.66% | 68.51% | 68.04% | 110 s |
| 6M              | 84.70% | 68.13% | 68.10% | 152 s |
| 7M              | 84.58% | 67.68% | 67.68% | 196 s |
| 8M              | 84.48% | 67.20% | 67.61% | 244 s |
| 9M              | 84.42% | 66.88% | 67.16% | 297 s |
| 10M             | **84.56%** | 66.44% | 66.67% | **354 s** |
| **Change 1M→10M** | **+0.63 pp** | −5.99 pp | −5.76 pp | — |

## Reading

Three observations confirm the main-paper claim across this second 10M dataset:

1. **IVF-TQ recall *improves* as the database grows**, consistent with the
   rate-distortion bound (Theorem 1, main paper): per-vector compression
   error is bounded by $(b, d, \delta)$ alone, and partition coverage —
   the only data-dependent layer — grows with $N$.

2. **Re-training does not fix the IVF-PQ degradation**: 354 s of cumulative
   retraining recovers only +0.23 pp at 10M (66.67% retrain vs. 66.44%
   stale). At every intermediate batch the retrain and stale variants are
   within ±0.5 pp. The codebook is not the binding constraint; the bit
   budget is.

3. **The SIFT-128 effect is sharper than Deep-96**: the recall gap widens
   by ~6.6 pp on SIFT-10M (11.5 → 18.1) versus ~2.5 pp on Deep-10M (5.4 → 7.9).
   We attribute this to dimensionality: SIFT's $d = 128$ produces denser
   per-coordinate distributions, and the additional sub-Gaussian tail mass
   that PQ's initial-sample codebook misses at moderate compression compounds
   more aggressively as the database grows.

For the multi-seed (3-seed paired-$t$) summary across sub-/bit-/super-matched
PQ memory regimes, see main paper **Table 6**.
