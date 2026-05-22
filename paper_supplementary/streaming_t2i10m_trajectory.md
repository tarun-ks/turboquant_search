# Streaming results at 10M scale: T2I-10M (single-seed per-batch trajectory)

Referenced from main paper §4.2. The 3-seed headline values are in
**Table 7 of the main paper**; this file reports a representative
single-seed per-batch view for context.

## Setup

- Dataset: T2I-10M (first 10M vectors of Yandex Text2Image-1B, 200-dim CLIP-style).
- Queries: 100K Text2Image query set, FAISS `IndexFlatIP` ground truth on normalised vectors.
- Training: first 1M vectors.
- Streaming: 9 batches of 1M each.
- IVF-TQ: $b = 4 +$ sign-bit (effective 5 bits/coord), 1032 bits/vec total
  ($4 \cdot 200 + 200 + 32 = 1032$).
- IVF-PQ regimes: sub-matched ($m_{\text{PQ}} = 100, b = 8$, 800 bits/vec,
  ≈0.78× IVF-TQ memory); bit-matched ($m_{\text{PQ}} = 100, b = 10$,
  1000 bits/vec, ≈0.97× IVF-TQ memory); super-matched ($m_{\text{PQ}} = 200,
  b = 8$, 1600 bits/vec, ≈1.55× IVF-TQ memory).
- Seed: 42 (representative; 3-seed summary in main Table 7).

## Per-batch trajectory: sub-matched regime ($m_{\text{PQ}} = 100$, $b = 8$)

| Vectors indexed | IVF-TQ | IVF-PQ (stale) | IVF-PQ (retrain/1M) | Retrain cum. |
|---|---|---|---|---|
| 1M (trained) | 81.61% | 76.63% | 76.63% |   0 s |
| 2M | 81.58% | 75.81% | 75.42% |  27 s |
| 10M | 80.86% | 73.55% | 73.32% | 520 s |

3-seed Δ (1M→10M, mean ± 95% CI):
IVF-TQ −0.76 ± 0.41 pp; IVF-PQ stale −3.24 ± 0.28 pp; IVF-PQ retrain −3.34 ± 0.49 pp.
IVF-TQ vs. PQ stale at 10M: **+7.41 ± 0.12 pp** ($p < 0.001$).

## Per-batch trajectory: bit-matched regime ($m_{\text{PQ}} = 100$, $b = 10$)

3-seed Δ (1M→10M, mean ± 95% CI):
IVF-TQ −0.76 ± 0.41 pp; IVF-PQ stale −0.89 ± 0.32 pp; IVF-PQ retrain −0.98 ± 0.23 pp.
IVF-TQ vs. PQ stale at 10M: **+0.29 ± 0.24 pp** ($p < 0.05$).
Retrain compute (3-seed mean ± 95% CI): **1327.5 ± 43.0 s**.

## Per-batch trajectory: super-matched regime ($m_{\text{PQ}} = 200$, $b = 8$)

3-seed Δ (1M→10M, mean ± 95% CI):
IVF-TQ −0.76 ± 0.41 pp; IVF-PQ stale +1.07 ± 0.45 pp; IVF-PQ retrain +1.05 ± 0.34 pp.
IVF-TQ vs. PQ stale at 10M: **−5.12 ± 0.45 pp** (rate-distortion regime).

## Reading

- T2I-10M joins **Deep-10M as stable at bit-matched memory** (Δ within ±1 pp);
  SIFT-10M still degrades at the same regime (−2.31 pp), confirming that the
  capacity threshold for PQ streaming stability is dataset-dependent.
- Per-batch retraining costs **1327.5 ± 43.0 s of cumulative compute** at the
  bit-matched regime — substantially more than Deep-10M (667 s) or SIFT-10M
  (821 s) at the same regime — with no recall benefit (retrain − stale paired
  Δ = −0.09 ± 0.28 pp, NS).
- IVF-TQ's trajectory is the same across all three regimes (single
  configuration, dataset-independent bit budget). For the 3-seed summary,
  see main paper **Table 7**.

## Data files

- Source CSVs: `experiments/results/streaming_t2i10m_*_multiseed.csv` (3 seeds × 30 rows each).
- Reproduction: `python experiments/streaming_multiseed.py --experiment t2i10m{,_pqmatched,_pqhigh} --seeds 42 123 7777`.
