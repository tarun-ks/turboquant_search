# Embedding-model swap on MS MARCO: full per-batch trajectories

Referenced from main paper §4.3. The headline 3-seed paired-$t$ values
are in **Table 8 of the main paper** (gentle and harsh swap summary rows);
this file reports the full per-batch trajectories.

## Setup

- Corpus: 1M unique passages from the BeIR mirror of MS MARCO.
- Training: 200K passages encoded by the *old* encoder.
- Streaming: remaining 800K passages encoded by the *new* encoder, in
  100K-vector batches.
- Queries: 5K disjoint MS MARCO query texts, encoded by the *new* encoder
  (simulating the production state after an encoder upgrade).
- Recall@10 against ground truth recomputed on the cumulative mixed-encoder database.
- Seeds: 42, 123, 7777 (mean ± 95% CI, paired-$t$ on within-seed differences).

## Gentle swap: all-MiniLM-L6-v2 → all-MiniLM-L12-v2 (cos = 0.51 on shared passages)

| Step | IVF-TQ | IVF-PQ stale | IVF-PQ retrain | retrain_t |
|---|---|---|---|---|
| Initial 200K (L6) | 91.76 ± 0.30% | 73.95 ± 0.26% | 73.98 ± 0.27% | 0 s |
| +100K L12 | 86.25 ± 0.37% | 72.64 ± 0.22% | 75.11 ± 0.31% | ~46 s |
| +200K L12 | 87.15 ± 0.40% | 72.66 ± 0.22% | 75.44 ± 0.43% | ~94 s |
| +300K L12 | 87.53 ± 0.32% | 72.66 ± 0.22% | 75.62 ± 0.29% | ~149 s |
| +400K L12 | 87.91 ± 0.36% | 72.48 ± 0.30% | 75.68 ± 0.35% | ~206 s |
| +500K L12 | 88.16 ± 0.44% | 72.43 ± 0.27% | 75.63 ± 0.28% | ~264 s |
| +600K L12 | 88.44 ± 0.34% | 72.43 ± 0.25% | 75.71 ± 0.29% | ~326 s |
| +700K L12 | 88.65 ± 0.25% | 72.34 ± 0.25% | 75.72 ± 0.31% | ~393 s |
| +800K L12 | **88.83 ± 0.31%** | 72.31 ± 0.22% | 75.56 ± 0.33% | **358.0 ± 21.0 s** |

**Paired Δ at +800K, IVF-TQ − IVF-PQ retrain: +13.27 ± 0.77 pp ($p < 0.001$).**

IVF-TQ drops ~5.5 pp on first contact with new-encoder vectors
(91.76 → 86.25 from initial L6-only state to +100K L12) then climbs back
to 88.83 ± 0.31% at +800K as more new-encoder vectors fill cells —
a self-healing dynamic consistent with the data-independence theorem
(Theorem 2, main paper). IVF-PQ retrain gains ~1.6 pp from its initial
baseline over 358 s of cumulative compute but never reaches IVF-TQ recall.

## Harsh swap: all-MiniLM-L6-v2 → BAAI/bge-small-en-v1.5 (cos = 0.24)

| Step | IVF-TQ | IVF-PQ stale | IVF-PQ retrain | retrain_t |
|---|---|---|---|---|
| Initial 200K (L6) | 76.95 ± 0.99% | 51.28 ± 0.32% | 51.32 ± 0.45% | 0 s |
| +100K BGE | 84.31 ± 0.78% | 51.56 ± 0.74% | 71.85 ± 0.40% | ~25 s |
| +200K BGE | 86.07 ± 0.70% | 51.83 ± 0.91% | 72.29 ± 0.53% | ~56 s |
| +300K BGE* | **86.96 ± 0.54%** | 52.05 ± 0.66% | 72.24 ± 0.43% | **91.8 ± 3.2 s** |

*BGE result reported at +300K to budget compute; trend is stable.

**Paired Δ at +300K, IVF-TQ − IVF-PQ retrain: +14.72 ± 0.43 pp ($p < 0.001$).**

The IVF-TQ trajectory *climbs substantially* as new-encoder vectors fill
cells: from 76.95 ± 0.99% at the L6-only trained state to 86.96 ± 0.54%
at the +300K compute-budgeted state, a +10.01 ± 1.04 pp gain across the
streaming portion alone. Each newly-arrived BGE vector is quantized at
full fidelity by the codebook designed only for $(b, d)$, so partition
coverage growth directly improves recall. IVF-PQ stale stays near 52%
because the codebook fitted on the L6 distribution is essentially noise
to BGE queries; PQ retraining recovers ~21 pp simply by adapting the
codebook but never closes the gap to IVF-TQ.

## Data files

- 3-seed source: `experiments/embed_swap_multiseed_minilm.json`, `experiments/embed_swap_multiseed_bge.json`.
- Reproduction: `python experiments/embed_swap_multiseed.py --target L12 --seeds 42 123 7777` (and `--target bge`).
