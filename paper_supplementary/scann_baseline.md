# ScaNN baseline (Linux-only): setup and full $L_s$ sweep

Referenced from main paper §5 (Million-Scale Comparison) and §7 (Limitations).
ScaNN dominates IVF-TQ on the static-recall axis at fixed compressed memory;
it shares the codebook-staleness limitation with PQ under streaming updates.

## Setup

- Google's ScaNN ([guo2020scann]) ships only Linux wheels via `pip install scann`.
- Runner: `experiments/scann_baseline.py` on Linux/Colab.
- Notebook: `experiments/scann_colab.ipynb`.
- Results: `experiments/scann_results.json`.

## Configuration

- Scoring: AsymmetricHashing (AH).
- `anisotropic_quantization_threshold` = 0.2.
- `dimensions_per_block` = 2.
- Reordering on top-100 candidates.
- Tree: $L = 2000$ leaves.
- Sweep: `num_leaves_to_search` ∈ {20, 50, 100, 200, 400}.
- Queries: 10K from the standard ann-benchmarks split.
- Ground truth: FAISS `IndexFlatIP` on normalised vectors.

## Full sweep at million-scale ($L = 2000$ leaves, reorder-100)

Memory column: compressed AH+tree footprint. ScaNN additionally stores
raw vectors for reorder (550 MB total on SIFT-1M, 412 MB on Deep-1M),
matching the HNSW order of magnitude.

| Dataset | $L_s$ | R@10 | QPS | Compressed | with reorder |
|---|---|---|---|---|---|
| SIFT-1M | 20  | 88.4% | 6.9K | 62.1 MB | 550 MB |
| SIFT-1M | 50  | 96.2% | 5.9K | 62.1 MB | 550 MB |
| SIFT-1M | 100 | 98.6% | 3.2K | 62.1 MB | 550 MB |
| SIFT-1M | 200 | 99.3% | 2.1K | 62.1 MB | 550 MB |
| SIFT-1M | 400 | 99.4% | 1.2K | 62.1 MB | 550 MB |
| Deep-1M | 20  | 91.1% | 8.9K | 46.6 MB | 413 MB |
| Deep-1M | 50  | 96.9% | 7.6K | 46.6 MB | 413 MB |
| Deep-1M | 100 | 98.8% | 4.8K | 46.6 MB | 413 MB |
| Deep-1M | 200 | 99.5% | 2.8K | 46.6 MB | 413 MB |
| Deep-1M | 400 | 99.7% | 1.5K | 46.6 MB | 413 MB |

## Reading

ScaNN dominates IVF-TQ on the static-recall axis at fixed compressed memory:
at ~96% recall, ScaNN uses roughly half the memory IVF-TQ requires.
We attribute this to ScaNN's anisotropic loss, which weights inner-product
preservation toward the score-magnitude regime that determines top-$k$
ranking, exceeding the per-coordinate-MSE optimality of Lloyd–Max in the
rate-distortion regime relevant to ANN.

ScaNN's anisotropic learned codebook nonetheless shares the codebook-staleness
limitation under streaming updates with PQ and OPQ: it is fitted to the
initial training sample, and the bias from that sample compounds as the
database grows (main paper §4). The IVF-TQ contribution is operational —
data-independence under streaming, not raw static recall — and ScaNN is
included as a strong learned-quantization baseline against which the
operational claim is measured.
