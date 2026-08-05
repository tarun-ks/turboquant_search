[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![PyPI](https://img.shields.io/pypi/v/turboquant-search)](https://pypi.org/project/turboquant-search/)

# TurboQuant Search

**One-line:** A codebook-free IVF index for vector search, with a benchmark suite and mechanism analysis for recall degradation under corpus growth.

- **Why:** No codebook re-training when your corpus grows. Across nine controlled cells (three 10M datasets × three PQ memory regimes × three seeds), per-batch PQ codebook retraining is statistically indistinguishable from no retraining in 8 of 9 cells while costing 667–1328s of cumulative compute per run. At sub-matched memory, IVF-TQ is more stable than IVF-PQ across all three datasets with $\Delta \in [-0.80, +0.56]$pp.
- **Install:** `pip install turboquant-search[all]`
- **Run:** `cd experiments && python run_benchmarks.py`

---

An IVF index whose residual compression layer is **codebook-free**: a fixed random rotation followed by precomputed Lloyd–Max scalar quantization that depends only on bit width and dimension. Building on [TurboQuant](https://arxiv.org/abs/2504.19874) (Zandieh et al., ICLR 2026). The IVF coarse partition is still trained by k-means; only the residual is data-independent.

**Why it matters.** Production ANN compression methods (PQ, OPQ, ScaNN, RaBitQ) fit a codebook to an initial training sample and reuse it as the database grows. The codebook silently goes stale. Across 3 seeds (42, 123, 7777) on streaming 10M ingestion at sub-matched memory ($\sim$0.75–0.78× IVF-TQ):

- Deep-10M: IVF-PQ drops $-3.23 \pm 0.49$pp; IVF-TQ holds at $-0.80 \pm 0.25$pp.
- SIFT-10M: IVF-PQ drops $-5.80 \pm 0.55$pp; IVF-TQ *improves* by $+0.56 \pm 0.10$pp.
- T2I-10M (200-dim, Yandex Text2Image-1B prefix): IVF-PQ drops $-3.24 \pm 0.28$pp; IVF-TQ holds at $-0.76 \pm 0.41$pp.

Per-batch PQ retraining is statistically indistinguishable from no retraining in 8 of 9 cells across (3 datasets × 3 memory regimes). At bit-matched memory ($\sim$0.95× IVF-TQ), IVF-PQ stabilises on Deep-10M and T2I-10M but **still degrades on SIFT-10M** ($-2.31$pp, $+6.79$pp behind IVF-TQ) — the capacity threshold for PQ streaming stability is dataset-dependent. **IVF-TQ's residual quantizer has no data-dependent codebook** — the $(b, d)$ configuration is fixed at build time and independent of corpus size.

## Quick benchmarks

Run the headline 1M-scale comparisons (flat TQ vs IVF-TQ vs FAISS PQ/OPQ/HNSW baselines) — about 15–20 minutes on a recent laptop, downloading datasets on first run:

```bash
pip install -e .[all]
cd experiments
python run_benchmarks.py
```

Output: `experiments/benchmark_results.json`.

Per-script index of every benchmark in [`experiments/README.md`](experiments/README.md). Example runs:

```bash
python experiments/streaming_with_retrain.py    # SIFT-1M streaming with periodic re-train
python experiments/streaming_10m.py             # Deep-10M streaming (~45 min, 10 GB cache)
python experiments/adaptive_ivftq_shift.py      # Adaptive partition refresh under worst-case shift
python experiments/compare_stage2.py            # Sign-bit refinement vs QJL at flat-TQ scale
```

### Reproducing the 9-cell streaming matrix (3 datasets × 3 PQ memory regimes)

All streaming results span 3 seeds (42, 123, 7777) with paired-$t$ CIs. See [`experiments/MULTISEED.md`](experiments/MULTISEED.md) for the full protocol. Wall-clock: ~3 h on a 16 GB M-series MacBook for the 1M-scale cells; T2I-10M needs more memory.

```bash
# Deep-10M (three PQ memory regimes)
python experiments/streaming_multiseed.py --experiment deep10m           --seeds 42 123 7777
python experiments/streaming_multiseed.py --experiment deep10m_pqmatched --seeds 42 123 7777
python experiments/streaming_multiseed.py --experiment deep10m_pqhigh    --seeds 42 123 7777

# SIFT-10M (three PQ memory regimes)
python experiments/streaming_multiseed.py --experiment sift10m           --seeds 42 123 7777
python experiments/streaming_multiseed.py --experiment sift10m_pqmatched --seeds 42 123 7777
python experiments/streaming_multiseed.py --experiment sift10m_pqhigh    --seeds 42 123 7777

# T2I-10M (three PQ memory regimes)
python experiments/streaming_multiseed.py --experiment t2i10m            --seeds 42 123 7777
python experiments/streaming_multiseed.py --experiment t2i10m_pqmatched  --seeds 42 123 7777
python experiments/streaming_multiseed.py --experiment t2i10m_pqhigh     --seeds 42 123 7777

# SIFT-1M three-condition control (original / shuffled-i.i.d. / mean-shift)
python experiments/streaming_multiseed.py --experiment sift1m            --seeds 42 123 7777

# Encoder-swap experiments (3-seed paired-t)
python experiments/embed_swap_multiseed.py --swap minilm  --seeds 42 123 7777   # L6 → L12 (gentle, cos 0.51)
python experiments/embed_swap_multiseed.py --swap bge     --seeds 42 123 7777   # L6 → BGE  (harsh, cos 0.24)

# Regenerate the multi-seed summary
python experiments/tables_from_multiseed.py
```

Outputs land under `experiments/results/`: per-experiment CSVs and a human-readable `multiseed_summary.txt`. Per-batch trajectory `.md` files for each dataset and regime are in [`paper_supplementary/`](paper_supplementary/) (supplementary analysis and full trajectory tables).

### Reproducing the corpus-growth mechanism results

These scripts isolate *why* IVF-PQ degrades as N grows — margin shrinkage overtaking fixed quantization error, not codebook staleness. Run in order; each step produces the CSV consumed by the next.

```bash
# Step 1: Oracle baseline (confirm staleness explanation is wrong)
# Each script: ~40 min for SIFT/Deep, ~70 min for T2I (Apple M-series, 16 GB)
# Output: experiments/results/streaming_oracle_{dataset}.csv
python experiments/streaming_oracle.py --dataset sift10m
python experiments/streaming_oracle.py --dataset deep10m
python experiments/streaming_oracle.py --dataset t2i10m

# Step 2: Uncompressed IVF (isolate coverage vs compression effects)
# Output: experiments/results/streaming_uncompressed_{dataset}.csv
python experiments/streaming_uncompressed.py --dataset sift10m
python experiments/streaming_uncompressed.py --dataset deep10m
python experiments/streaming_uncompressed.py --dataset t2i10m

# Step 3: Margin vs distortion at the decision boundary
# Output: experiments/results/rank_margin_{dataset}.csv
python experiments/rank_margin.py --dataset sift10m
python experiments/rank_margin.py --dataset deep10m
python experiments/rank_margin.py --dataset t2i10m

# Step 4: Per-query causal decomposition (3 seeds, ~5–8 min/seed)
# Output: experiments/results/perquery_{dataset}.csv  (10K rows × 3 seeds)
python experiments/perquery_analysis.py --dataset sift10m
python experiments/perquery_analysis.py --dataset deep10m
python experiments/perquery_analysis.py --dataset t2i10m

# Step 5: Bootstrap CIs on margin/err ratio — confirms ordering is real
# Output: experiments/results/bootstrap_ci_summary.csv
python experiments/bootstrap_ci.py
```

**Expected key numbers** (3 seeds, 10K queries each):

| Dataset | Coverage losses | err > margin (of ranking losses) | ratio CI (95%) |
|---|---|---|---|
| SIFT-10M | 0% | 94.5% ± 0.5% | [0.136, 0.147] |
| T2I-10M  | 0% | 89.8% ± 1.7% | [0.235, 0.252] |
| Deep-10M | 0% | 87.7% ± 0.6% | [0.265, 0.285] |

All three dataset CIs are fully separated — ratio strictly orders degradation severity. The SIFT–Deep crossing-fraction gap (94.5% vs 87.7%) is predicted by the ratio ordering (lower ratio → more boundary-adjacent neighbors → higher crossing fraction).

Memory: ~16 GB RAM needed for T2I-10M (200-d, 10M vectors). SIFT and Deep run on 8 GB.

## When to use this library

- **Streaming corpora.** Codebook-free residual compression eliminates the silent recall drift PQ/OPQ/ScaNN suffer when the database grows past the codebook-training sample. New vectors are encoded at full quality with no retraining.
- **Encoder upgrades.** Under embedding-model swap on 1M MS MARCO passages (3-seed paired-$t$, $p < 0.001$): IVF-TQ frozen exceeds an actively-retrained IVF-PQ baseline by $+13.27 \pm 0.77$pp on the gentle swap (L6 → L12, cosine 0.51) and by $+14.72 \pm 0.43$pp on the harsh swap (L6 → BGE-small, cosine 0.24).
- **Prototyping ANN compression.** No codebook configuration, no per-dataset tuning — set `bits=4` and the compression ratio is deterministic at `(b+1)/32` of float32 storage.
- **Readable reference.** Pure NumPy core + optional C++ inner loop (NEON-accelerated on Apple Silicon).

## Install

```bash
pip install turboquant-search            # core + CLI
pip install turboquant-search[all]       # + FAISS baselines + dataset loaders
```

## Quick Start

The flat `TurboQuantSearchIndex` below is the easiest starting point and works well for ≤100K vectors. **For larger corpora use `IVFTurboQuantIndex`** (next block) — the IVF version is the primary index in this library; flat is provided for prototyping and as a building block.

> **Note on the random-data example below.** Random Gaussian vectors are *pessimal* for any IVF-style index because they have no cluster structure. The example below works (it builds, indexes, searches) but produces a lower recall than you would see on real embeddings (SIFT, GloVe, BERT, OpenAI, Cohere, etc.). For realistic numbers, see [`experiments/run_benchmarks.py`](experiments/run_benchmarks.py).

```python
from turboquant_search import TurboQuantSearchIndex
import numpy as np

# Your embeddings (e.g., from sentence-transformers, OpenAI, etc.)
# Here we simulate 10K document embeddings of dimension 128
document_embeddings = np.random.randn(10000, 128).astype(np.float32)

# Create a compressed index — no training needed
index = TurboQuantSearchIndex(dim=128, bits=3)
index.add(document_embeddings)

# Search with a query embedding
query_embedding = np.random.randn(1, 128).astype(np.float32)
scores, top_k_indices = index.search(query_embedding, k=10)

print(f"Top 10 results: {top_k_indices[0]}")
print(f"Compression: {index.stats()['compression_ratio']}")
# -> '7.5x' (3-bit + sign-bit refinement)
```

Works with any embedding model — just pass in your vectors. No codebook training, no dataset-specific tuning.

### IVF-TQ: Sub-linear search at scale

```python
from turboquant_search import IVFTurboQuantIndex
import numpy as np

# 1M document embeddings
documents = np.random.randn(1_000_000, 128).astype(np.float32)

# Train coarse quantizer (k-means only — no codebook training)
index = IVFTurboQuantIndex(dim=128, nlist=1000, bits=4, nprobe=10)
index.train(documents)
index.add(documents)

# Search — scans ~1% of data, not all 1M vectors
scores, top_k = index.search(query, k=10)

# Add new vectors instantly — no retraining needed
new_vector = np.random.randn(128).astype(np.float32)
index.add_single(new_vector)  # compressed and indexed in microseconds
```

## Headline Results (1M-scale, 10K queries, seed=42)

Recall@10 at matched memory. Full numbers and all 10M-scale streaming results: see [`experiments/`](experiments/) and run `python experiments/run_benchmarks.py`.

### SIFT-1M (dim=128)

Two memory columns: **Packed** is the bits-perfect theoretical minimum (same
basis used by all published benchmarks). **Resident** is actual allocation.
FAISS methods use C++ byte-aligned storage, so their packed ≈ resident.
IVF-TQ uses numpy uint8 per coordinate, so resident is ~4× packed
(compression-only) or ~8× packed (with raw vectors for reranking).
See `experiments/measure_memory.py` for full measurements.

| Method | Recall@10 | Packed | Resident | Training |
|---|---|---|---|---|
| FAISS IVF-PQ m=64, n_p=80 | 73.2% | 62 MB | ~62 MB¹ | PQ codebook |
| FAISS OPQ+IVF-PQ m=128, n_p=80 | 97.0% | 123 MB | ~123 MB¹ | OPQ + PQ |
| FAISS HNSW M=32, ef=64 | 98.2% | 732 MB | ~732 MB¹ | None |
| ScaNN AH+tree, $L_s$=50 | 96.2% | 62 MB | ~62 MB¹ | ScaNN AH |
| **IVF-TQ 6-bit, n_p=20 (ours)** | **93.2%** | **111 MB** | **~388 MB²** | **k-means only** |
| **IVF-TQ 6-bit, n_p=40 (ours)** | **96.1%** | **111 MB** | **~388 MB²** | **k-means only** |

¹ FAISS and ScaNN store byte-aligned C++ arrays; packed ≈ resident.
² Compression-only (`store_raw_vectors=False`): numpy stores each b-bit index
  as a full uint8 byte — indices(128MB) + sign\_bits(128MB) + norms(4MB) +
  codes(128MB) + centroids(0.5MB). With raw vectors for reranking: ~900 MB.
  Use `IVFTurboQuantIndex(store_raw_vectors=False)` to avoid the raw-vector cost.

### Streaming on 10M scale (3 seeds, sub-matched PQ memory)

IVF-PQ's codebook is trained on the initial sample. As new vectors arrive, PQ compression degrades; periodic retraining is statistically indistinguishable from no retraining. IVF-TQ's compression is data-independent — recall holds or even *improves* as partition coverage grows.

| Dataset | IVF-TQ Δ (1M → 10M) | IVF-PQ (stale) Δ | IVF-PQ (retrain) Δ | retrain − stale |
|---|---|---|---|---|
| Deep-10M | $-0.80 \pm 0.25$pp | $-3.23 \pm 0.49$pp | $-3.17 \pm 0.15$pp | $+0.06 \pm 0.44$pp ($p{=}0.60$) |
| SIFT-10M | $+0.56 \pm 0.10$pp | $-5.80 \pm 0.55$pp | $-5.64 \pm 0.66$pp | $+0.17 \pm 0.50$pp ($p{=}0.29$) |
| T2I-10M | $-0.76 \pm 0.41$pp | $-3.24 \pm 0.28$pp | $-3.34 \pm 0.49$pp | $-0.10 \pm 0.30$pp ($p{=}0.29$) |

The same pattern persists at bit-matched and super-matched memory: across all 9 cells, retrain is indistinguishable from no-retrain in 8 of 9 (paired-$t$ $p \geq 0.14$); the 9th cell (SIFT-10M bit-matched) is $-0.08$pp opposite-sign, statistically significant but practically negligible. See per-batch trajectories under [`paper_supplementary/`](paper_supplementary/) (supplementary analysis).

## Supported Embedding Dimensions

Works with any embedding model. Common configurations:

| Model | Dim | Provider |
|-------|-----|----------|
| all-MiniLM-L6-v2 | 384 | sentence-transformers |
| bge-base-en-v1.5 | 768 | BAAI |
| bge-large-en-v1.5 | 1024 | BAAI |
| text-embedding-3-small | 1536 | OpenAI |
| text-embedding-3-large | 3072 | OpenAI |
| embed-v4 | 1024 | Cohere |
| voyage-3 | 1024 | Voyage AI |
| gemini-embedding-001 | 3072 | Google |
| nomic-embed-text-v1.5 | 768 | Nomic |

Just pass your vectors in — TurboQuant handles any dimension with the same compression ratio.

## Vector Database Integration

TurboQuant Search is a **standalone compressed index** — it is not a drop-in plugin for Pinecone, Weaviate, Qdrant, or Milvus. In its current form, it's best used for:

- **Standalone search** on small-to-medium datasets (up to ~1M vectors) where you want compression without a database
- **Prototyping** compression tradeoffs before committing to a production vector DB
- **Understanding** the TurboQuant algorithm — the code is readable NumPy, not optimized C++

To use with a vector DB, you would compress with TurboQuant, then store the compressed representation in the DB's raw storage layer — but this requires custom integration per DB. Most production vector DBs already have built-in PQ/SQ compression options that are more tightly integrated with their indexing.

## Limitations

- **Research-prototype kernel.** On SIFT-1M with the NEON-accelerated C++ kernel, IVF-TQ achieves ~13K QPS at $n_p$=20 (confirmed, seed=42). FAISS IVF-PQ achieves ~6.6K QPS at the same nprobe. Absolute throughput depends on nprobe; at lower nprobe FAISS benefits more from its FastScan-style SIMD-LUT kernels. Closing the gap needs a FastScan-style SIMD-LUT kernel; on the v2 roadmap.
- **IVF amplification matters at scale.** Flat-TQ recall benefits from sign-bit refinement (~+11.4pp over QJL across 6 flat-TQ cells), but the IVF-TQ advantage over IVF-PQ comes mainly from IVF amplification, not the residual quantizer alone. Under IVF, sign-bit vs Extended-RaBitQ-equivalent are within statistical noise.
- **Synthetic / uniform data.** IVF partitioning hurts when clusters are uniform. The IVF-TQ advantage appears on real data with natural cluster structure (SIFT, Deep, GloVe, T2I).
- **Stage 2 uses sign-bit refinement, not QJL.** Sign-bit is the choice for nearest-neighbour ranking; QJL is the choice for unbiased KV-cache attention scoring. See "Stage 2 design choice" below.

## How It Works

**Stage 1: Rotation + Lloyd-Max Quantization** — Multiply by a random orthogonal matrix (QR of Gaussian). Each coordinate becomes ~N(0, 1/d). Apply the optimal scalar quantizer for this distribution (b bits per coordinate). Store quantization indices + vector norm. The rotation is a dense d×d matrix multiply — O(d²) per vector. At d=128 this is fast; at d=768+ it becomes a real throughput bottleneck. A structured fast transform (Hadamard-style) would reduce this to O(d log d) but is not currently implemented.

**Stage 2: Sign-Bit Refinement** — Split each quantization bin at its centroid. Store 1 extra bit (above/below) per coordinate. This doubles effective resolution from 2^b to 2^(b+1) levels using the conditional expectation of each half-bin.

**Asymmetric Search** — Queries are rotated but not quantized. Inner products are preserved since the rotation is orthogonal: `<Pi*q, Pi*x> = <q, x>`.

## Stage 2 design choice

Stage 1 (rotation + Lloyd-Max) follows TurboQuant. Stage 2 diverges: TurboQuant
uses QJL (a 1-bit Gaussian random projection) for *unbiased* inner-product
estimation, which is the right choice for KV-cache attention scoring. This
library replaces QJL with a half-bin sign-bit refinement:

| | TurboQuant (QJL) | This library (Sign-Bit) |
|---|---|---|
| **Goal** | Unbiased inner product estimation | Low-variance ranking |
| **Method** | Gaussian random projection + sign bit | Split each Lloyd–Max bin at its centroid; store half-bin indicator |
| **Best for** | KV-cache scoring (needs unbiased estimates) | Nearest-neighbour search (needs correct ranking, not exact values) |
| **Empirical effect** | Baseline | **+11.4pp avg over QJL** on flat-TQ million-scale (6 cells across SIFT-1M, Deep-1M, GloVe-1M at 3 and 4 bits) |

For nearest-neighbour search the only thing that matters is rank order. Sign-bit
refinement accepts a small bias to cut variance, which is what reduces ranking
inversions and lifts recall. In an IVF wrapper the gain disappears because
IVF amplification already absorbs the residual variance; we ship sign-bit by
default because there is no setting in which it does worse than QJL.

## Running Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

## CLI

```bash
tqs benchmark --dataset synthetic     # Run benchmarks, print results table
tqs index --input vectors.npy --bits 3  # Index custom numpy embeddings (IVF-TQ by default; --flat for prototype)
tqs search --index tq.tqindex --query query.npy --top 10  # Search an existing index
```

## Project Structure

```
turboquant_search/
  core.py                # TurboQuant + IVF-TQ (rotation, Lloyd-Max, sign-bit, IVF partitioning)
  adaptive.py            # Adaptive IVF-TQ (partition-only refresh)
  faiss_baselines.py     # FAISS wrappers (Flat, PQ, IVF-PQ, OPQ, HNSW)
  benchmarks.py          # Benchmark runner
  datasets.py            # Dataset loaders (SIFT, Deep, GloVe, T2I, MS MARCO)
  cli.py                 # CLI entry point (tqs command)
csrc/                    # NEON-accelerated C++ inner loop (built by setup.py)
experiments/             # Reproducible benchmark scripts (see experiments/README.md)
paper_supplementary/     # Full trajectory tables and supplementary analysis
tests/                   # Unit tests
```

## Status

### Completed
* ✅ IVF-TQ hybrid index with $(b, d)$-only residual quantization
* ✅ Million-scale benchmarks on SIFT-1M, Deep-1M, GloVe-1M
* ✅ 10M-scale streaming benchmarks (Deep-10M, SIFT-10M, T2I-10M)
* ✅ Adaptive coarse-partition refresh
* ✅ Encoder-swap robustness (3-seed paired-$t$, MS MARCO)
* ✅ NEON-accelerated C++ inner loop
* ✅ Corpus-growth degradation mechanism: per-miss causal decomposition (3 datasets × 3 seeds); 87–94% of losses have err > margin, 0% coverage losses across all datasets
* ✅ Bootstrap CIs on margin/err ratio: all three datasets (SIFT/T2I/Deep) fully separated — ratio strictly orders degradation severity

### Future work
* FastScan-style SIMD LUTs (int8) to further reduce QPS gap with FAISS
* 100M+-scale validation
* GPU acceleration (CuPy / CUDA)
* Hadamard-style fast rotation to reduce per-vector cost from O(d²) to O(d log d) at high dimensions (d=768+)
* Extended graph-baseline sweep (DiskANN, SPANN, NSG beyond the HNSW comparison)

Contributions welcome.

## Acknowledgements

This work builds directly on Google Research's [TurboQuant](https://arxiv.org/abs/2504.19874) (Zandieh et al., ICLR 2026). The IVF-TQ contribution is the data-independent residual quantizer wrapped in an inverted-file index, the IVF-amplification analysis, and the streaming-stability evidence — not the underlying TurboQuant rotation + Lloyd–Max design, which is theirs.

## License

Apache 2.0

---

*Independent implementation inspired by the TurboQuant paper. Not affiliated with Google Research.*
