# Experiments

Each script in this directory runs one benchmark or ablation and writes a
self-contained `*_results.json`. The scripts are independent — you can run
any one without running the others first (datasets are auto-downloaded and
cached under `experiments/cache/`).

Files no longer used by the current benchmark suite live under `archive/`.

## Top-level runner

| Script | Purpose | Approx runtime |
|---|---|---|
| `run_benchmarks.py` | Runs the headline 1M-scale comparisons (sign-bit refinement, flat TQ vs IVF-TQ, FAISS PQ/OPQ/HNSW baselines, ScaNN where available). Output: `benchmark_results.json`. | 15–20 min |

## Sign-bit refinement (Stage 2)

| Script | Purpose |
|---|---|
| `compare_stage2.py` | Sign-bit vs QJL at 10K-scale (Synthetic, SIFT-128, GloVe-100, $b \in \{2,3,4\}$). |
| `compare_stage2_1m_multidataset.py` | Sign-bit vs QJL at 1M-scale (SIFT-1M, Deep-1M, GloVe-1M). |
| `qjl_index.py` | QJL Stage-2 baseline implementation used by the comparisons above. |
| `extended_rabitq_baseline.py` | Extended-RaBitQ-equivalent baseline for the IVF-TQ regime comparison (9-cell sweep). |

## Streaming and dynamic ingestion

| Script | Purpose |
|---|---|
| `streaming_ingestion.py` | SIFT-1M streaming under three ingestion conditions (original / shuffled-i.i.d. / mean-shift). |
| `streaming_with_retrain.py` | SIFT-1M streaming with periodic PQ codebook re-training (compute vs. recovery curve). |
| `streaming_10m.py` | Deep-10M streaming, 1M trained + 9 batches of 1M. |
| `streaming_sift10m.py` | SIFT-10M streaming, same protocol as Deep-10M. |
| `streaming_capacity_vs_bias.py` | Disambiguates codebook bias vs codebook capacity on SIFT-1M (three IVF-PQ variants evaluated on the same 1M database). |
| `embed_swap.py` | MS MARCO encoder-swap experiment (L6 → L12 and L6 → BGE-small). |

## Adaptive coarse-partition refresh

| Script | Purpose |
|---|---|
| `adaptive_ivftq_shift.py` | Worst-case rotation-shift recovery on Deep-1M (Adaptive IVF-TQ vs IVF-PQ retrain). |
| `adaptive_followups.py` | Refresh-frequency Pareto sweep. |

## Million- and 10M-scale comparison

| Script | Purpose |
|---|---|
| `scale_10m.py` | IVF-PQ + IVF-TQ at Deep-10M scale. |
| `run_hnsw_opq.py` | FAISS HNSW and OPQ baselines at 1M/10M. |
| `scann_baseline.py` | ScaNN runner (Linux-only; see `scann_colab.ipynb` for the Colab path). |
| `rabitq_baseline.py` | 1-bit RaBitQ baseline. |

## Bit-importance and cascade

| Script | Purpose |
|---|---|
| `ivf_rvq_tq.py` | IVF-RVQ-TQ index class (multi-bit Stage 2). Shared base used by the cascade scripts. |
| `rvq_tq_explore.py` | Flat RVQ-TQ exploration with `per_bin_lloyd_max_subcentroids`. |
| `rvq_tq_verify.py` | High-resolution Lloyd–Max codebook used by `ivf_rvq_tq.py`. |
| `sphere_decode.py` | Bit-importance ablation on TQ + sphere-decoding probe. |
| `cascade_pq_v2.py` | Properly-tuned IVF-PQ for the bit-importance comparison ($m{=}64$ mid-recall, $m{=}128$ high-recall). |
| `cascade_search.py` | Two-pass cascade-search implementation. |
| `cascade_robustness.py` / `cascade_robustness_cpp.py` | Multi-seed × nprobe × bit-budget cascade verification (Python + C++ refs). |
| `cascade_sift1m_seeds.py` | SIFT-1M multi-seed cascade verification. |
| `cascade_verify.py` | Cross-dataset cascade verification at multiple bit budgets. |

## Downstream NDCG

| Script | Purpose |
|---|---|
| `rag_ndcg_colab.ipynb` | NDCG@10 on BeIR/MS MARCO dev (Colab variant). |

## Negative results (explored alternatives)

| Script | Purpose |
|---|---|
| `rh_lsh_eval.py` | Random-hyperplane LSH coarse partition (data-independent; dominated by k-means). |
| `fa_highdim.py` | Frequency-adaptive bit allocation at high dim (failed extension). |

## Caches and outputs

`experiments/cache/` holds downloaded datasets (SIFT, Deep, GloVe, MS MARCO).
Total size around 10 GB. Excluded from git.

Each script writes its `*_results.json` next to itself; these are small
(< 1 MB each) and are committed so reviewers can inspect numbers without
re-running.

## Archive

`experiments/archive/` keeps exploratory and superseded scripts for
historical traceability. They are not part of the current benchmark suite.
