# Million-Scale Comparison: Deep-1M block

Referenced from main paper §5 (Million-Scale Comparison, Table 9).
The Deep-1M block was moved to this supplementary file to keep the main
paper within the PVLDB 12-page limit; SIFT-1M and Deep-10M remain in
the paper.

## Deep-1M (1M vectors, dim = 96)

10K queries, deterministic seed = 42, FAISS 1.13.2.

| Method | R@10 | QPS | Memory | Codebook training? |
|---|---|---|---|---|
| FAISS IVF-PQ m=48, $n_p$=80 | 85.1% | 8.8K | 46 MB | PQ |
| FAISS OPQ+IVF-PQ m=96, $n_p$=20 | 94.5% | 22K | 92 MB | OPQ + PQ |
| FAISS OPQ+IVF-PQ m=96, $n_p$=80 | 97.3% | 6.3K | 92 MB | OPQ + PQ |
| FAISS HNSW M=32, $ef_s$=64 | 97.6% | 93K | 610 MB | None |
| ScaNN AH+tree, $L_s$=50 | 96.9% | 7.6K | 47 MB† | ScaNN AH |
| ScaNN AH+tree, $L_s$=100 | 98.8% | 4.8K | 47 MB† | ScaNN AH |
| Ext. RaBitQ $B$=5, $n_p$=20§ | 89.7% | 2.7K | 61 MB | None |
| Ext. RaBitQ $B$=6, $n_p$=20§ | 92.5% | 2.8K | 73 MB | None |
| IVF-TQ 4-bit, $n_p$=20 (ours) | 89.9% | 14K | 61 MB | None |
| IVF-TQ 5-bit, $n_p$=20 (ours) | 92.8% | 13K | 73 MB | None |
| **IVF-TQ 6-bit, $n_p$=20 (ours)** | **94.2%** | **12K** | **84 MB** | **None** |
| **IVF-TQ 6-bit, $n_p$=40 (ours)** | **96.4%** | **6.9K** | **84 MB** | **None** |

†ScaNN's listed memory is the compressed AH+tree footprint; ScaNN
additionally stores raw vectors for reorder (412 MB total on Deep-1M).

§Extended RaBitQ rows are our Python reimplementation, matching the
official implementation's per-coordinate quantizer up to the $O(1/d)$
Gaussian-marginal correction proven in Theorem 1.

## Reading

The Deep-1M results mirror SIFT-1M (in the main paper): ScaNN at $L_s = 50$
reaches 96.9% R@10 at 47 MB — roughly half the memory IVF-TQ requires
at comparable recall (96.4% at 84 MB). At matched memory, IVF-TQ 6-bit
is within ~1pp of OPQ+IVF-PQ but is dominated by ScaNN by ~1pp.
HNSW dominates at high recall ($\geq 99\%$) but at $\sim 7\times$ the
memory footprint.
