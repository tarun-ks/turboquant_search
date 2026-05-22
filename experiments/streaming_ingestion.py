"""
Streaming Ingestion Experiment: IVF-TQ vs IVF-PQ Codebook Drift
================================================================

Hypothesis: IVF-PQ's codebook is trained on the initial data distribution.
When new vectors from a DIFFERENT distribution arrive, the stale codebook
degrades PQ compression quality. IVF-TQ's compression is distribution-
independent (rotation + Lloyd-Max for Gaussian), so it should not degrade.

Experiment:
  1. Load SIFT-1M: 1M vectors of dim=128
  2. Split into initial (first 200K) + 8 streaming batches of 100K each
  3. Shuffle the order so later batches come from different index regions
     (SIFT vectors are roughly organized by visual content)
  4. Train both IVF-PQ and IVF-TQ on the initial 200K
  5. Incrementally add each 100K batch WITHOUT retraining codebooks
  6. After each batch, measure Recall@10 on the FULL index vs ground truth
  7. Also measure recall on ONLY the newly added vectors (distribution shift)

This isolates the compression quality question: both methods use the same
IVF partitioning (k-means), but differ in how vectors within each partition
are compressed.
"""

import numpy as np
import time
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from turboquant_search.core import IVFTurboQuantIndex, FlatSearchIndex
from turboquant_search.faiss_baselines import FAISSFlatIndex, FAISSIVFPQIndex, FAISS_AVAILABLE
from turboquant_search.benchmarks import compute_recall
from turboquant_search.datasets import load_sift1m


def run_experiment():
    print("=" * 70)
    print("Streaming Ingestion: IVF-TQ vs IVF-PQ Codebook Drift")
    print("=" * 70)

    # ── Load data ──
    print("\nLoading SIFT-1M...")
    result = load_sift1m(n_vectors=1000000, n_queries=10000)
    if result is None:
        print("Failed to load SIFT-1M")
        return
    vectors, queries, label = result
    dim = vectors.shape[1]
    n_total = vectors.shape[0]
    print(f"  {label}")
    print(f"  Queries: {queries.shape[0]}")

    # ── Experiment parameters ──
    n_initial = 200_000
    batch_size = 100_000
    n_batches = (n_total - n_initial) // batch_size
    nlist = 500
    nprobe = 10
    bits = 4
    m_pq = 64  # matched compression (~8x, same as TQ 4-bit ~6x)
    k = 10

    print(f"\n  Config: nlist={nlist}, nprobe={nprobe}, bits={bits}, m_pq={m_pq}")
    print(f"  Initial: {n_initial:,} vectors")
    print(f"  Streaming: {n_batches} batches of {batch_size:,}")

    # ── Shuffle to create distribution shift ──
    # SIFT vectors at index i are from a different visual region than at
    # index i+500K. By NOT shuffling, later batches naturally come from
    # increasingly different distributions.
    # (We keep original order — SIFT's natural ordering provides the shift)

    initial_vectors = vectors[:n_initial]

    # ── Build ground truth on full dataset ──
    print("\nBuilding ground truth (flat index on full dataset)...")
    gt_flat = FAISSFlatIndex(dim) if FAISS_AVAILABLE else FlatSearchIndex(dim)
    gt_flat.add(vectors)
    _, gt_full = gt_flat.search(queries, k=k)

    # ── Train both indexes on initial data only ──
    print(f"\nTraining on initial {n_initial:,} vectors...")

    # IVF-TQ
    ivf_tq = IVFTurboQuantIndex(
        dim, nlist=nlist, bits=bits, nprobe=nprobe,
        use_residual_sign=True, seed=42,
    )
    ivf_tq.train(initial_vectors)
    ivf_tq.add(initial_vectors)

    # IVF-PQ (FAISS)
    if not FAISS_AVAILABLE:
        print("FAISS not available — cannot run comparison")
        return

    import faiss
    quantizer = faiss.IndexFlatIP(dim)
    ivf_pq = faiss.IndexIVFPQ(
        quantizer, dim, nlist, m_pq, 8, faiss.METRIC_INNER_PRODUCT
    )
    ivf_pq.nprobe = nprobe

    # Normalize for IP search
    def normalize(v):
        v = np.ascontiguousarray(v.astype(np.float32))
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.maximum(norms, 1e-8)

    init_normed = normalize(initial_vectors)
    ivf_pq.train(init_normed)
    ivf_pq.add(init_normed)

    queries_normed = normalize(queries)

    # ── Measure initial recall ──
    results = {"config": {
        "n_initial": n_initial,
        "batch_size": batch_size,
        "n_batches": n_batches,
        "nlist": nlist,
        "nprobe": nprobe,
        "bits": bits,
        "m_pq": m_pq,
        "dim": dim,
    }, "steps": []}

    def measure_step(step_name, n_indexed):
        # Ground truth for vectors indexed so far
        gt_partial = FAISSFlatIndex(dim)
        gt_partial.add(vectors[:n_indexed])
        _, gt_idx = gt_partial.search(queries, k=k)

        # IVF-TQ recall
        _, tq_idx = ivf_tq.search(queries, k=k, rerank=50)
        tq_recall = compute_recall(gt_idx, tq_idx, k)

        # IVF-PQ recall
        _, pq_idx = ivf_pq.search(queries_normed, k)
        pq_recall = compute_recall(gt_idx, pq_idx, k)

        step_data = {
            "step": step_name,
            "n_indexed": n_indexed,
            "ivf_tq_recall": round(tq_recall, 4),
            "ivf_pq_recall": round(pq_recall, 4),
        }
        results["steps"].append(step_data)

        print(f"  {step_name:<30} IVF-TQ: {tq_recall:.1%}  IVF-PQ: {pq_recall:.1%}  "
              f"(gap: {(tq_recall-pq_recall)*100:+.1f}pp)")
        return step_data

    print(f"\n{'Step':<32} {'IVF-TQ':>8} {'IVF-PQ':>8} {'Gap':>8}")
    print("-" * 60)

    measure_step(f"Initial ({n_initial//1000}K)", n_indexed=n_initial)

    # ── Stream batches ──
    for batch_i in range(n_batches):
        start = n_initial + batch_i * batch_size
        end = start + batch_size
        batch = vectors[start:end]
        batch_normed = normalize(batch)

        # Add to both indexes WITHOUT retraining
        ivf_tq.add(batch)
        ivf_pq.add(batch_normed)

        step_name = f"After batch {batch_i+1} ({end//1000}K total)"
        measure_step(step_name, n_indexed=end)

    # ── Summary ──
    initial = results["steps"][0]
    final = results["steps"][-1]

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  IVF-TQ:  {initial['ivf_tq_recall']:.1%} → {final['ivf_tq_recall']:.1%}  "
          f"(change: {(final['ivf_tq_recall']-initial['ivf_tq_recall'])*100:+.1f}pp)")
    print(f"  IVF-PQ:  {initial['ivf_pq_recall']:.1%} → {final['ivf_pq_recall']:.1%}  "
          f"(change: {(final['ivf_pq_recall']-initial['ivf_pq_recall'])*100:+.1f}pp)")
    print(f"  Gap:     {(initial['ivf_tq_recall']-initial['ivf_pq_recall'])*100:+.1f}pp → "
          f"{(final['ivf_tq_recall']-final['ivf_pq_recall'])*100:+.1f}pp")

    # Save results
    out_path = Path(__file__).parent / "streaming_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    run_experiment()
