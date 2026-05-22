"""
Streaming Ingestion with Periodic PQ Retraining Baseline
=========================================================

Extension of streaming_ingestion.py that adds a third method:
IVF-PQ with periodic codebook retraining every N vectors.

This answers the reviewer question: "Why not just retrain PQ periodically?"
Answer: you can, but it's expensive (seconds of downtime per retrain)
and IVF-TQ still wins on recall without any retraining.
"""

import numpy as np
import time
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from turboquant_search.core import IVFTurboQuantIndex, FlatSearchIndex
from turboquant_search.faiss_baselines import FAISSFlatIndex, FAISS_AVAILABLE
from turboquant_search.benchmarks import compute_recall
from turboquant_search.datasets import load_sift1m


def run_experiment():
    print("=" * 75)
    print("Streaming Ingestion: IVF-TQ vs IVF-PQ vs IVF-PQ (periodic retrain)")
    print("=" * 75)

    print("\nLoading SIFT-1M...")
    result = load_sift1m(n_vectors=1000000, n_queries=10000)
    if result is None:
        print("Failed to load SIFT-1M")
        return
    vectors, queries, label = result
    dim = vectors.shape[1]
    n_total = vectors.shape[0]
    print(f"  {label}")

    if not FAISS_AVAILABLE:
        print("FAISS not available")
        return

    import faiss

    n_initial = 200_000
    batch_size = 100_000
    n_batches = (n_total - n_initial) // batch_size
    nlist = 500
    nprobe = 10
    bits = 4
    m_pq = 64
    k = 10

    def normalize(v):
        v = np.ascontiguousarray(v.astype(np.float32))
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.maximum(norms, 1e-8)

    initial_vectors = vectors[:n_initial]
    init_normed = normalize(initial_vectors)
    queries_normed = normalize(queries)

    # ── IVF-TQ: train once, never retrain ──
    ivf_tq = IVFTurboQuantIndex(dim, nlist=nlist, bits=bits, nprobe=nprobe,
                                 use_residual_sign=True, seed=42)
    ivf_tq.train(initial_vectors)
    ivf_tq.add(initial_vectors)

    # ── IVF-PQ (stale): train once, never retrain ──
    def make_ivfpq(train_data):
        quantizer = faiss.IndexFlatIP(dim)
        idx = faiss.IndexIVFPQ(quantizer, dim, nlist, m_pq, 8,
                               faiss.METRIC_INNER_PRODUCT)
        idx.nprobe = nprobe
        idx.train(train_data)
        return idx

    ivf_pq_stale = make_ivfpq(init_normed)
    ivf_pq_stale.add(init_normed)

    # ── IVF-PQ (retrain every 200K): expensive but accurate ──
    ivf_pq_retrain = make_ivfpq(init_normed)
    ivf_pq_retrain.add(init_normed)
    all_normed_so_far = [init_normed]
    retrain_interval = 200_000  # retrain every 200K new vectors
    vectors_since_retrain = 0
    total_retrain_time = 0.0

    results = {"steps": []}

    print(f"\n{'Step':<30} {'IVF-TQ':>8} {'PQ stale':>9} {'PQ retrain':>11} {'Retrain cost':>13}")
    print("-" * 75)

    def measure(step_name, n_indexed):
        gt = FAISSFlatIndex(dim)
        gt.add(vectors[:n_indexed])
        _, gt_idx = gt.search(queries, k=k)

        _, tq_idx = ivf_tq.search(queries, k=k, rerank=50)
        tq_r = compute_recall(gt_idx, tq_idx, k)

        _, pq_s_idx = ivf_pq_stale.search(queries_normed, k)
        pq_s_r = compute_recall(gt_idx, pq_s_idx, k)

        _, pq_r_idx = ivf_pq_retrain.search(queries_normed, k)
        pq_r_r = compute_recall(gt_idx, pq_r_idx, k)

        step = {
            "step": step_name,
            "n_indexed": n_indexed,
            "ivf_tq": round(tq_r, 4),
            "ivf_pq_stale": round(pq_s_r, 4),
            "ivf_pq_retrain": round(pq_r_r, 4),
            "total_retrain_time_s": round(total_retrain_time, 2),
        }
        results["steps"].append(step)
        print(f"  {step_name:<28} {tq_r:>7.1%} {pq_s_r:>8.1%} {pq_r_r:>10.1%} {total_retrain_time:>10.1f}s")

    measure(f"Initial ({n_initial//1000}K)", n_initial)

    for batch_i in range(n_batches):
        start = n_initial + batch_i * batch_size
        end = start + batch_size
        batch = vectors[start:end]
        batch_normed = normalize(batch)

        # Add to all indexes
        ivf_tq.add(batch)
        ivf_pq_stale.add(batch_normed)

        # For retrain variant: add, then check if we need to retrain
        all_normed_so_far.append(batch_normed)
        vectors_since_retrain += batch_size

        if vectors_since_retrain >= retrain_interval:
            # Retrain from scratch on all data so far
            all_data = np.concatenate(all_normed_so_far)
            t0 = time.time()
            ivf_pq_retrain = make_ivfpq(all_data)
            ivf_pq_retrain.add(all_data)
            retrain_cost = time.time() - t0
            total_retrain_time += retrain_cost
            vectors_since_retrain = 0
        else:
            ivf_pq_retrain.add(batch_normed)

        measure(f"Batch {batch_i+1} ({end//1000}K)", end)

    # Summary
    init = results["steps"][0]
    final = results["steps"][-1]
    print(f"\n{'='*75}")
    print(f"Summary:")
    print(f"  IVF-TQ (no retrain):    {init['ivf_tq']:.1%} → {final['ivf_tq']:.1%}  "
          f"({(final['ivf_tq']-init['ivf_tq'])*100:+.1f}pp)  retrain cost: 0s")
    print(f"  IVF-PQ (stale):         {init['ivf_pq_stale']:.1%} → {final['ivf_pq_stale']:.1%}  "
          f"({(final['ivf_pq_stale']-init['ivf_pq_stale'])*100:+.1f}pp)  retrain cost: 0s")
    print(f"  IVF-PQ (retrain/200K):  {init['ivf_pq_retrain']:.1%} → {final['ivf_pq_retrain']:.1%}  "
          f"({(final['ivf_pq_retrain']-init['ivf_pq_retrain'])*100:+.1f}pp)  retrain cost: {total_retrain_time:.1f}s")

    out_path = Path(__file__).parent / "streaming_retrain_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


if __name__ == "__main__":
    run_experiment()
