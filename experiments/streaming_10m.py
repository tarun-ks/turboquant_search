"""
Streaming ingestion at 10M scale (Deep-10M).

Train all indexes on the first 1M vectors, then add 9 batches of 1M each.
Recompute Recall@10 against the cumulative database after every batch
using a fixed 10K-query set with seed=42.

Outputs experiments/streaming_10m_results.json.

Memory: peaks at ~10GB during the final FAISS Flat ground-truth pass.
Runtime: ~45-90 min on a recent Mac M-series laptop.
"""

import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.environ["TQS_THREADS"] = str(os.cpu_count() or 1)

from turboquant_search.core import IVFTurboQuantIndex
from turboquant_search.faiss_baselines import FAISS_AVAILABLE, FAISSFlatIndex
from turboquant_search.benchmarks import compute_recall

assert FAISS_AVAILABLE, "faiss-cpu required"
import faiss


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def normalize(v):
    v = np.ascontiguousarray(v.astype(np.float32))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(norms, 1e-8)


def load_deep10m_cached():
    cache = ROOT / "experiments" / "cache"
    vec_path = cache / "deep10m_vectors.npy"
    qry_path = cache / "deep10m_queries.npy"
    if not (vec_path.exists() and qry_path.exists()):
        log("Cache missing; run experiments/run_hnsw_opq.py --scale 10m first.")
        sys.exit(1)
    return np.load(vec_path, mmap_mode="r"), np.load(qry_path)


def main():
    vectors, queries = load_deep10m_cached()
    n_total, dim = vectors.shape
    nq = queries.shape[0]
    log(f"Deep-10M: {n_total:,} vectors, dim={dim}, {nq} queries")

    n_initial = 1_000_000
    batch_size = 1_000_000
    n_batches = 9
    nlist, nprobe, m_pq = 3162, 20, 48  # standard for Deep-10M

    init = np.asarray(vectors[:n_initial])
    init_normed = normalize(init)
    queries_normed = normalize(queries)

    # IVF-TQ
    log("Training IVF-TQ on first 1M...")
    ivf_tq = IVFTurboQuantIndex(dim, nlist=nlist, bits=4, nprobe=nprobe,
                                 use_residual_sign=True, seed=42)
    ivf_tq.train(init)
    ivf_tq.add(init)
    ivf_tq._raw_vectors = None  # save memory
    gc.collect()

    # IVF-PQ stale (codebook trained once on first 1M)
    log("Training IVF-PQ stale on first 1M...")
    quantizer = faiss.IndexFlatIP(dim)
    ivf_pq_stale = faiss.IndexIVFPQ(quantizer, dim, nlist, m_pq, 8,
                                     faiss.METRIC_INNER_PRODUCT)
    ivf_pq_stale.nprobe = nprobe
    ivf_pq_stale.train(init_normed)
    ivf_pq_stale.add(init_normed)

    # IVF-PQ retrain (codebook re-trained every batch)
    def make_ivfpq(data):
        q = faiss.IndexFlatIP(dim)
        idx = faiss.IndexIVFPQ(q, dim, nlist, m_pq, 8, faiss.METRIC_INNER_PRODUCT)
        idx.nprobe = nprobe
        idx.train(data)
        return idx

    log("Training IVF-PQ retrain on first 1M...")
    ivf_pq_retrain = make_ivfpq(init_normed)
    ivf_pq_retrain.add(init_normed)
    cumulative_normed = [init_normed]
    total_retrain_time = 0.0

    streaming_results = []

    def measure(step_label, n_indexed):
        log(f"  GT recompute against {n_indexed//1_000_000}M...")
        gt_idx = FAISSFlatIndex(dim)
        gt_idx.add(np.asarray(vectors[:n_indexed]))
        _, gt = gt_idx.search(queries, k=10)
        del gt_idx; gc.collect()

        _, tq_I = ivf_tq.search(queries, k=10, rerank=50)
        tq_r = compute_recall(gt, tq_I, 10)

        _, pq_s_I = ivf_pq_stale.search(queries_normed, 10)
        pq_s_r = compute_recall(gt, pq_s_I, 10)

        _, pq_r_I = ivf_pq_retrain.search(queries_normed, 10)
        pq_r_r = compute_recall(gt, pq_r_I, 10)

        entry = {
            "step": step_label,
            "n_indexed": n_indexed,
            "ivf_tq": round(tq_r * 100, 2),
            "ivf_pq_stale": round(pq_s_r * 100, 2),
            "ivf_pq_retrain": round(pq_r_r * 100, 2),
            "retrain_time_cumulative_s": round(total_retrain_time, 1),
        }
        streaming_results.append(entry)
        log(f"  {step_label}: TQ={tq_r:.1%} PQ_stale={pq_s_r:.1%} PQ_retrain={pq_r_r:.1%}")

    measure(f"Initial (1M)", n_initial)

    for batch_i in range(n_batches):
        start = n_initial + batch_i * batch_size
        end = min(start + batch_size, n_total)
        log(f"Batch {batch_i+1}: adding {start//1_000_000}M->{end//1_000_000}M...")
        batch = np.asarray(vectors[start:end])
        batch_normed = normalize(batch)

        ivf_tq.add(batch)
        ivf_tq._raw_vectors = None; gc.collect()

        ivf_pq_stale.add(batch_normed)

        # Retrain variant: re-train every batch (1M new vectors)
        cumulative_normed.append(batch_normed)
        all_normed = np.concatenate(cumulative_normed)
        t0 = time.time()
        ivf_pq_retrain = make_ivfpq(all_normed)
        ivf_pq_retrain.add(all_normed)
        total_retrain_time += time.time() - t0
        del all_normed; gc.collect()

        measure(f"Batch {batch_i+1} ({end//1_000_000}M)", end)

    out_path = ROOT / "experiments" / "streaming_10m_results.json"
    with open(out_path, "w") as f:
        json.dump({"steps": streaming_results,
                    "config": {
                        "dataset": "Deep-10M",
                        "n_initial": n_initial,
                        "batch_size": batch_size,
                        "n_batches": n_batches,
                        "nlist": nlist,
                        "nprobe": nprobe,
                        "m_pq": m_pq,
                    }}, f, indent=2)
    log(f"Saved {out_path}")


if __name__ == "__main__":
    main()
