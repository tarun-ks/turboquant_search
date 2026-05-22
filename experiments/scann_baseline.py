"""
ScaNN baseline (Linux-only — `pip install scann` does not have macOS wheels).

Implements a ScaNN wrapper that mirrors the FAISS-baseline interface used
by the rest of the experiments, using AsymmetricHashing scoring with the
anisotropic loss (Guo et al., 2020) and reordering on raw vectors.

Run on Colab (or any Linux machine with Python 3.10/3.11):
    pip install scann numpy faiss-cpu datasets
    python experiments/scann_baseline.py --datasets sift1m deep1m

Outputs experiments/scann_results.json with keys
    scann_<dataset>: {leaves_<L>: {recall10, qps, latency_ms, memory_mb, ...}}
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def normalize(v):
    v = np.ascontiguousarray(v.astype(np.float32))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(norms, 1e-8)


def compute_recall(gt, pred, k=10):
    n = gt.shape[0]
    hits = 0
    for i in range(n):
        hits += len(set(gt[i, :k]) & set(pred[i, :k]))
    return hits / (n * k)


def build_scann(db, num_leaves=2000, anisotropic_threshold=0.2,
                training_sample_size=250_000, reorder_n=100):
    import scann
    db = normalize(db)
    n, dim = db.shape
    builder = (
        scann.scann_ops_pybind.builder(db, 10, "dot_product")
        .tree(
            num_leaves=num_leaves,
            num_leaves_to_search=num_leaves // 20,  # default; overridden per-search
            training_sample_size=min(training_sample_size, n),
        )
        .score_ah(
            dimensions_per_block=2,
            anisotropic_quantization_threshold=anisotropic_threshold,
        )
        .reorder(reorder_n)
    )
    return builder.build()


def run_scann_sweep(db, queries, gt, num_leaves=2000,
                    leaves_to_search=(20, 50, 100, 200, 400)):
    print(f"  Building ScaNN: num_leaves={num_leaves}", flush=True)
    t0 = time.time()
    searcher = build_scann(db, num_leaves=num_leaves)
    build_time = time.time() - t0
    print(f"    build={build_time:.1f}s", flush=True)

    queries = normalize(queries)
    nq = queries.shape[0]
    sub = {}
    for L in leaves_to_search:
        # warmup
        searcher.search_batched(queries[:10], leaves_to_search=L)
        times = []
        I = None
        for _ in range(3):
            t0 = time.time()
            I, _ = searcher.search_batched(queries, leaves_to_search=L,
                                            final_num_neighbors=10)
            times.append(time.time() - t0)
        t = float(np.median(times))
        r = compute_recall(gt, I, 10)
        qps = nq / t if t > 0 else 0.0

        # Memory accounting:
        #   - PQ codes: n * (dim/2) bytes (1 byte per 2-dim block, 8-bit)
        #   - Codebook: 256 * (dim/2) * 2 * 4 = 1024 * dim bytes
        #   - Coarse partition centroids: num_leaves * dim * 4
        #   - Reorder data (raw vectors): n * dim * 4
        n, dim = db.shape
        codes = n * (dim // 2)
        codebook = 256 * (dim // 2) * 2 * 4
        coarse = num_leaves * dim * 4
        reorder = n * dim * 4
        compressed_mb = (codes + codebook + coarse) / (1024 * 1024)
        total_mb = compressed_mb + reorder / (1024 * 1024)

        sub[f"leaves{L}"] = {
            "recall10": round(r * 100, 1),
            "qps": round(qps),
            "latency_ms": round(t * 1000, 1),
            "memory_mb": round(compressed_mb, 1),
            "total_memory_mb": round(total_mb, 1),
            "compression": f"{(n * dim * 4) / (codes + codebook + coarse):.1f}x",
            "training": "Anisotropic AH codebook + tree partition",
        }
        print(f"    L={L:>3}: R@10={r:.1%}  {qps:.0f} QPS  "
              f"{t*1000:.1f}ms  {compressed_mb:.0f}MB", flush=True)
    return sub


def load_sift1m_via_npz():
    """Try the repo's loader; fall back to direct download for Colab."""
    try:
        from turboquant_search.datasets import load_sift1m
        r = load_sift1m(n_vectors=1_000_000, n_queries=10_000)
        if r is not None:
            v, q, _ = r
            return v, q
    except Exception:
        pass
    raise RuntimeError("SIFT-1M loader unavailable; run inside the repo.")


def load_deep1m_via_npz():
    try:
        from turboquant_search.datasets import load_deep1m
        r = load_deep1m(n_vectors=1_000_000, n_queries=10_000)
        if r is not None:
            v, q, _ = r
            return v, q
    except Exception:
        pass
    raise RuntimeError("Deep-1M loader unavailable; run inside the repo.")


def ground_truth_faiss(vectors, queries, k=10):
    import faiss
    v = normalize(vectors); q = normalize(queries)
    flat = faiss.IndexFlatIP(v.shape[1])
    flat.add(v)
    _, I = flat.search(q, k)
    return I


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["sift1m", "deep1m"])
    ap.add_argument("--out", default=str(ROOT / "experiments" / "scann_results.json"))
    args = ap.parse_args()

    results = {}
    for ds in args.datasets:
        print(f"\n=== {ds.upper()} ===", flush=True)
        if ds == "sift1m":
            v, q = load_sift1m_via_npz()
        elif ds == "deep1m":
            v, q = load_deep1m_via_npz()
        else:
            raise ValueError(f"unknown dataset: {ds}")
        n, dim = v.shape
        print(f"  loaded n={n}, dim={dim}, nq={q.shape[0]}", flush=True)
        print("  computing ground truth (FAISS Flat)...", flush=True)
        gt = ground_truth_faiss(v, q, k=10)
        sub = run_scann_sweep(v, q, gt, num_leaves=2000,
                              leaves_to_search=(20, 50, 100, 200, 400))
        results[f"scann_{ds}"] = sub

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
