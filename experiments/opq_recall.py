"""
OPQ+IVF-PQ static recall@10 on SIFT-1M and Deep-1M.

Establishes traceable baselines for the E&A comparison table.
Uses the same normalization protocol as recall_qps_sweep.py:
  - unit-L2 normalized vectors and queries
  - inner-product metric
  - ground truth computed by exact FlatIP on the same normalized vectors

Configs:
  SIFT-1M (d=128):  m=64,  m=128   (8-bit codes, nprobe sweep)
  Deep-1M  (d=96):  m=48,  m=96    (8-bit codes, nprobe sweep)

Note on nprobe: FAISSOPQIVFPQIndex wraps the inner IVF via IndexPreTransform,
so we use faiss.ParameterSpace to set nprobe on the outer index — the only way
that reliably propagates through the wrapper.

Output: experiments/results/opq_recall_results.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import faiss

from turboquant_search.benchmarks import compute_recall
from turboquant_search.datasets import load_sift1m, load_deep1m
from turboquant_search.faiss_baselines import FAISS_AVAILABLE

assert FAISS_AVAILABLE, "faiss-cpu required: pip install faiss-cpu"

RESULTS_DIR = ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT = RESULTS_DIR / "opq_recall_results.json"

NPROBES = [5, 10, 20, 40, 80, 160]
K = 10
NRUNS_QPS = 3
NLIST = 1000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return (v / np.maximum(norms, 1e-8)).astype(np.float32)


def measure_qps(fn, n_queries: int, n_runs: int = NRUNS_QPS) -> float:
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return n_queries / float(np.median(times))


def build_opq_ivfpq(vecs_normed: np.ndarray, m: int, nlist: int) -> faiss.Index:
    """Build OPQ+IVF-PQ. Returns the IndexPreTransform wrapping IndexIVFPQ."""
    dim = vecs_normed.shape[1]
    assert dim % m == 0, f"dim={dim} must be divisible by m={m}"

    opq = faiss.OPQMatrix(dim, m)
    opq.niter = 25

    quantizer = faiss.IndexFlatIP(dim)
    ivfpq = faiss.IndexIVFPQ(
        quantizer, dim, nlist, m, 8, faiss.METRIC_INNER_PRODUCT
    )
    index = faiss.IndexPreTransform(opq, ivfpq)

    t0 = time.perf_counter()
    index.train(vecs_normed)
    index.add(vecs_normed)
    build_time = time.perf_counter() - t0
    log(f"    Built in {build_time:.1f}s")
    return index


def run_opq_sweep(
    vecs_normed: np.ndarray,
    queries_normed: np.ndarray,
    gt: np.ndarray,
    m: int,
) -> list[dict]:
    dim = vecs_normed.shape[1]
    n = len(vecs_normed)
    assert dim % m == 0, f"dim={dim} not divisible by m={m}"

    # Memory: codes + PQ centroids + IVF centroids + OPQ rotation matrix
    code_bytes = n * m                              # m subvectors × 1 byte each
    pq_cents = m * 256 * (dim // m) * 4            # PQ codebook
    ivf_cents = NLIST * dim * 4                    # coarse centroids
    opq_matrix = dim * dim * 4                     # rotation matrix
    mem_mb = round((code_bytes + pq_cents + ivf_cents + opq_matrix) / 1e6, 1)

    log(f"    Building OPQ+IVF-PQ m={m} (nlist={NLIST}, 8-bit) …  memory≈{mem_mb} MB")
    index = build_opq_ivfpq(vecs_normed, m, NLIST)

    ps = faiss.ParameterSpace()
    rows = []
    for np_ in NPROBES:
        ps.set_index_parameter(index, "nprobe", np_)
        _, I = index.search(queries_normed, K)
        recall = round(compute_recall(gt, I, K) * 100, 2)
        qps = round(measure_qps(lambda: index.search(queries_normed, K), len(queries_normed)))
        rows.append({"nprobe": np_, "recall10": recall, "qps": qps, "memory_mb": mem_mb})
        log(f"      nprobe={np_:3d}  recall10={recall:.2f}%  QPS={qps:,}")
    return rows


def run_dataset(name: str, loader_fn) -> dict:
    log(f"\n=== {name} ===")
    result = loader_fn()
    if result is None:
        log(f"  SKIP: {name} not available")
        return {}
    vectors, queries, _ = result
    vecs = normalize(np.asarray(vectors, dtype=np.float32))
    qrys = normalize(np.asarray(queries, dtype=np.float32))
    dim = vecs.shape[1]
    log(f"  {len(vecs):,} × {dim}d  |  {len(qrys):,} queries")

    log("  Computing ground truth (FlatIP on normalized vectors) …")
    flat = faiss.IndexFlatIP(dim)
    flat.add(vecs)
    _, gt = flat.search(qrys, K)

    dataset_results = {}
    if dim == 128:
        configs = [64, 128]
    elif dim == 96:
        configs = [48, 96]
    else:
        # fallback: largest divisor ≤ dim//2 and dim itself
        configs = [c for c in range(dim // 2, 0, -1) if dim % c == 0][:2]

    for m in configs:
        key = f"opq_m{m}"
        dataset_results[key] = run_opq_sweep(vecs, qrys, gt, m)

    return dataset_results


def main() -> None:
    results = {}

    results["SIFT-1M"] = run_dataset(
        "SIFT-1M",
        lambda: load_sift1m(n_vectors=1_000_000, n_queries=10_000),
    )
    results["Deep-1M"] = run_dataset(
        "Deep-1M",
        lambda: load_deep1m(n_vectors=1_000_000, n_queries=10_000),
    )

    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults written to {OUTPUT}")

    log("\n--- Summary ---")
    for dataset, data in results.items():
        log(f"{dataset}:")
        for key, rows in data.items():
            best = max(rows, key=lambda r: r["recall10"])
            log(f"  {key}: best recall10={best['recall10']}% at nprobe={best['nprobe']}")


if __name__ == "__main__":
    main()
