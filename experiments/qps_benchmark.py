"""
Clean QPS benchmark for IVF-TQ with the C++ NEON kernel.

Resolves the Table 1 split: the paper claims 11K QPS (C++ kernel run),
the recall_qps_sweep showed 4.5K (Python fallback at time of sweep).
This script measures cleanly with the rebuilt cpython-314 kernel active.

Protocol
--------
* SIFT-1M, 10K queries (the standard benchmark set).
* IVF-TQ at every (bits, nprobe) config in Table 1: (4,20), (5,20), (6,20), (6,40).
* FAISS IVF-PQ for comparison: (m64/b10, np=20), (m128/b10, np=20).
* Warm-up: 5 passes of 1K queries (discarded).
* Timed: 10 passes of all 10K queries; QPS = total_queries / total_time.
* Kernel active check: if _tqs_cpp not importable, script errors out.
* Output: experiments/results/qps_benchmark_sift1m.csv

Usage: python experiments/qps_benchmark.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

# Verify C++ kernel is present BEFORE anything else.
try:
    import turboquant_search._tqs_cpp as _cpp
    print(f"[kernel] C++ kernel loaded: {_cpp.__file__}", flush=True)
except ImportError as e:
    raise SystemExit(
        f"C++ kernel not available: {e}\n"
        "Run: pip install pybind11>=2.11.0 && pip install -e . --no-build-isolation"
    )

from turboquant_search.core import IVFTurboQuantIndex
from turboquant_search.datasets import load_sift1m
from turboquant_search.benchmarks import compute_recall
from turboquant_search.faiss_baselines import FAISS_AVAILABLE

assert FAISS_AVAILABLE, "faiss-cpu required"
import faiss  # noqa: E402

from streaming_multiseed import set_all_seeds, normalize, make_ivfpq, log  # noqa: E402

RESULTS_DIR = ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
WARMUP_PASSES = 5
TIMED_PASSES = 10


def load_sift1m_data() -> Tuple[np.ndarray, np.ndarray]:
    log("Loading SIFT-1M…")
    result = load_sift1m(n_vectors=1_000_000, n_queries=10_000)
    assert result is not None, "SIFT-1M load failed"
    vectors, queries, _ = result
    return np.asarray(vectors, dtype=np.float32), np.asarray(queries, dtype=np.float32)


def measure_qps(index, queries: np.ndarray, k: int = 10,
                use_normed: bool = True) -> Tuple[float, float]:
    """Return (QPS, recall@10) for the given index and queries."""
    n_total, dim = queries.shape
    if use_normed:
        q = normalize(queries)
    else:
        q = queries

    # Warm-up
    warm_q = q[:1000]
    for _ in range(WARMUP_PASSES):
        index.search(warm_q, k)

    # Timed
    t0 = time.perf_counter()
    for _ in range(TIMED_PASSES):
        _, results = index.search(q, k)
    elapsed = time.perf_counter() - t0
    qps = (n_total * TIMED_PASSES) / elapsed
    return qps, results


def build_ivftq(vectors_normed: np.ndarray, dim: int,
                bits: int, nprobe: int, seed: int) -> IVFTurboQuantIndex:
    set_all_seeds(seed)
    nlist = 1000
    idx = IVFTurboQuantIndex(dim, nlist=nlist, bits=bits, nprobe=nprobe,
                             use_residual_sign=True, seed=seed)
    idx.train(vectors_normed)
    idx.add(vectors_normed)
    idx._raw_vectors = None
    return idx


def build_pq(vectors_normed: np.ndarray, dim: int,
             m: int, bits_sub: int, nprobe: int, seed: int) -> faiss.Index:
    set_all_seeds(seed)
    nlist = 1000
    idx = make_ivfpq(dim, nlist, m, bits_sub, nprobe, seed, vectors_normed)
    idx.add(vectors_normed)
    return idx


def main():
    vectors, queries = load_sift1m_data()
    dim = vectors.shape[1]
    log(f"SIFT-1M: {vectors.shape[0]:,} × {dim}d, {queries.shape[0]:,} queries")

    vectors_normed = normalize(vectors)
    queries_normed = normalize(queries)

    # GT for recall check
    log("Computing GT…")
    gi = faiss.IndexFlatIP(dim)
    gi.add(vectors_normed)
    _, gt = gi.search(queries_normed, 10)
    del gi

    configs: list[dict] = []

    # IVF-TQ configs from Table 1
    for bits, nprobe in [(4, 20), (5, 20), (6, 20), (6, 40)]:
        label = f"ivf_tq_b{bits}_np{nprobe}"
        log(f"\n[{label}] building…")
        set_all_seeds(SEED)
        idx = build_ivftq(vectors_normed, dim, bits, nprobe, SEED)
        qps, results = measure_qps(idx, queries, use_normed=False)
        recall = compute_recall(gt, results, 10) * 100
        mb = (vectors.shape[0] * dim * bits) / (8 * 1024 * 1024)  # rough compressed size
        log(f"  QPS={qps:,.0f}  recall@10={recall:.2f}%  ~{mb:.0f} MB")
        configs.append({
            "method": label,
            "bits": bits,
            "nprobe": nprobe,
            "qps": round(qps, 0),
            "recall_at_10": round(recall, 2),
            "kernel": "cpp_neon",
        })
        del idx

    # IVF-PQ configs for comparison
    for m, bits_sub, nprobe in [(64, 10, 20), (64, 10, 80), (128, 10, 20)]:
        label = f"ivf_pq_m{m}_b{bits_sub}_np{nprobe}"
        log(f"\n[{label}] building…")
        idx = build_pq(vectors_normed, dim, m, bits_sub, nprobe, SEED)
        idx.nprobe = nprobe
        t0 = time.perf_counter()
        for _ in range(WARMUP_PASSES):
            idx.search(queries_normed[:1000], 10)
        elapsed = 0.0
        for _ in range(TIMED_PASSES):
            t1 = time.perf_counter()
            _, results = idx.search(queries_normed, 10)
            elapsed += time.perf_counter() - t1
        qps = (queries_normed.shape[0] * TIMED_PASSES) / elapsed
        recall = compute_recall(gt, results, 10) * 100
        log(f"  QPS={qps:,.0f}  recall@10={recall:.2f}%")
        configs.append({
            "method": label,
            "bits": bits_sub,
            "nprobe": nprobe,
            "qps": round(qps, 0),
            "recall_at_10": round(recall, 2),
            "kernel": "faiss_ivfpq",
        })
        del idx

    df = pd.DataFrame(configs)
    out = RESULTS_DIR / "qps_benchmark_sift1m.csv"
    df.to_csv(out, index=False)

    log("\n=== QPS Benchmark Results (SIFT-1M, C++ kernel) ===")
    for _, row in df.iterrows():
        log(f"  {row['method']:35s}  QPS={row['qps']:>8,.0f}  recall={row['recall_at_10']:.2f}%")
    log(f"\nWritten: {out}")


if __name__ == "__main__":
    main()
