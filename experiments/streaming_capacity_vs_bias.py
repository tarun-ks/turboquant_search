"""
Capacity-vs-Bias Control Experiment (SIFT-1M).

We claim the dominant streaming-recall mechanism for IVF-PQ is
*initial-sample bias*, not codebook capacity at the 5-6 bits/dim regime.
PVLDB reviewers may push back: "How do you know it's bias, not capacity?
A 200K-trained PQ codebook may simply have too few centroids/levels to
serve a 1M database well, regardless of which 200K you trained on."

This script disambiguates by training three variants of IVF-PQ and evaluating
all three against the same 1M database:

    A) PQ-200K-initial    — codebook trained on the first 200K vectors
                            (the streaming "stale" condition).
    B) PQ-200K-random     — codebook trained on a uniformly random 200K
                            sample of the FULL 1M (same training size as A,
                            but eliminates initial-sample bias).
    C) PQ-1M              — codebook trained on the full 1M (capacity ceiling).

Predictions:
    A << B << C     => initial-sample bias is the dominant mechanism (paper claim).
    A == B << C     => capacity at 200K-trained codebook is the bottleneck.
    A == B == C     => neither bias nor capacity matter at this bit budget
                       (recall is bottlenecked elsewhere, e.g., partition).

This script is the controlled experiment that tells reviewers which of the
three our streaming-mechanism claim sits on. Reproduces the paragraph
"capacity-vs-bias control" referenced from Section 4.1.

Usage:
    python experiments/streaming_capacity_vs_bias.py
    # Output: streaming_capacity_vs_bias_results.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "code"))

from turboquant_search.datasets import load_sift1m
from turboquant_search.benchmarks import compute_recall
from turboquant_search.faiss_baselines import FAISSFlatIndex, FAISS_AVAILABLE


N_TRAIN_SMALL = 200_000
N_TOTAL = 1_000_000
N_QUERIES = 1_000
NLIST = 1000
NPROBE = 20
M_PQ = 64       # 4 bits/dim at d=128 -> 64 subquantisers @ 8 bits each
SEED = 42


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.ascontiguousarray(v.astype(np.float32))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(norms, 1e-8)


def build_groundtruth(vectors: np.ndarray, queries: np.ndarray, k: int = 10) -> np.ndarray:
    flat = FAISSFlatIndex(vectors.shape[1])
    flat.add(vectors)
    _, gt = flat.search(queries, k=k)
    return gt


def build_ivfpq(dim: int, train_data: np.ndarray, full_data: np.ndarray):
    """Train IVF-PQ on `train_data`, then index the entire `full_data`."""
    import faiss
    quantizer = faiss.IndexFlatIP(dim)
    idx = faiss.IndexIVFPQ(quantizer, dim, NLIST, M_PQ, 8, faiss.METRIC_INNER_PRODUCT)
    idx.nprobe = NPROBE
    t0 = time.time()
    idx.train(train_data)
    train_t = time.time() - t0
    t0 = time.time()
    idx.add(full_data)
    add_t = time.time() - t0
    return idx, train_t, add_t


def search_and_score(idx, queries: np.ndarray, gt: np.ndarray, k: int = 10):
    t0 = time.time()
    _, found = idx.search(queries, k)
    qps = queries.shape[0] / max(time.time() - t0, 1e-6)
    return compute_recall(gt, found, k), qps


def main():
    assert FAISS_AVAILABLE, "FAISS required for this experiment"

    print("Loading SIFT-1M...", flush=True)
    res = load_sift1m(n_vectors=N_TOTAL, n_queries=N_QUERIES)
    vectors, queries, _ = res
    dim = vectors.shape[1]

    # Use only the first N_QUERIES queries for speed.
    queries = queries[:N_QUERIES]

    print(f"Vectors {vectors.shape}, queries {queries.shape}", flush=True)
    vectors_n = normalize(vectors)
    queries_n = normalize(queries)

    print("Computing ground truth on full 1M...", flush=True)
    gt = build_groundtruth(vectors_n, queries_n, k=10)

    rng = np.random.RandomState(SEED)

    # Three training samples, all of size N_TRAIN_SMALL or N_TOTAL.
    train_samples = {
        "A_initial_200K": vectors_n[:N_TRAIN_SMALL],
        "B_random_200K":  vectors_n[rng.choice(N_TOTAL, N_TRAIN_SMALL, replace=False)],
        "C_full_1M":      vectors_n,
    }

    results = []
    for name, train_data in train_samples.items():
        print(f"\n=== {name}: training on {train_data.shape[0]} vectors ===", flush=True)
        idx, train_t, add_t = build_ivfpq(dim, train_data, vectors_n)
        recall, qps = search_and_score(idx, queries_n, gt, k=10)
        print(f"  R@10={recall*100:.2f}%  train={train_t:.1f}s  add={add_t:.1f}s  qps={qps:.0f}",
              flush=True)
        results.append({
            "variant": name,
            "n_train": int(train_data.shape[0]),
            "recall10": round(recall * 100, 2),
            "train_s": round(train_t, 2),
            "add_s": round(add_t, 2),
            "qps": round(qps, 0),
        })

    # Diagnostic: what does this say about bias vs capacity?
    a, b, c = results[0]["recall10"], results[1]["recall10"], results[2]["recall10"]
    bias_gap = b - a
    capacity_gap = c - b
    print("\n=== Diagnostic ===")
    print(f"  A (initial 200K)   : {a:.2f}%")
    print(f"  B (random 200K)    : {b:.2f}%")
    print(f"  C (full 1M)        : {c:.2f}%")
    print(f"  Bias gap (B-A)     : {bias_gap:+.2f}pp  (initial-sample bias)")
    print(f"  Capacity gap (C-B) : {capacity_gap:+.2f}pp  (200K codebook capacity)")
    if abs(bias_gap) > abs(capacity_gap):
        verdict = "bias-dominated (streaming-mechanism claim holds)"
    elif abs(capacity_gap) > 2 * abs(bias_gap):
        verdict = "capacity-dominated (paper claim weakens)"
    else:
        verdict = "mixed (re-frame Section 4.1 accordingly)"
    print(f"  Verdict             : {verdict}")

    out = {
        "dataset": "SIFT-1M",
        "n_total": N_TOTAL,
        "n_train_small": N_TRAIN_SMALL,
        "n_queries": int(queries.shape[0]),
        "nlist": NLIST,
        "nprobe": NPROBE,
        "m_pq": M_PQ,
        "results": results,
        "bias_gap": round(bias_gap, 2),
        "capacity_gap": round(capacity_gap, 2),
        "verdict": verdict,
    }

    out_path = Path(__file__).parent / "streaming_capacity_vs_bias_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
