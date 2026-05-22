"""
Head-to-head comparison: Sign-Bit Refinement vs QJL (Stage 2).

Both share identical Stage 1 (rotation + Lloyd-Max). Only Stage 2 differs:
  - Sign-bit: per-coordinate sign of (value - centroid), reconstruct with conditional mean
  - QJL: random Gaussian projection of residual, store signs, unbiased IP correction

This produces the empirical data needed for a preprint.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from turboquant_search.core import TurboQuantSearchIndex, FlatSearchIndex
from turboquant_search.faiss_baselines import FAISS_AVAILABLE, FAISSFlatIndex
from turboquant_search.benchmarks import compute_recall
from turboquant_search.datasets import load_synthetic, load_sift128, load_glove100
from experiments.qjl_index import QJLSearchIndex


def run_comparison(dataset_name, vectors, queries, label):
    """Run sign-bit vs QJL vs no-Stage2 on one dataset."""
    dim = vectors.shape[1]
    n = vectors.shape[0]

    # Ground truth
    if FAISS_AVAILABLE:
        gt = FAISSFlatIndex(dim)
    else:
        gt = FlatSearchIndex(dim)
    gt.add(vectors)
    _, gt_idx = gt.search(queries, k=50)

    k_values = [1, 5, 10, 50]

    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"  {n:,} vectors, dim={dim}, {queries.shape[0]} queries")
    print(f"{'='*80}")
    print()

    header = f"{'Method':<40} {'Memory':>8} {'Ratio':>7} {'Build':>8}"
    for k in k_values:
        header += f" {'R@'+str(k):>7}"
    print(header)
    print("-" * len(header))

    for bits in [2, 3, 4]:
        # --- No Stage 2 (Lloyd-Max only) ---
        tq_none = TurboQuantSearchIndex(dim, bits=bits, use_residual_sign=False, seed=42)
        tq_none.add(vectors)
        _, idx_none = tq_none.search(queries, k=50)

        row = f"TQ {bits}-bit (no Stage 2)                "[:40]
        row += f" {tq_none.memory_bytes/1e6:>7.2f}M {tq_none.compression_ratio:>6.1f}x"
        row += f" {tq_none.build_time:>7.3f}s"
        for k in k_values:
            r = compute_recall(gt_idx[:, :k], idx_none[:, :k], k)
            row += f" {r:>6.1%}"
        print(row)

        # --- Sign-bit refinement (our approach) ---
        tq_sign = TurboQuantSearchIndex(dim, bits=bits, use_residual_sign=True, seed=42)
        tq_sign.add(vectors)
        _, idx_sign = tq_sign.search(queries, k=50)

        row = f"TQ {bits}-bit + sign-bit refinement      "[:40]
        row += f" {tq_sign.memory_bytes/1e6:>7.2f}M {tq_sign.compression_ratio:>6.1f}x"
        row += f" {tq_sign.build_time:>7.3f}s"
        for k in k_values:
            r = compute_recall(gt_idx[:, :k], idx_sign[:, :k], k)
            row += f" {r:>6.1%}"
        print(row)

        # --- QJL (TurboQuant's Stage 2) ---
        qjl = QJLSearchIndex(dim, bits=bits, seed=42)
        qjl.add(vectors)
        _, idx_qjl = qjl.search(queries, k=50)

        row = f"TQ {bits}-bit + QJL (paper)              "[:40]
        row += f" {qjl.memory_bytes/1e6:>7.2f}M {qjl.compression_ratio:>6.1f}x"
        row += f" {qjl.build_time:>7.3f}s"
        for k in k_values:
            r = compute_recall(gt_idx[:, :k], idx_qjl[:, :k], k)
            row += f" {r:>6.1%}"
        print(row)

        print()


def main():
    print("Stage 2 Comparison: Sign-Bit Refinement vs QJL")
    print("=" * 80)
    print("Stage 1 (rotation + Lloyd-Max) is IDENTICAL for all methods.")
    print("Only Stage 2 differs: sign-bit refinement vs QJL vs none.")
    print()

    # Synthetic
    vectors, queries, label = load_synthetic(10000, 200, 128, 42)
    run_comparison("synthetic", vectors, queries, label)

    # SIFT-128
    result = load_sift128(10000, 200)
    if result is not None:
        vectors, queries, label = result
        run_comparison("sift-128", vectors, queries, label)

    # GloVe-100
    result = load_glove100(10000, 200)
    if result is not None:
        vectors, queries, label = result
        run_comparison("glove-100", vectors, queries, label)


if __name__ == "__main__":
    main()
