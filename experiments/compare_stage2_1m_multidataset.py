"""
Sign-bit refinement vs QJL at 1M scale, on three datasets.

Replicates compare_stage2.py at 1M scale on SIFT-1M, Deep-1M, GloVe-100 (1M
vectors), to validate that the sign-bit advantage holds beyond a single dataset.

Outputs experiments/stage2_1m_multidataset_results.json.
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from turboquant_search.core import TurboQuantSearchIndex, FlatSearchIndex
from turboquant_search.faiss_baselines import FAISS_AVAILABLE, FAISSFlatIndex
from turboquant_search.benchmarks import compute_recall
from turboquant_search.datasets import load_sift1m, load_deep1m, load_glove100
from experiments.qjl_index import QJLSearchIndex


N_QUERIES = 1000
BITS = [3, 4]
K = 10


def run_one(dataset_name, vectors, queries, label):
    dim = vectors.shape[1]
    n = vectors.shape[0]

    t0 = time.time()
    if FAISS_AVAILABLE:
        gt = FAISSFlatIndex(dim)
    else:
        gt = FlatSearchIndex(dim)
    gt.add(vectors)
    _, gt_idx = gt.search(queries, k=K)
    print(f"  Ground truth: {time.time()-t0:.1f}s")

    out = {"dataset": dataset_name, "label": label, "n": int(n), "dim": int(dim), "results": {}}

    for bits in BITS:
        bit_key = f"{bits}-bit"
        out["results"][bit_key] = {}

        for variant, factory in [
            ("no_stage2", lambda: TurboQuantSearchIndex(dim, bits=bits, use_residual_sign=False, seed=42)),
            ("signbit",   lambda: TurboQuantSearchIndex(dim, bits=bits, use_residual_sign=True,  seed=42)),
            ("qjl",       lambda: QJLSearchIndex(dim, bits=bits, seed=42)),
        ]:
            t0 = time.time()
            idx = factory()
            idx.add(vectors)
            _, pred_idx = idx.search(queries, k=K)
            recall = compute_recall(gt_idx[:, :K], pred_idx[:, :K], K)
            elapsed = time.time() - t0
            out["results"][bit_key][variant] = {"recall_at_10": float(recall), "elapsed_s": float(elapsed)}
            print(f"  {bit_key} {variant:>10}: R@10 = {recall:.4f} ({elapsed:.1f}s)")

    return out


def main():
    out_path = os.path.join(os.path.dirname(__file__), "stage2_1m_multidataset_results.json")

    all_results = {}

    for dataset_key, loader in [
        ("sift-1m",  lambda: load_sift1m(1_000_000, N_QUERIES)),
        ("deep-1m",  lambda: load_deep1m(1_000_000, N_QUERIES)),
        ("glove-1m", lambda: load_glove100(1_000_000, N_QUERIES)),
    ]:
        print(f"\n{'='*80}")
        print(f"  {dataset_key.upper()} @ 1M")
        print(f"{'='*80}")

        result = loader()
        if result is None:
            print(f"  SKIP: dataset {dataset_key} failed to load")
            continue

        vectors, queries, label = result
        print(f"  Loaded: {label}")
        all_results[dataset_key] = run_one(dataset_key, vectors, queries, label)

        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Wrote {out_path}")

    print(f"\nDone. Results in {out_path}")


if __name__ == "__main__":
    main()
