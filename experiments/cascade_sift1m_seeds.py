"""
SIFT-1M cascade-search multi-seed verification (4 seeds × 1 dataset).

Produces the SIFT-1M multi-seed cascade verification results.
Combined with cascade_robustness.py (which produces the Deep-1M rows), this
yields the full 8-cell verification matrix (4 seeds × 2 datasets).

Output: experiments/cascade_sift1m_seeds_results.json
"""

import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from turboquant_search.faiss_baselines import FAISSFlatIndex
from turboquant_search.benchmarks import compute_recall
from turboquant_search.datasets import load_sift1m
from experiments.ivf_rvq_tq import IVFRVQTQIndex
from experiments.cascade_search import search_cascade


def main(seeds=(42, 43, 44, 45)):
    print("Loading SIFT-1M ...")
    r = load_sift1m(1_000_000, 1000)
    if r is None:
        print("FAILED to load SIFT-1M")
        return
    v, q, _ = r
    dim = v.shape[1]
    print(f"  loaded n={v.shape[0]}, dim={dim}")

    print("Computing GT ...")
    gt_idx = FAISSFlatIndex(dim); gt_idx.add(v)
    _, gt = gt_idx.search(q, k=10)

    print(f"\n=== Multi-seed cascade verification @ SIFT-1M, b=5+1, np=40, msb=4, N=100 ===")
    out = []
    for seed in seeds:
        print(f"\n  seed={seed}: building IVF-RVQ-TQ ...")
        idx = IVFRVQTQIndex(dim=dim, nlist=1000, bits=5, refine_bits=1,
                            nprobe=40, seed=seed)
        idx.train(v); idx.add(v)

        # Baseline (full-precision search)
        t0 = time.time()
        _, pred = idx.search(q, k=10)
        base_t = time.time() - t0
        base_r = compute_recall(gt[:, :10], pred[:, :10], 10)

        # Cascade (msb=4, N=100)
        t0 = time.time()
        res = search_cascade(idx, q, k=10, top_msb_bits=4, rerank_n=100)
        cascade_t = time.time() - t0
        pred_c, _, _ = res
        cascade_r = compute_recall(gt[:, :10], pred_c[:, :10], 10)

        delta_pp = (cascade_r - base_r) * 100
        speedup = base_t / max(cascade_t, 1e-6)
        out.append({
            "seed": seed,
            "baseline": float(base_r),
            "cascade": float(cascade_r),
            "delta_pp": float(delta_pp),
            "speedup": float(speedup),
            "base_s": float(base_t),
            "cascade_s": float(cascade_t),
        })
        print(f"    base={base_r:.4f}  cascade={cascade_r:.4f}  Δ={delta_pp:+.3f}pp  speedup={speedup:.2f}×")

    out_path = os.path.join(os.path.dirname(__file__), "cascade_sift1m_seeds_results.json")
    with open(out_path, "w") as f:
        json.dump({"sift1m_seeds": out}, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
