"""Quick smoke test: build IVFTQIndex on Deep-1M, verify cascade C++ ≈ baseline."""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from turboquant_search.core import IVFTurboQuantIndex as IVFTQIndex
from turboquant_search.faiss_baselines import FAISSFlatIndex
from turboquant_search.benchmarks import compute_recall
from turboquant_search.datasets import load_deep1m
from experiments.cascade_search_cpp import search_cascade_cpp


def main():
    print("Loading Deep-1M ...")
    r = load_deep1m(1_000_000, 1000)
    v, q, _ = r
    dim = v.shape[1]
    print(f"  loaded n={v.shape[0]}, dim={dim}")

    print("Computing GT (FAISS Flat) ...")
    gt_idx = FAISSFlatIndex(dim); gt_idx.add(v)
    _, gt = gt_idx.search(q, k=10)

    print("Building IVFTQIndex (b=5+sign, nlist=1000, nprobe=40) ...")
    t0 = time.time()
    idx = IVFTQIndex(dim=dim, nlist=1000, bits=5, nprobe=40,
                     use_residual_sign=True, seed=42)
    idx.train(v)
    idx.add(v)
    print(f"  built in {time.time() - t0:.1f}s")

    # Baseline (C++ ivf_search via core.py)
    print("\n=== Baseline (C++ ivf_search) ===")
    t0 = time.time()
    _, pred_base = idx.search(q, k=10)
    base_t = time.time() - t0
    base_r = compute_recall(gt[:, :10], pred_base[:, :10], 10)
    print(f"  R@10 = {base_r:.4f}, time = {base_t:.2f}s, QPS = {q.shape[0]/base_t:.0f}")

    # C++ cascade
    print("\n=== C++ cascade (msb=4, N=100) ===")
    # warm caches
    _ = search_cascade_cpp(idx, q[:8], k=10, top_msb_bits=4, rerank_n=100)
    t0 = time.time()
    pred_casc = search_cascade_cpp(idx, q, k=10, top_msb_bits=4, rerank_n=100)
    casc_t = time.time() - t0
    casc_r = compute_recall(gt[:, :10], pred_casc[:, :10], 10)
    print(f"  R@10 = {casc_r:.4f}, time = {casc_t:.2f}s, QPS = {q.shape[0]/casc_t:.0f}")
    print(f"  Δ = {(casc_r - base_r) * 100:+.3f}pp,  speedup = {base_t/casc_t:.2f}×")

    # Sanity overlap: top-10 prediction ID overlap between baseline and cascade
    overlap = np.mean([
        len(set(pred_base[i].tolist()) & set(pred_casc[i].tolist())) / 10
        for i in range(q.shape[0])
    ])
    print(f"  top-10 ID overlap baseline vs cascade: {overlap*100:.1f}%")


if __name__ == "__main__":
    main()
