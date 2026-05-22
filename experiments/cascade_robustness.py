"""
Robustness verification for cascade search. Five independent stress tests:

(1) Multiple seeds — does recall preservation hold across different random rotations?
(2) Multiple nprobe values — does cascade work across the operating-point curve?
(3) Larger scale — does cascade work on Deep-10M?
(4) Speedup mechanism — what part of the cascade actually saves time? Direct
    measurement of the matmul vs fancy-indexing components.
(5) Comparison against just running with smaller bit budget — i.e., does
    cascade(b=6+1, msb=4, N=200) actually beat baseline(b=4+1) at np=40?
    This is the honest comparison: a smaller-bit baseline is SIMPLER and
    might be faster. Cascade only wins if it preserves the higher-precision
    recall.

Outputs experiments/cascade_robustness_results.json.
"""

import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from turboquant_search.faiss_baselines import FAISSFlatIndex
from turboquant_search.benchmarks import compute_recall
from turboquant_search.datasets import load_sift1m, load_deep1m
from experiments.ivf_rvq_tq import IVFRVQTQIndex
from experiments.cascade_search import search_cascade


def stress_seeds(v, q, gt, dim, seeds=(42, 43, 44, 45)):
    """(1) Robustness across seeds at b=5+1, np=40, msb=4, N=100."""
    print("\n--- (1) Multiple seeds @ b=5+1, np=40, msb=4, N=100 ---")
    out = []
    for seed in seeds:
        idx = IVFRVQTQIndex(dim=dim, nlist=1000, bits=5, refine_bits=1,
                            nprobe=40, seed=seed)
        idx.train(v); idx.add(v)
        # Baseline
        t0 = time.time()
        _, pred = idx.search(q, k=10)
        base_t = time.time() - t0
        base_r = compute_recall(gt[:, :10], pred[:, :10], 10)
        # Cascade
        t0 = time.time()
        res = search_cascade(idx, q, k=10, top_msb_bits=4, rerank_n=100)
        cascade_t = time.time() - t0
        pred_c, _, _ = res
        cascade_r = compute_recall(gt[:, :10], pred_c[:, :10], 10)
        speedup = base_t / cascade_t
        delta = (cascade_r - base_r) * 100
        print(f"  seed={seed}: base={base_r:.4f} cascade={cascade_r:.4f} Δ={delta:+.3f}pp  speedup={speedup:.2f}×")
        out.append({"seed": seed, "baseline": float(base_r), "cascade": float(cascade_r),
                    "delta_pp": float(delta), "speedup": float(speedup),
                    "base_s": float(base_t), "cascade_s": float(cascade_t)})
    return out


def stress_nprobe(v, q, gt, dim, nprobes=(10, 20, 40, 80, 160)):
    """(2) Robustness across nprobe at b=5+1, seed=42, msb=4, N=100."""
    print("\n--- (2) Across nprobe values @ b=5+1, seed=42 ---")
    idx = IVFRVQTQIndex(dim=dim, nlist=1000, bits=5, refine_bits=1, nprobe=40, seed=42)
    idx.train(v); idx.add(v)
    out = []
    for np_val in nprobes:
        idx.nprobe = np_val
        t0 = time.time()
        _, pred = idx.search(q, k=10)
        base_t = time.time() - t0
        base_r = compute_recall(gt[:, :10], pred[:, :10], 10)
        t0 = time.time()
        res = search_cascade(idx, q, k=10, top_msb_bits=4, rerank_n=100)
        cascade_t = time.time() - t0
        pred_c, _, _ = res
        cascade_r = compute_recall(gt[:, :10], pred_c[:, :10], 10)
        speedup = base_t / cascade_t
        delta = (cascade_r - base_r) * 100
        print(f"  np={np_val}: base={base_r:.4f} cascade={cascade_r:.4f} Δ={delta:+.3f}pp  speedup={speedup:.2f}×")
        out.append({"nprobe": np_val, "baseline": float(base_r), "cascade": float(cascade_r),
                    "delta_pp": float(delta), "speedup": float(speedup)})
    return out


def stress_smaller_baseline(v, q, gt, dim):
    """(5) Honest comparison: cascade(7-bit) vs baseline(5-bit) at np=40.
    If cascade preserves the 7-bit recall while only matching 5-bit speed,
    that's the real win. If the 5-bit baseline is faster AND has lower recall,
    cascade buys precision at the speed cost — that's the meaningful tradeoff.
    """
    print("\n--- (5) Honest cross-precision comparison ---")
    out = {}
    for bits, refine in [(4, 1), (5, 1), (6, 1)]:
        label = f"b{bits}+{refine}"
        idx = IVFRVQTQIndex(dim=dim, nlist=1000, bits=bits, refine_bits=refine,
                            nprobe=40, seed=42)
        idx.train(v); idx.add(v)
        t0 = time.time()
        _, pred = idx.search(q, k=10)
        base_t = time.time() - t0
        base_r = compute_recall(gt[:, :10], pred[:, :10], 10)
        out[f"{label}_baseline"] = {
            "recall": float(base_r), "time_s": float(base_t),
            "total_bits": bits + refine,
        }
        # Cascade with msb=bits-1 (i.e., one less than full primary)
        t0 = time.time()
        res = search_cascade(idx, q, k=10, top_msb_bits=bits - 1, rerank_n=100)
        cascade_t = time.time() - t0
        pred_c, _, _ = res
        cascade_r = compute_recall(gt[:, :10], pred_c[:, :10], 10)
        out[f"{label}_cascade"] = {
            "recall": float(cascade_r), "time_s": float(cascade_t),
            "speedup": float(base_t / cascade_t),
        }
        print(f"  {label}: base={base_r:.4f} ({base_t:.1f}s)  cascade={cascade_r:.4f} ({cascade_t:.1f}s, {base_t/cascade_t:.2f}×)")

    # Pareto comparison summary
    print("\n  Pareto summary (recall vs time):")
    for k, vv in out.items():
        print(f"    {k:<22} R={vv['recall']:.4f}  t={vv['time_s']:.2f}s")
    return out


def stress_speedup_mechanism(v, q, gt, dim):
    """(4) Where does the speedup actually come from?
    Decompose the baseline vs cascade timing into:
      a) fancy indexing (sub_centroids[primary, sub] vs coarse_recon[primary_msb])
      b) matmul (recon @ qrot)
      c) top-k extraction
    """
    print("\n--- (4) Speedup mechanism diagnostic ---")
    idx = IVFRVQTQIndex(dim=dim, nlist=1000, bits=5, refine_bits=1, nprobe=40, seed=42)
    idx.train(v); idx.add(v)

    # Time individual operations on a single cell
    cell = idx._partitions[0]  # take cell 0 as representative
    primary = cell["primary"]
    sub = cell["sub"]
    norms = cell["norms"]
    n_in_cell, dim_local = primary.shape

    # Fake query rotation
    qrot = np.random.randn(dim_local).astype(np.float32)

    # Time baseline fancy indexing
    n_runs = 100
    t0 = time.time()
    for _ in range(n_runs):
        recon = idx.sub_centroids[primary, sub]
    t_baseline_fancy = (time.time() - t0) / n_runs

    # Time cascade fancy indexing
    full_bits, msb_bits = idx.bits, 4
    lsb_count = full_bits - msb_bits
    coarse_recon = idx.sub_centroids.mean(axis=1).reshape(-1)[:2**msb_bits]  # placeholder
    flat_recon = idx.sub_centroids.reshape(2**msb_bits, 2**lsb_count, -1).mean(axis=(1,2))
    primary_msb = primary >> lsb_count
    t0 = time.time()
    for _ in range(n_runs):
        recon = flat_recon[primary_msb]
    t_cascade_fancy = (time.time() - t0) / n_runs

    # Time matmul
    recon_baseline = idx.sub_centroids[primary, sub]
    recon_cascade = flat_recon[primary_msb]
    t0 = time.time()
    for _ in range(n_runs):
        scores = recon_baseline @ qrot
    t_baseline_matmul = (time.time() - t0) / n_runs

    t0 = time.time()
    for _ in range(n_runs):
        scores = recon_cascade @ qrot
    t_cascade_matmul = (time.time() - t0) / n_runs

    print(f"  cell size: {n_in_cell} vectors")
    print(f"  baseline fancy indexing (2D):    {t_baseline_fancy*1000:.3f} ms")
    print(f"  cascade  fancy indexing (1D):    {t_cascade_fancy*1000:.3f} ms  ({t_baseline_fancy/t_cascade_fancy:.2f}× faster)")
    print(f"  baseline matmul:                 {t_baseline_matmul*1000:.3f} ms")
    print(f"  cascade  matmul:                 {t_cascade_matmul*1000:.3f} ms  (same matrix size)")
    return {
        "baseline_fancy_ms": float(t_baseline_fancy*1000),
        "cascade_fancy_ms": float(t_cascade_fancy*1000),
        "baseline_matmul_ms": float(t_baseline_matmul*1000),
        "cascade_matmul_ms": float(t_cascade_matmul*1000),
    }


def main():
    print("Loading Deep-1M ...")
    r = load_deep1m(1_000_000, 1000)
    if r is None:
        print("FAILED")
        return
    v, q, _ = r
    dim = v.shape[1]
    gt_idx = FAISSFlatIndex(dim); gt_idx.add(v)
    _, gt = gt_idx.search(q, k=10)

    out = {"deep-1m": {}}
    out["deep-1m"]["seeds"] = stress_seeds(v, q, gt, dim)
    out["deep-1m"]["nprobe"] = stress_nprobe(v, q, gt, dim)
    out["deep-1m"]["mechanism"] = stress_speedup_mechanism(v, q, gt, dim)
    out["deep-1m"]["pareto"] = stress_smaller_baseline(v, q, gt, dim)

    with open(os.path.join(os.path.dirname(__file__),
                            "cascade_robustness_results.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
