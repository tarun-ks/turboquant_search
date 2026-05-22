"""C++-measured cascade robustness: SIFT-1M and Deep-1M, 4 seeds each.

Mirrors cascade_robustness.py (which used IVFRVQTQIndex Python search) but
runs against the production IVFTurboQuantIndex with C++ ivf_search baseline
and the new C++ cascade_search.

Outputs experiments/cascade_robustness_cpp_results.json.
"""

import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from turboquant_search.core import IVFTurboQuantIndex
from turboquant_search.faiss_baselines import FAISSFlatIndex
from turboquant_search.benchmarks import compute_recall
from turboquant_search.datasets import load_sift1m, load_deep1m
from experiments.cascade_search_cpp import search_cascade_cpp


def _measure_once(idx, q, gt, k=10, top_msb_bits=4, rerank_n=100, repeats=3):
    """Time baseline + cascade with `repeats` runs each, taking the min."""
    base_times, casc_times = [], []
    base_pred = casc_pred = None
    for _ in range(repeats):
        t0 = time.time()
        _, base_pred = idx.search(q, k=k)
        base_times.append(time.time() - t0)
        t0 = time.time()
        casc_pred = search_cascade_cpp(idx, q, k=k,
                                        top_msb_bits=top_msb_bits, rerank_n=rerank_n)
        casc_times.append(time.time() - t0)
    base_t = min(base_times)
    casc_t = min(casc_times)
    base_r = compute_recall(gt[:, :k], base_pred[:, :k], k)
    casc_r = compute_recall(gt[:, :k], casc_pred[:, :k], k)
    overlap = np.mean([
        len(set(base_pred[i].tolist()) & set(casc_pred[i].tolist())) / k
        for i in range(q.shape[0])
    ])
    return {
        "baseline_recall": float(base_r),
        "cascade_recall": float(casc_r),
        "delta_pp": float((casc_r - base_r) * 100),
        "baseline_s": float(base_t),
        "cascade_s": float(casc_t),
        "speedup": float(base_t / casc_t),
        "qps_baseline": float(q.shape[0] / base_t),
        "qps_cascade": float(q.shape[0] / casc_t),
        "top10_overlap": float(overlap),
    }


def stress_seeds(name, v, q, gt, dim, seeds=(42, 43, 44, 45),
                 bits=5, nprobe=40, top_msb_bits=4, rerank_n=100, k=10):
    """Cascade vs baseline across 4 random rotation seeds at the canonical setup."""
    print(f"\n--- [{name}] seeds @ b={bits}+sign, np={nprobe}, msb={top_msb_bits}, N={rerank_n} ---")
    out = []
    for seed in seeds:
        idx = IVFTurboQuantIndex(dim=dim, nlist=1000, bits=bits,
                                  nprobe=nprobe, use_residual_sign=True, seed=seed)
        idx.train(v); idx.add(v)
        # Warmup
        _ = idx.search(q[:8], k=10)
        _ = search_cascade_cpp(idx, q[:8], k=10,
                                top_msb_bits=top_msb_bits, rerank_n=rerank_n)
        m = _measure_once(idx, q, gt, k=k,
                           top_msb_bits=top_msb_bits, rerank_n=rerank_n)
        m["seed"] = seed
        out.append(m)
        print(f"  seed={seed}: base={m['baseline_recall']:.4f} cascade={m['cascade_recall']:.4f} "
              f"Δ={m['delta_pp']:+.3f}pp  speedup={m['speedup']:.2f}×  "
              f"({m['qps_baseline']:.0f} → {m['qps_cascade']:.0f} QPS)")
    return out


def stress_nprobe(name, v, q, gt, dim, nprobes=(10, 20, 40, 80, 160),
                   bits=5, top_msb_bits=4, rerank_n=100, k=10):
    """Across nprobe values at seed=42."""
    print(f"\n--- [{name}] across nprobe @ b={bits}+sign, msb={top_msb_bits}, N={rerank_n} ---")
    idx = IVFTurboQuantIndex(dim=dim, nlist=1000, bits=bits,
                              nprobe=40, use_residual_sign=True, seed=42)
    idx.train(v); idx.add(v)
    _ = idx.search(q[:8], k=10)
    _ = search_cascade_cpp(idx, q[:8], k=10,
                            top_msb_bits=top_msb_bits, rerank_n=rerank_n)
    out = []
    for np_val in nprobes:
        idx.nprobe = np_val
        m = _measure_once(idx, q, gt, k=k,
                           top_msb_bits=top_msb_bits, rerank_n=rerank_n)
        m["nprobe"] = np_val
        out.append(m)
        print(f"  np={np_val:>3}: base={m['baseline_recall']:.4f} cascade={m['cascade_recall']:.4f} "
              f"Δ={m['delta_pp']:+.3f}pp  speedup={m['speedup']:.2f}×")
    return out


def main():
    out = {}
    for name, loader in [("deep-1m", lambda: load_deep1m(1_000_000, 1000)),
                          ("sift-1m", lambda: load_sift1m(1_000_000, 1000))]:
        print(f"\n{'='*60}\n{name.upper()}\n{'='*60}")
        r = loader()
        if r is None:
            print(f"  failed to load {name}"); continue
        v, q, _ = r
        dim = v.shape[1]
        print(f"  loaded n={v.shape[0]}, dim={dim}")
        gt_idx = FAISSFlatIndex(dim); gt_idx.add(v)
        _, gt = gt_idx.search(q, k=10)

        out[name] = {
            "seeds": stress_seeds(name, v, q, gt, dim),
            "nprobe": stress_nprobe(name, v, q, gt, dim),
        }
        with open(os.path.join(os.path.dirname(__file__),
                                "cascade_robustness_cpp_results.json"), "w") as f:
            json.dump(out, f, indent=2)

    # Summary
    print(f"\n{'='*60}\nSUMMARY (cascade vs C++ baseline, recall preservation)\n{'='*60}")
    for name in out:
        seeds = out[name]["seeds"]
        deltas = [s["delta_pp"] for s in seeds]
        spds = [s["speedup"] for s in seeds]
        print(f"{name}: 4-seed mean Δ={np.mean(deltas):+.3f}pp (range "
              f"[{min(deltas):+.3f}, {max(deltas):+.3f}])  "
              f"speedup mean={np.mean(spds):.2f}× (range [{min(spds):.2f}, {max(spds):.2f}])")


if __name__ == "__main__":
    main()
