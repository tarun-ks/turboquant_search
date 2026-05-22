"""
Verify cascade search on SIFT-1M and at other bit budgets.
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from turboquant_search.faiss_baselines import FAISSFlatIndex
from turboquant_search.benchmarks import compute_recall
from turboquant_search.datasets import load_sift1m, load_deep1m
from experiments.ivf_rvq_tq import IVFRVQTQIndex
from experiments.cascade_search import search_cascade


def run_dataset(name, loader):
    print(f"\n{'='*60}\n  {name.upper()}\n{'='*60}")
    r = loader()
    if r is None:
        return {}
    v, q, _ = r
    dim = v.shape[1]
    print(f"  loaded n={v.shape[0]}, dim={dim}")

    gt_idx = FAISSFlatIndex(dim); gt_idx.add(v)
    _, gt = gt_idx.search(q, k=10)

    out = {}
    # Test cascade at three bit budgets: (4+1)=5, (5+1)=6, (6+1)=7
    for bits, refine in [(4, 1), (5, 1), (6, 1)]:
        label = f"b{bits}+{refine}"
        print(f"\n  Building {label} ...")
        idx = IVFRVQTQIndex(dim=dim, nlist=1000, bits=bits,
                            refine_bits=refine, nprobe=40, seed=42)
        idx.train(v)
        idx.add(v)

        # Baseline
        idx.nprobe = 40
        t0 = time.time()
        _, pred = idx.search(q, k=10)
        base_t = time.time() - t0
        base_recall = compute_recall(gt[:, :10], pred[:, :10], 10)
        out[f"{label}_baseline"] = {
            "recall_at_10": float(base_recall),
            "search_s": round(base_t, 2),
            "total_bits": bits + refine,
        }
        print(f"    baseline np=40: R@10={base_recall:.4f} ({base_t:.1f}s)")

        # Cascade with several msb / N combos
        # Pick top_msb_bits = bits - 1 or bits - 2 for the "coarse" pass
        for top_msb in [bits - 2, bits - 1]:
            if top_msb < 2:
                continue
            for N in [50, 100, 200]:
                t0 = time.time()
                res = search_cascade(idx, q, k=10, top_msb_bits=top_msb, rerank_n=N)
                if res is None:
                    continue
                pred, p1, p2 = res
                t = time.time() - t0
                recall = compute_recall(gt[:, :10], pred[:, :10], 10)
                speedup = base_t / max(t, 1e-6)
                out[f"{label}_msb{top_msb}_N{N}"] = {
                    "recall_at_10": float(recall),
                    "total_s": round(t, 2),
                    "pass1_s": round(p1, 2),
                    "pass2_s": round(p2, 2),
                    "speedup_vs_baseline": round(speedup, 2),
                }
                print(f"    cascade msb={top_msb} N={N}: R@10={recall:.4f} "
                      f"({t:.1f}s, {speedup:.2f}× speedup)  Δ={(recall-base_recall)*100:+.2f}pp")
    return out


def main():
    results = {}
    for name, loader in [
        ("deep-1m", lambda: load_deep1m(1_000_000, 1000)),
        ("sift-1m", lambda: load_sift1m(1_000_000, 1000)),
    ]:
        results[name] = run_dataset(name, loader)
        with open(os.path.join(os.path.dirname(__file__),
                                "cascade_verify_results.json"), "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
