"""
Evaluate RH-IVF-TQ (fully data-independent ANN) against k-means IVF-TQ
and IVF-PQ on:
    1. Deep-1M static recall sweep (n_hyperplanes ∈ {8, 10, 12})
    2. Deep-1M with random-rotation distribution shift
       (the killer test: does data-independent partition stay flat
       while learned partition degrades?)

Outputs:
    experiments/rh_lsh_static_results.json
    experiments/rh_lsh_shift_results.json
"""

import gc, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.environ["TQS_THREADS"] = str(os.cpu_count() or 1)

from turboquant_search.core import IVFTurboQuantIndex
from turboquant_search.rh_lsh import RHLSHIVFTurboQuantIndex
from turboquant_search.faiss_baselines import FAISS_AVAILABLE, FAISSFlatIndex
from turboquant_search.benchmarks import compute_recall
from turboquant_search.datasets import load_deep1m
import faiss


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def normalize(v):
    v = np.ascontiguousarray(v.astype(np.float32))
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(n, 1e-8)


def run_static():
    log("=== Deep-1M static recall: RH-IVF-TQ vs IVF-TQ vs IVF-PQ ===")
    res = load_deep1m(n_vectors=1_000_000, n_queries=5_000)
    vectors_raw, queries_raw, _ = res
    vectors = normalize(vectors_raw); queries = normalize(queries_raw)
    dim = vectors.shape[1]
    flat = FAISSFlatIndex(dim); flat.add(vectors)
    _, gt = flat.search(queries, 10); del flat; gc.collect()

    out = {}

    # IVF-TQ baseline at nlist=1024 ≈ 2^10
    log("  IVF-TQ k-means, nlist=1024...")
    ivf = IVFTurboQuantIndex(dim, nlist=1024, bits=4, nprobe=20, seed=42)
    ivf.train(vectors); ivf.add(vectors)
    _, I = ivf.search(queries, k=10, rerank=0)
    out["ivf_tq_kmeans_rr0"] = round(compute_recall(gt, I, 10) * 100, 2)
    _, I = ivf.search(queries, k=10, rerank=50)
    out["ivf_tq_kmeans_rr50"] = round(compute_recall(gt, I, 10) * 100, 2)
    log(f"    rr=0: {out['ivf_tq_kmeans_rr0']:.2f}%  rr=50: {out['ivf_tq_kmeans_rr50']:.2f}%")
    del ivf; gc.collect()

    # RH-IVF-TQ at L ∈ {8, 10, 12} (cells = 256, 1024, 4096)
    for L in [8, 10, 12]:
        log(f"  RH-IVF-TQ L={L} (2^{L} = {2**L} cells)...")
        rh = RHLSHIVFTurboQuantIndex(dim, n_hyperplanes=L, bits=4, nprobe=20, seed=42)
        rh.add(vectors)
        for nprobe in [10, 20, 40, 80]:
            rh.nprobe = nprobe
            _, I = rh.search(queries, k=10, rerank=0)
            r0 = round(compute_recall(gt, I, 10) * 100, 2)
            _, I = rh.search(queries, k=10, rerank=50)
            r50 = round(compute_recall(gt, I, 10) * 100, 2)
            out[f"rh_L{L}_np{nprobe}_rr0"]  = r0
            out[f"rh_L{L}_np{nprobe}_rr50"] = r50
            log(f"    L={L} np={nprobe}: rr0={r0:.2f}%  rr50={r50:.2f}%  cells_occupied={rh.n_occupied_cells}")
        del rh; gc.collect()

    json.dump(out, open(ROOT / "experiments" / "rh_lsh_static_results.json", "w"), indent=2)
    log("  saved rh_lsh_static_results.json")


def run_shift():
    log("=== Deep-1M shift: RH-IVF-TQ vs IVF-TQ frozen under rotation ===")
    res = load_deep1m(n_vectors=1_000_000, n_queries=5_000)
    vectors_raw, queries_raw, _ = res
    vectors = normalize(vectors_raw); queries = normalize(queries_raw)
    dim = vectors.shape[1]

    rng = np.random.RandomState(2026)
    G = rng.randn(dim, dim).astype(np.float32)
    R, _ = np.linalg.qr(G)

    init = vectors[:200_000]
    new = vectors[200_000:1_000_000] @ R
    queries_B = queries @ R

    # All indexes start with init (in space A)
    log("  Building IVF-TQ frozen (k-means)...")
    ivf = IVFTurboQuantIndex(dim, nlist=1024, bits=4, nprobe=20, seed=42)
    ivf.train(init); ivf.add(init)

    log("  Building RH-IVF-TQ (no train)...")
    rh = RHLSHIVFTurboQuantIndex(dim, n_hyperplanes=10, bits=4, nprobe=20, seed=42)
    rh.add(init)

    rows = []

    def measure(step, db):
        flat = FAISSFlatIndex(dim); flat.add(db)
        _, gt = flat.search(queries_B, 10); del flat; gc.collect()
        _, I_kf  = ivf.search(queries_B, k=10, rerank=0)
        _, I_kfr = ivf.search(queries_B, k=10, rerank=50)
        _, I_rh  = rh.search(queries_B, k=10, rerank=0)
        _, I_rhr = rh.search(queries_B, k=10, rerank=50)
        r = {
            "step": step, "n_indexed": int(db.shape[0]),
            "ivf_tq_frozen_rr0":  round(compute_recall(gt, I_kf, 10) * 100, 2),
            "ivf_tq_frozen_rr50": round(compute_recall(gt, I_kfr, 10) * 100, 2),
            "rh_ivf_tq_rr0":      round(compute_recall(gt, I_rh, 10) * 100, 2),
            "rh_ivf_tq_rr50":     round(compute_recall(gt, I_rhr, 10) * 100, 2),
        }
        rows.append(r)
        log(f"  {step}: kmeans_rr0={r['ivf_tq_frozen_rr0']:.1f}  rh_rr0={r['rh_ivf_tq_rr0']:.1f}  "
            f"kmeans_rr50={r['ivf_tq_frozen_rr50']:.1f}  rh_rr50={r['rh_ivf_tq_rr50']:.1f}")

    measure("Initial 200K (A)", init)

    cumulative = [init]
    for i in range(8):
        s = i * 100_000; e = s + 100_000
        batch = new[s:e]
        ivf.add(batch); rh.add(batch)
        cumulative.append(batch)
        all_data = np.concatenate(cumulative)
        measure(f"+{(e // 1000)}K B", all_data)
        del all_data; gc.collect()

    json.dump({"steps": rows}, open(ROOT / "experiments" / "rh_lsh_shift_results.json", "w"), indent=2)
    log("  saved rh_lsh_shift_results.json")


if __name__ == "__main__":
    run_static()
    run_shift()
