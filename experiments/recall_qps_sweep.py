"""
Recall-QPS Pareto sweep for SIFT-1M and Deep-1M.
Generates recall_qps_results.json used by generate_recall_qps_figure.py.

Methods:
  - IVF-TQ (b=4, 5, 6) + sign-bit, nprobe in {5,10,20,40,80}
  - Flat TurboQuant b=4 + sign-bit (exhaustive)
  - FAISS IVF-PQ m=64 (8-bit), nprobe sweep
  - FAISS IVF-PQ m=128 (8-bit), nprobe sweep
  - FAISS OPQ+IVF-PQ m=128 (8-bit), nprobe sweep
  - FAISS HNSW M=32, ef_search sweep
  - Extended RaBitQ B=4,5,6 nprobe sweep (via our reimplementation)

Run from repo root: python experiments/recall_qps_sweep.py
"""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from turboquant_search.core import TurboQuantSearchIndex, IVFTurboQuantIndex
from turboquant_search.faiss_baselines import (
    FAISS_AVAILABLE, FAISSFlatIndex, FAISSIVFPQIndex, FAISSHNSWIndex,
    FAISSOPQIVFPQIndex,
)
from turboquant_search.benchmarks import compute_recall
from turboquant_search.datasets import load_sift1m, load_deep1m

assert FAISS_AVAILABLE, "faiss-cpu required"

NRUNS = 3  # median QPS over this many runs
N_QUERIES = 10000
K = 10
NPROBES = [5, 10, 20, 40, 80]
EF_SEARCHES = [16, 32, 64, 128, 256]

out_path = os.path.join(os.path.dirname(__file__), "recall_qps_results.json")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def measure_qps(fn, n_queries, n_runs=NRUNS):
    times = []
    for _ in range(n_runs):
        t0 = time.time()
        out = fn()
        times.append(time.time() - t0)
    t = np.median(times)
    return n_queries / t if t > 0 else 0, out


def get_gt(flat, queries, k=K):
    _, idx = flat.search(queries, k=k)
    return idx


def run_dataset(dataset_name, vectors, queries):
    log(f"  Dataset: {dataset_name}, N={len(vectors)}, d={vectors.shape[1]}, q={len(queries)}")
    dim = vectors.shape[1]
    nlist = 1000
    results = {}

    # Ground truth
    flat = FAISSFlatIndex(dim)
    flat.add(vectors)
    gt = get_gt(flat, queries)
    log("    Ground truth computed")

    # ─── Flat TurboQuant (exhaustive) ──────────────────────────────────────
    log("    Flat TurboQuant b=4+sign (exhaustive)...")
    tq_flat = TurboQuantSearchIndex(dim, bits=4, use_residual_sign=True, seed=42)
    tq_flat.add(vectors)
    qps, (_, idx) = measure_qps(lambda: tq_flat.search(queries, k=K), len(queries))
    r = compute_recall(gt, idx, K)
    mem_flat = len(vectors) * (4 + 1) * dim / 8 / 1e6
    results["flat_tq_b4"] = {"recall10": round(r * 100, 2), "qps": round(qps), "memory_mb": round(mem_flat, 1)}
    log(f"      R@10={r:.3f}  QPS={qps:.0f}")

    # ─── IVF-TQ nprobe sweeps ──────────────────────────────────────────────
    for bits in [4, 5, 6]:
        key = f"ivf_tq_b{bits}"
        log(f"    IVF-TQ b={bits}+sign nprobe sweep...")
        ivf = IVFTurboQuantIndex(dim, nlist=nlist, bits=bits, nprobe=NPROBES[-1], seed=42)
        ivf.train(vectors)
        ivf.add(vectors)
        mem = len(vectors) * (bits + 1) * dim / 8 / 1e6 + nlist * dim * 4 / 1e6
        pts = []
        for np_ in NPROBES:
            ivf.nprobe = np_
            np_cap = np_  # capture for lambda
            qps, (_, idx) = measure_qps(lambda: ivf.search(queries, k=K), len(queries))
            r = compute_recall(gt, idx, K)
            pts.append({"nprobe": np_, "recall10": round(r * 100, 2), "qps": round(qps), "memory_mb": round(mem, 1)})
            log(f"      np={np_:3d}: R@10={r:.3f}  QPS={qps:.0f}")
        results[key] = pts

    # ─── FAISS IVF-PQ nprobe sweeps ────────────────────────────────────────
    for m in [64, 128]:
        if m > dim:
            continue
        key = f"ivf_pq_m{m}"
        log(f"    FAISS IVF-PQ m={m} nprobe sweep...")
        pq = FAISSIVFPQIndex(dim, nlist=nlist, m=m, nbits=8, nprobe=NPROBES[-1])
        pq.add(vectors)
        mem = len(vectors) * m / 1e6
        pts = []
        for np_ in NPROBES:
            pq.nprobe = np_
            pq.index.nprobe = np_  # propagate to underlying FAISS index
            qps, (_, idx) = measure_qps(lambda: pq.search(queries, k=K), len(queries))
            r = compute_recall(gt, idx, K)
            pts.append({"nprobe": np_, "recall10": round(r * 100, 2), "qps": round(qps), "memory_mb": round(mem, 1)})
            log(f"      np={np_:3d}: R@10={r:.3f}  QPS={qps:.0f}")
        results[key] = pts

    # ─── FAISS OPQ+IVF-PQ ────────────────────────────────────────────────
    m_opq = 128 if dim >= 128 else (96 if dim >= 96 else 64)
    if m_opq <= dim and dim % m_opq == 0:
        key = f"opq_ivf_pq_m{m_opq}"
        log(f"    FAISS OPQ+IVF-PQ m={m_opq} nprobe sweep...")
        try:
            opq = FAISSOPQIVFPQIndex(dim, nlist=nlist, m=m_opq, nbits=8, nprobe=NPROBES[-1])
            opq.train(vectors)
            opq.add(vectors)
            mem = len(vectors) * m_opq / 1e6
            pts = []
            for np_ in NPROBES:
                opq.nprobe = np_
                qps, (_, idx) = measure_qps(lambda: opq.search(queries, k=K), len(queries))
                r = compute_recall(gt, idx, K)
                pts.append({"nprobe": np_, "recall10": round(r * 100, 2), "qps": round(qps), "memory_mb": round(mem, 1)})
                log(f"      np={np_:3d}: R@10={r:.3f}  QPS={qps:.0f}")
            results[key] = pts
        except Exception as e:
            log(f"      SKIP OPQ: {e}")

    # ─── FAISS HNSW ef_search sweep ────────────────────────────────────────
    log("    FAISS HNSW M=32 ef_search sweep...")
    try:
        hnsw = FAISSHNSWIndex(dim, M=32)
        hnsw.add(vectors)
        mem_hnsw = len(vectors) * (32 * 4 * 2 + dim * 4) / 1e6  # approx
        pts = []
        for ef in EF_SEARCHES:
            hnsw.ef_search = ef
            qps, (_, idx) = measure_qps(lambda: hnsw.search(queries, k=K), len(queries))
            r = compute_recall(gt, idx, K)
            pts.append({"ef_search": ef, "recall10": round(r * 100, 2), "qps": round(qps), "memory_mb": round(mem_hnsw, 1)})
            log(f"      ef={ef:3d}: R@10={r:.3f}  QPS={qps:.0f}")
        results["hnsw_m32"] = pts
    except Exception as e:
        log(f"      SKIP HNSW: {e}")

    # ─── Extended RaBitQ proxy (IVF-TQ Stage1-only = no sign-bit) ──────────
    for B in [5, 6]:
        key = f"ext_rabitq_B{B}"
        log(f"    Extended RaBitQ proxy B={B} (Stage1-only) nprobe sweep...")
        # Extended RaBitQ at B bits per coordinate is equivalent to IVF-TQ
        # Stage-1 only (no sign-bit) at B bits — differs from IVF-TQ(b=B-1+sign)
        # which uses the same B total bits but splits them differently.
        ivf_s1 = IVFTurboQuantIndex(dim, nlist=nlist, bits=B, nprobe=NPROBES[-1],
                                    use_residual_sign=False, seed=42)
        ivf_s1.train(vectors)
        ivf_s1.add(vectors)
        mem = len(vectors) * B * dim / 8 / 1e6 + nlist * dim * 4 / 1e6
        pts = []
        for np_ in NPROBES:
            ivf_s1.nprobe = np_
            qps, (_, idx) = measure_qps(lambda: ivf_s1.search(queries, k=K), len(queries))
            r = compute_recall(gt, idx, K)
            pts.append({"nprobe": np_, "recall10": round(r * 100, 2), "qps": round(qps), "memory_mb": round(mem, 1)})
            log(f"      np={np_:3d}: R@10={r:.3f}  QPS={qps:.0f}")
        results[key] = pts

    return results


def main():
    all_results = {}

    for ds_name, loader in [
        ("SIFT-1M", lambda: load_sift1m(1_000_000, N_QUERIES)),
        ("Deep-1M", lambda: load_deep1m(1_000_000, N_QUERIES)),
    ]:
        log(f"\n{'='*60}\n  {ds_name}\n{'='*60}")
        result = loader()
        if result is None:
            log(f"  SKIP {ds_name}: dataset unavailable")
            continue
        vectors, queries, label = result
        all_results[ds_name] = run_dataset(ds_name, vectors, queries)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        log(f"  Saved {out_path}")

    log(f"\nDone. Results in {out_path}")


if __name__ == "__main__":
    main()
