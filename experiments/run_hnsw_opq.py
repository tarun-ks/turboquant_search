"""
Run FAISS HNSW and OPQ+IVF-PQ baselines for Table 1 of the NeurIPS 2026
submission.

Scope:
    SIFT-1M  (dim=128) — HNSW M={16,32} ef_s sweep; OPQ m={64,128} nprobe sweep
    Deep-1M  (dim=96)  — same, OPQ m auto-adjusts to {48,96}
    Deep-10M (dim=96)  — single representative config of each (build time
                        for the full sweep at 10M is hours)

Results are merged into experiments/paper_results.json under keys
    hnsw_<dataset>  and  opq_<dataset>
to leave the existing IVF-TQ / IVF-PQ numbers untouched.

Ground truth at 10M is cached to experiments/cache/deep10m_gt.npy.

Usage:
    python experiments/run_hnsw_opq.py [--scale 1m|10m|all]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.environ["TQS_THREADS"] = str(os.cpu_count() or 1)

from turboquant_search.faiss_baselines import (
    FAISS_AVAILABLE, FAISSFlatIndex, FAISSHNSWIndex, FAISSOPQIVFPQIndex,
)
from turboquant_search.benchmarks import compute_recall
from turboquant_search.datasets import load_sift1m, load_deep1m

assert FAISS_AVAILABLE, "faiss-cpu is required: pip install faiss-cpu"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def measure_qps(search_fn, n_queries, n_runs=3):
    times = []
    out = None
    for _ in range(n_runs):
        t0 = time.time()
        out = search_fn()
        times.append(time.time() - t0)
    t = float(np.median(times))
    return n_queries / t if t > 0 else 0.0, t, out


def normalize(v):
    v = np.ascontiguousarray(v.astype(np.float32))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(norms, 1e-8)


def load_deep10m():
    """Mirrors experiments/scale_10m.py loader. Caches normalized arrays
    and ground truth to experiments/cache/."""
    cache = ROOT / "experiments" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    vec_path = cache / "deep10m_vectors.npy"
    qry_path = cache / "deep10m_queries.npy"
    gt_path = cache / "deep10m_gt.npy"

    if vec_path.exists() and qry_path.exists():
        vectors = np.load(vec_path, mmap_mode="r")
        queries = np.load(qry_path)
    else:
        log("Downloading Deep-10M from HuggingFace (one-time, ~3GB)...")
        from datasets import load_dataset
        train_ds = load_dataset("open-vdb/deep-image-96-angular", "train", split="train")
        test_ds = load_dataset("open-vdb/deep-image-96-angular", "test", split="test[:10000]")
        vectors = np.array(train_ds["emb"], dtype=np.float32)
        queries = np.array(test_ds["emb"], dtype=np.float32)
        vectors = normalize(vectors)
        queries = normalize(queries)
        np.save(vec_path, vectors)
        np.save(qry_path, queries)
        vectors = np.load(vec_path, mmap_mode="r")
        queries = np.load(qry_path)

    if gt_path.exists():
        gt = np.load(gt_path)
    else:
        log("Computing Deep-10M ground truth (FAISS Flat, ~10-20 min)...")
        flat = FAISSFlatIndex(vectors.shape[1])
        flat.add(np.asarray(vectors))
        t0 = time.time()
        _, gt = flat.search(queries, k=10)
        log(f"  GT computed in {time.time()-t0:.1f}s")
        np.save(gt_path, gt)

    return np.asarray(vectors), queries, gt


def run_hnsw_sweep(vectors, queries, gt, label, M_values, ef_search_values,
                   ef_construction=200, raw_mb_for_total=None):
    log(f"  HNSW sweep on {label}: M ∈ {M_values}, ef_s ∈ {ef_search_values}")
    nq = queries.shape[0]
    out = {}
    raw_vectors_mb = vectors.shape[0] * vectors.shape[1] * 4 / (1024 * 1024)
    for M in M_values:
        idx = FAISSHNSWIndex(vectors.shape[1], M=M, ef_construction=ef_construction,
                              ef_search=ef_search_values[0])
        t0 = time.time()
        idx.add(np.asarray(vectors))
        build = time.time() - t0
        log(f"    M={M} build={build:.1f}s mem={idx.memory_bytes/(1024*1024):.0f}MB")
        sub = {}
        for ef in ef_search_values:
            idx.ef_search = ef
            idx.search(queries[:10], k=10)  # warmup
            qps, t, (_, I) = measure_qps(lambda: idx.search(queries, k=10), nq)
            r = compute_recall(gt, I, 10)
            mem_mb = idx.memory_bytes / (1024 * 1024)
            entry = {
                "recall10": round(r * 100, 1),
                "qps": round(qps),
                "latency_ms": round(t * 1000, 1),
                "memory_mb": round(mem_mb, 1),
                "total_memory_mb": round(mem_mb, 1),  # HNSW already includes raw vectors
                "compression": "1.0x",
                "training": "None",
            }
            sub[f"ef{ef}"] = entry
            log(f"      ef={ef:3d}: R@10={r:.1%}  {qps:.0f} QPS  {t*1000:.1f}ms  {mem_mb:.0f}MB")
        out[f"M{M}"] = sub
        # Free graph before building next M
        del idx
        import gc; gc.collect()
    return out


def run_opq_sweep(vectors, queries, gt, label, m_values, nprobe_values,
                   nlist=1000, raw_mb_for_total=None):
    log(f"  OPQ+IVF-PQ sweep on {label}: m ∈ {m_values}, nprobe ∈ {nprobe_values}")
    nq = queries.shape[0]
    dim = vectors.shape[1]
    raw_vectors_mb = vectors.shape[0] * dim * 4 / (1024 * 1024)

    # Adjust m to divide dim evenly (matches FAISSOPQIVFPQIndex behavior).
    out = {}
    for m_target in m_values:
        m_eff = m_target
        while dim % m_eff != 0 and m_eff > 1:
            m_eff -= 1
        log(f"    m={m_target} -> effective m={m_eff}")
        idx = FAISSOPQIVFPQIndex(dim, nlist=nlist, m=m_target, nbits=8,
                                  nprobe=nprobe_values[0], opq_iter=25)
        t0 = time.time()
        idx.add(np.asarray(vectors))
        build = time.time() - t0
        log(f"      build={build:.1f}s mem={idx.memory_bytes/(1024*1024):.1f}MB")
        sub = {}
        for nprobe in nprobe_values:
            idx.nprobe = nprobe
            idx._ivfpq.nprobe = min(nprobe, idx._nlist)
            idx.search(queries[:10], k=10)  # warmup
            qps, t, (_, I) = measure_qps(lambda: idx.search(queries, k=10), nq)
            r = compute_recall(gt, I, 10)
            mem_mb = idx.memory_bytes / (1024 * 1024)
            entry = {
                "recall10": round(r * 100, 1),
                "qps": round(qps),
                "latency_ms": round(t * 1000, 1),
                "memory_mb": round(mem_mb, 1),
                "total_memory_mb": round(mem_mb + raw_vectors_mb, 1),
                "compression": f"{(raw_vectors_mb / mem_mb):.1f}x",
                "training": "OPQ + PQ codebook + IVF k-means",
            }
            sub[f"np{nprobe}"] = entry
            log(f"      np={nprobe:>3}: R@10={r:.1%}  {qps:.0f} QPS  {mem_mb:.0f}MB")
        out[f"m{m_eff}"] = sub
        del idx
        import gc; gc.collect()
    return out


def run_1m_scale(out: dict):
    nprobes = [5, 10, 20, 40, 80, 160]
    Ms = [16, 32]
    efs = [16, 32, 64, 128, 256]

    for ds_name, loader, m_values in [
        ("SIFT-1M", load_sift1m, [64, 128]),
        ("Deep-1M", load_deep1m, [64, 128]),  # auto-adjusts to 48, 96
    ]:
        log(f"=== {ds_name} ===")
        result = loader(n_vectors=1000000, n_queries=10000)
        if result is None:
            log(f"  SKIPPED {ds_name} (download failed)")
            continue
        vectors, queries, label = result
        log(f"  {label}")
        flat = FAISSFlatIndex(vectors.shape[1])
        flat.add(vectors)
        _, gt = flat.search(queries, k=10)

        out[f"hnsw_{ds_name}"] = run_hnsw_sweep(
            vectors, queries, gt, ds_name, Ms, efs)
        out[f"opq_{ds_name}"] = run_opq_sweep(
            vectors, queries, gt, ds_name, m_values, nprobes)


def run_10m_scale(out: dict):
    log("=== Deep-10M (HNSW+OPQ — single configs) ===")
    vectors, queries, gt = load_deep10m()
    log(f"  Deep-10M: {vectors.shape[0]:,} vectors, dim={vectors.shape[1]}, "
        f"{queries.shape[0]} queries")

    # HNSW M=32 with limited ef sweep (each build is ~30 min at 10M).
    out["hnsw_Deep-10M"] = run_hnsw_sweep(
        vectors, queries, gt, "Deep-10M",
        M_values=[32], ef_search_values=[32, 64, 128, 256])

    # OPQ at m=96 (dim=96 → matched-bit comparable to IVF-PQ m=96 ceiling)
    out["opq_Deep-10M"] = run_opq_sweep(
        vectors, queries, gt, "Deep-10M",
        m_values=[96], nprobe_values=[10, 20, 40, 80])


def merge_results(new_results: dict):
    path = ROOT / "experiments" / "paper_results.json"
    with open(path) as f:
        data = json.load(f)
    for k, v in new_results.items():
        data[k] = v
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    log(f"Merged {len(new_results)} new keys into {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=["1m", "10m", "all"], default="1m")
    args = ap.parse_args()

    out = {}
    if args.scale in ("1m", "all"):
        run_1m_scale(out)
    if args.scale in ("10m", "all"):
        run_10m_scale(out)

    merge_results(out)


if __name__ == "__main__":
    main()
