"""
Extended RaBitQ baseline (Gao, Gou & Xu, SIGMOD 2025) at million scale.

This is the baseline reviewers will demand. The argument in the main paper
(Section 7, Related Work) is that Extended RaBitQ at B>=2 is operationally
equivalent to TQ Stage 1 at the same b-bit budget for d >= 64, because the
rotated-unit-vector marginal Beta((d-1)/2,(d-1)/2) is within O(1/d) Kolmogorov
distance of N(0, 1/d). We measure that equivalent configuration here:

    Extended RaBitQ at B bits/dim   <=>   IVF-TQ Stage 1 only (use_residual_sign=False)

We compare three configurations at MATCHED memory:
  (a) Extended RaBitQ proxy = IVF-TQ Stage 1 only at (b+1) bits/dim
  (b) IVF-TQ Stage 1 + sign-bit (ours) at b bits Stage 1 + 1 bit Stage 2 = (b+1) bits/dim
  (c) IVF-PQ baseline at matched memory

The (a) vs (b) comparison directly measures the contribution of sign-bit refinement
on top of Extended RaBitQ-equivalent Stage 1 at IDENTICAL bit budget (and IDENTICAL
memory).

Datasets: SIFT-1M, Deep-1M, GloVe-1M (the three the benchmark suite reports).

Reproduces: Table 1 row "Extended RaBitQ" and the +X.X pp delta vs sign-bit.

Usage:
    python experiments/extended_rabitq_baseline.py
    # Output: extended_rabitq_results.json

Caveat: For full fidelity (especially at d < 64 or for the unbiased IP estimator
detail), substitute the official Extended RaBitQ implementation. The per-coordinate
marginal-density argument supports the proxy at million scale on SIFT-1M (d=128),
Deep-1M (d=96), and GloVe-1M (d=100).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "code"))

from turboquant_search.core import IVFTurboQuantIndex
from turboquant_search.faiss_baselines import FAISSFlatIndex, FAISSIVFPQIndex
from turboquant_search.benchmarks import compute_recall
from turboquant_search.datasets import load_sift1m, load_deep1m, load_glove100


def _load_glove1m(n_vectors: int = 1_000_000, n_queries: int = 1000):
    # GloVe-100-angular ships >1M rows; load_glove100 takes the n_vectors slice.
    return load_glove100(n_vectors, n_queries)


DATASETS = [
    ("SIFT-1M", load_sift1m, 128),
    ("Deep-1M", load_deep1m, 96),
    ("GloVe-1M", _load_glove1m, 100),
]

NLIST = 1000
NPROBE = 20
N_QUERIES = 1000
SEED = 42


def _build_groundtruth(vectors: np.ndarray, queries: np.ndarray, k: int = 10) -> np.ndarray:
    flat = FAISSFlatIndex(vectors.shape[1])
    flat.add(vectors)
    _, gt = flat.search(queries, k=k)
    return gt


def _run_extended_rabitq_proxy(vectors, queries, gt, *, bits, label: str):
    """Stage 1 only — Extended RaBitQ-equivalent at `bits` bits/dim."""
    dim = vectors.shape[1]
    t0 = time.time()
    idx = IVFTurboQuantIndex(
        dim=dim,
        bits=bits,
        use_residual_sign=False,  # disable Stage 2 -> Extended RaBitQ-equivalent
        nlist=NLIST,
        nprobe=NPROBE,
        seed=SEED,
    )
    idx.train(vectors)
    idx.add(vectors)
    train_add = time.time() - t0

    t0 = time.time()
    _, found = idx.search(queries, k=10)
    qps = N_QUERIES / max(time.time() - t0, 1e-6)
    recall = compute_recall(gt, found, 10)
    mem_mb = idx.memory_bytes / (1024 * 1024)

    return {
        "label": label,
        "bits_stage1": bits,
        "bits_stage2": 0,
        "bits_total": bits,
        "recall10": round(recall * 100, 2),
        "qps": round(qps, 0),
        "memory_mb": round(mem_mb, 1),
        "build_s": round(train_add, 1),
    }


def _run_ivf_tq_signbit(vectors, queries, gt, *, bits, label: str):
    """IVF-TQ with Stage 2 enabled (the proposed method) at `bits`+1 bits/dim total."""
    dim = vectors.shape[1]
    t0 = time.time()
    idx = IVFTurboQuantIndex(
        dim=dim,
        bits=bits,
        use_residual_sign=True,
        nlist=NLIST,
        nprobe=NPROBE,
        seed=SEED,
    )
    idx.train(vectors)
    idx.add(vectors)
    train_add = time.time() - t0

    t0 = time.time()
    _, found = idx.search(queries, k=10)
    qps = N_QUERIES / max(time.time() - t0, 1e-6)
    recall = compute_recall(gt, found, 10)
    mem_mb = idx.memory_bytes / (1024 * 1024)

    return {
        "label": label,
        "bits_stage1": bits,
        "bits_stage2": 1,
        "bits_total": bits + 1,
        "recall10": round(recall * 100, 2),
        "qps": round(qps, 0),
        "memory_mb": round(mem_mb, 1),
        "build_s": round(train_add, 1),
    }


def run_dataset(name, loader, dim):
    print(f"\n=== {name} (dim={dim}) ===", flush=True)
    print("Loading...", flush=True)
    res = loader(n_vectors=1_000_000, n_queries=N_QUERIES)
    if isinstance(res, tuple) and len(res) == 3:
        vectors, queries, _ = res
    else:
        vectors, queries = res

    print(f"Vectors {vectors.shape}, Queries {queries.shape}. Computing ground truth...", flush=True)
    gt = _build_groundtruth(vectors, queries, k=10)

    rows = []
    # Match memory: Extended RaBitQ at (b+1) bits = IVF-TQ Stage 1 at b bits + sign bit.
    # We sweep b in {3, 4, 5} (i.e., total bits 4, 5, 6).
    for b in (3, 4, 5):
        print(f"  -- bits_total={b+1} --", flush=True)

        rabitq_proxy = _run_extended_rabitq_proxy(
            vectors, queries, gt,
            bits=b + 1,
            label=f"Extended RaBitQ (proxy, B={b+1})",
        )
        print(f"    {rabitq_proxy['label']}: R@10={rabitq_proxy['recall10']:.2f}%  "
              f"{rabitq_proxy['memory_mb']:.0f} MB  {rabitq_proxy['qps']:.0f} QPS",
              flush=True)
        rows.append(rabitq_proxy)

        ours = _run_ivf_tq_signbit(
            vectors, queries, gt,
            bits=b,
            label=f"IVF-TQ (ours, b={b}+sign)",
        )
        print(f"    {ours['label']}: R@10={ours['recall10']:.2f}%  "
              f"{ours['memory_mb']:.0f} MB  {ours['qps']:.0f} QPS",
              flush=True)
        rows.append(ours)

        delta = ours["recall10"] - rabitq_proxy["recall10"]
        print(f"    DELTA (ours - Ext.RaBitQ proxy): {delta:+.2f}pp", flush=True)

    return {"dataset": name, "dim": dim, "nlist": NLIST, "nprobe": NPROBE,
            "n_queries": N_QUERIES, "seed": SEED, "rows": rows}


if __name__ == "__main__":
    out_path = Path(__file__).parent / "extended_rabitq_results.json"
    all_results = []
    for name, loader, dim in DATASETS:
        try:
            all_results.append(run_dataset(name, loader, dim))
        except Exception as e:  # noqa: BLE001
            print(f"FAILED on {name}: {e}", flush=True)
            all_results.append({"dataset": name, "error": str(e)})

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)

    # Summary across datasets
    print("\n=== SUMMARY: sign-bit advantage over Extended RaBitQ proxy at matched memory ===")
    for ds in all_results:
        if "rows" not in ds:
            continue
        rows = ds["rows"]
        for i in range(0, len(rows), 2):
            rab = rows[i]
            ours = rows[i + 1]
            delta = ours["recall10"] - rab["recall10"]
            print(f"  {ds['dataset']:10s} bits_total={ours['bits_total']}  "
                  f"Ext.RaBitQ={rab['recall10']:.1f}%  ours={ours['recall10']:.1f}%  "
                  f"Δ={delta:+.2f}pp")
