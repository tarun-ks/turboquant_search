"""
10M-scale benchmarks: Deep-10M (96-dim, ~10M vectors).
Addresses reviewer concern about only evaluating at 1M scale.
"""

import numpy as np
import time
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["TQS_THREADS"] = str(os.cpu_count() or 1)

from turboquant_search.core import IVFTurboQuantIndex, FlatSearchIndex
from turboquant_search.faiss_baselines import FAISSFlatIndex, FAISSIVFPQIndex, FAISS_AVAILABLE
from turboquant_search.benchmarks import compute_recall

assert FAISS_AVAILABLE
import faiss

results = {}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def normalize(v):
    v = np.ascontiguousarray(v.astype(np.float32))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(norms, 1e-8)


# ── Load Deep-10M ──
log("Loading Deep-10M from HuggingFace (first run downloads ~3GB)...")
from datasets import load_dataset

train_ds = load_dataset("open-vdb/deep-image-96-angular", "train", split="train")
test_ds = load_dataset("open-vdb/deep-image-96-angular", "test", split="test[:10000]")

vectors = np.array(train_ds["emb"], dtype=np.float32)
queries = np.array(test_ds["emb"], dtype=np.float32)
vectors = normalize(vectors)
queries = normalize(queries)
dim = vectors.shape[1]
n = vectors.shape[0]
nq = queries.shape[0]

log(f"Loaded: {n:,} vectors, {nq} queries, dim={dim}")

# ── Ground truth (brute force on 10M is feasible with FAISS) ──
log("Building ground truth (FAISS Flat on 10M)...")
flat = FAISSFlatIndex(dim)
flat.add(vectors)
t0 = time.time()
_, gt = flat.search(queries, k=10)
gt_time = time.time() - t0
log(f"  Ground truth: {gt_time:.1f}s")

nlist = 3162  # sqrt(10M) ≈ 3162

# ── FAISS IVF-PQ at various m ──
log("\n=== FAISS IVF-PQ baselines ===")
results["faiss"] = {}

vectors_normed = normalize(vectors)
queries_normed = normalize(queries)

for m in [48, 96]:
    log(f"\n  IVF-PQ m={m}:")
    sweep = {}
    quantizer = faiss.IndexFlatIP(dim)
    idx = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)
    idx.train(vectors_normed)
    idx.add(vectors_normed)

    for nprobe in [10, 20, 40, 80, 160]:
        idx.nprobe = nprobe
        times = []
        for _ in range(3):
            t0 = time.time()
            _, res = idx.search(queries_normed, 10)
            times.append(time.time() - t0)
        t = np.median(times)
        r = compute_recall(gt, res, 10)
        mem = (n * m + nlist * dim * 4) / (1024 * 1024)
        sweep[f"np{nprobe}"] = {"recall10": round(r * 100, 1), "qps": round(nq / t), "memory_mb": round(mem, 1)}
        log(f"    np={nprobe:>3}: R@10={r:.1%}  {nq/t:.0f} QPS  {mem:.0f} MB")
    results["faiss"][f"m{m}"] = sweep

# ── IVF-TQ ──
log("\n=== IVF-TQ 4-bit ===")
results["ivf_tq"] = {}

# Free FAISS indexes to reclaim memory for IVF-TQ
import gc
del idx
gc.collect()

ivf = IVFTurboQuantIndex(dim, nlist=nlist, bits=4, nprobe=10, seed=42)
log("  Training k-means on 10M...")
ivf.train(vectors)
log("  Adding vectors in 1M batches (no raw storage)...")
batch_add = 1_000_000
for start in range(0, n, batch_add):
    end = min(start + batch_add, n)
    ivf.add(vectors[start:end])
    ivf._raw_vectors = None  # free raw copy immediately
    import gc; gc.collect()
    log(f"    Added {end//1_000_000}M / {n//1_000_000}M")
log(f"  Index built: {ivf.memory_bytes/(1024*1024):.0f} MB")

for nprobe in [10, 20, 40]:
    ivf.nprobe = nprobe
    ivf.search(queries[:100], k=10)  # warmup
    times = []
    for _ in range(3):
        t0 = time.time()
        _, res = ivf.search(queries, k=10)
        times.append(time.time() - t0)
    t = np.median(times)
    r = compute_recall(gt, res, 10)
    mem = ivf.memory_bytes / (1024 * 1024)
    results["ivf_tq"][f"np{nprobe}"] = {
        "recall10": round(r * 100, 1),
        "qps": round(nq / t),
        "memory_mb": round(mem, 1),
    }
    log(f"  np={nprobe}: R@10={r:.1%}  {nq/t:.0f} QPS  {mem:.0f} MB")

# ── Save ──
out_path = Path(__file__).parent / "scale_10m_results.json"
with open(out_path, "w") as f:
    json.dump({"n_vectors": n, "n_queries": nq, "dim": dim, **results}, f, indent=2)
log(f"\nResults saved to {out_path}")
