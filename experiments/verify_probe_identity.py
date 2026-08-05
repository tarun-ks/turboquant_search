"""
Verify that IVFFlat and IVF-PQ coarse partitions are identical
when trained with the same seed on the same 1M data.

For each seed: trains both indexes, compares centroids, then checks
that all 10K queries probe exactly the same cells at nprobe=20.
"""

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))

import faiss
from streaming_multiseed import (
    set_all_seeds, normalize, make_ivfpq, _load_10m_cached, log,
)

NLIST, NPROBE = 3162, 20
# (dataset, vector_dim, m_pq, bits)
DATASETS = [("sift10m", 128, 64, 10), ("deep10m", 96, 48, 10), ("t2i10m", 200, 100, 10)]
SEEDS = [42, 123, 7777]


def make_ivfflat(dim, nlist, nprobe, seed, train_data):
    set_all_seeds(seed)
    quantizer = faiss.IndexFlatIP(dim)
    idx = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    if hasattr(idx, "cp") and hasattr(idx.cp, "seed"):
        idx.cp.seed = int(seed)
    idx.nprobe = nprobe
    idx.train(train_data)
    return idx


def centroids_of(idx):
    q = faiss.downcast_index(idx.quantizer)
    return q.reconstruct_n(0, idx.nlist)


def probe_sets(idx, queries):
    qc = queries @ centroids_of(idx).T
    top = np.argpartition(-qc, idx.nprobe, axis=1)[:, :idx.nprobe]
    return [frozenset(top[i].tolist()) for i in range(len(queries))]


results = {}
for dataset, dim, m_pq, bits in DATASETS:
    log(f"\n=== {dataset} (dim={dim}) ===")
    base, queries = _load_10m_cached(dataset)
    n1m = 1_000_000
    init_normed = normalize(np.asarray(base[:n1m]))
    queries_normed = normalize(queries)

    for seed in SEEDS:
        flat = make_ivfflat(dim, NLIST, NPROBE, seed, init_normed)
        pq   = make_ivfpq(dim, NLIST, m_pq, bits, NPROBE, seed, init_normed)

        c_flat = centroids_of(flat)
        c_pq   = centroids_of(pq)
        max_diff = float(np.max(np.abs(c_flat - c_pq)))

        ps_flat = probe_sets(flat, queries_normed)
        ps_pq   = probe_sets(pq, queries_normed)
        n_diff  = sum(1 for a, b in zip(ps_flat, ps_pq) if a != b)

        log(f"  seed={seed}: max_centroid_abs_diff={max_diff:.2e}  "
            f"queries_with_different_probe_sets={n_diff}/{len(queries_normed)}")
        results[(dataset, seed)] = {"max_diff": max_diff, "n_probe_diff": n_diff}

print("\n=== SUMMARY ===")
all_ok = True
for (ds, seed), v in results.items():
    ok = v["max_diff"] == 0.0 and v["n_probe_diff"] == 0
    print(f"  {ds} seed={seed}: centroid_diff={v['max_diff']:.2e}  "
          f"probe_diff={v['n_probe_diff']}  {'OK' if ok else 'MISMATCH'}")
    if not ok:
        all_ok = False
print(f"\nVERDICT: {'IDENTICAL — strong claim valid' if all_ok else 'MISMATCH — must use soft claim'}")
