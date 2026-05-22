"""
Adaptive IVF-TQ vs streaming distribution shift.

Setup:
    - Train all indexes on the first 200K vectors of Deep-1M ("space A").
    - Stream in 800K more vectors with a random orthogonal rotation
      applied (simulates an encoder swap to "space B"). Queries are
      pre-rotated to space B (production-deployed encoder).
    - After each batch, recompute Recall@10 against the cumulative
      mixed-space database.

We compare four indexes:
    1. IVF-TQ frozen (no refresh)
    2. Adaptive IVF-TQ — refresh every 100K vectors
    3. IVF-PQ stale (frozen codebook)
    4. IVF-PQ retrain (codebook re-trained every batch)

Output:
    experiments/adaptive_shift_results.json           (default, rerank=50)
    experiments/adaptive_shift_norerank_results.json  (when --rerank 0)

Usage:
    python experiments/adaptive_ivftq_shift.py             # default rerank=50
    python experiments/adaptive_ivftq_shift.py --rerank 0  # rr=0 column of Table 5
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.environ["TQS_THREADS"] = str(os.cpu_count() or 1)

from turboquant_search.core import IVFTurboQuantIndex
from turboquant_search.adaptive import AdaptiveIVFTurboQuantIndex
from turboquant_search.faiss_baselines import FAISS_AVAILABLE, FAISSFlatIndex
from turboquant_search.benchmarks import compute_recall
from turboquant_search.datasets import load_deep1m, load_sift1m

assert FAISS_AVAILABLE
import faiss


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def normalize(v):
    v = np.ascontiguousarray(v.astype(np.float32))
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(n, 1e-8)


def random_rotation(dim: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    G = rng.randn(dim, dim).astype(np.float32)
    Q, _ = np.linalg.qr(G)
    return Q


def main():
    global RERANK
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerank", type=int, default=50,
                        help="Re-rank top-N candidates with raw vectors. Use 0 for the rr=0 column of Table 5; 50 for the rr=50 column (default).")
    args = parser.parse_args()
    RERANK = args.rerank

    N_INIT = 200_000
    N_NEW = 800_000
    BATCH = 100_000
    NQ = 5_000
    NLIST = 500
    NPROBE = 20
    M_PQ = 96  # divides Deep-1M dim=96
    BITS = 4

    log(f"Loading Deep-1M ...")
    res = load_deep1m(n_vectors=N_INIT + N_NEW, n_queries=NQ)
    vectors_raw, queries_raw, label = res
    log(f"  {label}")

    dim = vectors_raw.shape[1]
    vectors = normalize(vectors_raw)
    queries = normalize(queries_raw)

    # Apply rotation R to (a) the streaming portion of the database and
    # (b) the queries — simulating an encoder swap.
    R = random_rotation(dim, seed=2026)
    init_db = vectors[:N_INIT]                       # space A
    new_db = vectors[N_INIT:N_INIT + N_NEW] @ R      # space B
    queries_B = queries @ R                           # post-swap encoder

    # cosine sample on shared passages — sanity check
    cos = float(np.mean(np.sum(init_db[:200] * (init_db[:200] @ R), axis=1)))
    log(f"  cos(orig, rotated) on shared sample = {cos:.3f}")

    # ──────────────────────────────────────────────────────────────
    # Build all four indexes on init_db (space A)
    # ──────────────────────────────────────────────────────────────
    log("Training IVF-TQ frozen ...")
    tq_frozen = IVFTurboQuantIndex(dim, nlist=NLIST, bits=BITS, nprobe=NPROBE,
                                    use_residual_sign=True, seed=42)
    tq_frozen.train(init_db)
    tq_frozen.add(init_db)

    log("Training Adaptive IVF-TQ ...")
    tq_adapt = AdaptiveIVFTurboQuantIndex(dim, nlist=NLIST, bits=BITS,
                                           nprobe=NPROBE,
                                           use_residual_sign=True, seed=42,
                                           refresh_every=BATCH,
                                           refresh_sample=50_000)
    tq_adapt.train(init_db)
    tq_adapt.add(init_db)

    log("Training IVF-PQ stale ...")
    quantizer = faiss.IndexFlatIP(dim)
    pq_stale = faiss.IndexIVFPQ(quantizer, dim, NLIST, M_PQ, 8,
                                 faiss.METRIC_INNER_PRODUCT)
    pq_stale.nprobe = NPROBE
    pq_stale.train(init_db); pq_stale.add(init_db)

    log("Training IVF-PQ retrain ...")
    def make_pq(data):
        q = faiss.IndexFlatIP(dim)
        idx = faiss.IndexIVFPQ(q, dim, NLIST, M_PQ, 8, faiss.METRIC_INNER_PRODUCT)
        idx.nprobe = NPROBE
        idx.train(data)
        return idx
    pq_retrain = make_pq(init_db); pq_retrain.add(init_db)
    cumulative = [init_db]
    pq_retrain_cum = 0.0

    rows = []

    def measure(step, db):
        flat = FAISSFlatIndex(dim); flat.add(db)
        _, gt = flat.search(queries_B, k=10)
        del flat; gc.collect()

        _, I_f = tq_frozen.search(queries_B, k=10, rerank=RERANK)
        _, I_a = tq_adapt.search(queries_B, k=10, rerank=RERANK)
        _, I_ps = pq_stale.search(queries_B, 10)
        _, I_pr = pq_retrain.search(queries_B, 10)

        r = {
            "step": step,
            "n_indexed": db.shape[0],
            "ivf_tq_frozen": round(compute_recall(gt, I_f, 10) * 100, 2),
            "ivf_tq_adapt":  round(compute_recall(gt, I_a, 10) * 100, 2),
            "ivf_pq_stale":  round(compute_recall(gt, I_ps, 10) * 100, 2),
            "ivf_pq_retrain":round(compute_recall(gt, I_pr, 10) * 100, 2),
            "pq_retrain_time_cum_s": round(pq_retrain_cum, 2),
            "tq_refresh_time_cum_s": round(tq_adapt.refresh_total_time, 2),
            "tq_refresh_count":      tq_adapt.refresh_count,
        }
        rows.append(r)
        log(f"  {step}: TQ_frozen={r['ivf_tq_frozen']}  TQ_adapt={r['ivf_tq_adapt']}  "
            f"PQ_stale={r['ivf_pq_stale']}  PQ_retrain={r['ivf_pq_retrain']}  "
            f"refresh_t={r['tq_refresh_time_cum_s']}s  retrain_t={r['pq_retrain_time_cum_s']}s")

    measure("Initial 200K (A)", init_db)

    for i in range(N_NEW // BATCH):
        s = i * BATCH; e = s + BATCH
        batch = new_db[s:e]
        tq_frozen.add(batch)
        tq_adapt.add(batch)  # auto-refresh fires here per refresh_every=BATCH
        pq_stale.add(batch)

        cumulative.append(batch)
        all_data = np.concatenate(cumulative)
        t0 = time.time()
        pq_retrain = make_pq(all_data)
        pq_retrain.add(all_data)
        pq_retrain_cum += time.time() - t0

        measure(f"+{(e // 1000)}K B", all_data)
        del all_data; gc.collect()

    suffix = "_norerank" if RERANK == 0 else ""
    out = ROOT / "experiments" / f"adaptive_shift{suffix}_results.json"
    with open(out, "w") as f:
        json.dump({
            "config": {
                "dataset": "Deep-1M (rotated to simulate encoder swap)",
                "n_init": N_INIT, "n_new": N_NEW, "batch": BATCH,
                "nlist": NLIST, "nprobe": NPROBE, "m_pq": M_PQ, "bits": BITS,
                "n_queries": NQ, "dim": dim,
                "cos_old_new_sample": cos,
            },
            "steps": rows,
            "final_summary": {
                "tq_refresh_total_time_s": tq_adapt.refresh_total_time,
                "tq_refresh_count": tq_adapt.refresh_count,
                "pq_retrain_total_time_s": pq_retrain_cum,
            },
        }, f, indent=2)
    log(f"Saved {out}")


if __name__ == "__main__":
    main()
