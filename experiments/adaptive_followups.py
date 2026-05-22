"""
Two follow-ups to the Adaptive IVF-TQ shift experiment:
    1. rerank=0 fairness — same workload, all indexes use no re-rank.
    2. refresh-frequency sweep — refresh every {50K, 100K, 250K, never}.

Outputs:
    experiments/adaptive_shift_norerank_results.json
    experiments/adaptive_freq_sweep_results.json
"""

import gc, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.environ["TQS_THREADS"] = str(os.cpu_count() or 1)

from turboquant_search.core import IVFTurboQuantIndex
from turboquant_search.adaptive import AdaptiveIVFTurboQuantIndex
from turboquant_search.faiss_baselines import FAISS_AVAILABLE, FAISSFlatIndex
from turboquant_search.benchmarks import compute_recall
from turboquant_search.datasets import load_deep1m
import faiss

assert FAISS_AVAILABLE


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def normalize(v):
    v = np.ascontiguousarray(v.astype(np.float32))
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(n, 1e-8)


def random_rotation(dim, seed):
    rng = np.random.RandomState(seed)
    G = rng.randn(dim, dim).astype(np.float32)
    Q, _ = np.linalg.qr(G)
    return Q


def setup():
    res = load_deep1m(n_vectors=1_000_000, n_queries=5_000)
    vectors_raw, queries_raw, _ = res
    dim = vectors_raw.shape[1]
    vectors = normalize(vectors_raw); queries = normalize(queries_raw)
    R = random_rotation(dim, seed=2026)
    init = vectors[:200_000]
    new = vectors[200_000:1_000_000] @ R
    queries_B = queries @ R
    return init, new, queries_B, dim


def make_pq(dim, nlist, m_pq, train, nprobe):
    q = faiss.IndexFlatIP(dim)
    idx = faiss.IndexIVFPQ(q, dim, nlist, m_pq, 8, faiss.METRIC_INNER_PRODUCT)
    idx.nprobe = nprobe; idx.train(train); return idx


# ──────────────────────────────────────────────────────────────────
# Experiment 1: rerank=0 fairness
# ──────────────────────────────────────────────────────────────────
def run_norerank():
    log("=== Follow-up 1: rerank=0 fairness ===")
    NLIST, NPROBE, M_PQ, BATCH = 500, 20, 96, 100_000
    init, new, queries_B, dim = setup()

    tq_frozen = IVFTurboQuantIndex(dim, nlist=NLIST, bits=4, nprobe=NPROBE,
                                    use_residual_sign=True, seed=42)
    tq_frozen.train(init); tq_frozen.add(init)
    tq_adapt = AdaptiveIVFTurboQuantIndex(dim, nlist=NLIST, bits=4, nprobe=NPROBE,
                                           use_residual_sign=True, seed=42,
                                           refresh_every=BATCH, refresh_sample=50_000)
    tq_adapt.train(init); tq_adapt.add(init)
    pq_stale = make_pq(dim, NLIST, M_PQ, init, NPROBE); pq_stale.add(init)
    pq_retrain = make_pq(dim, NLIST, M_PQ, init, NPROBE); pq_retrain.add(init)
    cumulative = [init]; pq_retrain_cum = 0.0
    rows = []

    def measure(step, db):
        flat = FAISSFlatIndex(dim); flat.add(db)
        _, gt = flat.search(queries_B, k=10); del flat; gc.collect()
        _, I_f = tq_frozen.search(queries_B, k=10, rerank=0)
        _, I_a = tq_adapt.search(queries_B, k=10, rerank=0)
        _, I_ps = pq_stale.search(queries_B, 10)
        _, I_pr = pq_retrain.search(queries_B, 10)
        r = {"step": step, "n_indexed": int(db.shape[0]),
             "ivf_tq_frozen": round(compute_recall(gt, I_f, 10) * 100, 2),
             "ivf_tq_adapt":  round(compute_recall(gt, I_a, 10) * 100, 2),
             "ivf_pq_stale":  round(compute_recall(gt, I_ps, 10) * 100, 2),
             "ivf_pq_retrain":round(compute_recall(gt, I_pr, 10) * 100, 2),
             "pq_retrain_time_cum_s": round(pq_retrain_cum, 2),
             "tq_refresh_time_cum_s": round(tq_adapt.refresh_total_time, 2)}
        rows.append(r)
        log(f"  {step}: TQfrz={r['ivf_tq_frozen']} TQadt={r['ivf_tq_adapt']} "
            f"PQstl={r['ivf_pq_stale']} PQrtr={r['ivf_pq_retrain']}")

    measure("Initial 200K", init)
    for i in range(8):
        s = i * BATCH; e = s + BATCH
        batch = new[s:e]
        tq_frozen.add(batch); tq_adapt.add(batch); pq_stale.add(batch)
        cumulative.append(batch)
        all_data = np.concatenate(cumulative)
        t0 = time.time()
        pq_retrain = make_pq(dim, NLIST, M_PQ, all_data, NPROBE); pq_retrain.add(all_data)
        pq_retrain_cum += time.time() - t0
        measure(f"+{(e//1000)}K B", all_data)
        del all_data; gc.collect()

    out = ROOT / "experiments" / "adaptive_shift_norerank_results.json"
    with open(out, "w") as f:
        json.dump({"steps": rows,
                    "config": {"rerank": 0, "nlist": NLIST, "nprobe": NPROBE,
                               "m_pq": M_PQ, "bits": 4}}, f, indent=2)
    log(f"Saved {out}")


# ──────────────────────────────────────────────────────────────────
# Experiment 2: refresh-frequency sweep
# ──────────────────────────────────────────────────────────────────
def run_freq_sweep():
    log("=== Follow-up 2: refresh-frequency sweep ===")
    NLIST, NPROBE, BATCH = 500, 20, 100_000
    init, new, queries_B, dim = setup()

    flat = FAISSFlatIndex(dim); flat.add(np.concatenate([init, new]))
    _, gt_full = flat.search(queries_B, k=10); del flat; gc.collect()

    results = {}
    for freq in [None, 25_000, 50_000, 100_000, 250_000]:
        label = f"every_{freq}" if freq else "never"
        log(f"  --- refresh frequency = {label} ---")
        idx = AdaptiveIVFTurboQuantIndex(dim, nlist=NLIST, bits=4, nprobe=NPROBE,
                                          use_residual_sign=True, seed=42,
                                          refresh_every=freq, refresh_sample=50_000)
        idx.train(init); idx.add(init)
        for i in range(8):
            s = i * BATCH; e = s + BATCH
            idx.add(new[s:e])

        _, I = idx.search(queries_B, k=10, rerank=50)
        r = compute_recall(gt_full, I, 10)
        results[label] = {
            "final_recall10": round(r * 100, 2),
            "refresh_count": idx.refresh_count,
            "refresh_total_time_s": round(idx.refresh_total_time, 2),
        }
        log(f"    R@10={r:.1%}  refreshes={idx.refresh_count}  "
            f"total_refresh_t={idx.refresh_total_time:.1f}s")
        del idx; gc.collect()

    out = ROOT / "experiments" / "adaptive_freq_sweep_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Saved {out}")


if __name__ == "__main__":
    run_norerank()
    run_freq_sweep()
