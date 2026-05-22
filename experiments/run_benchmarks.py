"""
Top-level benchmark runner for IVF-TQ.

Runs the headline 1M-scale comparisons (sign-bit refinement, flat TQ vs.
IVF-TQ, FAISS PQ/OPQ/HNSW baselines, ScaNN where available) and writes a
consolidated results JSON.

Usage:
    python experiments/run_benchmarks.py

Generates:
    experiments/benchmark_results.json

Estimated runtime: ~15 minutes (SIFT-1M and Deep-1M download + search).
"""

import numpy as np
import time
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["TQS_THREADS"] = str(os.cpu_count() or 1)

from turboquant_search.core import (
    TurboQuantSearchIndex, IVFTurboQuantIndex, FlatSearchIndex,
)
from turboquant_search.faiss_baselines import (
    FAISS_AVAILABLE, FAISSFlatIndex, FAISSPQIndex, FAISSIVFPQIndex,
)
from turboquant_search.benchmarks import compute_recall
from turboquant_search.datasets import (
    load_synthetic, load_sift128, load_glove100, load_sift1m, load_deep1m,
)

assert FAISS_AVAILABLE, "faiss-cpu is required: pip install faiss-cpu"

results = {}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def measure_qps(search_fn, n_queries, n_runs=5):
    """Measure median QPS over n_runs."""
    times = []
    for _ in range(n_runs):
        t0 = time.time()
        out = search_fn()
        times.append(time.time() - t0)
    t = np.median(times)
    return n_queries / t if t > 0 else 0, t, out


# ══════════════════════════════════════════════════════════════
# Table 2: Stage 2 comparison (sign-bit vs QJL, 10K scale)
# ══════════════════════════════════════════════════════════════
log("=" * 60)
log("Table 2: Sign-bit vs QJL (10K scale)")
log("=" * 60)

results["table2_stage2"] = {}

for ds_name, loader, ds_kwargs in [
    ("Synthetic", load_synthetic, {"n_vectors": 10000, "n_queries": 200, "dim": 128}),
    ("SIFT-128", load_sift128, {"n_vectors": 10000, "n_queries": 200}),
    ("GloVe-100", load_glove100, {"n_vectors": 10000, "n_queries": 200}),
]:
    log(f"  Loading {ds_name}...")
    result = loader(**ds_kwargs)
    if result is None:
        log(f"  SKIPPED {ds_name} (download failed)")
        continue
    vectors, queries, label = result
    dim = vectors.shape[1]

    # Ground truth
    flat = FAISSFlatIndex(dim)
    flat.add(vectors)
    _, gt = flat.search(queries, k=10)
    _, gt1 = flat.search(queries, k=1)

    from experiments.qjl_index import QJLSearchIndex

    ds_results = {}
    for bits in [2, 3, 4]:
        # No Stage 2
        tq_no = TurboQuantSearchIndex(dim, bits=bits, use_residual_sign=False, seed=42)
        tq_no.add(vectors)
        _, idx_no = tq_no.search(queries, k=10)
        _, idx_no1 = tq_no.search(queries, k=1)

        # QJL (TurboQuant's original Stage 2)
        qjl = QJLSearchIndex(dim, bits=bits, seed=42)
        qjl.add(vectors)
        _, idx_qjl = qjl.search(queries, k=10)
        _, idx_qjl1 = qjl.search(queries, k=1)

        # Sign-bit refinement (ours)
        tq_sign = TurboQuantSearchIndex(dim, bits=bits, use_residual_sign=True, seed=42)
        tq_sign.add(vectors)
        _, idx_sign = tq_sign.search(queries, k=10)
        _, idx_sign1 = tq_sign.search(queries, k=1)

        r_no = compute_recall(gt, idx_no, 10)
        r_qjl = compute_recall(gt, idx_qjl, 10)
        r_sign = compute_recall(gt, idx_sign, 10)
        r_no1 = compute_recall(gt1, idx_no1, 1)
        r_qjl1 = compute_recall(gt1, idx_qjl1, 1)
        r_sign1 = compute_recall(gt1, idx_sign1, 1)

        ds_results[f"{bits}-bit"] = {
            "no_stage2_r10": round(r_no * 100, 1),
            "qjl_r10": round(r_qjl * 100, 1),
            "signbit_r10": round(r_sign * 100, 1),
            "no_stage2_r1": round(r_no1 * 100, 1),
            "qjl_r1": round(r_qjl1 * 100, 1),
            "signbit_r1": round(r_sign1 * 100, 1),
        }
        log(f"    {ds_name} {bits}-bit: No-S2={r_no:.1%}  QJL={r_qjl:.1%}  Sign-bit={r_sign:.1%}")

    results["table2_stage2"][ds_name] = ds_results


# ══════════════════════════════════════════════════════════════
# Table 1 + Table 3: Million-scale results (SIFT-1M, Deep-1M)
# ══════════════════════════════════════════════════════════════
log("")
log("=" * 60)
log("Tables 1 & 3: Million-scale (SIFT-1M + Deep-1M)")
log("=" * 60)

results["table1_1m"] = {}

for ds_name, loader in [("SIFT-1M", load_sift1m), ("Deep-1M", load_deep1m)]:
    log(f"  Loading {ds_name}...")
    result = loader(n_vectors=1000000, n_queries=10000)
    if result is None:
        log(f"  SKIPPED {ds_name}")
        continue
    vectors, queries, label = result
    dim = vectors.shape[1]
    log(f"  {label}")

    # Ground truth
    flat = FAISSFlatIndex(dim)
    flat.add(vectors)
    _, gt = flat.search(queries, k=10)

    ds_results = {}

    # Flat TQ 4-bit
    log(f"    Flat TQ 4-bit...")
    tq = TurboQuantSearchIndex(dim, bits=4, use_residual_sign=True, seed=42)
    tq.add(vectors)
    _, idx = tq.search(queries, k=10)
    r = compute_recall(gt, idx, 10)
    ds_results["flat_tq_4bit"] = {"recall10": round(r * 100, 1)}
    log(f"      R@10={r:.1%}")

    # Flat PQ matched compression
    m_pq = dim // 2
    while dim % m_pq != 0 and m_pq > 1:
        m_pq -= 1
    log(f"    Flat PQ m={m_pq}...")
    pq = FAISSPQIndex(dim, m=m_pq, nbits=8)
    pq.add(vectors)
    _, idx = pq.search(queries, k=10)
    r = compute_recall(gt, idx, 10)
    ds_results[f"flat_pq_m{m_pq}"] = {"recall10": round(r * 100, 1)}
    log(f"      R@10={r:.1%}")

    nq = queries.shape[0]

    # FAISS IVF-PQ nprobe sweep at multiple m values (ceiling analysis)
    for m_val in [m_pq, 128] if dim == 128 else [m_pq]:
        if dim % m_val != 0:
            continue
        log(f"    FAISS IVF-PQ m={m_val} nprobe sweep...")
        faiss_sweep = {}
        for nprobe in [5, 10, 20, 40, 80, 160]:
            ivfpq = FAISSIVFPQIndex(dim, nlist=1000, m=m_val, nbits=8, nprobe=nprobe)
            ivfpq.add(vectors)
            qps, t, (_, idx) = measure_qps(lambda: ivfpq.search(queries, k=10), nq)
            r = compute_recall(gt, idx, 10)
            faiss_sweep[f"np{nprobe}"] = {
                "recall10": round(r * 100, 1),
                "qps": round(qps),
                "latency_ms": round(t * 1000, 1),
            }
            log(f"      np={nprobe}: R@10={r:.1%}  {qps:.0f} QPS  {t*1000:.1f}ms")
        ds_results[f"faiss_ivfpq_m{m_val}_sweep"] = faiss_sweep

    # PQ ceiling sweep (Table 2): all m values at np=160
    if dim == 128:
        log(f"    PQ ceiling sweep (m=8,16,32,64,128)...")
        ceiling_sweep = {}
        for m_val in [8, 16, 32, 64, 128]:
            ivfpq = FAISSIVFPQIndex(dim, nlist=1000, m=m_val, nbits=8, nprobe=160)
            ivfpq.add(vectors)
            _, idx = ivfpq.search(queries, k=10)
            r = compute_recall(gt, idx, 10)
            mem = ivfpq.memory_bytes / (1024 * 1024)
            ceiling_sweep[f"m{m_val}"] = {"recall10": round(r * 100, 1), "memory_mb": round(mem, 1)}
            log(f"      m={m_val}: ceiling R@10={r:.1%}  {mem:.0f} MB")
        ds_results["pq_ceiling_sweep"] = ceiling_sweep

    # IVF-TQ nprobe sweep (no rerank = compressed-only headline)
    log(f"    IVF-TQ 4-bit nprobe sweep...")
    tq_sweep = {}
    for nprobe in [3, 5, 7, 10, 15, 20]:
        ivf = IVFTurboQuantIndex(dim, nlist=1000, bits=4, nprobe=nprobe, seed=42)
        ivf.train(vectors)
        ivf.add(vectors)

        for rerank in [0, 50]:
            ivf.search(queries[:10], k=10, rerank=rerank)  # warmup + cache
            qps, t, (_, idx) = measure_qps(
                lambda rr=rerank: ivf.search(queries, k=10, rerank=rr), nq)
            r = compute_recall(gt, idx, 10)
            key = f"np{nprobe}" + (f"_rr{rerank}" if rerank else "")
            tq_sweep[key] = {
                "recall10": round(r * 100, 1),
                "qps": round(qps),
                "latency_ms": round(t * 1000, 1),
            }
            rr_str = f"+rr{rerank}" if rerank else ""
            log(f"      np={nprobe}{rr_str}: R@10={r:.1%}  {qps:.0f} QPS  {t*1000:.1f}ms")
    ds_results["ivf_tq_sweep"] = tq_sweep

    results["table1_1m"][ds_name] = ds_results


# ══════════════════════════════════════════════════════════════
# Table 3: IVF amplification (flat TQ vs IVF-TQ)
# ══════════════════════════════════════════════════════════════
log("")
log("=" * 60)
log("Table 3: IVF Amplification")
log("=" * 60)

results["table3_ivf_amp"] = {}

for ds_name, loader, kwargs, n in [
    ("SIFT-128", load_sift128, {"n_vectors": 10000, "n_queries": 200}, 10000),
    ("GloVe-100", load_glove100, {"n_vectors": 10000, "n_queries": 200}, 10000),
]:
    result = loader(**kwargs)
    if result is None:
        continue
    vectors, queries, _ = result
    dim = vectors.shape[1]

    flat = FAISSFlatIndex(dim)
    flat.add(vectors)
    _, gt = flat.search(queries, k=10)

    # Flat TQ
    tq = TurboQuantSearchIndex(dim, bits=4, use_residual_sign=True, seed=42)
    tq.add(vectors)
    _, idx = tq.search(queries, k=10)
    flat_r = compute_recall(gt, idx, 10)

    # IVF-TQ
    nlist = max(1, min(100, n // 39))
    ivf = IVFTurboQuantIndex(dim, nlist=nlist, bits=4, nprobe=10, seed=42)
    ivf.train(vectors)
    ivf.add(vectors)
    _, idx = ivf.search(queries, k=10)
    ivf_r = compute_recall(gt, idx, 10)

    results["table3_ivf_amp"][ds_name] = {
        "flat_tq": round(flat_r * 100, 1),
        "ivf_tq": round(ivf_r * 100, 1),
        "delta": round((ivf_r - flat_r) * 100, 1),
    }
    log(f"  {ds_name}: Flat TQ={flat_r:.1%}  IVF-TQ={ivf_r:.1%}  delta={ivf_r-flat_r:+.1%}")


# ══════════════════════════════════════════════════════════════
# Table 4: Streaming ingestion
# ══════════════════════════════════════════════════════════════
log("")
log("=" * 60)
log("Table 4: Streaming Ingestion (SIFT-1M)")
log("=" * 60)

result = load_sift1m(n_vectors=1000000, n_queries=10000)
if result is not None:
    vectors, queries, _ = result
    dim = vectors.shape[1]

    import faiss

    n_initial = 200_000
    batch_size = 100_000
    nlist, nprobe, m_pq = 500, 10, 64

    def normalize(v):
        v = np.ascontiguousarray(v.astype(np.float32))
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.maximum(norms, 1e-8)

    init = vectors[:n_initial]
    init_normed = normalize(init)
    queries_normed = normalize(queries)

    # IVF-TQ
    ivf_tq = IVFTurboQuantIndex(dim, nlist=nlist, bits=4, nprobe=nprobe,
                                 use_residual_sign=True, seed=42)
    ivf_tq.train(init)
    ivf_tq.add(init)

    # IVF-PQ stale
    quantizer = faiss.IndexFlatIP(dim)
    ivf_pq_stale = faiss.IndexIVFPQ(quantizer, dim, nlist, m_pq, 8,
                                     faiss.METRIC_INNER_PRODUCT)
    ivf_pq_stale.nprobe = nprobe
    ivf_pq_stale.train(init_normed)
    ivf_pq_stale.add(init_normed)

    # IVF-PQ retrain
    def make_ivfpq(data):
        q = faiss.IndexFlatIP(dim)
        idx = faiss.IndexIVFPQ(q, dim, nlist, m_pq, 8, faiss.METRIC_INNER_PRODUCT)
        idx.nprobe = nprobe
        idx.train(data)
        return idx

    ivf_pq_retrain = make_ivfpq(init_normed)
    ivf_pq_retrain.add(init_normed)
    all_normed = [init_normed]
    vecs_since_retrain = 0
    total_retrain_time = 0.0

    streaming_results = []

    def measure_streaming(step_name, n_indexed):
        gt_idx_obj = FAISSFlatIndex(dim)
        gt_idx_obj.add(vectors[:n_indexed])
        _, gt_idx = gt_idx_obj.search(queries, k=10)

        _, tq_idx = ivf_tq.search(queries, k=10, rerank=50)
        tq_r = compute_recall(gt_idx, tq_idx, 10)

        _, pq_s_idx = ivf_pq_stale.search(queries_normed, 10)
        pq_s_r = compute_recall(gt_idx, pq_s_idx, 10)

        _, pq_r_idx = ivf_pq_retrain.search(queries_normed, 10)
        pq_r_r = compute_recall(gt_idx, pq_r_idx, 10)

        entry = {
            "step": step_name, "n_indexed": n_indexed,
            "ivf_tq": round(tq_r * 100, 1),
            "ivf_pq_stale": round(pq_s_r * 100, 1),
            "ivf_pq_retrain": round(pq_r_r * 100, 1),
            "retrain_time_cumulative": round(total_retrain_time, 1),
        }
        streaming_results.append(entry)
        log(f"  {step_name}: TQ={tq_r:.1%}  PQ_stale={pq_s_r:.1%}  PQ_retrain={pq_r_r:.1%}")

    measure_streaming(f"Initial ({n_initial//1000}K)", n_initial)

    for batch_i in range(8):
        start = n_initial + batch_i * batch_size
        end = start + batch_size
        batch = vectors[start:end]
        batch_normed = normalize(batch)

        ivf_tq.add(batch)
        ivf_pq_stale.add(batch_normed)

        all_normed.append(batch_normed)
        vecs_since_retrain += batch_size

        if vecs_since_retrain >= 200_000:
            all_data = np.concatenate(all_normed)
            t0 = time.time()
            ivf_pq_retrain = make_ivfpq(all_data)
            ivf_pq_retrain.add(all_data)
            total_retrain_time += time.time() - t0
            vecs_since_retrain = 0
        else:
            ivf_pq_retrain.add(batch_normed)

        measure_streaming(f"Batch {batch_i+1} ({end//1000}K)", end)

    results["table4_streaming"] = streaming_results

# ══════════════════════════════════════════════════════════════
# Table 1 extras: FAISS HNSW (graph) and OPQ+IVF-PQ baselines
# ══════════════════════════════════════════════════════════════
log("")
log("=" * 60)
log("Table 1 extras: FAISS HNSW + OPQ+IVF-PQ (1M scale)")
log("=" * 60)

# These run in a separate module so the script can also be invoked
# standalone (e.g. python experiments/run_hnsw_opq.py --scale 1m).
from experiments.run_hnsw_opq import run_1m_scale

hnsw_opq_out: dict = {}
run_1m_scale(hnsw_opq_out)
results.update(hnsw_opq_out)


# ══════════════════════════════════════════════════════════════
# Save all results
# ══════════════════════════════════════════════════════════════
out_path = Path(__file__).parent / "benchmark_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

log("")
log("=" * 60)
log(f"All results saved to {out_path}")
log("=" * 60)
