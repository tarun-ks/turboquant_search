"""
Decisive matched re-ranking probe for the streaming comparison (R2 W3).

Question: does top-50 raw-vector re-ranking close the IVF-TQ vs IVF-PQ
streaming gap? If PQ staleness only *misranks* true neighbors *within* the
top-50 candidate set, re-ranking recovers them and the gap vanishes. If
staleness *ejects* true neighbors *out* of the top-50, re-ranking cannot
recover them and IVF-TQ's advantage survives. We do not know which a priori;
this measures it.

Matched protocol (both methods identical):
  * Same coarse partition params (nlist=3162, nprobe=20), same seeds.
  * Retrieve top-RERANK_DEPTH candidates from each index (approx score).
  * rr=0  : take top-10 of those candidates (pure compressed ranking).
  * rr=50 : re-score ALL candidates by EXACT inner product against the raw
            (unit-normalised) vectors via the SAME external_rerank() for both
            IVF-TQ and IVF-PQ, then take top-10.
  * GT is seed-independent per state (depends only on the data prefix), so we
    compute it once per state and reuse across seeds.

IVF-PQ here is the STALE index (train on first 1M, never retrain). The
retrain variant is omitted from this probe: its non-recovery is already
established (Tables 3-5 in the paper), and it is orthogonal to whether the
candidate SET contains the true neighbours.

Usage:
    python experiments/streaming_rerank.py --cell sift10m_pqmatched --seeds 42 123 7777
    python experiments/streaming_rerank.py --cell deep10m_pqmatched --seeds 42 123 7777
    python experiments/streaming_rerank.py --cell t2i10m_pqmatched  --seeds 42 123 7777

Output: experiments/results/streaming_rerank_<cell>.csv  (incremental)
Every measurement is printed to stdout as it happens (execution evidence).
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from turboquant_search.core import IVFTurboQuantIndex
from turboquant_search.faiss_baselines import FAISS_AVAILABLE, FAISSFlatIndex
from turboquant_search.benchmarks import compute_recall

assert FAISS_AVAILABLE, "faiss-cpu required"
import faiss  # noqa: E402

# Reuse the exact seeding / loading / IVF-PQ factory from the main harness so
# the protocol is identical to the tables we are augmenting.
from streaming_multiseed import (  # noqa: E402
    set_all_seeds, verify_determinism, normalize, make_ivfpq, _load_10m_cached, log,
)

RESULTS_DIR = ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RERANK_DEPTH = 50
NLIST, NPROBE = 3162, 20
BITS_TQ = 4  # IVF-TQ 4-bit + sign

# Bit-matched (headline) regime per dataset — mirrors streaming_multiseed cells.
CELLS = {
    "deep10m_pqmatched": dict(dataset="deep10m", m_pq=48,  bits_per_sub=10),
    "sift10m_pqmatched": dict(dataset="sift10m", m_pq=64,  bits_per_sub=10),
    "t2i10m_pqmatched":  dict(dataset="t2i10m",  m_pq=100, bits_per_sub=10),
}


def external_rerank(queries_normed: np.ndarray, cand_idx: np.ndarray,
                    base: np.ndarray, k: int) -> np.ndarray:
    """Exact-IP re-rank of candidate indices against raw unit-normalised vectors.

    Identical logic applied to BOTH IVF-TQ and IVF-PQ candidates so the only
    difference between methods is which candidate set they produce.
    """
    nq = queries_normed.shape[0]
    out = np.full((nq, k), -1, dtype=np.int64)
    for q in range(nq):
        c = cand_idx[q]
        c = c[c >= 0]
        if c.size == 0:
            continue
        raw = np.asarray(base[c]).astype(np.float32)
        nrm = np.linalg.norm(raw, axis=1, keepdims=True)
        raw = raw / np.maximum(nrm, 1e-8)
        scores = raw @ queries_normed[q]
        order = np.argsort(-scores)[:k]
        out[q, :order.size] = c[order]
    return out


def state_gt(base, queries, n_indexed, dim, cache):
    if n_indexed in cache:
        return cache[n_indexed]
    log(f"    GT recompute vs {n_indexed // 1_000_000}M (seed-independent, cached)…")
    gt_idx = FAISSFlatIndex(dim)
    gt_idx.add(np.asarray(base[:n_indexed]))
    _, gt = gt_idx.search(queries, k=10)
    del gt_idx
    gc.collect()
    cache[n_indexed] = gt
    return gt


def run_cell(cell_name: str, seeds: List[int]) -> List[dict]:
    cfg = CELLS[cell_name]
    dataset, m_pq, bits_per_sub = cfg["dataset"], cfg["m_pq"], cfg["bits_per_sub"]

    base, queries = _load_10m_cached(dataset)
    n_total, dim = base.shape
    queries_normed = normalize(queries)
    log(f"cell={cell_name} dataset={dataset} N={n_total:,} dim={dim} "
        f"m_pq={m_pq} b_per_sub={bits_per_sub} rerank_depth={RERANK_DEPTH}")

    n_initial, batch_size, n_batches = 1_000_000, 1_000_000, 9
    gt_cache: dict = {}
    rows: List[dict] = []

    for seed in seeds:
        log(f"\n=== seed {seed} (determinism {verify_determinism(seed)}) ===")
        t0 = time.time()
        set_all_seeds(seed)

        init = np.asarray(base[:n_initial])
        init_normed = normalize(init)

        log("  train IVF-TQ (4-bit+sign) on first 1M…")
        ivf_tq = IVFTurboQuantIndex(dim, nlist=NLIST, bits=BITS_TQ, nprobe=NPROBE,
                                    use_residual_sign=True, seed=seed)
        ivf_tq.train(init)
        ivf_tq.add(init)
        ivf_tq._raw_vectors = None  # we re-rank externally from the mmap base

        log(f"  train IVF-PQ stale (m={m_pq}, {bits_per_sub}-bit) on first 1M…")
        ivf_pq = make_ivfpq(dim, NLIST, m_pq, bits_per_sub, NPROBE, seed, init_normed)
        ivf_pq.add(init_normed)
        del init, init_normed
        gc.collect()

        def measure(state_idx: int, n_indexed: int):
            gt = state_gt(base, queries, n_indexed, dim, gt_cache)
            _, tq_cand = ivf_tq.search(queries, k=RERANK_DEPTH)
            _, pq_cand = ivf_pq.search(queries_normed, RERANK_DEPTH)

            tq_rr0 = compute_recall(gt, tq_cand[:, :10], 10) * 100
            pq_rr0 = compute_recall(gt, pq_cand[:, :10], 10) * 100
            tq_rr50 = compute_recall(
                gt, external_rerank(queries_normed, tq_cand, base, 10), 10) * 100
            pq_rr50 = compute_recall(
                gt, external_rerank(queries_normed, pq_cand, base, 10), 10) * 100

            log(f"    step={state_idx} N={n_indexed // 1_000_000}M | "
                f"TQ rr0={tq_rr0:.2f} rr50={tq_rr50:.2f} | "
                f"PQ rr0={pq_rr0:.2f} rr50={pq_rr50:.2f} | "
                f"gap(rr50)={tq_rr50 - pq_rr50:+.2f}")

            for index_name, rr0, rr50 in (("ivf_tq", tq_rr0, tq_rr50),
                                          ("ivf_pq_stale", pq_rr0, pq_rr50)):
                for rr, rec in ((0, rr0), (50, rr50)):
                    rows.append({
                        "seed": seed, "dataset": dataset, "cell": cell_name,
                        "m_pq": m_pq, "bits_per_sub": bits_per_sub,
                        "index": index_name, "rerank": rr,
                        "vectors_indexed": n_indexed, "recall10": round(rec, 4),
                    })

        measure(0, n_initial)
        for b in range(n_batches):
            start = n_initial + b * batch_size
            end = min(start + batch_size, n_total)
            log(f"  batch {b + 1}: add {start // 1_000_000}M→{end // 1_000_000}M…")
            batch = np.asarray(base[start:end])
            batch_normed = normalize(batch)
            ivf_tq.add(batch)
            ivf_tq._raw_vectors = None
            ivf_pq.add(batch_normed)
            del batch, batch_normed
            gc.collect()
            measure(b + 1, end)

        del ivf_tq, ivf_pq
        gc.collect()

        # incremental save after each seed
        pd.DataFrame(rows).to_csv(RESULTS_DIR / f"streaming_rerank_{cell_name}.csv",
                                  index=False)
        log(f"=== seed {seed} done in {time.time() - t0:.0f}s; {len(rows)} rows ===")

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=list(CELLS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 7777])
    args = ap.parse_args()

    rows = run_cell(args.cell, args.seeds)
    out = RESULTS_DIR / f"streaming_rerank_{args.cell}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    log(f"\nFinal CSV: {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
