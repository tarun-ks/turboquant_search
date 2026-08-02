"""
Oracle-codebook corpus-growth experiment (SIFT-10M, rr=0, compression-only).

Question: does IVF-PQ recall degrade as the corpus grows 1M -> 10M *even when
the codebook cannot be stale by construction*?

We grow the database 1M -> 10M in 9 batches of 1M and, at every checkpoint,
measure Recall@10 (nprobe=20, NO re-ranking) for four indexes on ONE axis:

  pq_oracle    : IVF-PQ whose coarse partition + PQ codebook are trained ONCE
                 on the COMPLETE final 10M corpus, then frozen. Cannot be stale
                 by construction (it already "knows" the final distribution).
                 At checkpoint N it holds the first N vectors, encoded with the
                 10M-trained codebook.
  pq_stale     : IVF-PQ trained on the first 1M, frozen (the v1 "stale" curve).
  pq_refreshed : IVF-PQ retrained on the cumulative first-N at every checkpoint
                 (never stale w.r.t. current data; the v1 "retrain" curve).
  ivf_tq       : IVF-TQ 4-bit+sign, codebook-free reference (the stable curve).

Interpretation:
  * If pq_oracle STILL degrades with N -> codebook staleness is NOT the driver;
    the degradation is a corpus-size / candidate-quality / rank-margin effect.
  * If pq_oracle is FLAT while pq_stale degrades -> staleness IS the driver.

GT is seed-independent per checkpoint (depends only on the data prefix) and is
cached across seeds. Every checkpoint is printed to stdout as it is measured.

Usage:
    python experiments/streaming_oracle.py --seeds 42 123 7777
Output: experiments/results/streaming_oracle_sift10m.csv  (incremental)
"""

from __future__ import annotations

import argparse
import gc
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

from streaming_multiseed import (  # noqa: E402
    set_all_seeds, verify_determinism, normalize, make_ivfpq, _load_10m_cached, log,
)

RESULTS_DIR = ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Bit-matched regimes per dataset (m_pq, bits_per_sub) — paper Tables 3-5.
DATASET_CFG = {"sift10m": (64, 10), "deep10m": (48, 10), "t2i10m": (100, 10)}
DATASET = "sift10m"
M_PQ, BITS_PER_SUB = DATASET_CFG[DATASET]
NLIST, NPROBE = 3162, 20
BITS_TQ = 4
INCLUDE_REFRESHED = True


def run(seeds: List[int]) -> List[dict]:
    base, queries = _load_10m_cached(DATASET)
    n_total, dim = base.shape
    queries_normed = normalize(queries)
    log(f"dataset={DATASET} N={n_total:,} dim={dim} m_pq={M_PQ} b={BITS_PER_SUB} "
        f"nprobe={NPROBE} (rr=0, compression-only)")

    n_initial, batch_size, n_batches = 1_000_000, 1_000_000, 9
    gt_cache: dict = {}
    rows: List[dict] = []

    def state_gt(n_indexed):
        if n_indexed in gt_cache:
            return gt_cache[n_indexed]
        log(f"    GT recompute vs {n_indexed // 1_000_000}M (cached across seeds)…")
        gi = FAISSFlatIndex(dim)
        gi.add(np.asarray(base[:n_indexed]))
        _, gt = gi.search(queries, k=10)
        del gi
        gc.collect()
        gt_cache[n_indexed] = gt
        return gt

    for seed in seeds:
        log(f"\n=== seed {seed} (determinism {verify_determinism(seed)}) ===")
        t0 = time.time()
        set_all_seeds(seed)

        init = np.asarray(base[:n_initial])
        init_normed = normalize(init)

        # ── ORACLE: train coarse+PQ on the FULL 10M, then freeze ──────────
        log("  training pq_oracle on FULL 10M (coarse+PQ), then freezing…")
        full_normed = normalize(np.asarray(base[:]))          # ~5 GB, transient
        pq_oracle = make_ivfpq(dim, NLIST, M_PQ, BITS_PER_SUB, NPROBE, seed, full_normed)
        del full_normed
        gc.collect()
        pq_oracle.add(init_normed)                            # start with first 1M

        # ── STALE: train on first 1M, freeze ──────────────────────────────
        log("  training pq_stale on first 1M, freezing…")
        pq_stale = make_ivfpq(dim, NLIST, M_PQ, BITS_PER_SUB, NPROBE, seed, init_normed)
        pq_stale.add(init_normed)

        # ── IVF-TQ reference (codebook-free) ──────────────────────────────
        log("  training ivf_tq (4-bit+sign) on first 1M…")
        ivf_tq = IVFTurboQuantIndex(dim, nlist=NLIST, bits=BITS_TQ, nprobe=NPROBE,
                                    use_residual_sign=True, seed=seed)
        ivf_tq.train(init)
        ivf_tq.add(init)
        ivf_tq._raw_vectors = None  # rr=0, free memory

        del init, init_normed
        gc.collect()

        def measure(state_idx, n_indexed):
            gt = state_gt(n_indexed)

            if INCLUDE_REFRESHED:
                # pq_refreshed: retrain on cumulative first-N at THIS checkpoint
                log(f"    training pq_refreshed on cumulative {n_indexed // 1_000_000}M…")
                cum_normed = normalize(np.asarray(base[:n_indexed]))
                pq_ref = make_ivfpq(dim, NLIST, M_PQ, BITS_PER_SUB, NPROBE, seed, cum_normed)
                pq_ref.add(cum_normed)
                del cum_normed
                gc.collect()
                _, r_I = pq_ref.search(queries_normed, 10)
                del pq_ref
                gc.collect()

            _, o_I = pq_oracle.search(queries_normed, 10)
            _, s_I = pq_stale.search(queries_normed, 10)
            _, t_I = ivf_tq.search(queries, k=10)

            recs = {
                "pq_oracle":    compute_recall(gt, o_I, 10) * 100,
                "pq_stale":     compute_recall(gt, s_I, 10) * 100,
                "ivf_tq":       compute_recall(gt, t_I, 10) * 100,
            }
            if INCLUDE_REFRESHED:
                recs["pq_refreshed"] = compute_recall(gt, r_I, 10) * 100
            rstr = f" refreshed={recs['pq_refreshed']:.2f}" if INCLUDE_REFRESHED else ""
            log(f"    step={state_idx} N={n_indexed // 1_000_000}M | "
                f"oracle={recs['pq_oracle']:.2f} stale={recs['pq_stale']:.2f}"
                f"{rstr} ivf_tq={recs['ivf_tq']:.2f}")
            for variant, rec in recs.items():
                rows.append({"seed": seed, "dataset": DATASET, "variant": variant,
                             "vectors_indexed": n_indexed, "recall10": round(rec, 4)})

        measure(0, n_initial)
        for b in range(n_batches):
            start = n_initial + b * batch_size
            end = min(start + batch_size, n_total)
            log(f"  batch {b + 1}: add {start // 1_000_000}M→{end // 1_000_000}M "
                f"to oracle/stale/ivf_tq…")
            batch = np.asarray(base[start:end])
            batch_normed = normalize(batch)
            pq_oracle.add(batch_normed)
            pq_stale.add(batch_normed)
            ivf_tq.add(batch)
            ivf_tq._raw_vectors = None
            del batch, batch_normed
            gc.collect()
            measure(b + 1, end)

        del pq_oracle, pq_stale, ivf_tq
        gc.collect()
        pd.DataFrame(rows).to_csv(RESULTS_DIR / f"streaming_oracle_{DATASET}.csv", index=False)
        log(f"=== seed {seed} done in {time.time() - t0:.0f}s; {len(rows)} rows ===")

    return rows


def main():
    global DATASET, M_PQ, BITS_PER_SUB, INCLUDE_REFRESHED
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sift10m", choices=list(DATASET_CFG))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 7777])
    ap.add_argument("--no-refreshed", action="store_true",
                    help="skip the per-batch retrain curve (confirmatory + expensive)")
    args = ap.parse_args()
    DATASET = args.dataset
    M_PQ, BITS_PER_SUB = DATASET_CFG[DATASET]
    INCLUDE_REFRESHED = not args.no_refreshed
    rows = run(args.seeds)
    out = RESULTS_DIR / f"streaming_oracle_{DATASET}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    log(f"\nFinal CSV: {out} ({len(rows)} rows)")

    df = pd.DataFrame(rows)
    variants = ["pq_oracle", "pq_stale", "ivf_tq"]
    if INCLUDE_REFRESHED:
        variants.insert(2, "pq_refreshed")
    log("\n=== corpus-growth deltas (10M - 1M), mean over seeds ===")
    for v in variants:
        s = df[df["variant"] == v]
        d1 = s[s["vectors_indexed"] == 1_000_000].set_index("seed")["recall10"]
        d10 = s[s["vectors_indexed"] == 10_000_000].set_index("seed")["recall10"]
        log(f"  {v:13s}: 1M={d1.mean():.2f} -> 10M={d10.mean():.2f}  "
            f"delta={(d10 - d1).mean():+.2f}pp")


if __name__ == "__main__":
    main()
