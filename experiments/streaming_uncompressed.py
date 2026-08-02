"""
Experiment 1: Uncompressed IVF control for the corpus-growth mechanism.

Same corpus-growth protocol as streaming_oracle.py (SIFT-10M, 1M->10M,
nprobe=20, rr=0, seeds 42/123/7777), but adds an UNCOMPRESSED IVF curve:
FAISS IndexIVFFlat = identical coarse partition style (k-means on first 1M,
frozen) with EXACT float32 distances, NO quantization.

Purpose (two-fold, per the experiment design):
  1. Split the mechanism:
       - uncompressed IVF degrades too  -> effect is IVF/nprobe coverage
         (coarse partition loses true neighbours from probed cells as N grows),
         NOT quantization distortion.
       - uncompressed IVF holds          -> effect is quantization-specific.
  2. Coherence check on our own headline. IVF-TQ is a LOSSY compression of the
     exact residuals; it must NOT beat exact residuals. We re-run ivf_tq here
     against the identical GT so the check is within-harness.

     STOP CONDITION: if uncompressed_ivf DEGRADES while ivf_tq HOLDS (i.e. lossy
     TQ beats exact residuals by a non-trivial margin), that is a red flag for a
     harness asymmetry (nprobe, candidate depth, GT, or normalization). The
     script prints an explicit COHERENCE verdict at every checkpoint so the
     asymmetry cannot be missed.

coarse partitions:
  * uncompressed_ivf : FAISS IndexIVFFlat, coarse k-means on first 1M (seed).
  * ivf_tq           : IVFTurboQuantIndex, own k-means on first 1M (seed).
Both freeze the coarse partition at 1M and grow the database; both use
nprobe=20 and the same GT (exact top-10 over the first N vectors).

Usage:
    python experiments/streaming_uncompressed.py --seeds 42 123 7777
Output: experiments/results/streaming_uncompressed_sift10m.csv  (incremental)
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
import faiss  # noqa: E402

from streaming_multiseed import (  # noqa: E402
    set_all_seeds, verify_determinism, normalize, _load_10m_cached, log,
)

RESULTS_DIR = ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATASET = "sift10m"
NLIST, NPROBE = 3162, 20
BITS_TQ = 4
VALID_DATASETS = ["sift10m", "deep10m", "t2i10m"]


def make_ivfflat(dim, nlist, nprobe, seed, train_data):
    """Uncompressed IVF: coarse k-means + EXACT float32 inner product."""
    set_all_seeds(seed)
    quantizer = faiss.IndexFlatIP(dim)
    idx = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    if hasattr(idx, "cp") and hasattr(idx.cp, "seed"):
        idx.cp.seed = int(seed)
    idx.train(train_data)
    idx.nprobe = nprobe
    return idx


def run(seeds: List[int]) -> List[dict]:
    base, queries = _load_10m_cached(DATASET)
    n_total, dim = base.shape
    queries_normed = normalize(queries)
    log(f"dataset={DATASET} N={n_total:,} dim={dim} nprobe={NPROBE} "
        f"(rr=0; uncompressed IVFFlat vs IVF-TQ coherence check)")

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

        log("  training uncompressed IVFFlat (coarse on first 1M, exact residuals)…")
        ivf_flat = make_ivfflat(dim, NLIST, NPROBE, seed, init_normed)
        ivf_flat.add(init_normed)

        log("  training ivf_tq (4-bit+sign) on first 1M…")
        ivf_tq = IVFTurboQuantIndex(dim, nlist=NLIST, bits=BITS_TQ, nprobe=NPROBE,
                                    use_residual_sign=True, seed=seed)
        ivf_tq.train(init)
        ivf_tq.add(init)
        ivf_tq._raw_vectors = None
        del init, init_normed
        gc.collect()

        def measure(state_idx, n_indexed):
            gt = state_gt(n_indexed)
            _, f_I = ivf_flat.search(queries_normed, 10)
            _, t_I = ivf_tq.search(queries, k=10)
            recs = {
                "uncompressed_ivf": compute_recall(gt, f_I, 10) * 100,
                "ivf_tq":           compute_recall(gt, t_I, 10) * 100,
            }
            # Coherence: lossy TQ should NOT beat exact residuals by more than noise.
            delta = recs["ivf_tq"] - recs["uncompressed_ivf"]
            verdict = "OK" if delta <= 0.5 else "RED-FLAG(TQ>exact)"
            log(f"    step={state_idx} N={n_indexed // 1_000_000}M | "
                f"uncompressed={recs['uncompressed_ivf']:.2f} "
                f"ivf_tq={recs['ivf_tq']:.2f} | TQ-exact={delta:+.2f} [{verdict}]")
            for variant, rec in recs.items():
                rows.append({"seed": seed, "dataset": DATASET, "variant": variant,
                             "vectors_indexed": n_indexed, "recall10": round(rec, 4)})

        measure(0, n_initial)
        for b in range(n_batches):
            start = n_initial + b * batch_size
            end = min(start + batch_size, n_total)
            log(f"  batch {b + 1}: add {start // 1_000_000}M→{end // 1_000_000}M…")
            batch = np.asarray(base[start:end])
            batch_normed = normalize(batch)
            ivf_flat.add(batch_normed)
            ivf_tq.add(batch)
            ivf_tq._raw_vectors = None
            del batch, batch_normed
            gc.collect()
            measure(b + 1, end)

        del ivf_flat, ivf_tq
        gc.collect()
        pd.DataFrame(rows).to_csv(RESULTS_DIR / f"streaming_uncompressed_{DATASET}.csv",
                                  index=False)
        log(f"=== seed {seed} done in {time.time() - t0:.0f}s; {len(rows)} rows ===")

    return rows


def main():
    global DATASET
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sift10m", choices=VALID_DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 7777])
    args = ap.parse_args()
    DATASET = args.dataset
    rows = run(args.seeds)
    out = RESULTS_DIR / f"streaming_uncompressed_{DATASET}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    log(f"\nFinal CSV: {out} ({len(rows)} rows)")

    df = pd.DataFrame(rows)
    log("\n=== corpus-growth deltas (10M - 1M), mean over seeds ===")
    for v in ["uncompressed_ivf", "ivf_tq"]:
        s = df[df["variant"] == v]
        d1 = s[s["vectors_indexed"] == 1_000_000].set_index("seed")["recall10"]
        d10 = s[s["vectors_indexed"] == 10_000_000].set_index("seed")["recall10"]
        log(f"  {v:16s}: 1M={d1.mean():.2f} -> 10M={d10.mean():.2f}  "
            f"delta={(d10 - d1).mean():+.2f}pp")
    # Final coherence verdict at 10M
    piv = df[df["vectors_indexed"] == 10_000_000].pivot_table(
        index="seed", columns="variant", values="recall10")
    worst = (piv["ivf_tq"] - piv["uncompressed_ivf"]).max()
    log(f"\nCOHERENCE @10M: max(TQ - exact) over seeds = {worst:+.2f}pp "
        f"({'OK' if worst <= 0.5 else 'RED FLAG — audit harness before Experiment 2'})")


if __name__ == "__main__":
    main()
