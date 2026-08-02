"""
Experiment 2: rank-margin / candidate analysis for the corpus-growth mechanism.

Confirms (or refutes) the mechanism inferred from Experiments 0-1:
  corpus growth -> (i) coverage TAILWIND (more true neighbours land in probed
  cells) but (ii) shrinking top-k MARGINS, so a quantizer's fixed score
  distortion flips more rankings. Net sign is set by distortion magnitude:
  exact -> up, IVF-TQ (low) -> flat, IVF-PQ (high) -> down.

Snapshots at N=1M and N=10M (SIFT-10M, nprobe=20), coarse partitions frozen on
the first 1M. Measures:

(a) CANDIDATE RECALL (coverage): fraction of the true top-10 whose assigned
    coarse cell is among the query's nprobe probed cells, BEFORE any ranking.
    Computed separately for the FAISS partition (IVF-PQ) and the TQ partition.
    Prediction: high and RISING with N -> misses are misrankings within the
    candidate set, not ejections from it.

(b) TOP-k MARGIN: distribution of s_10 - s_11 (exact-IP gap at the rank-10/11
    decision boundary). Prediction: SHRINKS with N.

(c) SCORE ERROR vs MARGIN: for each method, per-(query, true-neighbour)
    quantization score error |<q,x> - <q,x_hat>| (x_hat = the method's own
    reconstruction; PQ via faiss.reconstruct, TQ via its Lloyd-Max+sign decode).
    Prediction: RMS error is ~N-independent (fixed codebook) and larger for PQ
    than TQ; margins shrink; so error/margin (fraction of neighbours with
    error > margin) rises with N, more for PQ. That is the flip mechanism.

A self-check prints the correlation between each method's estimated neighbour
scores and the exact scores (should be high) so a broken decode is visible.

Usage: python experiments/rank_margin.py --seeds 42 123 --states 1 10
Output: experiments/results/rank_margin_sift10m.csv
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
    set_all_seeds, verify_determinism, normalize, make_ivfpq, _load_10m_cached, log,
)

RESULTS_DIR = ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_CFG = {"sift10m": (64, 10), "deep10m": (48, 10), "t2i10m": (100, 10)}
DATASET = "sift10m"
M_PQ, BITS_PER_SUB = DATASET_CFG[DATASET]
NLIST, NPROBE = 3162, 20
BITS_TQ = 4


def coverage(centroids, assign_of_neighbors, queries_normed, true_top10, nprobe):
    """Fraction of true top-10 whose coarse cell is in the query's nprobe cells."""
    qc = queries_normed @ centroids.T                          # (nq, nlist)
    probed = np.argpartition(-qc, nprobe, axis=1)[:, :nprobe]   # (nq, nprobe)
    nq = queries_normed.shape[0]
    hit = 0
    tot = 0
    for q in range(nq):
        pset = set(probed[q].tolist())
        cells = assign_of_neighbors[q]                          # (10,)
        hit += sum(1 for c in cells if c in pset)
        tot += len(cells)
    return 100.0 * hit / tot


def tq_reconstruct(ivftq, ids):
    """Reconstruct IVF-TQ's approximation for the given global vector ids."""
    # Build id -> (partition, position) once.
    if not hasattr(ivftq, "_id2pos"):
        id2pos = {}
        for l, il in enumerate(ivftq._invlists):
            for pos, vid in enumerate(il):
                id2pos[vid] = (l, pos)
        ivftq._id2pos = id2pos
    id2pos = ivftq._id2pos
    R = ivftq.rotation_matrix
    out = np.zeros((len(ids), ivftq.dim), dtype=np.float32)
    for k, vid in enumerate(ids):
        l, pos = id2pos[int(vid)]
        part = ivftq._partitions[l]
        idx = part["indices"][pos]              # (dim,)
        if ivftq.use_residual_sign and part["sign_bits"] is not None:
            sgn = part["sign_bits"][pos]
            recon_rot = ivftq.sub_centroids[idx, sgn] * part["norms"][pos]
        else:
            recon_rot = ivftq.tq_centroids[idx] * part["norms"][pos]
        residual_hat = recon_rot @ R            # inverse rotation
        out[k] = ivftq.coarse_centroids[l] + residual_hat
    return out


def pq_reconstruct(index, ids):
    index.make_direct_map()
    try:
        return np.stack([index.reconstruct(int(i)) for i in ids]).astype(np.float32)
    except Exception:
        out = np.zeros((len(ids), index.d), dtype=np.float32)
        for k, i in enumerate(ids):
            out[k] = index.reconstruct(int(i))
        return out


def run_snapshot(base, queries, queries_normed, dim, n_indexed, seed, rows):
    log(f"  snapshot N={n_indexed // 1_000_000}M seed={seed}")
    CHUNK = 1_000_000
    init_normed = normalize(np.asarray(base[:1_000_000]))

    def add_in_chunks(add_fn):
        for s in range(0, n_indexed, CHUNK):
            add_fn(normalize(np.asarray(base[s:min(s + CHUNK, n_indexed)])))
            gc.collect()

    # Exact top-11 via a chunked FAISS flat (stores one uncompressed copy; we
    # never also hold a full dense normalized array -> no OOM at 10M).
    gi = faiss.IndexFlatIP(dim)
    add_in_chunks(gi.add)
    gt_scores, gt_idx = gi.search(np.ascontiguousarray(queries_normed), 11)
    del gi; gc.collect()
    true_top10 = gt_idx[:, :10]
    margins = gt_scores[:, 9] - gt_scores[:, 10]          # (nq,)

    # Compressed indexes (coarse frozen on first 1M), added in chunks.
    pq = make_ivfpq(dim, NLIST, M_PQ, BITS_PER_SUB, NPROBE, seed, init_normed)
    add_in_chunks(pq.add)
    tq = IVFTurboQuantIndex(dim, nlist=NLIST, bits=BITS_TQ, nprobe=NPROBE,
                            use_residual_sign=True, seed=seed)
    tq.train(init_normed)
    add_in_chunks(tq.add)
    tq._raw_vectors = None

    # Method recall (sanity: should match the oracle/uncompressed runs).
    _, pqI = pq.search(queries_normed, 10)
    _, tqI = tq.search(queries, 10)
    pq_rec = compute_recall(true_top10, pqI, 10) * 100
    tq_rec = compute_recall(true_top10, tqI, 10) * 100

    # (a) coverage for both partitions. PQ coarse quantizer is an IndexFlatIP.
    faiss_cent = faiss.downcast_index(pq.quantizer).reconstruct_n(0, NLIST).astype(np.float32)
    tq_cent = tq.coarse_centroids
    nbr_flat = true_top10.reshape(-1)
    nbr_vecs = normalize(np.asarray(base[nbr_flat]))
    assign_faiss = np.argmax(nbr_vecs @ faiss_cent.T, axis=1).reshape(-1, 10)
    assign_tq = np.argmax(nbr_vecs @ tq_cent.T, axis=1).reshape(-1, 10)
    cov_faiss = coverage(faiss_cent, assign_faiss, queries_normed, true_top10, NPROBE)
    cov_tq = coverage(tq_cent, assign_tq, queries_normed, true_top10, NPROBE)

    # (c) score error on the true top-10 neighbours
    uids, inv = np.unique(nbr_flat, return_inverse=True)
    pq_hat = pq_reconstruct(pq, uids)
    tq_hat = tq_reconstruct(tq, uids)
    # est scores for each (query, neighbour) pair
    qrep = np.repeat(np.arange(len(queries)), 10)
    pq_est = np.sum(queries_normed[qrep] * pq_hat[inv], axis=1)
    tq_est = np.sum(queries_normed[qrep] * tq_hat[inv], axis=1)
    exact = gt_scores[:, :10].reshape(-1)
    err_pq = np.abs(exact - pq_est)
    err_tq = np.abs(exact - tq_est)
    # self-check correlations
    corr_pq = float(np.corrcoef(exact, pq_est)[0, 1])
    corr_tq = float(np.corrcoef(exact, tq_est)[0, 1])
    # per-boundary: compare error on the rank-10 neighbour to that query's margin
    err_pq_b10 = err_pq.reshape(-1, 10)[:, 9]
    err_tq_b10 = err_tq.reshape(-1, 10)[:, 9]
    frac_pq = 100.0 * np.mean(err_pq_b10 > margins)
    frac_tq = 100.0 * np.mean(err_tq_b10 > margins)

    log(f"    recall: PQ={pq_rec:.2f} TQ={tq_rec:.2f} | "
        f"cov(faiss)={cov_faiss:.2f} cov(tq)={cov_tq:.2f} | "
        f"margin med={np.median(margins):.4f} p10={np.percentile(margins,10):.4f}")
    log(f"    err RMS: PQ={np.sqrt(np.mean(err_pq**2)):.4f} "
        f"TQ={np.sqrt(np.mean(err_tq**2)):.4f} | "
        f"frac(err>margin @rank10): PQ={frac_pq:.1f}% TQ={frac_tq:.1f}% | "
        f"corr PQ={corr_pq:.3f} TQ={corr_tq:.3f}")

    rows.append({
        "seed": seed, "n_indexed": n_indexed,
        "pq_recall": round(pq_rec, 3), "tq_recall": round(tq_rec, 3),
        "cov_faiss": round(cov_faiss, 3), "cov_tq": round(cov_tq, 3),
        "margin_median": round(float(np.median(margins)), 5),
        "margin_p10": round(float(np.percentile(margins, 10)), 5),
        "err_rms_pq": round(float(np.sqrt(np.mean(err_pq**2))), 5),
        "err_rms_tq": round(float(np.sqrt(np.mean(err_tq**2))), 5),
        "frac_err_gt_margin_pq": round(float(frac_pq), 2),
        "frac_err_gt_margin_tq": round(float(frac_tq), 2),
        "corr_pq": round(corr_pq, 4), "corr_tq": round(corr_tq, 4),
    })
    del pq, tq, init_normed, nbr_vecs, pq_hat, tq_hat
    gc.collect()


def main():
    global DATASET, M_PQ, BITS_PER_SUB
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sift10m", choices=list(DATASET_CFG))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123])
    ap.add_argument("--states", nargs="+", type=int, default=[1, 10],
                    help="checkpoints in millions")
    args = ap.parse_args()
    DATASET = args.dataset
    M_PQ, BITS_PER_SUB = DATASET_CFG[DATASET]

    base, queries = _load_10m_cached(DATASET)
    _, dim = base.shape
    queries_normed = normalize(queries)
    log(f"dataset={DATASET} dim={dim} states={args.states}M seeds={args.seeds}")

    rows: List[dict] = []
    for seed in args.seeds:
        log(f"\n=== seed {seed} (determinism {verify_determinism(seed)}) ===")
        for nM in args.states:
            set_all_seeds(seed)
            t0 = time.time()
            run_snapshot(base, queries, queries_normed, dim, nM * 1_000_000, seed, rows)
            pd.DataFrame(rows).to_csv(RESULTS_DIR / f"rank_margin_{DATASET}.csv", index=False)
            log(f"    done in {time.time() - t0:.0f}s")

    log(f"\nFinal CSV: {RESULTS_DIR / f'rank_margin_{DATASET}.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
