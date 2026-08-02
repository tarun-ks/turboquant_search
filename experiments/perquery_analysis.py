"""
Unified per-query analysis: causal-miss decomposition + bootstrap data in one pass.

Single data structure read by both analyses:
  - bootstrap_ci.py  : reads margin_10m, err_pq_b10_10m, hit counts → CIs
  - causal decomposition: reads n_cov_lost, n_rank_margin, n_rank_other → proof check

Output: experiments/results/perquery_{DATASET}.csv
Columns (one row per (seed, query_idx)):
  seed, dataset, query_idx
  hit_count_pq_1m  : # true top10 retrieved by PQ at N=1M
  hit_count_pq_10m : # true top10 retrieved by PQ at N=10M
  hit_count_tq_1m  : # true top10 retrieved by TQ at N=1M
  hit_count_tq_10m : # true top10 retrieved by TQ at N=10M
  margin_10m       : exact score gap s_10 - s_11 at N=10M
  err_pq_b10_10m   : |<q,x10> - <q,x_hat_pq_10>| (rank-10 boundary PQ error)
  err_tq_b10_10m   : same for TQ
  n_lost           : # neighbors in GT@10M, in first 1M, hit@1M, miss@10M
  n_cov_lost       : of those, # with coarse cell not probed at N=10M
  n_rank_margin    : # ranking losses (cell probed, not top-k) with err > margin
  n_rank_other     : # ranking losses with err <= margin

Usage:
    python experiments/perquery_analysis.py --dataset sift10m --seeds 42 123 7777
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
from turboquant_search.faiss_baselines import FAISS_AVAILABLE
from turboquant_search.benchmarks import compute_recall

assert FAISS_AVAILABLE, "faiss-cpu required"
import faiss  # noqa: E402

from streaming_multiseed import (  # noqa: E402
    set_all_seeds, verify_determinism, normalize, make_ivfpq, _load_10m_cached, log,
)

RESULTS_DIR = ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_CFG = {"sift10m": (64, 10), "deep10m": (48, 10), "t2i10m": (100, 10)}
NLIST, NPROBE = 3162, 20
BITS_TQ = 4
CHUNK = 1_000_000


def _pq_reconstruct_batch(pq_index: faiss.Index, ids: np.ndarray) -> np.ndarray:
    pq_index.make_direct_map()
    dim = pq_index.d
    out = np.zeros((len(ids), dim), dtype=np.float32)
    for k, i in enumerate(ids):
        out[k] = pq_index.reconstruct(int(i))
    return out


def _tq_reconstruct_batch(ivftq, ids: np.ndarray) -> np.ndarray:
    if not hasattr(ivftq, "_id2pos"):
        id2pos: dict = {}
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
        idx = part["indices"][pos]
        if ivftq.use_residual_sign and part["sign_bits"] is not None:
            sgn = part["sign_bits"][pos]
            recon_rot = ivftq.sub_centroids[idx, sgn] * part["norms"][pos]
        else:
            recon_rot = ivftq.tq_centroids[idx] * part["norms"][pos]
        out[k] = ivftq.coarse_centroids[l] + recon_rot @ R
    return out


def _probed_cell_sets(queries_normed: np.ndarray, coarse_cent: np.ndarray,
                      nprobe: int) -> list:
    qc = queries_normed @ coarse_cent.T
    top = np.argpartition(-qc, nprobe, axis=1)[:, :nprobe]
    return [set(top[q].tolist()) for q in range(len(queries_normed))]


def _cell_of(vecs_normed: np.ndarray, coarse_cent: np.ndarray) -> np.ndarray:
    return np.argmax(vecs_normed @ coarse_cent.T, axis=1)


def run_one_seed(base, queries, queries_normed, dim, m_pq, bits_per_sub,
                 seed: int, out_rows: list) -> None:
    log(f"\n=== seed {seed} (det {verify_determinism(seed)[:20]}) ===")
    n_1m = CHUNK
    n_10m = len(base)
    t0 = time.time()

    init_normed = normalize(np.asarray(base[:n_1m]))

    # ── Build PQ (stale) and TQ, both frozen on first 1M ──────────────
    log("  training PQ + TQ on first 1M…")
    pq = make_ivfpq(dim, NLIST, m_pq, bits_per_sub, NPROBE, seed, init_normed)
    pq.add(init_normed)

    tq = IVFTurboQuantIndex(dim, nlist=NLIST, bits=BITS_TQ, nprobe=NPROBE,
                            use_residual_sign=True, seed=seed)
    tq.train(np.asarray(base[:n_1m]))
    tq.add(np.asarray(base[:n_1m]))
    tq._raw_vectors = None

    # ── GT@1M ──────────────────────────────────────────────────────────
    log("  GT@1M…")
    gi1 = faiss.IndexFlatIP(dim)
    gi1.add(init_normed)
    gt1_scores, gt1_idx = gi1.search(np.ascontiguousarray(queries_normed), 10)
    del gi1; gc.collect()
    true_top10_1m = gt1_idx   # (nq, 10)

    # ── PQ@1M and TQ@1M results ─────────────────────────────────────────
    _, pq_I_1m = pq.search(queries_normed, 10)
    _, tq_I_1m = tq.search(queries, 10)
    del init_normed; gc.collect()

    # ── Grow both indexes to 10M ────────────────────────────────────────
    log("  growing 1M→10M…")
    for s in range(n_1m, n_10m, CHUNK):
        end = min(s + CHUNK, n_10m)
        chunk_raw = np.asarray(base[s:end])
        chunk_normed = normalize(chunk_raw)
        pq.add(chunk_normed)
        tq.add(chunk_raw)
        tq._raw_vectors = None
        del chunk_raw, chunk_normed; gc.collect()
        log(f"    added through {end // CHUNK}M")

    # ── GT@10M (top-11 for margin) ──────────────────────────────────────
    log("  GT@10M…")
    gi10 = faiss.IndexFlatIP(dim)
    for s in range(0, n_10m, CHUNK):
        gi10.add(normalize(np.asarray(base[s:min(s + CHUNK, n_10m)])))
        gc.collect()
    gt10_scores, gt10_idx = gi10.search(np.ascontiguousarray(queries_normed), 11)
    del gi10; gc.collect()
    true_top10_10m = gt10_idx[:, :10]    # (nq, 10)
    margins = gt10_scores[:, 9] - gt10_scores[:, 10]  # (nq,)

    # ── PQ@10M and TQ@10M results ───────────────────────────────────────
    _, pq_I_10m = pq.search(queries_normed, 10)
    _, tq_I_10m = tq.search(queries, 10)

    pq_rec10 = compute_recall(true_top10_10m, pq_I_10m, 10) * 100
    tq_rec10 = compute_recall(true_top10_10m, tq_I_10m, 10) * 100
    log(f"  recall@10 at 10M — PQ={pq_rec10:.2f}% TQ={tq_rec10:.2f}%")

    # ── Per-query boundary error (ALL queries, rank-10 neighbor) ────────
    log("  computing per-query boundary errors (all queries)…")
    nbr_rank10 = true_top10_10m[:, 9]    # (nq,) — 10th true neighbor at 10M
    u_ids, inv_ids = np.unique(nbr_rank10, return_inverse=True)

    # Normalize from base (raw)
    nbr_vecs_r10 = normalize(np.asarray(base[u_ids]))          # (n_unique, dim)

    pq_hat_r10 = _pq_reconstruct_batch(pq, u_ids)              # (n_unique, dim)
    tq_hat_r10 = _tq_reconstruct_batch(tq, u_ids)              # (n_unique, dim)

    exact_r10 = np.sum(queries_normed * nbr_vecs_r10[inv_ids], axis=1)  # (nq,)
    pq_est_r10 = np.sum(queries_normed * pq_hat_r10[inv_ids], axis=1)   # (nq,)
    tq_est_r10 = np.sum(queries_normed * tq_hat_r10[inv_ids], axis=1)   # (nq,)
    err_pq_b10 = np.abs(exact_r10 - pq_est_r10)   # (nq,)
    err_tq_b10 = np.abs(exact_r10 - tq_est_r10)   # (nq,)
    del nbr_vecs_r10, pq_hat_r10, tq_hat_r10; gc.collect()

    # ── Per-query hit counts ─────────────────────────────────────────────
    nq = len(queries)
    true_top10_1m_sets  = [set(true_top10_1m[q].tolist()) for q in range(nq)]
    true_top10_10m_sets = [set(true_top10_10m[q].tolist()) for q in range(nq)]
    pq_I_1m_sets        = [set(pq_I_1m[q].tolist()) for q in range(nq)]
    pq_I_10m_sets       = [set(pq_I_10m[q].tolist()) for q in range(nq)]
    tq_I_1m_sets        = [set(tq_I_1m[q].tolist()) for q in range(nq)]
    tq_I_10m_sets       = [set(tq_I_10m[q].tolist()) for q in range(nq)]

    pq_hit_1m  = np.array([len(true_top10_1m_sets[q] & pq_I_1m_sets[q]) for q in range(nq)])
    pq_hit_10m = np.array([len(true_top10_10m_sets[q] & pq_I_10m_sets[q]) for q in range(nq)])
    tq_hit_1m  = np.array([len(true_top10_1m_sets[q] & tq_I_1m_sets[q]) for q in range(nq)])
    tq_hit_10m = np.array([len(true_top10_10m_sets[q] & tq_I_10m_sets[q]) for q in range(nq)])

    # ── Causal-miss decomposition (PQ, GT@10M vs GT@1M) ─────────────────
    log("  causal-miss decomposition…")
    # Coarse centroids for coverage check at 10M
    coarse_cent = faiss.downcast_index(pq.quantizer).reconstruct_n(0, NLIST).astype(np.float32)
    probed_at_10m = _probed_cell_sets(queries_normed, coarse_cent, NPROBE)

    # Neighbours in GT@10M that are in the first 1M → eligible for causal miss
    nbr_flat_10m = true_top10_10m.reshape(-1)
    in_1m_mask = nbr_flat_10m < n_1m
    u_nbr = np.unique(nbr_flat_10m[in_1m_mask])

    nbr_normed_all = normalize(np.asarray(base[u_nbr]))
    cell_all = _cell_of(nbr_normed_all, coarse_cent)
    id2cell = {int(vid): int(cell_all[k]) for k, vid in enumerate(u_nbr)}
    del nbr_normed_all, cell_all; gc.collect()

    # Reconstruct only these neighbours for error computation
    pq_hat_nbr = _pq_reconstruct_batch(pq, u_nbr)
    id2pos_pq = {int(vid): k for k, vid in enumerate(u_nbr)}

    n_lost_arr   = np.zeros(nq, dtype=np.int32)
    n_cov_arr    = np.zeros(nq, dtype=np.int32)
    n_rank_m_arr = np.zeros(nq, dtype=np.int32)
    n_rank_o_arr = np.zeros(nq, dtype=np.int32)

    for q in range(nq):
        true_nbrs = true_top10_10m_sets[q]
        ret_1m    = pq_I_1m_sets[q]
        ret_10m   = pq_I_10m_sets[q]
        margin_q  = float(margins[q])
        probed_q  = probed_at_10m[q]
        q_vec     = queries_normed[q]

        for n in true_nbrs:
            if n >= n_1m:
                continue           # not in first 1M → cannot have been retrieved @1M
            if n not in ret_1m:
                continue           # not a hit at 1M → not a "lost" neighbor
            if n in ret_10m:
                continue           # still retrieved @10M → not lost

            n_lost_arr[q] += 1
            if id2cell[n] not in probed_q:
                n_cov_arr[q] += 1
            else:
                pos = id2pos_pq[n]
                q_nbr_normed = normalize(np.asarray(base[n:n + 1]))[0]
                exact_s = float(np.dot(q_vec, q_nbr_normed))
                est_s   = float(np.dot(q_vec, pq_hat_nbr[pos]))
                err_n   = abs(exact_s - est_s)
                if err_n > margin_q:
                    n_rank_m_arr[q] += 1
                else:
                    n_rank_o_arr[q] += 1

    del pq, tq, pq_hat_nbr, id2pos_pq, id2cell; gc.collect()

    total_lost    = int(n_lost_arr.sum())
    total_cov     = int(n_cov_arr.sum())
    total_rank_m  = int(n_rank_m_arr.sum())
    total_rank_o  = int(n_rank_o_arr.sum())
    frac_cov = 100.0 * total_cov / max(1, total_lost)
    frac_mrg = 100.0 * total_rank_m / max(1, total_lost - total_cov)
    log(f"  causal miss: {total_lost} lost | cov={total_cov} ({frac_cov:.1f}%) | "
        f"rank_margin={total_rank_m} ({frac_mrg:.1f}% of ranking losses)")
    log(f"  elapsed {time.time() - t0:.0f}s")

    # ── Write per-query rows ─────────────────────────────────────────────
    for q in range(nq):
        out_rows.append({
            "seed": seed,
            "query_idx": q,
            "hit_count_pq_1m":  int(pq_hit_1m[q]),
            "hit_count_pq_10m": int(pq_hit_10m[q]),
            "hit_count_tq_1m":  int(tq_hit_1m[q]),
            "hit_count_tq_10m": int(tq_hit_10m[q]),
            "margin_10m":       round(float(margins[q]), 7),
            "err_pq_b10_10m":   round(float(err_pq_b10[q]), 7),
            "err_tq_b10_10m":   round(float(err_tq_b10[q]), 7),
            "n_lost":           int(n_lost_arr[q]),
            "n_cov_lost":       int(n_cov_arr[q]),
            "n_rank_margin":    int(n_rank_m_arr[q]),
            "n_rank_other":     int(n_rank_o_arr[q]),
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sift10m", choices=list(DATASET_CFG))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 7777])
    args = ap.parse_args()

    dataset = args.dataset
    m_pq, bits_per_sub = DATASET_CFG[dataset]

    base, queries = _load_10m_cached(dataset)
    _, dim = base.shape
    queries_normed = normalize(queries)
    log(f"dataset={dataset} N={len(base):,} dim={dim} m_pq={m_pq} b={bits_per_sub} "
        f"nprobe={NPROBE} seeds={args.seeds}")

    out_rows: list = []
    for seed in args.seeds:
        set_all_seeds(seed)
        run_one_seed(base, queries, queries_normed, dim, m_pq, bits_per_sub, seed, out_rows)
        df = pd.DataFrame(out_rows)
        df["dataset"] = dataset
        out = RESULTS_DIR / f"perquery_{dataset}.csv"
        df.to_csv(out, index=False)
        log(f"  written {out} ({len(df)} rows)")

    log(f"\nDone. perquery_{dataset}.csv: {len(out_rows)} rows "
        f"({len(out_rows) // len(args.seeds)} queries × {len(args.seeds)} seeds)")


if __name__ == "__main__":
    main()
