"""
Pairwise signed-error analysis: who drives score inversions?

For every in-probe missed true neighbor i and every returned false positive j
that outranks it, compute:
  e_i = ŝ_i - s_i  (true neighbor signed error)
  e_j = ŝ_j - s_j  (competitor signed error)
  m_ij = s_i - s_j  (pairwise exact margin, ≥ 0 since i is true top-k)
  z_ij = e_j - e_i  (pairwise signed perturbation)
  crossing: z_ij ≥ m_ij  (should be ~100% by definition of score miss)

Decomposition of crossing cause (four mutually exclusive, exhaustive categories):
  TN-only:   -e_i alone ≥ m_ij  AND  e_j < m_ij
  FP-only:    e_j alone ≥ m_ij  AND  -e_i < m_ij
  Both:       each error independently sufficient (-e_i ≥ m_ij AND e_j ≥ m_ij)
  Joint:      neither alone sufficient; only z_ij ≥ m_ij
  TN-dom:    -e_i > e_j  (neighbor component larger, regardless of sufficiency)
  FP-dom:     e_j > -e_i (competitor component larger)

Runs at 1M and 10M corpus sizes.
Output: experiments/results/pairwise_analysis_{dataset}.csv

Usage:
  python pairwise_score_analysis.py --dataset sift10m [--seeds 42 123 7777]
"""

import argparse
import gc
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))

from turboquant_search.faiss_baselines import FAISS_AVAILABLE
assert FAISS_AVAILABLE, "faiss-cpu required"
import faiss

from streaming_multiseed import (
    set_all_seeds, verify_determinism, normalize, make_ivfpq, _load_10m_cached, log,
)

RESULTS_DIR = ROOT / "experiments" / "results"
DATASET_CFG = {"sift10m": (64, 10), "deep10m": (48, 10), "t2i10m": (100, 10)}
NLIST, NPROBE = 3162, 20
CHUNK = 1_000_000


def _pq_reconstruct(pq_index, vec_ids: np.ndarray) -> np.ndarray:
    """Reconstruct PQ approximations for a batch of vector IDs."""
    recons = np.zeros((len(vec_ids), pq_index.d), dtype=np.float32)
    for k, vid in enumerate(vec_ids):
        recons[k] = pq_index.reconstruct(int(vid))
    return recons


def analyze_one_seed(base, queries, queries_normed, dim, m_pq, bits_per_sub, seed):
    n_10m = len(base)
    n_1m  = 1_000_000

    log(f"  seed={seed}  determinism={verify_determinism(seed)}")
    set_all_seeds(seed)

    # ── Build IVF-PQ (trained on first 1M, frozen) ──────────────────────
    init_normed = normalize(np.asarray(base[:n_1m]))
    pq = make_ivfpq(dim, NLIST, m_pq, bits_per_sub, NPROBE, seed, init_normed)
    pq.add(init_normed)  # make_ivfpq trains but does not add
    del init_normed; gc.collect()

    nq = len(queries_normed)

    # ── 1M analysis ─────────────────────────────────────────────────────
    gi1 = faiss.IndexFlatIP(dim)
    gi1.add(normalize(np.asarray(base[:n_1m])))
    gt1_scores, gt1_idx = gi1.search(np.ascontiguousarray(queries_normed), 11)
    true_top10_1m = gt1_idx[:, :10]
    margins_1m = gt1_scores[:, 9] - gt1_scores[:, 10]

    _, pq_I_1m = pq.search(queries_normed, 10)
    del gi1; gc.collect()

    rows_1m = _pairwise_decomposition(
        base, queries_normed, pq, n_1m,
        true_top10_1m, pq_I_1m, margins_1m, nq, seed, n_1m
    )

    # ── Add remaining vectors ────────────────────────────────────────────
    for s in range(n_1m, n_10m, CHUNK):
        end = min(s + CHUNK, n_10m)
        pq.add(normalize(np.asarray(base[s:end])))

    # ── 10M analysis ────────────────────────────────────────────────────
    gi10 = faiss.IndexFlatIP(dim)
    for s in range(0, n_10m, CHUNK):
        gi10.add(normalize(np.asarray(base[s:min(s+CHUNK, n_10m)])))
    gt10_scores, gt10_idx = gi10.search(np.ascontiguousarray(queries_normed), 11)
    true_top10_10m = gt10_idx[:, :10]
    margins_10m = gt10_scores[:, 9] - gt10_scores[:, 10]

    _, pq_I_10m = pq.search(queries_normed, 10)
    del gi10; gc.collect()

    rows_10m = _pairwise_decomposition(
        base, queries_normed, pq, n_10m,
        true_top10_10m, pq_I_10m, margins_10m, nq, seed, n_10m
    )

    del pq; gc.collect()
    return rows_1m + rows_10m


def _cell_of(vecs_normed: np.ndarray, coarse_cent: np.ndarray) -> np.ndarray:
    return np.argmax(vecs_normed @ coarse_cent.T, axis=1)


def _pairwise_decomposition(base, queries_normed, pq, N,
                             true_top10, pq_I, margins, nq, seed, corpus_n):
    """
    For each query: find missed in-probe true neighbors and the false positives
    that displaced them. Compute pairwise signed errors and decomposition.
    Returns list of per-query summary dicts.
    """
    import faiss as fi

    # Coarse centroids for coverage check
    coarse_cent = fi.downcast_index(pq.quantizer).reconstruct_n(0, NLIST).astype(np.float32)
    probed_cells = _probed_cell_sets(queries_normed, coarse_cent, NPROBE)

    # Collect all unique candidate IDs (true neighbors + returned results)
    true_sets = [set(v for v in true_top10[q].tolist() if 0 <= v < corpus_n)
                 for q in range(nq)]
    ret_sets  = [set(v for v in pq_I[q].tolist()       if 0 <= v < corpus_n)
                 for q in range(nq)]
    candidate_ids = set()
    for q in range(nq):
        candidate_ids |= true_sets[q] | ret_sets[q]
    candidate_ids = np.array(sorted(candidate_ids), dtype=np.int64)

    if len(candidate_ids) == 0:
        return []

    # Build id → assigned IVF cell mapping (load individually via mmap)
    log(f"    computing cell assignments for {len(candidate_ids)} candidates…")
    raw_vecs = normalize(
        np.vstack([np.asarray(base[int(v):int(v)+1]) for v in candidate_ids])
    )
    cell_arr = _cell_of(raw_vecs, coarse_cent)
    id2cell = {int(vid): int(cell_arr[k]) for k, vid in enumerate(candidate_ids)}

    # Collect all unique IDs that are in-probe missed neighbors or false positives
    all_ids_set = set()
    for q in range(nq):
        in_probe_missed = {v for v in true_sets[q] - ret_sets[q]
                           if id2cell.get(v, -1) in probed_cells[q]}
        false_pos = ret_sets[q] - true_sets[q]
        all_ids_set |= in_probe_missed | false_pos
    all_ids = np.array(sorted(all_ids_set), dtype=np.int64)

    if len(all_ids) == 0:
        return []

    # Reuse already-loaded raw_vecs for exact vectors (all_ids ⊆ candidate_ids)
    cand_id2idx = {int(vid): k for k, vid in enumerate(candidate_ids)}
    exact = raw_vecs[[cand_id2idx[int(v)] for v in all_ids]]  # (n, d), already normalized

    # PQ reconstructions via direct map
    log(f"    PQ-reconstructing {len(all_ids)} vectors…")
    pq.make_direct_map()
    recon = np.zeros((len(all_ids), pq.d), dtype=np.float32)
    for k, vid in enumerate(all_ids):
        try:
            recon[k] = pq.reconstruct(int(vid))
        except RuntimeError:
            pq_sa = np.zeros((1, pq.sa_code_size()), dtype=np.uint8)
            pq.sa_encode(exact[k:k+1], pq_sa)
            pq.sa_decode(pq_sa, recon[k:k+1])
    id2idx = {int(vid): k for k, vid in enumerate(all_ids)}

    rows = []
    total_pairs = 0
    tn_only_tot = fp_only_tot = both_tot = joint_tot = 0
    tn_dom_tot  = fp_dom_tot  = 0
    e_i_vals, e_j_vals, m_vals = [], [], []

    for q in range(nq):
        q_vec = queries_normed[q]
        in_probe_missed = sorted({v for v in (true_sets[q] - ret_sets[q])
                                   if id2cell.get(v, -1) in probed_cells[q]})
        false_pos       = sorted(ret_sets[q] - true_sets[q])

        if not in_probe_missed or not false_pos:
            continue

        for i in in_probe_missed:
            ki = id2idx[i]
            s_i  = float(np.dot(q_vec, exact[ki]))
            sh_i = float(np.dot(q_vec, recon[ki]))
            e_i  = sh_i - s_i

            # Only consider false positives that actually outrank i
            for j in false_pos:
                kj = id2idx[j]
                s_j  = float(np.dot(q_vec, exact[kj]))
                sh_j = float(np.dot(q_vec, recon[kj]))
                e_j  = sh_j - s_j

                # Only analyze pairs where j outranks i in approx scores
                if sh_j < sh_i:
                    continue

                m_ij = s_i - s_j   # exact margin (should be ≥ 0 if i is truly better)
                z_ij = e_j - e_i   # pairwise perturbation

                total_pairs += 1
                e_i_vals.append(e_i)
                e_j_vals.append(e_j)
                m_vals.append(m_ij)

                tn_sufficient = (-e_i) >= m_ij
                fp_sufficient = e_j    >= m_ij
                if   tn_sufficient and not fp_sufficient: tn_only_tot += 1
                elif fp_sufficient and not tn_sufficient: fp_only_tot += 1
                elif tn_sufficient and fp_sufficient:     both_tot    += 1
                else:                                     joint_tot   += 1

                if (-e_i) > e_j:  tn_dom_tot += 1
                else:             fp_dom_tot += 1

    if total_pairs == 0:
        return []

    e_i_arr = np.array(e_i_vals)
    e_j_arr = np.array(e_j_vals)
    m_arr   = np.array(m_vals)

    rows.append({
        "seed":        seed,
        "corpus_n":    corpus_n,
        "n_pairs":     total_pairs,
        "tn_only_pct": 100 * tn_only_tot / total_pairs,
        "fp_only_pct": 100 * fp_only_tot / total_pairs,
        "both_pct":    100 * both_tot    / total_pairs,
        "joint_pct":   100 * joint_tot   / total_pairs,
        "tn_dom_pct":  100 * tn_dom_tot  / total_pairs,
        "fp_dom_pct":  100 * fp_dom_tot  / total_pairs,
        "e_i_mean":    float(np.mean(e_i_arr)),
        "e_i_rms":     float(np.sqrt(np.mean(e_i_arr**2))),
        "e_j_mean":    float(np.mean(e_j_arr)),
        "e_j_rms":     float(np.sqrt(np.mean(e_j_arr**2))),
        "m_ij_median": float(np.median(m_arr)),
        "m_ij_mean":   float(np.mean(m_arr)),
        "neg_ei_mean": float(np.mean(-e_i_arr)),
        "ej_mean":     float(np.mean(e_j_arr)),
    })

    log(f"    corpus_n={corpus_n//1_000_000}M: {total_pairs} pairs | "
        f"TN-only {100*tn_only_tot/total_pairs:.1f}% "
        f"FP-only {100*fp_only_tot/total_pairs:.1f}% "
        f"Both {100*both_tot/total_pairs:.1f}% "
        f"Joint {100*joint_tot/total_pairs:.1f}% "
        f"TN-dom {100*tn_dom_tot/total_pairs:.1f}% "
        f"FP-dom {100*fp_dom_tot/total_pairs:.1f}%")
    return rows


def _probed_cell_sets(queries_normed, coarse_cent, nprobe):
    qc = queries_normed @ coarse_cent.T
    top = np.argpartition(-qc, nprobe, axis=1)[:, :nprobe]
    return [set(top[q].tolist()) for q in range(len(queries_normed))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sift10m", choices=list(DATASET_CFG))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 7777])
    args = ap.parse_args()

    dataset = args.dataset
    m_pq, bits_per_sub = DATASET_CFG[dataset]

    log(f"Pairwise score analysis: {dataset}, seeds={args.seeds}, "
        f"nlist={NLIST} nprobe={NPROBE}")
    base, queries = _load_10m_cached(dataset)
    queries_normed = normalize(queries)
    dim = queries.shape[1]

    all_rows = []
    for seed in args.seeds:
        log(f"\n=== seed {seed} ===")
        rows = analyze_one_seed(base, queries, queries_normed, dim,
                                m_pq, bits_per_sub, seed)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_path = RESULTS_DIR / f"pairwise_analysis_{dataset}.csv"
    df.to_csv(out_path, index=False)
    log(f"\nSaved {out_path}")

    # Print summary
    print("\n=== Pairwise decomposition summary ===")
    for N in sorted(df["corpus_n"].unique()):
        sub = df[df["corpus_n"] == N]
        print(f"\n{dataset} @ {N//1_000_000}M (n={len(sub)} seeds):")
        for col, label in [
            ("tn_only_pct", "TN-only (excl.)"),
            ("fp_only_pct", "FP-only (excl.)"),
            ("both_pct",    "Both sufficient"),
            ("joint_pct",   "Joint (neither alone)"),
            ("tn_dom_pct",  "TN-dominated (larger component)"),
            ("fp_dom_pct",  "FP-dominated (larger component)"),
        ]:
            print(f"  {label:30s}: {sub[col].mean():.1f}% ±{sub[col].std():.1f}%")


if __name__ == "__main__":
    main()
