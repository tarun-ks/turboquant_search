"""
Causal per-miss decomposition for the corpus-growth mechanism.

For each true top-10 neighbor RETRIEVED at N=1M by IVF-PQ (stale codebook)
but LOST at N=10M, classify the cause:

  (a) COVERAGE loss: the neighbour's coarse cell is not among the nprobe=20
      probed cells at N=10M — pure IVF allocation failure, unrelated to
      quantization.
  (b) RANKING loss (cell IS probed at 10M): the neighbour was present in
      the candidate set but displaced from the returned top-k by score
      errors. For these: was quantization err > exact top-k margin?

The fraction of ranking losses where err > margin is the direct causal test
of the margin-crossing mechanism.

  - HIGH frac (>70%)  → margin-crossing IS the cause; proof-strength claim.
  - LOW frac (<50%)   → something else (e.g. multi-candidate crowding)
                        dominates; frame as strong evidence, not proof.

Protocol
--------
IVF-PQ trained on first 1M, FROZEN (stale). Coarse partition = same 1M
k-means, frozen. Both N=1M and N=10M use the identical index; we just add
more vectors to it.

"lost" (q, n): n ∈ GT@10M, n < 1_000_000 (so it was present at 1M),
              n ∈ pq_results@1M[q], n ∉ pq_results@10M[q].

Margin = exact score gap s_10 - s_11 at N=10M (from FAISS flat GT top-11).
Coverage at 10M: q's nprobe probed cells = argmax_{nprobe}(q @ coarse_cent.T).

Output: experiments/results/causal_miss_{DATASET}.csv
Usage:  python experiments/causal_miss.py [--dataset sift10m] [--seeds 42 123 7777]
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
NLIST, NPROBE = 3162, 20


def pq_reconstruct_batch(pq_index, ids: np.ndarray) -> np.ndarray:
    """Reconstruct PQ approximations for given vector ids."""
    pq_index.make_direct_map()
    dim = pq_index.d
    out = np.zeros((len(ids), dim), dtype=np.float32)
    for k, i in enumerate(ids):
        out[k] = pq_index.reconstruct(int(i))
    return out


def probed_cells(queries_normed: np.ndarray, coarse_centroids: np.ndarray,
                 nprobe: int) -> list[set]:
    """For each query return the set of nprobe coarse cell indices it would probe."""
    qc = queries_normed @ coarse_centroids.T           # (nq, nlist)
    top = np.argpartition(-qc, nprobe, axis=1)[:, :nprobe]
    return [set(top[q].tolist()) for q in range(len(queries_normed))]


def run_one_seed(base, queries, queries_normed, dim, m_pq, bits_per_sub,
                 seed: int, rows: list):
    log(f"\n=== seed {seed} ===")
    n_1m = 1_000_000
    n_10m = len(base)

    init_normed = normalize(np.asarray(base[:n_1m]))
    CHUNK = 1_000_000

    # ── Build stale PQ (trained + frozen on first 1M) ─────────────────────
    log("  training IVF-PQ on first 1M, freezing…")
    pq = make_ivfpq(dim, NLIST, m_pq, bits_per_sub, NPROBE, seed, init_normed)
    pq.add(init_normed)

    # ── GT@1M and PQ results@1M ────────────────────────────────────────────
    log("  computing GT@1M and PQ@1M…")
    gi_1m = faiss.IndexFlatIP(dim)
    gi_1m.add(init_normed)
    gt_scores_1m, gt_idx_1m = gi_1m.search(np.ascontiguousarray(queries_normed), 11)
    del gi_1m; gc.collect()
    true_top10_1m = gt_idx_1m[:, :10]

    _, pq_I_1m = pq.search(queries_normed, 10)
    rec_1m = compute_recall(true_top10_1m, pq_I_1m, 10) * 100
    log(f"  PQ recall@10 at 1M: {rec_1m:.2f}%")

    # ── Grow index to 10M ──────────────────────────────────────────────────
    log("  growing PQ index 1M → 10M…")
    del init_normed; gc.collect()
    for s in range(n_1m, n_10m, CHUNK):
        chunk = normalize(np.asarray(base[s:min(s + CHUNK, n_10m)]))
        pq.add(chunk)
        del chunk; gc.collect()
        log(f"    added through {min(s + CHUNK, n_10m) // CHUNK}M")

    # ── GT@10M (top-11 for margin computation) ─────────────────────────────
    log("  computing GT@10M…")
    gi_10m = faiss.IndexFlatIP(dim)
    for s in range(0, n_10m, CHUNK):
        gi_10m.add(normalize(np.asarray(base[s:min(s + CHUNK, n_10m)])))
        gc.collect()
    gt_scores_10m, gt_idx_10m = gi_10m.search(np.ascontiguousarray(queries_normed), 11)
    del gi_10m; gc.collect()
    true_top10_10m = gt_idx_10m[:, :10]
    margins = gt_scores_10m[:, 9] - gt_scores_10m[:, 10]   # (nq,)

    # ── PQ results@10M ─────────────────────────────────────────────────────
    _, pq_I_10m = pq.search(queries_normed, 10)
    rec_10m = compute_recall(true_top10_10m, pq_I_10m, 10) * 100
    log(f"  PQ recall@10 at 10M: {rec_10m:.2f}%  Δ={rec_10m - rec_1m:+.2f}pp")

    # ── Coarse centroids (for coverage check at 10M) ───────────────────────
    coarse_cent = faiss.downcast_index(pq.quantizer).reconstruct_n(0, NLIST).astype(np.float32)
    probed_at_10m = probed_cells(queries_normed, coarse_cent, NPROBE)

    # ── Neighbour assignment at 10M ────────────────────────────────────────
    # We need to know which coarse cell each true-top-10 neighbour lands in.
    # Only neighbours that exist in the first 1M can have been retrieved at 1M.
    nq = len(queries)
    lost_total = 0
    coverage_loss = 0
    ranking_loss = 0
    ranking_margin_crossed = 0
    per_query_rows = []

    # Collect all unique neighbour ids from GT@10M that are in the first 1M
    # (so they could have been retrieved at 1M).
    nbr_flat_10m = true_top10_10m.reshape(-1)           # (nq*10,)
    in_1m_mask = nbr_flat_10m < n_1m
    unique_nbr_ids = np.unique(nbr_flat_10m[in_1m_mask])

    log(f"  reconstructing {len(unique_nbr_ids)} unique neighbours in first 1M…")
    pq_hat = pq_reconstruct_batch(pq, unique_nbr_ids)
    id_to_pos = {int(vid): k for k, vid in enumerate(unique_nbr_ids)}

    # Also need to know each neighbour's coarse cell assignment at 10M.
    # The PQ index uses the same coarse partition; the cell of each vector is
    # determined by argmax of its inner product with coarse centroids.
    nbr_normed = normalize(np.asarray(base[unique_nbr_ids]))
    cell_scores = nbr_normed @ coarse_cent.T
    nbr_cells = np.argmax(cell_scores, axis=1)   # (n_unique,)
    id_to_cell = {int(vid): int(nbr_cells[k]) for k, vid in enumerate(unique_nbr_ids)}
    del nbr_normed, cell_scores; gc.collect()

    log("  computing per-miss causal breakdown…")
    for q in range(nq):
        true_nbrs = set(true_top10_10m[q].tolist())
        retrieved_1m = set(pq_I_1m[q].tolist())
        retrieved_10m = set(pq_I_10m[q].tolist())
        margin_q = float(margins[q])
        probed_q = probed_at_10m[q]

        # Eligible: true neighbours in GT@10M that are in the first 1M
        eligible = [n for n in true_nbrs if n < n_1m]
        for n in eligible:
            if n not in retrieved_1m:
                continue   # not a "hit at 1M" → skip
            if n in retrieved_10m:
                continue   # still retrieved at 10M → not lost

            # This is a "lost" neighbour.
            lost_total += 1
            cell_n = id_to_cell[n]
            if cell_n not in probed_q:
                coverage_loss += 1
                mechanism = "coverage"
            else:
                # Ranking loss: cell was probed, but didn't make top-k.
                pos = id_to_pos[n]
                q_vec = queries_normed[q]
                x_hat = pq_hat[pos]
                exact_score = float(np.dot(q_vec, normalize(np.asarray(base[n:n+1]))[0]))
                est_score = float(np.dot(q_vec, x_hat))
                err = abs(exact_score - est_score)
                ranking_loss += 1
                if err > margin_q:
                    ranking_margin_crossed += 1
                    mechanism = "ranking_margin_crossed"
                else:
                    mechanism = "ranking_other"

            per_query_rows.append({
                "seed": seed,
                "query": q,
                "neighbor": n,
                "mechanism": mechanism,
                "margin_q": round(margin_q, 6),
            })

    n_ranking = ranking_loss
    frac_margin = 100.0 * ranking_margin_crossed / max(1, n_ranking)
    frac_coverage = 100.0 * coverage_loss / max(1, lost_total)
    frac_total_explained = 100.0 * (coverage_loss + ranking_margin_crossed) / max(1, lost_total)

    log(f"\n  ── Per-miss causal breakdown (seed={seed}) ──")
    log(f"  Lost neighbours (GT@10M, in first 1M, hit@1M, miss@10M): {lost_total}")
    log(f"  Coverage losses (cell not probed at 10M): {coverage_loss} "
        f"({frac_coverage:.1f}%)")
    log(f"  Ranking losses (cell probed, not top-k):  {n_ranking} "
        f"({100 - frac_coverage:.1f}%)")
    log(f"    of which err > margin:  {ranking_margin_crossed} "
        f"({frac_margin:.1f}% of ranking losses)")
    log(f"    of which err ≤ margin:  {n_ranking - ranking_margin_crossed}")
    log(f"  Total explained by mechanism: {frac_total_explained:.1f}%")
    log(f"  PQ recall: 1M={rec_1m:.2f}% → 10M={rec_10m:.2f}% (Δ={rec_10m-rec_1m:+.2f}pp)")

    rows.append({
        "seed": seed,
        "rec_1m": round(rec_1m, 3),
        "rec_10m": round(rec_10m, 3),
        "delta_pp": round(rec_10m - rec_1m, 3),
        "lost_total": lost_total,
        "coverage_loss": coverage_loss,
        "ranking_loss": n_ranking,
        "ranking_margin_crossed": ranking_margin_crossed,
        "frac_coverage_pct": round(frac_coverage, 2),
        "frac_margin_crossed_of_ranking_pct": round(frac_margin, 2),
        "frac_total_explained_pct": round(frac_total_explained, 2),
    })

    del pq, pq_hat, id_to_pos, id_to_cell, nbr_flat_10m
    gc.collect()


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

    rows: list = []
    for seed in args.seeds:
        set_all_seeds(seed)
        t0 = time.time()
        run_one_seed(base, queries, queries_normed, dim, m_pq, bits_per_sub, seed, rows)
        df = pd.DataFrame(rows)
        out = RESULTS_DIR / f"causal_miss_{dataset}.csv"
        df.to_csv(out, index=False)
        log(f"  seed {seed} done in {time.time() - t0:.0f}s; written {out}")

    df = pd.DataFrame(rows)
    log("\n=== Summary across seeds ===")
    cols = ["frac_coverage_pct", "frac_margin_crossed_of_ranking_pct",
            "frac_total_explained_pct", "delta_pp"]
    for col in cols:
        log(f"  {col}: {df[col].mean():.2f} ± {df[col].std():.2f}")


if __name__ == "__main__":
    main()
