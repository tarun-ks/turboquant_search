"""
Two follow-on experiments based on the MSB-dominance finding.

(A) Fill the missing IVF-TQ comparison cell: (b=6, b'=0) to compare against
    (b=5, b'=1) and (b=4, b'=2) at matched 6 bits/coord.

(B) Cascade search exploiting MSB importance:
    - Pass 1: score using only top-K MSBs of primary index (use coarsened
      sub_centroids that average the LSB-resolutions).
    - Pass 2: rerank top-N candidates using the FULL primary+sub encoding.
    Hypothesis: a fast-but-coarse first pass finds a high-recall candidate
    pool; the slower exact rerank on a small set preserves recall@10.

Outputs experiments/cascade_results.json.
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from turboquant_search.faiss_baselines import FAISS_AVAILABLE, FAISSFlatIndex
from turboquant_search.benchmarks import compute_recall
from turboquant_search.datasets import load_deep1m
from experiments.ivf_rvq_tq import IVFRVQTQIndex


def search_cascade(idx, queries, k=10, top_msb_bits=4, rerank_n=100):
    """
    Two-pass search.
    Pass 1: use only the top `top_msb_bits` of the primary index for ranking.
            Reconstruction value: average of all sub_centroids in the
            "coarse" bin formed by the masked-MSB primary bits.
    Pass 2: rerank top `rerank_n` from pass 1 using full precision.
    """
    full_bits = idx.bits
    if top_msb_bits >= full_bits:
        # nothing to gain
        return None

    # Build a coarse codebook: average sub_centroids over LSBs
    # full_levels = 2 ** full_bits, coarse_levels = 2 ** top_msb_bits
    # coarse_centroid[j] = mean over (i mod 2**(full_bits - top_msb_bits)) of sub_centroids[j*..., :]
    full_levels = 2 ** full_bits
    coarse_levels = 2 ** top_msb_bits
    lsb_count = full_bits - top_msb_bits
    n_lsb = 2 ** lsb_count
    n_sub = 2 ** idx.refine_bits if idx.refine_bits > 0 else 1

    # Pre-compute per-coord coarse "average centroid" for each (top_msb, sub) combo,
    # then average over sub too (since pass 1 does not use sub bits)
    if idx.refine_bits > 0:
        # average sub_centroids over (LSB, sub) for each coarse-MSB
        full_recon_vals = idx.sub_centroids  # shape (full_levels, n_sub)
        # Reshape to (coarse_levels, n_lsb, n_sub) and average
        flat_recon = full_recon_vals.reshape(coarse_levels, n_lsb, n_sub)
        coarse_recon = flat_recon.mean(axis=(1, 2)).astype(np.float32)  # (coarse_levels,)
    else:
        flat_recon = idx.tq_centroids.reshape(coarse_levels, n_lsb)
        coarse_recon = flat_recon.mean(axis=1).astype(np.float32)

    q = np.ascontiguousarray(queries.astype(np.float32))
    nq = q.shape[0]
    q_norms = np.linalg.norm(q, axis=1, keepdims=True)
    q_normed = q / np.maximum(q_norms, 1e-8)
    q_rotated = q_normed @ idx.rotation.T
    coarse_scores_qc = q_normed @ idx.coarse_centroids.T
    nprobe = min(idx.nprobe, idx.nlist)
    top_cells = np.argpartition(-coarse_scores_qc, nprobe, axis=1)[:, :nprobe]

    out_idx = np.full((nq, k), -1, dtype=np.int64)
    pass1_time = 0.0
    pass2_time = 0.0

    for qi in range(nq):
        cells = top_cells[qi]
        cell_qc = coarse_scores_qc[qi, cells]
        qrot = q_rotated[qi]

        # Pass 1: gather all candidates with coarse-MSB scoring
        t0 = time.time()
        all_ids, all_scores = [], []
        for ci, cell in enumerate(cells):
            part = idx._partitions[cell]
            if part is None:
                continue
            # Coarse primary index: top MSB bits
            primary_msb = part["primary"] >> lsb_count   # shape (n, dim)
            recon = coarse_recon[primary_msb]            # shape (n, dim)
            residual_score = (recon @ qrot) * part["norms"]
            total_score = cell_qc[ci] + residual_score
            all_ids.append(part["ids"])
            all_scores.append(total_score)
        if not all_ids:
            continue
        all_ids = np.concatenate(all_ids)
        all_scores = np.concatenate(all_scores)
        # Take top-rerank_n from pass 1
        rn = min(rerank_n, len(all_scores))
        top_pass1 = np.argpartition(-all_scores, rn - 1)[:rn]
        pass1_time += time.time() - t0

        # Pass 2: rerank with full precision
        t0 = time.time()
        cand_ids = all_ids[top_pass1]

        # Look up the full primary+sub for each candidate
        # Need to map cand_ids back to (cell, position within cell)
        # Simpler: rebuild full reconstruction for the candidates only
        # We can iterate cells and gather candidates from each
        cand_set = set(cand_ids.tolist())
        rerank_scores = []
        rerank_ids = []
        for ci, cell in enumerate(cells):
            part = idx._partitions[cell]
            if part is None:
                continue
            mask = np.isin(part["ids"], cand_ids)
            if not mask.any():
                continue
            primary_full = part["primary"][mask]
            if idx.refine_bits > 0:
                sub_full = part["sub"][mask]
                recon_full = idx.sub_centroids[primary_full, sub_full]
            else:
                recon_full = idx.tq_centroids[primary_full]
            norms_full = part["norms"][mask]
            scores_full = (recon_full @ qrot) * norms_full + cell_qc[ci]
            rerank_scores.append(scores_full)
            rerank_ids.append(part["ids"][mask])
        if not rerank_ids:
            continue
        rerank_ids = np.concatenate(rerank_ids)
        rerank_scores = np.concatenate(rerank_scores)
        kk = min(k, len(rerank_scores))
        top_local = np.argpartition(-rerank_scores, kk - 1)[:kk]
        top_local = top_local[np.argsort(-rerank_scores[top_local])]
        out_idx[qi, :kk] = rerank_ids[top_local]
        pass2_time += time.time() - t0

    return out_idx, pass1_time, pass2_time


def main():
    print("Loading Deep-1M ...")
    r = load_deep1m(1_000_000, 1000)
    if r is None:
        print("FAILED")
        return
    v, q, _ = r
    dim = v.shape[1]
    print(f"  loaded n={v.shape[0]}, dim={dim}")

    print("Computing GT ...")
    gt_idx = FAISSFlatIndex(dim)
    gt_idx.add(v)
    _, gt = gt_idx.search(q, k=10)

    results = {}

    # ── Experiment A: bit allocation at 6 bits/coord ───────────────────
    print("\n========== A. Bit allocation at 6 bits/coord ==========")
    for bits, refine, label in [(5, 1, "5+1"), (4, 2, "4+2"), (6, 0, "6+0")]:
        print(f"\n  Building IVF-RVQ-TQ b={bits} b'={refine} (label={label}) ...")
        t0 = time.time()
        idx = IVFRVQTQIndex(dim=dim, nlist=1000, bits=bits,
                            refine_bits=refine, nprobe=40, seed=42)
        idx.train(v)
        idx.add(v)
        build_t = time.time() - t0
        print(f"    built in {build_t:.1f}s")
        for nprobe in [20, 40]:
            idx.nprobe = nprobe
            t0 = time.time()
            _, pred = idx.search(q, k=10)
            search_t = time.time() - t0
            recall = compute_recall(gt[:, :10], pred[:, :10], 10)
            key = f"alloc_{label}_np{nprobe}"
            results[key] = {
                "bits": bits, "refine": refine, "total_bits": bits + refine,
                "nprobe": nprobe, "recall_at_10": float(recall),
                "search_s": round(search_t, 1),
            }
            print(f"    np={nprobe}: R@10 = {recall:.4f} ({search_t:.1f}s)")

    # ── Experiment B: cascade search at (b=5, b'=1) ───────────────────
    print("\n========== B. Cascade search ==========")
    print("  Building IVF-RVQ-TQ b=5 b'=1 (canonical 6-bit config) ...")
    idx_cascade = IVFRVQTQIndex(dim=dim, nlist=1000, bits=5,
                                 refine_bits=1, nprobe=40, seed=42)
    idx_cascade.train(v)
    idx_cascade.add(v)

    # Baseline (no cascade)
    idx_cascade.nprobe = 40
    t0 = time.time()
    _, pred = idx_cascade.search(q, k=10)
    base_t = time.time() - t0
    base_recall = compute_recall(gt[:, :10], pred[:, :10], 10)
    print(f"  baseline (no cascade) np=40: R@10={base_recall:.4f} ({base_t:.1f}s)")
    results["cascade_baseline"] = {
        "recall_at_10": float(base_recall), "search_s": round(base_t, 1)
    }

    # Cascade with various MSB / rerank-N settings
    for top_msb in [3, 4]:
        for rerank_n in [50, 100, 200, 500]:
            t0 = time.time()
            res = search_cascade(idx_cascade, q, k=10,
                                  top_msb_bits=top_msb, rerank_n=rerank_n)
            elapsed = time.time() - t0
            if res is None:
                print(f"  msb={top_msb} N={rerank_n}: SKIP (top_msb_bits >= primary)")
                continue
            pred, p1t, p2t = res
            recall = compute_recall(gt[:, :10], pred[:, :10], 10)
            key = f"cascade_msb{top_msb}_N{rerank_n}"
            results[key] = {
                "top_msb_bits": top_msb, "rerank_n": rerank_n,
                "recall_at_10": float(recall),
                "total_s": round(elapsed, 2),
                "pass1_s": round(p1t, 2),
                "pass2_s": round(p2t, 2),
            }
            print(f"  msb={top_msb} rerank_N={rerank_n}: R@10={recall:.4f} "
                  f"total={elapsed:.1f}s (p1={p1t:.1f}s p2={p2t:.1f}s)  "
                  f"Δ={(recall - base_recall)*100:+.2f}pp")

    out = os.path.join(os.path.dirname(__file__), "cascade_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
