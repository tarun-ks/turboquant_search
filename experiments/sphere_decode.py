"""
Two probe experiments on IVF-RVQ-TQ:

(1) Sphere-projected reconstruction. After standard TQ decode, the per-coord
    centroids do NOT lie exactly on the unit sphere; the squared sum
    deviates from 1 by O(1/sqrt(d)). A Bayesian-MAP decoder under the unit-
    sphere prior would re-normalize each reconstructed vector to unit norm.
    The standard TQ score is <q_rot, recon> * stored_norm, but stored_norm
    captures the residual norm in the rotated frame, not the recon norm.
    We test: does renormalising recon to ||recon||=1 improve recall?

(2) Bit-importance ablation. Corrupt specific bit positions and measure
    recall drop. Reveals whether sign-bit, low-MSB primary bits, or high-MSB
    primary bits matter most.

Outputs experiments/sphere_decode_results.json.
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


def search_with_decoder(idx, queries, k=10, decoder='standard'):
    """
    decoder: 'standard' = sub_centroids[primary, sub] then score via <q_rot, recon> * norm
             'sphere'   = same but normalise recon to unit norm before scoring
    """
    q = np.ascontiguousarray(queries.astype(np.float32))
    nq = q.shape[0]
    q_norms = np.linalg.norm(q, axis=1, keepdims=True)
    q_normed = q / np.maximum(q_norms, 1e-8)
    q_rotated = q_normed @ idx.rotation.T
    coarse = q_normed @ idx.coarse_centroids.T
    nprobe = min(idx.nprobe, idx.nlist)
    top_cells = np.argpartition(-coarse, nprobe, axis=1)[:, :nprobe]

    out_idx = np.full((nq, k), -1, dtype=np.int64)
    for qi in range(nq):
        cells = top_cells[qi]
        cell_scores = coarse[qi, cells]
        qrot = q_rotated[qi]
        all_ids, all_scores = [], []
        for ci, cell in enumerate(cells):
            part = idx._partitions[cell]
            if part is None:
                continue
            if idx.refine_bits > 0:
                recon = idx.sub_centroids[part["primary"], part["sub"]]
            else:
                recon = idx.tq_centroids[part["primary"]]

            if decoder == 'sphere':
                # renormalise each reconstructed vector to unit norm
                recon_norms = np.linalg.norm(recon, axis=1, keepdims=True)
                recon = recon / np.maximum(recon_norms, 1e-8)

            residual_score = (recon @ qrot) * part["norms"]
            total_score = cell_scores[ci] + residual_score
            all_ids.append(part["ids"])
            all_scores.append(total_score)
        if not all_ids:
            continue
        all_ids = np.concatenate(all_ids)
        all_scores = np.concatenate(all_scores)
        kk = min(k, len(all_scores))
        top_local = np.argpartition(-all_scores, kk - 1)[:kk]
        top_local = top_local[np.argsort(-all_scores[top_local])]
        out_idx[qi, :kk] = all_ids[top_local]
    return out_idx


def search_with_corruption(idx, queries, k=10, corrupt='none', frac=0.0, seed=43):
    """
    Corrupt specified bits before reconstruction.
    corrupt='primary_msb' / 'primary_lsb' / 'sign' / 'random'
    frac: fraction of vectors (or coords) to corrupt
    """
    rng = np.random.default_rng(seed)
    # Make a copy of partitions with corruption applied
    # We'll corrupt at decode time on a per-search basis (faster)
    q = np.ascontiguousarray(queries.astype(np.float32))
    nq = q.shape[0]
    q_norms = np.linalg.norm(q, axis=1, keepdims=True)
    q_normed = q / np.maximum(q_norms, 1e-8)
    q_rotated = q_normed @ idx.rotation.T
    coarse = q_normed @ idx.coarse_centroids.T
    nprobe = min(idx.nprobe, idx.nlist)
    top_cells = np.argpartition(-coarse, nprobe, axis=1)[:, :nprobe]

    n_levels = 2 ** idx.bits
    n_sub = 2 ** idx.refine_bits if idx.refine_bits > 0 else 1

    out_idx = np.full((nq, k), -1, dtype=np.int64)
    for qi in range(nq):
        cells = top_cells[qi]
        cell_scores = coarse[qi, cells]
        qrot = q_rotated[qi]
        all_ids, all_scores = [], []
        for ci, cell in enumerate(cells):
            part = idx._partitions[cell]
            if part is None:
                continue
            primary = part["primary"].copy()
            sub = part["sub"].copy() if part["sub"] is not None else None

            # Apply corruption
            n_in_cell, dim = primary.shape
            n_corrupt = int(frac * n_in_cell * dim)
            if n_corrupt > 0:
                flat_idx = rng.choice(n_in_cell * dim, size=n_corrupt, replace=False)
                rows, cols = flat_idx // dim, flat_idx % dim
                if corrupt == 'primary_msb':
                    # XOR top bit of primary
                    top_bit = 1 << (idx.bits - 1)
                    primary[rows, cols] = primary[rows, cols] ^ top_bit
                elif corrupt == 'primary_lsb':
                    primary[rows, cols] = primary[rows, cols] ^ 1
                elif corrupt == 'sign' and sub is not None:
                    sub[rows, cols] = sub[rows, cols] ^ 1
                elif corrupt == 'random':
                    primary[rows, cols] = rng.integers(0, n_levels, size=n_corrupt).astype(primary.dtype)

                primary = np.clip(primary, 0, n_levels - 1)

            if idx.refine_bits > 0:
                recon = idx.sub_centroids[primary, sub]
            else:
                recon = idx.tq_centroids[primary]
            residual_score = (recon @ qrot) * part["norms"]
            total_score = cell_scores[ci] + residual_score
            all_ids.append(part["ids"])
            all_scores.append(total_score)
        if not all_ids:
            continue
        all_ids = np.concatenate(all_ids)
        all_scores = np.concatenate(all_scores)
        kk = min(k, len(all_scores))
        top_local = np.argpartition(-all_scores, kk - 1)[:kk]
        top_local = top_local[np.argsort(-all_scores[top_local])]
        out_idx[qi, :kk] = all_ids[top_local]
    return out_idx


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

    # Build IVF-RVQ-TQ at the canonical IVF-TQ config (b=5 LM + 1-bit sign = "6-bit")
    print("Building IVF-RVQ-TQ (b=5, b'=1, 6-bit) ...")
    t0 = time.time()
    idx = IVFRVQTQIndex(dim=dim, nlist=1000, bits=5, refine_bits=1, nprobe=40, seed=42)
    idx.train(v)
    idx.add(v)
    print(f"  built in {time.time()-t0:.1f}s")

    results = {}

    # Experiment 1: sphere-projected decoder
    print("\n--- Experiment 1: sphere-projected decoder ---")
    for nprobe in [20, 40]:
        idx.nprobe = nprobe
        for decoder in ['standard', 'sphere']:
            t0 = time.time()
            pred = search_with_decoder(idx, q, k=10, decoder=decoder)
            recall = compute_recall(gt[:, :10], pred[:, :10], 10)
            elapsed = time.time() - t0
            key = f"np{nprobe}_{decoder}"
            results[key] = {"recall_at_10": float(recall), "elapsed_s": round(elapsed, 1)}
            print(f"  np={nprobe} decoder={decoder}: R@10 = {recall:.4f}  ({elapsed:.1f}s)")

    # Experiment 2: bit-importance ablation
    print("\n--- Experiment 2: bit-importance ablation (np=40) ---")
    idx.nprobe = 40
    # Baseline (no corruption)
    pred = search_with_corruption(idx, q, k=10, corrupt='none', frac=0.0)
    baseline_recall = compute_recall(gt[:, :10], pred[:, :10], 10)
    print(f"  baseline (no corruption): R@10 = {baseline_recall:.4f}")
    results["ablation_baseline"] = {"recall_at_10": float(baseline_recall)}

    for corrupt_type in ['primary_msb', 'primary_lsb', 'sign', 'random']:
        for frac in [0.05, 0.10, 0.20]:
            t0 = time.time()
            pred = search_with_corruption(idx, q, k=10, corrupt=corrupt_type,
                                           frac=frac, seed=43)
            recall = compute_recall(gt[:, :10], pred[:, :10], 10)
            elapsed = time.time() - t0
            key = f"corrupt_{corrupt_type}_frac{frac}"
            results[key] = {"recall_at_10": float(recall), "elapsed_s": round(elapsed, 1)}
            print(f"  {corrupt_type:>13} frac={frac:.2f}: R@10 = {recall:.4f}  Δ={(recall-baseline_recall)*100:+.2f}pp")

    out = os.path.join(os.path.dirname(__file__), "sphere_decode_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
