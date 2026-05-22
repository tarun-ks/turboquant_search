"""
RVQ-TQ exploration: multi-bit Stage-2 refinement on top of TurboQuant.

Hypothesis: at matched total bit budget, (b primary + b' refinement) bits should
roughly match (b+b') pure Lloyd-Max bits on per-coord MSE, but the hierarchical
structure may give a different ranking-error profile.

This is a self-contained experiment — does NOT modify turboquant_search/core.py.
Outputs experiments/rvq_tq_results.json.
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy.stats import norm

from turboquant_search.core import (
    _lloyd_max_codebook,
    TurboQuantSearchIndex,
    FlatSearchIndex,
)
from turboquant_search.faiss_baselines import FAISS_AVAILABLE, FAISSFlatIndex
from turboquant_search.benchmarks import compute_recall
from turboquant_search.datasets import load_deep1m


def per_bin_lloyd_max_subcentroids(bits: int, refine_bits: int, dim: int,
                                    primary_codebook=None):
    """
    For each primary Lloyd-Max bin, design a 2^refine_bits Lloyd-Max codebook
    for the within-bin truncated N(0, 1/d) distribution.

    Returns sub_centroids of shape (2**bits, 2**refine_bits) and
    sub_boundaries of shape (2**bits, 2**refine_bits - 1) — both in
    "normalized" units (i.e., scaled so the source is N(0, 1/d) on the unit sphere).

    primary_codebook : optional (centroids_raw, boundaries_raw) pair, both in
        STANDARD-NORMAL units. If provided, the sub-centroids are designed
        against THESE primary bin boundaries. If None, the cached Lloyd-Max
        codebook is used. CRITICAL: this must match whatever codebook the
        encoding step uses, otherwise sub-bin design and primary bin assignment
        disagree and reconstruction quality collapses (the b>=6 bug).
    """
    if primary_codebook is not None:
        centroids_raw, boundaries_raw = primary_codebook
    else:
        centroids_raw, boundaries_raw = _lloyd_max_codebook(bits)
    scale = np.sqrt(dim)
    centroids = centroids_raw / scale
    boundaries = boundaries_raw / scale

    n_levels = 2 ** bits
    n_sub = 2 ** refine_bits
    bin_lo = np.concatenate([[-6.0 / scale], boundaries])
    bin_hi = np.concatenate([boundaries, [6.0 / scale]])

    sub_centroids = np.zeros((n_levels, n_sub), dtype=np.float32)
    sub_boundaries = np.zeros((n_levels, n_sub - 1), dtype=np.float32)

    for i in range(n_levels):
        lo, hi = bin_lo[i], bin_hi[i]

        if n_sub == 1:
            sub_centroids[i, 0] = centroids[i]
            continue

        # Sample within-bin Gaussian density
        n_grid = 4000
        grid = np.linspace(lo, hi, n_grid)
        pdf = norm.pdf(grid * scale) * scale
        # Normalise to within-bin density
        if pdf.sum() > 0:
            pdf = pdf / pdf.sum()
        else:
            pdf = np.ones(n_grid) / n_grid

        # Lloyd iteration starting from quantile init
        cdf = np.cumsum(pdf)
        cdf = cdf / cdf[-1]
        # Initial centroids at the (k+0.5)/n_sub quantiles
        init_quantiles = (np.arange(n_sub) + 0.5) / n_sub
        c = np.array([grid[np.searchsorted(cdf, q)] for q in init_quantiles])

        for _ in range(80):
            # Decision boundaries are midpoints
            mids = (c[:-1] + c[1:]) / 2
            # Reassign each grid point
            assignments = np.searchsorted(mids, grid)
            # Update centroids to conditional means
            for k in range(n_sub):
                mask = assignments == k
                if pdf[mask].sum() > 1e-12:
                    c[k] = np.average(grid[mask], weights=pdf[mask])
                # else keep old centroid
            c = np.sort(c)

        sub_centroids[i] = c.astype(np.float32)
        sub_boundaries[i] = ((c[:-1] + c[1:]) / 2).astype(np.float32)

    return sub_centroids, sub_boundaries


class RVQTQIndex:
    """
    TurboQuant + multi-bit Stage-2 refinement (RVQ-TQ).

    bits: primary Lloyd-Max bits per coord.
    refine_bits: Stage-2 refinement bits per coord (0 = pure Lloyd-Max).
    """

    def __init__(self, dim: int, bits: int, refine_bits: int, seed: int = 42):
        self.dim = dim
        self.bits = bits
        self.refine_bits = refine_bits
        self.seed = seed

        # Random orthogonal rotation
        rng = np.random.default_rng(seed)
        H = rng.normal(size=(dim, dim))
        Q, _ = np.linalg.qr(H)
        self.rotation = Q.astype(np.float32)

        # Primary Lloyd-Max
        centroids_raw, boundaries_raw = _lloyd_max_codebook(bits)
        scale = np.sqrt(dim)
        self.centroids = (centroids_raw / scale).astype(np.float32)
        self.boundaries = (boundaries_raw / scale).astype(np.float32)

        # Multi-bit refinement codebook
        self.sub_centroids, self.sub_boundaries = per_bin_lloyd_max_subcentroids(
            bits, refine_bits, dim
        )

        self._indices = None        # (n, dim) primary bin index
        self._sub_indices = None    # (n, dim) sub-bin index
        self._norms = None
        self._n = 0

    def add(self, vectors):
        v = np.ascontiguousarray(vectors.astype(np.float32))
        rotated = v @ self.rotation.T
        norms = np.linalg.norm(rotated, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = rotated / norms

        # Primary quantization
        primary = np.digitize(normalized, self.boundaries).astype(np.uint16)
        primary = np.clip(primary, 0, 2 ** self.bits - 1)

        # Sub-bin quantization within each primary bin
        if self.refine_bits > 0:
            sub = np.zeros_like(primary, dtype=np.uint16)
            n_sub = 2 ** self.refine_bits
            for i in range(2 ** self.bits):
                mask = primary == i
                if not mask.any():
                    continue
                vals = normalized[mask]
                # Bucket by sub_boundaries[i]
                if n_sub > 1:
                    sub_idx = np.searchsorted(self.sub_boundaries[i], vals)
                    sub_idx = np.clip(sub_idx, 0, n_sub - 1)
                else:
                    sub_idx = np.zeros_like(vals, dtype=np.uint16)
                sub[mask] = sub_idx.astype(np.uint16)
        else:
            sub = np.zeros_like(primary, dtype=np.uint16)

        if self._indices is None:
            self._indices = primary
            self._sub_indices = sub
            self._norms = norms.reshape(-1)
        else:
            self._indices = np.concatenate([self._indices, primary])
            self._sub_indices = np.concatenate([self._sub_indices, sub])
            self._norms = np.concatenate([self._norms, norms.reshape(-1)])
        self._n += v.shape[0]

    def search(self, queries, k=10, query_batch=64):
        q = np.ascontiguousarray(queries.astype(np.float32))
        nq = q.shape[0]
        rotated_q = q @ self.rotation.T  # (nq, dim)

        # Reconstruct normalized database vectors using sub-centroids
        if self.refine_bits > 0:
            recon_norm = self.sub_centroids[self._indices, self._sub_indices]  # (n, dim)
        else:
            recon_norm = self.centroids[self._indices]

        # x_recon * norm scaled DB; precompute once
        recon_full = recon_norm * self._norms[:, np.newaxis]  # (n, dim) float32
        recon_full_T = np.ascontiguousarray(recon_full.T)     # (dim, n)

        out_scores = np.empty((nq, k), dtype=np.float32)
        out_idx = np.empty((nq, k), dtype=np.int64)
        for s in range(0, nq, query_batch):
            e = min(s + query_batch, nq)
            sub_scores = rotated_q[s:e] @ recon_full_T  # (B, n)
            B = e - s
            if k >= self._n:
                top_idx = np.argsort(-sub_scores, axis=1)[:, :k]
            else:
                part_idx = np.argpartition(-sub_scores, k, axis=1)[:, :k]
                row = np.arange(B)[:, None]
                top_idx = part_idx[row, np.argsort(-sub_scores[row, part_idx], axis=1)]
            row = np.arange(B)[:, None]
            out_idx[s:e] = top_idx
            out_scores[s:e] = sub_scores[row, top_idx]
        return out_scores, out_idx

    @property
    def memory_bytes_per_vec(self):
        # primary + refinement bits + norm
        return (self.bits + self.refine_bits) * self.dim / 8 + 4

    @property
    def total_bits_per_coord(self):
        return self.bits + self.refine_bits


def run():
    print("Loading Deep-1M ...")
    r = load_deep1m(1_000_000, 1000)
    if r is None:
        print("Deep-1M load failed.")
        return
    v, q, _ = r
    dim = v.shape[1]
    n = v.shape[0]
    print(f"  loaded n={n}, dim={dim}")

    # Ground truth
    print("Computing ground truth ...")
    t0 = time.time()
    if FAISS_AVAILABLE:
        gt_idx = FAISSFlatIndex(dim)
    else:
        gt_idx = FlatSearchIndex(dim)
    gt_idx.add(v)
    _, gt = gt_idx.search(q, k=10)
    print(f"  GT: {time.time() - t0:.1f}s")

    configs = []
    # Pure Lloyd-Max baselines (no refinement) at varying bits
    for b in [3, 4, 5, 6, 7]:
        configs.append((b, 0, f"pure-{b}bit"))
    # RVQ-TQ configurations
    for b, bp in [(3, 1), (3, 2), (3, 3), (4, 1), (4, 2), (4, 3)]:
        configs.append((b, bp, f"rvq-{b}+{bp}bit"))

    results = {}
    for bits, refine_bits, label in configs:
        total = bits + refine_bits
        print(f"\n[{label}] primary={bits}, refine={refine_bits}, total={total} bits/coord")
        try:
            t0 = time.time()
            idx = RVQTQIndex(dim, bits=bits, refine_bits=refine_bits, seed=42)
            idx.add(v)
            build_t = time.time() - t0

            t0 = time.time()
            _, pred = idx.search(q, k=10)
            search_t = time.time() - t0

            r = compute_recall(gt[:, :10], pred[:, :10], 10)
            mem_mb = (n * idx.memory_bytes_per_vec) / (1024 * 1024)
            qps = q.shape[0] / max(search_t, 1e-6)

            results[label] = {
                "primary_bits": bits,
                "refine_bits": refine_bits,
                "total_bits": total,
                "recall_at_10": float(r),
                "memory_mb": float(round(mem_mb, 1)),
                "build_s": float(round(build_t, 1)),
                "search_s": float(round(search_t, 1)),
                "qps": int(qps),
            }
            print(f"  R@10 = {r:.4f}  mem = {mem_mb:.1f} MB  build = {build_t:.1f}s  search = {search_t:.1f}s")
        except Exception as e:
            print(f"  FAILED: {e}")
            results[label] = {"error": str(e)}

        # Incremental write
        out = os.path.join(os.path.dirname(__file__), "rvq_tq_results.json")
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  wrote {out}")

    print("\n=== Summary ===")
    print(f"{'config':<20} {'bits':<6} {'R@10':<8} {'memory':<10}")
    for label, r in results.items():
        if "error" in r:
            print(f"{label:<20} ERROR: {r['error']}")
            continue
        print(f"{label:<20} {r['total_bits']:<6} {r['recall_at_10']*100:<7.2f}% {r['memory_mb']:<6.1f} MB")


if __name__ == "__main__":
    run()
