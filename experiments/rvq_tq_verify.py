"""
Verification: is the RVQ-TQ advantage at b>=6 a real result or a grid-resolution
artifact in the pure Lloyd-Max baseline?

We re-implement pure Lloyd-Max with high grid resolution (100K points), and
compare to RVQ-TQ at matched memory. Run on both Deep-1M and SIFT-1M.

Outputs experiments/rvq_tq_verify_results.json.
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy.stats import norm

from turboquant_search.faiss_baselines import FAISS_AVAILABLE, FAISSFlatIndex
from turboquant_search.benchmarks import compute_recall
from turboquant_search.datasets import load_deep1m, load_sift1m
from experiments.rvq_tq_explore import RVQTQIndex, per_bin_lloyd_max_subcentroids


def high_resolution_lloyd_max(bits: int, n_iter: int = 500, grid_size: int = 100000):
    """Pure Lloyd-Max with high grid resolution to rule out grid artifacts."""
    n_levels = 2 ** bits

    # Initial boundaries at uniform quantiles
    quantiles = np.linspace(0, 1, n_levels + 1)[1:-1]
    boundaries = norm.ppf(quantiles)

    # Use a wider grid and finer resolution
    x_grid = np.linspace(-5.0, 5.0, grid_size)
    pdf_vals = norm.pdf(x_grid)

    for _ in range(n_iter):
        centroids = np.zeros(n_levels)
        all_bounds = np.concatenate([[-np.inf], boundaries, [np.inf]])
        for i in range(n_levels):
            mask = (x_grid >= all_bounds[i]) & (x_grid < all_bounds[i + 1])
            if mask.sum() > 0:
                w = pdf_vals[mask]
                centroids[i] = (x_grid[mask] * w).sum() / max(w.sum(), 1e-15)
            else:
                centroids[i] = (all_bounds[i] + all_bounds[i + 1]) / 2.0
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0

    return centroids, boundaries


class HighResPureTQIndex:
    """Pure Lloyd-Max TurboQuant with high-resolution codebook design."""

    def __init__(self, dim: int, bits: int, seed: int = 42, grid_size: int = 100000):
        self.dim = dim
        self.bits = bits
        rng = np.random.default_rng(seed)
        H = rng.normal(size=(dim, dim))
        Q, _ = np.linalg.qr(H)
        self.rotation = Q.astype(np.float32)

        c_raw, b_raw = high_resolution_lloyd_max(bits, grid_size=grid_size)
        scale = np.sqrt(dim)
        self.centroids = (c_raw / scale).astype(np.float32)
        self.boundaries = (b_raw / scale).astype(np.float32)

        self._indices = None
        self._norms = None
        self._n = 0

    def add(self, vectors):
        v = np.ascontiguousarray(vectors.astype(np.float32))
        rotated = v @ self.rotation.T
        norms = np.linalg.norm(rotated, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = rotated / norms
        primary = np.digitize(normalized, self.boundaries).astype(np.uint16)
        primary = np.clip(primary, 0, 2 ** self.bits - 1)

        if self._indices is None:
            self._indices = primary
            self._norms = norms.reshape(-1)
        else:
            self._indices = np.concatenate([self._indices, primary])
            self._norms = np.concatenate([self._norms, norms.reshape(-1)])
        self._n += v.shape[0]

    def search(self, queries, k=10, query_batch=64):
        q = np.ascontiguousarray(queries.astype(np.float32))
        nq = q.shape[0]
        rotated_q = q @ self.rotation.T
        recon = self.centroids[self._indices] * self._norms[:, np.newaxis]
        recon_T = np.ascontiguousarray(recon.T)

        out_idx = np.empty((nq, k), dtype=np.int64)
        for s in range(0, nq, query_batch):
            e = min(s + query_batch, nq)
            scores = rotated_q[s:e] @ recon_T
            B = e - s
            part_idx = np.argpartition(-scores, k, axis=1)[:, :k]
            row = np.arange(B)[:, None]
            out_idx[s:e] = part_idx[row, np.argsort(-scores[row, part_idx], axis=1)]
        return None, out_idx


def run_dataset(name, loader):
    print(f"\n{'='*60}\n  {name.upper()}\n{'='*60}")
    r = loader()
    if r is None:
        print(f"  failed to load {name}")
        return {}
    v, q, _ = r
    dim = v.shape[1]
    n = v.shape[0]
    print(f"  loaded n={n}, dim={dim}")

    print("  computing GT ...")
    gt_idx = FAISSFlatIndex(dim) if FAISS_AVAILABLE else None
    gt_idx.add(v)
    _, gt = gt_idx.search(q, k=10)

    out = {}
    # High-resolution pure Lloyd-Max at b in {4,5,6,7}
    for b in [4, 5, 6, 7]:
        label = f"hires-pure-{b}bit"
        print(f"  [{label}] building ...")
        t0 = time.time()
        idx = HighResPureTQIndex(dim, bits=b, seed=42, grid_size=100000)
        idx.add(v)
        build_t = time.time() - t0
        t0 = time.time()
        _, pred = idx.search(q, k=10)
        search_t = time.time() - t0
        recall = compute_recall(gt[:, :10], pred[:, :10], 10)
        mem_mb = n * (b * dim / 8 + 4) / (1024 * 1024)
        out[label] = {
            "bits": b, "refine": 0, "total_bits": b,
            "recall_at_10": float(recall),
            "memory_mb": float(round(mem_mb, 1)),
            "build_s": round(build_t, 1),
        }
        print(f"    R@10={recall:.4f}  mem={mem_mb:.1f}MB  build={build_t:.1f}s")

    # RVQ-TQ at the same total memory
    for bits, refine in [(3, 1), (4, 1), (3, 2), (4, 2), (3, 3), (4, 3)]:
        label = f"rvq-{bits}+{refine}bit"
        print(f"  [{label}] building ...")
        t0 = time.time()
        idx = RVQTQIndex(dim, bits=bits, refine_bits=refine, seed=42)
        idx.add(v)
        build_t = time.time() - t0
        t0 = time.time()
        _, pred = idx.search(q, k=10)
        search_t = time.time() - t0
        recall = compute_recall(gt[:, :10], pred[:, :10], 10)
        mem_mb = n * idx.memory_bytes_per_vec / (1024 * 1024)
        out[label] = {
            "bits": bits, "refine": refine, "total_bits": bits + refine,
            "recall_at_10": float(recall),
            "memory_mb": float(round(mem_mb, 1)),
            "build_s": round(build_t, 1),
        }
        print(f"    R@10={recall:.4f}  mem={mem_mb:.1f}MB  build={build_t:.1f}s")

    return out


def main():
    results = {}
    for name, loader in [
        ("deep-1m", lambda: load_deep1m(1_000_000, 1000)),
        ("sift-1m", lambda: load_sift1m(1_000_000, 1000)),
    ]:
        results[name] = run_dataset(name, loader)
        out = os.path.join(os.path.dirname(__file__), "rvq_tq_verify_results.json")
        with open(out, "w") as f:
            json.dump(results, f, indent=2)

    print("\n=== Final summary ===")
    for ds, sub in results.items():
        print(f"\n{ds}:")
        # Group by total bits
        by_total = {}
        for label, r in sub.items():
            t = r["total_bits"]
            by_total.setdefault(t, []).append((label, r))
        for t in sorted(by_total):
            print(f"  total {t} bits:")
            entries = by_total[t]
            entries.sort(key=lambda x: -x[1]["recall_at_10"])
            for label, r in entries:
                print(f"    {label:<22} R@10={r['recall_at_10']*100:.2f}%  mem={r['memory_mb']:.1f}MB")


if __name__ == "__main__":
    main()
