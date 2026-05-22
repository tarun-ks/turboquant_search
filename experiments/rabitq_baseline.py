"""
RaBitQ baseline implementation (NumPy) for fair comparison.

RaBitQ (Gao & Long, SIGMOD 2024) uses:
  1. Random orthogonal rotation (same as TurboQuant)
  2. Binary quantization: store sign(rotated_x) as 1 bit per dim
  3. Correction factor x0 = ||x|| * cos(angle between x and binary_x)
  4. At search: estimate IP using binary inner product + correction

This is 1 bit/dim (vs TQ's 5 bits/dim), so much higher compression but
lower recall. The comparison shows the tradeoff space.

Reference: https://github.com/gaoj0017/RaBitQ
"""

import numpy as np
import time
from typing import Tuple


class RaBitQIndex:
    """IVF-RaBitQ: IVF partitioning with RaBitQ binary compression."""

    def __init__(self, dim: int, nlist: int = 100, nprobe: int = 10, seed: int = 42):
        self.dim = dim
        self.nlist = nlist
        self.nprobe = nprobe
        self.seed = seed

        # Padded dimension (multiple of 64 for efficiency)
        self.B = (dim + 63) // 64 * 64

        # Random orthogonal rotation
        rng = np.random.RandomState(seed)
        G = rng.randn(self.B, self.B).astype(np.float32)
        self.P, _ = np.linalg.qr(G)

        # IVF
        self.coarse_centroids = None
        self._trained = False

        # Per-partition storage
        self._invlists = [[] for _ in range(nlist)]
        self._partitions = [{"binary": None, "x0": None, "norms": None} for _ in range(nlist)]

        self._n_vectors = 0
        self.build_time = 0.0
        self.train_time = 0.0

    def train(self, vectors: np.ndarray):
        """Train IVF centroids via k-means."""
        vectors = vectors.astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors_normed = vectors / np.maximum(norms, 1e-8)

        actual_nlist = min(self.nlist, vectors.shape[0])
        t0 = time.time()
        from sklearn.cluster import MiniBatchKMeans
        kmeans = MiniBatchKMeans(n_clusters=actual_nlist, random_state=self.seed,
                                 batch_size=min(10000, vectors.shape[0]), n_init=3, max_iter=50)
        kmeans.fit(vectors_normed)
        self.coarse_centroids = kmeans.cluster_centers_.astype(np.float32)
        c_norms = np.linalg.norm(self.coarse_centroids, axis=1, keepdims=True)
        self.coarse_centroids = self.coarse_centroids / np.maximum(c_norms, 1e-8)
        self.nlist = actual_nlist
        self._invlists = [[] for _ in range(actual_nlist)]
        self._partitions = [{"binary": None, "x0": None, "norms": None} for _ in range(actual_nlist)]
        self._trained = True
        self.train_time = time.time() - t0

    def _rabitq_compress(self, residuals: np.ndarray):
        """Compress residuals using RaBitQ: rotate, binarize, compute x0."""
        n = residuals.shape[0]
        # Pad to B dimensions
        if self.dim < self.B:
            residuals_pad = np.pad(residuals, ((0, 0), (0, self.B - self.dim)), 'constant')
        else:
            residuals_pad = residuals

        norms = np.linalg.norm(residuals_pad, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = residuals_pad / norms

        # Rotate
        rotated = normalized @ self.P.T  # (n, B)

        # Binary quantization: sign bits
        binary = (rotated > 0).astype(np.uint8)  # (n, B)

        # Compute x0: correction factor
        # x0 = sum(|rotated_i|) / (sqrt(B) * ||rotated||)
        # Simplified: for unit vectors, x0 ≈ sqrt(2/pi) for Gaussian coordinates
        binary_float = binary.astype(np.float32) * 2 - 1  # {-1, +1}
        x0 = np.sum(rotated * binary_float, axis=1) / (np.sqrt(self.B) * np.linalg.norm(rotated, axis=1))

        return binary, x0, norms.reshape(-1)

    def add(self, vectors: np.ndarray, ids=None):
        """Add vectors to the index."""
        assert self._trained
        vectors = vectors.astype(np.float32)
        n = vectors.shape[0]
        if ids is None:
            ids = np.arange(self._n_vectors, self._n_vectors + n)

        t0 = time.time()
        v_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors_normed = vectors / np.maximum(v_norms, 1e-8)
        assignments = np.argmax(vectors_normed @ self.coarse_centroids.T, axis=1)

        for list_idx in range(self.nlist):
            mask = assignments == list_idx
            if not mask.any():
                continue
            vecs = vectors_normed[mask]
            vids = ids[mask] if ids is not None else np.where(mask)[0]
            residuals = vecs - self.coarse_centroids[list_idx]
            binary, x0, norms = self._rabitq_compress(residuals)

            self._invlists[list_idx].extend(vids.tolist())
            part = self._partitions[list_idx]
            if part["binary"] is None:
                part["binary"] = binary
                part["x0"] = x0
                part["norms"] = norms
            else:
                part["binary"] = np.concatenate([part["binary"], binary])
                part["x0"] = np.concatenate([part["x0"], x0])
                part["norms"] = np.concatenate([part["norms"], norms])

        self._n_vectors += n
        self.build_time += time.time() - t0

    def search(self, queries: np.ndarray, k: int = 10):
        """Search using RaBitQ binary distance estimation."""
        queries = queries.astype(np.float32)
        nq = queries.shape[0]
        k = min(k, self._n_vectors)

        q_norms = np.linalg.norm(queries, axis=1, keepdims=True)
        queries_normed = queries / np.maximum(q_norms, 1e-8)

        # Coarse search
        coarse_scores = queries_normed @ self.coarse_centroids.T
        actual_nprobe = min(self.nprobe, self.nlist)
        top_lists = np.argpartition(-coarse_scores, actual_nprobe, axis=1)[:, :actual_nprobe]

        # Pad and rotate queries
        if self.dim < self.B:
            queries_pad = np.pad(queries_normed, ((0, 0), (0, self.B - self.dim)), 'constant')
        else:
            queries_pad = queries_normed
        q_rotated = queries_pad @ self.P.T  # (nq, B)

        all_scores = np.full((nq, k), -np.inf, dtype=np.float32)
        all_indices = np.full((nq, k), -1, dtype=np.int64)

        for q_idx in range(nq):
            candidates_scores = []
            candidates_ids = []

            for p in range(actual_nprobe):
                list_idx = int(top_lists[q_idx, p])
                part = self._partitions[list_idx]
                if part["binary"] is None:
                    continue

                binary = part["binary"]  # (n_in_list, B)
                x0 = part["x0"]          # (n_in_list,)
                norms = part["norms"]     # (n_in_list,)
                n_in_list = binary.shape[0]

                # RaBitQ distance estimation:
                # IP ≈ norm * x0 * (2 * hamming_agreement / B - 1) * sqrt(B) + coarse_score
                q_binary = (q_rotated[q_idx] > 0).astype(np.uint8)  # (B,)
                # Hamming agreement = number of matching bits
                agreement = np.sum(binary == q_binary, axis=1)  # (n_in_list,)
                # Estimated residual IP
                est_ip = norms * x0 * (2.0 * agreement / self.B - 1.0) * np.sqrt(self.B)
                total = est_ip + coarse_scores[q_idx, list_idx]

                list_ids = np.array(self._invlists[list_idx], dtype=np.int64)
                candidates_scores.append(total)
                candidates_ids.append(list_ids)

            if not candidates_scores:
                continue

            scores = np.concatenate(candidates_scores)
            ids = np.concatenate(candidates_ids)
            actual_k = min(k, len(scores))
            if actual_k >= len(scores):
                top_k = np.argsort(-scores)[:actual_k]
            else:
                top_k = np.argpartition(-scores, actual_k)[:actual_k]
                top_k = top_k[np.argsort(-scores[top_k])]
            all_scores[q_idx, :actual_k] = scores[top_k]
            all_indices[q_idx, :actual_k] = ids[top_k]

        return all_scores, all_indices

    @property
    def memory_bytes(self):
        # 1 bit per dim (padded) + x0 float32 + norm float32
        bits_per_vector = self.B + 64  # B sign bits + 32 x0 + 32 norm
        vector_bytes = (self._n_vectors * bits_per_vector) // 8
        centroid_bytes = self.nlist * self.dim * 4
        return vector_bytes + centroid_bytes

    @property
    def compression_ratio(self):
        uncompressed = self._n_vectors * self.dim * 4
        return uncompressed / max(self.memory_bytes, 1)


if __name__ == "__main__":
    """Quick test and benchmark against IVF-TQ on SIFT-1M."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from turboquant_search.core import IVFTurboQuantIndex
    from turboquant_search.faiss_baselines import FAISSFlatIndex
    from turboquant_search.benchmarks import compute_recall
    from turboquant_search.datasets import load_sift1m

    print("Loading SIFT-1M...", flush=True)
    result = load_sift1m(n_vectors=1000000, n_queries=10000)
    vectors, queries, _ = result
    dim = vectors.shape[1]

    flat = FAISSFlatIndex(dim)
    flat.add(vectors)
    _, gt = flat.search(queries, k=10)

    nlist, nprobe = 1000, 10

    # RaBitQ
    print("Building IVF-RaBitQ...", flush=True)
    rb = RaBitQIndex(dim, nlist=nlist, nprobe=nprobe, seed=42)
    rb.train(vectors)
    rb.add(vectors)
    _, idx_rb = rb.search(queries, k=10)
    r_rb = compute_recall(gt, idx_rb, 10)
    mem_rb = rb.memory_bytes / (1024 * 1024)

    # IVF-TQ
    print("Building IVF-TQ...", flush=True)
    tq = IVFTurboQuantIndex(dim, nlist=nlist, bits=4, nprobe=nprobe, seed=42)
    tq.train(vectors)
    tq.add(vectors)
    _, idx_tq = tq.search(queries, k=10)
    r_tq = compute_recall(gt, idx_tq, 10)
    mem_tq = tq.memory_bytes / (1024 * 1024)

    print(f"\nResults (SIFT-1M, 10K queries, nlist={nlist}, nprobe={nprobe}):")
    print(f"  IVF-RaBitQ (1 bit/dim): R@10={r_rb:.1%}  {mem_rb:.0f} MB  ({rb.compression_ratio:.1f}x)")
    print(f"  IVF-TQ 4-bit (5 bit/dim): R@10={r_tq:.1%}  {mem_tq:.0f} MB  ({tq.compression_ratio:.1f}x)")

    results = {
        "rabitq": {"recall10": round(r_rb * 100, 1), "memory_mb": round(mem_rb, 1),
                   "compression": round(rb.compression_ratio, 1), "bits_per_dim": 1},
        "ivf_tq": {"recall10": round(r_tq * 100, 1), "memory_mb": round(mem_tq, 1),
                   "compression": round(tq.compression_ratio, 1), "bits_per_dim": 5},
    }

    import json
    out = Path(__file__).parent / "rabitq_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")
