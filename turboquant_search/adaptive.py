"""
Adaptive IVF-TQ: IVF-TQ with periodic, low-cost coarse-partition refresh.

Motivation
----------
The TQ residual compression is data-independent (its quality is bounded by
(bits, dim) only).
The IVF coarse partition, however, is a learned k-means and degrades under
distribution shift (e.g. embedding-encoder swap). Standard IVF-TQ freezes
the coarse partition at training time, leaving this layer vulnerable.

This module adds a `refresh()` operation that re-runs k-means on a
representative sample of the *currently indexed* vectors and re-assigns
every stored vector to its new partition. Crucially, the residual
compression layer is not re-trained — only the partition is. This is
~30x cheaper than retraining a PQ codebook (which couples partition and
codebook re-training) and yields a streaming-robust index.

Algorithm
---------
On `refresh(sample_size=50_000)`:
    1. Reconstruct (approximately) all currently-indexed vectors using
       stored TQ codes and old centroids.
    2. Sample `sample_size` reconstructed vectors uniformly across
       partitions to avoid sampling bias toward over-populated cells.
    3. Run k-means on the sample to obtain new coarse centroids.
    4. Re-assign every reconstructed vector to its new nearest centroid.
    5. Compute new residuals and re-quantize using the *fixed* TQ
       parameters (Π, Lloyd-Max codebook, sub-centroids).
    6. Discard old per-partition state; install new partitions.

Data-independence of the compression layer means step 5 requires no
training — only an O(N · d) re-encode pass. PQ-family methods cannot
do an analogous selective refresh because their compression is
data-dependent.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np

from .core import IVFTurboQuantIndex


class AdaptiveIVFTurboQuantIndex(IVFTurboQuantIndex):
    """IVF-TQ with cheap periodic coarse-partition refresh."""

    def __init__(self, *args, refresh_every: Optional[int] = None,
                 refresh_sample: int = 50_000, **kwargs):
        """
        Parameters
        ----------
        refresh_every : int or None
            If set, automatically call `refresh()` every time this many
            vectors have been added since the last refresh. None disables
            automatic refresh (caller invokes `refresh()` manually).
        refresh_sample : int
            Number of reconstructed vectors used for k-means in the
            refresh. 50K is enough for stable centroids on most workloads.
        """
        super().__init__(*args, **kwargs)
        self.refresh_every = refresh_every
        self.refresh_sample = refresh_sample
        self._adds_since_refresh = 0
        self.refresh_count = 0
        self.refresh_total_time = 0.0
        self.refresh_log: List[Dict] = []

    # ------------------------------------------------------------------
    # Reconstruction
    # ------------------------------------------------------------------
    def _reconstruct_partition(self, list_idx: int) -> np.ndarray:
        """Reconstruct all vectors stored in a single partition.

        Uses the per-coordinate sub-centroid (sign-bit refined) when
        available, else the bin centroid. Adds back the partition's
        coarse centroid and de-rotates.
        """
        part = self._partitions[list_idx]
        if part["indices"] is None or len(part["indices"]) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)

        indices = part["indices"]
        norms = part["norms"]

        # Reconstruct rotated unit residual
        if self.use_residual_sign and part["sign_bits"] is not None:
            unit = self.sub_centroids[indices, part["sign_bits"]]
        else:
            unit = self.tq_centroids[indices]
        unit = unit.astype(np.float32)

        # Scale by stored norms to recover rotated residual
        rotated_residual = unit * norms[:, None]

        # De-rotate (Π is orthogonal, so Π^T = Π^{-1})
        residual = rotated_residual @ self.rotation_matrix
        # Add coarse centroid to recover (approximately) the original vector
        return self.coarse_centroids[list_idx] + residual

    def _reconstruct_all(self) -> tuple[np.ndarray, np.ndarray]:
        """Reconstruct all indexed vectors and return (vectors, vector_ids)."""
        all_vecs: List[np.ndarray] = []
        all_ids: List[np.ndarray] = []
        for l in range(self.nlist):
            v = self._reconstruct_partition(l)
            if v.shape[0] == 0:
                continue
            all_vecs.append(v)
            all_ids.append(np.asarray(self._invlists[l], dtype=np.int64))
        if not all_vecs:
            return (np.zeros((0, self.dim), dtype=np.float32),
                    np.zeros((0,), dtype=np.int64))
        return np.concatenate(all_vecs), np.concatenate(all_ids)

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------
    def refresh(self, use_raw_if_available: bool = True,
                rng_seed: Optional[int] = None) -> Dict:
        """Re-run k-means on a sample of the currently indexed vectors,
        then re-encode all vectors against the new partition.

        Parameters
        ----------
        use_raw_if_available : bool
            If True and raw vectors are stored (rerank-mode), use them as
            ground truth for re-clustering. Otherwise use approximate
            reconstructions from compressed codes (self-contained path).
        rng_seed : int or None
            Optional override for the k-means random seed. None reuses
            self.seed + refresh_count to vary across refreshes.

        Returns
        -------
        dict
            Stats: {duration_s, sample_size, n_re_encoded, used_raw}.
        """
        assert self._trained, "Refresh requires a trained index."
        if self._n_vectors == 0:
            return {"duration_s": 0.0, "sample_size": 0,
                    "n_re_encoded": 0, "used_raw": False}

        t0 = time.time()
        used_raw = use_raw_if_available and self._raw_vectors is not None

        if used_raw:
            all_vecs = self._raw_vectors.copy()
            all_ids = np.arange(self._n_vectors, dtype=np.int64)
        else:
            all_vecs, all_ids = self._reconstruct_all()

        # Sample uniformly across partitions to avoid bias toward dense cells.
        sample = min(self.refresh_sample, all_vecs.shape[0])
        seed = rng_seed if rng_seed is not None else self.seed + self.refresh_count
        rng = np.random.RandomState(seed)
        # Stratified sample: take ~equal counts per non-empty partition.
        non_empty = [l for l in range(self.nlist) if self._invlists[l]]
        if non_empty:
            per_part = max(1, sample // len(non_empty))
            offsets = np.cumsum([0] + [len(self._invlists[l]) for l in range(self.nlist)])
            chosen = []
            cursor = 0
            for l in range(self.nlist):
                k = len(self._invlists[l])
                if k == 0:
                    cursor += k; continue
                take = min(per_part, k)
                idx = rng.choice(k, size=take, replace=False) + cursor
                chosen.append(idx)
                cursor += k
            chosen = np.concatenate(chosen)
            sample_vecs = all_vecs[chosen]
        else:
            sample_vecs = all_vecs[rng.choice(all_vecs.shape[0],
                                              size=sample, replace=False)]

        # Re-run k-means
        from sklearn.cluster import MiniBatchKMeans
        sample_normed = sample_vecs / np.maximum(
            np.linalg.norm(sample_vecs, axis=1, keepdims=True), 1e-8)
        kmeans = MiniBatchKMeans(
            n_clusters=self.nlist,
            random_state=seed,
            batch_size=min(10_000, sample_normed.shape[0]),
            n_init=3,
            max_iter=50,
        )
        kmeans.fit(sample_normed)
        new_centroids = kmeans.cluster_centers_.astype(np.float32)
        new_centroids /= np.maximum(np.linalg.norm(new_centroids, axis=1, keepdims=True), 1e-8)

        # Re-assign every indexed vector to the new partition
        all_normed = all_vecs / np.maximum(
            np.linalg.norm(all_vecs, axis=1, keepdims=True), 1e-8)
        new_assignments = np.argmax(all_normed @ new_centroids.T, axis=1)

        # Reset per-partition state and re-encode
        self._invlists = [[] for _ in range(self.nlist)]
        self._partitions = [
            {"indices": None, "norms": None, "sign_bits": None, "codes": None}
            for _ in range(self.nlist)
        ]

        for list_idx in range(self.nlist):
            mask = new_assignments == list_idx
            if not mask.any():
                continue
            vecs = all_normed[mask]
            vids = all_ids[mask]
            residuals = vecs - new_centroids[list_idx]
            indices, norms, sign_bits = self._tq_compress(residuals)

            if sign_bits is not None:
                codes = (indices * 2 + sign_bits).astype(np.uint8)
            else:
                codes = indices.astype(np.uint8)

            self._invlists[list_idx].extend(vids.tolist())
            part = self._partitions[list_idx]
            part["indices"] = indices
            part["norms"] = norms
            part["sign_bits"] = sign_bits
            part["codes"] = codes

        self.coarse_centroids = new_centroids
        if used_raw:
            self._raw_vectors = all_normed  # store re-normalized

        # Invalidate the C++ partition cache (the cache key is _n_vectors,
        # which doesn't change across a refresh — stale otherwise).
        if hasattr(self, "_cpp_partition_cache"):
            del self._cpp_partition_cache
        if hasattr(self, "_cpp_partition_cache_n"):
            del self._cpp_partition_cache_n

        duration = time.time() - t0
        self.refresh_count += 1
        self.refresh_total_time += duration
        self._adds_since_refresh = 0

        stats = {
            "duration_s": duration,
            "sample_size": sample_vecs.shape[0],
            "n_re_encoded": all_vecs.shape[0],
            "used_raw": used_raw,
            "refresh_count": self.refresh_count,
        }
        self.refresh_log.append(stats)
        return stats

    # ------------------------------------------------------------------
    # Add (with optional automatic refresh)
    # ------------------------------------------------------------------
    def add(self, vectors: np.ndarray, ids: Optional[np.ndarray] = None):
        super().add(vectors, ids=ids)
        self._adds_since_refresh += vectors.shape[0]
        if (self.refresh_every is not None
                and self._adds_since_refresh >= self.refresh_every):
            self.refresh()
