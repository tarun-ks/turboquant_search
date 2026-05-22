"""
QJL-based vector search index — TurboQuant's original Stage 2.

This implements TurboQuant's original Stage 2 faithfully:
  Stage 1: Random orthogonal rotation + Lloyd-Max quantization (same as core.py)
  Stage 2: QJL (Quantized Johnson-Lindenstrauss) correction using random
           Gaussian projections of the residual + 1-bit sign storage.

QJL gives UNBIASED inner product estimates (good for KV cache attention scores).
Sign-bit refinement (core.py) gives BIASED but LOWER VARIANCE estimates
(better for search ranking).

This file exists solely for empirical comparison — to measure the actual
recall difference between the two Stage 2 approaches.

Reference: QJL paper — arXiv:2406.03482
"""

import numpy as np
import time
from typing import Tuple

# Reuse Stage 1 machinery from core.py
from turboquant_search.core import (
    _lloyd_max_codebook,
    _get_rotation_matrix,
    _SCORE_MATRIX_LIMIT,
)


class QJLSearchIndex:
    """
    TurboQuant with QJL Stage 2 (TurboQuant's original approach).

    Stage 1 is identical to TurboQuantSearchIndex: rotation + Lloyd-Max.
    Stage 2 uses QJL instead of sign-bit refinement:
      - Compute residual: r = normalized - centroids[quantized_indices]
      - Project residual through random Gaussian matrix A: p = A @ r
      - Store sign(p) as 1 bit per coordinate
      - At search time, use the JL property for unbiased IP correction:
        <r, q> ≈ sqrt(pi/2) * ||r|| * (1/d) * sign(A@r)^T * (A@q)

    Memory cost is the same as sign-bit refinement: (bits+1)*dim + 32 bits
    per vector, plus 32 bits for residual norm.

    Parameters
    ----------
    dim : int
        Vector dimensionality.
    bits : int
        Lloyd-Max quantization bits (2, 3, or 4).
    seed : int
        Random seed.
    """

    def __init__(self, dim: int, bits: int = 3, seed: int = 42):
        self.dim = dim
        self.bits = bits
        self.seed = seed
        self.n_levels = 2 ** bits

        # Stage 1: same rotation and codebook as core.py
        self.rotation_matrix = _get_rotation_matrix(dim, seed)
        centroids_raw, boundaries_raw = _lloyd_max_codebook(bits)
        dim_scale = np.sqrt(dim)
        self.centroids = (centroids_raw / dim_scale).astype(np.float32)
        self.boundaries = (boundaries_raw / dim_scale).astype(np.float32)

        # Stage 2: QJL random projection matrix (separate seed from rotation)
        rng = np.random.RandomState(seed + 7777)
        self.qjl_matrix = rng.randn(dim, dim).astype(np.float32) / np.sqrt(dim)

        # Storage
        self._indices = None       # (n, dim) uint8 — Lloyd-Max bin indices
        self._norms = None         # (n,) float32 — vector norms
        self._qjl_signs = None     # (n, dim) uint8 — sign(A @ residual)
        self._residual_norms = None  # (n,) float32 — ||residual||
        self._n_vectors = 0

        self.build_time = 0.0
        self.memory_bytes = 0
        self.memory_bytes_uncompressed = 0

    def _rotate(self, vectors: np.ndarray) -> np.ndarray:
        return vectors @ self.rotation_matrix.T

    def _quantize_coords(self, rotated: np.ndarray):
        norms = np.linalg.norm(rotated, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = rotated / norms
        indices = np.digitize(normalized, self.boundaries).astype(np.uint8)
        reconstructed = self.centroids[indices] * norms
        return indices, reconstructed, norms.reshape(-1), normalized

    def add(self, vectors: np.ndarray):
        """Add vectors. Stage 1 = Lloyd-Max, Stage 2 = QJL."""
        assert vectors.shape[1] == self.dim
        vectors = vectors.astype(np.float32)
        vectors = np.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0)

        t0 = time.time()

        # Stage 1: rotate + quantize
        rotated = self._rotate(vectors)
        indices, reconstructed, norms, normalized = self._quantize_coords(rotated)

        # Stage 2: QJL on residual
        # residual = true normalized coords - quantized centroids
        residual = normalized - self.centroids[indices]  # (n, dim)
        residual_norms = np.linalg.norm(residual, axis=1)  # (n,)

        # Project residual through random Gaussian matrix, store signs
        projected = residual @ self.qjl_matrix.T  # (n, dim)
        qjl_signs = (projected >= 0).astype(np.uint8)  # (n, dim)

        # Store
        if self._indices is None:
            self._indices = indices
            self._norms = norms
            self._qjl_signs = qjl_signs
            self._residual_norms = residual_norms
        else:
            self._indices = np.concatenate([self._indices, indices])
            self._norms = np.concatenate([self._norms, norms])
            self._qjl_signs = np.concatenate([self._qjl_signs, qjl_signs])
            self._residual_norms = np.concatenate([self._residual_norms, residual_norms])

        self._n_vectors += vectors.shape[0]
        self.build_time = time.time() - t0

        # Memory: b bits per coord + 32 bits norm + 1 bit sign per coord + 32 bits residual norm
        bits_per_vector = self.bits * self.dim + 32 + self.dim + 32
        self.memory_bytes = (self._n_vectors * bits_per_vector) // 8
        self.memory_bytes_uncompressed = self._n_vectors * self.dim * 4

    def search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Asymmetric search using QJL for inner product correction.

        Score = <q_rot, Stage1_reconstruction> + QJL_correction

        Stage1 score: q_rot @ (centroids[indices] * norms).T
        QJL correction: sqrt(pi/2) * residual_norm * norm *
                        (1/dim) * sign(A@r)^T * (A @ q_normalized)

        The QJL correction is an unbiased estimate of <q_rot, residual * norm>.
        """
        queries = queries.astype(np.float32)
        queries = np.nan_to_num(queries, nan=0.0, posinf=0.0, neginf=0.0)
        nq = queries.shape[0]
        k = min(k, self._n_vectors)

        q_rotated = self._rotate(queries)

        # Stage 1 reconstruction: centroids[indices] * norms
        db_stage1 = self.centroids[self._indices] * self._norms[:, np.newaxis]

        # QJL: project query through same random matrix
        # q_projected = A @ q_rot^T, but we work with q_rot @ A^T
        q_projected = q_rotated @ self.qjl_matrix.T  # (nq, dim)

        # Convert sign bits to {-1, +1}
        signs_pm = self._qjl_signs.astype(np.float32) * 2 - 1  # (n, dim)

        # QJL scaling factor: sqrt(pi/2) for 1-bit quantization
        qjl_scale = np.sqrt(np.pi / 2.0)

        batch_size = max(1, _SCORE_MATRIX_LIMIT // max(self._n_vectors, 1))

        all_top_k_scores = np.empty((nq, k), dtype=np.float32)
        all_top_k_idx = np.empty((nq, k), dtype=np.int64)

        for start in range(0, nq, batch_size):
            end = min(start + batch_size, nq)
            batch_q_rot = q_rotated[start:end]
            batch_q_proj = q_projected[start:end]
            batch_nq = end - start

            # Stage 1 scores
            scores_stage1 = batch_q_rot @ db_stage1.T  # (batch_nq, n)

            # QJL correction: for each (query, db_vec) pair
            # correction = qjl_scale * residual_norm * norm * (signs_pm @ q_proj.T) / dim
            # signs_pm @ q_proj.T gives (n, batch_nq)
            qjl_dots = signs_pm @ batch_q_proj.T  # (n, batch_nq)
            correction = (
                qjl_scale
                * (self._residual_norms[:, np.newaxis] * self._norms[:, np.newaxis])
                * qjl_dots
                / self.dim
            )  # (n, batch_nq)

            scores = scores_stage1 + correction.T  # (batch_nq, n)

            # Top-k
            if k >= self._n_vectors:
                top_k_idx = np.argsort(-scores, axis=1)[:, :k]
            else:
                top_k_idx = np.argpartition(-scores, k, axis=1)[:, :k]
                for i in range(batch_nq):
                    order = np.argsort(-scores[i, top_k_idx[i]])
                    top_k_idx[i] = top_k_idx[i][order]

            top_k_scores = np.take_along_axis(scores, top_k_idx, axis=1)
            all_top_k_scores[start:end] = top_k_scores
            all_top_k_idx[start:end] = top_k_idx

        return all_top_k_scores, all_top_k_idx

    @property
    def compression_ratio(self) -> float:
        if self.memory_bytes == 0:
            return 0.0
        return self.memory_bytes_uncompressed / self.memory_bytes

    def stats(self) -> dict:
        return {
            "n_vectors": self._n_vectors,
            "dim": self.dim,
            "bits": self.bits,
            "stage2": "qjl",
            "memory_mb": self.memory_bytes / (1024 * 1024),
            "compression_ratio": f"{self.compression_ratio:.1f}x",
            "build_time_s": f"{self.build_time:.3f}",
        }
