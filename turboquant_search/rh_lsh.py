"""
RH-IVF-TQ — Random-Hyperplane LSH partitioning with canonical cell
centers + TQ residual compression. A fully training-free ANN index.

Both layers (partition and compression) are data-independent:
    1. Random rotation Π ........................ data-independent
    2. Random hyperplanes {h_1, ..., h_L} ........ data-independent
    3. Cell index by L-bit hash sign(<x, h_i>) ... data-independent
    4. Cell centroid by closed-form formula ...... data-independent
    5. TQ residual compression (rotation+Lloyd-Max+sign) ... data-independent

Cell centroid for hash code b ∈ {-1, +1}^L:
    c(b) = (1/Z) Σ_i b_i · h_i,   Z = ||Σ_i b_i · h_i||

This is the unit vector consistent with all L hash bits being correctly
classified, by symmetry of random Gaussian hyperplanes. Requires no
training and is invariant under any distribution shift of the data.

Search probes the n_probe nearest hash codes to the query's hash, ordered
by Hamming distance.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .core import (
    IVFTurboQuantIndex,
    _get_rotation_matrix,
    _get_sub_centroids,
    _lloyd_max_codebook,
)


class RHLSHIVFTurboQuantIndex(IVFTurboQuantIndex):
    """Random-Hyperplane-LSH IVF + TQ residual compression.

    Parameters
    ----------
    dim : int
        Vector dimensionality.
    n_hyperplanes : int
        Number of random hyperplanes L. The number of partitions is 2^L,
        but only occupied cells consume storage. L ≈ log2(n_vectors / 30)
        gives ~30 vectors per cell on average. Defaults to 10 (1024 cells).
    bits : int
        Bits per coordinate for TQ residual quantization.
    nprobe : int
        Number of cells to probe per query (ordered by Hamming distance
        from query's hash code).
    use_residual_sign : bool
        Whether to use sign-bit refinement (1 extra bit/coord).
    seed : int
        Random seed.
    """

    def __init__(self, dim: int, n_hyperplanes: int = 10, bits: int = 4,
                 nprobe: int = 10, use_residual_sign: bool = True,
                 seed: int = 42):
        # Skip IVFTurboQuantIndex.__init__ — we build our own state
        self.dim = dim
        self.L = int(n_hyperplanes)
        self.nlist = 2 ** self.L
        self.bits = bits
        self.nprobe = nprobe
        self.use_residual_sign = use_residual_sign
        self.seed = seed

        # Data-independent TQ parameters
        self.rotation_matrix = _get_rotation_matrix(dim, seed)
        centroids_raw, boundaries_raw = _lloyd_max_codebook(bits)
        s = np.sqrt(dim)
        self.tq_centroids = (centroids_raw / s).astype(np.float32)
        self.tq_boundaries = (boundaries_raw / s).astype(np.float32)
        if use_residual_sign:
            self.sub_centroids = _get_sub_centroids(bits, dim)

        # Data-independent random hyperplanes (Gaussian unit vectors)
        rng = np.random.RandomState(seed + 1)
        H = rng.randn(self.L, dim).astype(np.float32)
        H /= np.linalg.norm(H, axis=1, keepdims=True)
        self.hyperplanes = H  # (L, dim)

        # Sparse storage: only occupied cells consume memory
        self._cell_data: Dict[int, Dict] = {}     # hash_int -> {indices, norms, sign_bits, ids}
        self._raw_vectors: Optional[np.ndarray] = None
        self._n_vectors = 0
        self.build_time = 0.0
        self.train_time = 0.0
        self._trained = True  # no training required

    # ------------------------------------------------------------------
    # Hashing & canonical centroids
    # ------------------------------------------------------------------
    def _hash_codes(self, vectors: np.ndarray) -> np.ndarray:
        """Return (n,) int hash codes (L-bit packed into int)."""
        bits = (vectors @ self.hyperplanes.T) > 0          # (n, L) bool
        # Pack L bits into a single int64
        powers = (1 << np.arange(self.L)).astype(np.int64)  # (L,)
        return (bits.astype(np.int64) * powers).sum(axis=1) # (n,)

    def _hash_to_signed(self, hash_int: int) -> np.ndarray:
        """Unpack int hash to a signed vector in {-1, +1}^L."""
        bits = np.zeros(self.L, dtype=np.int64)
        for i in range(self.L):
            bits[i] = (hash_int >> i) & 1
        return (2 * bits - 1).astype(np.float32)            # {-1,+1}

    def _canonical_centroid(self, hash_int: int) -> np.ndarray:
        """Compute c(b) = normalize(Σ_i b_i * h_i) for hash b."""
        signed = self._hash_to_signed(hash_int)              # (L,)
        c = signed @ self.hyperplanes                        # (dim,)
        n = np.linalg.norm(c)
        return c / max(n, 1e-8)

    # ------------------------------------------------------------------
    # Index API
    # ------------------------------------------------------------------
    def train(self, vectors: np.ndarray):
        """No-op: this index requires no training."""
        pass

    def add(self, vectors: np.ndarray, ids: Optional[np.ndarray] = None):
        v = np.ascontiguousarray(vectors.astype(np.float32))
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        n = v.shape[0]
        if ids is None:
            ids = np.arange(self._n_vectors, self._n_vectors + n)

        t0 = time.time()

        # Normalize
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        v_normed = v / np.maximum(norms, 1e-8)

        # Hash to cells
        hashes = self._hash_codes(v_normed)                 # (n,) int64

        # Group by hash code; compute residuals against canonical centroid
        unique_hashes, inverse = np.unique(hashes, return_inverse=True)
        for hi, h in enumerate(unique_hashes):
            mask = inverse == hi
            cell_vecs = v_normed[mask]
            cell_ids = ids[mask]

            c = self._canonical_centroid(int(h))             # (dim,)
            residuals = cell_vecs - c                         # (n_cell, dim)

            # TQ-compress residuals
            rotated = residuals @ self.rotation_matrix.T
            r_norms = np.linalg.norm(rotated, axis=1, keepdims=True)
            r_norms = np.maximum(r_norms, 1e-8)
            normalized = rotated / r_norms
            indices = np.digitize(normalized, self.tq_boundaries).astype(np.uint8)
            sign_bits = None
            if self.use_residual_sign:
                sign_bits = (normalized >= self.tq_centroids[indices]).astype(np.uint8)

            # Append to cell
            if int(h) not in self._cell_data:
                self._cell_data[int(h)] = {
                    "indices": indices,
                    "norms": r_norms.reshape(-1),
                    "sign_bits": sign_bits,
                    "ids": cell_ids.tolist(),
                    "centroid": c,
                }
            else:
                d = self._cell_data[int(h)]
                d["indices"] = np.concatenate([d["indices"], indices])
                d["norms"] = np.concatenate([d["norms"], r_norms.reshape(-1)])
                if sign_bits is not None and d["sign_bits"] is not None:
                    d["sign_bits"] = np.concatenate([d["sign_bits"], sign_bits])
                d["ids"].extend(cell_ids.tolist())

        # Optional raw-vector storage for re-rank
        if self._raw_vectors is None:
            self._raw_vectors = v_normed
        else:
            self._raw_vectors = np.concatenate([self._raw_vectors, v_normed])

        self._n_vectors += n
        self.build_time += time.time() - t0

    def _decode_residuals(self, indices: np.ndarray, sign_bits: Optional[np.ndarray]) -> np.ndarray:
        """Reconstruct rotated unit residuals from compressed codes."""
        if self.use_residual_sign and sign_bits is not None:
            return self.sub_centroids[indices, sign_bits].astype(np.float32)
        return self.tq_centroids[indices].astype(np.float32)

    def search(self, queries: np.ndarray, k: int = 10, rerank: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        Q = np.ascontiguousarray(queries.astype(np.float32))
        Q = np.nan_to_num(Q, nan=0.0, posinf=0.0, neginf=0.0)
        nq = Q.shape[0]
        # Normalize queries
        Q = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-8)
        Qr = Q @ self.rotation_matrix.T

        # Compute query hash codes
        Q_hashes = self._hash_codes(Q)
        # Also compute SIGNED hash for Hamming-ordered probing — we use the
        # query's continuous hyperplane scores to order cells by *expected*
        # Hamming distance, which is more accurate than raw Hamming.
        Q_scores = Q @ self.hyperplanes.T                     # (nq, L)

        occupied_hashes = np.array(list(self._cell_data.keys()), dtype=np.int64)
        if occupied_hashes.size == 0:
            return np.zeros((nq, k), dtype=np.float32), np.zeros((nq, k), dtype=np.int64)

        # For each cell we know its signed hash bits from int decoding
        cell_signed = np.array([self._hash_to_signed(int(h)) for h in occupied_hashes])  # (n_cells, L)
        # Cell scores per query: <q_scores, cell_signed> — soft Hamming distance.
        # Argsort descending = most consistent cell first.
        cell_scores = Q_scores @ cell_signed.T                # (nq, n_cells)

        np_actual = min(self.nprobe, occupied_hashes.size)
        top_cells = np.argpartition(-cell_scores, np_actual - 1, axis=1)[:, :np_actual]

        # Precompute coarse score per (query, cell) using canonical centroid
        cell_centroids = np.stack([self._cell_data[int(h)]["centroid"]
                                   for h in occupied_hashes])  # (n_cells, dim)
        coarse = Q @ cell_centroids.T                          # (nq, n_cells) inner products

        out_ids = np.full((nq, k), -1, dtype=np.int64)
        out_scores = np.full((nq, k), -np.inf, dtype=np.float32)

        for qi in range(nq):
            cells = top_cells[qi]
            cand_ids: List[np.ndarray] = []
            cand_sc: List[np.ndarray] = []
            qr = Qr[qi]
            q = Q[qi]
            for ci in cells:
                h = int(occupied_hashes[ci])
                d = self._cell_data[h]
                if not d["ids"]:
                    continue
                # Decode residuals (rotated unit) and scale by stored norms
                unit = self._decode_residuals(d["indices"], d["sign_bits"])
                rec = unit * d["norms"][:, None]               # rotated residual
                # Inner product with rotated query gives residual score
                fine = rec @ qr
                s = coarse[qi, ci] + fine                       # (n_cell,)
                cand_ids.append(np.array(d["ids"], dtype=np.int64))
                cand_sc.append(s)
            if not cand_ids:
                continue
            ids_arr = np.concatenate(cand_ids)
            sc_arr = np.concatenate(cand_sc)
            top_k = min(k + max(rerank, 0), sc_arr.shape[0])
            top = np.argpartition(-sc_arr, top_k - 1)[:top_k]
            ci = ids_arr[top]; cs = sc_arr[top]
            if rerank > 0 and self._raw_vectors is not None:
                cs = self._raw_vectors[ci] @ q
            order = np.argsort(-cs)[:k]
            out_ids[qi, :len(order)] = ci[order]
            out_scores[qi, :len(order)] = cs[order]
        return out_scores, out_ids

    @property
    def memory_bytes(self) -> int:
        bits_per_vec = self.bits * self.dim + 32              # codes + norm
        if self.use_residual_sign:
            bits_per_vec += self.dim
        return self._n_vectors * bits_per_vec // 8

    @property
    def n_occupied_cells(self) -> int:
        return len(self._cell_data)
