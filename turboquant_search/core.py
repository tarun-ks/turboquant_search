"""
TurboQuant Search — Core Algorithm
===================================

Vector compression for similarity search, inspired by TurboQuant
(Zandieh et al., arXiv:2504.19874, 2025).

Technique:
  Stage 1: Random orthogonal rotation + Lloyd-Max optimal scalar
           quantization per coordinate.
  Stage 2: Sign-bit refinement — 1 extra bit per coordinate
           (above/below bin centroid), doubling effective resolution.

TurboQuant's original Stage 2 uses QJL for unbiased inner product
estimation (suited for KV cache). This library uses sign-bit refinement
instead because search ranking needs low variance, not unbiased
estimates. Empirically this gives +7-11pp recall improvement on
search benchmarks.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
import time


# ─────────────────────────────────────────────────────────────
# Module-level caches for expensive computations
# ─────────────────────────────────────────────────────────────

_LLOYD_MAX_CACHE: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
_ROTATION_CACHE: Dict[Tuple[int, int], np.ndarray] = {}

# Maximum elements in score matrix before batched search kicks in (~200MB)
_SCORE_MATRIX_LIMIT = 50_000_000


def _lloyd_max_codebook(bits: int, n_iter: int = 300, grid_size: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Lloyd-Max optimal codebook for the Beta(d/2, d/2) distribution.
    Results are cached by bit width since the codebook only depends on bits.

    After random orthogonal rotation, each coordinate of a unit vector follows
    approximately Beta(d/2, d/2) centered at 0. For large d, this is well
    approximated by a Gaussian N(0, 1/d). We use the Gaussian approximation
    for the Lloyd-Max optimization.

    Returns:
        centroids: (2^bits,) optimal reconstruction levels
        boundaries: (2^bits - 1,) decision boundaries
    """
    if bits in _LLOYD_MAX_CACHE:
        return _LLOYD_MAX_CACHE[bits]

    n_levels = 2 ** bits

    # Initialize with uniform quantile spacing on N(0,1)
    from scipy.stats import norm
    quantiles = np.linspace(0, 1, n_levels + 1)[1:-1]
    boundaries = norm.ppf(quantiles)

    # PDF for optimization - use standard normal
    x_grid = np.linspace(-4.0, 4.0, grid_size)
    pdf_vals = norm.pdf(x_grid)

    for _ in range(n_iter):
        # Compute centroids as conditional expectations
        centroids = np.zeros(n_levels)
        all_bounds = np.concatenate([[-np.inf], boundaries, [np.inf]])

        for i in range(n_levels):
            mask = (x_grid >= all_bounds[i]) & (x_grid < all_bounds[i + 1])
            if mask.sum() > 0:
                weighted = (x_grid[mask] * pdf_vals[mask]).sum()
                total = pdf_vals[mask].sum()
                centroids[i] = weighted / total if total > 0 else 0.0
            else:
                centroids[i] = (all_bounds[i] + all_bounds[i + 1]) / 2.0

        # Update boundaries as midpoints
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0

    result = (centroids, boundaries)
    _LLOYD_MAX_CACHE[bits] = result
    return result


def _get_rotation_matrix(dim: int, seed: int) -> np.ndarray:
    """Get or compute a cached random orthogonal rotation matrix via QR decomposition."""
    key = (dim, seed)
    if key in _ROTATION_CACHE:
        return _ROTATION_CACHE[key]

    rng = np.random.RandomState(seed)
    G = rng.randn(dim, dim).astype(np.float32)
    Q, _ = np.linalg.qr(G)
    _ROTATION_CACHE[key] = Q
    return Q


_SUB_CENTROID_CACHE: Dict[Tuple[int, int], np.ndarray] = {}


def _get_sub_centroids(bits: int, dim: int) -> np.ndarray:
    """
    Compute sub-centroids for sign-bit refinement.

    For each Lloyd-Max bin, splits it at the centroid and computes
    the conditional expectation for the lower and upper halves.
    This gives 2x finer reconstruction using 1 extra bit (the sign
    of the residual within each bin).

    Returns:
        sub_centroids: (2^bits, 2) array — [bin_idx, 0=lower / 1=upper]
    """
    key = (bits, dim)
    if key in _SUB_CENTROID_CACHE:
        return _SUB_CENTROID_CACHE[key]

    from scipy.stats import norm

    centroids_raw, boundaries_raw = _lloyd_max_codebook(bits)
    scale = np.sqrt(dim)
    centroids = centroids_raw / scale
    boundaries = boundaries_raw / scale

    n_levels = 2 ** bits
    all_bounds = np.concatenate([[-4.0 / scale], boundaries, [4.0 / scale]])

    grid = np.linspace(-4.0 / scale, 4.0 / scale, 50000)
    pdf = norm.pdf(grid * scale) * scale  # PDF of N(0, 1/dim)

    sub_centroids = np.zeros((n_levels, 2), dtype=np.float32)
    for i in range(n_levels):
        lo, hi = all_bounds[i], all_bounds[i + 1]
        mid = centroids[i]

        mask_lo = (grid >= lo) & (grid < mid)
        if mask_lo.sum() > 0:
            sub_centroids[i, 0] = np.average(grid[mask_lo], weights=pdf[mask_lo])
        else:
            sub_centroids[i, 0] = (lo + mid) / 2

        mask_hi = (grid >= mid) & (grid <= hi)
        if mask_hi.sum() > 0:
            sub_centroids[i, 1] = np.average(grid[mask_hi], weights=pdf[mask_hi])
        else:
            sub_centroids[i, 1] = (mid + hi) / 2

    _SUB_CENTROID_CACHE[key] = sub_centroids
    return sub_centroids


class TurboQuantSearchIndex:
    """
    TurboQuant-compressed vector search index.

    Compresses high-dimensional vectors using random orthogonal rotation
    followed by per-coordinate Lloyd-Max quantization, with optional
    sign-bit refinement that doubles effective resolution using 1 extra
    bit per coordinate.

    Parameters
    ----------
    dim : int
        Dimensionality of input vectors.
    bits : int
        Bits per coordinate for quantization (2, 3, or 4).
    use_residual_sign : bool
        Whether to apply sign-bit refinement (1 extra bit per coordinate).
        Doubles effective quantization levels. Default True.
    seed : int
        Random seed for reproducibility of the rotation matrix.
    """

    def __init__(self, dim: int, bits: int = 3, use_residual_sign: bool = True, seed: int = 42,
                 # Backward compat alias
                 use_qjl: bool = None):
        self.dim = dim
        self.bits = bits
        # use_qjl is kept as an alias for backward compatibility
        if use_qjl is not None:
            use_residual_sign = use_qjl
        self.use_qjl = use_residual_sign  # backward compat property
        self.use_residual_sign = use_residual_sign
        self.seed = seed
        self.n_levels = 2 ** bits

        # Cached random orthogonal rotation matrix via QR decomposition
        self.rotation_matrix = _get_rotation_matrix(dim, seed)

        # Cached Lloyd-Max codebook, scaled to match the distribution of
        # normalized rotated coordinates: N(0, 1/dim) instead of N(0, 1).
        centroids_raw, boundaries_raw = _lloyd_max_codebook(bits)
        dim_scale = np.sqrt(dim)
        self.centroids = (centroids_raw / dim_scale).astype(np.float32)
        self.boundaries = (boundaries_raw / dim_scale).astype(np.float32)

        # Sign-bit refinement: for each bin, pre-compute the conditional
        # centroid for the lower and upper halves. Storing 1 extra bit
        # (sign of residual) per coordinate doubles the effective resolution.
        if use_residual_sign:
            self.sub_centroids = _get_sub_centroids(bits, dim)

        # Storage
        self._indices = None       # (n, dim) uint8 quantization indices
        self._norms = None         # (n,) vector norms
        self._sign_bits = None     # (n, dim) 1-bit residual sign per coordinate
        self._n_vectors = 0

        # Metadata
        self.build_time = 0.0
        self.memory_bytes = 0
        self.memory_bytes_uncompressed = 0

    def _rotate(self, vectors: np.ndarray) -> np.ndarray:
        """Apply random orthogonal rotation."""
        return vectors @ self.rotation_matrix.T

    def _quantize_coords(self, rotated: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Scalar quantize each coordinate using Lloyd-Max codebook.

        Returns:
            indices: (n, dim) quantization bin indices
            reconstructed: (n, dim) dequantized values
            norms: (n,) vector norms
        """
        # Normalize by vector norms for unit-sphere quantization
        norms = np.linalg.norm(rotated, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = rotated / norms

        # Quantize using boundaries
        indices = np.digitize(normalized, self.boundaries).astype(np.uint8)

        # Reconstruct
        reconstructed = self.centroids[indices] * norms

        return indices, reconstructed, norms.reshape(-1)

    def _encode_sign_bits(self, normalized: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """
        Sign-bit refinement: store whether each coordinate's value is above
        or below its bin centroid. This 1 extra bit per coordinate doubles
        the effective quantization resolution.
        """
        residual = normalized - self.centroids[indices]
        return (residual >= 0).astype(np.uint8)

    def add(self, vectors: np.ndarray):
        """
        Add vectors to the index.

        Parameters
        ----------
        vectors : np.ndarray of shape (n, dim)
            Vectors to index. Will be compressed using TurboQuant.
        """
        assert vectors.shape[1] == self.dim, f"Expected dim={self.dim}, got {vectors.shape[1]}"
        vectors = vectors.astype(np.float32)
        # Replace NaN/inf with zeros to prevent matmul warnings
        vectors = np.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0)

        t0 = time.time()

        # Stage 1: Rotate and quantize
        rotated = self._rotate(vectors)
        indices, reconstructed, norms = self._quantize_coords(rotated)

        # Stage 2: Sign-bit refinement (1 extra bit per coordinate)
        sign_bits = None
        if self.use_residual_sign:
            normalized = rotated / np.maximum(norms[:, np.newaxis], 1e-8)
            sign_bits = self._encode_sign_bits(normalized, indices)

        # Store
        if self._indices is None:
            self._indices = indices
            self._norms = norms
            self._sign_bits = sign_bits
        else:
            self._indices = np.concatenate([self._indices, indices])
            self._norms = np.concatenate([self._norms, norms])
            if sign_bits is not None:
                self._sign_bits = np.concatenate([self._sign_bits, sign_bits])

        self._n_vectors += vectors.shape[0]
        self.build_time = time.time() - t0

        # Calculate memory usage
        bits_per_vector = self.bits * self.dim  # quantized coordinates
        bits_per_vector += 32  # norm (float32)
        if self.use_residual_sign:
            bits_per_vector += self.dim  # 1-bit sign per coordinate

        self.memory_bytes = (self._n_vectors * bits_per_vector) // 8
        self.memory_bytes_uncompressed = self._n_vectors * self.dim * 4  # float32

    def search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors using compressed representations.

        Uses asymmetric distance computation: query is kept in full precision,
        database vectors are in compressed form. Batches over queries when
        the score matrix would exceed the memory budget.

        Parameters
        ----------
        queries : np.ndarray of shape (nq, dim)
            Query vectors.
        k : int
            Number of neighbors to return.

        Returns
        -------
        distances : np.ndarray of shape (nq, k)
            Inner product scores (higher = more similar).
        indices : np.ndarray of shape (nq, k)
            Indices of nearest neighbors.
        """
        queries = queries.astype(np.float32)
        queries = np.nan_to_num(queries, nan=0.0, posinf=0.0, neginf=0.0)
        nq = queries.shape[0]
        k = min(k, self._n_vectors)

        # Rotate queries (but don't quantize — asymmetric search)
        q_rotated = self._rotate(queries)

        # ── C++ fast path: fused reconstruct + score + top-k ──
        try:
            from ._tqs_cpp import tq_flat_search as _cpp_flat_search
            use_sign = self.use_residual_sign and self._sign_bits is not None
            return _cpp_flat_search(
                self.sub_centroids if use_sign else np.empty((0, 2), dtype=np.float32),
                np.ascontiguousarray(self._indices),
                np.ascontiguousarray(self._sign_bits) if use_sign else np.empty((0, 0), dtype=np.uint8),
                np.ascontiguousarray(self._norms),
                self.centroids,
                np.ascontiguousarray(q_rotated),
                use_sign,
                k,
            )
        except ImportError:
            pass

        # ── NumPy fallback ──
        # Pre-compute database reconstruction using sign-bit refinement if available
        if self.use_residual_sign and self._sign_bits is not None:
            # Use sub-centroids: sub_centroids[bin_idx, sign_bit] for each coordinate
            db_reconstructed = self.sub_centroids[self._indices, self._sign_bits] * self._norms[:, np.newaxis]
        else:
            db_reconstructed = self.centroids[self._indices] * self._norms[:, np.newaxis]

        # Determine batch size to limit memory usage
        batch_size = max(1, _SCORE_MATRIX_LIMIT // max(self._n_vectors, 1))

        all_top_k_scores = np.empty((nq, k), dtype=np.float32)
        all_top_k_idx = np.empty((nq, k), dtype=np.int64)

        for start in range(0, nq, batch_size):
            end = min(start + batch_size, nq)
            batch_q = q_rotated[start:end]
            batch_nq = end - start

            # Compute inner products: q_rot . db_reconstructed
            # (Since rotation is orthogonal, <Rx, Ry> = <x, y>)
            scores = batch_q @ db_reconstructed.T  # (batch_nq, n)

            # Top-k selection
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

    def compress_with_details(self, vector: np.ndarray) -> dict:
        """
        Return intermediate compression results for visualization.

        Parameters
        ----------
        vector : np.ndarray of shape (dim,) or (1, dim)

        Returns
        -------
        dict with keys: original, rotated, norm, quantized_indices,
            reconstructed, residual, reconstruction_error,
            and if use_residual_sign: sign_bits, refined_reconstructed, refined_error
        """
        vector = vector.astype(np.float32)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)

        rotated = self._rotate(vector)
        indices, reconstructed, norms = self._quantize_coords(rotated)

        norms_val = float(norms) if np.ndim(norms) == 0 else float(norms[0])
        residual = (rotated - reconstructed).squeeze()

        result = {
            "original": vector.squeeze(),
            "rotated": rotated.squeeze(),
            "norm": norms_val,
            "quantized_indices": indices.squeeze(),
            "reconstructed": reconstructed.squeeze(),
            "residual": residual,
            "reconstruction_error": float(np.linalg.norm(residual)),
        }

        if self.use_residual_sign:
            normalized_v = rotated / np.maximum(norms_val, 1e-8)
            sign_bits = self._encode_sign_bits(normalized_v, indices)
            refined = self.sub_centroids[indices, sign_bits] * norms_val
            result["sign_bits"] = sign_bits.squeeze()
            result["refined_reconstructed"] = refined.squeeze()
            result["refined_error"] = float(np.linalg.norm((rotated - refined).squeeze()))

        return result

    @property
    def compression_ratio(self) -> float:
        """Compression ratio vs float32."""
        if self.memory_bytes == 0:
            return 0.0
        return self.memory_bytes_uncompressed / self.memory_bytes

    def stats(self) -> dict:
        """Return index statistics."""
        return {
            "n_vectors": self._n_vectors,
            "dim": self.dim,
            "bits": self.bits,
            "residual_mode": "sign-refine" if self.use_residual_sign else "none",
            "memory_mb": self.memory_bytes / (1024 * 1024),
            "memory_uncompressed_mb": self.memory_bytes_uncompressed / (1024 * 1024),
            "compression_ratio": f"{self.compression_ratio:.1f}x",
            "build_time_s": f"{self.build_time:.3f}",
        }


class IVFTurboQuantIndex:
    """
    IVF-TQ: Inverted File Index with TurboQuant compression.

    Combines IVF partitioning (k-means) with training-free TQ compression.
    Only the coarse quantizer (centroids) requires training — per-vector
    compression is instant, enabling incremental updates without codebook
    drift.

    Compared to IVF-PQ:
      - PQ codebooks are trained on data and degrade when distribution shifts.
      - TQ compression depends only on dimension and bit width, not data.
      - New vectors can be added to any partition without retraining.

    Parameters
    ----------
    dim : int
        Dimensionality of input vectors.
    nlist : int
        Number of IVF partitions (Voronoi cells).
    bits : int
        TQ bits per coordinate (2, 3, or 4).
    nprobe : int
        Number of partitions to search per query.
    use_residual_sign : bool
        Whether to apply sign-bit refinement. Default True.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, dim: int, nlist: int = 100, bits: int = 3,
                 nprobe: int = 10, use_residual_sign: bool = True,
                 seed: int = 42):
        self.dim = dim
        self.nlist = nlist
        self.bits = bits
        self.nprobe = nprobe
        self.use_residual_sign = use_residual_sign
        self.seed = seed

        # Shared TQ parameters — these do NOT depend on data
        self.rotation_matrix = _get_rotation_matrix(dim, seed)
        centroids_raw, boundaries_raw = _lloyd_max_codebook(bits)
        dim_scale = np.sqrt(dim)
        self.tq_centroids = (centroids_raw / dim_scale).astype(np.float32)
        self.tq_boundaries = (boundaries_raw / dim_scale).astype(np.float32)
        if use_residual_sign:
            self.sub_centroids = _get_sub_centroids(bits, dim)

        # Coarse quantizer — trained via k-means
        self.coarse_centroids = None  # (nlist, dim)
        self._trained = False

        # Per-partition compressed storage
        self._invlists: List[List[int]] = [[] for _ in range(nlist)]
        self._partitions: List[Dict] = [
            {"indices": None, "norms": None, "sign_bits": None, "codes": None}
            for _ in range(nlist)
        ]

        # Original normalized vectors for re-ranking (optional)
        self._raw_vectors: Optional[np.ndarray] = None

        self._n_vectors = 0
        self.build_time = 0.0
        self.train_time = 0.0

    def train(self, vectors: np.ndarray):
        """
        Train the coarse quantizer via k-means.

        This is the only training step. TQ compression within each
        partition requires no training at all.
        """
        vectors = vectors.astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors_normed = vectors / np.maximum(norms, 1e-8)

        actual_nlist = min(self.nlist, vectors.shape[0])

        t0 = time.time()
        from sklearn.cluster import MiniBatchKMeans
        kmeans = MiniBatchKMeans(
            n_clusters=actual_nlist,
            random_state=self.seed,
            batch_size=min(10000, vectors_normed.shape[0]),
            n_init=3,
            max_iter=50,
        )
        kmeans.fit(vectors_normed)
        self.coarse_centroids = kmeans.cluster_centers_.astype(np.float32)
        # Normalize centroids for IP search
        c_norms = np.linalg.norm(self.coarse_centroids, axis=1, keepdims=True)
        self.coarse_centroids = self.coarse_centroids / np.maximum(c_norms, 1e-8)

        self.nlist = actual_nlist
        self._invlists = [[] for _ in range(actual_nlist)]
        self._partitions = [
            {"indices": None, "norms": None, "sign_bits": None}
            for _ in range(actual_nlist)
        ]
        self._trained = True
        self.train_time = time.time() - t0

    def _tq_compress(self, residuals: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Compress residual vectors using TQ (no training)."""
        rotated = residuals @ self.rotation_matrix.T
        norms = np.linalg.norm(rotated, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = rotated / norms

        indices = np.digitize(normalized, self.tq_boundaries).astype(np.uint8)

        sign_bits = None
        if self.use_residual_sign:
            residual_from_centroid = normalized - self.tq_centroids[indices]
            sign_bits = (residual_from_centroid >= 0).astype(np.uint8)

        return indices, norms.reshape(-1), sign_bits

    def add(self, vectors: np.ndarray, ids: Optional[np.ndarray] = None):
        """
        Add vectors to the index. Assigns to nearest centroid and
        compresses residuals with TQ — no codebook training needed.
        """
        assert self._trained, "Must call train() before add()"
        vectors = vectors.astype(np.float32)
        vectors = np.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0)
        n = vectors.shape[0]

        if ids is None:
            ids = np.arange(self._n_vectors, self._n_vectors + n)

        t0 = time.time()

        # Normalize
        v_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors_normed = vectors / np.maximum(v_norms, 1e-8)

        # Assign to nearest centroid
        assignments = np.argmax(
            vectors_normed @ self.coarse_centroids.T, axis=1
        )

        # Compress residuals per partition
        for list_idx in range(self.nlist):
            mask = assignments == list_idx
            if not mask.any():
                continue

            vecs = vectors_normed[mask]
            vids = ids[mask]

            residuals = vecs - self.coarse_centroids[list_idx]
            indices, norms, sign_bits = self._tq_compress(residuals)

            # Precompute combined codes: code[d] = indices[d]*2 + sign_bits[d]
            # This halves memory loads in the C++ inner loop
            if sign_bits is not None:
                codes = (indices.astype(np.uint8) * 2 + sign_bits.astype(np.uint8)).astype(np.uint8)
            else:
                codes = indices.astype(np.uint8)

            self._invlists[list_idx].extend(vids.tolist())
            part = self._partitions[list_idx]
            if part["indices"] is None:
                part["indices"] = indices
                part["norms"] = norms
                part["sign_bits"] = sign_bits
                part["codes"] = codes
            else:
                part["indices"] = np.concatenate([part["indices"], indices])
                part["norms"] = np.concatenate([part["norms"], norms])
                if sign_bits is not None and part["sign_bits"] is not None:
                    part["sign_bits"] = np.concatenate(
                        [part["sign_bits"], sign_bits]
                    )
                part["codes"] = np.concatenate([part["codes"], codes])

        # Store raw vectors for re-ranking
        if self._raw_vectors is None:
            self._raw_vectors = vectors_normed
        else:
            self._raw_vectors = np.concatenate([self._raw_vectors, vectors_normed])

        self._n_vectors += n
        self.build_time += time.time() - t0

    def add_single(self, vector: np.ndarray, vector_id: Optional[int] = None):
        """
        Add a single vector without any retraining.

        This is the key advantage over IVF-PQ: per-vector compression
        requires no codebook, so new vectors can be added instantly
        without degrading compression quality.
        """
        vector = vector.astype(np.float32).reshape(1, -1)
        vid = vector_id if vector_id is not None else self._n_vectors
        self.add(vector, ids=np.array([vid]))

    def search(self, queries: np.ndarray, k: int = 10, rerank: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search nprobe partitions for k nearest neighbors.

        Uses TQ asymmetric distance on residuals within each probed
        partition, then merges results across partitions.

        Parameters
        ----------
        queries : np.ndarray of shape (nq, dim)
        k : int
            Number of neighbors to return.
        rerank : int
            If > 0, retrieve this many candidates from compressed search,
            then re-rank with exact inner products. E.g. rerank=100 gets
            top-100 from TQ, recomputes exact scores, returns top-k.
            Requires raw vectors to be stored (default when using add()).
        """
        queries = queries.astype(np.float32)
        queries = np.nan_to_num(queries, nan=0.0, posinf=0.0, neginf=0.0)
        nq = queries.shape[0]
        k = min(k, self._n_vectors)

        # If re-ranking, retrieve more candidates from compressed search
        search_k = max(k, rerank) if rerank > 0 else k

        # Normalize queries
        q_norms = np.linalg.norm(queries, axis=1, keepdims=True)
        queries_normed = queries / np.maximum(q_norms, 1e-8)

        # Rotate queries once (shared across all probes)
        q_rotated = queries_normed @ self.rotation_matrix.T

        # Coarse scores: <query, centroid>
        coarse_scores = queries_normed @ self.coarse_centroids.T  # (nq, nlist)

        # Find nprobe nearest centroids per query
        actual_nprobe = min(self.nprobe, self.nlist)
        if actual_nprobe >= self.nlist:
            top_lists = np.tile(np.arange(self.nlist), (nq, 1))
        else:
            top_lists = np.argpartition(
                -coarse_scores, actual_nprobe, axis=1
            )[:, :actual_nprobe]

        # ── C++ fast path with Python-level thread parallelism ──
        cpp_result = None
        try:
            from ._tqs_cpp import ivf_search as _cpp_ivf_search
            # Cache partition data to avoid rebuilding on every search
            if not hasattr(self, '_cpp_partition_cache') or self._cpp_partition_cache_n != self._n_vectors:
                partition_data = []
                for list_idx in range(self.nlist):
                    part = self._partitions[list_idx]
                    ids = np.array(self._invlists[list_idx], dtype=np.int64)
                    partition_data.append({
                        "indices": part["indices"],
                        "norms": part["norms"],
                        "sign_bits": part["sign_bits"],
                        "codes": part.get("codes"),
                        "ids": ids,
                    })
                self._cpp_partition_cache = partition_data
                self._cpp_partition_cache_n = self._n_vectors
            partition_data = self._cpp_partition_cache
            sub_centroids = self.sub_centroids if self.use_residual_sign else np.empty((0, 2), dtype=np.float32)
            q_rot_c = np.ascontiguousarray(q_rotated)
            cs_c = np.ascontiguousarray(coarse_scores)
            tl_c = top_lists.astype(np.int32)

            import os
            n_threads = min(nq, int(os.environ.get("TQS_THREADS", os.cpu_count() or 1)))

            if n_threads > 1 and nq >= 4:
                from concurrent.futures import ThreadPoolExecutor
                chunk_size = (nq + n_threads - 1) // n_threads

                def _search_chunk(start):
                    end = min(start + chunk_size, nq)
                    return _cpp_ivf_search(
                        sub_centroids, self.tq_centroids,
                        partition_data,
                        q_rot_c[start:end],
                        cs_c[start:end],
                        tl_c[start:end],
                        self.use_residual_sign,
                        search_k,
                    )

                with ThreadPoolExecutor(max_workers=n_threads) as pool:
                    futures = [pool.submit(_search_chunk, i * chunk_size)
                               for i in range(n_threads) if i * chunk_size < nq]
                    chunks = [f.result() for f in futures]

                all_scores = np.concatenate([c[0] for c in chunks], axis=0)
                all_indices = np.concatenate([c[1] for c in chunks], axis=0)
                cpp_result = (all_scores, all_indices)
            else:
                cpp_result = _cpp_ivf_search(
                    sub_centroids, self.tq_centroids,
                    partition_data, q_rot_c, cs_c, tl_c,
                    self.use_residual_sign, search_k,
                )
        except ImportError:
            pass

        if cpp_result is not None:
            all_scores, all_indices = cpp_result
        else:
            # ── NumPy fallback ──
            list_to_queries: Dict[int, List[int]] = {}
            for q_idx in range(nq):
                for l in top_lists[q_idx]:
                    list_to_queries.setdefault(int(l), []).append(q_idx)

            cand_scores: List[List[np.ndarray]] = [[] for _ in range(nq)]
            cand_ids: List[List[np.ndarray]] = [[] for _ in range(nq)]

            for list_idx, q_indices_list in list_to_queries.items():
                part = self._partitions[list_idx]
                if part["indices"] is None:
                    continue

                q_indices = np.array(q_indices_list)

                if self.use_residual_sign and part["sign_bits"] is not None:
                    db_recon = (
                        self.sub_centroids[part["indices"], part["sign_bits"]]
                        * part["norms"][:, np.newaxis]
                    )
                else:
                    db_recon = (
                        self.tq_centroids[part["indices"]]
                        * part["norms"][:, np.newaxis]
                    )

                batch_q = q_rotated[q_indices]
                fine = batch_q @ db_recon.T
                batch_coarse = coarse_scores[q_indices, list_idx]
                total = fine + batch_coarse[:, np.newaxis]

                list_ids = np.array(self._invlists[list_idx], dtype=np.int64)
                for i, q_idx in enumerate(q_indices_list):
                    cand_scores[q_idx].append(total[i])
                    cand_ids[q_idx].append(list_ids)

            all_scores = np.full((nq, search_k), -np.inf, dtype=np.float32)
            all_indices = np.full((nq, search_k), -1, dtype=np.int64)

            for q_idx in range(nq):
                if not cand_scores[q_idx]:
                    continue
                scores = np.concatenate(cand_scores[q_idx])
                ids = np.concatenate(cand_ids[q_idx])

                actual_k = min(search_k, len(scores))
                if actual_k >= len(scores):
                    top_k = np.argsort(-scores)[:actual_k]
                else:
                    top_k = np.argpartition(-scores, actual_k)[:actual_k]
                    top_k = top_k[np.argsort(-scores[top_k])]

                all_scores[q_idx, :actual_k] = scores[top_k]
                all_indices[q_idx, :actual_k] = ids[top_k]

        # ── Re-ranking: recompute exact inner products for top candidates ──
        if rerank > 0 and self._raw_vectors is not None:
            for q_idx in range(nq):
                cand_idx = all_indices[q_idx]
                valid = cand_idx >= 0
                if not valid.any():
                    continue
                cand_idx_valid = cand_idx[valid].astype(int)
                exact_scores = queries_normed[q_idx] @ self._raw_vectors[cand_idx_valid].T
                rerank_order = np.argsort(-exact_scores)[:k]
                all_scores[q_idx, :len(rerank_order)] = exact_scores[rerank_order]
                all_indices[q_idx, :len(rerank_order)] = cand_idx_valid[rerank_order]
                if len(rerank_order) < k:
                    all_scores[q_idx, len(rerank_order):] = -np.inf
                    all_indices[q_idx, len(rerank_order):] = -1
            all_scores = all_scores[:, :k]
            all_indices = all_indices[:, :k]

        return all_scores, all_indices

    @property
    def memory_bytes(self) -> int:
        bits_per_vector = self.bits * self.dim + 32  # quant indices + norm
        if self.use_residual_sign:
            bits_per_vector += self.dim  # sign bits
        vector_bytes = (self._n_vectors * bits_per_vector) // 8
        centroid_bytes = self.nlist * self.dim * 4
        return vector_bytes + centroid_bytes

    @property
    def compression_ratio(self) -> float:
        uncompressed = self._n_vectors * self.dim * 4
        if self.memory_bytes == 0:
            return 0.0
        return uncompressed / self.memory_bytes

    def stats(self) -> dict:
        list_sizes = [len(il) for il in self._invlists]
        return {
            "n_vectors": self._n_vectors,
            "dim": self.dim,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "bits": self.bits,
            "residual_mode": "sign-refine" if self.use_residual_sign else "none",
            "memory_mb": self.memory_bytes / (1024 * 1024),
            "memory_uncompressed_mb": (self._n_vectors * self.dim * 4)
            / (1024 * 1024),
            "compression_ratio": f"{self.compression_ratio:.1f}x",
            "train_time_s": f"{self.train_time:.3f}",
            "build_time_s": f"{self.build_time:.3f}",
            "avg_list_size": np.mean(list_sizes) if list_sizes else 0,
            "max_list_size": max(list_sizes) if list_sizes else 0,
        }


class FlatSearchIndex:
    """Brute-force exact search baseline."""

    def __init__(self, dim: int):
        self.dim = dim
        self._vectors = None
        self._n_vectors = 0
        self.build_time = 0.0

    def add(self, vectors: np.ndarray):
        vectors = vectors.astype(np.float32)
        t0 = time.time()
        if self._vectors is None:
            self._vectors = vectors
        else:
            self._vectors = np.concatenate([self._vectors, vectors])
        self._n_vectors += vectors.shape[0]
        self.build_time = time.time() - t0

    def search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        queries = queries.astype(np.float32)
        k = min(k, self._n_vectors)
        scores = queries @ self._vectors.T
        if k >= self._n_vectors:
            top_k_idx = np.argsort(-scores, axis=1)[:, :k]
        else:
            top_k_idx = np.argpartition(-scores, k, axis=1)[:, :k]
            for i in range(queries.shape[0]):
                order = np.argsort(-scores[i, top_k_idx[i]])
                top_k_idx[i] = top_k_idx[i][order]
        top_k_scores = np.take_along_axis(scores, top_k_idx, axis=1)
        return top_k_scores, top_k_idx

    @property
    def memory_bytes(self):
        return self._n_vectors * self.dim * 4

    def stats(self) -> dict:
        return {
            "n_vectors": self._n_vectors,
            "dim": self.dim,
            "bits": 32,
            "memory_mb": self.memory_bytes / (1024 * 1024),
            "compression_ratio": "1.0x (baseline)",
            "build_time_s": f"{self.build_time:.3f}",
        }


class ProductQuantizationIndex:
    """
    Product Quantization baseline for comparison.

    Splits vectors into subspaces and quantizes each independently
    using k-means clustering.
    """

    def __init__(self, dim: int, n_subspaces: int = 8, n_clusters: int = 256, seed: int = 42):
        assert dim % n_subspaces == 0
        self.dim = dim
        self.n_subspaces = n_subspaces
        self.sub_dim = dim // n_subspaces
        self.n_clusters = n_clusters
        self.seed = seed

        self._codes = None  # (n, n_subspaces) uint8
        self._codebooks = None  # (n_subspaces, n_clusters, sub_dim)
        self._n_vectors = 0
        self.build_time = 0.0

    def _train_codebooks(self, vectors: np.ndarray):
        """Train PQ codebooks using k-means on subspaces."""
        from sklearn.cluster import MiniBatchKMeans

        n = vectors.shape[0]
        actual_clusters = min(self.n_clusters, n)
        self._actual_clusters = actual_clusters
        self._codebooks = np.zeros((self.n_subspaces, actual_clusters, self.sub_dim), dtype=np.float32)

        for m in range(self.n_subspaces):
            sub_vectors = vectors[:, m * self.sub_dim:(m + 1) * self.sub_dim]
            kmeans = MiniBatchKMeans(
                n_clusters=actual_clusters,
                random_state=self.seed,
                batch_size=min(1000, n),
                n_init=1,
                max_iter=20
            )
            kmeans.fit(sub_vectors)
            self._codebooks[m] = kmeans.cluster_centers_

    def add(self, vectors: np.ndarray):
        vectors = vectors.astype(np.float32)
        t0 = time.time()

        # Train codebooks
        self._train_codebooks(vectors)

        # Encode
        codes = np.zeros((vectors.shape[0], self.n_subspaces), dtype=np.uint8)
        for m in range(self.n_subspaces):
            sub_vectors = vectors[:, m * self.sub_dim:(m + 1) * self.sub_dim]
            # Find nearest centroid
            dists = np.sum((sub_vectors[:, np.newaxis, :] - self._codebooks[m][np.newaxis, :, :]) ** 2, axis=2)
            codes[:, m] = np.argmin(dists, axis=1).astype(np.uint8)

        self._codes = codes
        self._n_vectors = vectors.shape[0]
        self.build_time = time.time() - t0

    def search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        queries = queries.astype(np.float32)
        nq = queries.shape[0]
        k = min(k, self._n_vectors)

        # Precompute distance tables
        # For inner product: score = sum_m q_m . codebook[m][code[m]]
        n_cb = self._codebooks.shape[1]
        dist_tables = np.zeros((nq, self.n_subspaces, n_cb), dtype=np.float32)
        for m in range(self.n_subspaces):
            q_sub = queries[:, m * self.sub_dim:(m + 1) * self.sub_dim]
            dist_tables[:, m, :] = q_sub @ self._codebooks[m].T

        # Compute scores using lookup
        scores = np.zeros((nq, self._n_vectors), dtype=np.float32)
        for m in range(self.n_subspaces):
            scores += dist_tables[:, m, :][:, self._codes[:, m]]

        # Top-k
        if k >= self._n_vectors:
            top_k_idx = np.argsort(-scores, axis=1)[:, :k]
        else:
            top_k_idx = np.argpartition(-scores, k, axis=1)[:, :k]
            for i in range(nq):
                order = np.argsort(-scores[i, top_k_idx[i]])
                top_k_idx[i] = top_k_idx[i][order]

        top_k_scores = np.take_along_axis(scores, top_k_idx, axis=1)
        return top_k_scores, top_k_idx

    @property
    def memory_bytes(self):
        # codes: n * n_subspaces * 8 bits
        # codebooks: n_subspaces * n_clusters * sub_dim * 32 bits
        code_bytes = self._n_vectors * self.n_subspaces
        codebook_bytes = self.n_subspaces * self.n_clusters * self.sub_dim * 4
        return code_bytes + codebook_bytes

    def stats(self) -> dict:
        uncompressed = self._n_vectors * self.dim * 4
        return {
            "n_vectors": self._n_vectors,
            "dim": self.dim,
            "bits": f"8 (PQ, {self.n_subspaces} subspaces)",
            "memory_mb": self.memory_bytes / (1024 * 1024),
            "compression_ratio": f"{uncompressed / self.memory_bytes:.1f}x",
            "build_time_s": f"{self.build_time:.3f}",
        }
