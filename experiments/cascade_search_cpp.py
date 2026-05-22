"""C++-accelerated cascade search.

Wraps the new turboquant_search._tqs_cpp.cascade_search entry point.
Operates on the production IVFTQIndex (turboquant_search.core) which already
maintains combined `codes` (= indices*2 + sign_bits) per partition. We
add a per-partition `msb_codes` cache (uint8, primary >> lsb_count).
"""
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os


def _ensure_cascade_caches(idx, top_msb_bits):
    """Pre-compute per-partition msb_codes for the requested MSB width.

    Cached on the index so repeated searches with the same top_msb_bits are free.
    """
    full_bits = idx.bits
    if top_msb_bits >= full_bits:
        raise ValueError(
            f"top_msb_bits={top_msb_bits} must be < full bits={full_bits}"
        )
    lsb_count = full_bits - top_msb_bits

    cached = getattr(idx, "_cascade_cache_msb_bits", None)
    if cached == top_msb_bits and getattr(idx, "_cascade_cache_n", 0) == idx._n_vectors:
        return  # already valid

    for cell in range(idx.nlist):
        part = idx._partitions[cell]
        if part is None or part.get("indices") is None:
            continue
        primary = part["indices"]
        # uint8 primary; MSB = primary >> lsb_count, fits in uint8.
        msb = (primary >> lsb_count).astype(np.uint8, copy=False)
        part["msb_codes"] = np.ascontiguousarray(msb)

    idx._cascade_cache_msb_bits = top_msb_bits
    idx._cascade_cache_n = idx._n_vectors


def _build_coarse_recon(idx, top_msb_bits):
    """Build the per-MSB-bin Pass-1 reconstruction (one scalar per MSB level)."""
    full_bits = idx.bits
    coarse_levels = 2 ** top_msb_bits
    n_lsb = 2 ** (full_bits - top_msb_bits)
    if idx.use_residual_sign:
        # sub_centroids: (n_levels, 2). Group n_lsb LSB-bins × 2 sign-halves per MSB.
        flat = idx.sub_centroids.reshape(coarse_levels, n_lsb, 2)
        coarse_recon = flat.mean(axis=(1, 2)).astype(np.float32)
    else:
        flat = idx.tq_centroids.reshape(coarse_levels, n_lsb)
        coarse_recon = flat.mean(axis=1).astype(np.float32)
    return coarse_recon


def search_cascade_cpp(idx, queries, k=10, top_msb_bits=4, rerank_n=100,
                       n_threads=None):
    """C++-accelerated two-pass cascade search.

    Parameters mirror the Python search_cascade in cascade_search.py.
    Returns: (out_indices, total_time_s)  — total_time_s measured around the
    C++ call only (excludes Python prep cost, which is amortised).
    """
    from turboquant_search._tqs_cpp import cascade_search as _cpp_cascade

    queries = np.ascontiguousarray(queries.astype(np.float32))
    nq = queries.shape[0]

    # Pre-compute MSB caches (one-shot per (idx, top_msb_bits) combination).
    _ensure_cascade_caches(idx, top_msb_bits)
    coarse_recon = _build_coarse_recon(idx, top_msb_bits)

    # Query rotation + coarse scoring (matches IVFTQIndex.search prep).
    q_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    q_normed = queries / np.maximum(q_norms, 1e-8)
    q_rotated = np.ascontiguousarray(q_normed @ idx.rotation_matrix.T, dtype=np.float32)
    coarse_scores = np.ascontiguousarray(q_normed @ idx.coarse_centroids.T,
                                          dtype=np.float32)
    nprobe = min(idx.nprobe, idx.nlist)
    if nprobe >= idx.nlist:
        top_lists = np.tile(np.arange(idx.nlist, dtype=np.int32), (nq, 1))
    else:
        top_lists = np.argpartition(
            -coarse_scores, nprobe, axis=1
        )[:, :nprobe].astype(np.int32)

    # Build partition data list (cached on idx for repeated searches).
    cache_key = (idx._n_vectors, top_msb_bits)
    if getattr(idx, "_cascade_partition_cache_key", None) != cache_key:
        partition_data = []
        for cell in range(idx.nlist):
            part = idx._partitions[cell]
            if part is None or part.get("indices") is None:
                partition_data.append({})
                continue
            ids = np.asarray(idx._invlists[cell], dtype=np.int64)
            partition_data.append({
                "msb_codes": part["msb_codes"],
                "codes": part["codes"],
                "norms": part["norms"],
                "ids": ids,
            })
        idx._cascade_partition_cache = partition_data
        idx._cascade_partition_cache_key = cache_key
    partition_data = idx._cascade_partition_cache

    sub_centroids = (
        idx.sub_centroids if idx.use_residual_sign
        else np.empty((idx.tq_centroids.size, 2), dtype=np.float32)
    )
    use_sign = idx.use_residual_sign

    if n_threads is None:
        n_threads = min(nq, int(os.environ.get("TQS_THREADS", os.cpu_count() or 1)))

    if n_threads > 1 and nq >= 4:
        chunk = (nq + n_threads - 1) // n_threads

        def _run(start):
            end = min(start + chunk, nq)
            return _cpp_cascade(
                coarse_recon, sub_centroids, partition_data,
                q_rotated[start:end], coarse_scores[start:end],
                top_lists[start:end], use_sign, k, rerank_n,
            )

        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            # Submit all first, THEN wait — otherwise list-comp blocks sequentially.
            futures = [pool.submit(_run, i * chunk)
                       for i in range(n_threads) if i * chunk < nq]
            chunks = [f.result() for f in futures]
        all_indices = np.concatenate([c[1] for c in chunks], axis=0)
    else:
        _, all_indices = _cpp_cascade(
            coarse_recon, sub_centroids, partition_data,
            q_rotated, coarse_scores, top_lists, use_sign, k, rerank_n,
        )

    return all_indices
