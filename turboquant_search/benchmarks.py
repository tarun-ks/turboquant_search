"""
Benchmark utilities for comparing TurboQuant vs FAISS baselines.

Uses real FAISS implementations for PQ and IVF-PQ baselines. Falls back
to NumPy implementations if faiss-cpu is not installed.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple

from .core import TurboQuantSearchIndex, IVFTurboQuantIndex, FlatSearchIndex, ProductQuantizationIndex
from .faiss_baselines import (
    FAISS_AVAILABLE,
    FAISSFlatIndex,
    FAISSPQIndex,
    FAISSIVFPQIndex,
)
from .datasets import (
    load_synthetic,
    load_sift128,
    load_glove100,
    DATASET_LOADERS,
)
# load_sift1m is available via DATASET_LOADERS["sift-1m"]


def compute_recall(ground_truth: np.ndarray, predictions: np.ndarray, k: int) -> float:
    """
    Compute Recall@k: fraction of true top-k neighbors found in predicted top-k.
    """
    nq = ground_truth.shape[0]
    gt_k = ground_truth[:, :k]
    pred_k = predictions[:, :k]

    recall_sum = 0.0
    for i in range(nq):
        gt_set = set(gt_k[i].tolist())
        pred_set = set(pred_k[i].tolist())
        recall_sum += len(gt_set & pred_set) / len(gt_set)

    return recall_sum / nq


# Keep for backward compatibility
def generate_synthetic_data(
    n_vectors: int = 10000,
    n_queries: int = 100,
    dim: int = 128,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic vectors. Delegates to datasets.load_synthetic."""
    vectors, queries, _ = load_synthetic(n_vectors, n_queries, dim, seed)
    return vectors, queries


def run_benchmark(
    dataset_name: str = "synthetic",
    n_vectors: int = 10000,
    n_queries: int = 200,
    dim: int = 128,
    k_values: List[int] = [1, 5, 10, 50],
    bit_widths: List[int] = [2, 3, 4],
    seed: int = 42,
    vectors: Optional[np.ndarray] = None,
    queries: Optional[np.ndarray] = None,
    progress_callback=None,
) -> Dict:
    """
    Run full benchmark: TurboQuant vs FAISS baselines.

    Uses real FAISS for Flat (ground truth), PQ, and IVF-PQ baselines.
    Falls back to NumPy implementations if faiss-cpu is unavailable.

    Parameters
    ----------
    dataset_name : str
        One of "synthetic", "sift-128", "glove-100", or provide vectors/queries directly.
    n_vectors, n_queries, dim : int
        Used for synthetic data generation. Ignored if vectors is provided.
    k_values : list of int
        Values of k for Recall@k.
    bit_widths : list of int
        TurboQuant bit widths to test.
    seed : int
        Random seed.
    vectors, queries : np.ndarray, optional
        Pre-loaded data. Overrides dataset_name.
    progress_callback : callable, optional
        Called with (step, total_steps, message).
    """
    # Count steps: ground_truth + PQ + IVF-PQ + PQ-matched + len(bit_widths)*2 + IVF-TQ
    total_steps = 5 + len(bit_widths) * 2
    step = 0

    def update(msg):
        nonlocal step
        step += 1
        if progress_callback:
            progress_callback(step, total_steps, msg)

    # ── Load data ──
    dataset_label = dataset_name
    if vectors is not None:
        vectors = vectors.astype(np.float32)
        if queries is None:
            rng = np.random.RandomState(seed)
            actual_nq = min(n_queries, vectors.shape[0] // 5)
            actual_nq = max(actual_nq, 10)
            qi = rng.choice(vectors.shape[0], size=actual_nq, replace=False)
            queries = vectors[qi].copy()
            queries += rng.randn(*queries.shape).astype(np.float32) * 0.05
            norms = np.linalg.norm(queries, axis=1, keepdims=True)
            queries = queries / np.maximum(norms, 1e-8)
        else:
            queries = queries.astype(np.float32)
        dataset_label = f"Custom ({vectors.shape[0]:,}, dim={vectors.shape[1]})"
    elif dataset_name == "synthetic":
        vectors, queries, dataset_label = load_synthetic(n_vectors, n_queries, dim, seed)
    elif dataset_name in DATASET_LOADERS:
        loader = DATASET_LOADERS[dataset_name]
        progress_fn = None
        if progress_callback:
            progress_fn = lambda msg: progress_callback(0, total_steps, msg)
        result = loader(n_vectors=n_vectors, n_queries=n_queries, progress_fn=progress_fn)
        if result is None:
            # Fall back to synthetic
            vectors, queries, dataset_label = load_synthetic(n_vectors, n_queries, dim, seed)
            dataset_label += " (fallback — download failed)"
        else:
            vectors, queries, dataset_label = result
    else:
        vectors, queries, dataset_label = load_synthetic(n_vectors, n_queries, dim, seed)

    n_vectors = vectors.shape[0]
    n_queries_actual = queries.shape[0]
    dim = vectors.shape[1]
    k_values = [k for k in k_values if k <= n_vectors]
    if not k_values:
        k_values = [1]
    max_k = max(k_values)

    results = {
        "config": {
            "n_vectors": n_vectors,
            "n_queries": n_queries_actual,
            "dim": dim,
            "k_values": k_values,
            "bit_widths": bit_widths,
            "dataset": dataset_label,
        },
        "methods": {},
    }

    # ── 1. Ground truth (FAISS Flat or NumPy Flat) ──
    update("Building ground truth index...")
    if FAISS_AVAILABLE:
        gt_index = FAISSFlatIndex(dim)
    else:
        gt_index = FlatSearchIndex(dim)
    gt_index.add(vectors)

    t0 = time.time()
    _, idx_gt = gt_index.search(queries, k=max_k)
    gt_search_time = time.time() - t0

    gt_indices = {k: idx_gt[:, :k] for k in k_values}

    gt_label = "Flat (FAISS)" if FAISS_AVAILABLE else "Flat (NumPy)"
    results["methods"][gt_label] = {
        "memory_mb": gt_index.memory_bytes / (1024 * 1024),
        "compression_ratio": 1.0,
        "build_time": gt_index.build_time,
        "search_time": gt_search_time,
        "recall": {k: 1.0 for k in k_values},
        "bits_per_dim": 32,
    }

    # ── 2. PQ baseline (FAISS or NumPy fallback) ──
    update("Building PQ index...")

    m_pq = 8
    while dim % m_pq != 0 and m_pq > 1:
        m_pq -= 1

    if FAISS_AVAILABLE:
        pq_index = FAISSPQIndex(dim, m=m_pq, nbits=8)
        pq_label = f"PQ (FAISS, m={m_pq})"
    else:
        pq_index = ProductQuantizationIndex(dim, n_subspaces=m_pq, seed=seed)
        pq_label = f"PQ (NumPy, m={m_pq})"
    pq_index.add(vectors)

    t0 = time.time()
    _, idx_pq = pq_index.search(queries, k=max_k)
    pq_search_time = time.time() - t0

    pq_recall = {k: compute_recall(gt_indices[k], idx_pq[:, :k], k) for k in k_values}
    uncompressed_bytes = n_vectors * dim * 4

    results["methods"][pq_label] = {
        "memory_mb": pq_index.memory_bytes / (1024 * 1024),
        "compression_ratio": uncompressed_bytes / max(pq_index.memory_bytes, 1),
        "build_time": pq_index.build_time,
        "search_time": pq_search_time,
        "recall": pq_recall,
        "bits_per_dim": 8.0 / max(dim / m_pq, 1),
    }

    # ── 3. IVF-PQ baseline (FAISS only) ──
    if FAISS_AVAILABLE:
        update("Building IVF-PQ index...")
        nlist = max(1, min(100, n_vectors // 39))
        ivfpq_index = FAISSIVFPQIndex(dim, nlist=nlist, m=m_pq, nbits=8, nprobe=10)
        ivfpq_index.add(vectors)

        t0 = time.time()
        _, idx_ivfpq = ivfpq_index.search(queries, k=max_k)
        ivfpq_search_time = time.time() - t0

        ivfpq_recall = {k: compute_recall(gt_indices[k], idx_ivfpq[:, :k], k) for k in k_values}

        results["methods"][f"IVF-PQ (FAISS, m={m_pq})"] = {
            "memory_mb": ivfpq_index.memory_bytes / (1024 * 1024),
            "compression_ratio": uncompressed_bytes / max(ivfpq_index.memory_bytes, 1),
            "build_time": ivfpq_index.build_time,
            "search_time": ivfpq_search_time,
            "recall": ivfpq_recall,
            "bits_per_dim": 8.0 / max(dim / m_pq, 1),
        }
    else:
        update("Skipping IVF-PQ (faiss-cpu not installed)...")

    # ── 3b. PQ at matched compression (~8x, same ballpark as TQ) ──
    # Default PQ (m=8) compresses ~24x — far more aggressive than TQ's 6-10x.
    # For a fair comparison, we also run PQ with m=dim//2 to match TQ's
    # compression range. Each sub-vector is 2 dims with 256 centroids.
    if FAISS_AVAILABLE:
        update("Building PQ (matched compression)...")
        m_matched = dim // 2
        while dim % m_matched != 0 and m_matched > 1:
            m_matched -= 1

        if m_matched > m_pq:
            pq_matched_index = FAISSPQIndex(dim, m=m_matched, nbits=8)
            pq_matched_index.add(vectors)

            t0 = time.time()
            _, idx_pq_matched = pq_matched_index.search(queries, k=max_k)
            pq_matched_time = time.time() - t0

            pq_matched_recall = {k: compute_recall(gt_indices[k], idx_pq_matched[:, :k], k) for k in k_values}

            results["methods"][f"PQ (FAISS, m={m_matched})"] = {
                "memory_mb": pq_matched_index.memory_bytes / (1024 * 1024),
                "compression_ratio": uncompressed_bytes / max(pq_matched_index.memory_bytes, 1),
                "build_time": pq_matched_index.build_time,
                "search_time": pq_matched_time,
                "recall": pq_matched_recall,
                "bits_per_dim": 8.0 / max(dim / m_matched, 1),
            }
    else:
        update("Skipping PQ-matched (faiss-cpu not installed)...")

    # ── 4. TurboQuant variants (flat brute-force) ──
    for bits in bit_widths:
        update(f"Building TurboQuant {bits}-bit...")
        tq = TurboQuantSearchIndex(dim, bits=bits, use_residual_sign=True, seed=seed)
        tq.add(vectors)

        update(f"Searching TurboQuant {bits}-bit...")
        t0 = time.time()
        _, idx_tq = tq.search(queries, k=max_k)
        tq_search_time = time.time() - t0

        tq_recall = {k: compute_recall(gt_indices[k], idx_tq[:, :k], k) for k in k_values}

        label = f"TurboQuant {bits}-bit"
        if tq.use_residual_sign:
            label += " + sign-refine"
        results["methods"][label] = {
            "memory_mb": tq.memory_bytes / (1024 * 1024),
            "compression_ratio": tq.compression_ratio,
            "build_time": tq.build_time,
            "search_time": tq_search_time,
            "recall": tq_recall,
            "bits_per_dim": bits + (1.0 if tq.use_residual_sign else 0.0),
        }

    # ── 5. IVF-TQ (sub-linear search with TQ compression) ──
    # Use same nlist/nprobe as IVF-PQ for fair comparison
    best_bits = max(bit_widths)
    update(f"Building IVF-TQ {best_bits}-bit...")
    ivftq_nlist = max(1, min(100, n_vectors // 39))
    ivftq_nprobe = min(10, ivftq_nlist)
    ivftq = IVFTurboQuantIndex(
        dim, nlist=ivftq_nlist, bits=best_bits, nprobe=ivftq_nprobe,
        use_residual_sign=True, seed=seed,
    )
    ivftq.train(vectors)
    ivftq.add(vectors)

    t0 = time.time()
    _, idx_ivftq = ivftq.search(queries, k=max_k)
    ivftq_search_time = time.time() - t0

    ivftq_recall = {k: compute_recall(gt_indices[k], idx_ivftq[:, :k], k) for k in k_values}

    results["methods"][f"IVF-TQ {best_bits}-bit (nprobe={ivftq_nprobe})"] = {
        "memory_mb": ivftq.memory_bytes / (1024 * 1024),
        "compression_ratio": ivftq.compression_ratio,
        "build_time": ivftq.train_time + ivftq.build_time,
        "search_time": ivftq_search_time,
        "recall": ivftq_recall,
        "bits_per_dim": best_bits + (1.0 if ivftq.use_residual_sign else 0.0),
    }

    return results


def format_results_table(results: Dict) -> str:
    """Format benchmark results as a readable table."""
    lines = []
    config = results["config"]
    lines.append(f"Dataset: {config.get('dataset', 'unknown')}")
    lines.append(f"  {config['n_vectors']:,} vectors, dim={config['dim']}, {config['n_queries']} queries")
    lines.append("")

    k_values = config["k_values"]
    header = f"{'Method':<28} {'Memory':>8} {'Ratio':>7} {'Build':>8} {'Search':>8}"
    for k in k_values:
        header += f" {'R@'+str(k):>7}"
    lines.append(header)
    lines.append("-" * len(header))

    for name, data in results["methods"].items():
        row = f"{name:<28} {data['memory_mb']:>7.1f}M {data['compression_ratio']:>6.1f}x"
        row += f" {data['build_time']:>7.3f}s {data['search_time']:>7.3f}s"
        for k in k_values:
            row += f" {data['recall'][k]:>6.1%}"
        lines.append(row)

    return "\n".join(lines)
