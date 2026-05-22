"""
Dataset loaders for vector search benchmarks.

Each loader returns (db_vectors, query_vectors, dataset_name) or None on failure.
All vectors are normalized to unit norm for inner product search.
"""

import numpy as np
from typing import Optional, Tuple


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """Normalize to unit norm."""
    vectors = vectors.astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-8)


def load_synthetic(
    n_vectors: int = 10000,
    n_queries: int = 200,
    dim: int = 128,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Generate synthetic clustered data. Always works, no download."""
    rng = np.random.RandomState(seed)

    n_clusters = max(10, n_vectors // 500)
    cluster_centers = rng.randn(n_clusters, dim).astype(np.float32)
    cluster_centers = _normalize(cluster_centers)

    vectors = []
    for i in range(n_vectors):
        c = i % n_clusters
        noise = rng.randn(dim).astype(np.float32) * 0.3
        v = cluster_centers[c] + noise
        vectors.append(v)
    vectors = _normalize(np.array(vectors, dtype=np.float32))

    queries = []
    for i in range(n_queries):
        if i < n_queries // 2:
            idx = rng.randint(n_vectors)
            noise = rng.randn(dim).astype(np.float32) * 0.1
            q = vectors[idx] + noise
        else:
            q = rng.randn(dim).astype(np.float32)
        queries.append(q)
    queries = _normalize(np.array(queries, dtype=np.float32))

    return vectors, queries, f"Synthetic clustered ({n_vectors:,}, dim={dim})"


def load_sift128(
    n_vectors: int = 10000,
    n_queries: int = 200,
    progress_fn=None,
) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
    """
    Load SIFT-128 vectors from HuggingFace Hub.

    Source: open-vdb/sift-128-euclidean (standard ANN benchmark).
    Returns None if download fails.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Warning: 'datasets' package not installed. pip install datasets")
        return None

    try:
        if progress_fn:
            progress_fn("Loading SIFT-128 (downloads ~100MB on first run)...")

        train_ds = load_dataset(
            "open-vdb/sift-128-euclidean", "train",
            split=f"train[:{n_vectors}]",
        )
        test_ds = load_dataset(
            "open-vdb/sift-128-euclidean", "test",
            split=f"test[:{n_queries}]",
        )

        if progress_fn:
            progress_fn("Processing vectors...")

        db_vectors = np.array(train_ds["emb"], dtype=np.float32)
        query_vectors = np.array(test_ds["emb"], dtype=np.float32)

        db_vectors = _normalize(db_vectors)
        query_vectors = _normalize(query_vectors)

        actual_n = db_vectors.shape[0]
        actual_nq = query_vectors.shape[0]

        return (
            db_vectors, query_vectors,
            f"SIFT-128 ({actual_n:,} db, {actual_nq} queries)"
        )

    except Exception as e:
        print(f"Warning: Could not load SIFT-128: {e}")
        return None


def load_glove100(
    n_vectors: int = 10000,
    n_queries: int = 200,
    progress_fn=None,
) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
    """
    Load GloVe-100 word embeddings from HuggingFace Hub.

    Source: open-vdb/glove-100-angular.
    Queries are sampled from the test split.
    Returns None if download fails.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Warning: 'datasets' package not installed. pip install datasets")
        return None

    try:
        if progress_fn:
            progress_fn("Loading GloVe-100 (downloads ~150MB on first run)...")

        train_ds = load_dataset(
            "open-vdb/glove-100-angular", "train",
            split=f"train[:{n_vectors}]",
        )
        test_ds = load_dataset(
            "open-vdb/glove-100-angular", "test",
            split=f"test[:{n_queries}]",
        )

        if progress_fn:
            progress_fn("Processing vectors...")

        db_vectors = np.array(train_ds["emb"], dtype=np.float32)
        query_vectors = np.array(test_ds["emb"], dtype=np.float32)

        db_vectors = _normalize(db_vectors)
        query_vectors = _normalize(query_vectors)

        actual_n = db_vectors.shape[0]
        actual_nq = query_vectors.shape[0]

        return (
            db_vectors, query_vectors,
            f"GloVe-100 ({actual_n:,} db, {actual_nq} queries)"
        )

    except Exception as e:
        print(f"Warning: Could not load GloVe-100: {e}")
        return None


def _read_fvecs(path: str) -> np.ndarray:
    """Read .fvecs format (standard ANN benchmark format)."""
    with open(path, "rb") as f:
        data = f.read()
    offset = 0
    vectors = []
    while offset < len(data):
        dim = int.from_bytes(data[offset:offset + 4], byteorder="little")
        offset += 4
        vec = np.frombuffer(data[offset:offset + dim * 4], dtype=np.float32)
        vectors.append(vec.copy())
        offset += dim * 4
    return np.array(vectors, dtype=np.float32)


def _read_ivecs(path: str) -> np.ndarray:
    """Read .ivecs format (standard ANN benchmark format)."""
    with open(path, "rb") as f:
        data = f.read()
    offset = 0
    vectors = []
    while offset < len(data):
        dim = int.from_bytes(data[offset:offset + 4], byteorder="little")
        offset += 4
        vec = np.frombuffer(data[offset:offset + dim * 4], dtype=np.int32)
        vectors.append(vec.copy())
        offset += dim * 4
    return np.array(vectors, dtype=np.int32)


def load_sift1m(
    n_vectors: int = 1000000,
    n_queries: int = 200,
    progress_fn=None,
) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
    """
    Load SIFT-1M dataset (1M 128-dim vectors from the standard ANN benchmark).

    Downloads from the texmex corpus mirror. ~170MB compressed.
    Returns None if download fails.
    """
    import os
    import tarfile
    from pathlib import Path

    cache_dir = Path.home() / ".cache" / "turboquant" / "sift1m"
    base_path = cache_dir / "sift" / "sift_base.fvecs"
    query_path = cache_dir / "sift" / "sift_query.fvecs"
    tar_path = cache_dir / "sift.tar.gz"

    # Check if already extracted
    if base_path.exists() and query_path.exists():
        if progress_fn:
            progress_fn("Loading SIFT-1M from cache...")
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        url = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
        # Try HTTP mirror first (more reliable)
        http_url = "http://corpus-texmex.irisa.fr/sift.tar.gz"

        if progress_fn:
            progress_fn("Downloading SIFT-1M (~170MB)...")

        # Download
        downloaded = False
        for dl_url in [http_url, url]:
            try:
                import urllib.request
                urllib.request.urlretrieve(dl_url, str(tar_path))
                downloaded = True
                break
            except Exception:
                continue

        if not downloaded:
            # Try with requests + tqdm
            try:
                import requests
                from tqdm import tqdm
                r = requests.get(http_url, stream=True, timeout=60)
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                with open(tar_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="SIFT-1M") as pbar:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
                downloaded = True
            except Exception as e:
                print(f"Download failed: {e}")
                return None

        if not downloaded:
            return None

        # Extract
        if progress_fn:
            progress_fn("Extracting SIFT-1M...")
        try:
            with tarfile.open(str(tar_path), "r:gz") as tar:
                tar.extractall(path=str(cache_dir))
        except Exception as e:
            print(f"Extraction failed: {e}")
            return None
        finally:
            if tar_path.exists():
                tar_path.unlink()

    if not base_path.exists():
        print("SIFT-1M: base vectors not found after extraction")
        return None

    try:
        if progress_fn:
            progress_fn("Reading SIFT-1M vectors...")

        db_vectors = _read_fvecs(str(base_path))
        query_vectors = _read_fvecs(str(query_path))

        # Subset if needed
        if n_vectors < db_vectors.shape[0]:
            db_vectors = db_vectors[:n_vectors]
        if n_queries < query_vectors.shape[0]:
            query_vectors = query_vectors[:n_queries]

        db_vectors = _normalize(db_vectors)
        query_vectors = _normalize(query_vectors)

        return (
            db_vectors, query_vectors,
            f"SIFT-1M ({db_vectors.shape[0]:,} db, {query_vectors.shape[0]} queries, dim=128)"
        )
    except Exception as e:
        print(f"Failed to read SIFT-1M: {e}")
        return None


def load_deep1m(
    n_vectors: int = 1000000,
    n_queries: int = 200,
    progress_fn=None,
) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
    """
    Load Deep-1M dataset (96-dim deep learning features, standard ANN benchmark).

    Source: open-vdb/deep-image-96-angular on HuggingFace Hub.
    Returns None if download fails.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Warning: 'datasets' package not installed. pip install datasets")
        return None

    try:
        if progress_fn:
            progress_fn("Loading Deep-1M (downloads on first run)...")

        train_ds = load_dataset(
            "open-vdb/deep-image-96-angular", "train",
            split=f"train[:{n_vectors}]",
        )
        test_ds = load_dataset(
            "open-vdb/deep-image-96-angular", "test",
            split=f"test[:{n_queries}]",
        )

        if progress_fn:
            progress_fn("Processing vectors...")

        db_vectors = np.array(train_ds["emb"], dtype=np.float32)
        query_vectors = np.array(test_ds["emb"], dtype=np.float32)

        db_vectors = _normalize(db_vectors)
        query_vectors = _normalize(query_vectors)

        actual_n = db_vectors.shape[0]
        actual_nq = query_vectors.shape[0]

        return (
            db_vectors, query_vectors,
            f"Deep-1M ({actual_n:,} db, {actual_nq} queries, dim=96)"
        )
    except Exception as e:
        print(f"Warning: Could not load Deep-1M: {e}")
        return None


# Registry of available datasets
DATASET_LOADERS = {
    "synthetic": load_synthetic,
    "sift-128": load_sift128,
    "glove-100": load_glove100,
    "sift-1m": load_sift1m,
    "deep-1m": load_deep1m,
}

DATASET_LABELS = {
    "synthetic": "Synthetic clustered (10K, dim=128)",
    "sift-128": "SIFT-128 (requires: pip install datasets)",
    "glove-100": "GloVe-100 (requires: pip install datasets)",
    "sift-1m": "SIFT-1M (downloads ~170MB, standard ANN benchmark)",
    "deep-1m": "Deep-1M (96-dim, standard ANN benchmark)",
}
