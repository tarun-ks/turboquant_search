"""
Pre-embedded dataset download and caching for TurboQuant Search demos.

Downloads pre-embedded datasets (Wikipedia paragraphs, arxiv abstracts) and
caches them in ~/.cache/turboquant/ for reuse across sessions.
"""

import json
import os
import hashlib
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np


CACHE_DIR = Path.home() / ".cache" / "turboquant"

# Common embedding models and their dimensions (for reference / UI dropdowns)
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": {"dim": 384, "provider": "sentence-transformers"},
    "all-mpnet-base-v2": {"dim": 768, "provider": "sentence-transformers"},
    "bge-small-en-v1.5": {"dim": 384, "provider": "BAAI"},
    "bge-base-en-v1.5": {"dim": 768, "provider": "BAAI"},
    "bge-large-en-v1.5": {"dim": 1024, "provider": "BAAI"},
    "text-embedding-3-small": {"dim": 1536, "provider": "OpenAI"},
    "text-embedding-3-large": {"dim": 3072, "provider": "OpenAI"},
    "text-embedding-ada-002": {"dim": 1536, "provider": "OpenAI"},
    "embed-v4": {"dim": 1024, "provider": "Cohere"},
    "voyage-3": {"dim": 1024, "provider": "Voyage AI"},
    "gemini-embedding-001": {"dim": 3072, "provider": "Google"},
    "nomic-embed-text-v1.5": {"dim": 768, "provider": "Nomic"},
}

# Dataset registry: name -> metadata
# Each dataset has: url_vectors, url_metadata, dim, count, model, description
DATASETS: Dict[str, dict] = {
    "wikipedia-384": {
        "description": "100K Wikipedia paragraphs — all-MiniLM-L6-v2 (384-dim)",
        "dim": 384,
        "count": 100_000,
        "model": "all-MiniLM-L6-v2",
        "filename_vectors": "wikipedia_100k_minilm.npy",
        "filename_metadata": "wikipedia_100k_minilm_meta.json",
        # TODO: Replace with actual HuggingFace Hub or public URL once uploaded
        "url_vectors": None,
        "url_metadata": None,
    },
    "wikipedia-768": {
        "description": "100K Wikipedia paragraphs — bge-base-en-v1.5 (768-dim)",
        "dim": 768,
        "count": 100_000,
        "model": "bge-base-en-v1.5",
        "filename_vectors": "wikipedia_100k_bge768.npy",
        "filename_metadata": "wikipedia_100k_bge768_meta.json",
        "url_vectors": None,
        "url_metadata": None,
    },
    "wikipedia-1536": {
        "description": "100K Wikipedia paragraphs — OpenAI-dim (1536-dim)",
        "dim": 1536,
        "count": 100_000,
        "model": "text-embedding-3-small (simulated)",
        "filename_vectors": "wikipedia_100k_oai1536.npy",
        "filename_metadata": "wikipedia_100k_oai1536_meta.json",
        "url_vectors": None,
        "url_metadata": None,
    },
    "arxiv-384": {
        "description": "100K arxiv abstracts — all-MiniLM-L6-v2 (384-dim)",
        "dim": 384,
        "count": 100_000,
        "model": "all-MiniLM-L6-v2",
        "filename_vectors": "arxiv_100k_minilm.npy",
        "filename_metadata": "arxiv_100k_minilm_meta.json",
        "url_vectors": None,
        "url_metadata": None,
    },
    "arxiv-1024": {
        "description": "100K arxiv abstracts — bge-large (1024-dim)",
        "dim": 1024,
        "count": 100_000,
        "model": "bge-large-en-v1.5 (simulated)",
        "filename_vectors": "arxiv_100k_bge1024.npy",
        "filename_metadata": "arxiv_100k_bge1024_meta.json",
        "url_vectors": None,
        "url_metadata": None,
    },
    # Backward-compatible aliases
    "wikipedia": {
        "description": "100K Wikipedia paragraphs — all-MiniLM-L6-v2 (384-dim)",
        "dim": 384,
        "count": 100_000,
        "model": "all-MiniLM-L6-v2",
        "filename_vectors": "wikipedia_100k_minilm.npy",
        "filename_metadata": "wikipedia_100k_minilm_meta.json",
        "url_vectors": None,
        "url_metadata": None,
    },
    "arxiv": {
        "description": "100K arxiv abstracts — all-MiniLM-L6-v2 (384-dim)",
        "dim": 384,
        "count": 100_000,
        "model": "all-MiniLM-L6-v2",
        "filename_vectors": "arxiv_100k_minilm.npy",
        "filename_metadata": "arxiv_100k_minilm_meta.json",
        "url_vectors": None,
        "url_metadata": None,
    },
}


def get_cache_dir() -> Path:
    """Return the cache directory, creating it if needed."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def list_datasets() -> Dict[str, str]:
    """Return {name: description} for all available datasets."""
    return {name: info["description"] for name, info in DATASETS.items()}


def _download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download a file with progress bar. Returns True on success."""
    try:
        import requests
        from tqdm import tqdm
    except ImportError:
        # Fallback without progress bar
        import urllib.request
        print(f"Downloading {desc or url}...")
        try:
            urllib.request.urlretrieve(url, str(dest))
            return True
        except Exception as e:
            print(f"Download failed: {e}")
            return False

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        with open(dest, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=desc or dest.name,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        if dest.exists():
            dest.unlink()
        return False


def _generate_synthetic_placeholder(
    dataset_name: str,
    dim: int,
    count: int,
    seed: int = 42,
) -> Tuple[np.ndarray, List[str]]:
    """
    Generate synthetic embeddings as a placeholder when real data isn't available.
    Uses clustered data to simulate realistic embedding distributions.
    """
    rng = np.random.RandomState(seed)
    n_clusters = 200

    # Clustered synthetic data
    centers = rng.randn(n_clusters, dim).astype(np.float32)
    norms = np.linalg.norm(centers, axis=1, keepdims=True)
    centers = centers / np.maximum(norms, 1e-8)

    vectors = []
    for i in range(count):
        c = i % n_clusters
        noise = rng.randn(dim).astype(np.float32) * 0.15
        v = centers[c] + noise
        vectors.append(v)
    vectors = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-8)

    # Generate placeholder text snippets
    if dataset_name == "wikipedia":
        texts = [f"Wikipedia paragraph #{i+1} (placeholder)" for i in range(count)]
    elif dataset_name == "arxiv":
        texts = [f"Arxiv abstract #{i+1} (placeholder)" for i in range(count)]
    else:
        texts = [f"Document #{i+1}" for i in range(count)]

    return vectors, texts


def load_dataset(
    name: str = "wikipedia",
    n_vectors: Optional[int] = None,
    n_queries: int = 200,
    seed: int = 42,
    force_download: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str], dict]:
    """
    Load a pre-embedded dataset, downloading and caching if needed.

    Parameters
    ----------
    name : str
        Dataset name: "wikipedia" or "arxiv".
    n_vectors : int, optional
        Number of vectors to use (None = all).
    n_queries : int
        Number of query vectors to sample.
    seed : int
        Random seed for query sampling.
    force_download : bool
        Re-download even if cached.

    Returns
    -------
    vectors : np.ndarray of shape (n, dim)
    queries : np.ndarray of shape (n_queries, dim)
    texts : list of str
        Source texts corresponding to vectors.
    info : dict
        Dataset metadata (dim, count, model, description).
    """
    if name not in DATASETS:
        available = ", ".join(DATASETS.keys())
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")

    ds = DATASETS[name]
    cache = get_cache_dir()
    vec_path = cache / ds["filename_vectors"]
    meta_path = cache / ds["filename_metadata"]

    vectors = None
    texts = None

    # Try loading from cache
    if vec_path.exists() and meta_path.exists() and not force_download:
        try:
            vectors = np.load(str(vec_path))
            with open(meta_path, "r") as f:
                meta = json.load(f)
            texts = meta.get("texts", [])
            print(f"Loaded {name} from cache ({vectors.shape[0]:,} vectors, dim={vectors.shape[1]})")
        except Exception as e:
            print(f"Cache corrupted, regenerating: {e}")
            vectors = None

    # Try downloading from URL
    if vectors is None and ds["url_vectors"] is not None:
        print(f"Downloading {name} dataset...")
        ok = _download_file(ds["url_vectors"], vec_path, f"{name} vectors")
        if ok and ds["url_metadata"] is not None:
            ok = _download_file(ds["url_metadata"], meta_path, f"{name} metadata")
        if ok:
            try:
                vectors = np.load(str(vec_path))
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                texts = meta.get("texts", [])
            except Exception:
                vectors = None

    # Fall back to synthetic placeholder
    if vectors is None:
        print(f"Generating synthetic {name} placeholder ({ds['count']:,} vectors, dim={ds['dim']})...")
        vectors, texts = _generate_synthetic_placeholder(
            name, ds["dim"], ds["count"], seed=seed
        )
        # Cache for reuse
        np.save(str(vec_path), vectors)
        meta = {
            "texts": texts,
            "model": ds["model"],
            "dim": ds["dim"],
            "count": len(texts),
            "synthetic": True,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f)
        print(f"Cached to {cache}")

    # Subset if requested
    total = vectors.shape[0]
    if n_vectors is not None and n_vectors < total:
        vectors = vectors[:n_vectors]
        texts = texts[:n_vectors]

    # Sample queries from the dataset (perturbed copies)
    rng = np.random.RandomState(seed)
    n_available = vectors.shape[0]
    actual_nq = min(n_queries, n_available // 5)
    actual_nq = max(actual_nq, 10)

    qi = rng.choice(n_available, size=actual_nq, replace=False)
    queries = vectors[qi].copy()
    queries += rng.randn(*queries.shape).astype(np.float32) * 0.05
    norms = np.linalg.norm(queries, axis=1, keepdims=True)
    queries = queries / np.maximum(norms, 1e-8)

    info = {
        "dim": ds["dim"],
        "count": vectors.shape[0],
        "model": ds["model"],
        "description": ds["description"],
        "name": name,
    }

    return vectors, queries, texts, info


def clear_cache(name: Optional[str] = None):
    """Remove cached datasets. If name is None, clear all."""
    cache = get_cache_dir()
    if name is not None:
        if name not in DATASETS:
            return
        ds = DATASETS[name]
        for fn in [ds["filename_vectors"], ds["filename_metadata"]]:
            p = cache / fn
            if p.exists():
                p.unlink()
                print(f"Removed {p}")
    else:
        import shutil
        if cache.exists():
            shutil.rmtree(cache)
            print(f"Cleared cache: {cache}")
