"""
Measure theoretical vs resident memory for IVFTurboQuantIndex on SIFT-1M.

Shows the gap between memory_bytes (packed theoretical footprint) and
memory_bytes_resident() (actual numpy allocations).

Usage:
    python experiments/measure_memory.py

Expected output (1M × 128-d, b=4, nlist=1024, signs enabled):
    Packed theoretical:   ~84 MB
    Resident (no raw):   ~388 MB  (indices + signs + norms)
    Resident (with raw): ~900 MB  (+ 512 MB raw_vectors)
    Float32 uncompressed: ~512 MB
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from turboquant_search import IVFTurboQuantIndex

DATA_DIR = Path.home() / "data"
SIFT_PATH = DATA_DIR / "sift" / "sift_base.fvecs"


def read_fvecs(path: Path, max_n: int | None = None) -> np.ndarray:
    with open(path, "rb") as f:
        raw = np.frombuffer(f.read(), dtype=np.float32)
    d = int(raw[0])
    n_total = len(raw) // (d + 1)
    n = min(n_total, max_n) if max_n else n_total
    vecs = raw[: n * (d + 1)].reshape(n, d + 1)[:, 1:]
    return np.ascontiguousarray(vecs)


def report(label: str, idx: IVFTurboQuantIndex):
    n = idx._n_vectors
    dim = idx.dim
    packed = idx.memory_bytes
    resident = idx.memory_bytes_resident()
    uncompressed = n * dim * 4

    print(f"\n{'─'*60}")
    print(f"{label}  (n={n:,}, dim={dim}, b={idx.bits})")
    print(f"{'─'*60}")
    print(f"  Packed theoretical:    {packed / 1e6:7.1f} MB")
    print(f"  Resident (numpy):      {resident / 1e6:7.1f} MB  "
          f"({resident / packed:.1f}× packed)")
    print(f"  Float32 uncompressed:  {uncompressed / 1e6:7.1f} MB")

    # Break down resident by component
    raw_bytes = idx._raw_vectors.nbytes if idx._raw_vectors is not None else 0
    index_bytes = resident - raw_bytes
    print(f"\n  Resident breakdown:")
    print(f"    Index arrays (indices/signs/norms/codes): "
          f"{index_bytes / 1e6:7.1f} MB")
    if idx._raw_vectors is not None:
        print(f"    Raw vectors (always stored):            "
              f"{raw_bytes / 1e6:7.1f} MB  ← dominant term")
    print(f"    Total:                                  "
          f"{resident / 1e6:7.1f} MB")


def main():
    if not SIFT_PATH.exists():
        print(f"SIFT base vectors not found at {SIFT_PATH}")
        print("Set DATA_DIR at the top of this script to the directory")
        print("containing sift/sift_base.fvecs.")
        sys.exit(1)

    print(f"Loading SIFT-1M from {SIFT_PATH}…")
    vecs = read_fvecs(SIFT_PATH, max_n=1_000_000)
    print(f"  Loaded {len(vecs):,} × {vecs.shape[1]}-d")

    # Normalize (inner-product mode, same as benchmark)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs_normed = vecs / np.maximum(norms, 1e-8)

    for bits in (4, 6):
        idx = IVFTurboQuantIndex(
            dim=vecs.shape[1],
            nlist=1024,
            bits=bits,
            nprobe=20,
            use_residual_sign=True,
        )
        print(f"\nTraining IVFTurboQuantIndex (b={bits})…")
        idx.train(vecs_normed[:100_000])
        print(f"Adding 1M vectors…")
        idx.add(vecs_normed)
        report(f"IVFTurboQuantIndex b={bits}", idx)

    print("\nDone.")


if __name__ == "__main__":
    main()
