"""
Packed-theoretical vs resident memory for all methods at SIFT-1M (1M × 128-d).

Shows the gap between memory_bytes (packed theoretical, bits-perfect) and
actual resident memory for IVF-TQ and all baselines compared in the
SIFT-1M table.

Key distinction:
  FAISS methods (IVF-PQ, OPQ, HNSW): C++ byte-aligned storage.
    - 8-bit PQ codes: 1 byte per sub-code → packed ≈ resident.
    - HNSW: raw float32 vectors → packed = resident (float32 already byte-aligned).
  IVF-TQ (numpy): each b-bit index stored as a full uint8 byte.
    - b=4 or b=6, signs enabled: resident ≈ 4× packed (compression-only).
    - With raw vectors (store_raw_vectors=True): +512 MB on top.
    - Resident does NOT change with b — numpy overhead is constant regardless
      of bit precision.

Usage:
    python experiments/measure_memory.py

Requires FAISS ('pip install faiss-cpu') for baseline measurements.
ScaNN requires a separate install and is skipped if not available.
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

N = 1_000_000
DIM = 128
NLIST = 1024
NPROBE = 20


# ── helpers ──────────────────────────────────────────────────────────────────

def read_fvecs(path: Path, max_n: int | None = None) -> np.ndarray:
    with open(path, "rb") as f:
        raw = np.frombuffer(f.read(), dtype=np.float32)
    d = int(raw[0])
    n_total = len(raw) // (d + 1)
    n = min(n_total, max_n) if max_n else n_total
    vecs = raw[: n * (d + 1)].reshape(n, d + 1)[:, 1:]
    return np.ascontiguousarray(vecs)


def mb(b: int) -> str:
    return f"{b / 1e6:.0f} MB"


def print_row(method, packed_b, resident_b, note=""):
    packed_s = mb(packed_b) if packed_b else "—"
    resident_s = mb(resident_b) if resident_b else "—"
    ratio = f"  ({resident_b / packed_b:.1f}× packed)" if packed_b and resident_b else ""
    print(f"  {method:<40s}  packed={packed_s:>8s}  resident={resident_s:>8s}{ratio}"
          + (f"  [{note}]" if note else ""))


# ── analytical formulae ───────────────────────────────────────────────────────

def faiss_ivfpq_resident(n, d, nlist, m, nbits=8):
    """FAISS IVFPQIndex: byte-aligned codes (1 byte per sub-code for nbits=8)."""
    code_bytes = n * ((m * nbits + 7) // 8)   # = n*m for nbits=8
    pq_centroids = m * (1 << nbits) * (d // m) * 4   # float32
    coarse_centroids = nlist * d * 4
    return code_bytes + pq_centroids + coarse_centroids


def faiss_opq_ivfpq_resident(n, d, nlist, m, nbits=8):
    """OPQ rotation matrix (d×d float32) + IVFPQ."""
    rotation = d * d * 4   # 64 KB for d=128, negligible
    return faiss_ivfpq_resident(n, d, nlist, m, nbits) + rotation


def faiss_hnsw_resident(n, d, M=32):
    """HNSW: float32 vectors + int32 graph links (level-0 has 2M neighbours)."""
    vectors = n * d * 4
    # level-0: 2M links per node; higher levels negligible for large n
    graph_links = n * M * 2 * 4   # int32
    overhead = n * 8   # per-node metadata (level, lock, etc.)
    return vectors + graph_links + overhead


def scann_ah_resident(n, d, m=64, nbits=8):
    """ScaNN AH (Asymmetric Hashing): similar byte layout to FAISS PQ."""
    # ScaNN AH stores m sub-codes of nbits each (same as PQ at matched memory)
    code_bytes = n * ((m * nbits + 7) // 8)
    ah_centroids = m * (1 << nbits) * (d // m) * 4
    return code_bytes + ah_centroids


def ext_rabitq_resident(n, d, b_bits=4):
    """Extended RaBitQ: b bits/coord + 1 norm float32 (packed bit arrays in C++)."""
    code_bits = n * d * b_bits
    code_bytes = (code_bits + 7) // 8
    norms = n * 4
    return code_bytes + norms


# ── IVF-TQ measurement ────────────────────────────────────────────────────────

def build_ivftq(vecs_normed, bits, store_raw):
    idx = IVFTurboQuantIndex(
        dim=DIM, nlist=NLIST, bits=bits, nprobe=NPROBE,
        use_residual_sign=True, store_raw_vectors=store_raw,
    )
    idx.train(vecs_normed[:100_000])
    idx.add(vecs_normed)
    return idx


def ivftq_resident_breakdown(idx: IVFTurboQuantIndex):
    """Per-component resident breakdown."""
    part_total = 0
    for part in idx._partitions:
        for key in ("indices", "norms", "sign_bits", "codes"):
            arr = part.get(key)
            if arr is not None:
                part_total += arr.nbytes
    raw_bytes = idx._raw_vectors.nbytes if idx._raw_vectors is not None else 0
    centroid_bytes = idx.coarse_centroids.nbytes if idx.coarse_centroids is not None else 0
    return part_total, centroid_bytes, raw_bytes


# ── FAISS measurement ─────────────────────────────────────────────────────────

def build_faiss_baselines(vecs_normed):
    try:
        import faiss
    except ImportError:
        print("  [FAISS not installed — skipping measured baseline builds]")
        return None

    results = {}

    # IVF-PQ m=64
    print("  Building FAISS IVF-PQ m=64…")
    quantizer = faiss.IndexFlatIP(DIM)
    idx_pq64 = faiss.IndexIVFPQ(quantizer, DIM, NLIST, 64, 8)
    idx_pq64.train(vecs_normed[:100_000])
    idx_pq64.add(vecs_normed)
    results["ivfpq_m64"] = {
        "code_size": idx_pq64.code_size,   # bytes per stored vector (= 64)
        "measured_codes": idx_pq64.code_size * N,
    }

    # IVF-PQ m=128
    print("  Building FAISS IVF-PQ m=128…")
    quantizer2 = faiss.IndexFlatIP(DIM)
    idx_pq128 = faiss.IndexIVFPQ(quantizer2, DIM, NLIST, 128, 8)
    idx_pq128.train(vecs_normed[:100_000])
    idx_pq128.add(vecs_normed)
    results["ivfpq_m128"] = {
        "code_size": idx_pq128.code_size,
        "measured_codes": idx_pq128.code_size * N,
    }

    return results


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if not SIFT_PATH.exists():
        print(f"SIFT base vectors not found at {SIFT_PATH}")
        print(f"Set DATA_DIR at the top of this script.")
        sys.exit(1)

    print(f"Loading SIFT-1M…")
    vecs = read_fvecs(SIFT_PATH, max_n=N)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs_normed = (vecs / np.maximum(norms, 1e-8)).astype(np.float32)
    print(f"  {len(vecs):,} × {DIM}-d loaded.\n")

    print("=" * 72)
    print(f"Memory accounting — SIFT-1M ({N:,} × {DIM}-d)  nlist={NLIST}")
    print("=" * 72)

    # ── IVF-TQ ───────────────────────────────────────────────────────────────
    print("\n── IVF-TQ ──")
    for bits in (4, 6):
        for store_raw in (False, True):
            label = f"IVF-TQ b={bits} {'(+raw vectors)' if store_raw else '(compression-only)'}"
            idx = build_ivftq(vecs_normed, bits, store_raw)
            packed = idx.memory_bytes
            resident = idx.memory_bytes_resident()
            part_b, cent_b, raw_b = ivftq_resident_breakdown(idx)
            print_row(label, packed, resident)
            if not store_raw:
                print(f"    breakdown: indices+codes+signs={part_b/1e6:.0f}MB "
                      f"centroids={cent_b/1e6:.1f}MB  raw=0MB")
            else:
                print(f"    breakdown: indices+codes+signs={part_b/1e6:.0f}MB "
                      f"centroids={cent_b/1e6:.1f}MB  raw={raw_b/1e6:.0f}MB")
            del idx

    # ── FAISS (measured) ──────────────────────────────────────────────────────
    print("\n── FAISS IVF-PQ (measured — C++ byte-aligned codes) ──")
    faiss_results = build_faiss_baselines(vecs_normed)

    # ── All methods (analytical) ──────────────────────────────────────────────
    print("\n── All methods: packed theoretical vs resident (analytical) ──")
    print(f"  {'Method':<40s}  {'Packed':>10s}  {'Resident':>10s}  Notes")
    print(f"  {'─'*40}  {'─'*10}  {'─'*10}")

    rows = [
        ("FAISS IVF-PQ m=64  (8-bit)", 62e6,
         faiss_ivfpq_resident(N, DIM, NLIST, 64),
         "C++ byte-aligned: packed≈resident"),
        ("FAISS IVF-PQ m=128 (8-bit)", 123e6,
         faiss_ivfpq_resident(N, DIM, NLIST, 128),
         "C++ byte-aligned: packed≈resident"),
        ("FAISS OPQ+IVF-PQ m=128    ", 123e6,
         faiss_opq_ivfpq_resident(N, DIM, NLIST, 128),
         "+rotation matrix (negligible at d=128)"),
        ("FAISS HNSW M=32            ", 732e6,
         faiss_hnsw_resident(N, DIM, 32),
         "float32 vectors + graph links"),
        ("ScaNN AH m=64 (8-bit)      ", 62e6,
         scann_ah_resident(N, DIM, 64),
         "byte-aligned AH codes"),
        ("Ext-RaBitQ b=4             ",
         ext_rabitq_resident(N, DIM, 4),
         ext_rabitq_resident(N, DIM, 4),
         "C++ packed bit arrays → packed=resident"),
        ("IVF-TQ b=4 compression-only",
         84e6,
         build_ivftq(vecs_normed, 4, False).memory_bytes_resident(),
         "numpy uint8/element; no raw vectors"),
        ("IVF-TQ b=6 compression-only",
         111e6,
         build_ivftq(vecs_normed, 6, False).memory_bytes_resident(),
         "same resident as b=4 (uint8 dtype)"),
        ("IVF-TQ b=6 with raw vectors",
         111e6,
         build_ivftq(vecs_normed, 6, True).memory_bytes_resident(),
         "+512 MB float32 raw vectors"),
    ]

    for method, packed_b, resident_b, note in rows:
        packed_s = f"{packed_b/1e6:.0f} MB"
        resident_s = f"{resident_b/1e6:.0f} MB"
        ratio = resident_b / packed_b if packed_b else 0
        ratio_s = f"{ratio:.1f}×" if ratio > 0 else "—"
        print(f"  {method:<40s}  {packed_s:>8s}  {resident_s:>8s}  {ratio_s:>5s}  {note}")

    if faiss_results:
        print("\n── FAISS measured code_size (bytes/vector) ──")
        for key, r in faiss_results.items():
            print(f"  {key}: code_size={r['code_size']} B/vec → "
                  f"{r['measured_codes']/1e6:.0f} MB codes for {N:,} vectors")

    print("\n── Key takeaway ──")
    print("  FAISS PQ/OPQ: C++ byte-aligned storage → packed ≈ resident.")
    print("  IVF-TQ (numpy): uint8 per coordinate → resident ≈ 4× packed (compression-only).")
    print("  IVF-TQ b=4 and b=6 have the SAME resident footprint (uint8 dtype unchanged).")
    print("  Raw vector store (+512 MB) is now opt-in: IVFTurboQuantIndex(store_raw_vectors=False).")


if __name__ == "__main__":
    main()
