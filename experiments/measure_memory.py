"""
Packed-theoretical vs resident memory for ALL methods at SIFT-1M (1M × 128-d).

Key distinction:
  FAISS (IVF-PQ, OPQ, HNSW): C++ byte-aligned storage.
    8-bit PQ codes: 1 byte per sub-code → packed = resident.
    HNSW: raw float32 vectors → packed = resident.
  IVF-TQ (numpy): each b-bit index stored as full uint8 byte.
    b=4 OR b=6, signs enabled: resident ≈ 4× packed (compression-only).
    b=4 and b=6 have IDENTICAL resident footprint (uint8 dtype is constant).
    With raw vectors (store_raw_vectors=True): +512 MB on top.
  Ext-RaBitQ: C++ bit-packed arrays → packed = resident.

Usage:
    python experiments/measure_memory.py

Requires faiss-cpu for FAISS baseline builds.
ScaNN requires a separate install; skipped if unavailable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from turboquant_search import IVFTurboQuantIndex

DATA_DIR = Path.home() / ".cache" / "turboquant" / "sift1m"
SIFT_PATH = DATA_DIR / "sift" / "sift_base.fvecs"

N = 1_000_000
DIM = 128
NLIST = 1024
NPROBE = 20


# ── helpers ──────────────────────────────────────────────────────────────────

def read_fvecs(path: Path, max_n: int | None = None) -> np.ndarray:
    with open(path, "rb") as f:
        raw_bytes = f.read()
    # Dimension header is int32 little-endian (NOT float32)
    d = int.from_bytes(raw_bytes[:4], byteorder="little")
    record_bytes = (d + 1) * 4   # 4 bytes for dim header + d × 4 bytes of float32
    n_total = len(raw_bytes) // record_bytes
    n = min(n_total, max_n) if max_n else n_total
    # Read entire file as float32; each record is [dim_as_f32, v0, v1, ..., vd-1]
    raw = np.frombuffer(raw_bytes[:n * record_bytes], dtype=np.float32)
    vecs = raw.reshape(n, d + 1)[:, 1:]   # drop the dim column
    return np.ascontiguousarray(vecs)


def mb(b: float) -> str:
    return f"{int(round(b / 1e6))} MB"


# ── IVF-TQ ────────────────────────────────────────────────────────────────────

def build_ivftq(vecs_normed: np.ndarray, bits: int,
                store_raw: bool) -> IVFTurboQuantIndex:
    idx = IVFTurboQuantIndex(
        dim=DIM, nlist=NLIST, bits=bits, nprobe=NPROBE,
        use_residual_sign=True, store_raw_vectors=store_raw,
    )
    idx.train(vecs_normed[:100_000])
    idx.add(vecs_normed)
    return idx


def ivftq_breakdown(idx: IVFTurboQuantIndex) -> dict:
    part_total = 0
    for part in idx._partitions:
        for key in ("indices", "norms", "sign_bits", "codes"):
            arr = part.get(key)
            if arr is not None:
                part_total += arr.nbytes
    raw_b = idx._raw_vectors.nbytes if idx._raw_vectors is not None else 0
    cent_b = idx.coarse_centroids.nbytes if idx.coarse_centroids is not None else 0
    return {"partition_arrays": part_total, "centroids": cent_b, "raw_vectors": raw_b}


# ── FAISS ─────────────────────────────────────────────────────────────────────

def build_faiss(vecs_normed: np.ndarray):
    try:
        import faiss
    except ImportError:
        print("  faiss-cpu not installed — FAISS builds skipped.")
        return None

    results = {}

    # IVF-PQ m=64
    print("  building FAISS IVF-PQ m=64 …")
    q64 = faiss.IndexFlatIP(DIM)
    idx64 = faiss.IndexIVFPQ(q64, DIM, NLIST, 64, 8)
    idx64.train(vecs_normed[:100_000])
    idx64.add(vecs_normed)
    pq64_code = idx64.code_size * idx64.ntotal
    pq64_pqcent = 64 * 256 * (DIM // 64) * 4
    pq64_coarse = NLIST * DIM * 4
    results["ivfpq_m64"] = {
        "code_size_per_vec": idx64.code_size,
        "codes": pq64_code,
        "pq_centroids": pq64_pqcent,
        "coarse_centroids": pq64_coarse,
        "total_resident": pq64_code + pq64_pqcent + pq64_coarse,
    }

    # IVF-PQ m=128
    print("  building FAISS IVF-PQ m=128 …")
    q128 = faiss.IndexFlatIP(DIM)
    idx128 = faiss.IndexIVFPQ(q128, DIM, NLIST, 128, 8)
    idx128.train(vecs_normed[:100_000])
    idx128.add(vecs_normed)
    pq128_code = idx128.code_size * idx128.ntotal
    pq128_pqcent = 128 * 256 * (DIM // 128) * 4
    pq128_coarse = NLIST * DIM * 4
    results["ivfpq_m128"] = {
        "code_size_per_vec": idx128.code_size,
        "codes": pq128_code,
        "pq_centroids": pq128_pqcent,
        "coarse_centroids": pq128_coarse,
        "total_resident": pq128_code + pq128_pqcent + pq128_coarse,
    }

    # HNSW M=32
    print("  building FAISS HNSW M=32 …")
    hnsw = faiss.IndexHNSWFlat(DIM, 32)
    hnsw.add(vecs_normed)
    hnsw_code = DIM * 4 * hnsw.ntotal   # float32 vectors (HNSW stores raw)
    hnsw_graph = N * 32 * 2 * 4                # int32 links (level-0, 2M per node)
    results["hnsw_m32"] = {
        "code_size_per_vec": DIM * 4,   # float32 vectors (no sa_code_size on HNSW)
        "vectors": hnsw_code,
        "graph_links_approx": hnsw_graph,
        "total_resident": hnsw_code + hnsw_graph,
    }

    return results


# ── analytical estimates for methods not built ────────────────────────────────

def ext_rabitq_analytical(n, d, b_bits):
    """C++ packed bit arrays: packed = resident."""
    code_bytes = (n * d * b_bits + 7) // 8
    norms = n * 4
    return code_bytes + norms


def scann_ah_analytical(n, d, m, nbits=8):
    """ScaNN AH: same layout as FAISS PQ at matched m."""
    code_bytes = n * ((m * nbits + 7) // 8)
    ah_cents = m * (1 << nbits) * (d // m) * 4
    return code_bytes + ah_cents


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if not SIFT_PATH.exists():
        print(f"SIFT base vectors not found at {SIFT_PATH}")
        print(f"Set DATA_DIR to the directory containing sift/sift_base.fvecs.")
        sys.exit(1)

    print(f"Loading SIFT-1M …")
    vecs = read_fvecs(SIFT_PATH, max_n=N)
    v_norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs_normed = (vecs / np.maximum(v_norms, 1e-8)).astype(np.float32)
    print(f"  {len(vecs):,} × {DIM}-d loaded.\n")

    # ── build IVF-TQ (4 configs) ──────────────────────────────────────────
    print("Building IVF-TQ indexes …")
    tq_results = {}
    for bits in (4, 6):
        for store_raw in (False, True):
            key = f"b{bits}_{'raw' if store_raw else 'noraw'}"
            print(f"  IVF-TQ b={bits} store_raw={store_raw} …")
            idx = build_ivftq(vecs_normed, bits, store_raw)
            tq_results[key] = {
                "packed": idx.memory_bytes,
                "resident": idx.memory_bytes_resident(),
                "breakdown": ivftq_breakdown(idx),
            }
            del idx   # free memory before next build

    # ── build FAISS baselines ─────────────────────────────────────────────
    print("\nBuilding FAISS baselines …")
    faiss_results = build_faiss(vecs_normed)
    del vecs_normed   # free

    # ── print results ─────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"Memory accounting — SIFT-1M ({N:,} × {DIM}-d)  nlist={NLIST}")
    print("=" * 78)
    print(f"\n{'Method':<42} {'Packed':>8} {'Resident':>9} {'Ratio':>6}  Notes")
    print("─" * 78)

    def row(name, packed_b, resident_b, note=""):
        ratio = resident_b / packed_b if packed_b else 0
        print(f"  {name:<40} {mb(packed_b):>8} {mb(resident_b):>9} "
              f"{ratio:>5.1f}×  {note}")

    # IVF-TQ rows
    for bits in (4, 6):
        for store_raw in (False, True):
            key = f"b{bits}_{'raw' if store_raw else 'noraw'}"
            r = tq_results[key]
            suffix = "(+raw vectors)" if store_raw else "(compression-only)"
            row(f"IVF-TQ b={bits} {suffix}", r["packed"], r["resident"])
            bd = r["breakdown"]
            print(f"    partition arrays={mb(bd['partition_arrays'])}  "
                  f"centroids={mb(bd['centroids'])}  "
                  f"raw_vectors={mb(bd['raw_vectors'])}")

    print()

    # FAISS measured rows
    if faiss_results:
        for key, r in faiss_results.items():
            if key == "ivfpq_m64":
                name = "FAISS IVF-PQ m=64  (8-bit)"
                packed = r["codes"]     # codes = packed for 8-bit (1 byte/sub-code)
                note = f"codes={mb(r['codes'])} pq_cents={mb(r['pq_centroids'])} coarse={mb(r['coarse_centroids'])}"
            elif key == "ivfpq_m128":
                name = "FAISS IVF-PQ m=128 (8-bit)"
                packed = r["codes"]
                note = f"codes={mb(r['codes'])} pq_cents={mb(r['pq_centroids'])} coarse={mb(r['coarse_centroids'])}"
            elif key == "hnsw_m32":
                name = "FAISS HNSW M=32"
                packed = r["vectors"]
                note = f"vectors={mb(r['vectors'])} graph={mb(r['graph_links_approx'])}"
            row(name, packed, r["total_resident"], note)

    # Analytical rows (not built)
    print("\n  --- analytical (C++ bit-packed, not built) ---")
    ext4 = ext_rabitq_analytical(N, DIM, 4)
    row("Ext-RaBitQ b=4 (C++ bit-packed)", ext4, ext4,
        "packed=resident: C++ packs bits tightly")
    scann = scann_ah_analytical(N, DIM, 64, 8)
    row("ScaNN AH m=64  (8-bit, analytical)", scann, scann,
        "same layout as FAISS PQ m=64")

    # Key takeaways
    print("\n" + "─" * 78)
    print("Key takeaways:")
    b4_noraw = tq_results["b4_noraw"]["resident"]
    b6_noraw = tq_results["b6_noraw"]["resident"]
    b4_raw   = tq_results["b4_raw"]["resident"]
    b6_raw   = tq_results["b6_raw"]["resident"]

    print(f"  IVF-TQ b=4 and b=6 have IDENTICAL resident: "
          f"{mb(b4_noraw)} / {mb(b6_noraw)} (uint8 per coord regardless of b)")
    if faiss_results and "ivfpq_m64" in faiss_results:
        faiss64 = faiss_results["ivfpq_m64"]["total_resident"]
        print(f"  At matched packed memory (~64 MB), IVF-TQ compression-only resident "
              f"is {b4_noraw/faiss64:.1f}× FAISS IVF-PQ m=64")
    print(f"  Raw-vector overhead: {mb(b6_raw - b6_noraw)} additional "
          f"({b6_raw/b6_noraw:.1f}× compression-only resident)")
    print(f"  store_raw_vectors=False saves {mb(b6_raw - b6_noraw)} at the cost of disabling rerank")


if __name__ == "__main__":
    main()
