"""
Download the first 10M vectors of Yandex Text2Image-1B + 10K queries.

Uses HTTP range requests so we don't pull the full 800GB base file.

Outputs:
    experiments/cache/text2image10m/text2image10m_vectors.npy   (10M × 200, float32)
    experiments/cache/text2image10m/text2image10m_queries.npy   (10K × 200, float32)

The streaming experiment recomputes ground truth against the cumulative
database at each batch, so we don't need to download the official 1B GT.

The official corpus is on Yandex storage with no auth required.
"""

import os
import struct
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "experiments" / "cache" / "text2image10m"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.1B.fbin"
QUERY_URL = "https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.public.100K.fbin"

# .fbin format:
#   header: 2 × int32  (num_vectors, dim)
#   data:   num_vectors × dim × float32, row-major

DIM_EXPECTED = 200
N_BASE = 10_000_000   # 10M slice
N_QUERIES = 10_000    # take first 10K of the 100K query file


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def http_range_download(url, byte_start, byte_end, dest_path, label):
    """Download bytes [byte_start, byte_end) inclusive-exclusive from url to dest_path.
    Resumes if a partial file exists."""
    dest_path = Path(dest_path)
    range_size = byte_end - byte_start
    if dest_path.exists() and dest_path.stat().st_size == range_size:
        log(f"  {label}: already on disk ({range_size / 1e9:.2f} GB), skipping")
        return
    req = urllib.request.Request(url)
    req.add_header("Range", f"bytes={byte_start}-{byte_end - 1}")
    log(f"  {label}: HTTP Range {byte_start}-{byte_end - 1} ({range_size / 1e9:.2f} GB)")
    t0 = time.time()
    total = 0
    chunk = 1 << 20  # 1 MB
    last_log = time.time()
    with urllib.request.urlopen(req) as r, open(dest_path, "wb") as out:
        while True:
            buf = r.read(chunk)
            if not buf:
                break
            out.write(buf)
            total += len(buf)
            if time.time() - last_log > 5:
                rate = total / (time.time() - t0) / 1e6
                pct = 100 * total / range_size
                log(f"    {label}: {total/1e9:.2f}/{range_size/1e9:.2f} GB ({pct:.1f}%), {rate:.1f} MB/s")
                last_log = time.time()
    dt = time.time() - t0
    log(f"  {label}: done in {dt/60:.1f} min, avg {total/dt/1e6:.1f} MB/s")


def read_fbin_header(path_or_bytes):
    """Read the 8-byte .fbin header → (num_vectors, dim)."""
    if isinstance(path_or_bytes, bytes):
        n, d = struct.unpack("<ii", path_or_bytes[:8])
    else:
        with open(path_or_bytes, "rb") as f:
            n, d = struct.unpack("<ii", f.read(8))
    return n, d


def download_base_slice():
    """Download header + first N_BASE vectors of the .fbin base file."""
    out = OUT_DIR / "base_10m.fbin"
    bytes_per_vec = DIM_EXPECTED * 4
    needed = 8 + N_BASE * bytes_per_vec    # 8GB+8B
    if out.exists() and out.stat().st_size == needed:
        log(f"  base slice already complete: {out}")
        return out
    http_range_download(BASE_URL, 0, needed, out, "base 10M slice")
    # Verify header
    n, d = read_fbin_header(out)
    log(f"  base header: claims {n:,} vectors × {d}-dim "
        f"(file holds the first {N_BASE:,})")
    if d != DIM_EXPECTED:
        raise RuntimeError(f"Expected dim={DIM_EXPECTED}, got {d}")
    return out


def download_queries():
    out = OUT_DIR / "query_100k.fbin"
    bytes_per_vec = DIM_EXPECTED * 4
    full_size = 8 + 100_000 * bytes_per_vec
    if out.exists() and out.stat().st_size == full_size:
        log(f"  queries already complete: {out}")
        return out
    http_range_download(QUERY_URL, 0, full_size, out, "queries 100K")
    return out


def fbin_to_npy(fbin_path, npy_path, n_take, dim=DIM_EXPECTED):
    if npy_path.exists():
        a = np.load(npy_path, mmap_mode="r")
        if a.shape == (n_take, dim):
            log(f"  npy already exists: {npy_path}")
            return npy_path
    log(f"  converting {fbin_path.name} → {npy_path.name} ({n_take:,} × {dim})")
    bytes_per_vec = dim * 4
    with open(fbin_path, "rb") as f:
        f.seek(8)  # skip header
        buf = f.read(n_take * bytes_per_vec)
    if len(buf) != n_take * bytes_per_vec:
        raise RuntimeError(f"Short read: got {len(buf)} bytes, expected {n_take * bytes_per_vec}")
    arr = np.frombuffer(buf, dtype=np.float32).reshape(n_take, dim).copy()
    log(f"  array stats: min={arr.min():.3f} max={arr.max():.3f} mean_norm={np.linalg.norm(arr, axis=1).mean():.3f}")
    np.save(npy_path, arr)
    log(f"  saved → {npy_path}")
    return npy_path


def main():
    log("=== Text2Image-10M download ===")
    log(f"  output dir: {OUT_DIR}")

    # 1. Download base slice (8 GB)
    base_path = download_base_slice()
    # 2. Download queries (80 MB)
    query_path = download_queries()

    # 3. Convert to .npy (one-time)
    fbin_to_npy(base_path, OUT_DIR / "text2image10m_vectors.npy", N_BASE)
    fbin_to_npy(query_path, OUT_DIR / "text2image10m_queries.npy", N_QUERIES)

    # 4. Sanity: cosine spread of base vectors (T2I uses learned dot-product
    #    embeddings, so norms may not be 1 — IVF-PQ and IVF-TQ in this paper
    #    normalize internally for IP scoring)
    base = np.load(OUT_DIR / "text2image10m_vectors.npy", mmap_mode="r")
    queries = np.load(OUT_DIR / "text2image10m_queries.npy", mmap_mode="r")
    log("=== summary ===")
    log(f"  base: {base.shape} dtype={base.dtype}")
    log(f"  queries: {queries.shape} dtype={queries.dtype}")
    base_norms = np.linalg.norm(base[:10000], axis=1)
    log(f"  base norm sample (10K): mean={base_norms.mean():.3f} "
        f"min={base_norms.min():.3f} max={base_norms.max():.3f}")
    log("Ready for streaming experiments. Use the .npy files in cache/text2image10m/.")


if __name__ == "__main__":
    main()
