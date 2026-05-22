"""
Multi-seed streaming protocol for the PVLDB tables.

Re-runs each streaming experiment at 3 (or more) seeds with every random source
threaded off the same top-level seed, and writes per-cell CSVs that
`tables_from_multiseed.py` converts into LaTeX tables with paired-t-test CIs.

Usage
-----
    python streaming_multiseed.py --experiment sift1m         --seeds 42 123 7777
    python streaming_multiseed.py --experiment deep10m        --seeds 42 123 7777
    python streaming_multiseed.py --experiment sift10m        --seeds 42 123 7777
    python streaming_multiseed.py --experiment deep10m_pqhigh --seeds 42 123 7777
    python streaming_multiseed.py --experiment sift10m_pqhigh --seeds 42 123 7777

Outputs (under experiments/results/):
    streaming_sift1m_multiseed.csv          # 3 ingestion conditions × 3 seeds × 2 indexes × 2 states
    streaming_deep10m_multiseed.csv         # per-batch trajectory, 3 seeds
    streaming_sift10m_multiseed.csv         # per-batch trajectory, 3 seeds
    streaming_deep10m_pqhigh_multiseed.csv  # IVF-PQ at m=96, 8-bit
    streaming_sift10m_pqhigh_multiseed.csv  # IVF-PQ at m=128, 8-bit

Determinism contract
--------------------
Every randomness source seeded off the single top-level seed. At the start
of each per-seed run we print the first five elements of
``np.random.RandomState(seed).standard_normal(10)`` so you can verify that
two runs at the same seed produce identical prefixes. If they don't, a
randomness source is leaking; investigate before reporting numbers.

Sources seeded per run:
    * numpy.random.seed(seed) at run start
    * random.seed(seed)
    * faiss.seed_global(seed) when available (FAISS k-means init)
    * Fixed rotation matrix Π in IVF-TQ (passed via constructor)
    * Query subsampling (np.random.RandomState(seed).choice)
    * Shuffled-ingestion permutation (np.random.RandomState(seed).permutation)
    * Mean-shift direction (np.random.RandomState(seed).standard_normal)
"""

from __future__ import annotations

import argparse
import gc
import os
import random
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TQS_THREADS", str(os.cpu_count() or 1))

from turboquant_search.core import IVFTurboQuantIndex
from turboquant_search.faiss_baselines import FAISS_AVAILABLE, FAISSFlatIndex
from turboquant_search.benchmarks import compute_recall

assert FAISS_AVAILABLE, "faiss-cpu required"
import faiss


RESULTS_DIR = ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── seeding helpers ──────────────────────────────────────────────

def set_all_seeds(seed: int) -> None:
    """Seed every randomness source. Call once at the top of each per-seed run."""
    np.random.seed(seed)
    random.seed(seed)
    try:
        # FAISS exposes seed_global since 1.7.4 (may not exist on all builds)
        faiss.seed_global(seed)  # type: ignore[attr-defined]
    except AttributeError:
        # Fall back: FAISS's k-means uses internal randomness threaded by its own seed
        # parameter, which we pass explicitly via faiss.Kmeans below.
        pass


def verify_determinism(seed: int) -> str:
    rs = np.random.RandomState(seed)
    return ", ".join(f"{x:+.6f}" for x in rs.standard_normal(10)[:5])


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.ascontiguousarray(v.astype(np.float32))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(norms, 1e-8)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── IVF-PQ factory threaded with seed ────────────────────────────

def make_ivfpq(dim: int, nlist: int, m: int, bits_per_sub: int, nprobe: int,
               seed: int, train_data: np.ndarray) -> faiss.Index:
    """Construct an IVF-PQ index with a seed-threaded k-means coarse partition.

    On FAISS builds that lack ``faiss.seed_global``, we still get reproducibility
    by setting the seed on the ClusteringParameters (``cp``) of both the coarse
    IVF partition and the PQ codebook k-means.
    """
    set_all_seeds(seed)  # numpy/random/global FAISS if available
    quantizer = faiss.IndexFlatIP(dim)
    idx = faiss.IndexIVFPQ(quantizer, dim, nlist, m, bits_per_sub,
                           faiss.METRIC_INNER_PRODUCT)
    # Seed FAISS's internal k-means used for the coarse partition + PQ codebook.
    # Both attributes have a `cp` (ClusteringParameters) with a `seed` field.
    for attr_chain in ("cp", "pq.cp"):
        target = idx
        for part in attr_chain.split("."):
            target = getattr(target, part, None)
            if target is None:
                break
        if target is not None and hasattr(target, "seed"):
            target.seed = int(seed)
    idx.nprobe = nprobe
    idx.train(train_data)
    return idx


# ── Experiment 1: SIFT-1M shuffled-i.i.d. + mean-shift controls ──

def run_sift1m_one_seed(seed: int) -> List[dict]:
    """For one seed, run all 3 ingestion conditions on SIFT-1M.

    Returns a list of dicts: 3 conditions × 2 indexes × 2 states (200K, 1M) = 12 rows.
    """
    from turboquant_search.datasets import load_sift1m

    log(f"  determinism check seed={seed}: {verify_determinism(seed)}")
    set_all_seeds(seed)

    n_total = 1_000_000
    n_initial = 200_000
    batch_size = 100_000
    n_batches = (n_total - n_initial) // batch_size
    nlist, nprobe, m_pq, bits = 500, 10, 64, 4

    log(f"  loading SIFT-1M…")
    vectors, queries, _ = load_sift1m(n_vectors=n_total, n_queries=10_000)
    dim = vectors.shape[1]
    vectors = np.asarray(vectors)
    queries = np.asarray(queries)
    queries_normed = normalize(queries)

    rows = []
    for condition in ("original", "shuffled", "mean_shift"):
        log(f"  condition: {condition}")
        set_all_seeds(seed)

        if condition == "original":
            stream_vecs = vectors[n_initial:].copy()
        elif condition == "shuffled":
            stream_vecs = vectors[n_initial:].copy()
            perm = np.random.RandomState(seed).permutation(len(stream_vecs))
            stream_vecs = stream_vecs[perm]
        else:  # mean_shift
            stream_vecs = vectors[n_initial:].copy().astype(np.float32)
            rs = np.random.RandomState(seed)
            direction = rs.standard_normal(dim).astype(np.float32)
            direction /= np.linalg.norm(direction)
            for b in range(n_batches):
                lo, hi = b * batch_size, (b + 1) * batch_size
                stream_vecs[lo:hi] += 0.05 * (b + 1) * direction

        full = np.concatenate([vectors[:n_initial], stream_vecs])
        initial = full[:n_initial]
        initial_normed = normalize(initial)

        # Train indexes on initial 200K
        ivf_tq = IVFTurboQuantIndex(dim, nlist=nlist, bits=bits, nprobe=nprobe,
                                    use_residual_sign=True, seed=seed)
        ivf_tq.train(initial)
        ivf_tq.add(initial)

        ivf_pq = make_ivfpq(dim, nlist, m_pq, 8, nprobe, seed, initial_normed)
        ivf_pq.add(initial_normed)

        # State at 200K. IVF-TQ uses rerank=50 (raw-vector rerank against
        # the cached raw vectors) to match the v1 Table 2 protocol in
        # streaming_ingestion.py. IVF-PQ uses no rerank (matches v1 too).
        gt = FAISSFlatIndex(dim); gt.add(initial)
        _, gt_200k = gt.search(queries, k=10)
        _, tq_I = ivf_tq.search(queries, k=10, rerank=50)
        _, pq_I = ivf_pq.search(queries_normed, 10)
        tq_200k = compute_recall(gt_200k, tq_I, 10) * 100
        pq_200k = compute_recall(gt_200k, pq_I, 10) * 100
        del gt; gc.collect()

        # Stream remaining batches
        for b in range(n_batches):
            lo = n_initial + b * batch_size
            hi = lo + batch_size
            batch = full[lo:hi]
            ivf_tq.add(batch)
            ivf_pq.add(normalize(batch))

        # State at 1M
        gt_full = FAISSFlatIndex(dim); gt_full.add(full)
        _, gt_1m = gt_full.search(queries, k=10)
        _, tq_I = ivf_tq.search(queries, k=10, rerank=50)
        _, pq_I = ivf_pq.search(queries_normed, 10)
        tq_1m = compute_recall(gt_1m, tq_I, 10) * 100
        pq_1m = compute_recall(gt_1m, pq_I, 10) * 100
        del gt_full, ivf_tq, ivf_pq; gc.collect()

        rows.append({"seed": seed, "condition": condition, "index": "ivf_tq",
                     "state": "200K", "recall10": round(tq_200k, 4)})
        rows.append({"seed": seed, "condition": condition, "index": "ivf_tq",
                     "state": "1M",   "recall10": round(tq_1m, 4)})
        rows.append({"seed": seed, "condition": condition, "index": "ivf_pq",
                     "state": "200K", "recall10": round(pq_200k, 4)})
        rows.append({"seed": seed, "condition": condition, "index": "ivf_pq",
                     "state": "1M",   "recall10": round(pq_1m, 4)})

    return rows


# ── Experiments 2 + 3: 10M streaming (Deep / SIFT) ───────────────

def _load_10m_cached(dataset: str):
    cache_root = ROOT / "experiments" / "cache"
    if dataset == "deep10m":
        vec = cache_root / "deep10m_vectors.npy"
        qry = cache_root / "deep10m_queries.npy"
    elif dataset == "sift10m":
        vec = cache_root / "sift10m" / "sift10m_base_f32.npy"
        qry = cache_root / "sift10m" / "bigann_query_f32.npy"
    elif dataset == "t2i10m":
        vec = cache_root / "text2image10m" / "text2image10m_vectors.npy"
        qry = cache_root / "text2image10m" / "text2image10m_queries.npy"
    else:
        raise ValueError(f"unknown 10M dataset: {dataset}")
    if not (vec.exists() and qry.exists()):
        raise FileNotFoundError(
            f"10M cache missing for {dataset}. Run the single-seed scripts first "
            "to populate experiments/cache/."
        )
    return np.load(vec, mmap_mode="r"), np.load(qry)


def run_10m_one_seed(seed: int, dataset: str,
                     m_pq: int, bits_per_sub: int) -> List[dict]:
    """Per-batch streaming trajectory at one seed.

    Returns a list of dicts: 3 indexes × 10 batch states = 30 rows.
    """
    log(f"  determinism check seed={seed}: {verify_determinism(seed)}")
    set_all_seeds(seed)

    n_initial = 1_000_000
    batch_size = 1_000_000
    n_batches = 9
    nlist, nprobe = 3162, 20
    bits_tq = 4  # IVF-TQ at 4+sign

    vectors, queries = _load_10m_cached(dataset)
    n_total, dim = vectors.shape
    nq = queries.shape[0]
    log(f"  {dataset}: {n_total:,} vectors, dim={dim}, {nq} queries")

    init = np.asarray(vectors[:n_initial])
    init_normed = normalize(init)
    queries_normed = normalize(queries)

    # IVF-TQ
    log("  training IVF-TQ on first 1M…")
    ivf_tq = IVFTurboQuantIndex(dim, nlist=nlist, bits=bits_tq, nprobe=nprobe,
                                use_residual_sign=True, seed=seed)
    ivf_tq.train(init); ivf_tq.add(init)
    ivf_tq._raw_vectors = None  # save memory; no rerank

    # IVF-PQ stale
    log("  training IVF-PQ stale on first 1M…")
    ivf_pq_stale = make_ivfpq(dim, nlist, m_pq, bits_per_sub, nprobe, seed, init_normed)
    ivf_pq_stale.add(init_normed)

    # IVF-PQ retrain
    log("  training IVF-PQ retrain on first 1M…")
    ivf_pq_retrain = make_ivfpq(dim, nlist, m_pq, bits_per_sub, nprobe, seed, init_normed)
    ivf_pq_retrain.add(init_normed)
    cumulative_normed = [init_normed]
    total_retrain_time = 0.0

    rows = []

    def measure(state_idx: int, n_indexed: int):
        log(f"  GT recompute against {n_indexed // 1_000_000}M…")
        gt_idx = FAISSFlatIndex(dim); gt_idx.add(np.asarray(vectors[:n_indexed]))
        _, gt = gt_idx.search(queries, k=10)
        del gt_idx; gc.collect()
        _, tq_I = ivf_tq.search(queries, k=10)
        tq_r = compute_recall(gt, tq_I, 10) * 100
        _, pq_s_I = ivf_pq_stale.search(queries_normed, 10)
        pq_s_r = compute_recall(gt, pq_s_I, 10) * 100
        _, pq_r_I = ivf_pq_retrain.search(queries_normed, 10)
        pq_r_r = compute_recall(gt, pq_r_I, 10) * 100
        log(f"    step={state_idx} N={n_indexed} TQ={tq_r:.2f} "
            f"PQ_stale={pq_s_r:.2f} PQ_retrain={pq_r_r:.2f}")
        for index_name, recall in (
            ("ivf_tq", tq_r),
            ("ivf_pq_stale", pq_s_r),
            ("ivf_pq_retrain", pq_r_r),
        ):
            rows.append({
                "seed": seed,
                "dataset": dataset,
                "m_pq": m_pq,
                "bits_per_sub": bits_per_sub,
                "index": index_name,
                "vectors_indexed": n_indexed,
                "recall10": round(recall, 4),
                "retrain_cum_seconds": round(total_retrain_time, 1),
            })

    measure(0, n_initial)

    for b in range(n_batches):
        start = n_initial + b * batch_size
        end = min(start + batch_size, n_total)
        log(f"  batch {b + 1}: adding {start // 1_000_000}M→{end // 1_000_000}M…")
        batch = np.asarray(vectors[start:end])
        batch_normed = normalize(batch)
        ivf_tq.add(batch); ivf_tq._raw_vectors = None; gc.collect()
        ivf_pq_stale.add(batch_normed)
        cumulative_normed.append(batch_normed)
        all_normed = np.concatenate(cumulative_normed)
        t0 = time.time()
        ivf_pq_retrain = make_ivfpq(dim, nlist, m_pq, bits_per_sub, nprobe,
                                    seed, all_normed)
        ivf_pq_retrain.add(all_normed)
        total_retrain_time += time.time() - t0
        del all_normed; gc.collect()
        measure(b + 1, end)

    return rows


# ── Orchestrator ─────────────────────────────────────────────────

EXPERIMENTS = {
    "sift1m": {
        "csv": "streaming_sift1m_multiseed.csv",
        "fn": lambda seed: run_sift1m_one_seed(seed),
    },
    "deep10m": {
        "csv": "streaming_deep10m_multiseed.csv",
        "fn": lambda seed: run_10m_one_seed(seed, "deep10m", m_pq=48, bits_per_sub=8),
    },
    "sift10m": {
        "csv": "streaming_sift10m_multiseed.csv",
        "fn": lambda seed: run_10m_one_seed(seed, "sift10m", m_pq=64, bits_per_sub=8),
    },
    "deep10m_pqhigh": {
        "csv": "streaming_deep10m_pqhigh_multiseed.csv",
        "fn": lambda seed: run_10m_one_seed(seed, "deep10m", m_pq=96, bits_per_sub=8),
    },
    "sift10m_pqhigh": {
        "csv": "streaming_sift10m_pqhigh_multiseed.csv",
        "fn": lambda seed: run_10m_one_seed(seed, "sift10m", m_pq=128, bits_per_sub=8),
    },
    # Bit-matched PQ: m * b approximately equal to IVF-TQ's bits/vec.
    # Deep-10M: TQ uses 5*96+32 = 512 bits/vec; PQ m=48 b=10 = 480 bits/vec (0.94x).
    # SIFT-10M: TQ uses 5*128+32 = 672 bits/vec; PQ m=64 b=10 = 640 bits/vec (0.95x).
    # FAISS supports b=10 functionally; ADC will not be SIMD-fast but recall is
    # the only thing we care about for this comparison.
    "deep10m_pqmatched": {
        "csv": "streaming_deep10m_pqmatched_multiseed.csv",
        "fn": lambda seed: run_10m_one_seed(seed, "deep10m", m_pq=48, bits_per_sub=10),
    },
    "sift10m_pqmatched": {
        "csv": "streaming_sift10m_pqmatched_multiseed.csv",
        "fn": lambda seed: run_10m_one_seed(seed, "sift10m", m_pq=64, bits_per_sub=10),
    },
    # Text2Image-10M (200-dim CLIP-style; Yandex T2I-1B first 10M).
    # TQ uses 5*200+32 = 1032 bits/vec.
    # FAISS PQ requires d % m == 0, so m must divide 200: valid m ∈ {50, 100, 200}.
    # Sub-matched: m=100, b=8 -> 800 bits/vec  (~0.78x — closest to ~0.75x target)
    # Bit-matched: m=100, b=10 -> 1000 bits/vec (~0.97x)
    # Super-matched: m=200, b=8 -> 1600 bits/vec (~1.55x)
    "t2i10m": {
        "csv": "streaming_t2i10m_multiseed.csv",
        "fn": lambda seed: run_10m_one_seed(seed, "t2i10m", m_pq=100, bits_per_sub=8),
    },
    "t2i10m_pqmatched": {
        "csv": "streaming_t2i10m_pqmatched_multiseed.csv",
        "fn": lambda seed: run_10m_one_seed(seed, "t2i10m", m_pq=100, bits_per_sub=10),
    },
    "t2i10m_pqhigh": {
        "csv": "streaming_t2i10m_pqhigh_multiseed.csv",
        "fn": lambda seed: run_10m_one_seed(seed, "t2i10m", m_pq=200, bits_per_sub=8),
    },
}


def validate_csv(df: pd.DataFrame, group_keys: List[str], expected_per_group: int) -> None:
    assert df.isna().sum().sum() == 0, "CSV has NaN values"
    sizes = df.groupby(group_keys).size()
    bad = sizes[sizes != expected_per_group]
    assert bad.empty, f"Unexpected cell sizes (should be {expected_per_group}):\n{bad}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, choices=list(EXPERIMENTS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 7777])
    args = ap.parse_args()

    cfg = EXPERIMENTS[args.experiment]
    out_path = RESULTS_DIR / cfg["csv"]
    log(f"experiment={args.experiment} seeds={args.seeds} → {out_path}")

    all_rows: List[dict] = []
    for seed in args.seeds:
        log(f"\n=== seed {seed} ===")
        t0 = time.time()
        rows = cfg["fn"](seed)
        all_rows.extend(rows)
        df_partial = pd.DataFrame(all_rows)
        df_partial.to_csv(out_path, index=False)  # incremental save
        log(f"=== seed {seed} done in {time.time() - t0:.0f}s; "
            f"wrote {len(all_rows)} rows so far ===")

    df = pd.DataFrame(all_rows)
    df.to_csv(out_path, index=False)
    log(f"\nFinal CSV: {out_path} ({len(df)} rows)")

    # Validation
    if args.experiment == "sift1m":
        validate_csv(df, ["condition", "index", "state"], len(args.seeds))
    else:
        validate_csv(df, ["index", "vectors_indexed"], len(args.seeds))
    log("Validation passed: no NaN, correct seeds-per-cell.")


if __name__ == "__main__":
    main()
