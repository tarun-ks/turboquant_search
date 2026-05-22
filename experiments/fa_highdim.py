"""
FA-IVF-TQ at dim=768 — final breakthrough attempt.

Hypothesis: at higher dimension (BERT-style 768-dim embeddings), the
bit-budget gradient may be steeper or shallower than at dim=96, changing
the regime where FA-IVF-TQ pays off. Two competing intuitions:

    A. Higher dim averages out per-coord errors, making bit precision
       per coord matter LESS. Demoting to 2-bit hurts less. FA-IVF-TQ
       wins by saving memory at little recall cost.

    B. Higher dim means more coords, each with smaller energy share.
       Score-gap-to-noise ratio matters more. Bit precision matters
       more. Demoting to 2-bit hurts MORE. FA-IVF-TQ loses worse.

The experiment decides empirically.

Data: synthesize 1M unit vectors at dim=768 with cluster structure
(100 random centers, isotropic Gaussian noise). Real BERT-like dimension
without requiring sentence-transformers (which isn't installed locally).

Same protocol as freq_adaptive_realistic_v2.py:
    1. Build uniform 4-bit IVF-TQ
    2. Run 5K-query Pareto-skewed warmup with top-50 hit tracking
    3. Identify hot=top-0.5%, cold=bottom-50% by hit count
    4. Build fresh FA-IVF-TQ with hot=6-bit, cold=2-bit
    5. Evaluate on UNSEEN held-out queries (100 popular + 1000 rare)

Comparison:
    A. Uniform 2-bit
    B. Uniform 4-bit
    C. Uniform 6-bit
    D. FA-IVF-TQ (hot=6, cold=2, discovered from history)
"""

import gc, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.environ["TQS_THREADS"] = str(os.cpu_count() or 1)

from turboquant_search.core import IVFTurboQuantIndex
from turboquant_search.freq_adaptive import FrequencyAdaptiveIVFTQ
from turboquant_search.faiss_baselines import FAISS_AVAILABLE, FAISSFlatIndex
from turboquant_search.benchmarks import compute_recall

assert FAISS_AVAILABLE


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def normalize(v):
    v = np.ascontiguousarray(v.astype(np.float32))
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(n, 1e-8)


def synthesize_clustered_high_dim(n=1_000_000, dim=768, n_clusters=100,
                                   noise_std=0.3, n_queries=2_000, seed=42):
    """Synthesize 1M unit vectors at dim=768 with cluster structure.

    Mimics BERT-like sentence embeddings: clustered distribution on the
    unit sphere with topical structure.
    """
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_clusters, dim).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    cluster_ids = rng.randint(0, n_clusters, size=n)
    vectors = centers[cluster_ids] + rng.randn(n, dim).astype(np.float32) * noise_std
    vectors = normalize(vectors)

    # Queries: a mix of cluster-near (popular regions) and uniform random (rare)
    pop_q = centers[rng.randint(0, n_clusters, size=n_queries // 2)] + \
            rng.randn(n_queries // 2, dim).astype(np.float32) * 0.1
    pop_q = normalize(pop_q)
    rare_q = rng.randn(n_queries - n_queries // 2, dim).astype(np.float32)
    rare_q = normalize(rare_q)
    queries = np.concatenate([pop_q, rare_q])
    return vectors, queries


def build_ivftq(vectors, dim, nlist, bits, nprobe, seed=42):
    idx = IVFTurboQuantIndex(dim, nlist=nlist, bits=bits, nprobe=nprobe,
                              use_residual_sign=True, seed=seed)
    idx.train(vectors); idx.add(vectors)
    return idx


def evaluate(idx, pop_eval, rare_eval, gt_pop, gt_rare, rerank=0):
    _, I = idx.search(pop_eval, k=10, rerank=rerank); pop = compute_recall(gt_pop, I, 10)
    _, I = idx.search(rare_eval, k=10, rerank=rerank); rare = compute_recall(gt_rare, I, 10)
    w = 0.8 * pop + 0.2 * rare
    return pop, rare, w


def main():
    DIM = 768
    N = 1_000_000
    N_CLUSTERS = 10_000
    NOISE = 0.05
    NLIST, NPROBE = 1000, 20

    log(f"Synthesizing {N:,} clustered unit vectors at dim={DIM} ...")
    t0 = time.time()
    vectors, queries = synthesize_clustered_high_dim(
        n=N, dim=DIM, n_clusters=N_CLUSTERS, noise_std=NOISE,
        n_queries=2_500, seed=42,
    )
    log(f"  synth done in {time.time()-t0:.1f}s; "
        f"vectors {vectors.shape}, queries {queries.shape}")

    rng = np.random.RandomState(2026)
    perm = rng.permutation(queries.shape[0])
    pop_train  = queries[perm[:100]]
    pop_eval   = queries[perm[100:200]]
    rare_train = queries[perm[200:1200]]
    rare_eval  = queries[perm[1200:2200]]

    log("Computing held-out ground truth...")
    flat = FAISSFlatIndex(DIM); flat.add(vectors)
    _, gt_pop  = flat.search(pop_eval, 10)
    _, gt_rare = flat.search(rare_eval, 10)
    del flat; gc.collect()

    results = {}

    # ── Warmup workload (Pareto-skewed) ──
    N_WARMUP = 5_000
    n_pop_warmup = 4000
    pop_warmup_idx = rng.choice(100, size=n_pop_warmup, replace=True)
    rare_warmup_idx = rng.choice(1000, size=N_WARMUP - n_pop_warmup, replace=False)
    warmup_q = np.concatenate([pop_train[pop_warmup_idx],
                                rare_train[rare_warmup_idx]])
    rng.shuffle(warmup_q)

    # ── Build uniform indexes ──
    for bits in [2, 4, 6]:
        log(f"Build uniform {bits}-bit IVF-TQ ...")
        t0 = time.time()
        idx = build_ivftq(vectors, DIM, NLIST, bits, NPROBE)
        log(f"  built in {time.time()-t0:.1f}s, mem={idx.memory_bytes/(1024*1024):.1f}MB")
        pop, rare, w = evaluate(idx, pop_eval, rare_eval, gt_pop, gt_rare, rerank=0)
        results[f"uniform_{bits}bit"] = {
            "memory_mb": round(idx.memory_bytes / (1024 * 1024), 1),
            "popular_recall10": round(pop * 100, 2),
            "rare_recall10":    round(rare * 100, 2),
            "weighted_80_20":   round(w * 100, 2),
        }
        log(f"  uniform {bits}-bit: pop={pop:.1%} rare={rare:.1%} w={w:.1%}")
        # Save the 4-bit index for warmup
        if bits == 4:
            u4 = idx
        else:
            del idx; gc.collect()

    # ── Phase 1: warmup hit tracking on uniform 4-bit ──
    log(f"Warmup: {N_WARMUP} queries with top-50 hit tracking on uniform 4-bit ...")
    hit = np.zeros(N, dtype=np.int32)
    chunk = 250
    for s in range(0, N_WARMUP, chunk):
        e = min(s + chunk, N_WARMUP)
        _, I = u4.search(warmup_q[s:e], k=50, rerank=0)
        for row in I:
            for v in row:
                if v >= 0: hit[int(v)] += 1
    log(f"  hit nonzero: {(hit>0).sum():,}  max={int(hit.max())}")
    del u4; gc.collect()

    # ── Phase 2: build FA-IVF-TQ with discovered bit widths ──
    HOT_FRAC = 0.005
    COLD_FRAC = 0.50
    n_hot = int(N * HOT_FRAC)
    n_cold = int(N * COLD_FRAC)
    sorted_by_hits = np.argsort(-hit)
    hot_set = set(sorted_by_hits[:n_hot].tolist())
    bottom = sorted_by_hits[-n_cold:]
    cold_set = set(int(x) for x in bottom if int(x) not in hot_set)
    bw = np.full(N, 4, dtype=np.int64)
    for v in hot_set:  bw[v] = 6
    for v in cold_set: bw[v] = 2
    log(f"Discovered: hot={len(hot_set):,} cold={len(cold_set):,} "
        f"medium={N - len(hot_set) - len(cold_set):,}")

    log("Build FA-IVF-TQ with discovered bit widths ...")
    t0 = time.time()
    fa = FrequencyAdaptiveIVFTQ(DIM, nlist=NLIST, nprobe=NPROBE, seed=42,
                                 bit_widths=(2, 4, 6), default_bits=4)
    fa.train(vectors); fa.add(vectors, bit_widths_per_vec=bw)
    log(f"  built in {time.time()-t0:.1f}s, avg_bits={fa.avg_bits_per_vector:.2f}, "
        f"mem={fa.memory_bytes/(1024*1024):.1f}MB")

    # ── Phase 3: held-out evaluation ──
    log("Held-out eval (UNSEEN queries) ...")
    _, I = fa.search(pop_eval, k=10, update_hits=False); fa_pop = compute_recall(gt_pop, I, 10)
    _, I = fa.search(rare_eval, k=10, update_hits=False); fa_rare = compute_recall(gt_rare, I, 10)
    fa_w = 0.8 * fa_pop + 0.2 * fa_rare
    log(f"  FA-IVF-TQ (discovered): pop={fa_pop:.1%} rare={fa_rare:.1%} w={fa_w:.1%}")
    results["fa_ivf_tq_discovered"] = {
        "memory_mb": round(fa.memory_bytes / (1024 * 1024), 1),
        "avg_bits_per_coord": round(fa.avg_bits_per_vector, 2),
        "popular_recall10": round(fa_pop * 100, 2),
        "rare_recall10":    round(fa_rare * 100, 2),
        "weighted_80_20":   round(fa_w * 100, 2),
        "hot_count": len(hot_set),
        "cold_count": len(cold_set),
        "memory_breakdown": fa.memory_breakdown(),
    }

    # ── Verdict ──
    u4_mb = results["uniform_4bit"]["memory_mb"]
    u4_w = results["uniform_4bit"]["weighted_80_20"]
    fa_mb = results["fa_ivf_tq_discovered"]["memory_mb"]
    fa_w_pct = results["fa_ivf_tq_discovered"]["weighted_80_20"]
    delta_mb = 100 * (u4_mb - fa_mb) / u4_mb
    delta_w = fa_w_pct - u4_w

    log(f"\n=== VERDICT (dim={DIM}) ===")
    log(f"Uniform 4-bit:  {u4_mb:.1f} MB   weighted R@10 = {u4_w:.2f}%")
    log(f"FA-IVF-TQ:      {fa_mb:.1f} MB   weighted R@10 = {fa_w_pct:.2f}%")
    log(f"Δ memory: {delta_mb:+.1f}%   Δ weighted R@10: {delta_w:+.2f}pp")
    if delta_w >= -2.0 and delta_mb >= 20:
        log(f"BREAKTHROUGH: significant memory savings at acceptable recall cost")
    elif delta_w >= 0:
        log(f"WIN: free memory savings at no recall cost")
    else:
        log(f"FAIL: recall hit too large for memory savings")

    out = ROOT / "experiments" / "fa_highdim_results.json"
    with open(out, "w") as f:
        json.dump({"config": {"dim": DIM, "n": N, "nlist": NLIST, "nprobe": NPROBE,
                              "n_clusters": N_CLUSTERS, "noise_std": NOISE},
                    "results": results}, f, indent=2, default=str)
    log(f"Saved {out}")


if __name__ == "__main__":
    main()
