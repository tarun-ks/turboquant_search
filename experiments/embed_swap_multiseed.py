"""
Multi-seed encoder-swap experiment.

Encodes passages + queries ONCE (cached to disk), then runs the streaming
index protocol for each seed. Outputs per-seed trajectories and paired-t
aggregates across seeds, mirroring the multi-seed-streaming convention in
streaming_multiseed.py.

Usage:
    # Encode + cache (heaviest step, run once per (passage set, model)):
    python embed_swap_multiseed.py --swap bge --seeds 42 123 7777 --reuse-cache

    # The cache lives at experiments/cache/embed_swap/{model_safe_name}_{N}.npy

Swaps:
    --swap minilm   :  all-MiniLM-L6-v2  ->  all-MiniLM-L12-v2  (gentle, cos~0.51)
    --swap bge      :  all-MiniLM-L6-v2  ->  BAAI/bge-small-en-v1.5  (harsh, cos~0.24)

Output: experiments/embed_swap_multiseed_{swap}.json
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

# Avoid HuggingFace tokenizers fork-after-parallelism deadlock that
# silently kills the process on macOS when faiss/sklearn touch
# OpenMP after the dataset loader has spawned workers.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# PyTorch (MPS) and FAISS each ship their own OpenMP runtime; loading
# both in the same process aborts silently on macOS. Allow duplicate
# OMP lib registration to make them co-exist.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

CACHE_DIR = ROOT / "experiments" / "cache" / "embed_swap"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Defaults: 200K initial, 800K new-encoder, 5K queries, 100K batches.
N_INIT = 200_000
N_NEW = 800_000
N_QUERIES = 5_000
BATCH = 100_000

MODEL_OLD = "sentence-transformers/all-MiniLM-L6-v2"
SWAPS = {
    "minilm": "sentence-transformers/all-MiniLM-L12-v2",
    "bge":    "BAAI/bge-small-en-v1.5",
}

# Harsh-swap (BGE) is compute-budgeted at +300K;
# gentle swap (L12) runs all 8 batches (+800K).
N_NEW_BUDGET = {
    "bge":    300_000,
    "minilm": 800_000,
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def model_safe(name):
    return name.replace("/", "_").replace("-", "_")


def load_passages(n_passages, n_queries, cache_key="msmarco"):
    """Load text passages + queries from MS MARCO. Cached as JSON for replay."""
    cache_file = CACHE_DIR / f"{cache_key}_n{n_passages}_q{n_queries}.json"
    if cache_file.exists():
        log(f"Reusing cached passage text from {cache_file.name}")
        with open(cache_file) as f:
            d = json.load(f)
        return d["passages"], d["queries"]
    from datasets import load_dataset
    log(f"Loading {n_passages + n_queries:,} passages from MS MARCO...")
    ds = load_dataset("ms_marco", "v2.1", split="train", streaming=True)
    passages, queries = [], []
    seen = set()
    for ex in ds:
        if len(queries) < n_queries:
            queries.append(ex["query"])
        for p in ex["passages"]["passage_text"]:
            if p in seen:
                continue
            seen.add(p)
            if len(passages) < n_passages:
                passages.append(p)
            if len(passages) >= n_passages and len(queries) >= n_queries:
                break
        if len(passages) >= n_passages and len(queries) >= n_queries:
            break
    with open(cache_file, "w") as f:
        json.dump({"passages": passages, "queries": queries}, f)
    log(f"Cached passage text → {cache_file.name}")
    return passages, queries


def encode_with_cache(model_name, texts, role, n, batch_size=512, device=None):
    """Encode texts; cache to disk keyed by (model, role, n).
    role: "passages" or "queries"."""
    cache_file = CACHE_DIR / f"{model_safe(model_name)}_{role}_n{n}.npy"
    if cache_file.exists():
        log(f"Reusing cached encoding {cache_file.name}")
        return np.load(cache_file)
    from sentence_transformers import SentenceTransformer
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    log(f"Encoding {len(texts):,} {role} with {model_name} on {device}...")
    m = SentenceTransformer(model_name, device=device)
    t0 = time.time()
    emb = m.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                   show_progress_bar=True, normalize_embeddings=True)
    dt = time.time() - t0
    log(f"  done in {dt/60:.1f} min ({len(texts)/dt:.0f} {role}/sec)")
    emb = emb.astype(np.float32)
    np.save(cache_file, emb)
    log(f"  saved → {cache_file.name}")
    # Free model GPU memory
    del m
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        try:
            torch.mps.empty_cache()
        except AttributeError:
            pass
    gc.collect()
    return emb


def sample_cos(model_old, model_new, sample_texts, device):
    """Cross-encoder cosine on a sample, to confirm gentle vs harsh swap."""
    from sentence_transformers import SentenceTransformer
    m_old = SentenceTransformer(model_old, device=device)
    m_new = SentenceTransformer(model_new, device=device)
    a = m_old.encode(sample_texts, batch_size=64, convert_to_numpy=True,
                     normalize_embeddings=True)
    b = m_new.encode(sample_texts, batch_size=64, convert_to_numpy=True,
                     normalize_embeddings=True)
    return float(np.mean((a * b).sum(axis=1)))


def run_one_seed(seed, init_emb, new_emb, queries_emb, n_new_budget,
                 nlist=500, nprobe=20, m_pq=96, batch=BATCH):
    """Run the streaming encoder-swap protocol for one seed.
    Returns list of per-step records."""
    import faiss
    # PyTorch's OpenMP runtime (initialised when sentence-transformers
    # encoded above) collides with FAISS's OpenMP on macOS. Forcing
    # FAISS to single-threaded sidesteps the silent abort.
    try:
        faiss.omp_set_num_threads(1)
    except AttributeError:
        pass
    from turboquant_search.core import IVFTurboQuantIndex
    from turboquant_search.faiss_baselines import FAISSFlatIndex
    from turboquant_search.benchmarks import compute_recall

    np.random.seed(seed)
    # FAISS internal RNG seed (affects k-means init)
    try:
        faiss.cvar.indexIVF_stats.reset()
    except AttributeError:
        pass

    dim = init_emb.shape[1]

    log(f"  [seed={seed}] training IVF-TQ on init (N={init_emb.shape[0]:,})...")
    ivf_tq = IVFTurboQuantIndex(dim, nlist=nlist, bits=4, nprobe=nprobe,
                                use_residual_sign=True, seed=seed)
    ivf_tq.train(init_emb)
    ivf_tq.add(init_emb)

    log(f"  [seed={seed}] training IVF-PQ stale (m_pq={m_pq})...")
    quantizer_stale = faiss.IndexFlatIP(dim)
    ivf_pq_stale = faiss.IndexIVFPQ(quantizer_stale, dim, nlist, m_pq, 8,
                                    faiss.METRIC_INNER_PRODUCT)
    ivf_pq_stale.cp.seed = int(seed)
    ivf_pq_stale.pq.cp.seed = int(seed) + 1
    ivf_pq_stale.nprobe = nprobe
    log(f"  [seed={seed}]   .train() start")
    ivf_pq_stale.train(init_emb)
    log(f"  [seed={seed}]   .train() done; .add() start")
    ivf_pq_stale.add(init_emb)
    log(f"  [seed={seed}]   .add() done")

    def make_ivfpq(data):
        q = faiss.IndexFlatIP(dim)
        idx = faiss.IndexIVFPQ(q, dim, nlist, m_pq, 8, faiss.METRIC_INNER_PRODUCT)
        idx.cp.seed = int(seed) + 100        # different seed for retrain to
        idx.pq.cp.seed = int(seed) + 101     # avoid coupling retrain init to stale
        idx.nprobe = nprobe
        idx.train(data)
        return idx

    log(f"  [seed={seed}] training IVF-PQ retrain on init...")
    ivf_pq_retrain = make_ivfpq(init_emb)
    ivf_pq_retrain.add(init_emb)
    cumulative = [init_emb]
    retrain_time = 0.0

    records = []

    def measure(label, cumulative_db):
        gt_idx = FAISSFlatIndex(dim)
        gt_idx.add(cumulative_db)
        _, gt = gt_idx.search(queries_emb, k=10)
        _, tq_I = ivf_tq.search(queries_emb, k=10, rerank=50)
        _, ps_I = ivf_pq_stale.search(queries_emb, 10)
        _, pr_I = ivf_pq_retrain.search(queries_emb, 10)
        tq_r = compute_recall(gt, tq_I, 10)
        ps_r = compute_recall(gt, ps_I, 10)
        pr_r = compute_recall(gt, pr_I, 10)
        rec = {
            "step": label,
            "n_indexed": int(cumulative_db.shape[0]),
            "ivf_tq": round(tq_r * 100, 2),
            "ivf_pq_stale": round(ps_r * 100, 2),
            "ivf_pq_retrain": round(pr_r * 100, 2),
            "retrain_time_cumulative_s": round(retrain_time, 1),
        }
        records.append(rec)
        log(f"    {label}: TQ={tq_r:.1%} PQ_stale={ps_r:.1%} PQ_retrain={pr_r:.1%}")

    measure("Initial (200K, old-enc)", init_emb)

    n_batches_budget = n_new_budget // batch
    for i in range(n_batches_budget):
        start, end = i * batch, (i + 1) * batch
        bsub = new_emb[start:end]
        ivf_tq.add(bsub)
        ivf_pq_stale.add(bsub)
        cumulative.append(bsub)
        all_data = np.concatenate(cumulative)
        t0 = time.time()
        ivf_pq_retrain = make_ivfpq(all_data)
        ivf_pq_retrain.add(all_data)
        retrain_time += time.time() - t0
        measure(f"+{end//1000}K new-enc", all_data)
        del all_data
        gc.collect()

    return records


def aggregate_paired_t(per_seed_records, key_metric="ivf_tq"):
    """Aggregate per-seed trajectories. Computes mean+CI per step, plus
    the harsh-swap headline: IVF-TQ vs IVF-PQ retrain at the final state."""
    n_seeds = len(per_seed_records)
    if n_seeds == 0:
        return {}
    n_steps = min(len(r) for r in per_seed_records)
    agg_steps = []
    for s in range(n_steps):
        step_label = per_seed_records[0][s]["step"]
        n_indexed = per_seed_records[0][s]["n_indexed"]
        cell = {"step": step_label, "n_indexed": n_indexed}
        for metric in ["ivf_tq", "ivf_pq_stale", "ivf_pq_retrain", "retrain_time_cumulative_s"]:
            vals = [r[s][metric] for r in per_seed_records]
            mn = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if n_seeds > 1 else 0.0
            # 95% CI via t critical at df=n_seeds-1 (for n=3: t=4.303)
            from math import sqrt
            t_crit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(n_seeds, 1.96)
            half = t_crit * sd / sqrt(n_seeds) if n_seeds > 1 else 0.0
            cell[metric] = round(mn, 3)
            cell[f"{metric}_ci95"] = round(half, 3)
        # Paired diff: tq - pq_retrain
        diffs = [r[s]["ivf_tq"] - r[s]["ivf_pq_retrain"] for r in per_seed_records]
        mn = float(np.mean(diffs))
        sd = float(np.std(diffs, ddof=1)) if n_seeds > 1 else 0.0
        from math import sqrt
        t_crit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(n_seeds, 1.96)
        half = t_crit * sd / sqrt(n_seeds) if n_seeds > 1 else 0.0
        # paired-t p-value: use 2-tailed t-distribution
        if n_seeds > 1 and sd > 0:
            t_stat = mn / (sd / sqrt(n_seeds))
            from scipy.stats import t as student_t
            p = float(2 * (1 - student_t.cdf(abs(t_stat), df=n_seeds - 1)))
        else:
            t_stat = float("inf") if mn != 0 else 0.0
            p = 0.0
        cell["paired_tq_minus_retrain"] = round(mn, 3)
        cell["paired_ci95"] = round(half, 3)
        cell["paired_t_stat"] = round(t_stat, 3)
        cell["paired_p"] = round(p, 4)
        agg_steps.append(cell)
    return agg_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swap", choices=["minilm", "bge"], default="bge",
                    help="Encoder swap to run. bge=harsh, minilm=gentle.")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 7777])
    ap.add_argument("--n-init", type=int, default=N_INIT)
    ap.add_argument("--n-new", type=int, default=N_NEW)
    ap.add_argument("--n-queries", type=int, default=N_QUERIES)
    ap.add_argument("--nlist", type=int, default=500)
    ap.add_argument("--nprobe", type=int, default=20)
    ap.add_argument("--reuse-cache", action="store_true", default=True)
    ap.add_argument("--batch-size", type=int, default=512,
                    help="Encoding batch size (smaller if MPS OOMs).")
    args = ap.parse_args()

    n_total = args.n_init + args.n_new
    n_new_budget = N_NEW_BUDGET[args.swap]
    model_new = SWAPS[args.swap]

    log(f"=== Multi-seed encoder swap ===")
    log(f"  swap: {args.swap}   old: {MODEL_OLD}   new: {model_new}")
    log(f"  N_init={args.n_init:,}  N_new={args.n_new:,}  N_new_budget={n_new_budget:,}")
    log(f"  seeds={args.seeds}")

    # 1. Load passages (cached)
    passages, queries_raw = load_passages(n_total, args.n_queries)
    init_passages = passages[:args.n_init]
    new_passages = passages[args.n_init:args.n_init + args.n_new]

    # 2. Encode (cached). MPS used if available.
    import torch
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    log(f"  encoder device: {device}")

    init_emb = encode_with_cache(MODEL_OLD, init_passages,
                                 role="init_passages", n=args.n_init,
                                 batch_size=args.batch_size, device=device)
    new_emb = encode_with_cache(model_new, new_passages,
                                role=f"new_passages_{args.swap}", n=args.n_new,
                                batch_size=args.batch_size, device=device)
    queries_emb = encode_with_cache(model_new, queries_raw,
                                    role=f"queries_{args.swap}", n=args.n_queries,
                                    batch_size=args.batch_size, device=device)

    # 3. Compute cross-encoder cosine on a sample (cached too)
    cos_file = CACHE_DIR / f"cos_{args.swap}.json"
    if cos_file.exists():
        with open(cos_file) as f:
            cos = json.load(f)["cos"]
        log(f"  cached cos(old, new) = {cos:.3f}")
    else:
        cos = sample_cos(MODEL_OLD, model_new, init_passages[:200], device)
        with open(cos_file, "w") as f:
            json.dump({"cos": cos}, f)
        log(f"  cos(old, new) on 200-sample = {cos:.3f}")

    # 4. Run streaming protocol per seed
    per_seed = []
    seed_summaries = {}
    for seed in args.seeds:
        log(f"--- Running seed={seed} ---")
        recs = run_one_seed(seed, init_emb, new_emb, queries_emb,
                            n_new_budget=n_new_budget,
                            nlist=args.nlist, nprobe=args.nprobe)
        per_seed.append(recs)
        seed_summaries[str(seed)] = recs
        # Periodic save in case run is interrupted
        out_path = ROOT / "experiments" / f"embed_swap_multiseed_{args.swap}.json"
        with open(out_path, "w") as f:
            json.dump({
                "config": {
                    "swap": args.swap,
                    "model_old": MODEL_OLD,
                    "model_new": model_new,
                    "n_init": args.n_init,
                    "n_new": args.n_new,
                    "n_new_budget": n_new_budget,
                    "batch": BATCH,
                    "n_queries": args.n_queries,
                    "nlist": args.nlist,
                    "nprobe": args.nprobe,
                    "cos_old_new": cos,
                    "seeds": args.seeds,
                    "device": device,
                },
                "per_seed": seed_summaries,
                "aggregated": aggregate_paired_t(per_seed),
            }, f, indent=2)
        log(f"  partial results saved → {out_path.name}")
        gc.collect()

    log("=== ALL SEEDS DONE ===")


if __name__ == "__main__":
    main()
