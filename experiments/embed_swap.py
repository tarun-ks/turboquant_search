"""
Distribution-shift via embedding-model swap.

Designed for Colab (free GPU) — encoding 1M passages with sentence-transformers
on CPU is too slow for laptop use.

Setup:
    Train an IVF-PQ codebook on N_INIT passages encoded with MODEL_OLD
    (e.g. all-MiniLM-L6-v2). Stream in N_NEW passages encoded with
    MODEL_NEW (e.g. all-MiniLM-L12-v2). Use a fixed query set encoded
    with MODEL_NEW (the production-deployed encoder). Recall@10 ground
    truth = exact top-10 over the cumulative mixed-encoder database
    (= what the user actually gets).

Outputs experiments/embed_swap_results.json.

Colab cell sequence:
    !pip install sentence-transformers faiss-cpu datasets numpy
    !python experiments/embed_swap.py
"""

import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


N_INIT = 200_000   # passages encoded with the old model
N_NEW = 800_000    # passages encoded with the new model
N_QUERIES = 5_000
BATCH = 100_000
MODEL_OLD = "sentence-transformers/all-MiniLM-L6-v2"   # 384-dim
MODEL_NEW = "sentence-transformers/all-MiniLM-L12-v2"  # 384-dim (same dim!)
DATASET = "ms_marco"  # or "BeIR/beir-corpus" subset


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def normalize(v):
    v = np.ascontiguousarray(v.astype(np.float32))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(norms, 1e-8)


def load_passages(n_passages, n_queries):
    """Load text passages for encoding. Tries MS MARCO first, falls back
    to a smaller BEIR corpus if MS MARCO is unavailable."""
    from datasets import load_dataset
    log(f"Loading {n_passages + n_queries} passages from MS MARCO...")
    ds = load_dataset("ms_marco", "v2.1", split="train",
                      streaming=True)
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
                return passages, queries
    return passages, queries


def encode(model_name, texts, batch_size=512):
    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Encoding {len(texts):,} texts with {model_name} on {device}...")
    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(texts, batch_size=batch_size,
                        convert_to_numpy=True, show_progress_bar=True,
                        normalize_embeddings=True)
    return emb.astype(np.float32)


def main():
    import faiss
    n_total = N_INIT + N_NEW
    passages, queries_raw = load_passages(n_total, N_QUERIES)
    log(f"Loaded {len(passages):,} passages, {len(queries_raw):,} queries.")

    init_passages = passages[:N_INIT]
    new_passages = passages[N_INIT:N_INIT + N_NEW]

    init_emb = encode(MODEL_OLD, init_passages)
    new_emb = encode(MODEL_NEW, new_passages)
    queries_emb = encode(MODEL_NEW, queries_raw)
    dim = init_emb.shape[1]
    log(f"dim={dim}; verifying L6/L12 produce comparable space...")
    # Cross-encoder cosine on a random sample (sanity check):
    sample = init_passages[:200]
    s_old = encode(MODEL_OLD, sample)
    s_new = encode(MODEL_NEW, sample)
    cos = float(np.mean((s_old * s_new).sum(axis=1)))
    log(f"Mean cos(L6(p), L12(p)) = {cos:.3f}  (lower = larger shift)")

    nlist = 500
    nprobe = 20
    m_pq = 96  # 384/4 = 96 subspaces
    queries_n = queries_emb  # already normalized

    # IVF-TQ
    from turboquant_search.core import IVFTurboQuantIndex
    from turboquant_search.faiss_baselines import FAISSFlatIndex
    from turboquant_search.benchmarks import compute_recall

    log("Training IVF-TQ on L6-encoded init passages...")
    ivf_tq = IVFTurboQuantIndex(dim, nlist=nlist, bits=4, nprobe=nprobe,
                                 use_residual_sign=True, seed=42)
    ivf_tq.train(init_emb)
    ivf_tq.add(init_emb)

    log("Training IVF-PQ stale on L6-encoded init passages...")
    quantizer = faiss.IndexFlatIP(dim)
    ivf_pq_stale = faiss.IndexIVFPQ(quantizer, dim, nlist, m_pq, 8,
                                     faiss.METRIC_INNER_PRODUCT)
    ivf_pq_stale.nprobe = nprobe
    ivf_pq_stale.train(init_emb)
    ivf_pq_stale.add(init_emb)

    def make_ivfpq(data):
        q = faiss.IndexFlatIP(dim)
        idx = faiss.IndexIVFPQ(q, dim, nlist, m_pq, 8, faiss.METRIC_INNER_PRODUCT)
        idx.nprobe = nprobe
        idx.train(data)
        return idx

    log("Training IVF-PQ retrain on L6-encoded init passages...")
    ivf_pq_retrain = make_ivfpq(init_emb)
    ivf_pq_retrain.add(init_emb)
    cumulative = [init_emb]
    retrain_time = 0.0

    results = []

    def measure(label, cumulative_db):
        gt_idx = FAISSFlatIndex(dim)
        gt_idx.add(cumulative_db)
        _, gt = gt_idx.search(queries_n, k=10)

        _, tq_I = ivf_tq.search(queries_n, k=10, rerank=50)
        _, ps_I = ivf_pq_stale.search(queries_n, 10)
        _, pr_I = ivf_pq_retrain.search(queries_n, 10)
        tq_r = compute_recall(gt, tq_I, 10)
        ps_r = compute_recall(gt, ps_I, 10)
        pr_r = compute_recall(gt, pr_I, 10)
        entry = {
            "step": label,
            "n_indexed": cumulative_db.shape[0],
            "ivf_tq": round(tq_r * 100, 2),
            "ivf_pq_stale": round(ps_r * 100, 2),
            "ivf_pq_retrain": round(pr_r * 100, 2),
            "retrain_time_cumulative_s": round(retrain_time, 1),
        }
        results.append(entry)
        log(f"  {label}: TQ={tq_r:.1%} PQ_stale={ps_r:.1%} PQ_retrain={pr_r:.1%}")

    measure("Initial (200K, L6 only)", init_emb)

    n_batches = N_NEW // BATCH
    for i in range(n_batches):
        start = i * BATCH
        end = start + BATCH
        batch = new_emb[start:end]
        ivf_tq.add(batch)
        ivf_pq_stale.add(batch)

        cumulative.append(batch)
        all_data = np.concatenate(cumulative)
        t0 = time.time()
        ivf_pq_retrain = make_ivfpq(all_data)
        ivf_pq_retrain.add(all_data)
        retrain_time += time.time() - t0

        measure(f"Batch {i+1} (+{(end)//1000}K L12)", all_data)
        del all_data; gc.collect()

    out = ROOT / "experiments" / "embed_swap_results.json"
    with open(out, "w") as f:
        json.dump({
            "steps": results,
            "swap_at_n": N_INIT,
            "config": {
                "model_old": MODEL_OLD, "model_new": MODEL_NEW,
                "n_init": N_INIT, "n_new": N_NEW,
                "batch": BATCH, "n_queries": N_QUERIES,
                "cos_old_new_sample": cos,
            },
        }, f, indent=2)
    log(f"Saved {out}")


if __name__ == "__main__":
    main()
