"""
IVF-RVQ-TQ: IVF + multi-bit Stage-2 refinement on residuals.

Tests whether the +1pp RVQ-TQ advantage over pure Lloyd-Max at high bit budget
transfers from flat TQ to the IVF-TQ setting.

Comparison:
  - IVF-TQ b-bit: b-bit primary LM + 1-bit sign (b+1 effective)
  - new "IVF-RVQ-TQ (b, b')": b-bit primary LM + b'-bit refinement (b+b' effective)

Outputs experiments/ivf_rvq_tq_results.json.
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy.stats import norm
from sklearn.cluster import MiniBatchKMeans

from turboquant_search.faiss_baselines import FAISS_AVAILABLE, FAISSFlatIndex
from turboquant_search.benchmarks import compute_recall
from turboquant_search.datasets import load_deep1m, load_sift1m
from experiments.rvq_tq_verify import high_resolution_lloyd_max
from experiments.rvq_tq_explore import per_bin_lloyd_max_subcentroids


class IVFRVQTQIndex:
    """IVF index with TurboQuant residual compression + b'-bit Stage 2 refinement."""

    def __init__(self, dim, nlist=1000, bits=4, refine_bits=2, nprobe=20, seed=42):
        self.dim = dim
        self.nlist = nlist
        self.bits = bits
        self.refine_bits = refine_bits
        self.nprobe = nprobe
        self.seed = seed

        # Random rotation
        rng = np.random.default_rng(seed)
        H = rng.normal(size=(dim, dim))
        Q, _ = np.linalg.qr(H)
        self.rotation = Q.astype(np.float32)

        # High-resolution Lloyd-Max primary codebook
        c_raw, b_raw = high_resolution_lloyd_max(bits, n_iter=500, grid_size=100000)
        scale = np.sqrt(dim)
        self.tq_centroids = (c_raw / scale).astype(np.float32)
        self.tq_boundaries = (b_raw / scale).astype(np.float32)

        # Per-primary-bin sub-centroids (RVQ-TQ refinement). CRITICAL: pass the
        # SAME primary codebook used for encoding above; otherwise the per-bin
        # sub-centroids are designed for the wrong bin boundaries and recall
        # collapses at b>=6 where cached vs high-res Lloyd-Max diverge.
        if refine_bits > 0:
            self.sub_centroids, self.sub_boundaries = per_bin_lloyd_max_subcentroids(
                bits, refine_bits, dim, primary_codebook=(c_raw, b_raw)
            )

        self.coarse_centroids = None
        self._partitions = None  # list of dicts per cell
        self._invlists = None    # list of lists of original ids per cell
        self._n = 0

    def train(self, vectors):
        v = np.ascontiguousarray(vectors.astype(np.float32))
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        v = v / np.maximum(norms, 1e-8)

        n_clusters = min(self.nlist, v.shape[0])
        km = MiniBatchKMeans(
            n_clusters=n_clusters, random_state=self.seed,
            batch_size=min(10000, v.shape[0]), n_init=3, max_iter=50,
        )
        km.fit(v)
        cc = km.cluster_centers_.astype(np.float32)
        cn = np.linalg.norm(cc, axis=1, keepdims=True)
        self.coarse_centroids = cc / np.maximum(cn, 1e-8)
        self.nlist = n_clusters
        self._partitions = [None] * n_clusters
        self._invlists = [[] for _ in range(n_clusters)]

    def add(self, vectors):
        assert self.coarse_centroids is not None, "call train() first"
        v = np.ascontiguousarray(vectors.astype(np.float32))
        v_norms = np.linalg.norm(v, axis=1, keepdims=True)
        v_normed = v / np.maximum(v_norms, 1e-8)

        # Assign to nearest centroid
        sims = v_normed @ self.coarse_centroids.T
        assignments = np.argmax(sims, axis=1)

        n_levels = 2 ** self.bits
        n_sub = 2 ** self.refine_bits if self.refine_bits > 0 else 1

        for cell_idx in range(self.nlist):
            mask = assignments == cell_idx
            if not mask.any():
                continue

            local_vecs = v_normed[mask]
            local_ids = np.where(mask)[0] + self._n
            residuals = local_vecs - self.coarse_centroids[cell_idx]

            rotated = residuals @ self.rotation.T
            r_norms = np.linalg.norm(rotated, axis=1, keepdims=True)
            r_norms = np.maximum(r_norms, 1e-8)
            normalized = rotated / r_norms

            primary = np.digitize(normalized, self.tq_boundaries).astype(np.uint16)
            primary = np.clip(primary, 0, n_levels - 1)

            if self.refine_bits > 0:
                sub = np.zeros_like(primary, dtype=np.uint16)
                for i in range(n_levels):
                    pmask = primary == i
                    if not pmask.any():
                        continue
                    vals = normalized[pmask]
                    if n_sub > 1:
                        sub_idx = np.searchsorted(self.sub_boundaries[i], vals)
                        sub_idx = np.clip(sub_idx, 0, n_sub - 1)
                    else:
                        sub_idx = np.zeros_like(vals, dtype=np.uint16)
                    sub[pmask] = sub_idx.astype(np.uint16)
            else:
                sub = None

            part = self._partitions[cell_idx]
            new_data = {
                "primary": primary,
                "sub": sub,
                "norms": r_norms.reshape(-1).astype(np.float32),
                "ids": np.asarray(local_ids, dtype=np.int64),
            }
            if part is None:
                self._partitions[cell_idx] = new_data
            else:
                part["primary"] = np.concatenate([part["primary"], new_data["primary"]])
                if part["sub"] is not None and new_data["sub"] is not None:
                    part["sub"] = np.concatenate([part["sub"], new_data["sub"]])
                part["norms"] = np.concatenate([part["norms"], new_data["norms"]])
                part["ids"] = np.concatenate([part["ids"], new_data["ids"]])

            for vid in local_ids.tolist():
                self._invlists[cell_idx].append(int(vid))

        self._n += v.shape[0]

    def search(self, queries, k=10, query_batch=32):
        q = np.ascontiguousarray(queries.astype(np.float32))
        nq = q.shape[0]
        q_norms = np.linalg.norm(q, axis=1, keepdims=True)
        q_normed = q / np.maximum(q_norms, 1e-8)
        q_rotated = q_normed @ self.rotation.T

        # Coarse scores
        coarse = q_normed @ self.coarse_centroids.T
        nprobe = min(self.nprobe, self.nlist)
        # top-nprobe partitions per query
        top_cells = np.argpartition(-coarse, nprobe, axis=1)[:, :nprobe]

        out_idx = np.full((nq, k), -1, dtype=np.int64)
        out_scores = np.full((nq, k), -np.inf, dtype=np.float32)

        for qi in range(nq):
            cells = top_cells[qi]
            cell_scores = coarse[qi, cells]  # exact coarse contribution
            qrot = q_rotated[qi]

            all_ids = []
            all_scores = []
            for ci, cell in enumerate(cells):
                part = self._partitions[cell]
                if part is None:
                    continue
                if self.refine_bits > 0:
                    recon = self.sub_centroids[part["primary"], part["sub"]]
                else:
                    recon = self.tq_centroids[part["primary"]]
                # residual contribution
                residual_score = (recon @ qrot) * part["norms"]
                total_score = cell_scores[ci] + residual_score
                all_ids.append(part["ids"])
                all_scores.append(total_score)

            if not all_ids:
                continue
            all_ids = np.concatenate(all_ids)
            all_scores = np.concatenate(all_scores)
            kk = min(k, len(all_scores))
            if kk == 0:
                continue
            top_local = np.argpartition(-all_scores, kk - 1)[:kk]
            top_local = top_local[np.argsort(-all_scores[top_local])]
            out_idx[qi, :kk] = all_ids[top_local]
            out_scores[qi, :kk] = all_scores[top_local]

        return out_scores, out_idx

    @property
    def memory_bytes_per_vec(self):
        return (self.bits + self.refine_bits) * self.dim / 8 + 4


def run_dataset(name, loader):
    print(f"\n{'='*60}\n  {name.upper()}\n{'='*60}")
    r = loader()
    if r is None:
        print(f"  failed to load {name}")
        return {}
    v, q, _ = r
    dim = v.shape[1]
    n = v.shape[0]
    print(f"  loaded n={n}, dim={dim}")

    print("  computing GT (FAISS Flat) ...")
    gt_idx = FAISSFlatIndex(dim) if FAISS_AVAILABLE else None
    gt_idx.add(v)
    _, gt = gt_idx.search(q, k=10)

    out = {}
    NPROBE_LIST = [20, 40]
    NLIST = 1000

    # Configurations: (bits, refine_bits, label)
    configs = [
        # standard (b-bit primary + 1-bit sign refinement, total b+1)
        (4, 1, "ivf-tq-paper-5bit"),     # 5 effective bits (canonical 4-bit config)
        (5, 1, "ivf-tq-paper-6bit"),     # 6 effective bits (canonical 5-bit config)
        (6, 1, "ivf-tq-paper-7bit"),     # 7 effective bits (canonical 6-bit config)
        # RVQ alternatives at the same total bit budget
        (3, 2, "ivf-rvq-3+2-5bit"),      # 5 effective bits, RVQ
        (4, 2, "ivf-rvq-4+2-6bit"),      # 6 effective bits, RVQ
        (3, 3, "ivf-rvq-3+3-6bit"),      # 6 effective bits, RVQ
        (4, 3, "ivf-rvq-4+3-7bit"),      # 7 effective bits, RVQ
        (5, 2, "ivf-rvq-5+2-7bit"),      # 7 effective bits, RVQ
    ]

    for bits, refine, label in configs:
        print(f"\n  [{label}] b={bits}, b'={refine}, total={bits+refine}")
        try:
            t0 = time.time()
            idx = IVFRVQTQIndex(dim=dim, nlist=NLIST, bits=bits,
                                refine_bits=refine, nprobe=20, seed=42)
            idx.train(v)
            train_t = time.time() - t0

            t0 = time.time()
            idx.add(v)
            add_t = time.time() - t0

            mem_mb = (n * idx.memory_bytes_per_vec) / (1024 * 1024)

            label_results = {
                "bits": bits, "refine": refine,
                "total_bits": bits + refine,
                "memory_mb": float(round(mem_mb, 1)),
                "train_s": round(train_t, 1),
                "add_s": round(add_t, 1),
                "by_nprobe": {},
            }

            for nprobe in NPROBE_LIST:
                idx.nprobe = nprobe
                t0 = time.time()
                _, pred = idx.search(q, k=10)
                search_t = time.time() - t0
                recall = compute_recall(gt[:, :10], pred[:, :10], 10)
                qps = q.shape[0] / max(search_t, 1e-6)
                label_results["by_nprobe"][f"np{nprobe}"] = {
                    "recall_at_10": float(recall),
                    "qps": int(qps),
                    "search_s": round(search_t, 1),
                }
                print(f"    np={nprobe}: R@10={recall:.4f}  {qps:.0f} QPS  ({search_t:.1f}s)")

            out[label] = label_results
        except Exception as e:
            import traceback
            print(f"    FAILED: {e}\n{traceback.format_exc()}")
            out[label] = {"error": str(e)}

        # incremental write
        out_path = os.path.join(os.path.dirname(__file__), "ivf_rvq_tq_results.json")
        with open(out_path, "w") as f:
            json.dump({name: out}, f, indent=2)

    return out


def main():
    results = {}
    for name, loader in [
        ("deep-1m", lambda: load_deep1m(1_000_000, 1000)),
        ("sift-1m", lambda: load_sift1m(1_000_000, 1000)),
    ]:
        results[name] = run_dataset(name, loader)
        out_path = os.path.join(os.path.dirname(__file__), "ivf_rvq_tq_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    print("\n=== Summary (matched memory) ===")
    for ds, sub in results.items():
        print(f"\n{ds}:")
        # Group by total bits
        by_total = {}
        for label, r in sub.items():
            if "error" in r:
                continue
            t = r["total_bits"]
            by_total.setdefault(t, []).append((label, r))
        for t in sorted(by_total):
            print(f"  total {t} bits:")
            entries = by_total[t]
            for label, r in entries:
                np20 = r["by_nprobe"].get("np20", {}).get("recall_at_10", float("nan"))
                np40 = r["by_nprobe"].get("np40", {}).get("recall_at_10", float("nan"))
                print(f"    {label:<30} np20={np20*100:.2f}%  np40={np40*100:.2f}%  mem={r['memory_mb']:.1f}MB")


if __name__ == "__main__":
    main()
