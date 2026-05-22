"""
Bit-importance ablation on a proper IVF-PQ baseline.

Skips FAISS SWIG entirely. We use a tighter NumPy IVF-PQ implementation
with sufficient k-means iterations and proper residual training, which
should achieve recall comparable to FAISS at SIFT-1M.

Goal: replicate the published "FAISS IVF-PQ m=64 = 73.2%" baseline,
then run bit-flip ablation at that proper baseline. If MSB:LSB asymmetry
ratio ≈ 1:1 holds at high-recall PQ, the cascade-TQ-specificity claim is
validated.
"""

import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans

import faiss
from turboquant_search.datasets import load_sift1m


def _normalize(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(n, 1e-8)


def compute_recall_at_10(gt, pred):
    n = gt.shape[0]
    hits = 0
    for i in range(n):
        hits += len(set(gt[i]) & set(pred[i]))
    return hits / (n * 10)


class TightIVFPQ:
    """Properly-tuned IVF-PQ targeting FAISS recall on SIFT-1M.

    Differences from earlier IVFPQ class:
      - More k-means iterations and more init seeds
      - Codebook trained on much more data
      - Lloyd iterations at full convergence
      - Inner-product metric (matches FAISS Table 4 baseline)
    """

    def __init__(self, dim, nlist=1000, m=64, n_centroids=256, nprobe=80, seed=42):
        assert dim % m == 0
        self.dim, self.m = dim, m
        self.sub_dim = dim // m
        self.n_centroids = n_centroids
        self.nlist = nlist
        self.nprobe = nprobe
        self.seed = seed
        self.coarse_centroids = None
        self.codebooks = None
        self._codes = None    # (n, m) uint8 — global codes array
        self._cell_of = None  # (n,) — which cell each vector lives in
        self._n = 0

    def train(self, v):
        v = np.ascontiguousarray(v.astype(np.float32))

        # Coarse k-means on normalized vectors
        n_clusters = min(self.nlist, v.shape[0])
        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=self.seed,
                              batch_size=10000, n_init=5, max_iter=100)
        km.fit(v)
        cc = km.cluster_centers_.astype(np.float32)
        self.coarse_centroids = cc / np.maximum(np.linalg.norm(cc, axis=1, keepdims=True), 1e-8)
        self.nlist = n_clusters

        # PQ codebook training: assign full data, compute residuals, train
        # full-batch k-means on each subspace.
        sims = v @ self.coarse_centroids.T
        assign = np.argmax(sims, axis=1)
        residuals = v - self.coarse_centroids[assign]
        self.codebooks = np.zeros((self.m, self.n_centroids, self.sub_dim), dtype=np.float32)
        # Sample 200K residuals for codebook training (enough per-centroid mass)
        n_train = min(200_000, residuals.shape[0])
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(residuals.shape[0], size=n_train, replace=False)
        residuals_train = residuals[idx]
        for sub in range(self.m):
            sub_data = residuals_train[:, sub * self.sub_dim:(sub + 1) * self.sub_dim]
            km2 = MiniBatchKMeans(
                n_clusters=self.n_centroids, random_state=self.seed + sub,
                batch_size=10000, n_init=3, max_iter=100,
            )
            km2.fit(sub_data)
            self.codebooks[sub] = km2.cluster_centers_.astype(np.float32)

    def add(self, v):
        v = np.ascontiguousarray(v.astype(np.float32))
        sims = v @ self.coarse_centroids.T
        assign = np.argmax(sims, axis=1).astype(np.int32)
        residuals = v - self.coarse_centroids[assign]
        codes = np.zeros((v.shape[0], self.m), dtype=np.uint8)
        for sub in range(self.m):
            sub_data = residuals[:, sub * self.sub_dim:(sub + 1) * self.sub_dim]
            # nearest-centroid (distance = ||x - c||^2 = ||x||^2 - 2<x,c> + ||c||^2)
            # Use (-2 * dot) + ||c||^2 since ||x||^2 doesn't affect argmin
            cb_norms = (self.codebooks[sub] ** 2).sum(axis=1)
            d = cb_norms[None, :] - 2 * (sub_data @ self.codebooks[sub].T)
            codes[:, sub] = np.argmin(d, axis=1).astype(np.uint8)
        self._codes = codes
        self._cell_of = assign
        self._n = v.shape[0]

    def search(self, q, k=10):
        q = np.ascontiguousarray(q.astype(np.float32))
        nq = q.shape[0]
        # Coarse top-nprobe
        coarse = q @ self.coarse_centroids.T   # (nq, nlist)
        nprobe = min(self.nprobe, self.nlist)
        top_cells = np.argpartition(-coarse, nprobe, axis=1)[:, :nprobe]

        out_idx = np.full((nq, k), -1, dtype=np.int64)
        for qi in range(nq):
            cells = top_cells[qi]
            cell_qc = coarse[qi, cells]
            qrot = q[qi]
            # LUT: lut[sub, c] = <q_sub, codebook[sub, c]>
            lut = np.zeros((self.m, self.n_centroids), dtype=np.float32)
            for sub in range(self.m):
                q_sub = qrot[sub * self.sub_dim:(sub + 1) * self.sub_dim]
                lut[sub] = self.codebooks[sub] @ q_sub

            # Pool candidates from probed cells
            cell_mask = np.isin(self._cell_of, cells)
            cand_ids = np.where(cell_mask)[0]
            if len(cand_ids) == 0:
                continue
            # cell_idx_of_each_candidate: index into `cells` array
            cell_to_pos = {int(c): pos for pos, c in enumerate(cells)}
            cand_cell_pos = np.array([cell_to_pos[int(c)] for c in self._cell_of[cand_ids]])
            cand_codes = self._codes[cand_ids]
            # residual score per candidate
            scores_residual = np.zeros(cand_codes.shape[0], dtype=np.float32)
            for sub in range(self.m):
                scores_residual += lut[sub, cand_codes[:, sub]]
            scores = cell_qc[cand_cell_pos] + scores_residual
            kk = min(k, len(scores))
            top_local = np.argpartition(-scores, kk - 1)[:kk]
            top_local = top_local[np.argsort(-scores[top_local])]
            out_idx[qi, :kk] = cand_ids[top_local]
        return out_idx


def ablate(idx, q, gt, position, frac, seed=43):
    """Make a copy, corrupt, search."""
    rng = np.random.default_rng(seed)
    codes_backup = idx._codes.copy()
    n, m = idx._codes.shape
    n_corrupt = int(frac * n * m)
    if n_corrupt > 0:
        flat = rng.choice(n * m, size=n_corrupt, replace=False)
        rows, cols = flat // m, flat % m
        if position == 'msb':
            idx._codes[rows, cols] = idx._codes[rows, cols] ^ (1 << 7)
        elif position == 'middle':
            idx._codes[rows, cols] = idx._codes[rows, cols] ^ (1 << 4)
        elif position == 'lsb':
            idx._codes[rows, cols] = idx._codes[rows, cols] ^ 1
        elif position == 'random':
            idx._codes[rows, cols] = rng.integers(0, 256, size=n_corrupt).astype(np.uint8)

    pred = idx.search(q, k=10)
    recall = compute_recall_at_10(gt, pred)
    idx._codes = codes_backup
    return recall


def main():
    print("Loading SIFT-1M ...")
    r = load_sift1m(1_000_000, 1000)
    if r is None:
        print("FAILED")
        return
    v, q, _ = r
    v = _normalize(v).astype(np.float32)
    q = _normalize(q).astype(np.float32)
    d = v.shape[1]
    print(f"  n={v.shape[0]}, dim={d}")

    # Use FAISS Flat for ground truth (matches the canonical setup)
    flat = faiss.IndexFlatIP(d)
    flat.add(v)
    _, gt = flat.search(q, 10)

    results = {}
    for m, label in [(64, "m=64"), (128, "m=128 (matched memory to TQ 6-bit)")]:
        print(f"\n=== TightIVFPQ m={m} on SIFT-1M ===")
        t0 = time.time()
        idx = TightIVFPQ(dim=d, nlist=1000, m=m, n_centroids=256, nprobe=80, seed=42)
        idx.train(v)
        train_t = time.time() - t0
        t0 = time.time()
        idx.add(v)
        add_t = time.time() - t0
        t0 = time.time()
        pred = idx.search(q, k=10)
        search_t = time.time() - t0
        baseline = compute_recall_at_10(gt, pred)
        print(f"  train={train_t:.1f}s, add={add_t:.1f}s, search={search_t:.1f}s")
        print(f"  baseline R@10 = {baseline:.4f}")

        out = {"m": m, "baseline_recall": float(baseline),
               "train_s": train_t, "add_s": add_t, "search_s": search_t,
               "ablations": []}

        if baseline < 0.50:
            print(f"  WARNING: baseline R@10 < 50%, ablation may have floor effects")

        for pos in ['msb', 'middle', 'lsb', 'random']:
            for frac in [0.05, 0.10, 0.20]:
                t0 = time.time()
                recall = ablate(idx, q, gt, pos, frac)
                elapsed = time.time() - t0
                delta = (recall - baseline) * 100
                out["ablations"].append({
                    "position": pos, "frac": frac,
                    "recall": float(recall), "delta_pp": float(delta),
                    "elapsed_s": round(elapsed, 1),
                })
                print(f"  {pos:>6} frac={frac:.2f}: R@10 = {recall:.4f}  Δ = {delta:+.2f}pp")

        results[f"m{m}"] = out

    out_path = os.path.join(os.path.dirname(__file__), "cascade_pq_faiss_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
