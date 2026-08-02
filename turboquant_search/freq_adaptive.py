"""
FA-IVF-TQ — Frequency-Adaptive IVF-TQ.

Per-vector variable bit budget driven by observed query frequency.
Vectors that are retrieved often ("hot") are stored at higher precision;
vectors that are rarely retrieved ("cold") are stored at lower precision.

Crucially: this is *only structurally possible with TQ*, because
re-encoding any vector at any precision is O(d²) work — O(d²) for the
dense rotation (d×d matrix multiply) plus O(d) for Lloyd–Max scalar
requantization — and no codebook retraining is required. PQ-family
methods would need to retrain a codebook to change a vector's bit budget,
defeating the purpose.  Note: O(d²) is tolerable at d=128 but becomes
a real bottleneck at d=768+; a structured fast transform (e.g., Hadamard)
is not currently implemented.

Implementation: maintain one IVF-TQ sub-index per supported bit width
(default {2, 4, 6}). All sub-indexes share the IVF coarse centroids.
Each vector lives in exactly one sub-index at any time. The `adapt`
method migrates a vector by re-encoding (data-independent) and moving
it between sub-indexes.

Search: query every sub-index in parallel, merge candidates by score.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .core import IVFTurboQuantIndex


class FrequencyAdaptiveIVFTQ:
    """IVF-TQ with per-vector variable precision."""

    def __init__(self, dim: int, nlist: int = 1000, nprobe: int = 20,
                 seed: int = 42, bit_widths: Tuple[int, ...] = (2, 4, 6),
                 default_bits: int = 4, use_residual_sign: bool = True):
        self.dim = dim
        self.nlist = nlist
        self.nprobe = nprobe
        self.seed = seed
        self.bit_widths = tuple(sorted(bit_widths))
        self.default_bits = default_bits
        self.use_residual_sign = use_residual_sign
        assert default_bits in self.bit_widths

        self._sub: Dict[int, IVFTurboQuantIndex] = {}
        self._vec_bits: Dict[int, int] = {}     # vid -> bit_width
        self._vec_data: Dict[int, np.ndarray] = {}  # vid -> raw vector (kept for migration)
        self._hit_counter: Dict[int, int] = {}  # vid -> retrieval count
        self._n_vectors = 0
        self._trained = False

    def train(self, vectors: np.ndarray):
        """Train ONE sub-index for k-means centroids; share with all."""
        master = IVFTurboQuantIndex(
            dim=self.dim, nlist=self.nlist, bits=self.default_bits,
            nprobe=self.nprobe, use_residual_sign=self.use_residual_sign,
            seed=self.seed,
        )
        master.train(vectors)
        # Build sister sub-indexes that share the master's coarse centroids
        for b in self.bit_widths:
            sub = IVFTurboQuantIndex(
                dim=self.dim, nlist=master.nlist, bits=b,
                nprobe=self.nprobe,
                use_residual_sign=self.use_residual_sign,
                seed=self.seed,
            )
            # Share the master's IVF centroids
            sub.coarse_centroids = master.coarse_centroids
            sub.nlist = master.nlist
            sub._invlists = [[] for _ in range(sub.nlist)]
            sub._partitions = [
                {"indices": None, "norms": None, "sign_bits": None, "codes": None}
                for _ in range(sub.nlist)
            ]
            sub._trained = True
            self._sub[b] = sub
        self._trained = True

    def add(self, vectors: np.ndarray,
            bit_widths_per_vec: Optional[np.ndarray] = None):
        """Add vectors at specified per-vector bit widths.
        If bit_widths_per_vec is None, all use default_bits."""
        assert self._trained, "train() first"
        n = vectors.shape[0]
        v = np.ascontiguousarray(vectors.astype(np.float32))
        v_norm = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-8)

        ids = np.arange(self._n_vectors, self._n_vectors + n)
        if bit_widths_per_vec is None:
            bit_widths_per_vec = np.full(n, self.default_bits, dtype=np.int64)
        assert bit_widths_per_vec.shape[0] == n

        for b in self.bit_widths:
            mask = bit_widths_per_vec == b
            if not mask.any():
                continue
            sub = self._sub[b]
            cell_ids = ids[mask]
            cell_vecs = v_norm[mask]
            sub.add(cell_vecs, ids=cell_ids)

        for i, vid in enumerate(ids):
            vid_int = int(vid)
            self._vec_bits[vid_int] = int(bit_widths_per_vec[i])
            self._vec_data[vid_int] = v_norm[i].copy()
            self._hit_counter[vid_int] = 0
        self._n_vectors += n

    def search(self, queries: np.ndarray, k: int = 10,
               cand_per_sub: Optional[int] = None,
               update_hits: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Search each sub-index for cand_per_sub candidates, merge by score."""
        Q = np.ascontiguousarray(queries.astype(np.float32))
        Q = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-8)
        nq = Q.shape[0]

        cps = cand_per_sub if cand_per_sub else max(k * 5, 50)

        # Search each sub-index
        sub_results = {}
        for b, sub in self._sub.items():
            sc, ids = sub.search(Q, k=cps, rerank=0)
            sub_results[b] = (sc, ids)

        out_ids = np.full((nq, k), -1, dtype=np.int64)
        out_sc = np.full((nq, k), -np.inf, dtype=np.float32)
        for qi in range(nq):
            all_ids: List[np.ndarray] = []
            all_sc: List[np.ndarray] = []
            for b in self.bit_widths:
                sc, ids = sub_results[b]
                # Filter -1 padding
                mask = ids[qi] >= 0
                all_ids.append(ids[qi][mask])
                all_sc.append(sc[qi][mask])
            if not all_ids:
                continue
            ids_arr = np.concatenate(all_ids)
            sc_arr = np.concatenate(all_sc)
            # Dedup (shouldn't happen in our setup but defensive)
            seen = set()
            keep = []
            for i in np.argsort(-sc_arr):
                vid = int(ids_arr[i])
                if vid in seen:
                    continue
                seen.add(vid)
                keep.append(i)
                if len(keep) >= k:
                    break
            for j, i in enumerate(keep):
                out_ids[qi, j] = ids_arr[i]
                out_sc[qi, j] = sc_arr[i]
                if update_hits:
                    self._hit_counter[int(ids_arr[i])] = self._hit_counter.get(int(ids_arr[i]), 0) + 1
        return out_sc, out_ids

    def adapt(self, hot_fraction: float = 0.02, cold_fraction: float = 0.50,
              hot_bits: int = 6, cold_bits: int = 2) -> Dict:
        """Promote top-N hot vectors to hot_bits, demote bottom-M cold to cold_bits.

        Migration cost: O(|migrated| · d) re-encoding only — no codebook
        retraining (TQ's data-independence in action).
        """
        t0 = time.time()
        n = len(self._vec_bits)
        if n == 0:
            return {"migrated": 0, "duration_s": 0.0}
        ids_sorted = sorted(self._hit_counter.items(), key=lambda x: x[1], reverse=True)
        hot_set = set(int(i) for i, _ in ids_sorted[:int(n * hot_fraction)])
        cold_set = set(int(i) for i, _ in ids_sorted[-int(n * cold_fraction):])
        # Conflict: a vector can't be both hot and cold
        cold_set -= hot_set

        migrated = 0
        for vid in hot_set | cold_set:
            target = hot_bits if vid in hot_set else cold_bits
            current = self._vec_bits.get(vid)
            if current == target:
                continue
            # Remove from current sub-index
            self._remove(vid, current)
            # Re-encode and add to target
            v = self._vec_data[vid]
            self._sub[target].add(v.reshape(1, -1), ids=np.array([vid]))
            self._vec_bits[vid] = target
            migrated += 1

        return {"migrated": migrated, "duration_s": time.time() - t0,
                "hot_count": len(hot_set), "cold_count": len(cold_set)}

    def _remove(self, vid: int, bits: int):
        """Remove vector vid from sub-index `bits`. O(N) in cell size."""
        sub = self._sub[bits]
        for cell_idx, ids_list in enumerate(sub._invlists):
            if vid in ids_list:
                pos = ids_list.index(vid)
                ids_list.pop(pos)
                part = sub._partitions[cell_idx]
                # Mask out position pos in arrays
                for key in ("indices", "norms", "sign_bits", "codes"):
                    if part[key] is not None:
                        part[key] = np.delete(part[key], pos, axis=0)
                sub._n_vectors -= 1
                # Invalidate C++ cache
                if hasattr(sub, "_cpp_partition_cache"):
                    del sub._cpp_partition_cache
                if hasattr(sub, "_cpp_partition_cache_n"):
                    del sub._cpp_partition_cache_n
                return
        raise KeyError(f"vid {vid} not found in sub-index {bits}")

    @property
    def memory_bytes(self) -> int:
        """Total memory of compressed codes only (excluding raw vectors)."""
        total = 0
        for b in self.bit_widths:
            total += self._sub[b].memory_bytes
        return total

    def memory_breakdown(self) -> Dict[int, Dict]:
        """Per-bit-width memory and vector count."""
        out = {}
        for b in self.bit_widths:
            n = self._sub[b]._n_vectors
            mem = self._sub[b].memory_bytes
            out[b] = {"n_vectors": n, "memory_bytes": mem,
                       "memory_mb": mem / (1024 * 1024)}
        return out

    @property
    def avg_bits_per_vector(self) -> float:
        if self._n_vectors == 0:
            return 0.0
        total_bits = sum(b * self._sub[b]._n_vectors * self.dim
                          for b in self.bit_widths)
        return total_bits / (self._n_vectors * self.dim)
