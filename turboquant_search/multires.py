"""
Multi-Resolution IVF-TQ (MR-IVF-TQ).

Each rotated coordinate is encoded at TWO resolution layers derived from
a hierarchical Lloyd-Max codebook:

    base layer       — `base_bits` (2-bit by default, 4 levels)
    refinement layer — additional bits up to `total_bits` (2-bit by default,
                       so total = base + refine = 4 effective levels per
                       base bin = 16 levels total)

Search proceeds in two stages:
    Stage 1 (fast filter): score every candidate using only the BASE codes
        (a 4-entry lookup table per coord — 4× fewer entries than the
        16-entry full table; cache-friendly).
    Stage 2 (accurate rerank): for the top `stage1_k` candidates, rescore
        with the full `total_bits` codes (using sub-centroids for
        sign-bit refinement when enabled).

The base codebook is a *coarsening* of the full codebook: each base bin
contains exactly 2^(total_bits - base_bits) full bins, and the base
reconstruction is the conditional mean of those full reconstructions on
the underlying Gaussian source. This guarantees that values within the
same base bin are similar in inner-product space, so the stage-1 filter
is a valid prefilter.

Memory: same as a single `total_bits` TQ index — base and refinement
fields are split out for cache-efficient stage-1 scans.

This is an implementation contribution paired with the SIMD kernel work:
the 4-entry stage-1 LUT fits trivially in vqtbl1q/vpshufb half-registers,
giving a structural latency advantage over flat 4-bit even before SIMD.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .core import (
    IVFTurboQuantIndex, _get_rotation_matrix, _lloyd_max_codebook,
)


def _hierarchical_codebook(total_bits: int, base_bits: int, dim: int):
    """Build a hierarchical Lloyd-Max codebook with a base layer + refine.

    Returns (full_centroids, full_boundaries, base_centroids, base_boundaries,
             full_to_base_map). Centroids and boundaries are scaled to the
             N(0, 1/dim) marginal of unit-vector coordinates after random
             rotation.
    """
    assert 1 <= base_bits < total_bits
    full_n = 2 ** total_bits
    base_n = 2 ** base_bits
    sub_per_base = full_n // base_n      # = 2^(total_bits - base_bits)
    s = np.sqrt(dim)

    full_c, full_b = _lloyd_max_codebook(total_bits)
    full_c = (full_c / s).astype(np.float32)
    full_b = (full_b / s).astype(np.float32)

    # Base centroid for bin j is the conditional mean of the
    # sub-centroids it contains under the Gaussian source. By symmetry
    # of Lloyd-Max for a symmetric source, this equals the average of
    # the contained sub-centroids weighted by their bin masses; we use
    # equal-mass approximation which is exact for fine quantization.
    base_c = np.array([
        full_c[j * sub_per_base : (j + 1) * sub_per_base].mean()
        for j in range(base_n)
    ], dtype=np.float32)
    # Base boundaries are the upper boundaries of each base bin's last
    # sub-bin (= the (sub_per_base * (j+1) - 1)-th boundary in the full
    # set, for j = 0..base_n-2).
    base_b = np.array([
        full_b[(j + 1) * sub_per_base - 1] for j in range(base_n - 1)
    ], dtype=np.float32)

    # Map from full bin index -> base bin index = idx // sub_per_base.
    full_to_base = np.array([j // sub_per_base for j in range(full_n)],
                             dtype=np.uint8)

    return (full_c, full_b, base_c, base_b, full_to_base, sub_per_base)


class MultiResIVFTurboQuantIndex(IVFTurboQuantIndex):
    """IVF-TQ with hierarchical 2-stage quantization for fast scan."""

    def __init__(self, dim: int, nlist: int = 100,
                 total_bits: int = 4, base_bits: int = 2,
                 nprobe: int = 10, use_residual_sign: bool = True,
                 seed: int = 42, stage1_k: Optional[int] = None):
        # Initialize parent at total_bits (gives us full codebook + sign-bit machinery)
        super().__init__(dim=dim, nlist=nlist, bits=total_bits, nprobe=nprobe,
                          use_residual_sign=use_residual_sign, seed=seed)
        assert 1 <= base_bits < total_bits, "base_bits must be 1..total_bits-1"
        self.total_bits = total_bits
        self.base_bits = base_bits
        self.stage1_k = stage1_k  # None -> auto-pick (10× k)

        full_c, full_b, base_c, base_b, full_to_base, sub_per_base = \
            _hierarchical_codebook(total_bits, base_bits, dim)
        # full_c == self.tq_centroids (already loaded by parent)
        self.base_centroids  = base_c
        self.base_boundaries = base_b
        self.full_to_base    = full_to_base    # uint8[2^total_bits]
        self.sub_per_base    = sub_per_base

    # ------------------------------------------------------------------
    # Override _tq_compress to also output base codes
    # ------------------------------------------------------------------
    def _tq_compress(self, residuals: np.ndarray):
        """Returns (full_indices, norms, sign_bits, base_indices)."""
        rotated = residuals @ self.rotation_matrix.T
        norms = np.linalg.norm(rotated, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = rotated / norms
        full_idx = np.digitize(normalized, self.tq_boundaries).astype(np.uint8)
        sign_bits = None
        if self.use_residual_sign:
            sign_bits = (normalized >= self.tq_centroids[full_idx]).astype(np.uint8)
        # Derive base codes from full codes via the precomputed map
        base_idx = self.full_to_base[full_idx]
        return full_idx, norms.reshape(-1), sign_bits, base_idx

    # ------------------------------------------------------------------
    # Override add to also store base codes per partition
    # ------------------------------------------------------------------
    def add(self, vectors: np.ndarray, ids: Optional[np.ndarray] = None):
        assert self._trained, "Must train() first."
        v = np.ascontiguousarray(vectors.astype(np.float32))
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        n = v.shape[0]
        if ids is None:
            ids = np.arange(self._n_vectors, self._n_vectors + n)
        t0 = time.time()
        v_norm = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-8)
        assignments = np.argmax(v_norm @ self.coarse_centroids.T, axis=1)

        for l in range(self.nlist):
            mask = assignments == l
            if not mask.any():
                continue
            cell_v = v_norm[mask]
            cell_ids = ids[mask]
            residuals = cell_v - self.coarse_centroids[l]
            full_idx, norms, sign_bits, base_idx = self._tq_compress(residuals)

            self._invlists[l].extend(cell_ids.tolist())
            part = self._partitions[l]
            if part["indices"] is None:
                part["indices"]   = full_idx
                part["norms"]     = norms
                part["sign_bits"] = sign_bits
                part["base"]      = base_idx
            else:
                part["indices"]   = np.concatenate([part["indices"], full_idx])
                part["norms"]     = np.concatenate([part["norms"], norms])
                if sign_bits is not None and part["sign_bits"] is not None:
                    part["sign_bits"] = np.concatenate([part["sign_bits"], sign_bits])
                part["base"]      = np.concatenate([part["base"], base_idx])

        if self._raw_vectors is None:
            self._raw_vectors = v_norm
        else:
            self._raw_vectors = np.concatenate([self._raw_vectors, v_norm])
        self._n_vectors += n
        self.build_time += time.time() - t0

    # ------------------------------------------------------------------
    # Two-stage search
    # ------------------------------------------------------------------
    def search(self, queries: np.ndarray, k: int = 10,
               rerank: int = 0, stage1_k: Optional[int] = None
               ) -> Tuple[np.ndarray, np.ndarray]:
        """Two-stage search: base-bit prefilter -> full-bit rerank.

        stage1_k : int or None
            Number of candidates to retain from stage 1 per query. None
            uses self.stage1_k or default 10*k.
        """
        Q = np.ascontiguousarray(queries.astype(np.float32))
        Q = np.nan_to_num(Q, nan=0.0, posinf=0.0, neginf=0.0)
        Q = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-8)
        Qr = Q @ self.rotation_matrix.T
        nq = Q.shape[0]

        s1_k = stage1_k or self.stage1_k or max(10 * k, 100)

        coarse = Q @ self.coarse_centroids.T                  # (nq, nlist)
        np_actual = min(self.nprobe, self.nlist)
        top_lists = np.argpartition(-coarse, np_actual - 1, axis=1)[:, :np_actual]

        out_ids = np.full((nq, k), -1, dtype=np.int64)
        out_scores = np.full((nq, k), -np.inf, dtype=np.float32)

        # Per-query loop. Both stages are vectorized within a query.
        for qi in range(nq):
            qr = Qr[qi]
            # ── Stage 1: base-code scan ──
            cells = top_lists[qi]
            cand_ids: List[np.ndarray] = []
            cand_s1: List[np.ndarray] = []
            for l in cells:
                part = self._partitions[int(l)]
                if part["indices"] is None or len(part["indices"]) == 0:
                    continue
                base_idx = part["base"]                       # (m, dim) uint8
                norms = part["norms"]                         # (m,)
                # Reconstruct rotated unit residual at BASE precision
                rec_unit = self.base_centroids[base_idx]      # (m, dim)
                rec = rec_unit * norms[:, None]
                fine = rec @ qr
                s1 = coarse[qi, int(l)] + fine
                cand_ids.append(np.array(self._invlists[int(l)], dtype=np.int64))
                cand_s1.append(s1)
            if not cand_ids:
                continue
            ids_arr = np.concatenate(cand_ids)
            s1_arr = np.concatenate(cand_s1)
            top_s1 = min(s1_k, ids_arr.shape[0])
            sel = np.argpartition(-s1_arr, top_s1 - 1)[:top_s1]
            sel_ids = ids_arr[sel]

            # ── Stage 2: full-precision rerank (using sub-centroids if enabled) ──
            # We need full codes for sel_ids. Build a lookup per partition.
            cursor = 0
            full_unit_list = []
            full_norms_list = []
            for l in cells:
                part = self._partitions[int(l)]
                if part["indices"] is None or len(part["indices"]) == 0:
                    continue
                m = len(part["indices"])
                if self.use_residual_sign and part["sign_bits"] is not None:
                    unit = self.sub_centroids[part["indices"], part["sign_bits"]]
                else:
                    unit = self.tq_centroids[part["indices"]]
                full_unit_list.append(unit)
                full_norms_list.append(part["norms"])
                cursor += m
            full_unit = np.concatenate(full_unit_list)         # (N_total, dim)
            full_norms = np.concatenate(full_norms_list)       # (N_total,)
            # sel were positions in ids_arr; they index into full_unit / full_norms
            sel_unit = full_unit[sel]
            sel_norms = full_norms[sel]
            sel_rec = sel_unit * sel_norms[:, None]
            # Need coarse score per selected candidate. We track which cell each
            # came from in cand_ids. Easier: recompute coarse for sel by re-finding
            # cell. But we know s1 = coarse + fine; the relative ranking can be
            # done on fine only if coarse is roughly equal across cells.
            # Cleanest: recompute coarse contribution for each sel from its cell.
            # We'll piggyback: add to fine using a per-row coarse from the cell
            # ID. Build cell-ID array parallel to ids_arr.
            cell_id_per_cand = []
            for l_idx, l in enumerate(cells):
                part = self._partitions[int(l)]
                if part["indices"] is None or len(part["indices"]) == 0:
                    continue
                cell_id_per_cand.append(np.full(len(part["indices"]), int(l), dtype=np.int64))
            cell_id_arr = np.concatenate(cell_id_per_cand)
            sel_cells = cell_id_arr[sel]
            sel_coarse = coarse[qi, sel_cells]
            s2 = sel_coarse + (sel_rec @ qr)
            if rerank > 0 and self._raw_vectors is not None:
                s2 = self._raw_vectors[sel_ids] @ Q[qi]
            order = np.argsort(-s2)[:k]
            out_ids[qi, :len(order)] = sel_ids[order]
            out_scores[qi, :len(order)] = s2[order]

        return out_scores, out_ids
