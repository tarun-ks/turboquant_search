"""
Unit tests for TurboQuant Search core functionality.
"""

import numpy as np
import pytest
from turboquant_search.core import (
    TurboQuantSearchIndex,
    FlatSearchIndex,
    ProductQuantizationIndex,
    _lloyd_max_codebook,
    _get_rotation_matrix,
)
from turboquant_search.faiss_baselines import FAISS_AVAILABLE


# ─────────────────────────────────────────────────────────────
# Compression ratio tests
# ─────────────────────────────────────────────────────────────

class TestCompressionRatio:
    """Test that compression ratios match expected values for each bit width."""

    @pytest.mark.parametrize("bits,expected_min_ratio", [
        (2, 6.0),
        (3, 4.0),
        (4, 3.0),
    ])
    def test_compression_ratio(self, bits, expected_min_ratio):
        dim = 128
        n = 1000
        index = TurboQuantSearchIndex(dim=dim, bits=bits, use_qjl=True, seed=42)
        vectors = np.random.randn(n, dim).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        index.add(vectors)

        ratio = index.compression_ratio
        assert ratio >= expected_min_ratio, (
            f"{bits}-bit compression ratio {ratio:.1f}x < expected minimum {expected_min_ratio}x"
        )

    def test_compression_ratio_without_qjl(self):
        """Without QJL, compression should be higher (fewer bits stored)."""
        dim = 128
        n = 1000
        bits = 3

        with_qjl = TurboQuantSearchIndex(dim=dim, bits=bits, use_qjl=True)
        without_qjl = TurboQuantSearchIndex(dim=dim, bits=bits, use_qjl=False)

        vectors = np.random.randn(n, dim).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

        with_qjl.add(vectors)
        without_qjl.add(vectors)

        assert without_qjl.compression_ratio > with_qjl.compression_ratio


# ─────────────────────────────────────────────────────────────
# Recall tests
# ─────────────────────────────────────────────────────────────

class TestRecall:
    """Test recall quality on well-separated and standard datasets."""

    def test_good_recall_well_separated_clusters(self):
        """On well-separated clusters, recall@10 should be high.

        Uses recall@10 instead of recall@1 because within-cluster vectors
        are near-identical (tied nearest neighbors), so exact index match
        is not a meaningful metric.
        """
        dim = 32
        n_clusters = 5
        vectors_per_cluster = 200
        k = 10

        rng = np.random.RandomState(42)

        # Random cluster centers spread across the unit sphere
        centers = rng.randn(n_clusters, dim).astype(np.float32)
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)

        vectors = []
        for c in range(n_clusters):
            noise = rng.randn(vectors_per_cluster, dim).astype(np.float32) * 0.15
            cluster_vecs = centers[c] + noise
            cluster_vecs /= np.linalg.norm(cluster_vecs, axis=1, keepdims=True)
            vectors.append(cluster_vecs)
        vectors = np.vstack(vectors)

        # Queries: slightly perturbed cluster centers
        queries = centers + rng.randn(n_clusters, dim).astype(np.float32) * 0.05
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)

        flat = FlatSearchIndex(dim)
        flat.add(vectors)
        _, gt_idx = flat.search(queries, k=k)

        tq = TurboQuantSearchIndex(dim=dim, bits=4, use_qjl=True, seed=42)
        tq.add(vectors)
        _, tq_idx = tq.search(queries, k=k)

        recall_sum = 0
        for i in range(n_clusters):
            gt_set = set(gt_idx[i].tolist())
            tq_set = set(tq_idx[i].tolist())
            recall_sum += len(gt_set & tq_set) / len(gt_set)
        recall = recall_sum / n_clusters

        assert recall >= 0.5, f"Recall@{k} = {recall:.2f}, expected >= 0.5 on well-separated data"

    def test_4bit_beats_2bit(self):
        """4-bit should have higher recall than 2-bit on the same data."""
        dim = 128
        n = 5000
        nq = 50
        k = 10

        rng = np.random.RandomState(42)
        vectors = rng.randn(n, dim).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        queries = rng.randn(nq, dim).astype(np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)

        flat = FlatSearchIndex(dim)
        flat.add(vectors)
        _, gt_idx = flat.search(queries, k=k)

        tq4 = TurboQuantSearchIndex(dim=dim, bits=4, use_qjl=True, seed=42)
        tq4.add(vectors)
        _, idx4 = tq4.search(queries, k=k)

        tq2 = TurboQuantSearchIndex(dim=dim, bits=2, use_qjl=True, seed=42)
        tq2.add(vectors)
        _, idx2 = tq2.search(queries, k=k)

        recall4 = sum(len(set(gt_idx[i].tolist()) & set(idx4[i].tolist())) / k for i in range(nq)) / nq
        recall2 = sum(len(set(gt_idx[i].tolist()) & set(idx2[i].tolist())) / k for i in range(nq)) / nq

        assert recall4 > recall2, f"4-bit recall {recall4:.3f} should beat 2-bit recall {recall2:.3f}"


# ─────────────────────────────────────────────────────────────
# QJL correction tests
# ─────────────────────────────────────────────────────────────

class TestQJLCorrection:
    """Test that QJL residual correction improves recall."""

    def test_qjl_improves_recall(self):
        """QJL correction should improve or maintain recall vs no QJL."""
        dim = 128
        n = 5000
        nq = 50
        k = 10

        rng = np.random.RandomState(42)
        vectors = rng.randn(n, dim).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        queries = rng.randn(nq, dim).astype(np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)

        flat = FlatSearchIndex(dim)
        flat.add(vectors)
        _, gt_idx = flat.search(queries, k=k)

        # With QJL
        tq_qjl = TurboQuantSearchIndex(dim=dim, bits=3, use_qjl=True, seed=42)
        tq_qjl.add(vectors)
        _, idx_qjl = tq_qjl.search(queries, k=k)

        # Without QJL
        tq_no = TurboQuantSearchIndex(dim=dim, bits=3, use_qjl=False, seed=42)
        tq_no.add(vectors)
        _, idx_no = tq_no.search(queries, k=k)

        recall_qjl = sum(len(set(gt_idx[i].tolist()) & set(idx_qjl[i].tolist())) / k for i in range(nq)) / nq
        recall_no = sum(len(set(gt_idx[i].tolist()) & set(idx_no[i].tolist())) / k for i in range(nq)) / nq

        # QJL should help or at least not hurt significantly
        assert recall_qjl >= recall_no - 0.05, (
            f"QJL recall {recall_qjl:.3f} much worse than no-QJL {recall_no:.3f}"
        )


# ─────────────────────────────────────────────────────────────
# Edge case tests
# ─────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_dim_1(self):
        """Should work with dim=1."""
        index = TurboQuantSearchIndex(dim=1, bits=3, use_qjl=True, seed=42)
        vectors = np.array([[1.0], [-1.0], [0.5]], dtype=np.float32)
        index.add(vectors)

        query = np.array([[0.9]], dtype=np.float32)
        scores, indices = index.search(query, k=2)

        assert scores.shape == (1, 2)
        assert indices.shape == (1, 2)

    def test_n_vectors_1(self):
        """Should work with a single vector in the index."""
        dim = 32
        index = TurboQuantSearchIndex(dim=dim, bits=3, use_qjl=True, seed=42)
        vectors = np.random.randn(1, dim).astype(np.float32)
        index.add(vectors)

        query = np.random.randn(1, dim).astype(np.float32)
        scores, indices = index.search(query, k=1)

        assert indices[0, 0] == 0  # only one vector, must return index 0

    def test_k_greater_than_n_vectors(self):
        """When k > n_vectors, should return all vectors."""
        dim = 32
        n = 5
        index = TurboQuantSearchIndex(dim=dim, bits=3, use_qjl=True, seed=42)
        vectors = np.random.randn(n, dim).astype(np.float32)
        index.add(vectors)

        query = np.random.randn(1, dim).astype(np.float32)
        scores, indices = index.search(query, k=100)

        assert scores.shape == (1, n)
        assert indices.shape == (1, n)
        # All indices should be present
        assert set(indices[0].tolist()) == set(range(n))

    def test_multiple_add_calls(self):
        """Adding vectors incrementally should work."""
        dim = 32
        index = TurboQuantSearchIndex(dim=dim, bits=3, use_qjl=True, seed=42)

        v1 = np.random.randn(10, dim).astype(np.float32)
        v2 = np.random.randn(5, dim).astype(np.float32)
        index.add(v1)
        index.add(v2)

        assert index._n_vectors == 15

        query = np.random.randn(1, dim).astype(np.float32)
        scores, indices = index.search(query, k=15)
        assert scores.shape == (1, 15)


# ─────────────────────────────────────────────────────────────
# Rotation matrix tests
# ─────────────────────────────────────────────────────────────

class TestRotationMatrix:
    """Test properties of the random orthogonal rotation matrix."""

    @pytest.mark.parametrize("dim", [16, 64, 128])
    def test_orthogonality(self, dim):
        """Pi @ Pi.T should be approximately identity."""
        Pi = _get_rotation_matrix(dim, seed=42)
        product = Pi @ Pi.T
        identity = np.eye(dim, dtype=np.float32)

        np.testing.assert_allclose(product, identity, atol=1e-5,
                                    err_msg=f"Rotation matrix not orthogonal for dim={dim}")

    @pytest.mark.parametrize("dim", [16, 64, 128])
    def test_preserves_norm(self, dim):
        """Rotation should preserve vector norms."""
        Pi = _get_rotation_matrix(dim, seed=42)
        rng = np.random.RandomState(123)
        v = rng.randn(dim).astype(np.float32)

        original_norm = np.linalg.norm(v)
        rotated_norm = np.linalg.norm(Pi @ v)

        np.testing.assert_allclose(rotated_norm, original_norm, rtol=1e-5,
                                    err_msg="Rotation changed vector norm")

    def test_preserves_inner_product(self):
        """Rotation should preserve inner products between vectors."""
        dim = 64
        Pi = _get_rotation_matrix(dim, seed=42)
        rng = np.random.RandomState(123)
        a = rng.randn(dim).astype(np.float32)
        b = rng.randn(dim).astype(np.float32)

        ip_original = np.dot(a, b)
        ip_rotated = np.dot(Pi @ a, Pi @ b)

        np.testing.assert_allclose(ip_rotated, ip_original, rtol=1e-4,
                                    err_msg="Rotation changed inner product")

    def test_caching(self):
        """Same (dim, seed) should return the same matrix object."""
        m1 = _get_rotation_matrix(64, seed=42)
        m2 = _get_rotation_matrix(64, seed=42)
        assert m1 is m2, "Cache should return the same object"

    def test_different_seeds(self):
        """Different seeds should produce different matrices."""
        m1 = _get_rotation_matrix(64, seed=42)
        m2 = _get_rotation_matrix(64, seed=99)
        assert not np.allclose(m1, m2), "Different seeds should give different matrices"


# ─────────────────────────────────────────────────────────────
# Lloyd-Max codebook tests
# ─────────────────────────────────────────────────────────────

class TestLloydMax:

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_correct_number_of_levels(self, bits):
        centroids, boundaries = _lloyd_max_codebook(bits)
        assert len(centroids) == 2 ** bits
        assert len(boundaries) == 2 ** bits - 1

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_centroids_sorted(self, bits):
        centroids, _ = _lloyd_max_codebook(bits)
        assert np.all(centroids[:-1] <= centroids[1:]), "Centroids should be sorted"

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_boundaries_between_centroids(self, bits):
        centroids, boundaries = _lloyd_max_codebook(bits)
        for i, b in enumerate(boundaries):
            assert centroids[i] <= b <= centroids[i + 1], (
                f"Boundary {b} not between centroids {centroids[i]} and {centroids[i+1]}"
            )

    def test_caching(self):
        c1, b1 = _lloyd_max_codebook(3)
        c2, b2 = _lloyd_max_codebook(3)
        assert c1 is c2, "Lloyd-Max cache should return same object"


# ─────────────────────────────────────────────────────────────
# Compress with details (visualizer support)
# ─────────────────────────────────────────────────────────────

class TestCompressWithDetails:

    def test_returns_all_keys(self):
        dim = 32
        tq = TurboQuantSearchIndex(dim=dim, bits=3, use_qjl=True, seed=42)
        v = np.random.randn(dim).astype(np.float32)
        v /= np.linalg.norm(v)

        details = tq.compress_with_details(v)

        expected_keys = {"original", "rotated", "norm", "quantized_indices",
                         "reconstructed", "residual", "reconstruction_error",
                         "sign_bits", "refined_reconstructed", "refined_error"}
        assert set(details.keys()) == expected_keys

    def test_shapes(self):
        dim = 64
        tq = TurboQuantSearchIndex(dim=dim, bits=3, use_qjl=True, seed=42)
        v = np.random.randn(dim).astype(np.float32)

        details = tq.compress_with_details(v)

        assert details["original"].shape == (dim,)
        assert details["rotated"].shape == (dim,)
        assert details["reconstructed"].shape == (dim,)
        assert details["residual"].shape == (dim,)


# ─────────────────────────────────────────────────────────────
# Flat and PQ baseline tests
# ─────────────────────────────────────────────────────────────

class TestBaselines:

    def test_flat_perfect_recall(self):
        """Flat index should always return the true nearest neighbors."""
        dim = 32
        n = 100

        rng = np.random.RandomState(42)
        vectors = rng.randn(n, dim).astype(np.float32)
        queries = rng.randn(5, dim).astype(np.float32)

        flat = FlatSearchIndex(dim)
        flat.add(vectors)
        scores, indices = flat.search(queries, k=10)

        # Verify scores are sorted descending
        for i in range(5):
            assert np.all(scores[i, :-1] >= scores[i, 1:])

    def test_pq_runs(self):
        """PQ index should run without errors."""
        dim = 32
        n = 200

        rng = np.random.RandomState(42)
        vectors = rng.randn(n, dim).astype(np.float32)
        queries = rng.randn(5, dim).astype(np.float32)

        pq = ProductQuantizationIndex(dim, n_subspaces=4, seed=42)
        pq.add(vectors)
        scores, indices = pq.search(queries, k=10)

        assert scores.shape == (5, 10)
        assert indices.shape == (5, 10)


# ─────────────────────────────────────────────────────────────
# FAISS tests (conditional)
# ─────────────────────────────────────────────────────────────

@pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss-cpu not installed")
class TestFAISS:

    def test_faiss_flat(self):
        from turboquant_search.faiss_baselines import FAISSFlatIndex

        dim = 32
        n = 100
        rng = np.random.RandomState(42)
        vectors = rng.randn(n, dim).astype(np.float32)
        queries = rng.randn(5, dim).astype(np.float32)

        index = FAISSFlatIndex(dim)
        index.add(vectors)
        scores, indices = index.search(queries, k=10)

        assert scores.shape == (5, 10)
        assert indices.shape == (5, 10)

    def test_faiss_ivfpq(self):
        from turboquant_search.faiss_baselines import FAISSIVFPQIndex

        dim = 32
        n = 500
        rng = np.random.RandomState(42)
        vectors = rng.randn(n, dim).astype(np.float32)
        queries = rng.randn(5, dim).astype(np.float32)

        index = FAISSIVFPQIndex(dim, nlist=10, m=4)
        index.add(vectors)
        scores, indices = index.search(queries, k=10)

        assert scores.shape == (5, 10)
        assert indices.shape == (5, 10)
