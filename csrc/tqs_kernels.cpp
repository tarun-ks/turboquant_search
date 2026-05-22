#include "tqs_kernels.h"
#include <algorithm>
#include <numeric>
#include <cstring>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define TQS_HAS_NEON 1
#else
#define TQS_HAS_NEON 0
#endif

namespace tqs {

// ═══════════════════════════════════════════════════════════════
// ADC table builder: table[d * n_entries + code] = q[d] * centroid[code]
// ═══════════════════════════════════════════════════════════════
static inline void build_adc_table(
    const float* q_row, int dim,
    const float* sub_centroids, int n_levels,
    const float* centroids,
    bool use_sign,
    float* table, int n_entries)
{
    if (use_sign) {
        for (int d = 0; d < dim; ++d) {
            float qd = q_row[d];
            float* row = table + d * n_entries;
            for (int l = 0; l < n_levels; ++l) {
                row[l * 2 + 0] = qd * sub_centroids[l * 2 + 0];
                row[l * 2 + 1] = qd * sub_centroids[l * 2 + 1];
            }
        }
    } else {
        for (int d = 0; d < dim; ++d) {
            float qd = q_row[d];
            float* row = table + d * n_entries;
            for (int l = 0; l < n_levels; ++l) {
                row[l] = qd * centroids[l];
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Fast inner loop: score one vector using combined codes + ADC table
// Uses NEON on ARM with 2 independent accumulators to break the
// dependency chain on `acc` and let the CPU pipeline overlap fadds.
// Theoretical 2× throughput vs single-accumulator on this loop.
// ═══════════════════════════════════════════════════════════════
static inline float score_one_vector(
    const float* __restrict__ table, int n_entries,
    const uint8_t* __restrict__ codes, int dim)
{
#if TQS_HAS_NEON
    // Two independent accumulators (no data dep between them).
    // Process 8 dims at a time → 2 SIMD adds per iteration that the
    // CPU can issue back-to-back without waiting for either one.
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    int d = 0;
    for (; d + 7 < dim; d += 8) {
        float v0 = table[(d + 0) * n_entries + codes[d + 0]];
        float v1 = table[(d + 1) * n_entries + codes[d + 1]];
        float v2 = table[(d + 2) * n_entries + codes[d + 2]];
        float v3 = table[(d + 3) * n_entries + codes[d + 3]];
        float v4 = table[(d + 4) * n_entries + codes[d + 4]];
        float v5 = table[(d + 5) * n_entries + codes[d + 5]];
        float v6 = table[(d + 6) * n_entries + codes[d + 6]];
        float v7 = table[(d + 7) * n_entries + codes[d + 7]];
        float32x4_t vals0 = {v0, v1, v2, v3};
        float32x4_t vals1 = {v4, v5, v6, v7};
        acc0 = vaddq_f32(acc0, vals0);   // independent of acc1
        acc1 = vaddq_f32(acc1, vals1);   // independent of acc0
    }
    // Tail: 4-wide
    for (; d + 3 < dim; d += 4) {
        float v0 = table[(d + 0) * n_entries + codes[d + 0]];
        float v1 = table[(d + 1) * n_entries + codes[d + 1]];
        float v2 = table[(d + 2) * n_entries + codes[d + 2]];
        float v3 = table[(d + 3) * n_entries + codes[d + 3]];
        float32x4_t vals = {v0, v1, v2, v3};
        acc0 = vaddq_f32(acc0, vals);
    }
    // Final reduction
    float32x4_t acc = vaddq_f32(acc0, acc1);
    float sum = vaddvq_f32(acc);
    for (; d < dim; ++d) {
        sum += table[d * n_entries + codes[d]];
    }
    return sum;
#else
    // Scalar fallback with 4 parallel accumulators
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    int d = 0;
    for (; d + 3 < dim; d += 4) {
        sum0 += table[(d + 0) * n_entries + codes[d + 0]];
        sum1 += table[(d + 1) * n_entries + codes[d + 1]];
        sum2 += table[(d + 2) * n_entries + codes[d + 2]];
        sum3 += table[(d + 3) * n_entries + codes[d + 3]];
    }
    float sum = sum0 + sum1 + sum2 + sum3;
    for (; d < dim; ++d) {
        sum += table[d * n_entries + codes[d]];
    }
    return sum;
#endif
}

// ═══════════════════════════════════════════════════════════════
// Candidate-parallel inner loop: score 4 vectors at once.
// Holds 4 independent scalar accumulators across the dim loop;
// for each dim, 4 INDEPENDENT loads from different rows allow the
// CPU to overlap memory latency. Final result is packed into a
// float32x4 and returned via output pointer.
// ═══════════════════════════════════════════════════════════════
static inline void score_4_vectors(
    const float* __restrict__ table, int n_entries,
    const uint8_t* __restrict__ codes_0,
    const uint8_t* __restrict__ codes_1,
    const uint8_t* __restrict__ codes_2,
    const uint8_t* __restrict__ codes_3,
    int dim,
    float* __restrict__ out)
{
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (int d = 0; d < dim; ++d) {
        const float* row = table + d * n_entries;
        s0 += row[codes_0[d]];   // 4 independent loads;
        s1 += row[codes_1[d]];   // CPU can pipeline aggressively
        s2 += row[codes_2[d]];
        s3 += row[codes_3[d]];
    }
    out[0] = s0; out[1] = s1; out[2] = s2; out[3] = s3;
}

// ═══════════════════════════════════════════════════════════════
// Score an entire partition with combined codes (fast path).
// Uses candidate-parallel scoring for the bulk (4 candidates per call,
// independent accumulators), and the multi-accumulator scalar path
// for the tail.
// ═══════════════════════════════════════════════════════════════
void score_partition_fast(
    const float* table, int n_entries, int dim,
    const uint8_t* codes, const float* norms,
    int n_in_list,
    float coarse_score,
    const int64_t* ids,
    std::vector<Candidate>& candidates)
{
    int j = 0;
    float dots[4];
    for (; j + 3 < n_in_list; j += 4) {
        score_4_vectors(
            table, n_entries,
            codes + static_cast<size_t>(j + 0) * dim,
            codes + static_cast<size_t>(j + 1) * dim,
            codes + static_cast<size_t>(j + 2) * dim,
            codes + static_cast<size_t>(j + 3) * dim,
            dim, dots);
        candidates.push_back({dots[0] * norms[j + 0] + coarse_score, ids[j + 0]});
        candidates.push_back({dots[1] * norms[j + 1] + coarse_score, ids[j + 1]});
        candidates.push_back({dots[2] * norms[j + 2] + coarse_score, ids[j + 2]});
        candidates.push_back({dots[3] * norms[j + 3] + coarse_score, ids[j + 3]});
    }
    for (; j < n_in_list; ++j) {
        float dot = score_one_vector(
            table, n_entries,
            codes + static_cast<size_t>(j) * dim, dim);
        candidates.push_back({dot * norms[j] + coarse_score, ids[j]});
    }
}

// ═══════════════════════════════════════════════════════════════
// Top-k: nth_element + sort (O(n) partition, O(k log k) sort)
// ═══════════════════════════════════════════════════════════════
static void topk_from_scores(
    const float* scores, int n, int k,
    float* out_scores, int64_t* out_indices)
{
    std::vector<int64_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);

    int actual_k = std::min(k, n);
    if (actual_k >= n) {
        std::sort(idx.begin(), idx.end(),
                  [scores](int64_t a, int64_t b) { return scores[a] > scores[b]; });
    } else {
        std::nth_element(idx.begin(), idx.begin() + actual_k, idx.end(),
                         [scores](int64_t a, int64_t b) { return scores[a] > scores[b]; });
        std::sort(idx.begin(), idx.begin() + actual_k,
                  [scores](int64_t a, int64_t b) { return scores[a] > scores[b]; });
    }

    for (int i = 0; i < actual_k; ++i) {
        out_scores[i] = scores[idx[i]];
        out_indices[i] = idx[i];
    }
    for (int i = actual_k; i < k; ++i) {
        out_scores[i] = -1e30f;
        out_indices[i] = -1;
    }
}

// ═══════════════════════════════════════════════════════════════
// Flat TQ search with ADC
// ═══════════════════════════════════════════════════════════════
void tq_flat_search(
    const float* sub_centroids, int n_levels,
    const uint8_t* indices, const uint8_t* sign_bits,
    const float* norms, int n_db, int dim,
    const float* centroids,
    const float* q_rotated, int nq,
    bool use_sign, int k,
    float* out_scores, int64_t* out_indices)
{
    int n_entries = use_sign ? n_levels * 2 : n_levels;

    #pragma omp parallel for schedule(dynamic)
    for (int q = 0; q < nq; ++q) {
        std::vector<float> table(dim * n_entries);
        build_adc_table(
            q_rotated + static_cast<size_t>(q) * dim, dim,
            sub_centroids, n_levels, centroids,
            use_sign, table.data(), n_entries);

        // Build combined codes on the fly for flat search
        std::vector<float> scores(n_db);
        if (use_sign) {
            for (int j = 0; j < n_db; ++j) {
                float sum = 0.0f;
                const uint8_t* idx_row = indices + static_cast<size_t>(j) * dim;
                const uint8_t* sign_row = sign_bits + static_cast<size_t>(j) * dim;
                for (int d = 0; d < dim; ++d) {
                    sum += table[d * n_entries + idx_row[d] * 2 + sign_row[d]];
                }
                scores[j] = sum * norms[j];
            }
        } else {
            for (int j = 0; j < n_db; ++j) {
                float sum = 0.0f;
                const uint8_t* idx_row = indices + static_cast<size_t>(j) * dim;
                for (int d = 0; d < dim; ++d) {
                    sum += table[d * n_entries + idx_row[d]];
                }
                scores[j] = sum * norms[j];
            }
        }

        topk_from_scores(
            scores.data(), n_db, k,
            out_scores + static_cast<size_t>(q) * k,
            out_indices + static_cast<size_t>(q) * k);
    }
}

// Kept for backward compatibility
void ivf_partition_score(
    const float* sub_centroids, int n_levels,
    const uint8_t* part_indices, const uint8_t* part_sign_bits,
    const float* part_norms, int n_in_list, int dim,
    const float* centroids,
    const float* q_rotated, int nq,
    const int* q_idx_list, int n_queries_for_list,
    const float* coarse_scores,
    bool use_sign,
    const int64_t* list_ids,
    std::vector<std::vector<Candidate>>& candidates)
{
    int n_entries = use_sign ? n_levels * 2 : n_levels;

    for (int qi = 0; qi < n_queries_for_list; ++qi) {
        int q = q_idx_list[qi];
        float coarse = coarse_scores[qi];
        const float* q_row = q_rotated + static_cast<size_t>(q) * dim;

        std::vector<float> table(dim * n_entries);
        build_adc_table(q_row, dim, sub_centroids, n_levels, centroids,
                        use_sign, table.data(), n_entries);

        for (int j = 0; j < n_in_list; ++j) {
            float sum = 0.0f;
            const uint8_t* idx_row = part_indices + static_cast<size_t>(j) * dim;
            if (use_sign) {
                const uint8_t* sign_row = part_sign_bits + static_cast<size_t>(j) * dim;
                for (int d = 0; d < dim; ++d) {
                    sum += table[d * n_entries + idx_row[d] * 2 + sign_row[d]];
                }
            } else {
                for (int d = 0; d < dim; ++d) {
                    sum += table[d * n_entries + idx_row[d]];
                }
            }
            candidates[q].push_back({sum * part_norms[j] + coarse, list_ids[j]});
        }
    }
}

void topk_select(
    const std::vector<Candidate>& cands, int k,
    float* out_scores, int64_t* out_indices)
{
    int n = static_cast<int>(cands.size());
    if (n == 0) {
        for (int i = 0; i < k; ++i) {
            out_scores[i] = -1e30f;
            out_indices[i] = -1;
        }
        return;
    }

    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);

    int actual_k = std::min(k, n);
    std::nth_element(idx.begin(), idx.begin() + actual_k, idx.end(),
                     [&cands](int a, int b) { return cands[a].score > cands[b].score; });
    std::sort(idx.begin(), idx.begin() + actual_k,
              [&cands](int a, int b) { return cands[a].score > cands[b].score; });

    for (int i = 0; i < actual_k; ++i) {
        out_scores[i] = cands[idx[i]].score;
        out_indices[i] = cands[idx[i]].id;
    }
    for (int i = actual_k; i < k; ++i) {
        out_scores[i] = -1e30f;
        out_indices[i] = -1;
    }
}

}  // namespace tqs
