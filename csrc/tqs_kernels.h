#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>

namespace tqs {

struct Candidate {
    float score;
    int64_t id;
};

// ── Flat TQ search (ADC + top-k) ──
void tq_flat_search(
    const float* sub_centroids, int n_levels,
    const uint8_t* indices, const uint8_t* sign_bits,
    const float* norms, int n_db, int dim,
    const float* centroids,
    const float* q_rotated, int nq,
    bool use_sign, int k,
    float* out_scores, int64_t* out_indices);

// ── IVF partition scoring ──
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
    std::vector<std::vector<Candidate>>& candidates);

// ── Top-k selection ──
void topk_select(
    const std::vector<Candidate>& candidates,
    int k,
    float* out_scores, int64_t* out_indices);

// ── Fast scoring with precomputed combined codes ──
// combined_code[d] = indices[d]*2 + sign_bits[d] (for sign case)
// or combined_code[d] = indices[d] (no sign)
// This halves memory loads in the inner loop.
void score_partition_fast(
    const float* table,       // (dim * n_entries) precomputed ADC table
    int n_entries, int dim,
    const uint8_t* codes,     // (n_in_list, dim) combined codes
    const float* norms,       // (n_in_list,)
    int n_in_list,
    float coarse_score,
    const int64_t* ids,       // (n_in_list,)
    std::vector<Candidate>& candidates);

}  // namespace tqs
