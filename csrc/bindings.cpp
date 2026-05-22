#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "tqs_kernels.h"
#include <vector>

namespace py = pybind11;

// ── Flat TQ search ──
py::tuple tq_flat_search_py(
    py::array_t<float, py::array::c_style> sub_centroids,
    py::array_t<uint8_t, py::array::c_style> indices,
    py::array_t<uint8_t, py::array::c_style> sign_bits,
    py::array_t<float, py::array::c_style> norms,
    py::array_t<float, py::array::c_style> centroids,
    py::array_t<float, py::array::c_style> q_rotated,
    bool use_sign,
    int k)
{
    auto idx_buf = indices.request();
    auto norms_buf = norms.request();
    auto q_buf = q_rotated.request();

    int n_db = static_cast<int>(idx_buf.shape[0]);
    int dim = static_cast<int>(idx_buf.shape[1]);
    int nq = static_cast<int>(q_buf.shape[0]);

    auto sc_buf = sub_centroids.request();
    auto cen_buf = centroids.request();
    int n_levels = use_sign ? static_cast<int>(sc_buf.shape[0])
                            : static_cast<int>(cen_buf.shape[0]);

    auto out_scores = py::array_t<float>({nq, k});
    auto out_indices = py::array_t<int64_t>({nq, k});

    tqs::tq_flat_search(
        static_cast<const float*>(sc_buf.ptr), n_levels,
        static_cast<const uint8_t*>(idx_buf.ptr),
        use_sign ? static_cast<const uint8_t*>(sign_bits.request().ptr) : nullptr,
        static_cast<const float*>(norms_buf.ptr),
        n_db, dim,
        static_cast<const float*>(cen_buf.ptr),
        static_cast<const float*>(q_buf.ptr),
        nq, use_sign, k,
        static_cast<float*>(out_scores.request().ptr),
        static_cast<int64_t*>(out_indices.request().ptr));

    return py::make_tuple(out_scores, out_indices);
}

// Inline ADC table builder (used from the OpenMP region)
static inline void build_adc_table_inline(
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

// ── Pre-extracted partition data ──
struct PartitionInfo {
    const uint8_t* codes;      // combined codes (fast path)
    const uint8_t* indices;    // separate indices (fallback)
    const uint8_t* sign_bits;  // separate sign bits (fallback)
    const float* norms;
    const int64_t* ids;
    int n_vectors;
    bool has_codes;            // whether combined codes are available
};

// ── IVF-TQ search with combined codes ──
py::tuple ivf_search_py(
    py::array_t<float, py::array::c_style> sub_centroids,
    py::array_t<float, py::array::c_style> tq_centroids,
    py::list partition_data_list,
    py::array_t<float, py::array::c_style> q_rotated,
    py::array_t<float, py::array::c_style> coarse_scores,
    py::array_t<int32_t, py::array::c_style> top_lists,
    bool use_sign,
    int k)
{
    auto q_buf = q_rotated.request();
    int nq = static_cast<int>(q_buf.shape[0]);
    int dim = static_cast<int>(q_buf.shape[1]);

    auto cs_buf = coarse_scores.request();
    auto tl_buf = top_lists.request();
    int nprobe = static_cast<int>(tl_buf.shape[1]);
    int nlist = static_cast<int>(partition_data_list.size());

    auto sc_buf = sub_centroids.request();
    auto tc_buf = tq_centroids.request();
    int n_levels = use_sign ? static_cast<int>(sc_buf.shape[0])
                            : static_cast<int>(tc_buf.shape[0]);
    int n_entries = use_sign ? n_levels * 2 : n_levels;

    const float* sc_ptr = static_cast<const float*>(sc_buf.ptr);
    const float* cen_ptr = static_cast<const float*>(tc_buf.ptr);
    const float* q_ptr = static_cast<const float*>(q_buf.ptr);
    const float* cs_ptr = static_cast<const float*>(cs_buf.ptr);
    const int32_t* tl_ptr = static_cast<const int32_t*>(tl_buf.ptr);

    // ── Phase 1: Extract partition data ONCE ──
    std::vector<py::array_t<uint8_t>> kept_codes(nlist);
    std::vector<py::array_t<uint8_t>> kept_idx(nlist);
    std::vector<py::array_t<uint8_t>> kept_sign(nlist);
    std::vector<py::array_t<float>> kept_norms(nlist);
    std::vector<py::array_t<int64_t>> kept_ids(nlist);
    std::vector<PartitionInfo> parts(nlist);

    for (int i = 0; i < nlist; ++i) {
        py::dict d = partition_data_list[i].cast<py::dict>();
        if (!d.contains("indices") || d["indices"].is_none()) {
            parts[i] = {nullptr, nullptr, nullptr, nullptr, nullptr, 0, false};
            continue;
        }

        kept_norms[i] = d["norms"].cast<py::array_t<float, py::array::c_style>>();
        kept_ids[i] = d["ids"].cast<py::array_t<int64_t, py::array::c_style>>();
        kept_idx[i] = d["indices"].cast<py::array_t<uint8_t, py::array::c_style>>();

        auto pi = kept_idx[i].request();
        parts[i].n_vectors = static_cast<int>(pi.shape[0]);
        parts[i].indices = static_cast<const uint8_t*>(pi.ptr);
        parts[i].norms = static_cast<const float*>(kept_norms[i].request().ptr);
        parts[i].ids = static_cast<const int64_t*>(kept_ids[i].request().ptr);

        // Try to get precomputed combined codes (fast path)
        if (d.contains("codes") && !d["codes"].is_none()) {
            kept_codes[i] = d["codes"].cast<py::array_t<uint8_t, py::array::c_style>>();
            parts[i].codes = static_cast<const uint8_t*>(kept_codes[i].request().ptr);
            parts[i].has_codes = true;
        } else {
            parts[i].codes = nullptr;
            parts[i].has_codes = false;
        }

        if (use_sign && d.contains("sign_bits") && !d["sign_bits"].is_none()) {
            kept_sign[i] = d["sign_bits"].cast<py::array_t<uint8_t, py::array::c_style>>();
            parts[i].sign_bits = static_cast<const uint8_t*>(kept_sign[i].request().ptr);
        } else {
            parts[i].sign_bits = nullptr;
        }
    }

    // ── Phase 2: Score all queries (OpenMP parallel over queries) ──
    auto out_scores = py::array_t<float>({nq, k});
    auto out_indices = py::array_t<int64_t>({nq, k});
    float* os_ptr = static_cast<float*>(out_scores.request().ptr);
    int64_t* oi_ptr = static_cast<int64_t*>(out_indices.request().ptr);

    // Release GIL so Python threads can run in parallel
    {
    py::gil_scoped_release release;

    for (int q = 0; q < nq; ++q) {
        const float* q_row = q_ptr + static_cast<size_t>(q) * dim;

        // Build ADC table once per query
        std::vector<float> table(dim * n_entries);
        build_adc_table_inline(q_row, dim, sc_ptr, n_levels, cen_ptr,
                               use_sign, table.data(), n_entries);

        // Score all probed partitions
        std::vector<tqs::Candidate> candidates;
        candidates.reserve(2048);

        for (int p = 0; p < nprobe; ++p) {
            int list_idx = tl_ptr[q * nprobe + p];
            if (list_idx < 0 || list_idx >= nlist) continue;

            const auto& part = parts[list_idx];
            if (part.n_vectors == 0 || !part.norms) continue;

            float coarse = cs_ptr[q * nlist + list_idx];

            if (part.has_codes) {
                // Fast path: use precomputed combined codes
                tqs::score_partition_fast(
                    table.data(), n_entries, dim,
                    part.codes, part.norms, part.n_vectors,
                    coarse, part.ids, candidates);
            } else {
                // Fallback: compute codes on the fly
                for (int j = 0; j < part.n_vectors; ++j) {
                    float sum = 0.0f;
                    const uint8_t* idx_row = part.indices + static_cast<size_t>(j) * dim;
                    if (use_sign && part.sign_bits) {
                        const uint8_t* sign_row = part.sign_bits + static_cast<size_t>(j) * dim;
                        for (int d = 0; d < dim; ++d) {
                            sum += table[d * n_entries + idx_row[d] * 2 + sign_row[d]];
                        }
                    } else {
                        for (int d = 0; d < dim; ++d) {
                            sum += table[d * n_entries + idx_row[d]];
                        }
                    }
                    candidates.push_back({sum * part.norms[j] + coarse, part.ids[j]});
                }
            }
        }

        tqs::topk_select(candidates, k,
                         os_ptr + static_cast<size_t>(q) * k,
                         oi_ptr + static_cast<size_t>(q) * k);
    }

    } // end GIL release scope

    return py::make_tuple(out_scores, out_indices);
}

// ── Cascade IVF-TQ search ──
// Two-pass search exploiting Lloyd–Max bin ordinality:
//   Pass 1: score every candidate using only top-msb bits of the primary
//           index against a coarsened (2^msb_bits)-entry LUT.
//   Pass 2: rerank top-rerank_n candidates using full primary+sign codes.
// Returns final top-k per query.
py::tuple cascade_search_py(
    py::array_t<float, py::array::c_style> coarse_recon,     // (n_msb,) Pass-1 codebook
    py::array_t<float, py::array::c_style> sub_centroids,    // (n_levels, 2) for use_sign, else (n_levels,)
    py::list partition_data_list,                            // each: {msb_codes, codes, norms, ids}
    py::array_t<float, py::array::c_style> q_rotated,        // (nq, dim)
    py::array_t<float, py::array::c_style> coarse_scores,    // (nq, nlist)
    py::array_t<int32_t, py::array::c_style> top_lists,      // (nq, nprobe)
    bool use_sign,
    int k,
    int rerank_n)
{
    auto q_buf = q_rotated.request();
    int nq = static_cast<int>(q_buf.shape[0]);
    int dim = static_cast<int>(q_buf.shape[1]);

    auto cs_buf = coarse_scores.request();
    auto tl_buf = top_lists.request();
    int nprobe = static_cast<int>(tl_buf.shape[1]);
    int nlist = static_cast<int>(partition_data_list.size());

    auto sc_buf = sub_centroids.request();
    int n_levels = static_cast<int>(sc_buf.shape[0]);
    int n_full = use_sign ? n_levels * 2 : n_levels;

    auto cr_buf = coarse_recon.request();
    int n_msb = static_cast<int>(cr_buf.shape[0]);

    const float* qrot_ptr = static_cast<const float*>(q_buf.ptr);
    const float* cs_ptr = static_cast<const float*>(cs_buf.ptr);
    const int32_t* tl_ptr = static_cast<const int32_t*>(tl_buf.ptr);
    const float* cr_ptr = static_cast<const float*>(cr_buf.ptr);
    const float* sc_ptr = static_cast<const float*>(sc_buf.ptr);

    // ── Phase 1: Extract partition data ONCE ──
    struct PartInfo {
        const uint8_t* msb_codes;   // (n, dim) Pass-1 codes (primary >> lsb_count)
        const uint8_t* codes;       // (n, dim) Pass-2 codes (primary*2 + sign for use_sign)
        const float* norms;
        const int64_t* ids;
        int n_vectors;
    };
    std::vector<py::array_t<uint8_t>> hold_msb(nlist), hold_codes(nlist);
    std::vector<py::array_t<float>> hold_norms(nlist);
    std::vector<py::array_t<int64_t>> hold_ids(nlist);
    std::vector<PartInfo> parts(nlist);

    for (int i = 0; i < nlist; ++i) {
        py::dict d = partition_data_list[i].cast<py::dict>();
        if (!d.contains("msb_codes") || d["msb_codes"].is_none()) {
            parts[i] = {nullptr, nullptr, nullptr, nullptr, 0};
            continue;
        }
        hold_msb[i] = d["msb_codes"].cast<py::array_t<uint8_t, py::array::c_style>>();
        hold_codes[i] = d["codes"].cast<py::array_t<uint8_t, py::array::c_style>>();
        hold_norms[i] = d["norms"].cast<py::array_t<float, py::array::c_style>>();
        hold_ids[i] = d["ids"].cast<py::array_t<int64_t, py::array::c_style>>();

        auto pm = hold_msb[i].request();
        parts[i].msb_codes = static_cast<const uint8_t*>(pm.ptr);
        parts[i].codes = static_cast<const uint8_t*>(hold_codes[i].request().ptr);
        parts[i].norms = static_cast<const float*>(hold_norms[i].request().ptr);
        parts[i].ids = static_cast<const int64_t*>(hold_ids[i].request().ptr);
        parts[i].n_vectors = static_cast<int>(pm.shape[0]);
    }

    // ── Phase 2: Score all queries ──
    auto out_scores = py::array_t<float>({nq, k});
    auto out_indices = py::array_t<int64_t>({nq, k});
    float* os_ptr = static_cast<float*>(out_scores.request().ptr);
    int64_t* oi_ptr = static_cast<int64_t*>(out_indices.request().ptr);

    {
    py::gil_scoped_release release;

    for (int q = 0; q < nq; ++q) {
        const float* q_row = qrot_ptr + static_cast<size_t>(q) * dim;

        // ── Pass 1 ADC table: dim × n_msb ──
        std::vector<float> pass1_tbl(static_cast<size_t>(dim) * n_msb);
        for (int d = 0; d < dim; ++d) {
            float qd = q_row[d];
            float* row = pass1_tbl.data() + static_cast<size_t>(d) * n_msb;
            for (int l = 0; l < n_msb; ++l) {
                row[l] = qd * cr_ptr[l];
            }
        }

        // Pass 1: score every probed partition. Track (cell, local) so Pass 2 can
        // look up the full code without an id->position search.
        struct CC { float score; int cell; int local_idx; int64_t id; };
        std::vector<CC> cands;
        cands.reserve(2048);

        const float* __restrict__ tbl1 = pass1_tbl.data();

        for (int p = 0; p < nprobe; ++p) {
            int cell = tl_ptr[q * nprobe + p];
            if (cell < 0 || cell >= nlist) continue;
            const auto& pi = parts[cell];
            if (pi.n_vectors == 0) continue;

            float coarse = cs_ptr[q * nlist + cell];

            int j = 0;
            // Candidate-parallel scoring: 4 vectors at a time, 4 independent
            // accumulators so the CPU can pipeline the indirect loads.
            // Mirrors tqs::score_4_vectors in tqs_kernels.cpp.
            for (; j + 3 < pi.n_vectors; j += 4) {
                const uint8_t* __restrict__ c0 =
                    pi.msb_codes + static_cast<size_t>(j + 0) * dim;
                const uint8_t* __restrict__ c1 =
                    pi.msb_codes + static_cast<size_t>(j + 1) * dim;
                const uint8_t* __restrict__ c2 =
                    pi.msb_codes + static_cast<size_t>(j + 2) * dim;
                const uint8_t* __restrict__ c3 =
                    pi.msb_codes + static_cast<size_t>(j + 3) * dim;
                float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
                for (int d = 0; d < dim; ++d) {
                    const float* __restrict__ row = tbl1 + static_cast<size_t>(d) * n_msb;
                    s0 += row[c0[d]];
                    s1 += row[c1[d]];
                    s2 += row[c2[d]];
                    s3 += row[c3[d]];
                }
                cands.push_back({s0 * pi.norms[j+0] + coarse, cell, j+0, pi.ids[j+0]});
                cands.push_back({s1 * pi.norms[j+1] + coarse, cell, j+1, pi.ids[j+1]});
                cands.push_back({s2 * pi.norms[j+2] + coarse, cell, j+2, pi.ids[j+2]});
                cands.push_back({s3 * pi.norms[j+3] + coarse, cell, j+3, pi.ids[j+3]});
            }
            // Tail
            for (; j < pi.n_vectors; ++j) {
                const uint8_t* code = pi.msb_codes + static_cast<size_t>(j) * dim;
                float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
                int d = 0;
                for (; d + 3 < dim; d += 4) {
                    s0 += tbl1[(d + 0) * n_msb + code[d + 0]];
                    s1 += tbl1[(d + 1) * n_msb + code[d + 1]];
                    s2 += tbl1[(d + 2) * n_msb + code[d + 2]];
                    s3 += tbl1[(d + 3) * n_msb + code[d + 3]];
                }
                float sum = s0 + s1 + s2 + s3;
                for (; d < dim; ++d) {
                    sum += tbl1[d * n_msb + code[d]];
                }
                cands.push_back({sum * pi.norms[j] + coarse, cell, j, pi.ids[j]});
            }
        }

        // Top-rerank_n by Pass-1 score
        int N = std::min(static_cast<int>(cands.size()), rerank_n);
        if (N == 0) {
            for (int i = 0; i < k; ++i) {
                os_ptr[static_cast<size_t>(q) * k + i] = -1e30f;
                oi_ptr[static_cast<size_t>(q) * k + i] = -1;
            }
            continue;
        }
        std::nth_element(cands.begin(), cands.begin() + N, cands.end(),
            [](const CC& a, const CC& b) { return a.score > b.score; });
        cands.resize(N);

        // ── Pass 2 ADC table: dim × n_full ──
        std::vector<float> pass2_tbl(static_cast<size_t>(dim) * n_full);
        if (use_sign) {
            for (int d = 0; d < dim; ++d) {
                float qd = q_row[d];
                float* row = pass2_tbl.data() + static_cast<size_t>(d) * n_full;
                for (int l = 0; l < n_levels; ++l) {
                    row[l * 2 + 0] = qd * sc_ptr[l * 2 + 0];
                    row[l * 2 + 1] = qd * sc_ptr[l * 2 + 1];
                }
            }
        } else {
            for (int d = 0; d < dim; ++d) {
                float qd = q_row[d];
                float* row = pass2_tbl.data() + static_cast<size_t>(d) * n_full;
                for (int l = 0; l < n_levels; ++l) {
                    row[l] = qd * sc_ptr[l];
                }
            }
        }

        // Re-score each top-N candidate at full precision
        for (int i = 0; i < N; ++i) {
            CC& c = cands[i];
            const auto& pi = parts[c.cell];
            const uint8_t* code = pi.codes + static_cast<size_t>(c.local_idx) * dim;
            float sum = 0.0f;
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            int d = 0;
            for (; d + 3 < dim; d += 4) {
                s0 += pass2_tbl[(d + 0) * n_full + code[d + 0]];
                s1 += pass2_tbl[(d + 1) * n_full + code[d + 1]];
                s2 += pass2_tbl[(d + 2) * n_full + code[d + 2]];
                s3 += pass2_tbl[(d + 3) * n_full + code[d + 3]];
            }
            sum = s0 + s1 + s2 + s3;
            for (; d < dim; ++d) {
                sum += pass2_tbl[d * n_full + code[d]];
            }
            float coarse = cs_ptr[q * nlist + c.cell];
            c.score = sum * pi.norms[c.local_idx] + coarse;
        }

        // Top-k from re-scored candidates
        int kk = std::min(k, N);
        std::nth_element(cands.begin(), cands.begin() + kk, cands.end(),
            [](const CC& a, const CC& b) { return a.score > b.score; });
        std::sort(cands.begin(), cands.begin() + kk,
            [](const CC& a, const CC& b) { return a.score > b.score; });
        for (int i = 0; i < kk; ++i) {
            os_ptr[static_cast<size_t>(q) * k + i] = cands[i].score;
            oi_ptr[static_cast<size_t>(q) * k + i] = cands[i].id;
        }
        for (int i = kk; i < k; ++i) {
            os_ptr[static_cast<size_t>(q) * k + i] = -1e30f;
            oi_ptr[static_cast<size_t>(q) * k + i] = -1;
        }
    }

    } // end GIL release scope

    return py::make_tuple(out_scores, out_indices);
}

PYBIND11_MODULE(_tqs_cpp, m) {
    m.doc() = "C++ accelerated search kernels for TurboQuant Search";
    m.def("tq_flat_search", &tq_flat_search_py,
          "Flat TQ search: ADC + top-k",
          py::arg("sub_centroids"), py::arg("indices"),
          py::arg("sign_bits"), py::arg("norms"),
          py::arg("centroids"), py::arg("q_rotated"),
          py::arg("use_sign"), py::arg("k"));
    m.def("ivf_search", &ivf_search_py,
          "IVF-TQ search with combined codes + OpenMP",
          py::arg("sub_centroids"), py::arg("tq_centroids"),
          py::arg("partition_data_list"), py::arg("q_rotated"),
          py::arg("coarse_scores"), py::arg("top_lists"),
          py::arg("use_sign"), py::arg("k"));
    m.def("cascade_search", &cascade_search_py,
          "Cascade IVF-TQ search: Pass-1 MSB filter + Pass-2 full re-rank",
          py::arg("coarse_recon"), py::arg("sub_centroids"),
          py::arg("partition_data_list"), py::arg("q_rotated"),
          py::arg("coarse_scores"), py::arg("top_lists"),
          py::arg("use_sign"), py::arg("k"), py::arg("rerank_n"));
}
