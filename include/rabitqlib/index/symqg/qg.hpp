#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <vector>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/index/query.hpp"
#include "rabitqlib/quantization/data_layout.hpp"
#include "rabitqlib/utils/buffer.hpp"
#include "rabitqlib/utils/memory.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/utils/space.hpp"
#include "rabitqlib/utils/visited_set.hpp"

namespace rabitqlib::symqg {

class QuantizedQuery {
   private:
    const float* rotated_query_;
    float k1xsumq_;
    float g_add_;

   public:
    QuantizedQuery(
        const float* rotated_query,
        const float* centroid,
        size_t padded_dim,
        MetricType metric_type
    );
    QuantizedQuery(const float* rotated_query, const QuantizedQuery& prepared);
    [[nodiscard]] const float* rotated_query() const;
    [[nodiscard]] float k1xsumq() const;
    [[nodiscard]] float g_add() const;
};

template <typename T = float>
class QuantizedGraph;

template <>
class QuantizedGraph<float> {
    friend class QGBuilder;
    friend struct QGConstructionTestAccess;

   private:
    size_t num_points_ = 0;    // num points
    size_t degree_bound_ = 0;  // degree bound
    size_t dim_ = 0;           // dimension
    size_t padded_dim_ = 0;    // padded dimension
    // Raw-vector distance.
    float (*raw_dist_func_)(const float*, const float*, size_t) = nullptr;
    PID entry_point_ = 0;  // Entry point of graph
    MetricType metric_type_ = MetricType::METRIC_L2;
    RotatorType rotator_type_ = RotatorType::FhtKacRotator;
    size_t quantization_bits_ = 0;  // 0: raw vectors, 4/8: packed RaBitQ vectors
    std::vector<float> centroid_;   // rotated global centroid for qg-quant
    ex_ipfunc quantized_ip_func_ = nullptr;

    using RowStorage =
        std::vector<float, memory::DefaultInitAlignedAllocator<float, 64, true>>;
    // Complete rows are contiguous in both raw and quantized modes:
    // vector/code, neighbor quantization data, then packed neighbor IDs.
    // Typed storage establishes float lifetimes for raw vectors; packed portions
    // are accessed only as bytes, never as float values.
    // Builders and load() fill rows before use; leave scalars uninitialized so
    // construction workers first-touch their own pages instead of zeroing serially.
    RowStorage data_;
    std::unique_ptr<Rotator<float>> rotator_;  // data rotator

    // Position of row data (raw vector or packed qg-quant vector), neighbor
    // quantization data, and neighbor IDs. Since every degree equals degree_bound_
    // (a multiple of 32), the degree does not need to be stored per vertex.
    size_t batch_data_offset_ = 0;  // offset of qg batch data
    size_t neighbor_offset_ = 0;    // offset of neighbors
    size_t row_offset_ = 0;         // length of entire row
    size_t ef_ = 0;
    bool ready_ = false;
    uint32_t seed_ = 42;

    // Owned by QGBuilder and never persisted.
    struct BuildCache {
        std::vector<float> rotated_vectors;
        std::vector<QuantizedQuery> queries;
    };

    [[nodiscard]] static size_t checked_add(size_t lhs, size_t rhs);

    [[nodiscard]] static size_t checked_multiply(size_t lhs, size_t rhs);

    [[nodiscard]] static size_t padded_dimension(size_t dim);

    void validate_configuration() const;

    void initialize_layout();

    void initialize();

    void copy_vectors(const float*, size_t);

    void set_quantization_centroid(const float* centroid);

    [[nodiscard]] char* get_row_data(PID data_id) {
        return reinterpret_cast<char*>(get_vector(data_id));
    }

    [[nodiscard]] const char* get_row_data(PID data_id) const {
        return reinterpret_cast<const char*>(get_vector(data_id));
    }

    [[nodiscard]] float* get_vector(PID data_id) {
        return data_.data() + ((row_offset_ / sizeof(float)) * data_id);
    }

    [[nodiscard]] const float* get_vector(PID data_id) const {
        return data_.data() + ((row_offset_ / sizeof(float)) * data_id);
    }

    [[nodiscard]] char* get_quantized_vector(PID data_id) { return get_row_data(data_id); }

    [[nodiscard]] const char* get_quantized_vector(PID data_id) const {
        return get_row_data(data_id);
    }

    void validate_search(const float*, uint32_t, const uint32_t*, const float*, size_t)
        const;

    void search_impl(const float*, uint32_t, uint32_t*, float*, size_t);

    void
    search_with_scratch_impl(const float*, uint32_t, uint32_t*, float*, float*, float*, float*, BatchQuery<float>&, buffer::SearchBuffer<float>&, buffer::SearchBuffer<float>&, VisitedSet&, PID, const float*);

    const float* prepare_build_query(
        PID,
        std::vector<float>&,
        std::optional<QuantizedQuery>&,
        const BuildCache* = nullptr
    ) const;

    float point_distance(const float*, const QuantizedQuery*, PID) const;

    float quantized_distance(const QuantizedQuery&, PID) const;

    void reconstruct_quantized_vector(PID, float*) const;

    [[nodiscard]] char* get_batch_data(PID data_id) {
        return get_row_data(data_id) + batch_data_offset_;
    }

    [[nodiscard]] const char* get_batch_data(PID data_id) const {
        return get_row_data(data_id) + batch_data_offset_;
    }

    [[nodiscard]] rabitqlib::detail::PackedArrayView<PID> get_neighbors(PID data_id) {
        return rabitqlib::detail::PackedArrayView<PID>(
            get_row_data(data_id) + neighbor_offset_
        );
    }

    [[nodiscard]] rabitqlib::detail::ConstPackedArrayView<PID> get_neighbors(PID data_id
    ) const {
        return rabitqlib::detail::ConstPackedArrayView<PID>(
            get_row_data(data_id) + neighbor_offset_
        );
    }

    void find_candidates(
        PID,
        size_t,
        std::vector<AnnCandidate<float>>&,
        VisitedSet&,
        const std::vector<uint32_t>&,
        const BuildCache* = nullptr
    ) const;

    void update_qg(
        PID,
        const std::vector<AnnCandidate<float>>&,
        const BuildCache* = nullptr,
        std::vector<float>* = nullptr
    );

    void
    update_results(buffer::SearchBuffer<float>&, VisitedSet&, const float*, const QuantizedQuery*);

    inline void scan_neighbors(
        const BatchQuery<float>&,
        PID,
        float*,
        buffer::SearchBuffer<float>&,
        VisitedSet&,
        size_t
    ) const;

   public:
    explicit QuantizedGraph(
        size_t num,
        size_t dim,
        size_t max_deg,
        MetricType metric_type = METRIC_L2,
        RotatorType rotator_type = RotatorType::FhtKacRotator,
        size_t quantization_bits = 0,
        uint32_t seed = std::random_device{}()
    );

    explicit QuantizedGraph();

    ~QuantizedGraph();

    QuantizedGraph(const QuantizedGraph&) = delete;
    QuantizedGraph& operator=(const QuantizedGraph&) = delete;
    QuantizedGraph(QuantizedGraph&&) noexcept;
    QuantizedGraph& operator=(QuantizedGraph&&) noexcept;

    [[nodiscard]] size_t num_vertices() const;

    [[nodiscard]] size_t dimension() const;

    [[nodiscard]] size_t degree_bound() const;

    [[nodiscard]] PID entry_point() const;

    [[nodiscard]] MetricType metric_type() const;

    [[nodiscard]] size_t quantization_bits() const;

    [[nodiscard]] bool is_quantized() const;

    void set_ep(PID entry);

    void save(const char*) const;

    void load(const char*);

    void set_ef(size_t);

    /* search and copy results to KNN */
    void search(
        const float* __restrict__ query,
        uint32_t knn,
        uint32_t* __restrict__ results,
        float* __restrict__ dists
    );

    /**
     * Search contiguous row-major queries, using the window set by set_ef().
     * queries contains num_queries * dimension() floats; results and dists
     * each hold num_queries * knn elements. Empty batches do no work.
     * Scratch is reused within each worker. num_threads defaults to one;
     * zero selects the available logical CPU count, capped by the batch.
     * Input and output buffers must not overlap. Concurrent searches require
     * separate outputs and no index mutation, including set_ef().
     */
    void search_batch(
        const float* queries,
        size_t num_queries,
        uint32_t knn,
        uint32_t* results,
        float* dists,
        size_t num_threads = 1
    );

    // Like search_batch, but ef belongs to this call and never reads or changes
    // the default set by set_ef(). Concurrent calls may use different windows.
    void search_batch_with_ef(
        const float* queries,
        size_t num_queries,
        uint32_t knn,
        uint32_t* results,
        float* dists,
        size_t ef,
        size_t num_threads = 1
    );

    /**
     * Search using caller-owned scratch buffers.
     *
     * This avoids per-query allocations and can use a previous result as a
     * warm-start hint. The search and result buffers must be sized for ef and
     * knn respectively. rotated_query_scratch, est_dist_scratch, and
     * lut_float_scratch require padded_dim(), degree_bound(), and
     * padded_dim() * 4 elements respectively. The visited set must be
     * initialized before the first search; an uninitialized set raises
     * std::invalid_argument. Concurrent searches require separate scratch
     * buffers, including visited sets.
     *
     * An optional k1xsumq must come from BatchQuery::k1xsumq() for the same
     * query and this graph's rotation. A null pointer recomputes the sum.
     */
    void search_with_scratch(
        const float* __restrict__ query,
        uint32_t knn,
        uint32_t* __restrict__ results,
        float* __restrict__ dists,
        float* __restrict__ rotated_query_scratch,
        float* __restrict__ est_dist_scratch,
        float* __restrict__ lut_float_scratch,
        BatchQuery<float>& batch_query,
        buffer::SearchBuffer<float>& search_pool,
        buffer::SearchBuffer<float>& result_pool,
        VisitedSet& visited,
        PID hint = kPidMax,
        const float* k1xsumq = nullptr
    );

    [[nodiscard]] size_t padded_dim() const { return this->padded_dim_; }
};

// Preserve C++17 deduction for callers that omit the float template argument.
QuantizedGraph()->QuantizedGraph<float>;
QuantizedGraph(
    size_t,
    size_t,
    size_t,
    MetricType = METRIC_L2,
    RotatorType = RotatorType::FhtKacRotator,
    size_t = 0,
    uint32_t = std::random_device{}()
)
    ->QuantizedGraph<float>;

}  // namespace rabitqlib::symqg
