#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "rabitqlib/core/metric.hpp"                   // IWYU pragma: export
#include "rabitqlib/core/types.hpp"                    // IWYU pragma: export
#include "rabitqlib/index/detail/visited_set_fwd.hpp"  // IWYU pragma: export
#include "rabitqlib/index/types.hpp"                   // IWYU pragma: export
#include "rabitqlib/utils/memory.hpp"
#include "rabitqlib/utils/packed.hpp"

namespace rabitqlib {
template <typename T>
class Rotator;
template <typename T>
class BatchQuery;
namespace buffer {
template <typename T>
class SearchBuffer;
}
}  // namespace rabitqlib

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
class QuantizedGraph<float>;

namespace detail {
struct SearchScratchView;

void search_with_workspace(
    QuantizedGraph<float>& graph,
    const float* query,
    uint32_t k,
    uint32_t* result_ids,
    float* distances,
    SearchScratchView scratch,
    PID hint = kPidMax,
    const float* k1xsumq = nullptr
);
}  // namespace detail

template <>
class QuantizedGraph<float> {
    friend class QGBuilder;
    friend void detail::search_with_workspace(
        QuantizedGraph<float>& graph,
        const float* query,
        uint32_t k,
        uint32_t* result_ids,
        float* distances,
        detail::SearchScratchView scratch,
        PID hint,
        const float* k1xsumq
    );
    friend struct QGConstructionTestAccess;

   private:
    size_t max_elements_ = 0;
    size_t max_degree_ = 0;
    size_t dim_ = 0;
    size_t padded_dim_ = 0;
    float (*raw_dist_func_)(const float*, const float*, size_t) = nullptr;
    PID entry_point_id_ = 0;
    MetricType metric_type_ = METRIC_L2;
    RotatorType rotator_type_ = RotatorType::FhtKacRotator;
    size_t quantization_bits_ = 0;  // 0: raw vectors, 4/8: packed RaBitQ vectors
    std::vector<float> centroid_;   // rotated global centroid for qg-quant
    float (*quantized_ip_func_)(const float*, const uint8_t*, size_t) = nullptr;

    using RowStorage =
        std::vector<float, memory::DefaultInitAlignedAllocator<float, 64, true>>;
    // Complete rows are contiguous in both raw and quantized modes:
    // vector/code, neighbor quantization data, then packed neighbor IDs.
    // Typed storage establishes float lifetimes for raw vectors; packed portions
    // are accessed only as bytes, never as float values.
    // Builders and load() fill rows before use; leave scalars uninitialized so
    // construction workers first-touch their own pages instead of zeroing serially.
    RowStorage data_;
    std::unique_ptr<Rotator<float>> rotator_;

    // Position of row data (raw vector or packed qg-quant vector), neighbor
    // quantization data, and neighbor IDs. Since every degree equals max_degree_
    // (a multiple of 32), the degree does not need to be stored per vertex.
    size_t batch_data_offset_ = 0;
    size_t neighbor_offset_ = 0;
    size_t record_bytes_ = 0;  // length of entire row
    size_t default_ef_search_ = 0;
    bool ready_ = false;
    uint32_t seed_ = 42;

    [[nodiscard]] static size_t checked_add(size_t lhs, size_t rhs);

    [[nodiscard]] static size_t checked_multiply(size_t lhs, size_t rhs);

    [[nodiscard]] static size_t padded_dimension(size_t dim);

    void validate_configuration() const;

    void initialize_layout();

    void initialize(bool randomize_rotation = true);

    // Write the graph file at exactly this path.
    void write_file(const char* filename) const;

    void copy_vectors(const float* data, size_t num_workers);

    void set_quantization_centroid(const float* centroid);

    [[nodiscard]] char* get_row_data(PID internal_id) {
        return reinterpret_cast<char*>(get_vector(internal_id));
    }

    [[nodiscard]] const char* get_row_data(PID internal_id) const {
        return reinterpret_cast<const char*>(get_vector(internal_id));
    }

    [[nodiscard]] float* get_vector(PID internal_id) {
        return data_.data() + ((record_bytes_ / sizeof(float)) * internal_id);
    }

    [[nodiscard]] const float* get_vector(PID internal_id) const {
        return data_.data() + ((record_bytes_ / sizeof(float)) * internal_id);
    }

    [[nodiscard]] char* get_quantized_vector(PID internal_id) {
        return get_row_data(internal_id);
    }

    [[nodiscard]] const char* get_quantized_vector(PID internal_id) const {
        return get_row_data(internal_id);
    }

    void validate_search(
        const float* query,
        uint32_t k,
        const uint32_t* result_ids,
        const float* distances,
        size_t ef_search
    ) const;

    void search_impl(
        const float* query,
        uint32_t k,
        uint32_t* result_ids,
        float* distances,
        size_t ef_search
    );

    void search_with_scratch_impl(
        const float* query,
        uint32_t k,
        uint32_t* result_ids,
        float* distances,
        detail::SearchScratchView scratch,
        PID hint,
        const float* k1xsumq
    );

    float point_distance(
        const float* raw_query, const QuantizedQuery* quantized_query, PID internal_id
    ) const;

    float quantized_distance(const QuantizedQuery& query, PID internal_id) const;

    void reconstruct_quantized_vector(PID internal_id, float* reconstructed) const;

    [[nodiscard]] char* get_batch_data(PID internal_id) {
        return get_row_data(internal_id) + batch_data_offset_;
    }

    [[nodiscard]] const char* get_batch_data(PID internal_id) const {
        return get_row_data(internal_id) + batch_data_offset_;
    }

    [[nodiscard]] rabitqlib::detail::PackedArrayView<PID> get_neighbors(PID internal_id) {
        return rabitqlib::detail::PackedArrayView<PID>(
            get_row_data(internal_id) + neighbor_offset_
        );
    }

    [[nodiscard]] rabitqlib::detail::ConstPackedArrayView<PID> get_neighbors(
        PID internal_id
    ) const {
        return rabitqlib::detail::ConstPackedArrayView<PID>(
            get_row_data(internal_id) + neighbor_offset_
        );
    }

    // Inputs are in this graph's rotated, padded domain. rotated_source is the
    // source vertex, used as the quantization centroid; rotated_data contains one
    // vector per neighbor, in ID order.
    void encode_neighbors(
        PID internal_id,
        const std::vector<PID>& new_neighbors,
        const float* rotated_data,
        const float* rotated_source
    );

    void update_results(
        buffer::SearchBuffer<float>& result_pool,
        VisitedSet& visited,
        const float* query,
        const QuantizedQuery* quantized_query
    );

    void scan_neighbors(
        const BatchQuery<float>& query_wrapper,
        PID internal_id,
        float* estimated_distances,
        buffer::SearchBuffer<float>& search_pool,
        VisitedSet& visited,
        size_t num_neighbors
    ) const;

   public:
    explicit QuantizedGraph(
        size_t max_elements,
        size_t dim,
        size_t max_degree,
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

    // Capacity, nonremoved points, and stored records, respectively. Until building
    // completes, both point counts are zero. This index has no remove operation.
    [[nodiscard]] size_t max_elements() const;
    [[nodiscard]] size_t num_points() const;
    [[nodiscard]] size_t num_stored_points() const;

    // Configured graph size, also used by builders before rows are searchable.
    [[nodiscard]] size_t num_vertices() const;

    [[nodiscard]] size_t dimension() const;

    [[nodiscard]] size_t degree_bound() const;

    [[nodiscard]] PID entry_point() const;

    [[nodiscard]] MetricType metric_type() const;

    [[nodiscard]] size_t quantization_bits() const;

    [[nodiscard]] bool is_quantized() const;

    void set_ep(PID entry);

    void save(const char* filename) const;

    void load(const char* filename);

    void set_ef(size_t ef_search);

    void search(
        const float* __restrict__ query,
        uint32_t k,
        uint32_t* __restrict__ result_ids,
        float* __restrict__ distances
    );

    /**
     * Search contiguous row-major queries, using the window set by set_ef().
     * queries contains num_queries * dimension() floats; result_ids and distances
     * each hold num_queries * k elements. Empty batches do no work.
     * Scratch is reused within each worker. num_threads defaults to one;
     * zero selects the available logical CPU count, capped by the batch.
     * Input and output buffers must not overlap. Concurrent searches require
     * separate outputs and no index mutation, including set_ef().
     */
    void search_batch(
        const float* queries,
        size_t num_queries,
        uint32_t k,
        uint32_t* result_ids,
        float* distances,
        size_t num_threads = 1
    );

    // Like search_batch, but ef_search belongs to this call and never reads or changes
    // the default set by set_ef(). Concurrent calls may use different windows.
    void search_batch_with_ef(
        const float* queries,
        size_t num_queries,
        uint32_t k,
        uint32_t* result_ids,
        float* distances,
        size_t ef_search,
        size_t num_threads = 1
    );

    /**
     * Search using caller-owned scratch buffers.
     *
     * Reuses scratch and can use a previous result as a warm-start hint.
     * BatchQuery may allocate LUT storage on its first use or when its shape changes;
     * this is not the stable core v1 zero-allocation interface. Search and result
     * buffers must be sized for ef_search and k respectively. rotated_query and
     * lookup_table need D and 4 * D floats, where D is the graph's stored padded
     * dimension; this advanced caller must already know D. dimension() is the original
     * input dimension. estimated_distances needs degree_bound() floats. The visited set
     * must be initialized before the first search; an uninitialized set raises
     * std::invalid_argument. Concurrent searches require separate scratch
     * buffers, including visited sets.
     *
     * An optional k1xsumq must come from BatchQuery::k1xsumq() for the same
     * query and this graph's rotation. A null pointer recomputes the sum.
     */
    void search_with_scratch(
        const float* __restrict__ query,
        uint32_t k,
        uint32_t* __restrict__ result_ids,
        float* __restrict__ distances,
        float* __restrict__ rotated_query,
        float* __restrict__ estimated_distances,
        float* __restrict__ lookup_table,
        BatchQuery<float>& batch_query,
        buffer::SearchBuffer<float>& search_pool,
        buffer::SearchBuffer<float>& result_pool,
        VisitedSet& visited,
        PID hint = kPidMax,
        const float* k1xsumq = nullptr
    );

    [[nodiscard]] size_t padded_dim() const { return padded_dim_; }
};

// Preserve C++17 deduction for callers that omit the float template argument.
QuantizedGraph() -> QuantizedGraph<float>;
QuantizedGraph(
    size_t,
    size_t,
    size_t,
    MetricType = METRIC_L2,
    RotatorType = RotatorType::FhtKacRotator,
    size_t = 0,
    uint32_t = std::random_device{}()
) -> QuantizedGraph<float>;

}  // namespace rabitqlib::symqg
