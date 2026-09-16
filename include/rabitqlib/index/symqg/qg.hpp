#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
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

    using RowStorage = std::vector<float, memory::AlignedAllocator<float, 1 << 22, true>>;
    // Complete rows are contiguous in both raw and quantized modes:
    // vector/code, neighbor quantization data, then packed neighbor IDs.
    // Typed storage establishes float lifetimes for raw vectors; packed portions
    // are accessed only as bytes, never as float values.
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

    [[nodiscard]] static size_t checked_add(size_t lhs, size_t rhs);

    [[nodiscard]] static size_t checked_multiply(size_t lhs, size_t rhs);

    [[nodiscard]] static size_t padded_dimension(size_t dim);

    void validate_configuration() const;

    void initialize_layout();

    void initialize();

    void copy_vectors(const float*, size_t);

    void set_quantization_centroid(const float* centroid);

    [[nodiscard]] char* get_row_data(PID data_id);

    [[nodiscard]] const char* get_row_data(PID data_id) const;

    [[nodiscard]] float* get_vector(PID data_id);

    [[nodiscard]] const float* get_vector(PID data_id) const;

    [[nodiscard]] char* get_quantized_vector(PID data_id);

    [[nodiscard]] const char* get_quantized_vector(PID data_id) const;

    void prepare_query(const float*, std::vector<float>&, std::optional<QuantizedQuery>&)
        const;

    const float*
    prepare_build_query(PID, std::vector<float>&, std::optional<QuantizedQuery>&) const;

    float point_distance(const float*, const QuantizedQuery*, PID) const;

    float quantized_distance(const QuantizedQuery&, PID) const;

    void reconstruct_quantized_vector(PID, float*) const;

    [[nodiscard]] char* get_batch_data(PID data_id);

    [[nodiscard]] const char* get_batch_data(PID data_id) const;

    [[nodiscard]] rabitqlib::detail::PackedArrayView<PID> get_neighbors(PID data_id);

    [[nodiscard]] rabitqlib::detail::ConstPackedArrayView<PID> get_neighbors(PID data_id
    ) const;

    void
    find_candidates(PID, size_t, std::vector<AnnCandidate<float>>&, VisitedSet&, const std::vector<uint32_t>&)
        const;

    void update_qg(PID, const std::vector<AnnCandidate<float>>&);

    void
    update_results(buffer::SearchBuffer<float>&, VisitedSet&, const float*, const QuantizedQuery*);

    void scan_neighbors(
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
        size_t quantization_bits = 0
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
};

// Preserve C++17 deduction for callers that omit the float template argument.
QuantizedGraph()->QuantizedGraph<float>;
QuantizedGraph(
    size_t,
    size_t,
    size_t,
    MetricType = METRIC_L2,
    RotatorType = RotatorType::FhtKacRotator,
    size_t = 0
)
    ->QuantizedGraph<float>;

}  // namespace rabitqlib::symqg
