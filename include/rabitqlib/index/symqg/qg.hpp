#pragma once

#include <omp.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/fastscan/fastscan.hpp"
#include "rabitqlib/index/estimator.hpp"
#include "rabitqlib/index/query.hpp"
#include "rabitqlib/quantization/data_layout.hpp"
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/utils/array.hpp"
#include "rabitqlib/utils/buffer.hpp"
#include "rabitqlib/utils/memory.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/utils/space.hpp"
#include "rabitqlib/utils/visited_pool.hpp"
#include "rabitqlib/utils/visited_set.hpp"

namespace rabitqlib::symqg {

template <typename T = float>
class QuantizedGraph {
    friend class QGBuilder;

   private:
    size_t num_points_ = 0;                                     // num points
    size_t degree_bound_ = 0;                                   // degree bound
    size_t dim_ = 0;                                            // dimension
    size_t padded_dim_ = 0;                                     // padded dimension
    T (*raw_dist_func_)(const T*, const T*, size_t) = nullptr;  // raw-vector distance
    PID entry_point_ = 0;                                       // Entry point of graph
    MetricType metric_type_ = MetricType::METRIC_L2;
    RotatorType rotator_type_ = RotatorType::FhtKacRotator;

    Array<
        char,
        std::vector<size_t>,
        memory::AlignedAllocator<
            char,
            1 << 22,
            true>>
        data_;                       // vectors + graph + quantization codes + factors
    Rotator<T>* rotator_ = nullptr;  // data rotator
    std::unique_ptr<VisitedListPool> visited_list_pool_ = nullptr;

    // Position of different data in each row (RawData + QuantizationCodes + Factors +
    // neighborIDs) Since we guarantee the degree for each vertex equals degree_bound
    // (multiple of 32), we do not need to store the degree for each vertex
    size_t batch_data_offset_ = 0;  // offset of qg batch data
    size_t neighbor_offset_ = 0;    // offset of neighbors
    size_t row_offset_ = 0;         // length of entire row
    size_t ef_ = 0;

    struct StorageLayout {
        size_t padded_dim;
        size_t batch_data_offset;
        size_t neighbor_offset;
        size_t row_offset;
        size_t total_bytes;
        size_t rotator_bytes;
    };

    static constexpr size_t kSerializedMetadataBytes =
        (sizeof(size_t) * 4) + sizeof(PID) + sizeof(RotatorType) + sizeof(MetricType);

    [[nodiscard]] static size_t checked_add(
        size_t first, size_t second, const char* message
    );

    [[nodiscard]] static size_t checked_multiply(
        size_t first, size_t second, const char* message
    );

    [[nodiscard]] StorageLayout compute_storage_layout() const;

    void initialize();

    void initialize(const StorageLayout& layout);

    void swap(QuantizedGraph& other) noexcept;

    void copy_vectors(const T*);

    [[nodiscard]] T* get_vector(PID data_id) {
        assert(static_cast<size_t>(data_id) < num_points_);
        return reinterpret_cast<T*>(
            data_.data() + (row_offset_ * static_cast<size_t>(data_id))
        );
    }

    [[nodiscard]] const T* get_vector(PID data_id) const {
        assert(static_cast<size_t>(data_id) < num_points_);
        return reinterpret_cast<const T*>(
            data_.data() + (row_offset_ * static_cast<size_t>(data_id))
        );
    }

    [[nodiscard]] char* get_batch_data(PID data_id) {
        assert(static_cast<size_t>(data_id) < num_points_);
        return data_.data() + (row_offset_ * static_cast<size_t>(data_id)) +
               batch_data_offset_;
    }

    [[nodiscard]] const char* get_batch_data(PID data_id) const {
        assert(static_cast<size_t>(data_id) < num_points_);
        return data_.data() + (row_offset_ * static_cast<size_t>(data_id)) +
               batch_data_offset_;
    }

    [[nodiscard]] PID* get_neighbors(PID data_id) {
        assert(static_cast<size_t>(data_id) < num_points_);
        return reinterpret_cast<PID*>(
            data_.data() + (row_offset_ * static_cast<size_t>(data_id)) + neighbor_offset_
        );
    }

    [[nodiscard]] const PID* get_neighbors(PID data_id) const {
        assert(static_cast<size_t>(data_id) < num_points_);
        return reinterpret_cast<const PID*>(
            data_.data() + (row_offset_ * static_cast<size_t>(data_id)) + neighbor_offset_
        );
    }

    void
    find_candidates(PID, size_t, std::vector<AnnCandidate<T>>&, VisitedSet&, const std::vector<uint32_t>&)
        const;

    void update_qg(PID, const std::vector<AnnCandidate<T>>&);

    void update_results(buffer::SearchBuffer<T>&, VisitedSet&, const T*);

    void scan_neighbors(
        const BatchQuery<T>&, PID, T*, buffer::SearchBuffer<T>&, VisitedSet&, size_t
    ) const;

   public:
    explicit QuantizedGraph(
        size_t num,
        size_t dim,
        size_t max_deg,
        MetricType metric_type = METRIC_L2,
        RotatorType rotator_type = RotatorType::FhtKacRotator
    );

    explicit QuantizedGraph() = default;

    ~QuantizedGraph();

    [[nodiscard]] auto num_vertices() const { return this->num_points_; }

    [[nodiscard]] auto dimension() const { return this->dim_; }

    [[nodiscard]] auto degree_bound() const { return this->degree_bound_; }

    [[nodiscard]] auto entry_point() const { return this->entry_point_; }

    [[nodiscard]] auto metric_type() const { return this->metric_type_; }

    void set_ep(PID entry) { this->entry_point_ = entry; };

    void save(const char*) const;

    void load(const char*);

    void set_ef(size_t);

    /* search and copy results to KNN */
    void search(const T* __restrict__ query, uint32_t knn, uint32_t* __restrict__ results);
    void search(
        const T* __restrict__ query,
        uint32_t knn,
        uint32_t* __restrict__ results,
        T* __restrict__ dists
    );
};

template <typename T>
inline QuantizedGraph<T>::QuantizedGraph(
    size_t num, size_t dim, size_t max_deg, MetricType metric_type, RotatorType rotator_type
)
    : num_points_(num)
    , degree_bound_(max_deg)
    , dim_(dim)
    , padded_dim_(dim)
    , raw_dist_func_((metric_type == METRIC_IP) ? dot_product_dis<T> : euclidean_sqr<T>)
    , metric_type_(metric_type)
    , rotator_type_(rotator_type) {
    initialize();
}

template <typename T>
inline QuantizedGraph<T>::~QuantizedGraph() {
    delete this->rotator_;
}

template <typename T>
inline void QuantizedGraph<T>::copy_vectors(const T* data) {
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_points_; ++i) {
        const T* src = data + (dim_ * i);
        T* dst = get_vector(i);
        std::copy(src, src + dim_, dst);
    }
    std::cout << "\tVectors Copied\n";
}

template <typename T>
inline void QuantizedGraph<T>::save(const char* filename) const {
    std::cout << "Saving quantized graph to " << filename << '\n';
    std::ofstream output(filename, std::ios::binary);
    if (!output.is_open()) {
        throw std::ios_base::failure("cannot open quantized graph for writing");
    }

    /* Basic variants */
    output.write(reinterpret_cast<const char*>(&num_points_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&degree_bound_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&dim_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&padded_dim_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&entry_point_), sizeof(PID));
    output.write(reinterpret_cast<const char*>(&rotator_type_), sizeof(RotatorType));
    output.write(reinterpret_cast<const char*>(&metric_type_), sizeof(MetricType));

    /* Data */
    data_.save(output);

    /* Rotator */
    this->rotator_->save(output);

    output.close();
    if (output.fail()) {
        throw std::ios_base::failure("failed to save complete quantized graph");
    }
    std::cout << "\tQuantized graph saved!\n";
}

template <typename T>
inline void QuantizedGraph<T>::load(const char* filename) {
    std::cout << "loading quantized graph " << filename << '\n';

    std::ifstream input(filename, std::ios::binary);
    if (!input.is_open()) {
        throw std::ios_base::failure("cannot open quantized graph for reading");
    }

    QuantizedGraph candidate;

    /* Basic variants */
    input.read(reinterpret_cast<char*>(&candidate.num_points_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&candidate.degree_bound_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&candidate.dim_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&candidate.padded_dim_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&candidate.entry_point_), sizeof(PID));
    input.read(reinterpret_cast<char*>(&candidate.rotator_type_), sizeof(RotatorType));
    input.read(reinterpret_cast<char*>(&candidate.metric_type_), sizeof(MetricType));
    if (!input.good()) {
        throw std::ios_base::failure("failed to read quantized graph metadata");
    }

    const auto payload_position = input.tellg();
    if (payload_position == std::streampos(-1)) {
        throw std::ios_base::failure("failed to determine quantized graph payload offset");
    }

    const StorageLayout layout = candidate.compute_storage_layout();
    if (candidate.padded_dim_ != layout.padded_dim) {
        throw std::runtime_error("invalid padded dimension in quantized graph");
    }

    size_t expected_file_bytes = checked_add(
        kSerializedMetadataBytes,
        layout.total_bytes,
        "quantized graph serialized size overflow"
    );
    expected_file_bytes = checked_add(
        expected_file_bytes,
        layout.rotator_bytes,
        "quantized graph serialized size overflow"
    );
    if (expected_file_bytes >
        static_cast<size_t>(std::numeric_limits<std::streamoff>::max())) {
        throw std::length_error("quantized graph file is too large to read");
    }

    input.seekg(0, std::ios::end);
    const std::streamoff file_bytes = input.tellg();
    if (file_bytes < 0) {
        throw std::ios_base::failure("failed to determine quantized graph file size");
    }
    if (file_bytes != static_cast<std::streamoff>(expected_file_bytes)) {
        throw std::ios_base::failure("quantized graph file size does not match metadata");
    }
    input.seekg(payload_position);
    if (!input.good()) {
        throw std::ios_base::failure("failed to seek to quantized graph payload");
    }

    candidate.raw_dist_func_ =
        (candidate.metric_type_ == METRIC_IP) ? dot_product_dis<T> : euclidean_sqr<T>;
    candidate.initialize(layout);

    /* Data */
    constexpr size_t kLoadChunkBytes = size_t{16} * 1024 * 1024;
    const size_t max_stream_bytes =
        static_cast<size_t>(std::numeric_limits<std::streamsize>::max());
    if (layout.row_offset > max_stream_bytes) {
        throw std::length_error("quantized graph row is too large to deserialize");
    }

    const size_t rows_per_chunk =
        std::max<size_t>(size_t{1}, kLoadChunkBytes / layout.row_offset);
    PID maximum_neighbor = 0;
    size_t first_point = 0;
    size_t loaded_bytes = 0;
    while (first_point < candidate.num_points_) {
        const size_t point_count =
            std::min(rows_per_chunk, candidate.num_points_ - first_point);
        const size_t chunk_bytes = checked_multiply(
            point_count, layout.row_offset, "quantized graph load chunk overflow"
        );

        input.read(
            candidate.data_.data() + loaded_bytes, static_cast<std::streamsize>(chunk_bytes)
        );
        if (!input.good()) {
            throw std::ios_base::failure("failed to read complete quantized graph data");
        }

        const char* const chunk = candidate.data_.data() + loaded_bytes;
        for (size_t local_point = 0; local_point < point_count; ++local_point) {
            const auto* neighbors = reinterpret_cast<const PID*>(
                chunk + (local_point * layout.row_offset) + layout.neighbor_offset
            );
            for (size_t edge = 0; edge < candidate.degree_bound_; ++edge) {
                maximum_neighbor = std::max(maximum_neighbor, neighbors[edge]);
            }
        }

        loaded_bytes = checked_add(
            loaded_bytes, chunk_bytes, "quantized graph loaded byte count overflow"
        );
        first_point += point_count;
    }
    if (loaded_bytes != layout.total_bytes) {
        throw std::runtime_error("quantized graph data layout mismatch");
    }
    if (static_cast<size_t>(maximum_neighbor) >= candidate.num_points_) {
        throw std::runtime_error("out-of-range neighbor ID in quantized graph");
    }

    /* Rotator */
    candidate.rotator_->load(input);
    if (!input.good()) {
        throw std::ios_base::failure("failed to read quantized graph rotator");
    }

    candidate.ef_ = ef_;
    swap(candidate);
    std::cout << "Quantized graph loaded!\n";
}

template <typename T>
inline void QuantizedGraph<T>::set_ef(size_t cur_ef) {
    this->ef_ = cur_ef;
}

/**
 * @brief search on qg
 *
 * @param query     unrotated query vector, dimension_ elements
 * @param knn       num of nearest neighbors
 * @param results   search result
 */
template <typename T>
inline void QuantizedGraph<T>::search(
    const T* __restrict__ query, uint32_t k, uint32_t* __restrict__ results
) {
    std::vector<T> rotated_query(padded_dim_);
    rotator_->rotate(query, rotated_query.data());

    // init query
    BatchQuery<T> q_obj(rotated_query.data(), padded_dim_);

    buffer::SearchBuffer<T> search_pool(ef_);
    // init search buffer
    search_pool.insert(this->entry_point_, std::numeric_limits<T>::max());

    buffer::SearchBuffer res_pool(k);  // result buffer
    auto* vis = visited_list_pool_->get_free_vislist();

    std::vector<T> est_dist(degree_bound_);  // estimated distances

    while (search_pool.has_next()) {
        PID cur_node = search_pool.pop();
        if (vis->get(cur_node)) {
            continue;
        }
        vis->set(cur_node);

        q_obj.set_g_add(raw_dist_func_(query, get_vector(cur_node), dim_));

        scan_neighbors(
            q_obj, cur_node, est_dist.data(), search_pool, *vis, this->degree_bound_
        );
        res_pool.insert(cur_node, q_obj.g_add());
    }

    update_results(res_pool, *vis, query);
    visited_list_pool_->release_vis_list(vis);
    res_pool.copy_results(results);
}

template <typename T>
inline void QuantizedGraph<T>::search(
    const T* __restrict__ query,
    uint32_t k,
    uint32_t* __restrict__ results,
    T* __restrict__ dists
) {
    std::vector<T> rotated_query(padded_dim_);
    rotator_->rotate(query, rotated_query.data());

    // init query
    BatchQuery<T> q_obj(rotated_query.data(), padded_dim_);

    buffer::SearchBuffer<T> search_pool(ef_);
    // init search buffer
    search_pool.insert(this->entry_point_, std::numeric_limits<T>::max());

    buffer::SearchBuffer res_pool(k);  // result buffer
    auto* vis = visited_list_pool_->get_free_vislist();

    std::vector<T> est_dist(degree_bound_);  // estimated distances

    while (search_pool.has_next()) {
        PID cur_node = search_pool.pop();
        if (vis->get(cur_node)) {
            continue;
        }
        vis->set(cur_node);

        q_obj.set_g_add(raw_dist_func_(query, get_vector(cur_node), dim_));

        scan_neighbors(
            q_obj, cur_node, est_dist.data(), search_pool, *vis, this->degree_bound_
        );
        res_pool.insert(cur_node, q_obj.g_add());
    }

    update_results(res_pool, *vis, query);
    visited_list_pool_->release_vis_list(vis);
    res_pool.copy_results(results, dists);
}

// scan a data row (including data vec and quantization codes for its neighbors)
// store estimated distance & return exact distnace for current vertex
template <typename T>
void QuantizedGraph<T>::scan_neighbors(
    const BatchQuery<T>& q_obj,
    PID data_id,
    T* est_dist,
    buffer::SearchBuffer<T>& search_pool,
    VisitedSet& vis,
    size_t cur_degree
) const {
    const auto* batch_data = get_batch_data(data_id);
    for (size_t i = 0; i < cur_degree; i += fastscan::kBatchSize) {
        qg_batch_estdist(batch_data, q_obj, padded_dim_, est_dist + i);
        batch_data += QGBatchDataMap<T>::data_bytes(padded_dim_);
    }

    const PID* ptr_nb = get_neighbors(data_id);
    for (size_t i = 0; i < cur_degree; ++i) {
        PID cur_neighbor = ptr_nb[i];
        T dist = est_dist[i];

        if (search_pool.is_full(dist) || vis.get(cur_neighbor)) {
            continue;
        }
        search_pool.insert(cur_neighbor, dist);  // update search buffer
        memory::mem_prefetch_l2(
            reinterpret_cast<const char*>(get_vector(search_pool.next_id())), 10
        );
    }
}

template <typename T>
inline void QuantizedGraph<T>::update_results(
    buffer::SearchBuffer<T>& result_pool, VisitedSet& vis, const T* query
) {
    if (result_pool.is_full()) {
        return;
    }

    auto data = result_pool.data();
    for (auto record : data) {
        PID* ptr_nb = get_neighbors(record.id);
        for (uint32_t i = 0; i < this->degree_bound_; ++i) {
            PID cur_neighbor = ptr_nb[i];
            if (!vis.get(cur_neighbor)) {
                vis.set(cur_neighbor);
                result_pool.insert(
                    cur_neighbor, raw_dist_func_(query, get_vector(cur_neighbor), dim_)
                );
            }
        }
        if (result_pool.is_full()) {
            break;
        }
    }
}

template <typename T>
inline size_t QuantizedGraph<T>::checked_add(
    size_t first, size_t second, const char* message
) {
    if (second > std::numeric_limits<size_t>::max() - first) {
        throw std::length_error(message);
    }
    return first + second;
}

template <typename T>
inline size_t QuantizedGraph<T>::checked_multiply(
    size_t first, size_t second, const char* message
) {
    if (first != 0 && second > std::numeric_limits<size_t>::max() / first) {
        throw std::length_error(message);
    }
    return first * second;
}

template <typename T>
inline auto QuantizedGraph<T>::compute_storage_layout() const -> StorageLayout {
    if (num_points_ == 0 || num_points_ > buffer::kSearchBufferMaxPointCount) {
        throw std::invalid_argument("quantized graph point count must be between 1 and 2^31"
        );
    }
    if (dim_ == 0) {
        throw std::invalid_argument("quantized graph dimension must be positive");
    }
    if (degree_bound_ == 0 || degree_bound_ % fastscan::kBatchSize != 0) {
        throw std::invalid_argument(
            "quantized graph degree must be a positive multiple of 32"
        );
    }
    if (metric_type_ != METRIC_L2 && metric_type_ != METRIC_IP) {
        throw std::invalid_argument("invalid quantized graph metric type");
    }
    if (rotator_type_ != RotatorType::MatrixRotator &&
        rotator_type_ != RotatorType::FhtKacRotator) {
        throw std::invalid_argument("invalid quantized graph rotator type");
    }
    if (rotator_type_ == RotatorType::FhtKacRotator) {
        if (!std::is_same_v<T, float>) {
            throw std::invalid_argument("FHT-Kac rotation requires float data");
        }
        if (dim_ < 64 || dim_ >= 4096) {
            throw std::invalid_argument(
                "FHT-Kac rotation requires a dimension between 64 and 4095"
            );
        }
    }
    if (static_cast<size_t>(entry_point_) >= num_points_) {
        throw std::invalid_argument("quantized graph entry point is out of range");
    }
    if (dim_ > std::numeric_limits<size_t>::max() - 63) {
        throw std::length_error("quantized graph padded dimension overflow");
    }

    StorageLayout layout{};
    layout.padded_dim = ((dim_ + 63) / 64) * 64;
    layout.batch_data_offset =
        checked_multiply(dim_, sizeof(T), "quantized graph vector bytes overflow");
    const size_t packed_code_bits = checked_multiply(
        layout.padded_dim, fastscan::kBatchSize, "quantized graph packed-code size overflow"
    );
    const size_t factor_bytes = checked_multiply(
        sizeof(T), fastscan::kBatchSize * 2, "quantized graph factor size overflow"
    );
    const size_t batch_bytes = checked_add(
        packed_code_bits / 8, factor_bytes, "quantized graph batch size overflow"
    );
    const size_t all_batch_bytes = checked_multiply(
        batch_bytes,
        degree_bound_ / fastscan::kBatchSize,
        "quantized graph batch storage overflow"
    );
    layout.neighbor_offset = checked_add(
        layout.batch_data_offset,
        all_batch_bytes,
        "quantized graph neighbor offset overflow"
    );
    const size_t neighbor_bytes = checked_multiply(
        degree_bound_, sizeof(PID), "quantized graph neighbor storage overflow"
    );
    layout.row_offset = checked_add(
        layout.neighbor_offset, neighbor_bytes, "quantized graph row size overflow"
    );
    layout.total_bytes = checked_multiply(
        num_points_, layout.row_offset, "quantized graph total storage overflow"
    );
    if (rotator_type_ == RotatorType::FhtKacRotator) {
        layout.rotator_bytes =
            checked_multiply(
                layout.padded_dim, 4, "quantized graph rotator size overflow"
            ) /
            8;
    } else {
        layout.rotator_bytes = checked_multiply(
            checked_multiply(
                dim_, layout.padded_dim, "quantized graph rotator size overflow"
            ),
            sizeof(T),
            "quantized graph rotator size overflow"
        );
    }
    return layout;
}

template <typename T>
inline void QuantizedGraph<T>::initialize() {
    initialize(compute_storage_layout());
}

template <typename T>
inline void QuantizedGraph<T>::initialize(const StorageLayout& layout) {
    std::unique_ptr<Rotator<T>> next_rotator(
        choose_rotator<T>(dim_, rotator_type_, layout.padded_dim)
    );
    if (next_rotator->size() != layout.padded_dim ||
        next_rotator->dump_bytes() != layout.rotator_bytes) {
        throw std::runtime_error("rotator returned an invalid storage layout");
    }

    using DataArray =
        Array<char, std::vector<size_t>, memory::AlignedAllocator<char, 1 << 22, true>>;
    DataArray next_data(std::vector<size_t>{num_points_, layout.row_offset});
    auto next_visited_list_pool = std::make_unique<VisitedListPool>(1, num_points_);

    data_ = std::move(next_data);
    delete rotator_;
    rotator_ = next_rotator.release();
    visited_list_pool_ = std::move(next_visited_list_pool);
    padded_dim_ = layout.padded_dim;
    batch_data_offset_ = layout.batch_data_offset;
    neighbor_offset_ = layout.neighbor_offset;
    row_offset_ = layout.row_offset;
}

template <typename T>
inline void QuantizedGraph<T>::swap(QuantizedGraph& other) noexcept {
    using std::swap;
    swap(num_points_, other.num_points_);
    swap(degree_bound_, other.degree_bound_);
    swap(dim_, other.dim_);
    swap(padded_dim_, other.padded_dim_);
    swap(raw_dist_func_, other.raw_dist_func_);
    swap(entry_point_, other.entry_point_);
    swap(metric_type_, other.metric_type_);
    swap(rotator_type_, other.rotator_type_);
    swap(data_, other.data_);
    swap(rotator_, other.rotator_);
    swap(visited_list_pool_, other.visited_list_pool_);
    swap(batch_data_offset_, other.batch_data_offset_);
    swap(neighbor_offset_, other.neighbor_offset_);
    swap(row_offset_, other.row_offset_);
    swap(ef_, other.ef_);
}

// find candidate neighbors for cur_id, exclude the vertex itself
template <typename T>
inline void QuantizedGraph<T>::find_candidates(
    PID cur_id,
    size_t search_ef,
    std::vector<AnnCandidate<T>>& results,
    VisitedSet& vis,
    const std::vector<uint32_t>& degrees
) const {
    const T* query = get_vector(cur_id);
    std::vector<T> rotated_query(padded_dim_);
    rotator_->rotate(query, rotated_query.data());

    // init query
    BatchQuery<T> q_obj(rotated_query.data(), padded_dim_);

    // insert entry point to initialize search buffer
    buffer::SearchBuffer tmp_pool(search_ef);
    tmp_pool.insert(this->entry_point_, 1e10);
    memory::mem_prefetch_l1(
        reinterpret_cast<const char*>(get_vector(this->entry_point_)), 10
    );

    /* Current version of fast scan compute 32 distances */
    std::vector<T> est_dist(degree_bound_);  // estimated distances
    while (tmp_pool.has_next()) {
        auto cur_candi = tmp_pool.pop();
        if (vis.get(cur_candi)) {
            continue;
        }
        vis.set(cur_candi);
        auto cur_degree = degrees[cur_candi];
        q_obj.set_g_add(raw_dist_func_(query, get_vector(cur_candi), dim_));
        scan_neighbors(q_obj, cur_candi, est_dist.data(), tmp_pool, vis, cur_degree);
        if (cur_candi != cur_id) {
            results.emplace_back(cur_candi, q_obj.g_add());
        }
    }
}

// based on new neighbor lists to update quantization code and factors
template <typename T>
inline void QuantizedGraph<T>::update_qg(
    PID cur_id, const std::vector<AnnCandidate<T>>& new_neighbors
) {
    size_t cur_degree = new_neighbors.size();

    if (cur_degree == 0) {
        return;
    }
    // copy neighbors
    PID* neighbor_ptr = get_neighbors(cur_id);
    for (size_t i = 0; i < cur_degree; ++i) {
        neighbor_ptr[i] = new_neighbors[i].id;
    }

    // rotated data
    std::vector<T> rotated_data(cur_degree * padded_dim_);
    std::vector<T> rotated_centroid(padded_dim_);
    for (size_t i = 0; i < cur_degree; ++i) {
        const T* neighbor_vec = get_vector(new_neighbors[i].id);
        this->rotator_->rotate(neighbor_vec, &rotated_data[i * padded_dim_]);
    }
    this->rotator_->rotate(get_vector(cur_id), rotated_centroid.data());

    // quantize batches for current vertex
    auto* batch_data = get_batch_data(cur_id);
    const auto* data = rotated_data.data();
    for (size_t i = 0; i < cur_degree; i += fastscan::kBatchSize) {
        quant::quantize_qg_batch(
            data,
            rotated_centroid.data(),
            std::min(cur_degree - i, fastscan::kBatchSize),
            padded_dim_,
            batch_data,
            metric_type_
        );

        data += fastscan::kBatchSize * padded_dim_;
        batch_data += QGBatchDataMap<T>::data_bytes(padded_dim_);
    }
}
}  // namespace rabitqlib::symqg
