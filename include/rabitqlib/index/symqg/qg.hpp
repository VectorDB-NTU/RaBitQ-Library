#pragma once

#include <omp.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <optional>
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
#include "rabitqlib/utils/io.hpp"
#include "rabitqlib/utils/memory.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/utils/space.hpp"
#include "rabitqlib/utils/visited_pool.hpp"
#include "rabitqlib/utils/visited_set.hpp"

namespace rabitqlib::symqg {

template <typename T>
class QuantizedQuery {
   private:
    const T* rotated_query_;
    T k1xsumq_;
    T g_add_;

   public:
    QuantizedQuery(
        const T* rotated_query, const T* centroid, size_t padded_dim, MetricType metric_type
    )
        : rotated_query_(rotated_query) {
        k1xsumq_ =
            std::accumulate(rotated_query, rotated_query + padded_dim, static_cast<T>(0)) /
            -2;

        g_add_ = metric_type == METRIC_IP
                     ? -dot_product(rotated_query, centroid, padded_dim)
                     : euclidean_sqr(rotated_query, centroid, padded_dim);
    }

    [[nodiscard]] const T* rotated_query() const { return rotated_query_; }
    [[nodiscard]] T k1xsumq() const { return k1xsumq_; }
    [[nodiscard]] T g_add() const { return g_add_; }
};

template <typename T = float>
class QuantizedGraph {
    friend class QGBuilder;

   private:
    size_t num_points_ = 0;                           // num points
    size_t degree_bound_ = 0;                         // degree bound
    size_t dim_ = 0;                                  // dimension
    size_t padded_dim_ = 0;                           // padded dimension
    T (*raw_dist_func_)(const T*, const T*, size_t);  // dist func for raw vector
    PID entry_point_ = 0;                             // Entry point of graph
    MetricType metric_type_ = MetricType::METRIC_L2;
    RotatorType rotator_type_ = RotatorType::FhtKacRotator;
    size_t quantization_bits_ = 0;   // 0: raw vectors, 4/8: packed RaBitQ vectors
    const T* build_data_ = nullptr;  // non-owning, used only while QGBuilder runs
    std::vector<T> centroid_;        // rotated global centroid for qg-quant
    ex_ipfunc quantized_ip_func_ = nullptr;

    Array<
        char,
        std::vector<size_t>,
        memory::AlignedAllocator<
            char,
            1 << 22,
            true>>
        data_;  // vectors/codes + graph quantization data + edges
    std::unique_ptr<Rotator<T>> rotator_;  // data rotator
    std::unique_ptr<VisitedListPool> visited_list_pool_ = nullptr;

    // Position of row data (raw vector or packed qg-quant vector), neighbor
    // quantization data, and neighbor IDs. Since every degree equals degree_bound_
    // (a multiple of 32), the degree does not need to be stored per vertex.
    size_t batch_data_offset_ = 0;  // offset of qg batch data
    size_t neighbor_offset_ = 0;    // offset of neighbors
    size_t row_offset_ = 0;         // length of entire row
    size_t ef_ = 0;

    void validate_configuration() const;

    void initialize();

    void copy_vectors(const T*);

    void set_quantization_centroid(const T* centroid);

    [[nodiscard]] T* get_vector(PID data_id) {
        return reinterpret_cast<T*>(&data_.at(row_offset_ * data_id));
    }

    [[nodiscard]] const T* get_vector(PID data_id) const {
        return reinterpret_cast<const T*>(&data_.at(row_offset_ * data_id));
    }

    [[nodiscard]] const T* get_build_vector(PID data_id) const {
        return build_data_ + (dim_ * data_id);
    }

    [[nodiscard]] char* get_quantized_vector(PID data_id) {
        return &data_.at(row_offset_ * data_id);
    }

    [[nodiscard]] const char* get_quantized_vector(PID data_id) const {
        return &data_.at(row_offset_ * data_id);
    }

    void prepare_query(const T*, std::vector<T>&, std::optional<QuantizedQuery<T>>&) const;

    T point_distance(const T*, const QuantizedQuery<T>*, PID) const;

    T quantized_distance(const QuantizedQuery<T>&, PID) const;

    void reconstruct_quantized_vector(PID, T*) const;

    [[nodiscard]] char* get_batch_data(PID data_id) {
        return &data_.at((row_offset_ * data_id) + batch_data_offset_);
    }

    [[nodiscard]] const char* get_batch_data(PID data_id) const {
        return &data_.at((row_offset_ * data_id) + batch_data_offset_);
    }

    [[nodiscard]] PID* get_neighbors(PID data_id) {
        return reinterpret_cast<PID*>(&data_.at((row_offset_ * data_id) + neighbor_offset_)
        );
    }

    [[nodiscard]] const PID* get_neighbors(PID data_id) const {
        return reinterpret_cast<const PID*>(
            &data_.at((row_offset_ * data_id) + neighbor_offset_)
        );
    }

    void
    find_candidates(PID, size_t, std::vector<AnnCandidate<T>>&, VisitedSet&, const std::vector<uint32_t>&)
        const;

    void update_qg(PID, const std::vector<AnnCandidate<T>>&);

    void
    update_results(buffer::SearchBuffer<T>&, VisitedSet&, const T*, const QuantizedQuery<T>*);

    void scan_neighbors(
        const BatchQuery<T>&, PID, T*, buffer::SearchBuffer<T>&, VisitedSet&, size_t
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

    explicit QuantizedGraph() = default;

    ~QuantizedGraph() = default;

    [[nodiscard]] auto num_vertices() const { return this->num_points_; }

    [[nodiscard]] auto dimension() const { return this->dim_; }

    [[nodiscard]] auto degree_bound() const { return this->degree_bound_; }

    [[nodiscard]] auto entry_point() const { return this->entry_point_; }

    [[nodiscard]] auto metric_type() const { return this->metric_type_; }

    [[nodiscard]] auto quantization_bits() const { return this->quantization_bits_; }

    [[nodiscard]] bool is_quantized() const { return quantization_bits_ != 0; }

    void set_ep(PID entry) { this->entry_point_ = entry; };

    void save(const char*) const;

    void load(const char*);

    void set_ef(size_t);

    /* search and copy results to KNN */
    void search(
        const T* __restrict__ query,
        uint32_t knn,
        uint32_t* __restrict__ results,
        T* __restrict__ dists
    );
};

template <typename T>
inline QuantizedGraph<T>::QuantizedGraph(
    size_t num,
    size_t dim,
    size_t max_deg,
    MetricType metric_type,
    RotatorType rotator_type,
    size_t quantization_bits
)
    : num_points_(num)
    , degree_bound_(max_deg)
    , dim_(dim)
    , padded_dim_(dim)
    , raw_dist_func_((metric_type == METRIC_IP) ? dot_product_dis<T> : euclidean_sqr<T>)
    , metric_type_(metric_type)
    , rotator_type_(rotator_type)
    , quantization_bits_(quantization_bits) {
    validate_configuration();
    initialize();
}

template <typename T>
inline void QuantizedGraph<T>::validate_configuration() const {
    validate_metric_type(metric_type_);
    if (degree_bound_ == 0 || degree_bound_ % fastscan::kBatchSize != 0) {
        throw std::invalid_argument(
            "QuantizedGraph degree bound must be a positive multiple of 32"
        );
    }
    if (degree_bound_ >= num_points_) {
        throw std::invalid_argument(
            "QuantizedGraph degree bound must be smaller than the number of points"
        );
    }
    if (num_points_ > buffer::kSearchBufferMaxPointCount) {
        throw std::invalid_argument(
            "QuantizedGraph point count exceeds the search-buffer ID limit"
        );
    }
    if (quantization_bits_ != 0 && quantization_bits_ != 4 && quantization_bits_ != 8) {
        throw std::invalid_argument(
            "QuantizedGraph quantization bits must be 0 (vanilla), 4, or 8"
        );
    }
    if (quantization_bits_ != 0 && !std::is_same_v<T, float>) {
        throw std::invalid_argument("QuantizedGraph qg-quant currently requires float data"
        );
    }
}

template <typename T>
inline void QuantizedGraph<T>::copy_vectors(const T* data) {
    build_data_ = data;
    if (quantization_bits_ != 0) {
        if constexpr (!std::is_same_v<T, float>) {
            throw std::logic_error("qg-quant currently requires float data");
        } else {
            if (centroid_.size() != padded_dim_) {
                throw std::logic_error(
                    "qg-quant centroid must be set before copying vectors"
                );
            }
#pragma omp parallel
            {
                std::vector<T> rotated_data(padded_dim_);
                std::vector<uint8_t> quantized_data(padded_dim_);
#pragma omp for schedule(dynamic)
                for (size_t i = 0; i < num_points_; ++i) {
                    rotator_->rotate(data + (dim_ * i), rotated_data.data());
                    ExDataMap<T> output(
                        get_quantized_vector(i), padded_dim_, quantization_bits_
                    );
                    T unused_f_error = 0;
                    quant::quantize_full_single(
                        rotated_data.data(),
                        centroid_.data(),
                        padded_dim_,
                        quantization_bits_,
                        quantized_data.data(),
                        output.f_add_ex(),
                        output.f_rescale_ex(),
                        unused_f_error,
                        metric_type_
                    );
                    quant::rabitq_impl::ex_bits::packing_rabitqplus_code(
                        quantized_data.data(),
                        output.ex_code(),
                        padded_dim_,
                        quantization_bits_
                    );
                }
            }
            return;
        }
    }
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_points_; ++i) {
        const T* src = data + (dim_ * i);
        T* dst = get_vector(i);
        std::copy(src, src + dim_, dst);
    }
}

template <typename T>
inline void QuantizedGraph<T>::set_quantization_centroid(const T* centroid) {
    if (quantization_bits_ == 0) {
        return;
    }
    centroid_.resize(padded_dim_);
    rotator_->rotate(centroid, centroid_.data());
}

template <typename T>
inline void QuantizedGraph<T>::save(const char* filename) const {
    std::ofstream output(filename, std::ios::binary);
    if (!output.is_open()) {
        throw std::runtime_error("Cannot open quantized graph file for writing");
    }

    constexpr uint64_t kFormatMagic = 0x5147524142495451ULL;  // "QGRABITQ"
    constexpr uint32_t kFormatVersion = 1;
    if (quantization_bits_ != 0) {
        output.write(reinterpret_cast<const char*>(&kFormatMagic), sizeof(kFormatMagic));
        output.write(
            reinterpret_cast<const char*>(&kFormatVersion), sizeof(kFormatVersion)
        );
    }

    /* Basic variants */
    output.write(reinterpret_cast<const char*>(&num_points_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&degree_bound_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&dim_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&padded_dim_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&entry_point_), sizeof(PID));
    output.write(reinterpret_cast<const char*>(&rotator_type_), sizeof(RotatorType));
    output.write(reinterpret_cast<const char*>(&metric_type_), sizeof(MetricType));
    if (quantization_bits_ != 0) {
        output.write(
            reinterpret_cast<const char*>(&quantization_bits_), sizeof(quantization_bits_)
        );
        output.write(
            reinterpret_cast<const char*>(centroid_.data()), padded_dim_ * sizeof(T)
        );
    }

    /* Data */
    data_.save(output);

    /* Rotator */
    this->rotator_->save(output);

    output.close();
}

template <typename T>
inline void QuantizedGraph<T>::load(const char* filename) {
    /* Check existence */
    if (!file_exists(filename)) {
        throw std::runtime_error("Quantized graph file does not exist");
    }

    std::ifstream input(filename, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("Cannot open quantized graph file");
    }

    constexpr uint64_t kFormatMagic = 0x5147524142495451ULL;  // "QGRABITQ"
    constexpr uint32_t kFormatVersion = 1;
    uint64_t magic = 0;
    input.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic == kFormatMagic) {
        uint32_t version = 0;
        input.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (version != kFormatVersion) {
            throw std::runtime_error("Unsupported QuantizedGraph file version");
        }
    } else {
        // Files produced before qg-quant have no header and always contain raw vectors.
        input.clear();
        input.seekg(0);
    }

    /* Basic variants */
    input.read(reinterpret_cast<char*>(&num_points_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&degree_bound_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&dim_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&padded_dim_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&entry_point_), sizeof(PID));
    input.read(reinterpret_cast<char*>(&rotator_type_), sizeof(RotatorType));
    input.read(reinterpret_cast<char*>(&metric_type_), sizeof(MetricType));
    if (magic == kFormatMagic) {
        input.read(
            reinterpret_cast<char*>(&quantization_bits_), sizeof(quantization_bits_)
        );
    } else {
        quantization_bits_ = 0;
    }

    raw_dist_func_ = (metric_type_ == METRIC_IP) ? dot_product_dis<T> : euclidean_sqr<T>;

    validate_configuration();
    initialize();

    if (quantization_bits_ != 0) {
        centroid_.resize(padded_dim_);
        input.read(reinterpret_cast<char*>(centroid_.data()), padded_dim_ * sizeof(T));
    }

    /* Data */
    data_.load(input);

    /* Rotator */
    this->rotator_->load(input);
    if (rotator_->size() != padded_dim_) {
        throw std::runtime_error("Invalid padded dimension in quantized graph file");
    }

    input.close();
}

template <typename T>
inline void QuantizedGraph<T>::set_ef(size_t cur_ef) {
    this->ef_ = cur_ef;
}

template <typename T>
inline void QuantizedGraph<T>::search(
    const T* __restrict__ query,
    uint32_t k,
    uint32_t* __restrict__ results,
    T* __restrict__ dists
) {
    std::vector<T> rotated_query(padded_dim_);
    std::optional<QuantizedQuery<T>> quantized_query;
    prepare_query(query, rotated_query, quantized_query);
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

        q_obj.set_g_add(
            point_distance(query, quantized_query ? &*quantized_query : nullptr, cur_node)
        );

        scan_neighbors(
            q_obj, cur_node, est_dist.data(), search_pool, *vis, this->degree_bound_
        );
        res_pool.insert(cur_node, q_obj.g_add());
    }

    update_results(res_pool, *vis, query, quantized_query ? &*quantized_query : nullptr);
    visited_list_pool_->release_vis_list(vis);
    res_pool.copy_results(results, dists);
}

template <typename T>
inline void QuantizedGraph<T>::prepare_query(
    const T* query,
    std::vector<T>& rotated_query,
    std::optional<QuantizedQuery<T>>& quantized_query
) const {
    rotator_->rotate(query, rotated_query.data());

    if (quantization_bits_ != 0) {
        quantized_query.emplace(
            rotated_query.data(), centroid_.data(), padded_dim_, metric_type_
        );
    }
}

template <typename T>
inline T QuantizedGraph<T>::point_distance(
    const T* raw_query, const QuantizedQuery<T>* quantized_query, PID data_id
) const {
    if (quantized_query != nullptr) {
        return quantized_distance(*quantized_query, data_id);
    }
    return raw_dist_func_(raw_query, get_vector(data_id), dim_);
}

// Scan a data row and store estimated neighbor distances. The caller scores the current
// vertex from either its raw vector (vanilla QG) or its 4/8-bit code (qg-quant).
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
    buffer::SearchBuffer<T>& result_pool,
    VisitedSet& vis,
    const T* query,
    const QuantizedQuery<T>* quantized_query
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
                    cur_neighbor, point_distance(query, quantized_query, cur_neighbor)
                );
            }
        }
        if (result_pool.is_full()) {
            break;
        }
    }
}

// initialize const offsets & data array
template <typename T>
inline void QuantizedGraph<T>::initialize() {
    rotator_.reset(
        choose_rotator<float>(dim_, rotator_type_, round_up_to_multiple(dim_, 64))
    );
    padded_dim_ = rotator_->size();

    /* check size */
    assert(padded_dim_ % 64 == 0);
    assert(padded_dim_ >= dim_);

    this->batch_data_offset_ =
        quantization_bits_ == 0 ? dim_ * sizeof(T)
                                : ExDataMap<T>::data_bytes(padded_dim_, quantization_bits_);
    this->neighbor_offset_ =
        batch_data_offset_ + (QGBatchDataMap<T>::data_bytes(padded_dim_) *
                              (degree_bound_ / fastscan::kBatchSize));
    this->row_offset_ = neighbor_offset_ + (degree_bound_ * sizeof(PID));

    data_ = Array<char, std::vector<size_t>, memory::AlignedAllocator<char, 1 << 22, true>>(
        std::vector<size_t>{num_points_, row_offset_}
    );

    visited_list_pool_ = std::make_unique<VisitedListPool>(1, num_points_);
    if (quantization_bits_ != 0) {
        quantized_ip_func_ = select_excode_ipfunc(quantization_bits_);
    }
}

template <typename T>
inline T QuantizedGraph<T>::quantized_distance(const QuantizedQuery<T>& query, PID data_id)
    const {
    if constexpr (!std::is_same_v<T, float>) {
        throw std::logic_error("qg-quant currently requires float data");
    } else {
        ConstExDataMap<T> data(
            get_quantized_vector(data_id), padded_dim_, quantization_bits_
        );
        return quant::full_est_dist(
            data.ex_code(),
            query.rotated_query(),
            quantized_ip_func_,
            padded_dim_,
            quantization_bits_,
            data.f_add_ex(),
            data.f_rescale_ex(),
            query.g_add(),
            query.k1xsumq()
        );
    }
}

template <typename T>
inline void QuantizedGraph<T>::reconstruct_quantized_vector(PID data_id, T* reconstructed)
    const {
    ConstExDataMap<T> data(get_quantized_vector(data_id), padded_dim_, quantization_bits_);
    std::vector<uint8_t> quantized_data(padded_dim_);
    if (quantization_bits_ == 8) {
        std::copy(data.ex_code(), data.ex_code() + padded_dim_, quantized_data.begin());
    } else {
        for (size_t i = 0; i < padded_dim_; i += 16) {
            uint64_t packed = 0;
            std::memcpy(&packed, data.ex_code() + (i / 2), sizeof(packed));
            for (size_t j = 0; j < 8; ++j) {
                const uint8_t pair = static_cast<uint8_t>(packed >> (j * 8));
                quantized_data[i + j] = pair & 0x0f;
                quantized_data[i + 8 + j] = pair >> 4;
            }
        }
    }
    quant::reconstruct_full_vec(
        quantized_data.data(),
        centroid_.data(),
        padded_dim_,
        quantization_bits_,
        data.f_rescale_ex(),
        reconstructed,
        metric_type_
    );
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
    const T* query = get_build_vector(cur_id);
    std::vector<T> rotated_query(padded_dim_);
    std::optional<QuantizedQuery<T>> quantized_query;
    prepare_query(query, rotated_query, quantized_query);
    BatchQuery<T> q_obj(rotated_query.data(), padded_dim_);

    // insert entry point to initialize search buffer
    buffer::SearchBuffer tmp_pool(search_ef);
    tmp_pool.insert(this->entry_point_, std::numeric_limits<T>::max());

    /* Current version of fast scan compute 32 distances */
    std::vector<T> est_dist(degree_bound_);  // estimated distances
    while (tmp_pool.has_next()) {
        auto cur_candi = tmp_pool.pop();
        if (vis.get(cur_candi)) {
            continue;
        }
        vis.set(cur_candi);
        auto cur_degree = degrees[cur_candi];
        q_obj.set_g_add(
            point_distance(query, quantized_query ? &*quantized_query : nullptr, cur_candi)
        );
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
        const T* neighbor_vec = get_build_vector(new_neighbors[i].id);
        this->rotator_->rotate(neighbor_vec, &rotated_data[i * padded_dim_]);
    }
    if (quantization_bits_ == 0) {
        this->rotator_->rotate(get_build_vector(cur_id), rotated_centroid.data());
    } else {
        reconstruct_quantized_vector(cur_id, rotated_centroid.data());
    }

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
