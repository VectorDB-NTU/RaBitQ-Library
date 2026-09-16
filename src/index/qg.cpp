#include "rabitqlib/index/symqg/qg.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <ios>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/fastscan/fastscan.hpp"
#include "rabitqlib/index/query.hpp"
#include "rabitqlib/quantization/data_layout.hpp"
#include "rabitqlib/quantization/pack_excode.hpp"
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/simd/estimator_dispatch.hpp"
#include "rabitqlib/utils/buffer.hpp"
#include "rabitqlib/utils/io.hpp"
#include "rabitqlib/utils/memory.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/utils/space.hpp"
#include "rabitqlib/utils/visited_set.hpp"

namespace rabitqlib::symqg {

QuantizedQuery::QuantizedQuery(
    const float* rotated_query,
    const float* centroid,
    size_t padded_dim,
    MetricType metric_type
)
    : rotated_query_(rotated_query) {
    k1xsumq_ = std::accumulate(rotated_query, rotated_query + padded_dim, 0.0F) / -2;
    g_add_ = metric_type == METRIC_IP ? -dot_product(rotated_query, centroid, padded_dim)
                                      : euclidean_sqr(rotated_query, centroid, padded_dim);
}
const float* QuantizedQuery::rotated_query() const { return rotated_query_; }
float QuantizedQuery::k1xsumq() const { return k1xsumq_; }
float QuantizedQuery::g_add() const { return g_add_; }

size_t QuantizedGraph::checked_add(size_t lhs, size_t rhs) {
    if (lhs > std::numeric_limits<size_t>::max() - rhs) {
        throw std::length_error("QuantizedGraph storage size exceeds size_t");
    }
    return lhs + rhs;
}

size_t QuantizedGraph::checked_multiply(size_t lhs, size_t rhs) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        throw std::length_error("QuantizedGraph storage size exceeds size_t");
    }
    return lhs * rhs;
}

size_t QuantizedGraph::padded_dimension(size_t dim) {
    return (checked_add(dim, 63) / 64) * 64;
}

char* QuantizedGraph::get_row_data(PID data_id) {
    return reinterpret_cast<char*>(get_vector(data_id));
}

const char* QuantizedGraph::get_row_data(PID data_id) const {
    return reinterpret_cast<const char*>(get_vector(data_id));
}

float* QuantizedGraph::get_vector(PID data_id) {
    return data_.data() + ((row_offset_ / sizeof(float)) * data_id);
}

const float* QuantizedGraph::get_vector(PID data_id) const {
    return data_.data() + ((row_offset_ / sizeof(float)) * data_id);
}

char* QuantizedGraph::get_quantized_vector(PID data_id) { return get_row_data(data_id); }

const char* QuantizedGraph::get_quantized_vector(PID data_id) const {
    return get_row_data(data_id);
}

char* QuantizedGraph::get_batch_data(PID data_id) {
    return get_row_data(data_id) + batch_data_offset_;
}

const char* QuantizedGraph::get_batch_data(PID data_id) const {
    return get_row_data(data_id) + batch_data_offset_;
}

rabitqlib::detail::PackedArrayView<PID> QuantizedGraph::get_neighbors(PID data_id) {
    return rabitqlib::detail::PackedArrayView<PID>(
        get_row_data(data_id) + neighbor_offset_
    );
}

rabitqlib::detail::ConstPackedArrayView<PID> QuantizedGraph::get_neighbors(PID data_id
) const {
    return rabitqlib::detail::ConstPackedArrayView<PID>(
        get_row_data(data_id) + neighbor_offset_
    );
}

size_t QuantizedGraph::num_vertices() const { return this->num_points_; }

size_t QuantizedGraph::dimension() const { return this->dim_; }

size_t QuantizedGraph::degree_bound() const { return this->degree_bound_; }

PID QuantizedGraph::entry_point() const { return this->entry_point_; }

MetricType QuantizedGraph::metric_type() const { return this->metric_type_; }

size_t QuantizedGraph::quantization_bits() const { return this->quantization_bits_; }

bool QuantizedGraph::is_quantized() const { return quantization_bits_ != 0; }

void QuantizedGraph::set_ep(PID entry) {
    if (entry >= num_points_) {
        throw std::invalid_argument("QuantizedGraph entry point is out of range");
    }
    entry_point_ = entry;
}

QuantizedGraph::QuantizedGraph() = default;

QuantizedGraph::~QuantizedGraph() = default;

QuantizedGraph::QuantizedGraph(QuantizedGraph&&) noexcept = default;

QuantizedGraph& QuantizedGraph::operator=(QuantizedGraph&&) noexcept = default;

QuantizedGraph::QuantizedGraph(
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
    , raw_dist_func_((metric_type == METRIC_IP) ? dot_product_dis<float> : euclidean_sqr<float>)
    , metric_type_(metric_type)
    , rotator_type_(rotator_type)
    , quantization_bits_(quantization_bits) {
    validate_configuration();
    initialize();
}

void QuantizedGraph::validate_configuration() const {
    validate_metric_type(metric_type_);
    if (dim_ == 0) {
        throw std::invalid_argument("QuantizedGraph dimension must be positive");
    }
    if (rotator_type_ != RotatorType::MatrixRotator &&
        rotator_type_ != RotatorType::FhtKacRotator) {
        throw std::invalid_argument("QuantizedGraph rotator type is invalid");
    }
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
    if (entry_point_ >= num_points_) {
        throw std::invalid_argument("QuantizedGraph entry point is out of range");
    }
    if (quantization_bits_ != 0 && quantization_bits_ != 4 && quantization_bits_ != 8) {
        throw std::invalid_argument(
            "QuantizedGraph quantization bits must be 0 (vanilla), 4, or 8"
        );
    }
}

void QuantizedGraph::copy_vectors(const float* data, size_t num_threads) {
    const int thread_count = static_cast<int>(num_threads);
    if (quantization_bits_ != 0) {
        if (centroid_.size() != padded_dim_) {
            throw std::logic_error("qg-quant centroid must be set before copying vectors");
        }
#pragma omp parallel num_threads(thread_count)
        {
            std::vector<float> rotated_data(padded_dim_);
            std::vector<uint8_t> quantized_data(padded_dim_);
#pragma omp for schedule(dynamic)
            for (size_t i = 0; i < num_points_; ++i) {
                rotator_->rotate(data + (dim_ * i), rotated_data.data());
                ExDataMap<float> output(
                    get_quantized_vector(i), padded_dim_, quantization_bits_
                );
                float f_add;
                float f_rescale;
                float unused_f_error = 0;
                quant::quantize_full_single(
                    rotated_data.data(),
                    centroid_.data(),
                    padded_dim_,
                    quantization_bits_,
                    quantized_data.data(),
                    f_add,
                    f_rescale,
                    unused_f_error,
                    metric_type_
                );
                output.f_add_ex() = f_add;
                output.f_rescale_ex() = f_rescale;
                quant::rabitq_impl::ex_bits::packing_rabitqplus_code(
                    quantized_data.data(), output.ex_code(), padded_dim_, quantization_bits_
                );
            }
        }
        return;
    }
#pragma omp parallel for schedule(dynamic) num_threads(thread_count)
    for (size_t i = 0; i < num_points_; ++i) {
        const float* src = data + (dim_ * i);
        float* dst = get_vector(i);
        std::copy(src, src + dim_, dst);
    }
}

void QuantizedGraph::set_quantization_centroid(const float* centroid) {
    if (quantization_bits_ == 0) {
        return;
    }
    centroid_.resize(padded_dim_);
    rotator_->rotate(centroid, centroid_.data());
}

void QuantizedGraph::save(const char* filename) const {
    if (!ready_ || rotator_ == nullptr) {
        throw std::logic_error("QuantizedGraph must be built or loaded before save");
    }
    if (filename == nullptr || filename[0] == '\0') {
        throw std::invalid_argument("QuantizedGraph save filename must not be empty");
    }
    std::ofstream output(filename, std::ios::binary);
    if (!output.is_open()) {
        throw std::runtime_error("Cannot open quantized graph file for writing");
    }
    output.exceptions(std::ios::badbit | std::ios::failbit);

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
            reinterpret_cast<const char*>(centroid_.data()),
            static_cast<std::streamsize>(padded_dim_ * sizeof(float))
        );
    }

    /* Data */
    output.write(
        get_row_data(0),
        static_cast<std::streamsize>(checked_multiply(num_points_, row_offset_))
    );

    /* Rotator */
    this->rotator_->save(output);

    output.flush();
    output.close();
}

void QuantizedGraph::load(const char* filename) {
    if (filename == nullptr || filename[0] == '\0') {
        throw std::invalid_argument("QuantizedGraph load filename must not be empty");
    }
    /* Check existence */
    if (!file_exists(filename)) {
        throw std::runtime_error("Quantized graph file does not exist");
    }

    std::ifstream input(filename, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("Cannot open quantized graph file");
    }

    auto read_exact = [&](void* destination, size_t bytes, const char* field) {
        if (bytes > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
            throw std::runtime_error("QuantizedGraph field is too large to read");
        }
        input.read(
            reinterpret_cast<char*>(destination), static_cast<std::streamsize>(bytes)
        );
        if (!input) {
            throw std::runtime_error(
                std::string("Truncated QuantizedGraph file while reading ") + field
            );
        }
    };
    auto read_value = [&](auto& value, const char* field) {
        read_exact(&value, sizeof(value), field);
    };

    constexpr uint64_t kFormatMagic = 0x5147524142495451ULL;  // "QGRABITQ"
    constexpr uint32_t kFormatVersion = 1;
    uint64_t magic = 0;
    read_value(magic, "format marker");
    if (magic == kFormatMagic) {
        uint32_t version = 0;
        read_value(version, "format version");
        if (version != kFormatVersion) {
            throw std::runtime_error("Unsupported QuantizedGraph file version");
        }
    } else {
        // Files produced before qg-quant have no header and always contain raw vectors.
        input.clear();
        input.seekg(0);
    }

    QuantizedGraph loaded;
    size_t stored_padded_dim = 0;
    read_value(loaded.num_points_, "point count");
    read_value(loaded.degree_bound_, "degree bound");
    read_value(loaded.dim_, "dimension");
    read_value(stored_padded_dim, "padded dimension");
    read_value(loaded.entry_point_, "entry point");
    read_value(loaded.rotator_type_, "rotator type");
    read_value(loaded.metric_type_, "metric type");
    if (magic == kFormatMagic) {
        read_value(loaded.quantization_bits_, "quantization bits");
    } else {
        loaded.quantization_bits_ = 0;
    }

    loaded.raw_dist_func_ =
        (loaded.metric_type_ == METRIC_IP) ? dot_product_dis<float> : euclidean_sqr<float>;
    loaded.validate_configuration();
    loaded.padded_dim_ = padded_dimension(loaded.dim_);
    if (stored_padded_dim != loaded.padded_dim_) {
        throw std::runtime_error("Invalid padded dimension in quantized graph file");
    }
    loaded.initialize_layout();

    const size_t centroid_bytes = loaded.quantization_bits_ == 0
                                      ? 0
                                      : checked_multiply(loaded.padded_dim_, sizeof(float));
    const size_t data_bytes = checked_multiply(loaded.num_points_, loaded.row_offset_);
    const size_t rotator_bytes =
        loaded.rotator_type_ == RotatorType::MatrixRotator
            ? checked_multiply(
                  checked_multiply(sizeof(float), loaded.dim_), loaded.padded_dim_
              )
            : checked_multiply(loaded.padded_dim_, size_t{4}) / 8;
    const size_t expected_payload_bytes =
        checked_add(checked_add(centroid_bytes, data_bytes), rotator_bytes);

    const auto payload_position = input.tellg();
    if (payload_position < 0) {
        throw std::runtime_error("Cannot determine QuantizedGraph payload position");
    }
    const size_t file_size = get_filesize(filename);
    const size_t payload_offset = static_cast<size_t>(payload_position);
    if (payload_offset > file_size ||
        file_size - payload_offset != expected_payload_bytes) {
        throw std::runtime_error("Invalid QuantizedGraph payload size");
    }

    loaded.initialize();

    if (loaded.quantization_bits_ != 0) {
        loaded.centroid_.resize(loaded.padded_dim_);
        read_exact(loaded.centroid_.data(), centroid_bytes, "quantization centroid");
    }

    read_exact(loaded.get_row_data(0), data_bytes, "graph data");

    for (PID source = 0; source < loaded.num_points_; ++source) {
        const auto neighbors = loaded.get_neighbors(source);
        for (size_t i = 0; i < loaded.degree_bound_; ++i) {
            if (neighbors[i] >= loaded.num_points_) {
                throw std::runtime_error("Invalid QuantizedGraph neighbor ID");
            }
        }
    }

    loaded.rotator_->load(input);
    if (!input) {
        throw std::runtime_error("Truncated QuantizedGraph file while reading rotator");
    }

    input.close();
    // ef is a runtime search setting rather than persisted index state. Preserve
    // the target object's value, matching the previous in-place load behavior.
    loaded.ef_ = ef_;
    loaded.ready_ = true;
    *this = std::move(loaded);
}

void QuantizedGraph::set_ef(size_t cur_ef) {
    if (cur_ef == 0) {
        throw std::invalid_argument("QuantizedGraph ef must be positive");
    }
    this->ef_ = cur_ef;
}

void QuantizedGraph::search(
    const float* __restrict__ query,
    uint32_t k,
    uint32_t* __restrict__ results,
    float* __restrict__ dists
) {
    if (!ready_ || rotator_ == nullptr) {
        throw std::logic_error("QuantizedGraph must be built or loaded before search");
    }
    if (query == nullptr || results == nullptr || dists == nullptr) {
        throw std::invalid_argument("QuantizedGraph search buffers must not be null");
    }
    if (k == 0 || k > num_points_) {
        throw std::invalid_argument("QuantizedGraph k must be between 1 and num_points");
    }
    if (ef_ < k) {
        throw std::invalid_argument("QuantizedGraph ef must be at least k");
    }

    std::vector<float> rotated_query(padded_dim_);
    std::optional<QuantizedQuery> quantized_query;
    prepare_query(query, rotated_query, quantized_query);
    BatchQuery<float> q_obj(rotated_query.data(), padded_dim_, metric_type_);

    buffer::SearchBuffer<float> search_pool(ef_);
    // init search buffer
    search_pool.insert(this->entry_point_, std::numeric_limits<float>::max());

    buffer::SearchBuffer res_pool(k);  // result buffer
    thread_local VisitedSet visited;
    thread_local size_t visited_size = 0;
    if (visited_size != num_points_) {
        visited.initialize(num_points_, num_points_ / 10);
        visited_size = num_points_;
    }
    visited.clear();
    auto* vis = &visited;

    std::vector<float> est_dist(degree_bound_);  // estimated distances

    while (search_pool.has_next()) {
        PID cur_node = search_pool.pop();
        if (vis->get(cur_node)) {
            continue;
        }
        vis->set(cur_node);

        const float vertex_distance =
            point_distance(query, quantized_query ? &*quantized_query : nullptr, cur_node);
        q_obj.set_g_add(vertex_distance);

        scan_neighbors(
            q_obj, cur_node, est_dist.data(), search_pool, *vis, this->degree_bound_
        );
        res_pool.insert(cur_node, vertex_distance);
    }

    update_results(res_pool, *vis, query, quantized_query ? &*quantized_query : nullptr);
    if (res_pool.size() != k) {
        throw std::runtime_error("QuantizedGraph search could not produce k results");
    }
    res_pool.copy_results(results, dists);
}

void QuantizedGraph::prepare_query(
    const float* query,
    std::vector<float>& rotated_query,
    std::optional<QuantizedQuery>& quantized_query
) const {
    rotator_->rotate(query, rotated_query.data());

    if (quantization_bits_ != 0) {
        quantized_query.emplace(
            rotated_query.data(), centroid_.data(), padded_dim_, metric_type_
        );
    }
}

float QuantizedGraph::point_distance(
    const float* raw_query, const QuantizedQuery* quantized_query, PID data_id
) const {
    if (quantized_query != nullptr) {
        return quantized_distance(*quantized_query, data_id);
    }
    return raw_dist_func_(raw_query, get_vector(data_id), dim_);
}

// Scan a data row and store estimated neighbor distances. The caller scores the current
// vertex from either its raw vector (vanilla QG) or its 4/8-bit code (qg-quant).
void QuantizedGraph::scan_neighbors(
    const BatchQuery<float>& q_obj,
    PID data_id,
    float* est_dist,
    buffer::SearchBuffer<float>& search_pool,
    VisitedSet& vis,
    size_t cur_degree
) const {
    const auto* batch_data = get_batch_data(data_id);
    const auto neighbors = get_neighbors(data_id);
    for (size_t begin = 0; begin < cur_degree; begin += fastscan::kBatchSize) {
        const float threshold = search_pool.top_dist();
        uint32_t candidate_mask = simd::qg_batch_estdist_mask(
            batch_data, q_obj, padded_dim_, est_dist + begin, threshold
        );
        batch_data += QGBatchDataMap<float>::data_bytes(padded_dim_);

        // Construction can leave a partial batch with stale IDs in its unused lanes.
        const size_t remaining = cur_degree - begin;
        if (remaining < fastscan::kBatchSize) {
            candidate_mask &= (uint32_t{1} << remaining) - 1;
        }

        while (candidate_mask != 0) {
            const auto lane = static_cast<size_t>(__builtin_ctz(candidate_mask));
            candidate_mask &= candidate_mask - 1;
            const size_t i = begin + lane;
            PID cur_neighbor = neighbors[i];
            float dist = est_dist[i];

            if (search_pool.is_full(dist) || vis.get(cur_neighbor)) {
                continue;
            }
            search_pool.insert(cur_neighbor, dist);  // update search buffer
            memory::mem_prefetch_l2(get_row_data(search_pool.next_id()), 10);
        }
    }
}

void QuantizedGraph::update_results(
    buffer::SearchBuffer<float>& result_pool,
    VisitedSet& vis,
    const float* query,
    const QuantizedQuery* quantized_query
) {
    if (result_pool.is_full()) {
        return;
    }

    const auto& pool_data = result_pool.data();
    const std::vector<AnnCandidate<float>> data(
        pool_data.begin(), pool_data.begin() + static_cast<ptrdiff_t>(result_pool.size())
    );
    for (const auto& record : data) {
        auto neighbors = get_neighbors(record.id);
        for (uint32_t i = 0; i < this->degree_bound_; ++i) {
            PID cur_neighbor = neighbors[i];
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
void QuantizedGraph::initialize_layout() {
    if (quantization_bits_ == 0) {
        batch_data_offset_ = checked_multiply(dim_, sizeof(float));
    } else {
        const size_t code_bits = checked_multiply(padded_dim_, quantization_bits_);
        batch_data_offset_ = checked_add(code_bits / 8, checked_multiply(sizeof(float), 2));
    }

    const size_t binary_batch_bytes =
        checked_multiply(padded_dim_, fastscan::kBatchSize) / 8;
    const size_t factor_bytes =
        checked_multiply(checked_multiply(sizeof(float), fastscan::kBatchSize), size_t{2});
    const size_t batch_bytes = checked_add(binary_batch_bytes, factor_bytes);
    neighbor_offset_ = checked_add(
        batch_data_offset_,
        checked_multiply(batch_bytes, degree_bound_ / fastscan::kBatchSize)
    );
    row_offset_ =
        checked_add(neighbor_offset_, checked_multiply(degree_bound_, sizeof(PID)));
}

void QuantizedGraph::initialize() {
    padded_dim_ = padded_dimension(dim_);
    rotator_.reset(choose_rotator<float>(dim_, rotator_type_, padded_dim_));

    assert(padded_dim_ % 64 == 0);
    assert(padded_dim_ >= dim_);

    initialize_layout();

    const size_t data_bytes = checked_multiply(num_points_, row_offset_);
    assert(row_offset_ % sizeof(float) == 0);
    data_ = RowStorage(data_bytes / sizeof(float));

    if (quantization_bits_ != 0) {
        quantized_ip_func_ = select_excode_ipfunc(quantization_bits_);
    }
}

float QuantizedGraph::quantized_distance(const QuantizedQuery& query, PID data_id) const {
    ConstExDataMap<float> data(
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

void QuantizedGraph::reconstruct_quantized_vector(PID data_id, float* reconstructed) const {
    ConstExDataMap<float> data(
        get_quantized_vector(data_id), padded_dim_, quantization_bits_
    );
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

// Construction sources come from owned raw rows or from the existing RaBitQ codes.
// Reconstructed sources are already rotated; never pass them through prepare_query.
const float* QuantizedGraph::prepare_build_query(
    PID id, std::vector<float>& rotated, std::optional<QuantizedQuery>& prepared
) const {
    if (is_quantized()) {
        rotated.resize(padded_dim_);
        reconstruct_quantized_vector(id, rotated.data());
        prepared.emplace(rotated.data(), centroid_.data(), padded_dim_, metric_type_);
        return rotated.data();
    }
    prepared.reset();
    return get_vector(id);
}

// find candidate neighbors for cur_id, exclude the vertex itself
void QuantizedGraph::find_candidates(
    PID cur_id,
    size_t search_ef,
    std::vector<AnnCandidate<float>>& results,
    VisitedSet& vis,
    const std::vector<uint32_t>& degrees
) const {
    std::vector<float> rotated_query(padded_dim_);
    std::optional<QuantizedQuery> quantized_query;
    const float* query = prepare_build_query(cur_id, rotated_query, quantized_query);
    if (!is_quantized()) {
        rotator_->rotate(query, rotated_query.data());
    }
    BatchQuery<float> q_obj(rotated_query.data(), padded_dim_, metric_type_);

    // insert entry point to initialize search buffer
    buffer::SearchBuffer tmp_pool(search_ef);
    tmp_pool.insert(this->entry_point_, std::numeric_limits<float>::max());

    /* Current version of fast scan compute 32 distances */
    std::vector<float> est_dist(degree_bound_);  // estimated distances
    while (tmp_pool.has_next()) {
        auto cur_candi = tmp_pool.pop();
        if (vis.get(cur_candi)) {
            continue;
        }
        vis.set(cur_candi);
        auto cur_degree = degrees[cur_candi];
        const float vertex_distance =
            point_distance(query, quantized_query ? &*quantized_query : nullptr, cur_candi);
        q_obj.set_g_add(vertex_distance);
        scan_neighbors(q_obj, cur_candi, est_dist.data(), tmp_pool, vis, cur_degree);
        if (cur_candi != cur_id) {
            results.emplace_back(cur_candi, vertex_distance);
        }
    }
}

// based on new neighbor lists to update quantization code and factors
void QuantizedGraph::update_qg(
    PID cur_id, const std::vector<AnnCandidate<float>>& new_neighbors
) {
    size_t cur_degree = new_neighbors.size();

    if (cur_degree == 0) {
        return;
    }
    // copy neighbors
    auto neighbors = get_neighbors(cur_id);
    for (size_t i = 0; i < cur_degree; ++i) {
        neighbors[i] = new_neighbors[i].id;
    }

    // rotated data
    std::vector<float> rotated_data(cur_degree * padded_dim_);
    std::vector<float> rotated_centroid(padded_dim_);
    for (size_t i = 0; i < cur_degree; ++i) {
        if (quantization_bits_ == 0) {
            const float* neighbor_vec = get_vector(new_neighbors[i].id);
            this->rotator_->rotate(neighbor_vec, &rotated_data[i * padded_dim_]);
        } else {
            reconstruct_quantized_vector(
                new_neighbors[i].id, &rotated_data[i * padded_dim_]
            );
        }
    }
    if (quantization_bits_ == 0) {
        this->rotator_->rotate(get_vector(cur_id), rotated_centroid.data());
    } else {
        reconstruct_quantized_vector(cur_id, rotated_centroid.data());
    }

    // quantize batches for current vertex
    auto* batch_data = get_batch_data(cur_id);
    for (size_t i = 0; i < cur_degree; i += fastscan::kBatchSize) {
        const size_t batch_size = std::min(cur_degree - i, fastscan::kBatchSize);
        quant::quantize_qg_batch(
            rotated_data.data() + (i * padded_dim_),
            rotated_centroid.data(),
            batch_size,
            padded_dim_,
            batch_data,
            metric_type_
        );
        if (batch_size < fastscan::kBatchSize) {
            QGBatchDataMap<float> batch(batch_data, padded_dim_);
            std::fill(
                batch.f_add() + static_cast<ptrdiff_t>(batch_size),
                batch.f_add() + fastscan::kBatchSize,
                static_cast<float>(0)
            );
            std::fill(
                batch.f_rescale() + static_cast<ptrdiff_t>(batch_size),
                batch.f_rescale() + fastscan::kBatchSize,
                static_cast<float>(0)
            );
        }

        batch_data += QGBatchDataMap<float>::data_bytes(padded_dim_);
    }
}
}  // namespace rabitqlib::symqg
