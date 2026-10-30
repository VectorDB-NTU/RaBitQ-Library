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

#include "rabitqlib/core/data_layout.hpp"
#include "rabitqlib/core/encoder.hpp"
#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/core/pack_excode.hpp"
#include "rabitqlib/core/rotator.hpp"
#include "rabitqlib/fastscan/data_layout.hpp"
#include "rabitqlib/fastscan/detail/estimator_dispatch.hpp"
#include "rabitqlib/fastscan/encoder.hpp"
#include "rabitqlib/fastscan/pack_codes.hpp"
#include "rabitqlib/fastscan/query.hpp"
#include "rabitqlib/index/detail/buffer.hpp"
#include "rabitqlib/index/detail/parallel_exception.hpp"
#include "rabitqlib/index/detail/visited_set.hpp"
#include "rabitqlib/index/symqg/detail/search_scratch.hpp"
#include "rabitqlib/index/types.hpp"
#include "rabitqlib/utils/bitops.hpp"
#include "rabitqlib/utils/io.hpp"
#include "rabitqlib/utils/memory.hpp"
#include "rabitqlib/utils/path.hpp"
#include "rabitqlib/utils/space.hpp"
#include "rabitqlib/utils/tools.hpp"

namespace rabitqlib::symqg {

namespace {
// Native-endian numeric tag; little-endian files contain the bytes "QTIBARGQ".
constexpr uint64_t kFormatMagic = 0x5147524142495451ULL;
constexpr uint32_t kFormatVersion = 1;
}  // namespace

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
QuantizedQuery::QuantizedQuery(const float* rotated_query, const QuantizedQuery& prepared)
    : rotated_query_(rotated_query), k1xsumq_(prepared.k1xsumq_), g_add_(prepared.g_add_) {}
const float* QuantizedQuery::rotated_query() const { return rotated_query_; }
float QuantizedQuery::k1xsumq() const { return k1xsumq_; }
float QuantizedQuery::g_add() const { return g_add_; }

size_t QuantizedGraph<float>::checked_add(size_t lhs, size_t rhs) {
    if (lhs > std::numeric_limits<size_t>::max() - rhs) {
        throw std::length_error("QuantizedGraph storage size exceeds size_t");
    }
    return lhs + rhs;
}

size_t QuantizedGraph<float>::checked_multiply(size_t lhs, size_t rhs) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        throw std::length_error("QuantizedGraph storage size exceeds size_t");
    }
    return lhs * rhs;
}

size_t QuantizedGraph<float>::padded_dimension(size_t dim) {
    // Use the same rotated domain as the other indexes.
    return (checked_add(dim, 31) / 32) * 32;
}

size_t QuantizedGraph<float>::num_vertices() const { return this->max_elements_; }

size_t QuantizedGraph<float>::max_elements() const { return max_elements_; }

size_t QuantizedGraph<float>::num_points() const { return ready_ ? max_elements_ : 0; }

size_t QuantizedGraph<float>::num_stored_points() const {
    return ready_ ? max_elements_ : 0;
}

size_t QuantizedGraph<float>::dimension() const { return this->dim_; }

size_t QuantizedGraph<float>::degree_bound() const { return this->max_degree_; }

PID QuantizedGraph<float>::entry_point() const { return this->entry_point_id_; }

MetricType QuantizedGraph<float>::metric_type() const { return this->metric_type_; }

size_t QuantizedGraph<float>::quantization_bits() const { return this->quantization_bits_; }

bool QuantizedGraph<float>::is_quantized() const { return quantization_bits_ != 0; }

void QuantizedGraph<float>::set_ep(PID entry) {
    if (entry >= max_elements_) {
        throw std::invalid_argument("QuantizedGraph entry point is out of range");
    }
    entry_point_id_ = entry;
}

QuantizedGraph<float>::QuantizedGraph() = default;

QuantizedGraph<float>::~QuantizedGraph() = default;

QuantizedGraph<float>::QuantizedGraph(QuantizedGraph<float>&&) noexcept = default;

QuantizedGraph<float>& QuantizedGraph<float>::operator=(QuantizedGraph<float>&&) noexcept =
    default;

QuantizedGraph<float>::QuantizedGraph(
    size_t max_elements,
    size_t dim,
    size_t max_degree,
    MetricType metric_type,
    RotatorType rotator_type,
    size_t quantization_bits,
    uint32_t seed
)
    : max_elements_(max_elements)
    , max_degree_(max_degree)
    , dim_(dim)
    , padded_dim_(dim)
    , raw_dist_func_(
          (metric_type == METRIC_IP) ? dot_product_dis<float> : euclidean_sqr<float>
      )
    , metric_type_(metric_type)
    , rotator_type_(rotator_type)
    , quantization_bits_(quantization_bits)
    , seed_(seed) {
    validate_configuration();
    padded_dim_ = padded_dimension(dim_);
    initialize();
}

void QuantizedGraph<float>::validate_configuration() const {
    validate_metric_type(metric_type_);
    if (dim_ == 0) {
        throw std::invalid_argument("QuantizedGraph dimension must be positive");
    }
    if (rotator_type_ != RotatorType::MatrixRotator &&
        rotator_type_ != RotatorType::FhtKacRotator) {
        throw std::invalid_argument("QuantizedGraph rotator type is invalid");
    }
    if (max_degree_ == 0 || max_degree_ % fastscan::kBatchSize != 0) {
        throw std::invalid_argument(
            "QuantizedGraph degree bound must be a positive multiple of 32"
        );
    }
    if (max_degree_ >= max_elements_) {
        throw std::invalid_argument(
            "QuantizedGraph degree bound must be smaller than the number of points"
        );
    }
    if (max_elements_ > buffer::kSearchBufferMaxPointCount) {
        throw std::invalid_argument(
            "QuantizedGraph point count exceeds the search-buffer ID limit"
        );
    }
    if (entry_point_id_ >= max_elements_) {
        throw std::invalid_argument("QuantizedGraph entry point is out of range");
    }
    if (quantization_bits_ != 0 && quantization_bits_ != 4 && quantization_bits_ != 8) {
        throw std::invalid_argument(
            "QuantizedGraph quantization bits must be 0 (vanilla), 4, or 8"
        );
    }
}

void QuantizedGraph<float>::copy_vectors(const float* data, size_t num_workers) {
    const int omp_num_threads = static_cast<int>(num_workers);
    if (quantization_bits_ != 0) {
        if (centroid_.size() != padded_dim_) {
            throw std::logic_error("qg-quant centroid must be set before copying vectors");
        }
        {
            index_detail::ParallelException failure;
#pragma omp parallel num_threads(omp_num_threads)
            {
                std::vector<float> rotated_data;
                std::vector<uint8_t> quantized_data;
                try {
                    rotated_data.resize(padded_dim_);
                    quantized_data.resize(padded_dim_);
                } catch (...) { failure.capture(); }
#pragma omp for schedule(static)
                for (std::ptrdiff_t index = 0;
                     index < static_cast<std::ptrdiff_t>(max_elements_);
                     ++index) {
                    if (failure.failed()) {
                        continue;
                    }
                    try {
                        const size_t i = static_cast<size_t>(index);
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
                            quantized_data.data(),
                            output.ex_code(),
                            padded_dim_,
                            quantization_bits_
                        );
                    } catch (...) { failure.capture(); }
                }
            }
            failure.rethrow();
        }
        return;
    }
#pragma omp parallel for schedule(static) num_threads(omp_num_threads)
    for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(max_elements_);
         ++index) {
        const size_t i = static_cast<size_t>(index);
        const float* src = data + (dim_ * i);
        float* dst = get_vector(i);
        std::copy(src, src + dim_, dst);
    }
}

void QuantizedGraph<float>::set_quantization_centroid(const float* centroid) {
    if (quantization_bits_ == 0) {
        return;
    }
    centroid_.resize(padded_dim_);
    rotator_->rotate(centroid, centroid_.data());
}

void QuantizedGraph<float>::save(const char* filename) const {
    if (!ready_ || rotator_ == nullptr) {
        throw std::logic_error("QuantizedGraph must be built or loaded before save");
    }
    if (filename == nullptr || filename[0] == '\0') {
        throw std::invalid_argument("QuantizedGraph save filename must not be empty");
    }
    std::ofstream output(rabitqlib::io_impl::filesystem_path(filename), std::ios::binary);
    if (!output.is_open()) {
        throw std::runtime_error("Cannot open quantized graph file for writing");
    }
    output.exceptions(std::ios::badbit | std::ios::failbit);

    if (quantization_bits_ != 0) {
        output.write(reinterpret_cast<const char*>(&kFormatMagic), sizeof(kFormatMagic));
        output.write(
            reinterpret_cast<const char*>(&kFormatVersion), sizeof(kFormatVersion)
        );
    }

    /* Basic variants */
    output.write(reinterpret_cast<const char*>(&max_elements_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&max_degree_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&dim_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&padded_dim_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&entry_point_id_), sizeof(PID));
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
        static_cast<std::streamsize>(checked_multiply(max_elements_, record_bytes_))
    );

    /* Rotator */
    this->rotator_->save(output);

    output.flush();
    output.close();
}

void QuantizedGraph<float>::load(const char* filename) {
    if (filename == nullptr || filename[0] == '\0') {
        throw std::invalid_argument("QuantizedGraph load filename must not be empty");
    }
    /* Check existence */
    if (!file_exists(filename)) {
        throw std::runtime_error("Quantized graph file does not exist");
    }

    std::ifstream input(rabitqlib::io_impl::filesystem_path(filename), std::ios::binary);
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

    QuantizedGraph<float> loaded;
    size_t stored_padded_dim = 0;
    read_value(loaded.max_elements_, "point count");
    read_value(loaded.max_degree_, "degree bound");
    read_value(loaded.dim_, "dimension");
    read_value(stored_padded_dim, "padded dimension");
    read_value(loaded.entry_point_id_, "entry point");
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
    if (stored_padded_dim < loaded.dim_ || stored_padded_dim % 32 != 0) {
        throw std::runtime_error("Invalid padded dimension in quantized graph file");
    }
    // Older files use 64-coordinate padding. Their saved rotation and codes must
    // retain that dimension, including when initialize() allocates the row storage.
    loaded.padded_dim_ = stored_padded_dim;
    loaded.initialize_layout();

    const size_t centroid_bytes = loaded.quantization_bits_ == 0
                                      ? 0
                                      : checked_multiply(loaded.padded_dim_, sizeof(float));
    const size_t data_bytes = checked_multiply(loaded.max_elements_, loaded.record_bytes_);
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

    loaded.initialize(false);

    if (loaded.quantization_bits_ != 0) {
        loaded.centroid_.resize(loaded.padded_dim_);
        read_exact(loaded.centroid_.data(), centroid_bytes, "quantization centroid");
    }

    read_exact(loaded.get_row_data(0), data_bytes, "graph data");

    for (PID source = 0; source < loaded.max_elements_; ++source) {
        const auto neighbors = loaded.get_neighbors(source);
        for (size_t i = 0; i < loaded.max_degree_; ++i) {
            if (neighbors[i] >= loaded.max_elements_) {
                throw std::runtime_error("Invalid QuantizedGraph neighbor ID");
            }
        }
    }

    loaded.rotator_->load(input);
    if (!input) {
        throw std::runtime_error("Truncated QuantizedGraph file while reading rotator");
    }

    input.close();
    // ef_search is a runtime search setting rather than persisted index state. Preserve
    // the target object's value, matching the previous in-place load behavior.
    loaded.default_ef_search_ = default_ef_search_;
    loaded.ready_ = true;
    *this = std::move(loaded);
}

void QuantizedGraph<float>::set_ef(size_t ef_search) {
    if (ef_search == 0) {
        throw std::invalid_argument("QuantizedGraph ef must be positive");
    }
    this->default_ef_search_ = ef_search;
}

void QuantizedGraph<float>::search(
    const float* __restrict__ query,
    uint32_t k,
    uint32_t* __restrict__ result_ids,
    float* __restrict__ distances
) {
    search_impl(query, k, result_ids, distances, default_ef_search_);
}

void QuantizedGraph<float>::search_impl(
    const float* query, uint32_t k, uint32_t* result_ids, float* distances, size_t ef_search
) {
    validate_search(query, k, result_ids, distances, ef_search);
    detail::SearchScratch scratch(padded_dim_, max_degree_, ef_search, k);
    thread_local VisitedSet visited;
    thread_local size_t visited_size = 0;
    if (visited_size != max_elements_) {
        visited.initialize(max_elements_, max_elements_ / 10);
        visited_size = max_elements_;
    }
    search_with_scratch_impl(
        query, k, result_ids, distances, scratch.view(visited), kPidMax, nullptr
    );
}

void QuantizedGraph<float>::search_batch(
    const float* queries,
    size_t num_queries,
    uint32_t k,
    uint32_t* result_ids,
    float* distances,
    size_t num_threads
) {
    search_batch_with_ef(
        queries, num_queries, k, result_ids, distances, default_ef_search_, num_threads
    );
}

void QuantizedGraph<float>::search_batch_with_ef(
    const float* queries,
    size_t num_queries,
    uint32_t k,
    uint32_t* result_ids,
    float* distances,
    size_t ef_search,
    size_t num_threads
) {
    if (num_queries == 0) {
        return;
    }
    validate_search(queries, k, result_ids, distances, ef_search);
    const size_t max_values = std::numeric_limits<size_t>::max() / sizeof(float);
    if (num_queries > max_values / dim_ || num_queries > max_values / k) {
        throw std::length_error("QuantizedGraph query batch is too large");
    }
    if (num_queries == 1) {
        search_impl(queries, k, result_ids, distances, ef_search);
        return;
    }

    const int num_workers = rabitqlib::detail::openmp_team_size(num_threads, num_queries);
    index_detail::ParallelException failure;
#pragma omp parallel num_threads(num_workers) if (num_workers > 1)
    {
        // Allocate and release scratch on its worker, retaining it across queries.
        std::optional<detail::SearchScratch> scratch;
        VisitedSet visited;
        try {
            scratch.emplace(padded_dim_, max_degree_, ef_search, k);
            visited.initialize(max_elements_, max_elements_ / 10);
        } catch (...) { failure.capture(); }
#pragma omp for schedule(dynamic)
        for (std::ptrdiff_t query_index = 0;
             query_index < static_cast<std::ptrdiff_t>(num_queries);
             ++query_index) {
            if (failure.failed()) {
                continue;
            }
            const size_t i = static_cast<size_t>(query_index);
            try {
                auto& local = *scratch;
                search_with_scratch_impl(
                    queries + i * dim_,
                    k,
                    result_ids + i * k,
                    distances + i * k,
                    local.view(visited),
                    kPidMax,
                    nullptr
                );
            } catch (...) { failure.capture(); }
        }
    }
    failure.rethrow();
}

void QuantizedGraph<float>::validate_search(
    const float* query,
    uint32_t k,
    const uint32_t* result_ids,
    const float* distances,
    size_t ef_search
) const {
    if (!ready_ || rotator_ == nullptr) {
        throw std::logic_error("QuantizedGraph must be built or loaded before search");
    }
    if (query == nullptr || result_ids == nullptr || distances == nullptr) {
        throw std::invalid_argument("QuantizedGraph search buffers must not be null");
    }
    if (k == 0 || k > max_elements_) {
        throw std::invalid_argument("QuantizedGraph k must be between 1 and num_points");
    }
    if (ef_search < k) {
        throw std::invalid_argument("QuantizedGraph ef must be at least k");
    }
}

float QuantizedGraph<float>::point_distance(
    const float* raw_query, const QuantizedQuery* quantized_query, PID data_id
) const {
    if (quantized_query != nullptr) {
        return quantized_distance(*quantized_query, data_id);
    }
    return raw_dist_func_(raw_query, get_vector(data_id), dim_);
}

// Scan a data row and store estimated neighbor distances. The caller scores the current
// vertex from either its raw vector (vanilla QG) or its 4/8-bit code (qg-quant).
void QuantizedGraph<float>::scan_neighbors(
    const BatchQuery<float>& query_wrapper,
    PID data_id,
    float* estimated_distances,
    buffer::SearchBuffer<float>& search_pool,
    VisitedSet& visited,
    size_t num_neighbors
) const {
    const auto* batch_data = get_batch_data(data_id);
    const auto neighbors = get_neighbors(data_id);
    for (size_t begin = 0; begin < num_neighbors; begin += fastscan::kBatchSize) {
        const float threshold = search_pool.top_dist();
        uint32_t candidate_mask = simd::qg_batch_estdist_mask(
            batch_data, query_wrapper, padded_dim_, estimated_distances + begin, threshold
        );
        batch_data += QGBatchDataMap<float>::data_bytes(padded_dim_);

        // Construction can leave a partial batch with stale IDs in its unused lanes.
        const size_t remaining = num_neighbors - begin;
        if (remaining < fastscan::kBatchSize) {
            candidate_mask &= (uint32_t{1} << remaining) - 1;
        }

        while (candidate_mask != 0) {
            const auto lane = static_cast<size_t>(bitops::countr_zero32(candidate_mask));
            candidate_mask &= candidate_mask - 1;
            const size_t i = begin + lane;
            PID neighbor_id = neighbors[i];
            float dist = estimated_distances[i];

            if (search_pool.is_full(dist) || visited.get(neighbor_id)) {
                continue;
            }
            search_pool.insert(neighbor_id, dist);  // update search buffer
            memory::mem_prefetch_l2(get_row_data(search_pool.next_id()), 10);
        }
    }
}

void QuantizedGraph<float>::update_results(
    buffer::SearchBuffer<float>& result_pool,
    VisitedSet& visited,
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
        for (uint32_t i = 0; i < this->max_degree_; ++i) {
            PID neighbor_id = neighbors[i];
            if (!visited.get(neighbor_id)) {
                visited.set(neighbor_id);
                result_pool.insert(
                    neighbor_id, point_distance(query, quantized_query, neighbor_id)
                );
            }
        }
        if (result_pool.is_full()) {
            break;
        }
    }
}

void QuantizedGraph<float>::initialize_layout() {
    if (quantization_bits_ == 0) {
        batch_data_offset_ = checked_multiply(dim_, sizeof(float));
    } else {
        batch_data_offset_ = ExDataMap<float>::data_bytes(padded_dim_, quantization_bits_);
    }

    const size_t batch_bytes = QGBatchDataMap<float>::data_bytes(padded_dim_);
    neighbor_offset_ = checked_add(
        batch_data_offset_,
        checked_multiply(batch_bytes, max_degree_ / fastscan::kBatchSize)
    );
    record_bytes_ =
        checked_add(neighbor_offset_, checked_multiply(max_degree_, sizeof(PID)));
}

void QuantizedGraph<float>::initialize(bool randomize_rotation) {
    if (randomize_rotation) {
        rotator_.reset(choose_rotator<float>(dim_, rotator_type_, padded_dim_, seed_));
    } else {
        rotator_ = rotator_impl::make_rotator_for_loading(dim_, rotator_type_, padded_dim_);
    }

    assert(padded_dim_ % 32 == 0);
    assert(padded_dim_ >= dim_);

    initialize_layout();

    const size_t data_bytes = checked_multiply(max_elements_, record_bytes_);
    assert(record_bytes_ % sizeof(float) == 0);
    data_ = RowStorage(data_bytes / sizeof(float));

    if (quantization_bits_ != 0) {
        quantized_ip_func_ = select_excode_ipfunc(quantization_bits_);
    }
}

float QuantizedGraph<float>::quantized_distance(
    const QuantizedQuery& query, PID data_id
) const {
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

void QuantizedGraph<float>::reconstruct_quantized_vector(
    PID data_id, float* reconstructed
) const {
    ConstExDataMap<float> data(
        get_quantized_vector(data_id), padded_dim_, quantization_bits_
    );
    std::vector<uint8_t> quantized_data(padded_dim_);
    quant::rabitq_impl::ex_bits::unpacking_rabitqplus_code(
        data.ex_code(), quantized_data.data(), padded_dim_, quantization_bits_
    );
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

// Encode a logical neighbor-ID row, including physical storage for tail batches.
void QuantizedGraph<float>::encode_neighbors(
    PID cur_id,
    const std::vector<PID>& new_neighbors,
    const float* rotated_data,
    const float* rotated_centroid
) {
    size_t num_neighbors = new_neighbors.size();

    if (num_neighbors == 0) {
        return;
    }
    auto neighbor_ptr = get_neighbors(cur_id);
    for (size_t i = 0; i < num_neighbors; ++i) {
        neighbor_ptr[i] = new_neighbors[i];
    }

    // quantize batches for current vertex
    auto* batch_data = get_batch_data(cur_id);
    for (size_t i = 0; i < num_neighbors; i += fastscan::kBatchSize) {
        const size_t batch_size = std::min(num_neighbors - i, fastscan::kBatchSize);
        quant::quantize_qg_batch(
            rotated_data + (i * padded_dim_),
            rotated_centroid,
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
void QuantizedGraph<float>::search_with_scratch(
    const float* __restrict__ query,
    uint32_t k,
    uint32_t* __restrict__ result_ids,
    float* __restrict__ distances,
    float* __restrict__ rotated_query_scratch,
    float* __restrict__ est_dist_scratch,
    float* __restrict__ lut_float_scratch,
    BatchQuery<float>& batch_query,
    buffer::SearchBuffer<float>& search_pool,
    buffer::SearchBuffer<float>& result_pool,
    VisitedSet& visited,
    PID hint,
    const float* k1xsumq
) {
    detail::search_with_workspace(
        *this,
        query,
        k,
        result_ids,
        distances,
        {rotated_query_scratch,
         est_dist_scratch,
         lut_float_scratch,
         batch_query,
         search_pool,
         result_pool,
         visited},
        hint,
        k1xsumq
    );
}

void detail::search_with_workspace(
    QuantizedGraph<float>& graph,
    const float* query,
    uint32_t k,
    uint32_t* result_ids,
    float* distances,
    detail::SearchScratchView scratch,
    PID hint,
    const float* k1xsumq
) {
    graph.validate_search(query, k, result_ids, distances, graph.default_ef_search_);
    graph.search_with_scratch_impl(query, k, result_ids, distances, scratch, hint, k1xsumq);
}

void QuantizedGraph<float>::search_with_scratch_impl(
    const float* __restrict__ query,
    uint32_t k,
    uint32_t* __restrict__ result_ids,
    float* __restrict__ distances,
    detail::SearchScratchView scratch,
    PID hint,
    const float* k1xsumq
) {
    if (scratch.rotated_query == nullptr || scratch.estimated_distances == nullptr ||
        scratch.lookup_table == nullptr) {
        throw std::invalid_argument("QuantizedGraph search buffers must not be null");
    }
    if (hint != kPidMax && hint >= max_elements_) {
        throw std::invalid_argument("SymphonyQG search hint is outside the graph");
    }
    if (!scratch.visited.initialized()) {
        throw std::invalid_argument("SymphonyQG visited set must be initialized");
    }

    rotator_->rotate(query, scratch.rotated_query);
    std::optional<QuantizedQuery> quantized_query;
    if (quantization_bits_ != 0) {
        quantized_query.emplace(
            scratch.rotated_query, centroid_.data(), padded_dim_, metric_type_
        );
    }
    const auto* quantized = quantized_query ? &*quantized_query : nullptr;
    scratch.batch_query.reset(
        scratch.rotated_query, padded_dim_, scratch.lookup_table, metric_type_, k1xsumq
    );
    scratch.search_pool.clear();
    scratch.result_pool.clear();
    scratch.visited.clear();

    if (hint != kPidMax && hint != entry_point_id_) {
        scratch.search_pool.insert(hint, point_distance(query, quantized, hint));
    }
    scratch.search_pool.insert(entry_point_id_, std::numeric_limits<float>::max());

    while (scratch.search_pool.has_next()) {
        const PID current = scratch.search_pool.pop();
        if (scratch.visited.get(current)) {
            continue;
        }
        scratch.visited.set(current);

        const float vertex_distance = point_distance(query, quantized, current);
        scratch.batch_query.set_g_add(vertex_distance);
        scan_neighbors(
            scratch.batch_query,
            current,
            scratch.estimated_distances,
            scratch.search_pool,
            scratch.visited,
            max_degree_
        );
        scratch.result_pool.insert(current, vertex_distance);
    }

    update_results(scratch.result_pool, scratch.visited, query, quantized);
    if (scratch.result_pool.size() != k) {
        throw std::runtime_error("QuantizedGraph search could not produce k results");
    }
    scratch.result_pool.copy_results(result_ids, distances);
}

}  // namespace rabitqlib::symqg
