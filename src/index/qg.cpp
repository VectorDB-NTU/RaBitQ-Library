#include "rabitqlib/index/symqg/qg.hpp"

#include <algorithm>
#include <array>
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

#include "../core/blob.hpp"
#include "file_replacement.hpp"
#include "format2.hpp"
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
#include "rabitqlib/index/format_error.hpp"
#include "rabitqlib/index/symqg/detail/search_scratch.hpp"
#include "rabitqlib/index/types.hpp"
#include "rabitqlib/utils/bitops.hpp"
#include "rabitqlib/utils/input_checks.hpp"
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

// Index format 2 sections of SymphonyQG (docs/formats.md, "Section kinds").
constexpr uint32_t kParamsSection = 768;
constexpr uint32_t kCentroidSection = 769;
constexpr uint32_t kRecordsSection = 770;
constexpr size_t kParamsBytes = 32;
// The stored initialization: 0 is unknown.
constexpr uint32_t kInitPiPNN = 1;
constexpr uint32_t kInitRandom = 2;
}  // namespace

QuantizedQuery::QuantizedQuery(
    const float* rotated_query,
    const float* centroid,
    size_t padded_dim,
    MetricType metric_type
)
    : rotated_query_(rotated_query) {
    g_k1xsumq_ = std::accumulate(rotated_query, rotated_query + padded_dim, 0.0F) / -2;
    g_add_ = metric_type == METRIC_IP ? -dot_product(rotated_query, centroid, padded_dim)
                                      : euclidean_sqr(rotated_query, centroid, padded_dim);
}
QuantizedQuery::QuantizedQuery(const float* rotated_query, const QuantizedQuery& prepared)
    : rotated_query_(rotated_query)
    , g_k1xsumq_(prepared.g_k1xsumq_)
    , g_add_(prepared.g_add_) {}
const float* QuantizedQuery::rotated_query() const { return rotated_query_; }
float QuantizedQuery::g_k1xsumq() const { return g_k1xsumq_; }
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

size_t QuantizedGraph<float>::num_vertices() const { return max_elements_; }

size_t QuantizedGraph<float>::max_elements() const { return max_elements_; }

size_t QuantizedGraph<float>::num_points() const { return ready_ ? max_elements_ : 0; }

size_t QuantizedGraph<float>::num_stored_points() const {
    return ready_ ? max_elements_ : 0;
}

size_t QuantizedGraph<float>::dimension() const { return dim_; }

size_t QuantizedGraph<float>::degree_bound() const { return max_degree_; }

PID QuantizedGraph<float>::entry_point() const { return entry_point_id_; }

MetricType QuantizedGraph<float>::metric_type() const { return metric_type_; }

size_t QuantizedGraph<float>::quantization_bits() const { return quantization_bits_; }

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
    , metric_type_(metric_type)
    , rotator_type_(rotator_type)
    , quantization_bits_(quantization_bits)
    , seed_(seed)
    , seed_known_(true) {
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
            throw std::logic_error(
                "QuantizedGraph quantization centroid must be set before copying vectors"
            );
        }
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
    if (read_only_) {
        throw std::logic_error(
            "QuantizedGraph was loaded from a file with features this release cannot "
            "update, so it cannot save"
        );
    }
    index_detail::save_replacing(filename, {}, [this](const std::string& path) {
        write_file(path.c_str());
    });
}

void QuantizedGraph<float>::write_file(const char* filename) const {
    namespace f2 = index_detail::format2;
    std::ofstream output(io_impl::filesystem_path(filename), std::ios::binary);
    if (!output.is_open()) {
        throw std::runtime_error("QuantizedGraph cannot open file for writing");
    }
    output.exceptions(std::ios::badbit | std::ios::failbit);

    f2::Header header;
    header.index_kind = f2::IndexKind::SymphonyQg;
    header.metric = metric_type_ == METRIC_IP ? f2::Metric::InnerProduct : f2::Metric::L2;
    header.dim = dim_;
    header.padded_dim = padded_dim_;
    header.code_bits = static_cast<uint32_t>(quantization_bits_);
    header.raw_type = quantization_bits_ == 0 ? f2::RawType::Float32 : f2::RawType::None;
    header.id_bits = 32;
    if (seed_known_) {
        header.seed = seed_;
    }
    header.num_points = max_elements_;

    std::vector<uint8_t> rotator(
        rabitqlib::detail::rotator_blob_bytes(rotator_type_, dim_, padded_dim_)
    );
    rabitqlib::detail::encode_rotator_blob(
        *rotator_, rotator_type_, dim_, seed(), rotator.data()
    );
    std::array<uint8_t, kParamsBytes> params{};
    f2::put_u64(params.data(), max_degree_);
    f2::put_u64(params.data() + 8, entry_point_id_);
    f2::put_u64(params.data() + 16, ef_construction_);
    uint32_t init = 0;
    if (initialization_.has_value()) {
        init = *initialization_ == QGInitialization::PiPNN ? kInitPiPNN : kInitRandom;
    }
    f2::put_u32(params.data() + 24, init);

    const auto bytes = [](const auto& buffer) {
        return
            [&buffer](f2::SectionSink& sink) { sink.write(buffer.data(), buffer.size()); };
    };
    std::vector<f2::SectionWrite> sections = {
        {f2::kRotatorSection, f2::kSectionRequired, rotator.size(), bytes(rotator)},
        {kParamsSection,
         f2::kSectionRequired | f2::kSectionChecksummed,
         params.size(),
         bytes(params)},
    };
    if (quantization_bits_ != 0) {
        sections.push_back(
            {kCentroidSection,
             f2::kSectionRequired,
             padded_dim_ * sizeof(float),
             [this](f2::SectionSink& sink) {
                 sink.write(centroid_.data(), padded_dim_ * sizeof(float));
             }}
        );
    }
    const size_t data_bytes = checked_multiply(max_elements_, record_bytes_);
    sections.push_back(
        {kRecordsSection,
         f2::kSectionRequired,
         data_bytes,
         [this, data_bytes](f2::SectionSink& sink) {
             sink.write(get_row_data(0), data_bytes);
         }}
    );
    f2::write_file(output, header, sections, RABITQ_LIBRARY_VERSION);
    output.close();
}

void QuantizedGraph<float>::read_format2(std::istream& input) {
    namespace f2 = index_detail::format2;
    const std::vector<uint32_t> known_kinds = {
        f2::kRotatorSection, kParamsSection, kCentroidSection, kRecordsSection};
    f2::Reader reader(
        input, f2::IndexKind::SymphonyQg, known_kinds, [](const f2::Header& header) {
            if (header.metric == f2::Metric::Cosine) {
                throw f2::FormatError(
                    "QuantizedGraph index file uses the cosine metric, which this release "
                    "does not support"
                );
            }
            const bool raw =
                header.code_bits == 0 && header.raw_type == f2::RawType::Float32;
            const bool quantized = (header.code_bits == 4 || header.code_bits == 8) &&
                                   header.raw_type == f2::RawType::None;
            if (header.id_bits != 32 || (!raw && !quantized)) {
                throw f2::FormatError(
                    "QuantizedGraph index file stores neither raw vectors nor 4- or 8-bit "
                    "codes with 32-bit IDs"
                );
            }
            if (header.seed.has_value() &&
                *header.seed > std::numeric_limits<uint32_t>::max()) {
                throw f2::FormatError("QuantizedGraph index file has a seed above 32 bits");
            }
        }
    );
    const f2::Header& header = reader.header();
    metric_type_ = header.metric == f2::Metric::InnerProduct ? METRIC_IP : METRIC_L2;
    dim_ = static_cast<size_t>(header.dim);
    padded_dim_ = static_cast<size_t>(header.padded_dim);
    quantization_bits_ = header.code_bits;
    max_elements_ = static_cast<size_t>(header.num_points);
    seed_known_ = header.seed.has_value();
    seed_ = static_cast<uint32_t>(header.seed.value_or(0));

    const auto require = [&reader](uint32_t kind, std::optional<uint64_t> size) {
        const f2::Section* section = reader.find(kind);
        if (section == nullptr) {
            throw f2::FormatError(
                "QuantizedGraph index file has no section kind " + std::to_string(kind)
            );
        }
        if (size.has_value() && section->size != *size) {
            throw f2::FormatError(
                "QuantizedGraph index file section kind " + std::to_string(kind) + " has " +
                std::to_string(section->size) + " bytes; expected " + std::to_string(*size)
            );
        }
        return section;
    };
    const auto read_all = [&reader](const f2::Section& section) {
        std::vector<uint8_t> bytes(static_cast<size_t>(section.size));
        reader.read(section, 0, bytes.data(), bytes.size());
        return bytes;
    };

    const std::vector<uint8_t> blob = read_all(*require(f2::kRotatorSection, std::nullopt));
    rabitqlib::detail::DecodedRotator rotator;
    try {
        rotator = rabitqlib::detail::decode_rotator_blob(blob.data(), blob.size());
    } catch (const std::invalid_argument& error) {
        throw f2::FormatError(
            std::string("QuantizedGraph index file has an invalid rotator (") +
            error.what() + ")"
        );
    }
    if (rotator.dim != dim_ || rotator.padded_dim != padded_dim_) {
        throw f2::FormatError(
            "QuantizedGraph index file rotator does not match its dimensions"
        );
    }
    rotator_type_ = rotator.type;

    const std::vector<uint8_t> params = read_all(*require(kParamsSection, kParamsBytes));
    const uint64_t max_degree = f2::get_u64(params.data());
    const uint64_t entry_point = f2::get_u64(params.data() + 8);
    const uint32_t init = f2::get_u32(params.data() + 24);
    if (max_degree > max_elements_ || entry_point >= max_elements_ || init > kInitRandom ||
        f2::get_u32(params.data() + 28) != 0) {
        throw f2::FormatError("QuantizedGraph index file has invalid graph parameters");
    }
    max_degree_ = static_cast<size_t>(max_degree);
    entry_point_id_ = static_cast<PID>(entry_point);
    ef_construction_ = static_cast<size_t>(f2::get_u64(params.data() + 16));
    if (init != 0) {
        initialization_ =
            init == kInitPiPNN ? QGInitialization::PiPNN : QGInitialization::Random;
    }
    try {
        validate_configuration();
    } catch (const std::invalid_argument& error) {
        throw f2::FormatError(
            std::string("QuantizedGraph index file is invalid: ") + error.what()
        );
    }

    size_t data_bytes = 0;
    try {
        initialize_layout();
        data_bytes = checked_multiply(max_elements_, record_bytes_);
    } catch (const std::length_error&) {
        throw f2::FormatError(
            "QuantizedGraph index file describes records too large to address"
        );
    }
    const f2::Section* centroid = nullptr;
    if (quantization_bits_ != 0) {
        centroid = require(kCentroidSection, padded_dim_ * sizeof(float));
    } else if (reader.find(kCentroidSection) != nullptr) {
        throw f2::FormatError(
            "QuantizedGraph index file has a centroid but stores raw vectors"
        );
    }
    const f2::Section* records = require(kRecordsSection, data_bytes);

    initialize(false);
    rotator_ = std::move(rotator.rotator);
    if (centroid != nullptr) {
        centroid_.resize(padded_dim_);
        reader.open(*centroid).read(centroid_.data(), padded_dim_ * sizeof(float));
    }
    reader.open(*records).read(get_row_data(0), data_bytes);
    read_only_ = reader.read_only();
    for (PID source = 0; source < max_elements_; ++source) {
        const auto neighbors = get_neighbors(source);
        for (size_t i = 0; i < max_degree_; ++i) {
            if (neighbors[i] >= max_elements_) {
                throw FormatError("QuantizedGraph neighbor ID is invalid");
            }
        }
    }
}

std::optional<uint32_t> QuantizedGraph<float>::seed() const {
    return seed_known_ ? std::optional<uint32_t>(seed_) : std::nullopt;
}

std::optional<size_t> QuantizedGraph<float>::ef_construction() const {
    return ef_construction_ != 0 ? std::optional<size_t>(ef_construction_) : std::nullopt;
}

std::optional<QGInitialization> QuantizedGraph<float>::initialization() const {
    return initialization_;
}

void QuantizedGraph<float>::load(const char* filename) {
    if (filename == nullptr || filename[0] == '\0') {
        throw std::invalid_argument("QuantizedGraph load filename must not be empty");
    }
    if (!file_exists(filename)) {
        throw std::runtime_error("QuantizedGraph file does not exist");
    }

    std::ifstream input(io_impl::filesystem_path(filename), std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("QuantizedGraph cannot open file");
    }
    if (index_detail::format2::has_magic(
            input, index_detail::format2::IndexKind::SymphonyQg
        )) {
        QuantizedGraph<float> loaded;
        loaded.read_format2(input);
        // ef_search is a runtime search setting rather than persisted index state.
        loaded.default_ef_search_ = default_ef_search_;
        loaded.ready_ = true;
        *this = std::move(loaded);
        return;
    }

    auto read_exact = [&](void* destination, size_t bytes, const char* field) {
        if (bytes > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
            throw FormatError("QuantizedGraph field is too large to read");
        }
        input.read(
            reinterpret_cast<char*>(destination), static_cast<std::streamsize>(bytes)
        );
        if (!input) {
            throw FormatError(
                std::string("QuantizedGraph file is truncated while reading ") + field
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
            throw FormatError("QuantizedGraph file version is unsupported");
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

    try {
        loaded.validate_configuration();
    } catch (const std::invalid_argument& error) {
        throw FormatError(error.what());  // stored settings that do not fit
    }
    if (stored_padded_dim < loaded.dim_ || stored_padded_dim % 32 != 0) {
        throw FormatError("QuantizedGraph file has an invalid padded dimension");
    }
    // Older files use 64-coordinate padding. Their saved rotation and codes must
    // retain that dimension, including when initialize() allocates the row storage.
    loaded.padded_dim_ = stored_padded_dim;
    size_t centroid_bytes = 0;
    size_t data_bytes = 0;
    size_t expected_payload_bytes = 0;
    try {
        loaded.initialize_layout();
        centroid_bytes = loaded.quantization_bits_ == 0
                             ? 0
                             : checked_multiply(loaded.padded_dim_, sizeof(float));
        data_bytes = checked_multiply(loaded.max_elements_, loaded.record_bytes_);
        const size_t rotator_bytes =
            loaded.rotator_type_ == RotatorType::MatrixRotator
                ? checked_multiply(
                      checked_multiply(sizeof(float), loaded.dim_), loaded.padded_dim_
                  )
                : checked_multiply(loaded.padded_dim_, size_t{4}) / 8;
        expected_payload_bytes =
            checked_add(checked_add(centroid_bytes, data_bytes), rotator_bytes);
    } catch (const std::length_error&) {
        throw FormatError("QuantizedGraph file describes records too large to address");
    }

    const auto payload_position = input.tellg();
    if (payload_position < 0) {
        throw std::runtime_error("QuantizedGraph cannot determine payload position");
    }
    const size_t file_size = get_filesize(filename);
    const size_t payload_offset = static_cast<size_t>(payload_position);
    if (payload_offset > file_size ||
        file_size - payload_offset != expected_payload_bytes) {
        throw FormatError("QuantizedGraph payload size is invalid");
    }

    try {
        loaded.initialize(false);
    } catch (const std::invalid_argument& error) {
        // Dimensions the stored rotation does not support, such as FHT/Kac below 64.
        throw FormatError(std::string("QuantizedGraph file is invalid: ") + error.what());
    }

    if (loaded.quantization_bits_ != 0) {
        loaded.centroid_.resize(loaded.padded_dim_);
        read_exact(loaded.centroid_.data(), centroid_bytes, "quantization centroid");
    }

    read_exact(loaded.get_row_data(0), data_bytes, "graph data");

    for (PID source = 0; source < loaded.max_elements_; ++source) {
        const auto neighbors = loaded.get_neighbors(source);
        for (size_t i = 0; i < loaded.max_degree_; ++i) {
            if (neighbors[i] >= loaded.max_elements_) {
                throw FormatError("QuantizedGraph neighbor ID is invalid");
            }
        }
    }

    loaded.rotator_->load(input);
    if (!input) {
        throw FormatError("QuantizedGraph file is truncated while reading rotator");
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
    default_ef_search_ = ef_search;
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
    check_coordinates(query, dim_, dim_, "QuantizedGraph search query");
    // Pools never need more slots than points; search pads the rest of k.
    detail::SearchScratch scratch(
        padded_dim_,
        max_degree_,
        std::min(ef_search, max_elements_),
        std::min<size_t>(k, max_elements_)
    );
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
    check_coordinates(queries, num_queries * dim_, dim_, "QuantizedGraph search queries");

    const int num_workers = rabitqlib::detail::openmp_team_size(num_threads, num_queries);
    index_detail::ParallelException failure;
#pragma omp parallel num_threads(num_workers) if (num_workers > 1)
    {
        // Allocate and release scratch on its worker, retaining it across queries.
        std::optional<detail::SearchScratch> scratch;
        VisitedSet visited;
        try {
            scratch.emplace(
                padded_dim_,
                max_degree_,
                std::min(ef_search, max_elements_),
                std::min<size_t>(k, max_elements_)
            );
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
    // Fewer than k results are padded, so k may exceed the point count.
    if (k == 0 || k > buffer::kSearchBufferMaxPointCount) {
        throw std::invalid_argument("QuantizedGraph k must be between 1 and 2^31");
    }
    if (ef_search < k) {
        throw std::invalid_argument("QuantizedGraph ef must be at least k");
    }
}

float QuantizedGraph<float>::point_distance(
    const float* raw_query, const QuantizedQuery* quantized_query, PID internal_id
) const {
    if (quantized_query != nullptr) {
        return quantized_distance(*quantized_query, internal_id);
    }
    return raw_dist_func_(raw_query, get_vector(internal_id), dim_);
}

// Scan a data row and store estimated neighbor distances. The caller scores the current
// vertex from either its raw vector (vanilla QG) or its 4/8-bit code (qg-quant).
void QuantizedGraph<float>::scan_neighbors(
    const BatchQuery<float>& query_wrapper,
    PID internal_id,
    float* estimated_distances,
    buffer::SearchBuffer<float>& search_pool,
    VisitedSet& visited,
    size_t num_neighbors
) const {
    const auto* batch_data = get_batch_data(internal_id);
    const auto neighbors = get_neighbors(internal_id);
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
        pool_data.begin(),
        pool_data.begin() + static_cast<std::ptrdiff_t>(result_pool.size())
    );
    for (const auto& record : data) {
        auto neighbors = get_neighbors(record.id);
        for (size_t i = 0; i < max_degree_; ++i) {
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

    raw_dist_func_ =
        metric_type_ == METRIC_IP ? dot_product_dis<float> : euclidean_sqr<float>;
    if (quantization_bits_ != 0) {
        quantized_ip_func_ = select_excode_ipfunc(quantization_bits_);
    }
}

float QuantizedGraph<float>::quantized_distance(
    const QuantizedQuery& query, PID internal_id
) const {
    ConstExDataMap<float> data(
        get_quantized_vector(internal_id), padded_dim_, quantization_bits_
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
        query.g_k1xsumq()
    );
}

void QuantizedGraph<float>::reconstruct_quantized_vector(
    PID internal_id, float* reconstructed
) const {
    ConstExDataMap<float> data(
        get_quantized_vector(internal_id), padded_dim_, quantization_bits_
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
    PID internal_id,
    const std::vector<PID>& new_neighbors,
    const float* rotated_data,
    const float* rotated_source
) {
    size_t num_neighbors = new_neighbors.size();

    if (num_neighbors == 0) {
        return;
    }
    auto neighbor_ptr = get_neighbors(internal_id);
    for (size_t i = 0; i < num_neighbors; ++i) {
        neighbor_ptr[i] = new_neighbors[i];
    }

    // Quantize neighbor batches against the source vertex.
    auto* batch_data = get_batch_data(internal_id);
    for (size_t i = 0; i < num_neighbors; i += fastscan::kBatchSize) {
        const size_t batch_size = std::min(num_neighbors - i, fastscan::kBatchSize);
        quant::quantize_qg_batch(
            rotated_data + (i * padded_dim_),
            rotated_source,
            batch_size,
            padded_dim_,
            batch_data,
            metric_type_
        );
        if (batch_size < fastscan::kBatchSize) {
            QGBatchDataMap<float> batch(batch_data, padded_dim_);
            std::fill(
                batch.f_add() + static_cast<std::ptrdiff_t>(batch_size),
                batch.f_add() + fastscan::kBatchSize,
                0.0F
            );
            std::fill(
                batch.f_rescale() + static_cast<std::ptrdiff_t>(batch_size),
                batch.f_rescale() + fastscan::kBatchSize,
                0.0F
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
    float* __restrict__ rotated_query,
    float* __restrict__ estimated_distances,
    float* __restrict__ lookup_table,
    BatchQuery<float>& batch_query,
    buffer::SearchBuffer<float>& search_pool,
    buffer::SearchBuffer<float>& result_pool,
    VisitedSet& visited,
    PID hint,
    const float* g_k1xsumq
) {
    detail::search_with_workspace(
        *this,
        query,
        k,
        result_ids,
        distances,
        {rotated_query,
         estimated_distances,
         lookup_table,
         batch_query,
         search_pool,
         result_pool,
         visited},
        hint,
        g_k1xsumq
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
    const float* g_k1xsumq
) {
    graph.validate_search(query, k, result_ids, distances, graph.default_ef_search_);
    if (scratch.result_pool.capacity() != k) {
        throw std::invalid_argument("QuantizedGraph result pool capacity must equal k");
    }
    check_coordinates(query, graph.dim_, graph.dim_, "QuantizedGraph search query");
    graph.search_with_scratch_impl(
        query, k, result_ids, distances, scratch, hint, g_k1xsumq
    );
}

void QuantizedGraph<float>::search_with_scratch_impl(
    const float* __restrict__ query,
    uint32_t k,
    uint32_t* __restrict__ result_ids,
    float* __restrict__ distances,
    detail::SearchScratchView scratch,
    PID hint,
    const float* g_k1xsumq
) {
    if (scratch.rotated_query == nullptr || scratch.estimated_distances == nullptr ||
        scratch.lookup_table == nullptr) {
        throw std::invalid_argument("QuantizedGraph search buffers must not be null");
    }
    if (hint != kPidMax && hint >= max_elements_) {
        throw std::invalid_argument("QuantizedGraph search hint is outside the graph");
    }
    if (!scratch.visited.initialized()) {
        throw std::invalid_argument("QuantizedGraph visited set must be initialized");
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
        scratch.rotated_query, padded_dim_, scratch.lookup_table, metric_type_, g_k1xsumq
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
    // Pad when the graph reaches fewer than k points. The pool holds at most k.
    const size_t found = scratch.result_pool.size();
    scratch.result_pool.copy_results(result_ids, distances);
    std::fill(result_ids + found, result_ids + k, kPidMax);
    std::fill(distances + found, distances + k, std::numeric_limits<float>::infinity());
}

}  // namespace rabitqlib::symqg
