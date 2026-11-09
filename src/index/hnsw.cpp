#include "rabitqlib/index/hnsw/hnsw.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <ios>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../core/blob.hpp"
#include "cosine.hpp"
#include "file_replacement.hpp"
#include "format2.hpp"
#include "rabitqlib/clustering/rabitqkmeans.hpp"
#include "rabitqlib/core/data_layout.hpp"
#include "rabitqlib/core/encoder.hpp"
#include "rabitqlib/core/estimator.hpp"
#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/core/pack_excode.hpp"
#include "rabitqlib/core/query.hpp"
#include "rabitqlib/core/rotator.hpp"
#include "rabitqlib/index/detail/buffer.hpp"
#include "rabitqlib/index/detail/hnsw_dispatch.hpp"
#include "rabitqlib/index/detail/parallel_for.hpp"
#include "rabitqlib/index/detail/visited_pool.hpp"
#include "rabitqlib/index/format_error.hpp"
#include "rabitqlib/index/hnsw/hnsw_quant.hpp"
#include "rabitqlib/index/types.hpp"
#include "rabitqlib/utils/input_checks.hpp"
#include "rabitqlib/utils/memory.hpp"
#include "rabitqlib/utils/path.hpp"
#include "rabitqlib/utils/space.hpp"
#include "rabitqlib/utils/tools.hpp"

namespace rabitqlib::hnsw {

namespace {
// Index format 2 sections of HNSW-RaBitQ (docs/formats.md, "Section kinds").
constexpr uint32_t kParamsSection = 512;
constexpr uint32_t kCentroidsSection = 513;
constexpr uint32_t kRecordsSection = 514;
constexpr uint32_t kLevelsSection = 515;
constexpr uint32_t kUpperLinksSection = 516;
constexpr size_t kParamsBytes = 72;
}  // namespace

HierarchicalNSW::HierarchicalNSW() = default;
HierarchicalNSW::~HierarchicalNSW() = default;
void HierarchicalNSW::AlignedDeleter::operator()(char* pointer) const noexcept {
    memory::aligned_deallocate(pointer);
}
void HierarchicalNSW::LinkDeleter::operator()(char* pointer) const noexcept {
    std::free(pointer);
}
int HierarchicalNSW::random_level(double multiplier) {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double level = -std::log(distribution(level_generator_)) * multiplier;
    return static_cast<int>(level);
}

HierarchicalNSW::HierarchicalNSW(
    size_t max_elements,
    size_t dim,
    size_t total_bits,
    size_t M,
    size_t ef_construction,
    size_t random_seed,
    MetricType metric_type
)
    : metric_type_(index_detail::stored_metric(metric_type))
    , cosine_(metric_type == METRIC_COSINE)
    , link_list_locks_(max_elements)
    , element_levels_(max_elements)
    , raw_dist_func_(
          (metric_type_ == METRIC_IP) ? dot_product_dis<float> : euclidean_sqr<float>
      ) {
    static_assert(
        buffer::kSearchBufferMaxPointCount <= kRemovedMask,
        "cluster IDs must leave the removal bit free"
    );
    validate_metric_type(metric_type_);
    if (M < 2) {
        throw std::invalid_argument("HNSW M must be at least 2");
    }
    if (max_elements == 0) {
        throw std::invalid_argument("HNSW max_elements must be greater than zero");
    }
    max_elements_ = max_elements;
    dim_ = dim;
    rotator_.reset(
        choose_rotator<float>(
            dim,
            RotatorType::FhtKacRotator,
            round_up_to_multiple(dim_, 32),
            static_cast<uint32_t>(random_seed)
        )
    );
    padded_dim_ = rotator_->size();
    assert(padded_dim_ % 32 == 0);
    assert(padded_dim_ >= dim_);
    ex_bits_ = total_bits - 1;

    if (total_bits < 1 || total_bits > 9) {
        throw std::invalid_argument("HNSW quantization bits must be in [1, 9]");
    }

    ip_func_ = select_excode_ipfunc(ex_bits_);

    M_ = std::min(M, kMaxM);

    upper_layer_max_degree_ = M_;
    base_layer_max_degree_ = M_ * 2;
    ef_construction_ = std::max(ef_construction, M_);

    set_record_layout();
    data_level0_memory_.reset(
        memory::huge_page_allocate<char>(max_elements_ * level0_record_bytes_)
    );

    level_generator_.seed(random_seed);
    seed_ = random_seed;

    visited_set_pool_ = std::make_unique<VisitedSetPool>(1, max_elements_);

    link_lists_.reserve(max_elements_);
    level_multiplier_ = 1 / std::log(1.0 * static_cast<double>(M_));

    query_config_ = quant::faster_config(padded_dim_, SplitSingleQuery<float>::kNumBits);
}

void HierarchicalNSW::set_record_layout() {
    bin_data_bytes_ = BinDataMap<float>::data_bytes(padded_dim_);
    ex_data_bytes_ = ExDataMap<float>::data_bytes(padded_dim_, ex_bits_);
    // Level 0: link count and links, cluster ID, external label, then the codes.
    level0_links_bytes_ = (base_layer_max_degree_ + 1) * sizeof(PID);
    label_offset_ = level0_links_bytes_ + sizeof(PID);
    bin_data_offset_ = label_offset_ + sizeof(PID);
    ex_data_offset_ = bin_data_offset_ + bin_data_bytes_;
    level0_record_bytes_ = ex_data_offset_ + ex_data_bytes_;
    upper_layer_links_bytes_ = (upper_layer_max_degree_ + 1) * sizeof(PID);
}

void HierarchicalNSW::save(const char* filename) const {
    if (level0_record_bytes_ == 0) {
        throw std::logic_error("HNSW index must be initialized before save");
    }
    if (num_stored_points_ == 0) {
        throw std::logic_error("HNSW index has no points to save; add points first");
    }
    if (filename == nullptr || filename[0] == '\0') {
        throw std::invalid_argument("HNSW index filename must not be empty");
    }
    check_writable("save");
    index_detail::save_replacing(filename, {}, [this](const std::string& path) {
        write_file(path.c_str());
    });
}

void HierarchicalNSW::write_file(const char* filename) const {
    namespace f2 = index_detail::format2;
    std::ofstream output(io_impl::filesystem_path(filename), std::ios::binary);
    if (!output.is_open()) {
        throw std::runtime_error("HNSW cannot open index file for writing");
    }
    const size_t element_count = num_stored_points_.load();

    f2::Header header;
    header.index_kind = f2::IndexKind::Hnsw;
    header.metric = f2::header_metric(metric_type());
    header.dim = dim_;
    header.padded_dim = padded_dim_;
    header.code_bits = static_cast<uint32_t>(ex_bits_ + 1);
    header.id_bits = 32;
    header.seed = seed_;
    header.num_points = element_count;

    // The rotation was drawn from the low 32 bits of the seed.
    std::vector<uint8_t> rotator(
        rabitqlib::detail::rotator_blob_bytes(RotatorType::FhtKacRotator, dim_, padded_dim_)
    );
    rabitqlib::detail::encode_rotator_blob(
        *rotator_,
        RotatorType::FhtKacRotator,
        dim_,
        seed_.has_value() ? std::optional<uint32_t>(static_cast<uint32_t>(*seed_))
                          : std::nullopt,
        rotator.data()
    );
    std::array<uint8_t, kParamsBytes> params{};
    f2::put_u64(params.data(), max_elements_);
    f2::put_u64(params.data() + 8, num_clusters_);
    f2::put_u64(params.data() + 16, M_);
    f2::put_u64(params.data() + 24, upper_layer_max_degree_);
    f2::put_u64(params.data() + 32, base_layer_max_degree_);
    f2::put_u64(params.data() + 40, ef_construction_);
    uint64_t multiplier_bits = 0;
    std::memcpy(&multiplier_bits, &level_multiplier_, sizeof(multiplier_bits));
    f2::put_u64(params.data() + 48, multiplier_bits);
    f2::put_u32(
        params.data() + 56, static_cast<uint32_t>(static_cast<int32_t>(max_level_))
    );
    f2::put_u64(params.data() + 64, entry_point_id_);
    std::vector<uint8_t> levels(sizeof(uint32_t) * element_count);
    uint64_t upper_bytes = 0;
    for (size_t i = 0; i < element_count; ++i) {
        f2::put_u32(levels.data() + (sizeof(uint32_t) * i), element_levels_[i]);
        upper_bytes += uint64_t(element_levels_[i]) * upper_layer_links_bytes_;
    }

    const auto bytes = [](const auto& buffer) {
        return
            [&buffer](f2::SectionSink& sink) { sink.write(buffer.data(), buffer.size()); };
    };
    const std::vector<f2::SectionWrite> sections = {
        {f2::kRotatorSection, f2::kSectionRequired, rotator.size(), bytes(rotator)},
        {kParamsSection,
         f2::kSectionRequired | f2::kSectionChecksummed,
         params.size(),
         bytes(params)},
        {kCentroidsSection,
         f2::kSectionRequired,
         sizeof(float) * num_clusters_ * padded_dim_,
         [this](f2::SectionSink& sink) {
             sink.write(
                 centroids_memory_.get(), sizeof(float) * num_clusters_ * padded_dim_
             );
         }},
        {kRecordsSection,
         f2::kSectionRequired,
         element_count * level0_record_bytes_,
         [&](f2::SectionSink& sink) {
             sink.write(data_level0_memory_.get(), element_count * level0_record_bytes_);
         }},
        {kLevelsSection, f2::kSectionRequired, levels.size(), bytes(levels)},
        {kUpperLinksSection,
         f2::kSectionRequired,
         upper_bytes,
         [&](f2::SectionSink& sink) {
             for (size_t i = 0; i < element_count; ++i) {
                 if (element_levels_[i] > 0) {
                     sink.write(
                         link_lists_[i].get(),
                         static_cast<size_t>(element_levels_[i]) * upper_layer_links_bytes_
                     );
                 }
             }
         }},
    };
    // write_file checks the stream after flushing; close reports a late failure.
    f2::write_file(output, header, sections, RABITQ_LIBRARY_VERSION);
    output.close();
    if (!output) {
        throw std::runtime_error("HNSW index file cannot be written");
    }
}

void HierarchicalNSW::load(const char* filename) {
    if (filename == nullptr || filename[0] == '\0') {
        throw std::invalid_argument("HNSW index filename must not be empty");
    }
    std::ifstream input(io_impl::filesystem_path(filename), std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("HNSW cannot open index file");
    }

    // Keep the current index intact until the complete file has been validated.
    HierarchicalNSW loaded;
    if (index_detail::format2::has_magic(input, index_detail::format2::IndexKind::Hnsw)) {
        loaded.read_format2(input);
    } else {
        loaded.read_legacy(input);
    }

    loaded.visited_set_pool_ = std::make_unique<VisitedSetPool>(1, loaded.max_elements_);
    loaded.raw_dist_func_ =
        (loaded.metric_type_ == METRIC_IP) ? dot_product_dis<float> : euclidean_sqr<float>;
    loaded.ip_func_ = select_excode_ipfunc(loaded.ex_bits_);
    loaded.query_config_ =
        quant::faster_config(loaded.padded_dim_, SplitSingleQuery<float>::kNumBits);

    // All swaps below are non-allocating. The temporary releases the old state.
    using std::swap;
    swap(construction_data_, loaded.construction_data_);
    swap(max_elements_, loaded.max_elements_);
    const size_t old_count = num_stored_points_.load();
    num_stored_points_ = loaded.num_stored_points_.load();
    loaded.num_stored_points_ = old_count;
    swap(num_removed_, loaded.num_removed_);
    swap(level0_record_bytes_, loaded.level0_record_bytes_);
    swap(upper_layer_links_bytes_, loaded.upper_layer_links_bytes_);
    swap(M_, loaded.M_);
    swap(upper_layer_max_degree_, loaded.upper_layer_max_degree_);
    swap(base_layer_max_degree_, loaded.base_layer_max_degree_);
    swap(ef_construction_, loaded.ef_construction_);
    swap(metric_type_, loaded.metric_type_);
    swap(cosine_, loaded.cosine_);
    swap(level_multiplier_, loaded.level_multiplier_);
    swap(max_level_, loaded.max_level_);
    link_list_locks_.swap(loaded.link_list_locks_);
    swap(entry_point_id_, loaded.entry_point_id_);
    swap(level0_links_bytes_, loaded.level0_links_bytes_);
    swap(bin_data_offset_, loaded.bin_data_offset_);
    swap(ex_data_offset_, loaded.ex_data_offset_);
    swap(label_offset_, loaded.label_offset_);
    swap(bin_data_bytes_, loaded.bin_data_bytes_);
    swap(ex_data_bytes_, loaded.ex_data_bytes_);
    swap(ex_bits_, loaded.ex_bits_);
    swap(data_level0_memory_, loaded.data_level0_memory_);
    swap(link_lists_, loaded.link_lists_);
    element_levels_.swap(loaded.element_levels_);
    swap(num_clusters_, loaded.num_clusters_);
    swap(dim_, loaded.dim_);
    swap(padded_dim_, loaded.padded_dim_);
    swap(centroids_memory_, loaded.centroids_memory_);
    label_lookup_.swap(loaded.label_lookup_);
    swap(visited_set_pool_, loaded.visited_set_pool_);
    swap(ip_func_, loaded.ip_func_);
    swap(rotator_, loaded.rotator_);
    swap(query_config_, loaded.query_config_);
    swap(raw_dist_func_, loaded.raw_dist_func_);
    swap(seed_, loaded.seed_);
    swap(read_only_, loaded.read_only_);
}

// The unversioned layout of releases up to 0.8.
void HierarchicalNSW::read_legacy(std::ifstream& input) {
    auto invalid_file = []() -> void {
        throw FormatError("HNSW index file is invalid or truncated");
    };
    auto read_header = [&](auto& value) {
        input.read(reinterpret_cast<char*>(&value), sizeof(value));
        if (!input) {
            invalid_file();
        }
    };
    auto checked_product = [&](size_t left, size_t right) {
        if (right != 0 && left > std::numeric_limits<size_t>::max() / right) {
            invalid_file();
        }
        return left * right;
    };

    size_t element_count = 0;
    PID label_offset = 0;
    read_header(max_elements_);
    read_header(element_count);
    read_header(dim_);
    read_header(padded_dim_);
    read_header(num_clusters_);
    read_header(ex_bits_);
    read_header(metric_type_);
    read_header(bin_data_bytes_);
    read_header(ex_data_bytes_);
    read_header(level0_links_bytes_);
    read_header(bin_data_offset_);
    read_header(ex_data_offset_);
    read_header(label_offset);
    label_offset_ = label_offset;
    read_header(level0_record_bytes_);
    read_header(upper_layer_links_bytes_);
    read_header(max_level_);
    read_header(entry_point_id_);
    read_header(M_);
    read_header(upper_layer_max_degree_);
    read_header(base_layer_max_degree_);
    read_header(level_multiplier_);
    read_header(ef_construction_);

    if (max_elements_ == 0 || max_elements_ > buffer::kSearchBufferMaxPointCount ||
        element_count > max_elements_ ||
        num_clusters_ > buffer::kSearchBufferMaxPointCount ||
        (element_count != 0 && num_clusters_ == 0) || dim_ < 64 ||
        dim_ > rotator_impl::FhtKacRotator::kMaxDim || padded_dim_ < dim_ ||
        padded_dim_ % 32 != 0 || padded_dim_ > rotator_impl::FhtKacRotator::kMaxDim ||
        ex_bits_ > 8 || (metric_type_ != METRIC_L2 && metric_type_ != METRIC_IP) ||
        M_ < 2 || M_ > kMaxM || upper_layer_max_degree_ != M_ ||
        base_layer_max_degree_ != 2 * M_ || ef_construction_ < M_ ||
        (element_count == 0 && (max_level_ != -1 || entry_point_id_ != kPidMax)) ||
        (element_count != 0 && (max_level_ < 0 || entry_point_id_ >= element_count))) {
        invalid_file();
    }

    // The stored layout must be the one the validated parameters derive.
    const auto stored_layout = record_layout();
    set_record_layout();
    if (record_layout() != stored_layout || !std::isfinite(level_multiplier_) ||
        level_multiplier_ <= 0.0) {
        invalid_file();
    }

    const size_t centroids_bytes =
        checked_product(checked_product(num_clusters_, padded_dim_), sizeof(float));
    const size_t data_capacity_bytes = checked_product(max_elements_, level0_record_bytes_);
    const size_t data_bytes = checked_product(element_count, level0_record_bytes_);
    // Validate pointer-table capacity before allocating it.
    (void)checked_product(max_elements_, sizeof(LinkStorage));
    const size_t link_headers_bytes = checked_product(element_count, sizeof(uint32_t));
    rotator_ = rotator_impl::make_rotator_for_loading(
        dim_, RotatorType::FhtKacRotator, padded_dim_
    );
    const size_t rotator_bytes = rotator_->dump_bytes();

    const auto payload_start = input.tellg();
    input.seekg(0, std::ios::end);
    const auto payload_end = input.tellg();
    if (!input || payload_start < 0 || payload_end < payload_start) {
        invalid_file();
    }
    size_t remaining = static_cast<size_t>(payload_end - payload_start);
    input.seekg(payload_start);
    if (centroids_bytes > remaining || data_bytes > remaining - centroids_bytes ||
        link_headers_bytes > remaining - centroids_bytes - data_bytes ||
        rotator_bytes > remaining - centroids_bytes - data_bytes - link_headers_bytes) {
        invalid_file();
    }
    auto read_payload = [&](char* destination, size_t bytes) {
        if (bytes > remaining ||
            bytes > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
            invalid_file();
        }
        if (bytes != 0) {
            input.read(destination, static_cast<std::streamsize>(bytes));
            if (!input) {
                invalid_file();
            }
        }
        remaining -= bytes;
    };

    if (centroids_bytes != 0) {
        centroids_memory_.reset(memory::huge_page_allocate<char>(centroids_bytes));
        read_payload(centroids_memory_.get(), centroids_bytes);
    }
    data_level0_memory_.reset(memory::huge_page_allocate<char>(data_capacity_bytes));
    read_payload(data_level0_memory_.get(), data_bytes);

    element_levels_ = std::vector<int>(max_elements_);
    link_list_locks_ = std::vector<std::mutex>(max_elements_);
    link_lists_.reserve(max_elements_);
    link_lists_.resize(element_count);
    for (size_t i = 0; i < element_count; ++i) {
        uint32_t link_list_bytes = 0;
        read_payload(reinterpret_cast<char*>(&link_list_bytes), sizeof(link_list_bytes));
        if (link_list_bytes != 0) {
            if (link_list_bytes % upper_layer_links_bytes_ != 0 ||
                link_list_bytes / upper_layer_links_bytes_ >
                    static_cast<size_t>(max_level_) ||
                link_list_bytes > remaining) {
                invalid_file();
            }
            link_lists_[i].reset(static_cast<char*>(std::malloc(link_list_bytes)));
            if (link_lists_[i] == nullptr) {
                throw std::bad_alloc();
            }
            element_levels_[i] =
                static_cast<int>(link_list_bytes / upper_layer_links_bytes_);
        }
        num_stored_points_ = i + 1;
        if (link_list_bytes != 0) {
            read_payload(link_lists_[i].get(), link_list_bytes);
        }
    }
    validate_graph();

    if (rotator_bytes > remaining) {
        invalid_file();
    }
    rotator_->load(input);
    if (!input) {
        invalid_file();
    }
    remaining -= rotator_bytes;
    if (remaining != 0) {
        invalid_file();
    }
}

void HierarchicalNSW::read_format2(std::istream& input) {
    namespace f2 = index_detail::format2;
    const std::vector<uint32_t> known_kinds = {
        f2::kRotatorSection,
        kParamsSection,
        kCentroidsSection,
        kRecordsSection,
        kLevelsSection,
        kUpperLinksSection};
    f2::Reader reader(
        input, f2::IndexKind::Hnsw, known_kinds, [](const f2::Header& header) {
            if (header.id_bits != 32 || header.raw_type != f2::RawType::None) {
                throw f2::FormatError("HNSW index file stores 64-bit IDs or raw vectors");
            }
            if (header.code_bits == 0 || header.dim < 64) {
                throw f2::FormatError(
                    "HNSW index file has an invalid code width or dimension"
                );
            }
            if (header.num_points == 0 ||
                header.num_points > buffer::kSearchBufferMaxPointCount) {
                throw f2::FormatError(
                    "HNSW index file has an invalid point count " +
                    std::to_string(header.num_points)
                );
            }
        }
    );
    const f2::Header& header = reader.header();
    metric_type_ = index_detail::stored_metric(f2::metric_type(header.metric));
    cosine_ = header.metric == f2::Metric::Cosine;
    dim_ = static_cast<size_t>(header.dim);
    padded_dim_ = static_cast<size_t>(header.padded_dim);
    ex_bits_ = header.code_bits - 1;
    seed_ = header.seed;
    const auto element_count = static_cast<size_t>(header.num_points);

    const auto require = [&reader](uint32_t kind, std::optional<uint64_t> size) {
        const f2::Section* section = reader.find(kind);
        if (section == nullptr) {
            throw f2::FormatError(
                "HNSW index file has no section kind " + std::to_string(kind)
            );
        }
        if (size.has_value() && section->size != *size) {
            throw f2::FormatError(
                "HNSW index file section kind " + std::to_string(kind) + " has " +
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

    const std::vector<uint8_t> params = read_all(*require(kParamsSection, kParamsBytes));
    max_elements_ = static_cast<size_t>(f2::get_u64(params.data()));
    num_clusters_ = static_cast<size_t>(f2::get_u64(params.data() + 8));
    M_ = static_cast<size_t>(f2::get_u64(params.data() + 16));
    upper_layer_max_degree_ = static_cast<size_t>(f2::get_u64(params.data() + 24));
    base_layer_max_degree_ = static_cast<size_t>(f2::get_u64(params.data() + 32));
    ef_construction_ = static_cast<size_t>(f2::get_u64(params.data() + 40));
    const uint64_t multiplier_bits = f2::get_u64(params.data() + 48);
    std::memcpy(&level_multiplier_, &multiplier_bits, sizeof(level_multiplier_));
    max_level_ = static_cast<int>(static_cast<int32_t>(f2::get_u32(params.data() + 56)));
    const uint64_t entry_point = f2::get_u64(params.data() + 64);
    if (max_elements_ < element_count ||
        max_elements_ > buffer::kSearchBufferMaxPointCount || num_clusters_ == 0 ||
        num_clusters_ > buffer::kSearchBufferMaxPointCount || M_ < 2 || M_ > kMaxM ||
        upper_layer_max_degree_ != M_ || base_layer_max_degree_ != 2 * M_ ||
        ef_construction_ < M_ || !std::isfinite(level_multiplier_) ||
        level_multiplier_ <= 0.0 || max_level_ < 0 || entry_point >= element_count ||
        f2::get_u32(params.data() + 60) != 0) {
        throw f2::FormatError("HNSW index file has invalid graph parameters");
    }
    entry_point_id_ = static_cast<PID>(entry_point);
    set_record_layout();

    const std::vector<uint8_t> blob = read_all(*require(f2::kRotatorSection, std::nullopt));
    rabitqlib::detail::DecodedRotator rotator;
    try {
        rotator = rabitqlib::detail::decode_rotator_blob(blob.data(), blob.size());
    } catch (const std::invalid_argument& error) {
        throw f2::FormatError(
            std::string("HNSW index file has an invalid rotator (") + error.what() + ")"
        );
    }
    if (rotator.type != RotatorType::FhtKacRotator || rotator.dim != dim_ ||
        rotator.padded_dim != padded_dim_) {
        throw f2::FormatError("HNSW index file rotator does not match its dimensions");
    }
    rotator_ = std::move(rotator.rotator);

    const size_t centroids_bytes = sizeof(float) * num_clusters_ * padded_dim_;
    const f2::Section* centroids = require(kCentroidsSection, centroids_bytes);
    const f2::Section* records =
        require(kRecordsSection, element_count * level0_record_bytes_);
    const std::vector<uint8_t> levels =
        read_all(*require(kLevelsSection, sizeof(uint32_t) * element_count));
    uint64_t upper_bytes = 0;
    element_levels_ = std::vector<int>(max_elements_);
    for (size_t i = 0; i < element_count; ++i) {
        const uint32_t level = f2::get_u32(levels.data() + (sizeof(uint32_t) * i));
        if (level > static_cast<uint32_t>(max_level_)) {
            throw f2::FormatError("HNSW index file has a point above the maximum level");
        }
        element_levels_[i] = static_cast<int>(level);
        // Guard the sum, so that a wrapped total cannot match the section size.
        const uint64_t bytes = uint64_t{level} * upper_layer_links_bytes_;
        if (bytes > std::numeric_limits<uint64_t>::max() - upper_bytes) {
            throw f2::FormatError("HNSW index file has an invalid level structure");
        }
        upper_bytes += bytes;
    }
    const f2::Section* upper = require(kUpperLinksSection, upper_bytes);

    centroids_memory_.reset(memory::huge_page_allocate<char>(centroids_bytes));
    f2::SectionInput centroids_input = reader.open(*centroids);
    centroids_input.read(centroids_memory_.get(), centroids_bytes);
    data_level0_memory_.reset(
        memory::huge_page_allocate<char>(max_elements_ * level0_record_bytes_)
    );
    f2::SectionInput records_input = reader.open(*records);
    records_input.read(data_level0_memory_.get(), element_count * level0_record_bytes_);
    link_list_locks_ = std::vector<std::mutex>(max_elements_);
    link_lists_.reserve(max_elements_);
    link_lists_.resize(element_count);
    f2::SectionInput upper_input = reader.open(*upper);
    for (size_t i = 0; i < element_count; ++i) {
        const size_t bytes =
            static_cast<size_t>(element_levels_[i]) * upper_layer_links_bytes_;
        if (bytes == 0) {
            continue;
        }
        link_lists_[i].reset(static_cast<char*>(std::malloc(bytes)));
        if (link_lists_[i] == nullptr) {
            throw std::bad_alloc();
        }
        upper_input.read(link_lists_[i].get(), bytes);
    }
    num_stored_points_ = element_count;
    validate_graph();
    read_only_ = reader.read_only();
}

void HierarchicalNSW::check_writable(const char* operation) const {
    if (read_only_) {
        throw std::logic_error(
            std::string(
                "HNSW index was loaded from a file with features this release "
                "cannot update, so it cannot "
            ) +
            operation
        );
    }
}

void HierarchicalNSW::validate_graph() {
    const auto invalid = []() -> void {
        throw FormatError("HNSW index file is invalid or truncated");
    };
    const size_t element_count = num_stored_points_;
    for (size_t i = 0; i < element_count; ++i) {
        const char* row = data_level0_memory_.get() + i * level0_record_bytes_;
        PID cluster_id = 0;
        PID label = 0;
        uint16_t level0_degree = 0;
        std::memcpy(&level0_degree, row, sizeof(level0_degree));
        std::memcpy(&cluster_id, row + level0_links_bytes_, sizeof(cluster_id));
        std::memcpy(&label, row + label_offset_, sizeof(label));
        if ((cluster_id & ~kRemovedMask) >= num_clusters_ ||
            level0_degree > base_layer_max_degree_) {
            invalid();
        }
        num_removed_ += (cluster_id & kRemovedMask) != 0;
        for (size_t edge = 0; edge < level0_degree; ++edge) {
            PID neighbor = 0;
            std::memcpy(&neighbor, row + sizeof(PID) + edge * sizeof(PID), sizeof(PID));
            if (neighbor >= element_count) {
                invalid();
            }
        }
        if (!label_lookup_.emplace(label, static_cast<PID>(i)).second) {
            invalid();
        }
        for (int level = 0; level < element_levels_[i]; ++level) {
            const char* link = link_lists_[i].get() + level * upper_layer_links_bytes_;
            uint16_t degree = 0;
            std::memcpy(&degree, link, sizeof(degree));
            if (degree > upper_layer_max_degree_) {
                invalid();
            }
            for (size_t edge = 0; edge < degree; ++edge) {
                PID neighbor = 0;
                std::memcpy(
                    &neighbor, link + sizeof(PID) + edge * sizeof(PID), sizeof(PID)
                );
                if (neighbor >= element_count) {
                    invalid();
                }
            }
        }
    }
    for (size_t i = 0; i < element_count; ++i) {
        for (int level = 1; level <= element_levels_[i]; ++level) {
            const char* link = link_lists_[i].get() +
                               (static_cast<size_t>(level) - 1) * upper_layer_links_bytes_;
            uint16_t degree = 0;
            std::memcpy(&degree, link, sizeof(degree));
            for (size_t edge = 0; edge < degree; ++edge) {
                PID neighbor = 0;
                std::memcpy(
                    &neighbor, link + sizeof(PID) + edge * sizeof(PID), sizeof(PID)
                );
                if (element_levels_[neighbor] < level) {
                    invalid();
                }
            }
        }
    }
    if (element_count != 0 && element_levels_[entry_point_id_] != max_level_) {
        invalid();
    }
}

void HierarchicalNSW::construct(
    size_t num_clusters,
    const float* centroids,
    size_t num_points,
    const float* data,
    PID* cluster_ids,
    size_t num_threads,
    bool fast_quantization
) {
    if (num_clusters_ != 0 || num_stored_points_ != 0) {
        throw std::logic_error("HNSW index is already trained, constructed or loaded");
    }
    if (num_clusters == 0 || num_clusters > buffer::kSearchBufferMaxPointCount) {
        throw std::invalid_argument("HNSW cluster count is out of range");
    }
    if (num_points == 0 || num_points > max_elements_ ||
        num_points > buffer::kSearchBufferMaxPointCount) {
        throw std::invalid_argument("HNSW point count is out of range");
    }
    if (centroids == nullptr || data == nullptr || cluster_ids == nullptr) {
        throw std::invalid_argument("HNSW construction inputs must not be null");
    }
    if (num_clusters > std::numeric_limits<size_t>::max() / sizeof(float) / padded_dim_) {
        throw std::invalid_argument("HNSW centroids exceed addressable storage");
    }
    for (size_t i = 0; i < num_points; ++i) {
        if (cluster_ids[i] >= num_clusters) {
            throw std::invalid_argument("HNSW cluster ID is out of range");
        }
    }
    check_coordinates(data, num_points * dim_, dim_, "HNSW construction data");
    check_coordinates(centroids, num_clusters * dim_, dim_, "HNSW construction centroids");
    std::vector<float> normalized;
    if (cosine_) {
        normalized = index_detail::normalized_rows(data, num_points, dim_, num_threads);
        data = normalized.data();
    }

    try {
        num_clusters_ = num_clusters;
        const size_t centroids_bytes = num_clusters_ * padded_dim_ * sizeof(float);
        centroids_memory_.reset(memory::huge_page_allocate<char>(centroids_bytes));

        for (size_t i = 0; i < num_clusters; ++i) {
            rotator_->rotate(
                centroids + (i * dim_),
                reinterpret_cast<float*>(centroids_memory_.get()) + (i * padded_dim_)
            );
        }

        link_exact(num_points, data, cluster_ids, num_threads, fast_quantization);
    } catch (...) {
        // Also discard the centroids, retaining capacity and rotation for a retry.
        centroids_memory_.reset();
        num_clusters_ = 0;
        throw;
    }
}

void HierarchicalNSW::link_exact(
    size_t num_points,
    const float* data,
    const PID* cluster_ids,
    size_t num_threads,
    bool fast_quantization
) {
    try {
        quant::RabitqConfig config;
        if (fast_quantization) {
            config = quant::faster_config(padded_dim_, ex_bits_ + 1);
        }

        construction_data_ = data;
        index_detail::parallel_for(
            0, num_points, num_threads, [&](size_t index, size_t /*thread*/) {
                add_point(index, cluster_ids[index], config);
            }
        );
    } catch (...) {
        // parallel_for joins every worker before propagating an exception.
        // Discard the partial graph, retaining capacity, rotation and centroids.
        std::fill_n(element_levels_.begin(), num_stored_points_.load(), 0);
        link_lists_.clear();
        num_stored_points_ = 0;
        num_removed_ = 0;
        label_lookup_.clear();
        entry_point_id_ = kPidMax;
        max_level_ = -1;
        construction_data_ = nullptr;
        throw;
    }
    construction_data_ = nullptr;
}

void HierarchicalNSW::set_centroids(size_t num_clusters, const float* centroids) {
    if (level0_record_bytes_ == 0) {
        throw std::logic_error("HNSW index must be initialized before setting centroids");
    }
    if (num_stored_points_ != 0) {
        throw std::logic_error("HNSW cannot replace the centroids of an index with points");
    }
    if (num_clusters == 0 || num_clusters > buffer::kSearchBufferMaxPointCount) {
        throw std::invalid_argument("HNSW cluster count is out of range");
    }
    if (centroids == nullptr) {
        throw std::invalid_argument("HNSW centroids must not be null");
    }
    if (num_clusters > std::numeric_limits<size_t>::max() / sizeof(float) / padded_dim_) {
        throw std::invalid_argument("HNSW centroids exceed addressable storage");
    }
    check_coordinates(centroids, num_clusters * dim_, dim_, "HNSW centroids");

    // construct rotates centroids the same way, so both paths quantize identically.
    AlignedStorage rotated(
        memory::huge_page_allocate<char>(num_clusters * padded_dim_ * sizeof(float))
    );
    for (size_t i = 0; i < num_clusters; ++i) {
        rotator_->rotate(
            centroids + (i * dim_),
            reinterpret_cast<float*>(rotated.get()) + (i * padded_dim_)
        );
    }
    centroids_memory_ = std::move(rotated);
    num_clusters_ = num_clusters;
}

void HierarchicalNSW::train(
    size_t num_clusters,
    size_t num_points,
    const float* data,
    const rabitqkmeans::RaBitQKMeansParameters& cp
) {
    if (level0_record_bytes_ == 0) {
        throw std::logic_error("HNSW index must be initialized before training");
    }
    if (num_stored_points_ != 0) {
        throw std::logic_error("HNSW cannot retrain an index with points");
    }
    if (num_clusters == 0 || num_clusters > buffer::kSearchBufferMaxPointCount) {
        throw std::invalid_argument("HNSW cluster count is out of range");
    }
    if (data == nullptr) {
        throw std::invalid_argument("HNSW training data must not be null");
    }
    if (num_points < num_clusters) {
        throw std::invalid_argument("HNSW training needs at least num_clusters points");
    }
    if (num_points > std::numeric_limits<size_t>::max() / sizeof(float) / dim_) {
        throw std::invalid_argument("HNSW training data exceeds addressable storage");
    }
    // k-means checks the data for finite, in-range coordinates in parallel with
    // the bound check_coordinates uses; set_centroids checks its centroids. The cosine
    // metric checks the data before normalizing it, as add does.
    std::vector<float> normalized;
    if (cosine_) {
        check_coordinates(data, num_points * dim_, dim_, "HNSW training data");
        normalized = index_detail::normalized_rows(data, num_points, dim_, cp.num_threads);
        data = normalized.data();
    }
    rabitqkmeans::RaBitQKMeans kmeans(dim_, num_clusters, cp);
    kmeans.train_centroids(num_points, data);
    set_centroids(num_clusters, kmeans.centroids.data());
}

template <class Distance>
PID HierarchicalNSW::greedy_descend(
    PID current_id, int from_level, int to_level, Distance distance
) {
    float current_distance = distance(current_id);
    for (int level = from_level; level > to_level; --level) {
        bool changed = true;
        while (changed) {
            changed = false;
            std::unique_lock<std::mutex> lock(link_list_locks_[current_id]);
            const PID* links = upper_layer_links(current_id, level);
            const size_t num_neighbors = link_count(links);
            const PID* neighbors = links + 1;
            for (size_t i = 0; i < num_neighbors; ++i) {
                PID candidate_id = neighbors[i];
                assert(candidate_id < num_stored_points_);
                float candidate_distance = distance(candidate_id);
                if (candidate_distance < current_distance) {
                    current_distance = candidate_distance;
                    current_id = candidate_id;
                    changed = true;
                }
            }
        }
    }
    return current_id;
}

template <class Distance, class Prefetch>
maxheap<std::pair<float, PID>> HierarchicalNSW::search_layer(
    PID entry_point_id, int level, Distance distance, Prefetch prefetch
) {
    auto visited = visited_set_pool_->acquire();

    maxheap<std::pair<float, PID>> top_candidates;
    minheap<std::pair<float, PID>> candidate_set;

    float dist_k = distance(entry_point_id);
    top_candidates.emplace(dist_k, entry_point_id);
    candidate_set.emplace(dist_k, entry_point_id);
    visited->set(entry_point_id);

    while (!candidate_set.empty()) {
        std::pair<float, PID> current = candidate_set.top();
        if (current.first > dist_k && top_candidates.size() == ef_construction_) {
            break;
        }
        candidate_set.pop();

        PID current_id = current.second;
        std::unique_lock<std::mutex> lock(link_list_locks_[current_id]);

        const PID* links =
            (level == 0) ? level0_links(current_id) : upper_layer_links(current_id, level);
        const size_t num_neighbors = link_count(links);
        const PID* neighbors = links + 1;

        // Slots beyond the link count hold stale or unwritten IDs.
        for (size_t j = 0; j < std::min<size_t>(2, num_neighbors); ++j) {
            prefetch(neighbors[j]);
        }

        for (size_t j = 0; j < num_neighbors; ++j) {
            PID candidate_id = neighbors[j];
            assert(candidate_id < num_stored_points_);
            if (visited->get(candidate_id)) {
                continue;
            }
            visited->set(candidate_id);

            if (j < num_neighbors - 1) {
                prefetch(*(neighbors + j + 1));
            }

            float candidate_distance = distance(candidate_id);
            if (top_candidates.size() < ef_construction_ || dist_k > candidate_distance) {
                candidate_set.emplace(candidate_distance, candidate_id);
                top_candidates.emplace(candidate_distance, candidate_id);
                if (top_candidates.size() > ef_construction_) {
                    top_candidates.pop();
                }
                if (!top_candidates.empty()) {
                    dist_k = top_candidates.top().first;
                }
            }
        }
    }
    visited_set_pool_->release(std::move(visited));
    return top_candidates;
}

void HierarchicalNSW::add_point(
    PID label, PID cluster_id, const quant::RabitqConfig& config
) {
    int node_level = 0;
    PID new_node_id = 0;
    {
        // construct passes distinct labels below its validated capacity.
        std::unique_lock<std::mutex> label_lookup_lock(label_lookup_lock_);
        assert(label_lookup_.find(label) == label_lookup_.end());
        assert(num_stored_points_ < max_elements_);

        new_node_id = num_stored_points_;
        link_lists_.emplace_back();
        ++num_stored_points_;
        label_lookup_[label] = new_node_id;
        node_level = random_level(level_multiplier_);
    }

    std::unique_lock<std::mutex> node_lock(link_list_locks_[new_node_id]);

    element_levels_[new_node_id] = node_level;
    std::unique_lock<std::mutex> entry_point_lock(entry_point_mutex_);
    int previous_max_level = max_level_;
    PID current_id = entry_point_id_;
    if (node_level <= previous_max_level) {
        entry_point_lock.unlock();
    }

    // Initialize the new record.
    std::memset(
        data_level0_memory_.get() + (new_node_id * level0_record_bytes_),
        0,
        level0_record_bytes_
    );

    // Initialisation of label and cluster id
    set_external_label(new_node_id, label);
    set_raw_cluster_id(new_node_id, cluster_id);

    // Quantize raw data and initialize quantized data
    std::vector<float> rotated_data(padded_dim_);
    rotator_->rotate(construction_data_ + (label * dim_), rotated_data.data());
    quant::quantize_split_single(
        rotated_data.data(),
        reinterpret_cast<float*>(centroids_memory_.get()) + (cluster_id * padded_dim_),
        padded_dim_,
        ex_bits_,
        bin_data(new_node_id),
        ex_data(new_node_id),
        metric_type_,
        config
    );

    // If the current vertex is at level >0, it needs some space to store the extra edges.
    if (node_level > 0) {
        link_lists_[new_node_id].reset(
            static_cast<char*>(std::malloc((upper_layer_links_bytes_ * node_level) + 1))
        );
        if (link_lists_[new_node_id] == nullptr) {
            throw std::bad_alloc();
        }
        std::memset(
            link_lists_[new_node_id].get(), 0, (upper_layer_links_bytes_ * node_level) + 1
        );
    }

    if (current_id != kPidMax) {
        const auto distance = [this, new_node_id](PID id) {
            return raw_distance(id, new_node_id);
        };
        const auto prefetch = [this](PID id) {
            memory::mem_prefetch_l1(
                reinterpret_cast<const char*>(
                    construction_data_ + (external_label(id) * dim_)
                ),
                padded_dim_ / 16
            );
        };
        if (node_level < previous_max_level) {
            current_id =
                greedy_descend(current_id, previous_max_level, node_level, distance);
        }

        for (int level = std::min(node_level, previous_max_level); level >= 0; --level) {
            maxheap<std::pair<float, PID>> top_candidates =
                search_layer(current_id, level, distance, prefetch);
            current_id = mutually_connect_new_element(new_node_id, top_candidates, level);
        }
    }

    // A taller node becomes the entry point; entry_point_lock is still held then.
    if (node_level > previous_max_level) {
        entry_point_id_ = new_node_id;
        max_level_ = node_level;
    }
}

namespace {
template <class Distance>
void prune_neighbors(
    maxheap<std::pair<float, PID>>& top_candidates, size_t M, Distance distance
) {
    if (top_candidates.size() < M) {
        return;
    }

    minheap<std::pair<float, PID>> closest_first;
    std::vector<std::pair<float, PID>> selected;
    while (!top_candidates.empty()) {
        closest_first.emplace(top_candidates.top());
        top_candidates.pop();
    }

    while (!closest_first.empty()) {
        if (selected.size() >= M) {
            break;
        }
        std::pair<float, PID> current_pair = closest_first.top();
        float query_distance = current_pair.first;
        closest_first.pop();
        bool keep = true;

        for (std::pair<float, PID> selected_pair : selected) {
            float current_distance = distance(selected_pair.second, current_pair.second);
            if (current_distance < query_distance) {
                keep = false;
                break;
            }
        }
        if (keep) {
            selected.push_back(current_pair);
        }
    }

    for (std::pair<float, PID> current_pair : selected) {
        top_candidates.emplace(current_pair);
    }
}
}  // namespace

template <class Distance, class ReverseEdgeDistance>
PID HierarchicalNSW::mutually_connect(
    PID new_node_id,
    maxheap<std::pair<float, PID>>& top_candidates,
    int level,
    Distance distance,
    ReverseEdgeDistance reverse_edge_distance
) {
    size_t max_degree = level > 0 ? upper_layer_max_degree_ : base_layer_max_degree_;
    prune_neighbors(top_candidates, M_, distance);
    if (top_candidates.size() > M_) {
        throw std::runtime_error("HNSW pruning must not return more than M candidates");
    }

    std::vector<PID> selected_neighbors;
    selected_neighbors.reserve(M_);
    while (!top_candidates.empty()) {
        selected_neighbors.push_back(top_candidates.top().second);
        top_candidates.pop();
    }
    if (selected_neighbors.empty()) {
        throw std::runtime_error("HNSW selected no neighbors for the new element");
    }
    PID next_closest_entry_point = selected_neighbors.back();

    {
        PID* new_links = (level == 0) ? level0_links(new_node_id)
                                      : upper_layer_links(new_node_id, level);
        if (*new_links > 0) {
            throw std::runtime_error("HNSW new element must have an empty link list");
        }

        set_link_count(new_links, selected_neighbors.size());
        auto* neighbors = new_links + 1;
        for (size_t i = 0; i < selected_neighbors.size(); ++i) {
            if (neighbors[i] != 0) {
                throw std::runtime_error("HNSW new element link list is not zeroed");
            }
            if (level > element_levels_[selected_neighbors[i]]) {
                throw std::runtime_error("HNSW cannot link on a nonexistent level");
            }
            neighbors[i] = selected_neighbors[i];
        }
    }

    for (auto selected_neighbor : selected_neighbors) {
        std::unique_lock<std::mutex> lock(link_list_locks_[selected_neighbor]);

        PID* neighbor_links = (level == 0) ? level0_links(selected_neighbor)
                                           : upper_layer_links(selected_neighbor, level);
        size_t neighbor_degree = link_count(neighbor_links);

        if (neighbor_degree > max_degree) {
            throw std::runtime_error("HNSW neighbor degree exceeds the layer maximum");
        }
        if (selected_neighbor == new_node_id) {
            throw std::runtime_error("HNSW cannot connect an element to itself");
        }
        if (level > element_levels_[selected_neighbor]) {
            throw std::runtime_error("HNSW cannot link on a nonexistent level");
        }

        auto* neighbors = neighbor_links + 1;

        bool already_linked = false;
        for (size_t j = 0; j < neighbor_degree; ++j) {
            if (neighbors[j] == new_node_id) {
                already_linked = true;
                break;
            }
        }

        if (!already_linked) {
            if (neighbor_degree < max_degree) {
                neighbors[neighbor_degree] = new_node_id;
                set_link_count(neighbor_links, neighbor_degree + 1);
            } else {
                float new_neighbor_distance = reverse_edge_distance(selected_neighbor);
                maxheap<std::pair<float, PID>> candidates;
                candidates.emplace(new_neighbor_distance, new_node_id);
                for (size_t j = 0; j < neighbor_degree; ++j) {
                    candidates.emplace(
                        distance(neighbors[j], selected_neighbor), neighbors[j]
                    );
                }

                prune_neighbors(candidates, max_degree, distance);

                int new_degree = 0;
                while (!candidates.empty()) {
                    neighbors[new_degree] = candidates.top().second;
                    candidates.pop();
                    ++new_degree;
                }
                set_link_count(neighbor_links, new_degree);
            }
        }
    }

    return next_closest_entry_point;
}

PID HierarchicalNSW::mutually_connect_new_element(
    PID new_node_id, maxheap<std::pair<float, PID>>& top_candidates, int level
) {
    return mutually_connect(
        new_node_id,
        top_candidates,
        level,
        [this](PID target, PID query) { return raw_distance(target, query); },
        // Preserve the original operand order for a newly proposed reverse edge.
        [this, new_node_id](PID neighbor) { return raw_distance(neighbor, new_node_id); }
    );
}

std::vector<std::vector<std::pair<float, PID>>> HierarchicalNSW::search(
    const float* queries, size_t num_queries, size_t k, size_t ef_search, size_t num_threads
) {
    if (level0_record_bytes_ == 0) {
        throw std::logic_error("HNSW index must be initialized before search");
    }
    if (k == 0) {
        throw std::invalid_argument("HNSW k must be greater than zero");
    }
    if (num_queries != 0) {
        if (queries == nullptr) {
            throw std::invalid_argument("HNSW search queries must not be null");
        }
        check_coordinates(queries, num_queries * dim_, dim_, "HNSW search queries");
    }
    std::vector<std::vector<std::pair<float, PID>>> results(num_queries);
    index_detail::parallel_for(
        0,
        num_queries,
        num_threads,
        [&](size_t index, size_t /*thread*/) {
            std::vector<float> rotated_query(padded_dim_);
            const float* query = queries + (index * dim_);
            std::vector<float> normalized_query;
            if (cosine_) {
                normalized_query.resize(dim_);
                index_detail::normalize_vector(query, normalized_query.data(), dim_);
                query = normalized_query.data();
            }
            rotator_->rotate(query, rotated_query.data());
            results[index] = detail::search_knn_sorted(
                *this, rotated_query.data(), k, std::max(ef_search, k)
            );
        },
        /*reuse_workers=*/true
    );
    return results;
}

std::vector<std::pair<float, PID>> HierarchicalNSW::search_knn_sorted_with(
    detail::SearchLayersFn search_layers,
    const float* rotated_query,
    size_t k,
    size_t ef_search
) const {
    std::vector<std::pair<float, PID>> result;
    if (num_stored_points_ == 0) {
        return result;
    }
    if (ef_search == 0) {
        ef_search = kDefaultEfSearch;
    }
    // No candidate pool can hold more distinct points than are stored, so a huge
    // ef_search must not turn into a huge allocation.
    ef_search = std::min(ef_search, num_stored_points_.load());

    SplitSingleQuery<float> query_wrapper(
        rotated_query, padded_dim_, ex_bits_, query_config_, metric_type_
    );

    // Preprocess - get the distance from query to all centroids
    std::vector<float> q_to_centroids(num_clusters_);

    if (metric_type_ == METRIC_L2) {
        for (size_t i = 0; i < num_clusters_; ++i) {
            q_to_centroids[i] = std::sqrt(raw_dist_func_(
                rotated_query,
                reinterpret_cast<const float*>(centroids_memory_.get()) + (i * padded_dim_),
                padded_dim_
            ));
        }
    } else if (metric_type_ == METRIC_IP) {
        q_to_centroids.resize(2 * num_clusters_);
        // First half stores dot(query, centroid); second half stores ||query - centroid||.
        for (size_t i = 0; i < num_clusters_; ++i) {
            q_to_centroids[i] = dot_product(
                rotated_query,
                reinterpret_cast<const float*>(centroids_memory_.get()) + (i * padded_dim_),
                padded_dim_
            );
            q_to_centroids[i + num_clusters_] = std::sqrt(euclidean_sqr(
                rotated_query,
                reinterpret_cast<const float*>(centroids_memory_.get()) + (i * padded_dim_),
                padded_dim_
            ));
        }
    }

    auto visited = visited_set_pool_->acquire();
    // Keep the ef_search best estimates in a bounded sorted buffer.
    buffer::SearchBuffer<float> search_pool(
        std::min(std::max(ef_search, k), num_stored_points_.load())
    );
    std::vector<AnnCandidate<float>> nearest(k);
    const size_t num_results = search_layers(
        *this,
        query_wrapper,
        q_to_centroids.data(),
        *visited,
        search_pool,
        k,
        nearest.data()
    );
    visited_set_pool_->release(std::move(visited));

    result.reserve(num_results);
    for (size_t i = 0; i < num_results; ++i) {
        result.emplace_back(nearest[i].distance, external_label(nearest[i].id));
    }
    // Candidate selection uses distance only. Preserve that selection, then match
    // the former heap's (distance, external label) ordering within each tie group.
    for (auto first = result.begin(); first != result.end();) {
        auto last = first + 1;
        while (last != result.end() && last->first == first->first) {
            ++last;
        }
        std::sort(first, last);
        first = last;
    }
    return result;
}

size_t HierarchicalNSW::remove(const PID* labels, size_t num_points) {
    if (data_level0_memory_ == nullptr || centroids_memory_ == nullptr ||
        rotator_ == nullptr || num_clusters_ == 0) {
        throw std::logic_error("HNSW index must be constructed or loaded before remove");
    }
    check_writable("remove");
    if (num_points == 0) {
        return 0;
    }
    if (labels == nullptr) {
        throw std::invalid_argument("HNSW remove labels must not be null");
    }

    // Resolve every label before marking any point, so an unknown label removes
    // nothing.
    std::vector<PID> internal_ids(num_points);
    {
        std::unique_lock<std::mutex> label_lookup_lock(label_lookup_lock_);
        for (size_t i = 0; i < num_points; ++i) {
            const auto found = label_lookup_.find(labels[i]);
            if (found == label_lookup_.end()) {
                throw std::invalid_argument("HNSW remove label is not in the index");
            }
            internal_ids[i] = found->second;
        }
    }

    size_t removed = 0;
    for (const PID internal_id : internal_ids) {
        if (!is_removed(internal_id)) {
            const PID marked = raw_cluster_id(internal_id) | kRemovedMask;
            set_raw_cluster_id(internal_id, marked);
            ++removed;
        }
    }
    num_removed_ += removed;
    return removed;
}

// Code per dimension is (sign << ex_bits) + ex, up to nine bits, scaled by
// f_rescale_ex.
void HierarchicalNSW::reconstruct_rotated(PID internal_id, float* rotated) const {
    thread_local std::vector<uint16_t> combined;
    thread_local std::vector<uint8_t> ex_codes;
    combined.resize(padded_dim_);

    const ConstBinDataMap<float> bin(bin_data(internal_id), padded_dim_);
    quant::rabitq_impl::unpack_sign_bits(bin.bin_code(), padded_dim_, combined.data());

    const auto* centroid = reinterpret_cast<const float*>(centroids_memory_.get()) +
                           (stored_cluster_id(internal_id) * padded_dim_);

    if (ex_bits_ == 0) {
        quant::reconstruct_full_vec<float, uint16_t>(
            combined.data(),
            centroid,
            padded_dim_,
            1,
            bin.f_rescale(),
            rotated,
            metric_type_
        );
        return;
    }

    ex_codes.resize(padded_dim_);
    const ConstExDataMap<float> ex(ex_data(internal_id), padded_dim_, ex_bits_);
    quant::rabitq_impl::ex_bits::unpacking_rabitqplus_code(
        ex.ex_code(), ex_codes.data(), padded_dim_, ex_bits_
    );
    for (size_t d = 0; d < padded_dim_; ++d) {
        combined[d] = static_cast<uint16_t>((combined[d] << ex_bits_) + ex_codes[d]);
    }

    quant::reconstruct_full_vec<float, uint16_t>(
        combined.data(),
        centroid,
        padded_dim_,
        ex_bits_ + 1,
        ex.f_rescale_ex(),
        rotated,
        metric_type_
    );
}

// Only `query` is reconstructed; `target` is scored from its codes, as QGBuilder
// does. Callers hold `query` fixed across a loop, so it is prepared once.
float HierarchicalNSW::quantized_distance(
    PID target, PID query, detail::QuantQuery& cached
) const {
    if (cached.id != query || !cached.prepared.has_value()) {
        cached.rotated.resize(padded_dim_);
        reconstruct_rotated(query, cached.rotated.data());
        cached.prepared.emplace(
            cached.rotated.data(), padded_dim_, ex_bits_, query_config_, metric_type_
        );
        cached.id = query;
    }

    const auto* centroid = reinterpret_cast<const float*>(centroids_memory_.get()) +
                           (stored_cluster_id(target) * padded_dim_);
    const float dist_sqr = euclidean_sqr(cached.rotated.data(), centroid, padded_dim_);
    const float g_error = std::sqrt(dist_sqr);
    const float g_add = (metric_type_ == METRIC_IP)
                            ? -dot_product(cached.rotated.data(), centroid, padded_dim_)
                            : dist_sqr;

    if (ex_bits_ == 0) {
        return estimate_split_single(
                   bin_data(target), *cached.prepared, padded_dim_, g_add, g_error
        )
            .distance;
    }
    return estimate_split_single_full(
               bin_data(target),
               ex_data(target),
               ip_func_,
               *cached.prepared,
               padded_dim_,
               ex_bits_,
               g_add,
               g_error
    )
        .distance;
}

PID HierarchicalNSW::mutually_connect_quant(
    PID new_node_id,
    maxheap<std::pair<float, PID>>& top_candidates,
    int level,
    detail::QuantQuery& query
) {
    return mutually_connect(
        new_node_id,
        top_candidates,
        level,
        [this, &query](PID target, PID source) {
            return quantized_distance(target, source, query);
        },
        // Quantized scoring is directional: reconstruct the existing neighbor.
        [this, new_node_id, &query](PID neighbor) {
            return quantized_distance(new_node_id, neighbor, query);
        }
    );
}

// The slot, the level and the link list are all decided by the caller, so this
// can run on many threads at once and the labels do not depend on their order.
void HierarchicalNSW::add_point_quant(
    PID new_node_id, int node_level, LinkStorage& link_list, detail::QuantQuery& query
) {
    std::unique_lock<std::mutex> node_lock(link_list_locks_[new_node_id]);

    std::unique_lock<std::mutex> entry_point_lock(entry_point_mutex_);
    int previous_max_level = max_level_;
    PID current_id = entry_point_id_;
    if (node_level <= previous_max_level) {
        entry_point_lock.unlock();
    }
    const PID previous_entry_point = current_id;

    link_lists_[new_node_id] = std::move(link_list);
    element_levels_[new_node_id] = node_level;

    try {
        if (current_id != kPidMax) {
            const auto distance = [this, new_node_id, &query](PID id) {
                return quantized_distance(id, new_node_id, query);
            };
            if (node_level < previous_max_level) {
                current_id =
                    greedy_descend(current_id, previous_max_level, node_level, distance);
            }

            for (int level = std::min(node_level, previous_max_level); level >= 0;
                 --level) {
                maxheap<std::pair<float, PID>> top_candidates =
                    search_layer(current_id, level, distance, [](PID) {});
                current_id =
                    mutually_connect_quant(new_node_id, top_candidates, level, query);
            }
        }

        if (node_level > previous_max_level) {
            entry_point_id_ = new_node_id;
            max_level_ = node_level;
        }
    } catch (...) {
        // Upper layers may already expose this node. Keep a route to the old
        // graph if base-layer linking was interrupted before choosing neighbors.
        PID* base_links = level0_links(new_node_id);
        if (previous_entry_point != kPidMax && link_count(base_links) == 0) {
            base_links[1] = previous_entry_point;
            set_link_count(base_links, 1);
        }
        // An interrupted promotion has no incoming links above the previous
        // maximum level. Discard those levels so save/load sees a valid graph.
        if (node_level > previous_max_level) {
            element_levels_[new_node_id] = std::max(0, previous_max_level);
            if (element_levels_[new_node_id] == 0) {
                link_lists_[new_node_id].reset();
            }
        }
        throw;
    }
}

// A capacity change reallocates the level-0 block, invalidating pointers into it.
// This must not run while another thread searches; equal capacity is a no-op.
void HierarchicalNSW::resize(size_t new_max_elements) {
    if (level0_record_bytes_ == 0) {
        throw std::logic_error("HNSW index must be initialized before resize");
    }
    check_writable("resize");
    if (new_max_elements == 0) {
        throw std::invalid_argument("HNSW resize capacity must be greater than zero");
    }
    if (new_max_elements < num_stored_points_) {
        throw std::invalid_argument("HNSW resize capacity is below the element count");
    }
    if (new_max_elements > buffer::kSearchBufferMaxPointCount) {
        throw std::invalid_argument("HNSW resize capacity exceeds the supported ID range");
    }
    if (new_max_elements == max_elements_) {
        return;
    }
    if (new_max_elements > std::numeric_limits<size_t>::max() / level0_record_bytes_) {
        throw std::invalid_argument("HNSW resize capacity exceeds addressable storage");
    }

    // Everything new is built before anything old is released, so a failure
    // leaves the index as it was. realloc is not an option: the level-0 block
    // comes from aligned_alloc, which realloc does not keep aligned.
    const size_t level0_bytes = new_max_elements * level0_record_bytes_;
    AlignedStorage level0(memory::huge_page_allocate<char>(level0_bytes));
    std::memset(level0.get(), 0, level0_bytes);
    std::memcpy(
        level0.get(), data_level0_memory_.get(), num_stored_points_ * level0_record_bytes_
    );

    std::vector<LinkStorage> links;
    links.reserve(new_max_elements);

    std::vector<int> levels(element_levels_);
    levels.resize(new_max_elements);
    std::vector<std::mutex> locks(new_max_elements);
    auto pool = std::make_unique<VisitedSetPool>(1, new_max_elements);

    // All potentially throwing work is complete. Moving the live owners into
    // reserved slots cannot allocate or throw; unused capacity stays uninitialized.
    for (auto& link : link_lists_) {
        links.push_back(std::move(link));
    }
    data_level0_memory_ = std::move(level0);
    link_lists_ = std::move(links);
    element_levels_.swap(levels);
    link_list_locks_.swap(locks);
    visited_set_pool_ = std::move(pool);
    max_elements_ = new_max_elements;
}

std::vector<PID> HierarchicalNSW::add(
    const float* data,
    size_t num_points,
    const PID* cluster_ids,
    bool fast_quantization,
    size_t num_threads
) {
    if (data_level0_memory_ == nullptr || centroids_memory_ == nullptr ||
        rotator_ == nullptr || num_clusters_ == 0) {
        throw std::logic_error(
            "HNSW index must be trained, constructed or loaded before add"
        );
    }
    check_writable("add");
    if (num_points == 0) {
        return {};
    }
    if (data == nullptr) {
        throw std::invalid_argument("HNSW add inputs must not be null");
    }
    if (num_points > max_elements_ - num_stored_points_) {
        throw std::invalid_argument("HNSW add exceeds the remaining capacity");
    }
    if (num_points > buffer::kSearchBufferMaxPointCount - num_stored_points_) {
        throw std::invalid_argument("HNSW add point count exceeds the supported ID range");
    }
    check_coordinates(data, num_points * dim_, dim_, "HNSW add data");
    std::vector<float> normalized;
    if (cosine_) {
        normalized = index_detail::normalized_rows(data, num_points, dim_, num_threads);
        data = normalized.data();
    }
    std::vector<PID> labels(num_points);
    // Either the caller names the cluster of every point, or each one goes to its
    // nearest centroid, ordered the way a query is routed.
    std::vector<PID> assigned(num_points, 0);
    if (cluster_ids != nullptr) {
        for (size_t i = 0; i < num_points; ++i) {
            if (cluster_ids[i] >= num_clusters_) {
                throw std::invalid_argument("HNSW cluster ID is out of range");
            }
            assigned[i] = cluster_ids[i];
        }
    } else {
        const auto* centroids = reinterpret_cast<const float*>(centroids_memory_.get());
        index_detail::parallel_for(0, num_points, num_threads, [&](size_t i, size_t) {
            std::vector<float> rotated(padded_dim_);
            rotator_->rotate(data + (i * dim_), rotated.data());
            float best = std::numeric_limits<float>::max();
            for (size_t c = 0; c < num_clusters_; ++c) {
                const float dist = raw_dist_func_(
                    rotated.data(), centroids + (c * padded_dim_), padded_dim_
                );
                if (dist < best) {
                    best = dist;
                    assigned[i] = static_cast<PID>(c);
                }
            }
        });
    }

    if (num_stored_points_ == 0) {
        // The exact vectors are at hand, so link them as construct would.
        link_exact(num_points, data, assigned.data(), num_threads, fast_quantization);
        std::iota(labels.begin(), labels.end(), PID{0});
        return labels;
    }

    quant::RabitqConfig config;
    if (fast_quantization) {
        config = quant::faster_config(padded_dim_, ex_bits_ + 1);
    }

    // Drawn here because the generator is shared: the inserts below run in
    // parallel and must not touch it.
    std::vector<int> levels(num_points);
    for (size_t i = 0; i < num_points; ++i) {
        levels[i] = random_level(level_multiplier_);
    }

    // Every link list is allocated before any of them is installed, so running
    // out of memory leaves the index untouched.
    std::vector<LinkStorage> link_lists;
    link_lists.reserve(num_points);
    for (size_t i = 0; i < num_points; ++i) {
        char* raw = nullptr;
        if (levels[i] > 0) {
            raw = static_cast<char*>(
                std::calloc(1, (upper_layer_links_bytes_ * levels[i]) + 1)
            );
            if (raw == nullptr) {
                throw std::bad_alloc();
            }
        }
        link_lists.emplace_back(raw);
    }

    const size_t num_workers = std::min(resolve_num_threads(num_threads), num_points);
    std::vector<detail::QuantQuery> queries(num_workers);

    // Prepare every code before publishing the slots. If quantization or worker
    // creation fails, the graph and point count are unchanged.
    {
        std::unique_lock<std::mutex> label_lookup_lock(label_lookup_lock_);
        if (num_stored_points_ + num_points > max_elements_) {
            throw std::runtime_error("HNSW add exceeds the remaining capacity");
        }
        const auto first = static_cast<PID>(num_stored_points_.load());
        index_detail::parallel_for(0, num_points, num_threads, [&](size_t i, size_t) {
            const auto id = static_cast<PID>(first + i);
            labels[i] = id;
            std::memset(
                data_level0_memory_.get() + (id * level0_record_bytes_),
                0,
                level0_record_bytes_
            );
            set_external_label(id, id);
            set_raw_cluster_id(id, assigned[i]);
            std::vector<float> rotated_data(padded_dim_);
            rotator_->rotate(data + (i * dim_), rotated_data.data());
            quant::quantize_split_single(
                rotated_data.data(),
                reinterpret_cast<float*>(centroids_memory_.get()) +
                    (assigned[i] * padded_dim_),
                padded_dim_,
                ex_bits_,
                bin_data(id),
                ex_data(id),
                metric_type_,
                config
            );
        });
        size_t registered = 0;
        try {
            for (; registered < num_points; ++registered) {
                label_lookup_[labels[registered]] = labels[registered];
            }
        } catch (...) {
            for (size_t i = 0; i < registered; ++i) {
                label_lookup_.erase(labels[i]);
            }
            throw;
        }
        // Unstarted inserts are valid level-0 nodes with owned codes. Their
        // upper-layer lists remain under RAII until an insert takes ownership.
        link_lists_.resize(num_stored_points_ + num_points);
        for (size_t i = 0; i < num_points; ++i) {
            element_levels_[labels[i]] = 0;
        }
        num_stored_points_ += num_points;
    }

    index_detail::parallel_for(0, num_points, num_workers, [&](size_t i, size_t thread) {
        add_point_quant(labels[i], levels[i], link_lists[i], queries[thread]);
    });

    return labels;
}

}  // namespace rabitqlib::hnsw
