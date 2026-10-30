#include "rabitqlib/index/hnsw/hnsw.hpp"

#include <algorithm>
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
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

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
#include "rabitqlib/index/hnsw/hnsw_quant.hpp"
#include "rabitqlib/index/types.hpp"
#include "rabitqlib/utils/memory.hpp"
#include "rabitqlib/utils/path.hpp"
#include "rabitqlib/utils/space.hpp"
#include "rabitqlib/utils/tools.hpp"

namespace rabitqlib::hnsw {

HierarchicalNSW::HierarchicalNSW() = default;
HierarchicalNSW::~HierarchicalNSW() = default;
void HierarchicalNSW::AlignedDeleter::operator()(char* pointer) const noexcept {
    memory::aligned_deallocate(pointer);
}
void HierarchicalNSW::LinkDeleter::operator()(char* pointer) const noexcept {
    std::free(pointer);
}
int HierarchicalNSW::get_random_level(double reverse_size) {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double r = -log(distribution(level_generator_)) * reverse_size;
    return static_cast<int>(r);
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
    : metric_type_(metric_type)
    , label_op_locks_(kMaxLabelOperationLock)
    , link_list_locks_(max_elements)
    , element_levels_(max_elements)
    , raw_dist_func_(
          (metric_type == METRIC_IP) ? dot_product_dis<float> : euclidean_sqr<float>
      ) {
    static_assert(
        buffer::kSearchBufferMaxPointCount <= kRemovedMask,
        "cluster IDs must leave the removal bit free"
    );
    validate_metric_type(metric_type);
    if (M < 2) {
        throw std::invalid_argument("HNSW M must be at least 2");
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
    /* check size */
    assert(padded_dim_ % 32 == 0);
    assert(padded_dim_ >= dim_);
    ex_bits_ = total_bits - 1;

    if (total_bits < 1 || total_bits > 9) {
        throw std::invalid_argument("HNSW quantization bits must be in [1, 9]");
    };

    assert(padded_dim_ % 32 == 0);

    ip_func_ = select_excode_ipfunc(ex_bits_);

    if (M <= 10000) {
        M_ = M;
    } else {
        M_ = 10000;
    }

    upper_layer_max_degree_ = M_;
    base_layer_max_degree_ = M_ * 2;
    ef_construction_ = std::max(ef_construction, M_);
    default_ef_search_ = 10;

    bin_data_bytes_ = BinDataMap<float>::data_bytes(padded_dim_);
    ex_data_bytes_ = ExDataMap<float>::data_bytes(padded_dim_, ex_bits_);
    level0_links_bytes_ = base_layer_max_degree_ * sizeof(PID) + sizeof(PID);
    label_offset_ =
        level0_links_bytes_ + sizeof(PID);  // (# of edges + edges) + (cluster_id)
    bin_data_offset_ =
        label_offset_ +
        sizeof(PID);  // (# of edges + edges) + (cluster_id) + (external label)
    ex_data_offset_ =
        bin_data_offset_ + bin_data_bytes_;  // (# of edges + edges) + (cluster_id)
                                             // + (external label) + (BinData)
    level0_record_bytes_ = ex_data_offset_ + ex_data_bytes_;
    data_level0_memory_.reset(
        memory::huge_page_allocate<char>(max_elements_ * level0_record_bytes_)
    );
    if (data_level0_memory_ == nullptr) {
        throw std::runtime_error("Not enough memory");
    }

    level_generator_.seed(random_seed);

    num_stored_points_ = 0;

    visited_list_pool_ = std::make_unique<VisitedListPool>(1, max_elements_);

    // initializations for special treatment of the first node
    entry_point_id_ = -1;
    max_level_ = -1;

    link_lists_.reserve(max_elements_);
    upper_layer_links_bytes_ = upper_layer_max_degree_ * sizeof(PID) + sizeof(PID);
    mult_ = 1 / log(1.0 * static_cast<double>(M_));

    this->query_config_ =
        quant::faster_config(padded_dim_, SplitSingleQuery<float>::kNumBits);
}

void HierarchicalNSW::save(const char* filename) const {
    if (level0_record_bytes_ == 0) {
        throw std::logic_error("HNSW index must be initialized before save");
    }
    std::ofstream output(rabitqlib::io_impl::filesystem_path(filename), std::ios::binary);
    if (!output.is_open()) {
        throw std::runtime_error("HNSW: cannot open index file for writing");
    }

    output.write(reinterpret_cast<const char*>(&max_elements_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&num_stored_points_), sizeof(size_t));

    output.write(reinterpret_cast<const char*>(&dim_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&padded_dim_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&num_clusters_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&ex_bits_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&metric_type_), sizeof(metric_type_));

    output.write(reinterpret_cast<const char*>(&bin_data_bytes_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&ex_data_bytes_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&level0_links_bytes_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&bin_data_offset_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&ex_data_offset_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&label_offset_), sizeof(PID));
    output.write(reinterpret_cast<const char*>(&level0_record_bytes_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&upper_layer_links_bytes_), sizeof(size_t));

    output.write(reinterpret_cast<const char*>(&max_level_), sizeof(int));
    output.write(reinterpret_cast<const char*>(&entry_point_id_), sizeof(PID));

    output.write(reinterpret_cast<const char*>(&M_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&upper_layer_max_degree_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&base_layer_max_degree_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&mult_), sizeof(double));
    output.write(reinterpret_cast<const char*>(&ef_construction_), sizeof(size_t));

    output.write(
        reinterpret_cast<const char*>(centroids_memory_.get()),
        static_cast<std::streamsize>(num_clusters_ * padded_dim_ * sizeof(float))
    );

    output.write(
        reinterpret_cast<const char*>(data_level0_memory_.get()),
        static_cast<std::streamsize>(num_stored_points_ * level0_record_bytes_)
    );

    for (size_t i = 0; i < num_stored_points_; i++) {
        unsigned int link_list_size =
            element_levels_[i] > 0 ? upper_layer_links_bytes_ * element_levels_[i] : 0;
        output.write(reinterpret_cast<const char*>(&link_list_size), sizeof(unsigned int));
        if (link_list_size != 0) {
            output.write(link_lists_[i].get(), link_list_size);
        }
    }

    rotator_->save(output);
    output.close();
    if (!output) {
        throw std::runtime_error("HNSW: failed to write index file");
    }
}

void HierarchicalNSW::load(const char* filename) {
    if (filename == nullptr || filename[0] == '\0') {
        throw std::invalid_argument("HNSW: index filename must not be empty");
    }
    std::ifstream input(rabitqlib::io_impl::filesystem_path(filename), std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("HNSW: cannot open index file");
    }

    auto invalid_file = []() -> void {
        throw std::runtime_error("HNSW: invalid or truncated index file");
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

    // Keep the current index intact until the complete file has been validated.
    HierarchicalNSW loaded;
    size_t element_count = 0;
    PID label_offset = 0;
    read_header(loaded.max_elements_);
    read_header(element_count);
    read_header(loaded.dim_);
    read_header(loaded.padded_dim_);
    read_header(loaded.num_clusters_);
    read_header(loaded.ex_bits_);
    read_header(loaded.metric_type_);
    read_header(loaded.bin_data_bytes_);
    read_header(loaded.ex_data_bytes_);
    read_header(loaded.level0_links_bytes_);
    read_header(loaded.bin_data_offset_);
    read_header(loaded.ex_data_offset_);
    read_header(label_offset);
    loaded.label_offset_ = label_offset;
    read_header(loaded.level0_record_bytes_);
    read_header(loaded.upper_layer_links_bytes_);
    read_header(loaded.max_level_);
    read_header(loaded.entry_point_id_);
    read_header(loaded.M_);
    read_header(loaded.upper_layer_max_degree_);
    read_header(loaded.base_layer_max_degree_);
    read_header(loaded.mult_);
    read_header(loaded.ef_construction_);

    if (loaded.max_elements_ == 0 ||
        loaded.max_elements_ > buffer::kSearchBufferMaxPointCount ||
        element_count > loaded.max_elements_ ||
        loaded.num_clusters_ > buffer::kSearchBufferMaxPointCount ||
        (element_count != 0 && loaded.num_clusters_ == 0) || loaded.dim_ < 64 ||
        loaded.dim_ > rotator_impl::FhtKacRotator::kMaxDim ||
        loaded.padded_dim_ < loaded.dim_ || loaded.padded_dim_ % 32 != 0 ||
        loaded.padded_dim_ > rotator_impl::FhtKacRotator::kMaxDim || loaded.ex_bits_ > 8 ||
        (loaded.metric_type_ != METRIC_L2 && loaded.metric_type_ != METRIC_IP) ||
        loaded.M_ < 2 || loaded.M_ > 10000 || loaded.upper_layer_max_degree_ != loaded.M_ ||
        loaded.base_layer_max_degree_ != 2 * loaded.M_ ||
        loaded.ef_construction_ < loaded.M_ ||
        (element_count == 0 &&
         (loaded.max_level_ != -1 || loaded.entry_point_id_ != kPidMax)) ||
        (element_count != 0 &&
         (loaded.max_level_ < 0 || loaded.entry_point_id_ >= element_count))) {
        invalid_file();
    }

    const size_t expected_level0 = (loaded.base_layer_max_degree_ + 1) * sizeof(PID);
    const size_t expected_bin = BinDataMap<float>::data_bytes(loaded.padded_dim_);
    const size_t expected_ex =
        ExDataMap<float>::data_bytes(loaded.padded_dim_, loaded.ex_bits_);
    const size_t expected_label = expected_level0 + sizeof(PID);
    const size_t expected_bin_offset = expected_label + sizeof(PID);
    const size_t expected_ex_offset = expected_bin_offset + expected_bin;
    const size_t expected_level0_record_bytes = expected_ex_offset + expected_ex;
    if (loaded.bin_data_bytes_ != expected_bin || loaded.ex_data_bytes_ != expected_ex ||
        loaded.level0_links_bytes_ != expected_level0 ||
        loaded.label_offset_ != expected_label ||
        loaded.bin_data_offset_ != expected_bin_offset ||
        loaded.ex_data_offset_ != expected_ex_offset ||
        loaded.level0_record_bytes_ != expected_level0_record_bytes ||
        loaded.upper_layer_links_bytes_ !=
            (loaded.upper_layer_max_degree_ + 1) * sizeof(PID) ||
        !std::isfinite(loaded.mult_) || loaded.mult_ <= 0.0) {
        invalid_file();
    }

    const size_t centroids_bytes = checked_product(
        checked_product(loaded.num_clusters_, loaded.padded_dim_), sizeof(float)
    );
    const size_t data_capacity_bytes =
        checked_product(loaded.max_elements_, loaded.level0_record_bytes_);
    const size_t data_bytes = checked_product(element_count, loaded.level0_record_bytes_);
    // Validate pointer-table capacity before allocating it.
    (void)checked_product(loaded.max_elements_, sizeof(LinkStorage));
    const size_t link_headers_bytes = checked_product(element_count, sizeof(unsigned int));
    loaded.rotator_ = rotator_impl::make_rotator_for_loading(
        loaded.dim_, RotatorType::FhtKacRotator, loaded.padded_dim_
    );
    const size_t rotator_bytes = loaded.rotator_->dump_bytes();

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
        loaded.centroids_memory_.reset(memory::huge_page_allocate<char>(centroids_bytes));
        if (loaded.centroids_memory_ == nullptr) {
            throw std::bad_alloc();
        }
        read_payload(loaded.centroids_memory_.get(), centroids_bytes);
    }
    loaded.data_level0_memory_.reset(memory::huge_page_allocate<char>(data_capacity_bytes));
    if (loaded.data_level0_memory_ == nullptr) {
        throw std::bad_alloc();
    }
    read_payload(loaded.data_level0_memory_.get(), data_bytes);

    loaded.element_levels_ = std::vector<int>(loaded.max_elements_);
    loaded.link_list_locks_ = std::vector<std::mutex>(loaded.max_elements_);
    loaded.label_op_locks_ = std::vector<std::mutex>(kMaxLabelOperationLock);
    loaded.link_lists_.reserve(loaded.max_elements_);
    loaded.link_lists_.resize(element_count);
    for (size_t i = 0; i < element_count; ++i) {
        const char* row =
            loaded.data_level0_memory_.get() + i * loaded.level0_record_bytes_;
        PID cluster_id = 0;
        PID label = 0;
        unsigned short level0_degree = 0;
        std::memcpy(&level0_degree, row, sizeof(level0_degree));
        std::memcpy(&cluster_id, row + loaded.level0_links_bytes_, sizeof(cluster_id));
        std::memcpy(&label, row + loaded.label_offset_, sizeof(label));
        if ((cluster_id & ~kRemovedMask) >= loaded.num_clusters_ ||
            level0_degree > loaded.base_layer_max_degree_) {
            invalid_file();
        }
        loaded.num_removed_ += (cluster_id & kRemovedMask) != 0;
        for (size_t edge = 0; edge < level0_degree; ++edge) {
            PID neighbor = 0;
            std::memcpy(&neighbor, row + sizeof(PID) + edge * sizeof(PID), sizeof(PID));
            if (neighbor >= element_count) {
                invalid_file();
            }
        }
        if (!loaded.label_lookup_.emplace(label, static_cast<PID>(i)).second) {
            invalid_file();
        }

        unsigned int link_list_size = 0;
        read_payload(reinterpret_cast<char*>(&link_list_size), sizeof(link_list_size));
        if (link_list_size != 0) {
            if (link_list_size % loaded.upper_layer_links_bytes_ != 0 ||
                link_list_size / loaded.upper_layer_links_bytes_ >
                    static_cast<size_t>(loaded.max_level_) ||
                link_list_size > remaining) {
                invalid_file();
            }
            loaded.link_lists_[i].reset(static_cast<char*>(std::malloc(link_list_size)));
            if (loaded.link_lists_[i] == nullptr) {
                throw std::bad_alloc();
            }
            loaded.element_levels_[i] =
                static_cast<int>(link_list_size / loaded.upper_layer_links_bytes_);
        }
        loaded.num_stored_points_ = i + 1;
        if (link_list_size != 0) {
            read_payload(loaded.link_lists_[i].get(), link_list_size);
            for (int level = 0; level < loaded.element_levels_[i]; ++level) {
                const char* link =
                    loaded.link_lists_[i].get() + level * loaded.upper_layer_links_bytes_;
                unsigned short degree = 0;
                std::memcpy(&degree, link, sizeof(degree));
                if (degree > loaded.upper_layer_max_degree_) {
                    invalid_file();
                }
                for (size_t edge = 0; edge < degree; ++edge) {
                    PID neighbor = 0;
                    std::memcpy(
                        &neighbor, link + sizeof(PID) + edge * sizeof(PID), sizeof(PID)
                    );
                    if (neighbor >= element_count) {
                        invalid_file();
                    }
                }
            }
        }
    }
    for (size_t i = 0; i < element_count; ++i) {
        for (int level = 1; level <= loaded.element_levels_[i]; ++level) {
            const char* link =
                loaded.link_lists_[i].get() +
                (static_cast<size_t>(level) - 1) * loaded.upper_layer_links_bytes_;
            unsigned short degree = 0;
            std::memcpy(&degree, link, sizeof(degree));
            for (size_t edge = 0; edge < degree; ++edge) {
                PID neighbor = 0;
                std::memcpy(
                    &neighbor, link + sizeof(PID) + edge * sizeof(PID), sizeof(PID)
                );
                if (loaded.element_levels_[neighbor] < level) {
                    invalid_file();
                }
            }
        }
    }
    if (element_count != 0 &&
        loaded.element_levels_[loaded.entry_point_id_] != loaded.max_level_) {
        invalid_file();
    }

    if (rotator_bytes > remaining) {
        invalid_file();
    }
    loaded.rotator_->load(input);
    if (!input) {
        invalid_file();
    }
    remaining -= rotator_bytes;
    if (remaining != 0) {
        invalid_file();
    }

    loaded.visited_list_pool_ = std::make_unique<VisitedListPool>(1, loaded.max_elements_);
    loaded.raw_dist_func_ =
        (loaded.metric_type_ == METRIC_IP) ? dot_product_dis<float> : euclidean_sqr<float>;
    loaded.ip_func_ = select_excode_ipfunc(loaded.ex_bits_);
    loaded.default_ef_search_ = 10;
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
    swap(default_ef_search_, loaded.default_ef_search_);
    swap(metric_type_, loaded.metric_type_);
    swap(mult_, loaded.mult_);
    swap(max_level_, loaded.max_level_);
    label_op_locks_.swap(loaded.label_op_locks_);
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
    swap(visited_list_pool_, loaded.visited_list_pool_);
    swap(ip_func_, loaded.ip_func_);
    swap(rotator_, loaded.rotator_);
    swap(query_config_, loaded.query_config_);
    swap(raw_dist_func_, loaded.raw_dist_func_);
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
        throw std::logic_error("HNSW index is already constructed or loaded");
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

    try {
        num_clusters_ = num_clusters;
        const size_t centroids_bytes = num_clusters_ * padded_dim_ * sizeof(float);
        centroids_memory_.reset(memory::huge_page_allocate<char>(centroids_bytes));
        if (centroids_memory_ == nullptr) {
            throw std::runtime_error(
                "Not enough memory: HNSW failed to allocate centroids"
            );
        }

        for (size_t i = 0; i < num_clusters; ++i) {
            this->rotator_->rotate(
                centroids + (i * dim_),
                reinterpret_cast<float*>(centroids_memory_.get()) + (i * padded_dim_)
            );
        }

        quant::RabitqConfig config;
        if (fast_quantization) {
            config = quant::faster_config(padded_dim_, ex_bits_ + 1);
        }

        construction_data_ = data;
        rabitqlib::index_detail::parallel_for(
            0, num_points, num_threads, [&](size_t idx, size_t /*threadId*/) {
                add_point(idx, cluster_ids[idx], config);
            }
        );
    } catch (...) {
        // parallel_for joins every worker before propagating an exception.
        // Discard the partial graph, retaining capacity and rotation for a retry.
        std::fill_n(element_levels_.begin(), num_stored_points_.load(), 0);
        link_lists_.clear();
        num_stored_points_ = 0;
        num_removed_ = 0;
        label_lookup_.clear();
        entry_point_id_ = kPidMax;
        max_level_ = -1;
        centroids_memory_.reset();
        num_clusters_ = 0;
        construction_data_ = nullptr;
        throw;
    }
    construction_data_ = nullptr;
}

void HierarchicalNSW::add_point(
    PID label, PID cluster_id, const quant::RabitqConfig& config
) {
    std::unique_lock<std::mutex> label_lock(get_label_op_mutex(label));

    int node_level = 0;
    PID new_node_id = 0;
    {
        std::unique_lock<std::mutex> label_lookup_lock(label_lookup_lock_);
        if (label_lookup_.find(label) != label_lookup_.end()) {
            throw std::runtime_error(
                "Currently not support replacement of existing elements, only support "
                "inserting elements with distinct labels"
            );
        }

        if (num_stored_points_ >= max_elements_) {
            throw std::runtime_error("The number of elements exceeds the specified limit");
        }

        new_node_id = num_stored_points_;
        link_lists_.emplace_back();
        num_stored_points_++;
        label_lookup_[label] = new_node_id;
        node_level = get_random_level(mult_);
    }

    std::unique_lock<std::mutex> node_lock(link_list_locks_[new_node_id]);

    element_levels_[new_node_id] = node_level;
    std::unique_lock<std::mutex> entry_point_lock(global_);
    int previous_max_level = max_level_;
    PID current_id = entry_point_id_;
    if (node_level <= previous_max_level) {
        entry_point_lock.unlock();
    }

    // initialize the current memory.
    memset(
        data_level0_memory_.get() + (new_node_id * level0_record_bytes_),
        0,
        level0_record_bytes_
    );

    // Initialisation of label and cluster id
    memcpy(get_external_label_pt(new_node_id), &label, sizeof(PID));
    memcpy(get_clusterid_pt(new_node_id), &cluster_id, sizeof(PID));

    // Quantize raw data and initialize quantized data
    std::vector<float> rotated_data(padded_dim_);
    rotator_->rotate(construction_data_ + (label * dim_), rotated_data.data());
    quant::quantize_split_single(
        rotated_data.data(),
        reinterpret_cast<float*>(centroids_memory_.get()) + (cluster_id * padded_dim_),
        padded_dim_,
        ex_bits_,
        get_bindata_by_internalid(new_node_id),
        get_exdata_by_internalid(new_node_id),
        metric_type_,
        config
    );

    // If the current vertex is at level >0, it needs some space to store the extra edges.
    if (node_level > 0) {
        link_lists_[new_node_id].reset(
            static_cast<char*>(std::malloc((upper_layer_links_bytes_ * node_level) + 1))
        );
        if (link_lists_[new_node_id] == nullptr) {
            throw std::runtime_error(
                "Not enough memory: add_point failed to allocate linklist"
            );
        }
        memset(
            link_lists_[new_node_id].get(), 0, (upper_layer_links_bytes_ * node_level) + 1
        );
    }

    if (static_cast<signed>(current_id) != -1) {
        if (node_level < previous_max_level) {
            float current_distance = get_data_dist(current_id, new_node_id);
            for (int level = previous_max_level; level > node_level; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    std::unique_lock<std::mutex> lock(link_list_locks_[current_id]);
                    const PID* data = get_linklist(current_id, level);
                    const size_t size = get_list_count(data);

                    const PID* datal = data + 1;
                    for (size_t i = 0; i < size; i++) {
                        PID cand = datal[i];
                        if (cand > max_elements_) {
                            throw std::runtime_error(
                                "HNSW construct upper-layer traversal: invalid candidate "
                                "ID " +
                                std::to_string(cand) + " at level " +
                                std::to_string(level) +
                                " (capacity=" + std::to_string(max_elements_) + ")"
                            );
                        }
                        float d = get_data_dist(cand, new_node_id);
                        if (d < current_distance) {
                            current_distance = d;
                            current_id = cand;
                            changed = true;
                        }
                    }
                }
            }
        }

        for (int level = std::min(node_level, previous_max_level); level >= 0; level--) {
            maxheap<std::pair<float, PID>> top_candidates =
                search_base_layer(current_id, new_node_id, level);
            current_id = mutually_connect_new_element(new_node_id, top_candidates, level);
        }
    } else {
        // Do nothing for the first element
        entry_point_id_ = 0;
        max_level_ = node_level;
    }

    // Releasing lock for the maximum level
    if (node_level > previous_max_level) {
        entry_point_id_ = new_node_id;
        max_level_ = node_level;
    }
}

maxheap<std::pair<float, PID>> HierarchicalNSW::search_base_layer(
    PID entry_point_id, PID new_node_id, int layer
) {
    auto visited = visited_list_pool_->get_free_vislist();

    maxheap<std::pair<float, PID>> top_candidates;
    minheap<std::pair<float, PID>> candidate_set;

    float lower_bound = get_data_dist(entry_point_id, new_node_id);
    top_candidates.emplace(lower_bound, entry_point_id);
    candidate_set.emplace(lower_bound, entry_point_id);
    visited->set(entry_point_id);

    while (!candidate_set.empty()) {
        std::pair<float, PID> curr_el_pair = candidate_set.top();
        if (curr_el_pair.first > lower_bound && top_candidates.size() == ef_construction_) {
            break;
        }
        candidate_set.pop();

        PID current_id = curr_el_pair.second;

        std::unique_lock<std::mutex> lock(link_list_locks_[current_id]);

        const PID* data =
            (layer == 0) ? get_linklist0(current_id) : get_linklist(current_id, layer);
        const size_t size = get_list_count(data);
        const PID* datal = data + 1;

        rabitqlib::memory::mem_prefetch_l1(
            reinterpret_cast<const char*>(
                construction_data_ + (get_external_label(*datal) * dim_)
            ),
            padded_dim_ / 16
        );

        rabitqlib::memory::mem_prefetch_l1(
            reinterpret_cast<const char*>(
                construction_data_ + (get_external_label(*(datal + 1)) * dim_)
            ),
            padded_dim_ / 16
        );

        for (size_t j = 0; j < size; j++) {
            PID candidate_id = *(datal + j);
            if (visited->get(candidate_id)) {
                continue;
            }
            visited->set(candidate_id);

            if (j < size - 1) {
                rabitqlib::memory::mem_prefetch_l1(
                    reinterpret_cast<const char*>(
                        construction_data_ + (get_external_label(*(datal + j + 1)) * dim_)
                    ),
                    padded_dim_ / 16
                );
            }

            float candidate_distance = get_data_dist(candidate_id, new_node_id);
            if (top_candidates.size() < ef_construction_ ||
                lower_bound > candidate_distance) {
                candidate_set.emplace(candidate_distance, candidate_id);
                top_candidates.emplace(candidate_distance, candidate_id);
                if (top_candidates.size() > ef_construction_) {
                    top_candidates.pop();
                }
                if (!top_candidates.empty()) {
                    lower_bound = top_candidates.top().first;
                }
            }
        }
    }
    visited_list_pool_->release_vis_list(std::move(visited));
    return top_candidates;
}

namespace detail {
template <class Distance>
void prune_neighbors(
    maxheap<std::pair<float, PID>>& top_candidates, size_t M, Distance distance
) {
    if (top_candidates.size() < M) {
        return;
    }

    minheap<std::pair<float, PID>> queue_closest;
    std::vector<std::pair<float, PID>> return_list;
    while (top_candidates.size() > 0) {
        queue_closest.emplace(top_candidates.top());
        top_candidates.pop();
    }

    while (queue_closest.size() > 0) {
        if (return_list.size() >= M) {
            break;
        }
        std::pair<float, PID> current_pair = queue_closest.top();
        float query_distance = current_pair.first;
        queue_closest.pop();
        bool good = true;

        for (std::pair<float, PID> second_pair : return_list) {
            float current_distance = distance(second_pair.second, current_pair.second);
            if (current_distance < query_distance) {
                good = false;
                break;
            }
        }
        if (good) {
            return_list.push_back(current_pair);
        }
    }

    for (std::pair<float, PID> current_pair : return_list) {
        top_candidates.emplace(current_pair);
    }
}
}  // namespace detail

template <class Distance, class ReverseEdgeDistance>
PID HierarchicalNSW::mutually_connect(
    PID new_node_id,
    maxheap<std::pair<float, PID>>& top_candidates,
    int level,
    Distance distance,
    ReverseEdgeDistance reverse_edge_distance
) {
    size_t max_m = level > 0 ? upper_layer_max_degree_ : base_layer_max_degree_;
    detail::prune_neighbors(top_candidates, M_, distance);
    if (top_candidates.size() > M_) {
        throw std::runtime_error(
            "Should be not be more than M_ candidates returned by the heuristic"
        );
    }

    std::vector<PID> selected_neighbors;
    selected_neighbors.reserve(M_);
    while (top_candidates.size() > 0) {
        selected_neighbors.push_back(top_candidates.top().second);
        top_candidates.pop();
    }
    if (selected_neighbors.empty()) {
        throw std::runtime_error("No neighbors selected for the new element");
    }
    PID next_closest_entry_point = selected_neighbors.back();

    {
        PID* ll_cur =
            (level == 0) ? get_linklist0(new_node_id) : get_linklist(new_node_id, level);
        if (*ll_cur > 0) {
            throw std::runtime_error(
                "The newly inserted element should have blank link list"
            );
        }

        set_list_count(ll_cur, selected_neighbors.size());
        auto* data = ll_cur + 1;
        for (size_t idx = 0; idx < selected_neighbors.size(); idx++) {
            if (data[idx] != 0) {
                throw std::runtime_error("Possible memory corruption");
            }
            if (level > element_levels_[selected_neighbors[idx]]) {
                throw std::runtime_error("Trying to make a link on a non-existent level");
            }
            data[idx] = selected_neighbors[idx];
        }
    }

    for (auto selected_neighbor : selected_neighbors) {
        std::unique_lock<std::mutex> lock(link_list_locks_[selected_neighbor]);

        PID* ll_other = (level == 0) ? get_linklist0(selected_neighbor)
                                     : get_linklist(selected_neighbor, level);
        size_t sz_link_list_other = get_list_count(ll_other);

        if (sz_link_list_other > max_m) {
            throw std::runtime_error("Bad value of sz_link_list_other");
        }
        if (selected_neighbor == new_node_id) {
            throw std::runtime_error("Trying to connect an element to itself");
        }
        if (level > element_levels_[selected_neighbor]) {
            throw std::runtime_error("Trying to make a link on a non-existent level");
        }

        auto* data = ll_other + 1;

        bool is_cur_c_present = false;
        for (size_t j = 0; j < sz_link_list_other; j++) {
            if (data[j] == new_node_id) {
                is_cur_c_present = true;
                break;
            }
        }

        if (!is_cur_c_present) {
            if (sz_link_list_other < max_m) {
                data[sz_link_list_other] = new_node_id;
                set_list_count(ll_other, sz_link_list_other + 1);
            } else {
                float new_neighbor_distance = reverse_edge_distance(selected_neighbor);
                maxheap<std::pair<float, PID>> candidates;
                candidates.emplace(new_neighbor_distance, new_node_id);
                for (size_t j = 0; j < sz_link_list_other; j++) {
                    candidates.emplace(distance(data[j], selected_neighbor), data[j]);
                }

                detail::prune_neighbors(candidates, max_m, distance);

                int indx = 0;
                while (candidates.size() > 0) {
                    data[indx] = candidates.top().second;
                    candidates.pop();
                    indx++;
                }
                set_list_count(ll_other, indx);
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
        [this](PID target, PID query) { return get_data_dist(target, query); },
        // Preserve the original operand order for a newly proposed reverse edge.
        [this, new_node_id](PID neighbor) { return get_data_dist(neighbor, new_node_id); }
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
    std::vector<std::vector<std::pair<float, PID>>> results(num_queries);
    rabitqlib::index_detail::parallel_for(
        0,
        num_queries,
        num_threads,
        [&](size_t idx, size_t /*threadId*/) {
            std::vector<float> rotated_query(padded_dim_);
            this->rotator_->rotate(queries + (idx * dim_), rotated_query.data());
            results[idx] = detail::search_knn_sorted(
                *this, rotated_query.data(), k, std::max(ef_search, k)
            );
        },
        /*reuse_workers=*/true
    );
    return results;
}

size_t HierarchicalNSW::remove(const PID* labels, size_t num_points) {
    if (data_level0_memory_ == nullptr || centroids_memory_ == nullptr ||
        rotator_ == nullptr || num_clusters_ == 0) {
        throw std::logic_error("HNSW index must be constructed or loaded before remove");
    }
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
    for (const PID id : internal_ids) {
        if (!is_removed(id)) {
            const PID marked = get_raw_clusterid(id) | kRemovedMask;
            std::memcpy(get_clusterid_pt(id), &marked, sizeof(PID));
            ++removed;
        }
    }
    num_removed_ += removed;
    return removed;
}

// Code per dimension is (sign << ex_bits) + ex, up to nine bits, scaled by
// f_rescale_ex.
void HierarchicalNSW::reconstruct_rotated(PID id, float* out) const {
    thread_local std::vector<uint16_t> combined;
    thread_local std::vector<uint8_t> ex_codes;
    combined.resize(padded_dim_);

    const ConstBinDataMap<float> bin(get_bindata_by_internalid(id), padded_dim_);
    const uint8_t* sign_code = bin.bin_code();
    for (size_t word = 0; word * 64 < padded_dim_; ++word) {
        uint64_t bits = 0;
        const size_t base = word * 64;
        const size_t span = std::min<size_t>(64, padded_dim_ - base);
        std::memcpy(&bits, sign_code + (word * sizeof(uint64_t)), span / 8);
        for (size_t j = 0; j < span; ++j) {
            combined[base + j] = static_cast<uint16_t>((bits >> (span - 1 - j)) & 1ULL);
        }
    }

    const auto* centroid = reinterpret_cast<const float*>(centroids_memory_.get()) +
                           (get_clusterid_by_internalid(id) * padded_dim_);

    if (ex_bits_ == 0) {
        quant::reconstruct_full_vec<float, uint16_t>(
            combined.data(), centroid, padded_dim_, 1, bin.f_rescale(), out, metric_type_
        );
        return;
    }

    ex_codes.resize(padded_dim_);
    const ConstExDataMap<float> ex(get_exdata_by_internalid(id), padded_dim_, ex_bits_);
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
        out,
        metric_type_
    );
}

// Only `query` is reconstructed; `target` is scored from its codes, as QGBuilder
// does. Callers hold `query` fixed across a loop, so it is prepared once.
float HierarchicalNSW::get_quant_dist(
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
                           (get_clusterid_by_internalid(target) * padded_dim_);
    const float dist_sqr = euclidean_sqr(cached.rotated.data(), centroid, padded_dim_);
    const float g_error = std::sqrt(dist_sqr);
    const float g_add = (metric_type_ == METRIC_IP)
                            ? -dot_product(cached.rotated.data(), centroid, padded_dim_)
                            : dist_sqr;

    if (ex_bits_ == 0) {
        return estimate_split_single(
                   get_bindata_by_internalid(target),
                   *cached.prepared,
                   padded_dim_,
                   g_add,
                   g_error
        )
            .distance;
    }
    return estimate_split_single_full(
               get_bindata_by_internalid(target),
               get_exdata_by_internalid(target),
               ip_func_,
               *cached.prepared,
               padded_dim_,
               ex_bits_,
               g_add,
               g_error
    )
        .distance;
}

maxheap<std::pair<float, PID>> HierarchicalNSW::search_base_layer_quant(
    PID entry_point_id, PID new_node_id, int layer, detail::QuantQuery& query
) {
    auto visited = visited_list_pool_->get_free_vislist();

    maxheap<std::pair<float, PID>> top_candidates;
    minheap<std::pair<float, PID>> candidate_set;

    float lower_bound = get_quant_dist(entry_point_id, new_node_id, query);
    top_candidates.emplace(lower_bound, entry_point_id);
    candidate_set.emplace(lower_bound, entry_point_id);
    visited->set(entry_point_id);

    while (!candidate_set.empty()) {
        std::pair<float, PID> curr_el_pair = candidate_set.top();
        if (curr_el_pair.first > lower_bound && top_candidates.size() == ef_construction_) {
            break;
        }
        candidate_set.pop();

        PID current_id = curr_el_pair.second;
        std::unique_lock<std::mutex> lock(link_list_locks_[current_id]);

        const PID* data =
            (layer == 0) ? get_linklist0(current_id) : get_linklist(current_id, layer);
        const size_t size = get_list_count(data);
        const PID* datal = data + 1;

        for (size_t j = 0; j < size; j++) {
            PID candidate_id = datal[j];
            if (candidate_id >= num_stored_points_) {
                throw std::runtime_error(
                    "HNSW add layer search: invalid candidate ID " +
                    std::to_string(candidate_id) + " at level " + std::to_string(layer) +
                    " (stored_points=" + std::to_string(num_stored_points_.load()) + ")"
                );
            }
            if (visited->get(candidate_id)) {
                continue;
            }
            visited->set(candidate_id);

            float candidate_distance = get_quant_dist(candidate_id, new_node_id, query);
            if (top_candidates.size() < ef_construction_ ||
                lower_bound > candidate_distance) {
                candidate_set.emplace(candidate_distance, candidate_id);
                top_candidates.emplace(candidate_distance, candidate_id);
                if (top_candidates.size() > ef_construction_) {
                    top_candidates.pop();
                }
                if (!top_candidates.empty()) {
                    lower_bound = top_candidates.top().first;
                }
            }
        }
    }
    visited_list_pool_->release_vis_list(std::move(visited));
    return top_candidates;
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
            return get_quant_dist(target, source, query);
        },
        // Quantized scoring is directional: reconstruct the existing neighbor.
        [this, new_node_id, &query](PID neighbor) {
            return get_quant_dist(new_node_id, neighbor, query);
        }
    );
}

// The slot, the level and the link list are all decided by the caller, so this
// can run on many threads at once and the labels do not depend on their order.
void HierarchicalNSW::add_point_quant(
    PID new_node_id, int node_level, LinkStorage& link_list, detail::QuantQuery& query
) {
    std::unique_lock<std::mutex> node_lock(link_list_locks_[new_node_id]);

    std::unique_lock<std::mutex> entry_point_lock(global_);
    int previous_max_level = max_level_;
    PID current_id = entry_point_id_;
    if (node_level <= previous_max_level) {
        entry_point_lock.unlock();
    }
    const PID previous_entry_point = current_id;

    link_lists_[new_node_id] = std::move(link_list);
    element_levels_[new_node_id] = node_level;

    try {
        if (static_cast<signed>(current_id) != -1) {
            if (node_level < previous_max_level) {
                float current_distance = get_quant_dist(current_id, new_node_id, query);
                for (int level = previous_max_level; level > node_level; level--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        std::unique_lock<std::mutex> lock(link_list_locks_[current_id]);
                        const PID* data = get_linklist(current_id, level);
                        const size_t size = get_list_count(data);
                        const PID* datal = data + 1;
                        for (size_t i = 0; i < size; i++) {
                            PID cand = datal[i];
                            if (cand >= num_stored_points_) {
                                throw std::runtime_error(
                                    "HNSW add upper-layer traversal: invalid candidate "
                                    "ID " +
                                    std::to_string(cand) + " at level " +
                                    std::to_string(level) + " (stored_points=" +
                                    std::to_string(num_stored_points_.load()) + ")"
                                );
                            }
                            float d = get_quant_dist(cand, new_node_id, query);
                            if (d < current_distance) {
                                current_distance = d;
                                current_id = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            for (int level = std::min(node_level, previous_max_level); level >= 0;
                 level--) {
                maxheap<std::pair<float, PID>> top_candidates =
                    search_base_layer_quant(current_id, new_node_id, level, query);
                current_id =
                    mutually_connect_quant(new_node_id, top_candidates, level, query);
            }
        } else {
            entry_point_id_ = new_node_id;
            max_level_ = node_level;
        }

        if (node_level > previous_max_level) {
            entry_point_id_ = new_node_id;
            max_level_ = node_level;
        }
    } catch (...) {
        // Upper layers may already expose this node. Keep a route to the old
        // graph if base-layer linking was interrupted before choosing neighbors.
        PID* base_links = get_linklist0(new_node_id);
        if (previous_entry_point != kPidMax && get_list_count(base_links) == 0) {
            base_links[1] = previous_entry_point;
            set_list_count(base_links, 1);
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
    if (new_max_elements < num_stored_points_) {
        throw std::invalid_argument("HNSW resize: capacity below the element count");
    }
    if (new_max_elements > buffer::kSearchBufferMaxPointCount) {
        throw std::invalid_argument("HNSW resize: capacity exceeds the supported ID range");
    }
    if (new_max_elements == max_elements_) {
        return;
    }
    if (new_max_elements > std::numeric_limits<size_t>::max() / level0_record_bytes_) {
        throw std::invalid_argument("HNSW resize: capacity exceeds addressable storage");
    }

    // Everything new is built before anything old is released, so a failure
    // leaves the index as it was. realloc is not an option: the level-0 block
    // comes from aligned_alloc, which realloc does not keep aligned.
    const size_t level0_bytes = new_max_elements * level0_record_bytes_;
    AlignedStorage level0(memory::huge_page_allocate<char>(level0_bytes));
    if (level0 == nullptr) {
        throw std::bad_alloc();
    }
    std::memset(level0.get(), 0, level0_bytes);
    std::memcpy(
        level0.get(), data_level0_memory_.get(), num_stored_points_ * level0_record_bytes_
    );

    std::vector<LinkStorage> links;
    links.reserve(new_max_elements);

    std::vector<int> levels(element_levels_);
    levels.resize(new_max_elements);
    std::vector<std::mutex> locks(new_max_elements);
    auto pool = std::make_unique<VisitedListPool>(1, new_max_elements);

    // All potentially throwing work is complete. Moving the live owners into
    // reserved slots cannot allocate or throw; unused capacity stays uninitialized.
    for (auto& link : link_lists_) {
        links.push_back(std::move(link));
    }
    data_level0_memory_ = std::move(level0);
    link_lists_ = std::move(links);
    element_levels_.swap(levels);
    link_list_locks_.swap(locks);
    visited_list_pool_ = std::move(pool);
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
        throw std::logic_error("HNSW index must be constructed or loaded before add");
    }
    if (num_points == 0) {
        return {};
    }
    if (data == nullptr) {
        throw std::invalid_argument("HNSW add inputs must not be null");
    }
    if (num_points > max_elements_ - num_stored_points_) {
        throw std::invalid_argument("HNSW add: not enough capacity");
    }
    if (num_points > buffer::kSearchBufferMaxPointCount - num_stored_points_) {
        throw std::invalid_argument("HNSW add: point count exceeds the supported ID range");
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
        rabitqlib::index_detail::parallel_for(
            0, num_points, num_threads, [&](size_t i, size_t) {
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
            }
        );
    }

    quant::RabitqConfig config;
    if (fast_quantization) {
        config = quant::faster_config(padded_dim_, ex_bits_ + 1);
    }

    // Drawn here because the generator is shared: the inserts below run in
    // parallel and must not touch it.
    std::vector<int> levels(num_points);
    for (size_t i = 0; i < num_points; ++i) {
        levels[i] = get_random_level(mult_);
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
                throw std::runtime_error(
                    "Not enough memory: HNSW add failed to allocate a link list"
                );
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
            throw std::runtime_error("The number of elements exceeds the specified limit");
        }
        const auto first = static_cast<PID>(num_stored_points_.load());
        rabitqlib::index_detail::parallel_for(
            0, num_points, num_threads, [&](size_t i, size_t) {
                const auto id = static_cast<PID>(first + i);
                labels[i] = id;
                std::memset(
                    data_level0_memory_.get() + (id * level0_record_bytes_),
                    0,
                    level0_record_bytes_
                );
                std::memcpy(get_external_label_pt(id), &id, sizeof(PID));
                std::memcpy(get_clusterid_pt(id), &assigned[i], sizeof(PID));
                std::vector<float> rotated_data(padded_dim_);
                rotator_->rotate(data + (i * dim_), rotated_data.data());
                quant::quantize_split_single(
                    rotated_data.data(),
                    reinterpret_cast<float*>(centroids_memory_.get()) +
                        (assigned[i] * padded_dim_),
                    padded_dim_,
                    ex_bits_,
                    get_bindata_by_internalid(id),
                    get_exdata_by_internalid(id),
                    metric_type_,
                    config
                );
            }
        );
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

    rabitqlib::index_detail::parallel_for(
        0, num_points, num_workers, [&](size_t i, size_t worker) {
            add_point_quant(labels[i], levels[i], link_lists[i], queries[worker]);
        }
    );

    return labels;
}

}  // namespace rabitqlib::hnsw
