// HNSW is developed from the [HNSW library](https://github.com/nmslib/hnswlib)
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rabitqlib/core/metric.hpp"  // IWYU pragma: export
#include "rabitqlib/core/types.hpp"   // IWYU pragma: export
#include "rabitqlib/index/detail/hnsw_dispatch.hpp"
#include "rabitqlib/index/types.hpp"  // IWYU pragma: export

namespace rabitqlib {
template <typename T>
class Rotator;
template <typename T>
class SplitSingleQuery;
class VisitedListPool;
namespace detail {
struct DistanceEstimate;
}
}  // namespace rabitqlib

namespace rabitqlib::hnsw {
namespace detail {
struct QuantQuery;
}

template <typename T>
using maxheap = std::priority_queue<T>;

template <typename T>
using minheap = std::priority_queue<T, std::vector<T>, std::greater<T>>;

class HierarchicalNSW {
   public:
    explicit HierarchicalNSW();
    // dim is the original input dimension; total_bits includes the sign bit.
    // random_seed controls graph levels and (through its low 32 bits) rotation.
    // Parallel insertion order can still vary with thread scheduling.
    explicit HierarchicalNSW(
        size_t max_elements,
        size_t dim,
        size_t total_bits,
        size_t M,
        size_t ef_construction,
        size_t random_seed = 100,
        MetricType metric_type = METRIC_L2
    );
    ~HierarchicalNSW();

    [[nodiscard]] size_t dimension() const { return dim_; }
    [[nodiscard]] size_t num_clusters() const { return num_clusters_; }
    [[nodiscard]] size_t nbits() const { return ex_bits_ + 1; }
    [[nodiscard]] size_t M() const { return M_; }
    [[nodiscard]] size_t ef_construction() const { return ef_construction_; }
    [[nodiscard]] MetricType metric_type() const { return metric_type_; }
    // Logical capacity, nonremoved points, and stored records (including removals).
    [[nodiscard]] size_t max_elements() const { return max_elements_; }
    [[nodiscard]] size_t num_points() const {
        return num_stored_points_.load() - num_removed_;
    }
    [[nodiscard]] size_t num_stored_points() const { return num_stored_points_; }
    [[nodiscard]] PID cluster_id_of(PID internal_id) const {
        return get_clusterid_by_internalid(internal_id);
    }

    void save(const char* filename) const;
    void load(const char* filename);

    // Centroids and data use the original dimension; cluster_ids has num_points entries.
    // Input buffers must remain valid until this synchronous call returns or throws;
    // the index retains no pointers to them afterward.
    // fast_quantization selects fixed-scale extra-bit quantization instead of exact scale
    // search, trading quantization accuracy for speed. One-bit storage is unaffected.
    void construct(
        size_t num_clusters,
        const float* centroids,
        size_t num_points,
        const float* data,
        PID* cluster_ids,
        size_t num_threads = std::numeric_limits<size_t>::max(),
        bool fast_quantization = false
    );

    /**
     * @brief Insert points into a constructed or loaded index.
     *
     * Linked with the stored quantized codes, not the exact vectors construct uses,
     * which are not kept. Centroids and rotation are never retrained. Point `i` gets
     * the label `num_stored_points() + i` whatever `num_threads` is. Throws rather than
     * growing past `max_elements()`. Not safe to call while another thread searches.
     * Preparation failures leave the graph and point count unchanged. A failure
     * during linking may leave a partially linked batch counted in the index,
     * which remains safe to search, save, load, and destroy.
     *
     * @param data New points, `num_points * dim` floats in the original coordinates
     * @param num_points Number of new points
     * @param cluster_ids Cluster of each point in `[0, num_clusters)`, or nullptr to
     * route each one to its nearest centroid
     * @param fast_quantization Use fixed-scale extra-bit quantization for speed at a
     * possible loss of accuracy; has no effect on one-bit storage
     * @param num_threads Threads used to route and insert. Labels never depend on it,
     * but links do with more than one thread, so the default of one keeps the graph
     * deterministic.
     * @return The label given to each new point
     */
    std::vector<PID> add(
        const float* data,
        size_t num_points,
        const PID* cluster_ids = nullptr,
        bool fast_quantization = false,
        size_t num_threads = 1
    );

    /**
     * @brief Change capacity, allowing growth or shrinkage down to num_stored_points().
     *
     * An unchanged capacity is a no-op. Otherwise reallocates the base layer and
     * copies stored records, requiring room for both allocations and invalidating
     * pointers into the base layer. Work includes initializing the new capacity.
     * The index is unchanged if this throws.
     * Requires initialization through the parameterized constructor or `load`.
     *
     * @param new_max_elements The new capacity, at least `num_stored_points()`
     */
    void resize(size_t new_max_elements);

    /**
     * @brief Exclude points from all later search results.
     *
     * A removed point keeps its codes and its place in the graph: search and `add`
     * still traverse it and may link new points to it, but search never returns it,
     * so a query may get fewer than k results. A removed point still counts in
     * `num_stored_points()` and occupies capacity, but no longer counts in num_points().
     * Removal is stored in the index, so it survives save and load,
     * and is idempotent. Removed points cannot be restored. Every label is checked
     * before anything changes. Not safe to call while another thread searches or adds.
     *
     * @param labels Labels of the points to remove, as returned by `add` or `search`
     * @param num_points Number of labels
     * @return Number of points newly removed
     */
    size_t remove(const PID* labels, size_t num_points);

    // Queries use the original dimension. Each result is (distance, external label),
    // where distance estimates squared L2 or 1 - dot according to the index metric.
    std::vector<std::vector<std::pair<float, PID>>> search(
        const float* queries,
        size_t num_queries,
        size_t k,
        size_t ef_search,
        size_t num_threads
    );

   private:
    // Borrows construct() input only while its workers run; cleared on success or failure.
    const float* construction_data_{nullptr};

    // Bounded priority queue implemented as a sorted vector.
    struct Candidate {
        float est_dist;
        PID id;
    };

    class BoundedKNN {
       public:
        explicit BoundedKNN(size_t capacity) : capacity_(capacity) {}

        // Insert a candidate in sorted order (ascending by est_dist).
        void insert(const Candidate& cand) {
            auto it = std::upper_bound(
                queue_.begin(),
                queue_.end(),
                cand,
                [](const Candidate& a, const Candidate& b) {
                    return a.est_dist < b.est_dist;
                }
            );
            queue_.insert(it, cand);
            // If we exceed capacity, drop the worst candidate (largest est_dist).
            if (queue_.size() > capacity_) {
                queue_.pop_back();
            }
        }

        // Returns the worst (largest est_dist) candidate.
        [[nodiscard]] const Candidate& worst() const { return queue_.back(); }

        [[nodiscard]] size_t size() const { return queue_.size(); }

        [[nodiscard]] const std::vector<Candidate>& candidates() const { return queue_; }

       private:
        size_t capacity_;
        // Sorted in ascending order by est_dist so that the worst is at the back.
        std::vector<Candidate> queue_;
    };

    friend struct HnswPruningTestAccess;
    friend std::vector<std::pair<float, PID>> detail::search_knn_sorted_neon(
        HierarchicalNSW&, const float*, size_t, size_t
    );
    friend std::vector<std::pair<float, PID>> detail::search_knn_sorted_generic(
        HierarchicalNSW&, const float*, size_t, size_t
    );
    friend std::vector<std::pair<float, PID>> detail::search_knn_sorted_avx2(
        HierarchicalNSW&, const float*, size_t, size_t
    );
    friend std::vector<std::pair<float, PID>> detail::search_knn_sorted_avx512_core(
        HierarchicalNSW&, const float*, size_t, size_t
    );
    friend std::vector<std::pair<float, PID>> detail::search_knn_sorted_avx512_popcnt(
        HierarchicalNSW&, const float*, size_t, size_t
    );

    // `remove` sets the high bit of a point's stored cluster ID. construct and load
    // keep num_clusters_ at or below 2^31, so a real cluster ID never has it.
    static constexpr PID kRemovedMask = 0x80000000U;
    // Used when a search passes ef_search == 0.
    static constexpr size_t kDefaultEfSearch = 10;
    size_t max_elements_{0};
    mutable std::atomic<size_t> num_stored_points_{0};
    size_t num_removed_{0};  // Recovered from stored cluster-ID markers when loading.
    size_t level0_record_bytes_{0};      // One complete level-0 record.
    size_t upper_layer_links_bytes_{0};  // One node's adjacency list at one upper level.
    size_t M_{0};
    size_t upper_layer_max_degree_{0};
    size_t base_layer_max_degree_{0};
    size_t ef_construction_{0};
    MetricType metric_type_{METRIC_L2};

    double mult_{0.0};
    int max_level_{0};

    std::mutex global_;
    std::vector<std::mutex> link_list_locks_;

    PID entry_point_id_{0};

    size_t level0_links_bytes_{
        0};  // One node's level-0 adjacency list, including its header.

    size_t bin_data_offset_{0}, ex_data_offset_{0}, label_offset_{0};
    size_t bin_data_bytes_{0}, ex_data_bytes_{0};
    size_t ex_bits_{0};

    struct AlignedDeleter {
        void operator()(char* pointer) const noexcept;
    };
    struct LinkDeleter {
        void operator()(char* pointer) const noexcept;
    };
    using AlignedStorage = std::unique_ptr<char, AlignedDeleter>;
    using LinkStorage = std::unique_ptr<char, LinkDeleter>;

    // Layout: (# of edges + edges) + (cluster_id) + (External_id) + (BinData) + (ExData)
    AlignedStorage data_level0_memory_;
    // Reserve capacity without initializing unused slots; size tracks live nodes.
    std::vector<LinkStorage> link_lists_;
    std::vector<int> element_levels_;

    size_t num_clusters_{0};
    size_t dim_{0};
    size_t padded_dim_{0};

    AlignedStorage centroids_memory_;

    mutable std::mutex label_lookup_lock_;
    std::unordered_map<PID, PID> label_lookup_;

    std::default_random_engine level_generator_;

    std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};

    float (*ip_func_)(const float*, const uint8_t*, size_t){nullptr};

    std::unique_ptr<Rotator<float>> rotator_;

    quant::RabitqConfig query_config_;

    using EstimateRecord = rabitqlib::detail::DistanceEstimate;

    float (*raw_dist_func_)(const float* __restrict__, const float* __restrict__, size_t){
        nullptr};

    PID get_external_label(PID internal_id) const {
        PID return_label;
        memcpy(
            &return_label,
            (data_level0_memory_.get() + (internal_id * level0_record_bytes_) +
             label_offset_),
            sizeof(PID)
        );
        return return_label;
    }

    void set_external_label(PID internal_id, PID label) {
        memcpy(
            (data_level0_memory_.get() + (internal_id * level0_record_bytes_) +
             label_offset_),
            &label,
            sizeof(PID)
        );
    }

    char* get_bindata_by_internalid(PID internal_id) {
        return data_level0_memory_.get() + (internal_id * level0_record_bytes_) +
               bin_data_offset_;
    }

    const char* get_bindata_by_internalid(PID internal_id) const {
        return data_level0_memory_.get() + (internal_id * level0_record_bytes_) +
               bin_data_offset_;
    }

    char* get_exdata_by_internalid(PID internal_id) {
        return data_level0_memory_.get() + (internal_id * level0_record_bytes_) +
               ex_data_offset_;
    }

    const char* get_exdata_by_internalid(PID internal_id) const {
        return data_level0_memory_.get() + (internal_id * level0_record_bytes_) +
               ex_data_offset_;
    }

    // The stored cluster ID with the removal mark, as save and load see it.
    PID get_raw_clusterid(PID internal_id) const {
        return *(reinterpret_cast<const PID*>(
            data_level0_memory_.get() + (internal_id * level0_record_bytes_) +
            level0_links_bytes_
        ));
    }

    PID get_clusterid_by_internalid(PID internal_id) const {
        return get_raw_clusterid(internal_id) & ~kRemovedMask;
    }

    bool is_removed(PID internal_id) const {
        return (get_raw_clusterid(internal_id) & kRemovedMask) != 0;
    }

    // Store a cluster ID, including any removal mark.
    void set_raw_clusterid(PID internal_id, PID cluster_id) {
        memcpy(
            data_level0_memory_.get() + (internal_id * level0_record_bytes_) +
                level0_links_bytes_,
            &cluster_id,
            sizeof(PID)
        );
    }

    // Derive record sizes and offsets from padded_dim_, ex_bits_ and the layer degrees.
    void set_record_layout();

    // The fields set_record_layout derives, in a fixed order for comparison.
    [[nodiscard]] std::array<size_t, 8> record_layout() const {
        return {
            bin_data_bytes_,
            ex_data_bytes_,
            level0_links_bytes_,
            label_offset_,
            bin_data_offset_,
            ex_data_offset_,
            level0_record_bytes_,
            upper_layer_links_bytes_};
    }

    int get_random_level(double reverse_size);

    // Write the index file at exactly this path.
    void write_file(const char* filename) const;

    PID* get_linklist(PID internal_id, int level) {
        return reinterpret_cast<PID*>(
            link_lists_[internal_id].get() + ((level - 1) * upper_layer_links_bytes_)
        );
    }

    const PID* get_linklist(PID internal_id, int level) const {
        return reinterpret_cast<const PID*>(
            link_lists_[internal_id].get() + ((level - 1) * upper_layer_links_bytes_)
        );
    }

    PID* get_linklist0(PID internal_id) {
        return reinterpret_cast<PID*>(
            data_level0_memory_.get() + (internal_id * level0_record_bytes_)
        );
    }

    const PID* get_linklist0(PID internal_id) const {
        return reinterpret_cast<const PID*>(
            data_level0_memory_.get() + (internal_id * level0_record_bytes_)
        );
    }

    static unsigned short int get_list_count(const PID* ptr) {
        return *(reinterpret_cast<const unsigned short int*>(ptr));
    }

    static void set_list_count(PID* ptr, unsigned short int size) {
        *(reinterpret_cast<unsigned short int*>(ptr)) = size;
    }

    // ANN Search
    template <class Kernel>
    void get_bin_est_direct(
        const std::vector<float>&,
        const SplitSingleQuery<float>&,
        PID,
        HierarchicalNSW::EstimateRecord&
    ) const;

    template <class Kernel>
    void get_full_est_direct(
        const std::vector<float>&,
        const SplitSingleQuery<float>&,
        PID,
        HierarchicalNSW::EstimateRecord&
    ) const;

    template <class Kernel>
    std::vector<std::pair<float, PID>> search_knn_sorted_direct(
        const float*, size_t, size_t
    ) const;

    template <class Kernel>
    void search_base_layer_direct(
        PID entry_point_id,
        size_t ef_search,
        size_t k,
        const SplitSingleQuery<float>& query_wrapper,
        const std::vector<float>& q_to_centroids,
        BoundedKNN& nearest_candidates
    ) const;

    // Construction
    // Currently only support index construction with non-quantized vectors
    float get_data_dist(PID obj1, PID obj2) const {
        PID label1 = get_external_label(obj1);
        PID label2 = get_external_label(obj2);
        return raw_dist_func_(
            construction_data_ + (label1 * dim_), construction_data_ + (label2 * dim_), dim_
        );
    }

    void add_point(PID, PID, const quant::RabitqConfig&);

    // Insertion steps shared by raw and quantized scoring. distance(id) scores an
    // existing node against the node being inserted; prefetch(id) may warm its data.
    template <class Distance>
    PID greedy_descend(PID, int from_level, int to_level, Distance);
    template <class Distance, class Prefetch>
    maxheap<std::pair<float, PID>> search_layer(PID, int, Distance, Prefetch);

    PID mutually_connect_new_element(PID, maxheap<std::pair<float, PID>>&, int);

    template <class Distance, class ReverseEdgeDistance>
    PID mutually_connect(
        PID, maxheap<std::pair<float, PID>>&, int, Distance, ReverseEdgeDistance
    );

    // The same steps once the original data is gone, implemented in hnsw.cpp.
    void reconstruct_rotated(PID, float*) const;
    float get_quant_dist(PID target, PID query, detail::QuantQuery&) const;
    void add_point_quant(PID, int, LinkStorage&, detail::QuantQuery&);
    PID mutually_connect_quant(
        PID, maxheap<std::pair<float, PID>>&, int, detail::QuantQuery&
    );
};

struct EstimateRecord {
    float est_dist;
    float low_dist;
};

}  // namespace rabitqlib::hnsw
