// HNSW is developed from the [HNSW library](https://github.com/nmslib/hnswlib)
#pragma once

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
class VisitedSetPool;
namespace detail {
struct DistanceEstimate;
}
namespace rabitqkmeans {
struct RaBitQKMeansParameters;
}
}  // namespace rabitqlib

namespace rabitqlib::hnsw {
namespace detail {
struct QuantQuery;
struct HnswSearchAccess;
}  // namespace detail

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
    // Centroids are set, by train, set_centroids or construct, or the index was loaded
    // with clusters.
    [[nodiscard]] bool is_trained() const { return num_clusters_ != 0; }
    // Cluster of a stored record by internal ID (0 through num_stored_points() - 1;
    // these coincide with labels only for points added in order). Removed points
    // keep their cluster.
    [[nodiscard]] PID cluster_id_of(PID internal_id) const {
        if (data_level0_memory_ == nullptr) {
            throw std::logic_error(
                "HNSW index must be constructed or loaded before cluster lookup"
            );
        }
        if (internal_id >= num_stored_points_.load()) {
            throw std::invalid_argument("HNSW internal ID is out of range");
        }
        return stored_cluster_id(internal_id);
    }

    // Requires at least one stored point; a trained index must add points first.
    void save(const char* filename) const;
    void load(const char* filename);

    /**
     * @brief Replace the centroids of an initialized index without stored points.
     *
     * The index is unchanged if this throws. Later adds route and quantize against
     * these centroids; the first add into the empty graph links like construct.
     *
     * @param num_clusters Number of centroids, from 1 through 2^31
     * @param centroids Centroid vectors (num_clusters * dim) in the original coordinates
     */
    void set_centroids(size_t num_clusters, const float* centroids);

    /**
     * @brief Train centroids with RaBitQKMeans and set them as set_centroids does.
     *
     * Only centroids are trained: points are not assigned until add. The index is
     * unchanged if this throws.
     *
     * @param num_clusters Number of centroids, at most num_points
     * @param num_points Number of training points
     * @param data Training points (num_points * dim)
     * @param cp Clustering parameters, for example default_training_parameters()
     */
    void train(
        size_t num_clusters,
        size_t num_points,
        const float* data,
        const rabitqkmeans::RaBitQKMeansParameters& cp
    );

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
     * @brief Insert points into a trained, constructed or loaded index.
     *
     * Into an empty graph, points are linked with their exact vectors as construct
     * links them, and a failure leaves the graph empty. Otherwise they are linked with
     * the stored quantized codes, since the exact vectors are not kept. Centroids and
     * rotation are never retrained. Point `i` gets
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
    // ef_search is raised to k and clamped to the stored point count. An empty index,
    // trained or not, returns no results.
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

    friend struct HnswTestAccess;
    friend struct detail::HnswSearchAccess;

    // `remove` sets the high bit of a point's stored cluster ID. construct and load
    // keep num_clusters_ at or below 2^31, so a real cluster ID never has it.
    static constexpr PID kRemovedMask = 0x80000000U;
    // Used when a search passes ef_search == 0.
    static constexpr size_t kDefaultEfSearch = 10;
    // The constructor clamps M to this value and load rejects files above it.
    static constexpr size_t kMaxM = 10000;
    size_t max_elements_{0};
    std::atomic<size_t> num_stored_points_{0};
    size_t num_removed_{0};  // Recovered from stored cluster-ID markers when loading.
    size_t level0_record_bytes_{0};      // One complete level-0 record.
    size_t upper_layer_links_bytes_{0};  // One node's adjacency list at one upper level.
    size_t M_{0};
    size_t upper_layer_max_degree_{0};
    size_t base_layer_max_degree_{0};
    size_t ef_construction_{0};
    MetricType metric_type_{METRIC_L2};

    double level_multiplier_{0.0};
    // max_level_ == -1 and entry_point_id_ == kPidMax mark an empty graph.
    int max_level_{-1};

    // Serializes entry-point and maximum-level updates during insertion.
    std::mutex entry_point_mutex_;
    std::vector<std::mutex> link_list_locks_;

    PID entry_point_id_{kPidMax};

    // One node's level-0 adjacency list, including its header.
    size_t level0_links_bytes_{0};

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
    // Reserve capacity without initializing unused slots; size tracks stored nodes,
    // including removed ones.
    std::vector<LinkStorage> link_lists_;
    std::vector<int> element_levels_;

    size_t num_clusters_{0};
    size_t dim_{0};
    size_t padded_dim_{0};

    AlignedStorage centroids_memory_;

    std::mutex label_lookup_lock_;
    std::unordered_map<PID, PID> label_lookup_;

    std::default_random_engine level_generator_;

    std::unique_ptr<VisitedSetPool> visited_set_pool_{nullptr};

    float (*ip_func_)(const float*, const uint8_t*, size_t){nullptr};

    std::unique_ptr<Rotator<float>> rotator_;

    quant::RabitqConfig query_config_;

    using DistanceEstimate = rabitqlib::detail::DistanceEstimate;

    float (*raw_dist_func_)(const float* __restrict__, const float* __restrict__, size_t){
        nullptr};

    [[nodiscard]] PID external_label(PID internal_id) const {
        PID label;
        std::memcpy(
            &label,
            (data_level0_memory_.get() + (internal_id * level0_record_bytes_) +
             label_offset_),
            sizeof(PID)
        );
        return label;
    }

    void set_external_label(PID internal_id, PID label) {
        std::memcpy(
            (data_level0_memory_.get() + (internal_id * level0_record_bytes_) +
             label_offset_),
            &label,
            sizeof(PID)
        );
    }

    [[nodiscard]] char* bin_data(PID internal_id) {
        return data_level0_memory_.get() + (internal_id * level0_record_bytes_) +
               bin_data_offset_;
    }

    [[nodiscard]] const char* bin_data(PID internal_id) const {
        return data_level0_memory_.get() + (internal_id * level0_record_bytes_) +
               bin_data_offset_;
    }

    [[nodiscard]] char* ex_data(PID internal_id) {
        return data_level0_memory_.get() + (internal_id * level0_record_bytes_) +
               ex_data_offset_;
    }

    [[nodiscard]] const char* ex_data(PID internal_id) const {
        return data_level0_memory_.get() + (internal_id * level0_record_bytes_) +
               ex_data_offset_;
    }

    // The stored cluster ID with the removal mark, as save and load see it.
    [[nodiscard]] PID stored_cluster_id(PID internal_id) const {
        return raw_cluster_id(internal_id) & ~kRemovedMask;
    }
    [[nodiscard]] PID raw_cluster_id(PID internal_id) const {
        return *(reinterpret_cast<const PID*>(
            data_level0_memory_.get() + (internal_id * level0_record_bytes_) +
            level0_links_bytes_
        ));
    }

    [[nodiscard]] bool is_removed(PID internal_id) const {
        return (raw_cluster_id(internal_id) & kRemovedMask) != 0;
    }

    // Store a cluster ID, including any removal mark.
    void set_raw_cluster_id(PID internal_id, PID cluster_id) {
        std::memcpy(
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

    int random_level(double multiplier);

    // Write the index file at exactly this path.
    void write_file(const char* filename) const;

    // A link list is a count header followed by neighbor IDs.
    [[nodiscard]] PID* upper_layer_links(PID internal_id, int level) {
        return reinterpret_cast<PID*>(
            link_lists_[internal_id].get() + ((level - 1) * upper_layer_links_bytes_)
        );
    }

    [[nodiscard]] const PID* upper_layer_links(PID internal_id, int level) const {
        return reinterpret_cast<const PID*>(
            link_lists_[internal_id].get() + ((level - 1) * upper_layer_links_bytes_)
        );
    }

    [[nodiscard]] PID* level0_links(PID internal_id) {
        return reinterpret_cast<PID*>(
            data_level0_memory_.get() + (internal_id * level0_record_bytes_)
        );
    }

    [[nodiscard]] const PID* level0_links(PID internal_id) const {
        return reinterpret_cast<const PID*>(
            data_level0_memory_.get() + (internal_id * level0_record_bytes_)
        );
    }

    [[nodiscard]] static uint16_t link_count(const PID* links) {
        return *(reinterpret_cast<const uint16_t*>(links));
    }

    static void set_link_count(PID* links, uint16_t count) {
        *(reinterpret_cast<uint16_t*>(links)) = count;
    }

    // ANN Search. search_knn_sorted_with prepares the query in generic code and runs
    // one ISA's search_layers_direct (instantiated in src/index/hnsw_search_*.cpp).
    std::vector<std::pair<float, PID>> search_knn_sorted_with(
        detail::SearchLayersFn search_layers,
        const float* rotated_query,
        size_t k,
        size_t ef_search
    ) const;

    template <class Kernel>
    void estimate_bin_direct(
        const float* q_to_centroids,
        const SplitSingleQuery<float>& query_wrapper,
        PID candidate_id,
        DistanceEstimate& estimate
    ) const;

    template <class Kernel>
    void estimate_full_direct(
        const float* q_to_centroids,
        const SplitSingleQuery<float>& query_wrapper,
        PID candidate_id,
        DistanceEstimate& estimate
    ) const;

    template <class Kernel>
    size_t search_layers_direct(
        const SplitSingleQuery<float>& query_wrapper,
        const float* q_to_centroids,
        VisitedSet& visited,
        buffer::SearchBuffer<float>& search_pool,
        size_t k,
        AnnCandidate<float>* nearest
    ) const;

    template <class Kernel, class Results>
    void search_base_layer_direct(
        PID entry_point_id,
        size_t k,
        const SplitSingleQuery<float>& query_wrapper,
        const float* q_to_centroids,
        VisitedSet& visited,
        buffer::SearchBuffer<float>& search_pool,
        Results& nearest_candidates
    ) const;

    // Raw-vector distance used by construct; add scores links with quantized_distance.
    [[nodiscard]] float raw_distance(PID target, PID query) const {
        PID target_label = external_label(target);
        PID query_label = external_label(query);
        return raw_dist_func_(
            construction_data_ + (target_label * dim_),
            construction_data_ + (query_label * dim_),
            dim_
        );
    }

    void add_point(PID label, PID cluster_id, const quant::RabitqConfig& config);

    // Link validated points into an empty graph that has centroids, scoring with their
    // exact vectors. Labels are 0 through num_points - 1. A failure empties the graph.
    void link_exact(
        size_t num_points,
        const float* data,
        const PID* cluster_ids,
        size_t num_threads,
        bool fast_quantization
    );

    // Insertion steps shared by raw and quantized scoring. distance(id) scores an
    // existing node against the node being inserted; prefetch(id) may warm its data.
    template <class Distance>
    PID greedy_descend(PID current_id, int from_level, int to_level, Distance distance);
    template <class Distance, class Prefetch>
    maxheap<std::pair<float, PID>> search_layer(
        PID entry_point_id, int level, Distance distance, Prefetch prefetch
    );

    PID mutually_connect_new_element(
        PID new_node_id, maxheap<std::pair<float, PID>>& top_candidates, int level
    );

    template <class Distance, class ReverseEdgeDistance>
    PID mutually_connect(
        PID new_node_id,
        maxheap<std::pair<float, PID>>& top_candidates,
        int level,
        Distance distance,
        ReverseEdgeDistance reverse_edge_distance
    );

    // The same steps once the original data is gone, implemented in hnsw.cpp.
    void reconstruct_rotated(PID internal_id, float* rotated) const;
    float quantized_distance(PID target, PID query, detail::QuantQuery& cached) const;
    void add_point_quant(
        PID new_node_id, int node_level, LinkStorage& link_list, detail::QuantQuery& query
    );
    PID mutually_connect_quant(
        PID new_node_id,
        maxheap<std::pair<float, PID>>& top_candidates,
        int level,
        detail::QuantQuery& query
    );
};

struct EstimateRecord {
    float est_dist;
    float low_dist;
};

}  // namespace rabitqlib::hnsw
