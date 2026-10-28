#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include "rabitqlib/core/metric.hpp"  // IWYU pragma: export
#include "rabitqlib/core/types.hpp"   // IWYU pragma: export
#include "rabitqlib/index/ivf/cluster.hpp"
#include "rabitqlib/index/ivf/types.hpp"  // IWYU pragma: export
#include "rabitqlib/index/types.hpp"      // IWYU pragma: export
#include "rabitqlib/utils/memory.hpp"

namespace rabitqlib {
template <typename T>
class Rotator;
template <typename T>
class SplitBatchQuery;
namespace buffer {
template <typename T>
class SearchBuffer;
}
}  // namespace rabitqlib

namespace rabitqlib::ivf {
class Initializer;

class IVF {
   private:
    friend struct IvfTestAccess;

    using ByteStorage =
        std::vector<std::byte, memory::AlignedAllocator<std::byte, 64, true>>;
    using FloatStorage = std::vector<float, memory::AlignedAllocator<float, 64, true>>;
    using IdStorage = std::vector<PID, memory::AlignedAllocator<PID, 64, true>>;

    // One cluster's own storage, independent of every other cluster's.
    class ClusterStorage {
       public:
        explicit ClusterStorage(bool raw_reranking) : raw_reranking_(raw_reranking) {}

        ByteStorage batch_storage;  // 1-bit code and factors
        ByteStorage ex_storage;     // extra-bit codes and packed factors
        FloatStorage raw_storage;   // original vectors for raw reranking
        IdStorage id_storage;       // point IDs of this cluster

        [[nodiscard]] char* batch_data() {
            return reinterpret_cast<char*>(batch_storage.data());
        }

        [[nodiscard]] const char* batch_data() const {
            return reinterpret_cast<const char*>(batch_storage.data());
        }

        [[nodiscard]] char* ex_data() {
            return raw_reranking_ ? reinterpret_cast<char*>(raw_storage.data())
                                  : reinterpret_cast<char*>(ex_storage.data());
        }

        [[nodiscard]] const char* ex_data() const {
            return raw_reranking_ ? reinterpret_cast<const char*>(raw_storage.data())
                                  : reinterpret_cast<const char*>(ex_storage.data());
        }

        [[nodiscard]] PID* ids() { return id_storage.data(); }

        [[nodiscard]] const PID* ids() const { return id_storage.data(); }

       private:
        bool raw_reranking_;  // raw vectors replace extra-bit codes
    };

    InitializerType initializer_type_ = InitializerType::Auto;
    std::unique_ptr<Initializer> initializer_;     // initializer for candidate clusters
    std::vector<ClusterStorage> cluster_storage_;  // one entry per cluster
    size_t max_elements_ = 0;  // Logical capacity; construct fills it, add grows it.
    size_t num_removed_ = 0;   // Recovered from removal markers when loading.
    size_t dim_ = 0;
    size_t padded_dim_ = 0;
    size_t num_clusters_ = 0;
    bool raw_reranking_ = false;  // raw vectors replace extra-bit codes
    bool ready_ = false;          // construction or loading completed
    size_t ex_bits_ = 0;          // total bits = ex_bits_ + 1
    RotatorType rotator_type_ = RotatorType::FhtKacRotator;
    std::unique_ptr<Rotator<float>> rotator_;
    std::vector<Cluster> clusters_;
    MetricType metric_type_ = rabitqlib::METRIC_L2;
    float (*ip_func_)(const float*, const uint8_t*, size_t) = nullptr;

    static constexpr uint64_t kRawFormatMagic = 0x3157415251464252ULL;
    static constexpr uint32_t kRawFormatVersion = 1;
    static constexpr uint64_t kFormatMagic = 0x3158444951424152ULL;  // "RABQIDX1"
    static constexpr uint32_t kFormatVersion = 2;

    void quantize_cluster(
        Cluster&,
        const std::vector<PID>&,
        const float*,
        const float*,
        float*,
        float*,
        const quant::RabitqConfig&
    );

    void extend_cluster(
        const Cluster&,
        const std::vector<PID>&,
        const float*,
        const float*,
        PID,
        char*,
        char*,
        PID*,
        char*,
        const quant::RabitqConfig&
    ) const;

    // True if `num_points` points cannot be addressed by the storage size computations
    [[nodiscard]] bool exceeds_storage(size_t num_points) const;

    [[nodiscard]] static size_t ids_bytes(size_t cluster_size) {
        return sizeof(PID) * cluster_size;
    }

    // get num of bytes used for 1-bit code and corresponding factors
    [[nodiscard]] size_t batch_data_bytes(size_t cluster_size) const;

    [[nodiscard]] inline size_t rerank_vector_bytes() const;

    [[nodiscard]] inline size_t ex_data_bytes(size_t cluster_size) const;

    // Size every buffer of a cluster's storage for `num_points` points
    void resize_storage(ClusterStorage& cs, size_t num_points) const;

    void reserve_storage(ClusterStorage& cs, size_t num_points) const;

    [[nodiscard]] bool has_capacity(const ClusterStorage& cs, size_t num_points) const;

    void allocate_memory(const std::vector<size_t>&);

    void init_clusters(const std::vector<size_t>&);

    void free_memory();

    void swap(IVF& other) noexcept;

    inline void search_cluster(
        const Cluster&,
        const SplitBatchQuery<float>&,
        buffer::SearchBuffer<float>&,
        bool,
        const float*
    ) const;

    inline void scan_one_batch(
        const char* batch_data,
        const char* ex_data,
        const PID* ids,
        const SplitBatchQuery<float>& query_wrapper,
        buffer::SearchBuffer<float>& result_pool,
        size_t num_points,
        bool,
        const float*
    ) const;

    inline void validate_search(const float*, size_t, size_t, const PID*) const;

   public:
    explicit IVF() noexcept;
    // dim is the original input dimension; total_bits is 1-9 for quantized
    // storage (including the sign bit), or 32 for raw-vector reranking.
    explicit IVF(
        size_t num_points,
        size_t dim,
        size_t num_clusters,
        size_t total_bits,
        MetricType metric_type = rabitqlib::METRIC_L2,
        RotatorType rotator_type = RotatorType::FhtKacRotator,
        InitializerType initializer = InitializerType::Auto
    );

    ~IVF();

    // Logical capacity, including slots occupied by removed points. add grows it.
    [[nodiscard]] size_t max_elements() const { return max_elements_; }
    // Unbuilt indexes have no stored or searchable points.
    [[nodiscard]] size_t num_points() const {
        return ready_ ? max_elements_ - num_removed_ : 0;
    }
    [[nodiscard]] size_t num_stored_points() const { return ready_ ? max_elements_ : 0; }
    [[nodiscard]] size_t dimension() const { return dim_; }
    [[nodiscard]] size_t nbits() const { return raw_reranking_ ? 32 : ex_bits_ + 1; }
    [[nodiscard]] MetricType metric_type() const { return metric_type_; }
    [[nodiscard]] RotatorType rotator_type() const { return rotator_type_; }
    [[nodiscard]] InitializerType initializer_type() const { return initializer_type_; }

    // fast_quantization selects fixed-scale extra-bit quantization instead of exact scale
    // search, trading quantization accuracy for speed. One-bit and raw storage are
    // unaffected.
    void construct(
        const float* data,
        const float* centroids,
        const PID* cluster_ids,
        bool fast_quantization = false,
        size_t num_threads = std::numeric_limits<size_t>::max()
    );

    /**
     * @brief Append points to a constructed or loaded index without the original data.
     *
     * Points are quantized against the centroids and rotator the index already holds, so
     * the centroids never move: recall can degrade if the added data drifts away from the
     * data the centroids were trained on. Point `i` of `data` receives PID
     * `num_stored_points() + i`. The index is unchanged if this throws.
     *
     * Only clusters that receive points are touched. Each one is rebuilt next to its old
     * storage, so a call needs extra memory for, and takes time proportional to, those
     * clusters plus `num_points`; prefer a few large calls to many small ones. A cluster
     * with enough capacity from `reserve` grows in place instead. `construct` expects
     * `max_elements()` rows afterwards.
     * Not safe to call while another thread searches, adds or removes.
     *
     * @param data New points (num_points * dim)
     * @param num_points Number of new points
     * @param cluster_ids Cluster of each new point, or nullptr to route each point to its
     * nearest centroid the same way a query is routed
     * @param fast_quantization Use fixed-scale extra-bit quantization for speed at a
     * possible loss of accuracy; has no effect on one-bit or raw storage
     * @param num_threads Threads used for routing and quantization
     */
    void add(
        const float* data,
        size_t num_points,
        const PID* cluster_ids = nullptr,
        bool fast_quantization = false,
        size_t num_threads = std::numeric_limits<size_t>::max()
    );

    /**
     * @brief Preallocate room for a cluster to hold `num_points` points.
     *
     * Later `add` calls grow the cluster in place, without copying it, until it holds more
     * than `num_points`. Does nothing if the cluster already has that capacity. Reserved
     * capacity is not saved: a loaded index has none. The index is unchanged if this
     * throws. Not safe to call while another thread searches, adds or removes.
     *
     * @param cluster_id Cluster to grow
     * @param num_points Total number of points the cluster can hold, including its
     * current ones
     */
    void reserve(PID cluster_id, size_t num_points);

    /**
     * @brief Exclude points from all later search results.
     *
     * A removed point stays in num_stored_points() and occupies capacity, but no longer
     * counts in num_points(). Search may
     * return fewer than k results (unfilled slots use kPidMax). Removal is stored in the
     * index, so it survives save and load, and is idempotent. Removed points cannot be
     * restored. Not safe to call while another thread searches or adds.
     *
     * @param ids_to_remove PIDs to remove
     * @param num_points Number of PIDs
     * @return Number of points newly removed
     */
    size_t remove(const PID* ids_to_remove, size_t num_points);

    void save(const char* filename) const;

    void load(const char* filename);

    // Automatically use HACC for 4-9 quantized bits; use standard FastScan otherwise.
    // Distances estimate squared L2 or 1 - dot according to the index metric.
    void search(
        const float* query,
        size_t k,
        size_t nprobe,
        PID* result_ids,
        float* distances = nullptr
    ) const;

    void search(
        const float* query, size_t k, size_t nprobe, PID* result_ids, bool use_hacc
    ) const;

    void search(
        const float* query,
        size_t k,
        size_t nprobe,
        PID* result_ids,
        float* distances,
        bool use_hacc
    ) const;

    /**
     * Search contiguous row-major queries using at most num_threads workers.
     * queries contains num_queries * dimension() floats. result_ids and optional
     * distances each hold num_queries * k elements. Empty batches do no work.
     * use_hacc defaults to the same automatic policy as search().
     * Input and output buffers must not overlap. Concurrent calls are safe if
     * the index is not being modified and their output buffers are separate.
     */
    void search_batch(
        const float* queries,
        size_t num_queries,
        size_t k,
        size_t nprobe,
        PID* result_ids,
        float* distances = nullptr,
        std::optional<bool> use_hacc = std::nullopt,
        size_t num_threads = 1
    ) const;

    [[nodiscard]] size_t padded_dim() const { return this->padded_dim_; }

    [[nodiscard]] size_t num_clusters() const { return this->num_clusters_; }
};

}  // namespace rabitqlib::ivf
