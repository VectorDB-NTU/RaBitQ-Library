#pragma once

#include <omp.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <ios>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/fastscan/fastscan.hpp"
#include "rabitqlib/index/estimator.hpp"
#include "rabitqlib/index/ivf/cluster.hpp"
#include "rabitqlib/index/ivf/initializer.hpp"
#include "rabitqlib/index/query.hpp"
#include "rabitqlib/quantization/data_layout.hpp"
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/utils/buffer.hpp"
#include "rabitqlib/utils/memory.hpp"
#include "rabitqlib/utils/path.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/utils/space.hpp"
#include "rabitqlib/utils/tools.hpp"

namespace rabitqlib::ivf {
namespace detail {
void insert_candidates(buffer::SearchBuffer<float>&, const PID*, const float*, size_t);
}  // namespace detail

class IVF {
   private:
    using ByteStorage =
        std::vector<std::byte, memory::AlignedAllocator<std::byte, 64, true>>;
    using FloatStorage = std::vector<float, memory::AlignedAllocator<float, 64, true>>;
    using IdStorage = std::vector<PID, memory::AlignedAllocator<PID, 64, true>>;

    InitializerType initializer_type_ = InitializerType::Auto;
    std::unique_ptr<Initializer> initer_;            // initializer for candidate clusters
    ByteStorage batch_storage_;                      // 1-bit code and factors
    ByteStorage ex_storage_;                         // extra-bit codes and packed factors
    FloatStorage raw_storage_;                       // original vectors for raw reranking
    IdStorage id_storage_;                           // point IDs organized by cluster
    size_t num_ = 0;                                 // num of data points
    size_t dim_ = 0;                                 // dimension of data points
    size_t padded_dim_ = 0;                          // dimension after padding
    size_t num_cluster_ = 0;                         // num of centroids (clusters)
    bool raw_reranking_ = false;                     // raw vectors replace extra-bit codes
    bool ready_ = false;                             // construction or loading completed
    size_t ex_bits_ = 0;                             // total bits = ex_bits_ + 1
    RotatorType type_ = RotatorType::FhtKacRotator;  // type of rotator
    std::unique_ptr<Rotator<float>> rotator_;        // Data Rotator
    std::vector<Cluster> cluster_lst_;               // List of clusters in ivf
    MetricType metric_type_ = rabitqlib::METRIC_L2;  // metric type
    float (*ip_func_)(const float*, const uint8_t*, size_t) = nullptr;

    static constexpr uint64_t kRawFormatMagic = 0x3157415251464252ULL;
    static constexpr uint32_t kRawFormatVersion = 1;
    static constexpr uint64_t kFormatMagic = 0x3158444951424152ULL;  // "RABQIDX1"
    static constexpr uint32_t kFormatVersion = 2;

    void
    quantize_cluster(Cluster&, const std::vector<PID>&, const float*, const float*, float*, float*, const quant::RabitqConfig&);

    void
    extend_cluster(const Cluster&, const std::vector<PID>&, const float*, const float*, PID, char*, char*, PID*, const quant::RabitqConfig&)
        const;

    // True if `n` points cannot be addressed by the storage size computations
    [[nodiscard]] bool exceeds_storage(size_t n) const {
        const size_t max_size = std::numeric_limits<size_t>::max();
        const size_t batch_bytes = BatchDataMap<float>::data_bytes(padded_dim_);
        const size_t rerank_bytes = rerank_vector_bytes();
        return n > max_size / sizeof(PID) || n > max_size / sizeof(float) / padded_dim_ ||
               n > max_size / batch_bytes ||
               (rerank_bytes != 0 && n > max_size / rerank_bytes);
    }

    [[nodiscard]] size_t ids_bytes() const { return sizeof(PID) * num_; }

    // get num of bytes used for 1-bit code and corresponding factors
    [[nodiscard]] size_t batch_data_bytes(const std::vector<size_t>& cluster_sizes) const {
        assert(cluster_sizes.size() == num_cluster_);  // num of clusters
        size_t total_blocks = 0;
        for (auto size : cluster_sizes) {
            total_blocks += div_round_up(size, fastscan::kBatchSize);
        }
        return total_blocks * BatchDataMap<float>::data_bytes(padded_dim_);
    }

    [[nodiscard]] size_t rerank_vector_bytes() const {
        return raw_reranking_ ? sizeof(float) * dim_
                              : ExDataMap<float>::data_bytes(padded_dim_, ex_bits_);
    }

    [[nodiscard]] size_t ex_data_bytes() const { return rerank_vector_bytes() * num_; }

    [[nodiscard]] char* batch_data() {
        return reinterpret_cast<char*>(batch_storage_.data());
    }

    [[nodiscard]] const char* batch_data() const {
        return reinterpret_cast<const char*>(batch_storage_.data());
    }

    [[nodiscard]] char* ex_data() {
        return raw_reranking_ ? reinterpret_cast<char*>(raw_storage_.data())
                              : reinterpret_cast<char*>(ex_storage_.data());
    }

    [[nodiscard]] const char* ex_data() const {
        return raw_reranking_ ? reinterpret_cast<const char*>(raw_storage_.data())
                              : reinterpret_cast<const char*>(ex_storage_.data());
    }

    [[nodiscard]] PID* ids() { return id_storage_.data(); }

    [[nodiscard]] const PID* ids() const { return id_storage_.data(); }

    void allocate_memory(const std::vector<size_t>&);

    void init_clusters(const std::vector<size_t>&);

    void free_memory() {
        initer_.reset();
        ByteStorage().swap(batch_storage_);
        ByteStorage().swap(ex_storage_);
        FloatStorage().swap(raw_storage_);
        IdStorage().swap(id_storage_);
    }

    void swap(IVF& other) noexcept {
        using std::swap;
        swap(initializer_type_, other.initializer_type_);
        swap(initer_, other.initer_);
        swap(batch_storage_, other.batch_storage_);
        swap(ex_storage_, other.ex_storage_);
        swap(raw_storage_, other.raw_storage_);
        swap(id_storage_, other.id_storage_);
        swap(num_, other.num_);
        swap(dim_, other.dim_);
        swap(padded_dim_, other.padded_dim_);
        swap(num_cluster_, other.num_cluster_);
        swap(raw_reranking_, other.raw_reranking_);
        swap(ready_, other.ready_);
        swap(ex_bits_, other.ex_bits_);
        swap(type_, other.type_);
        swap(rotator_, other.rotator_);
        swap(cluster_lst_, other.cluster_lst_);
        swap(metric_type_, other.metric_type_);
        swap(ip_func_, other.ip_func_);
    }

    void
    search_cluster(const Cluster&, const SplitBatchQuery<float>&, buffer::SearchBuffer<float>&, bool, const float*)
        const;

    void
    scan_one_batch(const char* batch_data, const char* ex_data, const PID* ids, const SplitBatchQuery<float>& q_obj, buffer::SearchBuffer<float>& knns, size_t num_points, bool, const float*)
        const;

    void validate_search(const float*, size_t, size_t, const PID*) const;

   public:
    explicit IVF() = default;
    explicit IVF(
        size_t,
        size_t,
        size_t,
        size_t,
        MetricType metric_type = rabitqlib::METRIC_L2,
        RotatorType type = RotatorType::FhtKacRotator,
        InitializerType initializer = InitializerType::Auto
    );

    ~IVF();

    // Number of stored points, including any removed with `remove`. `add` increases it.
    [[nodiscard]] size_t max_elements() const { return num_; }
    [[nodiscard]] size_t dimension() const { return dim_; }
    [[nodiscard]] size_t nbits() const { return raw_reranking_ ? 32 : ex_bits_ + 1; }
    [[nodiscard]] MetricType metric_type() const { return metric_type_; }
    [[nodiscard]] RotatorType rotator_type() const { return type_; }
    [[nodiscard]] InitializerType initializer_type() const { return initializer_type_; }

    void construct(const float*, const float*, const PID*, bool, size_t);

    /**
     * @brief Append points to a constructed or loaded index without the original data.
     *
     * Points are quantized against the centroids and rotator the index already holds, so
     * the centroids never move: recall can degrade if the added data drifts away from the
     * data the centroids were trained on. Point `i` of `data` receives PID
     * `max_elements() + i`. The index is unchanged if this throws.
     *
     * The grown index is built next to the old one, so a call needs memory for both and
     * takes time proportional to the whole index, not just to `n`: prefer a few large
     * calls to many small ones. `construct` expects `max_elements()` rows afterwards.
     * Not safe to call while another thread searches, adds or removes.
     *
     * @param data New points (n * dim)
     * @param n Number of new points
     * @param cluster_ids Cluster of each new point, or nullptr to route each point to its
     * nearest centroid the same way a query is routed
     * @param faster Same meaning as in construct
     * @param num_threads Threads used for routing and quantization
     */
    void add(
        const float* data,
        size_t n,
        const PID* cluster_ids = nullptr,
        bool faster = false,
        size_t num_threads = std::numeric_limits<size_t>::max()
    );

    /**
     * @brief Exclude points from all later search results.
     *
     * A removed point keeps its storage, still counts in `max_elements()`, and search may
     * return fewer than k results (unfilled slots use kPidMax). Removal is stored in the
     * index, so it survives save and load, and is idempotent. Removed points cannot be
     * restored. Not safe to call while another thread searches or adds.
     *
     * @param ids_to_remove PIDs to remove
     * @param n Number of PIDs
     * @return Number of points newly removed
     */
    size_t remove(const PID* ids_to_remove, size_t n);

    void save(const char*) const;

    void load(const char*);

    // Automatically use HACC for 4-9 quantized bits; use standard FastScan otherwise.
    void search(const float*, size_t, size_t, PID*, float* = nullptr) const;

    void search(const float*, size_t, size_t, PID*, bool) const;

    void search(const float*, size_t, size_t, PID*, float*, bool) const;

    /**
     * Search contiguous row-major queries using at most num_threads workers.
     * queries contains num_queries * dimension() floats. results and optional
     * dists each hold num_queries * k elements. Empty batches do no work.
     * use_hacc defaults to the same automatic policy as search().
     * Input and output buffers must not overlap. Concurrent calls are safe if
     * the index is not being modified and their output buffers are separate.
     */
    void search_batch(
        const float* queries,
        size_t num_queries,
        size_t k,
        size_t nprobe,
        PID* results,
        float* dists = nullptr,
        std::optional<bool> use_hacc = std::nullopt,
        size_t num_threads = 1
    ) const;

    [[nodiscard]] size_t padded_dim() const { return this->padded_dim_; }

    [[nodiscard]] size_t num_clusters() const { return this->num_cluster_; }
};

inline IVF::IVF(
    size_t n,
    size_t dim,
    size_t cluster_num,
    size_t bits,
    MetricType metric_type,
    RotatorType type,
    InitializerType initializer
)
    : initializer_type_(resolve_initializer_type(initializer, cluster_num))
    , num_(n)
    , dim_(dim)
    , padded_dim_(dim)
    , num_cluster_(cluster_num)
    , raw_reranking_(bits == 32)
    , ex_bits_(bits == 32 ? 0 : bits - 1)
    , type_(type)
    , metric_type_(metric_type) {
    validate_metric_type(metric_type);
    if ((bits < 1 || bits > 9) && bits != 32) {
        throw std::invalid_argument("IVF bits must be in [1, 9] or 32 for raw reranking");
    };
    if (n == 0 || n > buffer::kSearchBufferMaxPointCount) {
        throw std::invalid_argument("IVF point count exceeds the supported ID range");
    }
    if (cluster_num == 0 || cluster_num > buffer::kSearchBufferMaxPointCount) {
        throw std::invalid_argument("IVF cluster count exceeds the supported ID range");
    }
    if (dim == 0 || dim > (std::numeric_limits<size_t>::max() / 32) - 64) {
        throw std::invalid_argument("IVF dimension is invalid or too large");
    }
    padded_dim_ = round_up_to_multiple(dim_, 32);
    const size_t max_size = std::numeric_limits<size_t>::max();
    if (exceeds_storage(n) || cluster_num > max_size / sizeof(float) / padded_dim_ ||
        (type == RotatorType::MatrixRotator && dim > max_size / sizeof(float) / padded_dim_
        )) {
        throw std::invalid_argument("IVF configuration exceeds addressable storage");
    }
    rotator_.reset(choose_rotator<float>(dim, type, padded_dim_));
    /* check size */
    assert(padded_dim_ % 32 == 0);
    assert(padded_dim_ >= dim_);
}

inline IVF::~IVF() { free_memory(); }

/**
 * @brief Construct clusters in IVF
 *
 * @param data Data objects (N*DIM)
 * @param centroids Centroid vectors (K*DIM)
 * @param cluster_ids Cluster ID for each data object
 */
inline void IVF::construct(
    const float* data,
    const float* centroids,
    const PID* cluster_ids,
    bool faster = false,
    size_t num_threads = std::numeric_limits<size_t>::max()
) {
    if (num_ == 0 || dim_ == 0 || num_cluster_ == 0 || rotator_ == nullptr) {
        throw std::logic_error("IVF must be configured before construction");
    }
    if (data == nullptr || centroids == nullptr || cluster_ids == nullptr) {
        throw std::invalid_argument("IVF construction inputs must not be null");
    }

    // get id list for each cluster
    std::vector<size_t> counts(num_cluster_, 0);
    std::vector<std::vector<PID>> id_lists(num_cluster_);
    for (size_t i = 0; i < num_; ++i) {
        PID cid = cluster_ids[i];
        if (cid >= num_cluster_) {
            throw std::invalid_argument("Cluster ID is out of range");
        }
        id_lists[cid].push_back(static_cast<PID>(i));
        counts[cid] += 1;
    }

    allocate_memory(counts);

    // init the cluster list
    init_clusters(counts);

    // all rotated centroids
    std::vector<float> rotated_centroids(num_cluster_ * padded_dim_);

    quant::RabitqConfig config;
    if (faster) {
        config = quant::faster_config(padded_dim_, ex_bits_ + 1);
    }

    num_threads = std::min(resolve_num_threads(num_threads), num_cluster_);
    const size_t scratch_points =
        std::min(fastscan::kBatchSize, *std::max_element(counts.begin(), counts.end()));
    // Each worker reuses one batch, independently of the largest cluster size.
    // Allocate before entering OpenMP so allocation failures reach the caller.
    std::vector<std::vector<float>> rotated_blocks(num_threads);
    for (auto& block : rotated_blocks) {
        block.resize(scratch_points * padded_dim_);
    }
    /* Quantize each cluster */
#pragma omp parallel for schedule(dynamic) num_threads(num_threads)
    for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(num_cluster_);
         ++index) {
        const size_t i = static_cast<size_t>(index);
        const float* cur_centroid = centroids + (i * dim_);
        float* cur_rotated_c = &rotated_centroids[i * padded_dim_];
        Cluster& cp = cluster_lst_[i];
        quantize_cluster(
            cp,
            id_lists[i],
            data,
            cur_centroid,
            cur_rotated_c,
            rotated_blocks[static_cast<size_t>(omp_get_thread_num())].data(),
            config
        );
    }

    this->initer_->add_vectors(rotated_centroids.data(), num_threads);
    ready_ = true;
}

inline void IVF::allocate_memory(const std::vector<size_t>& cluster_sizes) {
    ready_ = false;
    free_memory();
    cluster_lst_.clear();

    if (initializer_type_ == InitializerType::Flat) {
        this->initer_ =
            std::make_unique<FlatInitializer>(padded_dim_, num_cluster_, metric_type_);
    } else if (initializer_type_ == InitializerType::FlatRaBitQ) {
        this->initer_ = std::make_unique<FlatRaBitQInitializer>(
            padded_dim_, num_cluster_, metric_type_
        );
    } else {
        this->initer_ =
            std::make_unique<HNSWInitializer>(padded_dim_, num_cluster_, metric_type_);
    }
    batch_storage_ = ByteStorage(batch_data_bytes(cluster_sizes));
    if (rerank_vector_bytes() > 0) {
        if (raw_reranking_) {
            raw_storage_ = FloatStorage(num_ * dim_);
        } else {
            ex_storage_ = ByteStorage(ex_data_bytes());
        }
    }
    id_storage_ = IdStorage(num_);

    this->ip_func_ = select_excode_ipfunc(ex_bits_);
}

/**
 * @brief Initialize the cluster list by finding each cluster's storage offsets.
 */
inline void IVF::init_clusters(const std::vector<size_t>& cluster_sizes) {
    this->cluster_lst_.reserve(num_cluster_);
    size_t added_vectors = 0;
    size_t added_batches = 0;
    for (size_t i = 0; i < num_cluster_; ++i) {
        // find data location for current cluster
        size_t num = cluster_sizes[i];
        size_t num_batches = div_round_up(num, fastscan::kBatchSize);

        char* current_batch_data =
            batch_data() + (BatchDataMap<float>::data_bytes(padded_dim_) * added_batches);
        char* current_ex_data = rerank_vector_bytes() > 0
                                    ? ex_data() + (added_vectors * rerank_vector_bytes())
                                    : nullptr;
        PID* cluster_ids = ids() + added_vectors;

        Cluster cur_cluster(num, current_batch_data, current_ex_data, cluster_ids);
        this->cluster_lst_.push_back(std::move(cur_cluster));

        added_vectors += num;
        added_batches += num_batches;
    }
}

inline void IVF::quantize_cluster(
    Cluster& cp,
    const std::vector<PID>& IDs,
    const float* data,
    const float* cur_centroid,
    float* rotated_centroid,
    float* rotated_data,
    const quant::RabitqConfig& config
) {
    size_t num_points = IDs.size();
    if (cp.num() != num_points) {
        throw std::invalid_argument("Cluster size and ID count differ");
    }

    // copy ids
    std::copy(IDs.begin(), IDs.end(), cp.ids());

    // rotate centroid
    this->rotator_->rotate(cur_centroid, rotated_centroid);

    char* batch_data = cp.batch_data();
    char* ex_data = cp.ex_data();
    for (size_t i = 0; i < num_points; i += fastscan::kBatchSize) {
        size_t n = std::min(fastscan::kBatchSize, num_points - i);
        for (size_t j = 0; j < n; ++j) {
            const float* vector = data + (IDs[i + j] * dim_);
            rotator_->rotate(vector, rotated_data + (j * padded_dim_));
            if (raw_reranking_) {
                std::memcpy(
                    cp.ex_data() + ((i + j) * rerank_vector_bytes()),
                    vector,
                    rerank_vector_bytes()
                );
            }
        }

        quant::quantize_split_batch(
            rotated_data,
            rotated_centroid,
            n,
            padded_dim_,
            ex_bits_,
            batch_data,
            ex_data,
            metric_type_,
            config
        );

        batch_data += BatchDataMap<float>::data_bytes(padded_dim_);
        if (ex_bits_ > 0) {
            ex_data += ExDataMap<float>::data_bytes(padded_dim_, ex_bits_) * n;
        }
    }
}

inline void IVF::add(
    const float* data, size_t n, const PID* cluster_ids, bool faster, size_t num_threads
) {
    if (!ready_ || initer_ == nullptr || rotator_ == nullptr ||
        cluster_lst_.size() != num_cluster_ || batch_storage_.empty() ||
        id_storage_.size() != num_) {
        throw std::logic_error("IVF index must be constructed or loaded before add");
    }
    if (n == 0) {
        return;
    }
    if (data == nullptr) {
        throw std::invalid_argument("IVF add data must not be null");
    }
    if (n > buffer::kSearchBufferMaxPointCount - num_) {
        throw std::invalid_argument("IVF point count exceeds the supported ID range");
    }
    const size_t new_num = num_ + n;
    if (exceeds_storage(new_num)) {
        throw std::invalid_argument("IVF configuration exceeds addressable storage");
    }

    /* Assign every new point to a cluster */
    std::vector<PID> assigned(n);
    if (cluster_ids != nullptr) {
        for (size_t i = 0; i < n; ++i) {
            if (cluster_ids[i] >= num_cluster_) {
                throw std::invalid_argument("Cluster ID is out of range");
            }
            assigned[i] = cluster_ids[i];
        }
    } else {
        parallel_for(0, n, num_threads, [&](size_t row, size_t /*thread_id*/) {
            std::vector<float> rotated(padded_dim_);
            std::vector<AnnCandidate<float>> nearest(1);
            rotator_->rotate(data + (row * dim_), rotated.data());
            initer_->centroids_distances(rotated.data(), 1, nearest);
            assigned[row] = nearest[0].id;
        });
    }

    std::vector<std::vector<PID>> rows_of(num_cluster_);
    for (size_t i = 0; i < n; ++i) {
        rows_of[assigned[i]].push_back(static_cast<PID>(i));
    }
    std::vector<size_t> new_sizes(num_cluster_);
    for (size_t i = 0; i < num_cluster_; ++i) {
        new_sizes[i] = cluster_lst_[i].num() + rows_of[i].size();
    }

    /* Build the grown storage on the side so the index is unchanged on failure */
    ByteStorage new_batch(batch_data_bytes(new_sizes));
    ByteStorage new_ex;
    FloatStorage new_raw;
    const size_t rerank_bytes = rerank_vector_bytes();
    if (raw_reranking_) {
        new_raw = FloatStorage(new_num * dim_);
    } else if (rerank_bytes > 0) {
        new_ex = ByteStorage(new_num * rerank_bytes);
    }
    IdStorage new_ids(new_num);
    char* new_rerank = raw_reranking_ ? reinterpret_cast<char*>(new_raw.data())
                                      : reinterpret_cast<char*>(new_ex.data());

    std::vector<size_t> batch_offset(num_cluster_);
    std::vector<size_t> vector_offset(num_cluster_);
    size_t added_batches = 0;
    size_t added_vectors = 0;
    for (size_t i = 0; i < num_cluster_; ++i) {
        batch_offset[i] = added_batches;
        vector_offset[i] = added_vectors;
        added_batches += div_round_up(new_sizes[i], fastscan::kBatchSize);
        added_vectors += new_sizes[i];
    }

    quant::RabitqConfig config;
    if (faster) {
        config = quant::faster_config(padded_dim_, ex_bits_ + 1);
    }

    const size_t batch_bytes = BatchDataMap<float>::data_bytes(padded_dim_);
    const PID first_pid = static_cast<PID>(num_);
    // An exception must not leave an OpenMP region, so keep the first one and rethrow it.
    std::exception_ptr failure;
#pragma omp parallel for schedule(dynamic) num_threads(resolve_num_threads(num_threads))
    for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(num_cluster_);
         ++index) {
        const size_t i = static_cast<size_t>(index);
        try {
            extend_cluster(
                cluster_lst_[i],
                rows_of[i],
                data,
                initer_->centroid(static_cast<PID>(i)),
                first_pid,
                reinterpret_cast<char*>(new_batch.data()) + (batch_offset[i] * batch_bytes),
                rerank_bytes > 0 ? new_rerank + (vector_offset[i] * rerank_bytes) : nullptr,
                new_ids.data() + vector_offset[i],
                config
            );
        } catch (...) {
#pragma omp critical(rabitq_ivf_add_failure)
            if (!failure) {
                failure = std::current_exception();
            }
        }
    }
    if (failure) {
        std::rethrow_exception(failure);
    }

    /* Commit. Rebuilding the cluster list reuses its capacity, so it cannot throw. */
    batch_storage_.swap(new_batch);
    ex_storage_.swap(new_ex);
    raw_storage_.swap(new_raw);
    id_storage_.swap(new_ids);
    num_ = new_num;
    cluster_lst_.clear();
    init_clusters(new_sizes);
}

/**
 * @brief Copy one cluster into its grown storage and append new points to it.
 *
 * Full batches, rerank data and IDs are copied unchanged. The old partial tail batch is
 * unpacked and repacked together with the new points, so the result is byte-identical to
 * quantizing the same points in the same order in one construct call.
 */
inline void IVF::extend_cluster(
    const Cluster& old_cluster,
    const std::vector<PID>& rows,
    const float* data,
    const float* rotated_centroid,
    PID first_pid,
    char* batch_dst,
    char* rerank_dst,
    PID* ids_dst,
    const quant::RabitqConfig& config
) const {
    const size_t old_num = old_cluster.num();
    const size_t batch_bytes = BatchDataMap<float>::data_bytes(padded_dim_);
    const size_t rerank_bytes = rerank_vector_bytes();
    const size_t cols = padded_dim_ / 8;

    if (rows.empty()) {
        const size_t old_batches = div_round_up(old_num, fastscan::kBatchSize);
        std::copy_n(old_cluster.batch_data(), old_batches * batch_bytes, batch_dst);
        std::copy_n(old_cluster.ex_data(), old_num * rerank_bytes, rerank_dst);
        std::copy_n(old_cluster.ids(), old_num, ids_dst);
        return;
    }

    const size_t full_batches = old_num / fastscan::kBatchSize;
    const size_t kept = old_num - (full_batches * fastscan::kBatchSize);
    const size_t tail_num = kept + rows.size();

    std::copy_n(old_cluster.batch_data(), full_batches * batch_bytes, batch_dst);
    std::copy_n(old_cluster.ex_data(), old_num * rerank_bytes, rerank_dst);
    std::copy_n(old_cluster.ids(), old_num, ids_dst);

    /* Codes and factors of every point in the tail, old points first */
    std::vector<uint8_t> codes(tail_num * cols);
    std::vector<float> f_add(tail_num);
    std::vector<float> f_rescale(tail_num);
    std::vector<float> f_error(tail_num);
    if (kept > 0) {
        ConstBatchDataMap<float> old_tail(
            old_cluster.batch_data() + (full_batches * batch_bytes), padded_dim_
        );
        fastscan::unpack_codes(padded_dim_, old_tail.bin_code(), kept, codes.data());
        old_tail.f_add().copy_to(f_add.data(), kept);
        old_tail.f_rescale().copy_to(f_rescale.data(), kept);
        old_tail.f_error().copy_to(f_error.data(), kept);
    }

    std::vector<float> rotated(padded_dim_);
    for (size_t j = 0; j < rows.size(); ++j) {
        const float* vector = data + (static_cast<size_t>(rows[j]) * dim_);
        const size_t lane = kept + j;
        rotator_->rotate(vector, rotated.data());
        quant::quantize_compact_one_bit(
            rotated.data(),
            rotated_centroid,
            padded_dim_,
            codes.data() + (lane * cols),
            f_add[lane],
            f_rescale[lane],
            f_error[lane],
            metric_type_
        );
        char* rerank = rerank_dst + ((old_num + j) * rerank_bytes);
        if (raw_reranking_) {
            std::memcpy(rerank, vector, rerank_bytes);
        } else if (ex_bits_ > 0) {
            quant::quantize_compact_ex_bits(
                rotated.data(),
                rotated_centroid,
                padded_dim_,
                ex_bits_,
                rerank,
                metric_type_,
                config
            );
        }
        ids_dst[old_num + j] = first_pid + rows[j];
    }

    /* Repack the tail into batches */
    char* batch = batch_dst + (full_batches * batch_bytes);
    for (size_t start = 0; start < tail_num; start += fastscan::kBatchSize) {
        const size_t count = std::min(fastscan::kBatchSize, tail_num - start);
        BatchDataMap<float> cur_batch(batch, padded_dim_);
        fastscan::pack_codes(
            padded_dim_, codes.data() + (start * cols), count, cur_batch.bin_code()
        );
        for (size_t i = 0; i < count; ++i) {
            cur_batch.f_add()[i] = f_add[start + i];
            cur_batch.f_rescale()[i] = f_rescale[start + i];
            cur_batch.f_error()[i] = f_error[start + i];
        }
        batch += batch_bytes;
    }
}

inline size_t IVF::remove(const PID* ids_to_remove, size_t n) {
    if (!ready_ || initer_ == nullptr || rotator_ == nullptr ||
        cluster_lst_.size() != num_cluster_ || batch_storage_.empty() ||
        id_storage_.size() != num_) {
        throw std::logic_error("IVF index must be constructed or loaded before remove");
    }
    if (n == 0) {
        return 0;
    }
    if (ids_to_remove == nullptr) {
        throw std::invalid_argument("IVF remove IDs must not be null");
    }

    // Validate every ID before changing anything
    std::vector<bool> marked(num_, false);
    for (size_t i = 0; i < n; ++i) {
        if (ids_to_remove[i] >= num_) {
            throw std::invalid_argument("IVF remove ID is out of range");
        }
        marked[ids_to_remove[i]] = true;
    }

    // A point whose f_add is +inf has +inf estimated and lower-bound distances in every
    // scan path, so it never enters the result buffer or triggers reranking.
    constexpr float kRemoved = std::numeric_limits<float>::infinity();
    const size_t batch_bytes = BatchDataMap<float>::data_bytes(padded_dim_);
    size_t removed = 0;
    for (const auto& cluster : cluster_lst_) {
        const PID* cluster_ids = cluster.ids();
        for (size_t pos = 0; pos < cluster.num(); ++pos) {
            if (!marked[cluster_ids[pos]]) {
                continue;
            }
            BatchDataMap<float> cur_batch(
                cluster.batch_data() + ((pos / fastscan::kBatchSize) * batch_bytes),
                padded_dim_
            );
            const size_t lane = pos % fastscan::kBatchSize;
            if (static_cast<float>(cur_batch.f_add()[lane]) != kRemoved) {
                cur_batch.f_add()[lane] = kRemoved;
                ++removed;
            }
        }
    }
    return removed;
}

inline void IVF::save(const char* filename) const {
    if (!ready_ || initer_ == nullptr || rotator_ == nullptr ||
        cluster_lst_.size() != num_cluster_ || batch_storage_.empty() ||
        id_storage_.size() != num_) {
        throw std::logic_error("Cannot save an unconstructed IVF index");
    }
    if (filename == nullptr || filename[0] == '\0') {
        throw std::invalid_argument("IVF save filename must not be empty");
    }

    std::ofstream output(rabitqlib::io_impl::filesystem_path(filename), std::ios::binary);
    output.exceptions(std::ios::failbit | std::ios::badbit);
    const uint32_t flags = raw_reranking_ ? 1 : 0;
    const uint64_t stored_padded_dim = padded_dim_;
    output.write(reinterpret_cast<const char*>(&kFormatMagic), sizeof(kFormatMagic));
    output.write(reinterpret_cast<const char*>(&kFormatVersion), sizeof(kFormatVersion));
    output.write(reinterpret_cast<const char*>(&flags), sizeof(flags));
    output.write(
        reinterpret_cast<const char*>(&stored_padded_dim), sizeof(stored_padded_dim)
    );

    const auto stored_initializer = static_cast<uint32_t>(initializer_type_);
    output.write(
        reinterpret_cast<const char*>(&stored_initializer), sizeof(stored_initializer)
    );

    /* Save meta data */
    output.write(reinterpret_cast<const char*>(&num_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&dim_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&num_cluster_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&ex_bits_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&type_), sizeof(type_));
    output.write(reinterpret_cast<const char*>(&metric_type_), sizeof(metric_type_));

    /* Save number of vectors of each cluster */
    std::vector<size_t> cluster_sizes;
    cluster_sizes.reserve(num_cluster_);
    for (const auto& cur_cluster : cluster_lst_) {
        cluster_sizes.push_back(cur_cluster.num());
    }
    output.write(
        reinterpret_cast<const char*>(cluster_sizes.data()),
        static_cast<std::streamsize>(sizeof(size_t) * num_cluster_)
    );

    /* Save rotator */
    this->rotator_->save(output);

    /* Save data */
    this->initer_->save(output, filename);
    output.write(
        batch_data(), static_cast<std::streamsize>(batch_data_bytes(cluster_sizes))
    );
    if (ex_data_bytes() != 0) {
        output.write(ex_data(), static_cast<std::streamsize>(ex_data_bytes()));
    }
    output.write(
        reinterpret_cast<const char*>(ids()), static_cast<std::streamsize>(ids_bytes())
    );

    output.close();
}

inline void IVF::load(const char* filename) {
    if (filename == nullptr || filename[0] == '\0') {
        throw std::invalid_argument("IVF load filename must not be empty");
    }
    std::ifstream input(rabitqlib::io_impl::filesystem_path(filename), std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("Cannot open IVF index file");
    }

    IVF loaded;
    input.exceptions(std::ios::failbit | std::ios::badbit);
    input.seekg(0, std::ios::end);
    const auto file_bytes = static_cast<size_t>(input.tellg());
    input.seekg(0);
    uint64_t magic = 0;
    input.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    const bool explicit_padding = magic == kFormatMagic;
    if (explicit_padding) {
        uint32_t version = 0, flags = 0;
        uint64_t stored_padded_dim = 0;
        input.read(reinterpret_cast<char*>(&version), sizeof(version));
        input.read(reinterpret_cast<char*>(&flags), sizeof(flags));
        input.read(reinterpret_cast<char*>(&stored_padded_dim), sizeof(stored_padded_dim));
        if (version != 1 && version != kFormatVersion) {
            throw std::runtime_error("Unsupported IVF index version");
        }
        if (flags > 1 || stored_padded_dim > std::numeric_limits<size_t>::max()) {
            throw std::runtime_error("Invalid IVF index metadata");
        }
        loaded.raw_reranking_ = flags == 1;
        loaded.padded_dim_ = static_cast<size_t>(stored_padded_dim);
        if (version >= 2) {
            uint32_t initializer = 0;
            input.read(reinterpret_cast<char*>(&initializer), sizeof(initializer));
            if (initializer < static_cast<uint32_t>(InitializerType::Flat) ||
                initializer > static_cast<uint32_t>(InitializerType::HNSW)) {
                throw std::runtime_error("Invalid IVF initializer type in index file");
            }
            loaded.initializer_type_ = static_cast<InitializerType>(initializer);
        }
    } else if (magic == kRawFormatMagic) {
        loaded.raw_reranking_ = true;
        uint32_t version = 0;
        input.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (version != kRawFormatVersion) {
            throw std::runtime_error("Unsupported raw IVF index version");
        }
    } else {
        input.seekg(0);
    }

    /* Load meta data */
    input.read(reinterpret_cast<char*>(&loaded.num_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&loaded.dim_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&loaded.num_cluster_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&loaded.ex_bits_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&loaded.type_), sizeof(loaded.type_));
    input.read(reinterpret_cast<char*>(&loaded.metric_type_), sizeof(loaded.metric_type_));
    validate_metric_type(loaded.metric_type_);
    // Missing routing metadata means a historical file. Its payload layout must
    // follow the original threshold, independently of the current Auto policy.
    if (loaded.initializer_type_ == InitializerType::Auto) {
        loaded.initializer_type_ =
            loaded.num_cluster_ < 20000 ? InitializerType::Flat : InitializerType::HNSW;
    }
    if (loaded.num_ == 0 || loaded.num_ > buffer::kSearchBufferMaxPointCount ||
        loaded.dim_ == 0 || loaded.num_cluster_ == 0 ||
        loaded.num_cluster_ > buffer::kSearchBufferMaxPointCount || loaded.ex_bits_ > 8 ||
        (loaded.raw_reranking_ && loaded.ex_bits_ != 0) ||
        loaded.dim_ > (std::numeric_limits<size_t>::max() / 32) - 64) {
        throw std::runtime_error("Invalid IVF index metadata");
    }
    if (!explicit_padding) {
        // Both historical formats omit padded_dim and always used 64-coordinate blocks.
        loaded.padded_dim_ = round_up_to_multiple(loaded.dim_, 64);
    }
    if (loaded.padded_dim_ < loaded.dim_ || loaded.padded_dim_ % 32 != 0 ||
        loaded.padded_dim_ > (std::numeric_limits<size_t>::max() / 32) - 64) {
        throw std::runtime_error("Invalid padded dimension in IVF index file");
    }
    // Bound every payload by the actual file before allocating or multiplying sizes.
    size_t remaining = file_bytes - static_cast<size_t>(input.tellg());
    const auto consume = [&remaining](size_t count, size_t bytes) {
        if (bytes != 0 && count > remaining / bytes) {
            throw std::runtime_error("Invalid or truncated IVF index payload");
        }
        remaining -= count * bytes;
    };
    consume(loaded.num_cluster_, sizeof(size_t));
    consume(loaded.num_, sizeof(PID));
    consume(loaded.num_, loaded.rerank_vector_bytes());
    if (loaded.type_ == RotatorType::MatrixRotator) {
        consume(loaded.dim_, sizeof(float) * loaded.padded_dim_);
    } else if (loaded.type_ == RotatorType::FhtKacRotator) {
        consume(1, loaded.padded_dim_ / 2);
    } else {
        throw std::runtime_error("Invalid IVF rotator type");
    }
    if (loaded.initializer_type_ != InitializerType::HNSW) {
        consume(loaded.num_cluster_, sizeof(float) * loaded.padded_dim_);
    }
    if (loaded.initializer_type_ == InitializerType::FlatRaBitQ) {
        consume(loaded.padded_dim_, sizeof(float));
        consume(
            div_round_up(loaded.num_cluster_, fastscan::kBatchSize),
            BatchDataMap<float>::data_bytes(loaded.padded_dim_)
        );
    }

    /* Load number of vectors of each cluster */
    std::vector<size_t> cluster_sizes(loaded.num_cluster_, 0);
    input.read(
        reinterpret_cast<char*>(cluster_sizes.data()),
        static_cast<std::streamsize>(sizeof(size_t) * loaded.num_cluster_)
    );

    size_t total = 0;
    for (size_t size : cluster_sizes) {
        if (size > loaded.num_ - total) {
            throw std::runtime_error("Invalid cluster counts in IVF index file");
        }
        total += size;
        consume(
            div_round_up(size, fastscan::kBatchSize),
            BatchDataMap<float>::data_bytes(loaded.padded_dim_)
        );
    }
    if (total != loaded.num_) {
        throw std::runtime_error("Invalid cluster counts in IVF index file");
    }
    loaded.rotator_.reset(
        choose_rotator<float>(loaded.dim_, loaded.type_, loaded.padded_dim_)
    );

    /* Load rotator */
    loaded.rotator_->load(input);

    /* Load data */
    loaded.allocate_memory(cluster_sizes);
    loaded.initer_->load(input, filename);
    input.read(
        loaded.batch_data(),
        static_cast<std::streamsize>(loaded.batch_data_bytes(cluster_sizes))
    );
    if (loaded.ex_data_bytes() != 0) {
        input.read(loaded.ex_data(), static_cast<std::streamsize>(loaded.ex_data_bytes()));
    }
    input.read(
        reinterpret_cast<char*>(loaded.ids()),
        static_cast<std::streamsize>(loaded.ids_bytes())
    );
    for (size_t i = 0; i < loaded.num_; ++i) {
        if (loaded.ids()[i] >= loaded.num_ ||
            loaded.ids()[i] >= buffer::kSearchBufferCheckedMask) {
            throw std::runtime_error("Invalid point ID in IVF index file");
        }
    }

    /* Init each cluster */
    loaded.init_clusters(cluster_sizes);
    loaded.ready_ = true;

    input.close();
    swap(loaded);
}

inline void IVF::search(
    const float* query, size_t k, size_t nprobe, PID* results, float* dists
) const {
    search(query, k, nprobe, results, dists, !raw_reranking_ && ex_bits_ > 2);
}

inline void IVF::search(
    const float* __restrict__ query,
    size_t k,
    size_t nprobe,
    PID* __restrict__ results,
    bool use_hacc
) const {
    this->search(query, k, nprobe, results, nullptr, use_hacc);
}

inline void IVF::search(
    const float* __restrict__ query,
    size_t k,
    size_t nprobe,
    PID* __restrict__ results,
    float* __restrict__ dists,
    bool use_hacc
) const {
    validate_search(query, k, nprobe, results);

    nprobe = std::min(nprobe, num_cluster_);  // corner case
    std::vector<float> rotated_query(padded_dim_);
    this->rotator_->rotate(query, rotated_query.data());

    // use initer to get closest nprobe centroids
    std::vector<AnnCandidate<float>> centroid_dist(nprobe);
    this->initer_->centroids_distances(rotated_query.data(), nprobe, centroid_dist);

    buffer::SearchBuffer knns(k);

    SplitBatchQuery<float> q_obj(
        rotated_query.data(), padded_dim_, ex_bits_, metric_type_, use_hacc
    );

    for (size_t i = 0; i < nprobe; ++i) {
        PID cid = centroid_dist[i].id;
        float dist = centroid_dist[i].distance;
        const Cluster& cur_cluster = cluster_lst_[cid];

        if (metric_type_ == METRIC_L2) {
            q_obj.set_g_add(dist);
        } else if (metric_type_ == METRIC_IP) {
            const float residual_norm = std::sqrt(
                euclidean_sqr(rotated_query.data(), initer_->centroid(cid), padded_dim_)
            );
            auto g_add_ip = dot_product<float>(
                rotated_query.data(), initer_->centroid(cid), padded_dim_
            );
            q_obj.set_g_add(residual_norm, g_add_ip);
        } else {
            // unsupported
            throw std::invalid_argument("Quantization only supports L2 and IP metrics");
        }
        // q_obj.set_g_add(dist);
        search_cluster(cur_cluster, q_obj, knns, use_hacc, query);
    }

    const size_t found = knns.size();
    if (dists != nullptr) {
        knns.copy_results(results, dists);
        std::fill(dists + found, dists + k, std::numeric_limits<float>::infinity());
    } else {
        knns.copy_results(results);
    }
    std::fill(results + found, results + k, kPidMax);
}

inline void IVF::search_batch(
    const float* queries,
    size_t num_queries,
    size_t k,
    size_t nprobe,
    PID* results,
    float* dists,
    std::optional<bool> use_hacc,
    size_t num_threads
) const {
    if (num_queries == 0) {
        return;
    }
    validate_search(queries, k, nprobe, results);
    const size_t max_values = std::numeric_limits<size_t>::max() / sizeof(float);
    if (num_queries > max_values / dim_ || num_queries > max_values / k) {
        throw std::length_error("IVF query batch is too large");
    }
    nprobe = std::min(nprobe, num_cluster_);
    const bool hacc = use_hacc.value_or(!raw_reranking_ && ex_bits_ > 2);
    const size_t workers = num_queries == 1 || num_threads == 1
                               ? 1
                               : std::min(resolve_num_threads(num_threads), num_queries);
    if (workers == 1) {
        for (size_t i = 0; i < num_queries; ++i) {
            search(
                queries + i * dim_,
                k,
                nprobe,
                results + i * k,
                dists == nullptr ? nullptr : dists + i * k,
                hacc
            );
        }
        return;
    }
    std::atomic<bool> failed{false};
    std::exception_ptr error;
    const auto capture_error = [&] {
#pragma omp critical(rabitq_ivf_search_error)
        {
            if (!error) {
                error = std::current_exception();
            }
        }
        failed.store(true, std::memory_order_relaxed);
    };
#pragma omp parallel for num_threads(workers) schedule(dynamic)
    for (std::ptrdiff_t query_index = 0;
         query_index < static_cast<std::ptrdiff_t>(num_queries);
         ++query_index) {
        if (failed.load(std::memory_order_relaxed)) {
            continue;
        }
        const size_t i = static_cast<size_t>(query_index);
        try {
            search(
                queries + i * dim_,
                k,
                nprobe,
                results + i * k,
                dists == nullptr ? nullptr : dists + i * k,
                hacc
            );
        } catch (...) { capture_error(); }
    }
    if (error) {
        std::rethrow_exception(error);
    }
}

inline void IVF::validate_search(
    const float* query, size_t k, size_t nprobe, const PID* results
) const {
    if (!ready_ || initer_ == nullptr || rotator_ == nullptr ||
        cluster_lst_.size() != num_cluster_ || batch_storage_.empty() ||
        id_storage_.size() != num_) {
        throw std::logic_error("IVF index must be constructed or loaded before search");
    }
    if (query == nullptr) {
        throw std::invalid_argument("IVF search query must not be null");
    }
    if (results == nullptr) {
        throw std::invalid_argument("IVF search results must not be null");
    }
    if (k == 0 || k > num_ || k > buffer::kSearchBufferMaxPointCount) {
        throw std::invalid_argument("IVF search k must be between 1 and the point count");
    }
    if (nprobe == 0) {
        throw std::invalid_argument("IVF search nprobe must be positive");
    }
}

inline void IVF::search_cluster(
    const Cluster& cur_cluster,
    const SplitBatchQuery<float>& q_obj,
    buffer::SearchBuffer<float>& knns,
    bool use_hacc,
    const float* query
) const {
    size_t iter = cur_cluster.num() / fastscan::kBatchSize;
    size_t remain = cur_cluster.num() - (iter * fastscan::kBatchSize);

    const char* batch_data = cur_cluster.batch_data();
    const char* ex_data = cur_cluster.ex_data();
    const PID* ids = cur_cluster.ids();

    /* Compute distances block by block */
    for (size_t i = 0; i < iter; ++i) {
        scan_one_batch(
            batch_data, ex_data, ids, q_obj, knns, fastscan::kBatchSize, use_hacc, query
        );

        batch_data += BatchDataMap<float>::data_bytes(padded_dim_);
        if (rerank_vector_bytes() > 0) {
            ex_data += rerank_vector_bytes() * fastscan::kBatchSize;
        }
        ids += fastscan::kBatchSize;
    }

    if (remain > 0) {
        // scan the last block
        scan_one_batch(batch_data, ex_data, ids, q_obj, knns, remain, use_hacc, query);
    }
}

inline void IVF::scan_one_batch(
    const char* batch_data,
    const char* ex_data,
    const PID* ids,
    const SplitBatchQuery<float>& q_obj,
    buffer::SearchBuffer<float>& knns,
    size_t num_points,
    bool use_hacc,
    const float* query
) const {
    std::array<float, fastscan::kBatchSize> est_distance;  // estimated distance
    std::array<float, fastscan::kBatchSize> low_distance;  // lower distance
    std::array<float, fastscan::kBatchSize> ip_x0_qr;      // inner product of the 1st bit

    split_batch_estdist(
        batch_data,
        q_obj,
        padded_dim_,
        est_distance.data(),
        low_distance.data(),
        ip_x0_qr.data(),
        use_hacc
    );

    float distk = knns.top_dist();

    // Without reranking data, return the one-bit estimates directly.
    if (ex_bits_ == 0 && !raw_reranking_) {
        detail::insert_candidates(knns, ids, est_distance.data(), num_points);
        return;
    }

    // incremental distance computation - V2
    for (size_t i = 0; i < num_points; ++i) {
        float lower_dist = low_distance[i];
        if (lower_dist < distk) {
            PID id = ids[i];
            float ex_dist;
            if (raw_reranking_) {
                const auto* vector = reinterpret_cast<const float*>(ex_data);
                ex_dist = metric_type_ == METRIC_L2 ? euclidean_sqr(query, vector, dim_)
                                                    : dot_product_dis(query, vector, dim_);
            } else {
                ex_dist = split_distance_boosting(
                    ex_data, ip_func_, q_obj, padded_dim_, ex_bits_, ip_x0_qr[i]
                );
            }
            knns.insert(id, ex_dist);
            distk = knns.top_dist();
        }
        ex_data += rerank_vector_bytes();
    }
}
}  // namespace rabitqlib::ivf
