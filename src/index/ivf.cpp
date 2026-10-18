#include "rabitqlib/index/ivf/ivf.hpp"

#include <omp.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <ios>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "ivf_search.hpp"
#include "rabitqlib/core/data_layout.hpp"
#include "rabitqlib/core/encoder.hpp"
#include "rabitqlib/core/estimator.hpp"
#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/core/rotator.hpp"
#include "rabitqlib/fastscan/data_layout.hpp"
#include "rabitqlib/fastscan/encoder.hpp"
#include "rabitqlib/fastscan/estimator.hpp"
#include "rabitqlib/fastscan/pack_codes.hpp"
#include "rabitqlib/fastscan/query.hpp"
#include "rabitqlib/index/detail/buffer.hpp"
#include "rabitqlib/index/detail/parallel_exception.hpp"
#include "rabitqlib/index/detail/parallel_for.hpp"
#include "rabitqlib/index/ivf/cluster.hpp"
#include "rabitqlib/index/ivf/initializer.hpp"
#include "rabitqlib/index/types.hpp"
#include "rabitqlib/utils/path.hpp"
#include "rabitqlib/utils/space.hpp"
#include "rabitqlib/utils/tools.hpp"

namespace rabitqlib::ivf {

IVF::IVF() noexcept = default;
IVF::~IVF() = default;

bool IVF::exceeds_storage(size_t n) const {
    const size_t max_size = std::numeric_limits<size_t>::max();
    const size_t batch_bytes = BatchDataMap<float>::data_bytes(padded_dim_);
    const size_t rerank_bytes = rerank_vector_bytes();
    return n > max_size / sizeof(PID) || n > max_size / sizeof(float) / padded_dim_ ||
           n > max_size / batch_bytes || (rerank_bytes != 0 && n > max_size / rerank_bytes);
}

size_t IVF::batch_data_bytes(size_t cluster_size) const {
    return div_round_up(cluster_size, fastscan::kBatchSize) *
           BatchDataMap<float>::data_bytes(padded_dim_);
}

// Keep the original inlining intent inside this translation unit, especially the
// per-vector stride calculation. Public consumers only need the declarations.
inline size_t IVF::rerank_vector_bytes() const {
    return raw_reranking_ ? sizeof(float) * dim_
                          : ExDataMap<float>::data_bytes(padded_dim_, ex_bits_);
}

inline size_t IVF::ex_data_bytes(size_t cluster_size) const {
    return rerank_vector_bytes() * cluster_size;
}

void IVF::resize_storage(ClusterStorage& cs, size_t n) const {
    cs.batch_storage.resize(batch_data_bytes(n));
    if (rerank_vector_bytes() > 0) {
        if (raw_reranking_) {
            cs.raw_storage.resize(n * dim_);
        } else {
            cs.ex_storage.resize(ex_data_bytes(n));
        }
    }
    cs.id_storage.resize(n);
}

void IVF::reserve_storage(ClusterStorage& cs, size_t n) const {
    cs.batch_storage.reserve(batch_data_bytes(n));
    if (rerank_vector_bytes() > 0) {
        if (raw_reranking_) {
            cs.raw_storage.reserve(n * dim_);
        } else {
            cs.ex_storage.reserve(ex_data_bytes(n));
        }
    }
    cs.id_storage.reserve(n);
}

bool IVF::has_capacity(const ClusterStorage& cs, size_t n) const {
    if (cs.batch_storage.capacity() < batch_data_bytes(n) || cs.id_storage.capacity() < n) {
        return false;
    }
    if (rerank_vector_bytes() == 0) {
        return true;
    }
    return raw_reranking_ ? cs.raw_storage.capacity() >= n * dim_
                          : cs.ex_storage.capacity() >= ex_data_bytes(n);
}

void IVF::free_memory() {
    initializer_.reset();
    std::vector<ClusterStorage>().swap(cluster_storage_);
}

void IVF::swap(IVF& other) noexcept {
    using std::swap;
    swap(initializer_type_, other.initializer_type_);
    swap(initializer_, other.initializer_);
    swap(cluster_storage_, other.cluster_storage_);
    swap(max_elements_, other.max_elements_);
    swap(num_removed_, other.num_removed_);
    swap(dim_, other.dim_);
    swap(padded_dim_, other.padded_dim_);
    swap(num_clusters_, other.num_clusters_);
    swap(raw_reranking_, other.raw_reranking_);
    swap(ready_, other.ready_);
    swap(ex_bits_, other.ex_bits_);
    swap(rotator_type_, other.rotator_type_);
    swap(rotator_, other.rotator_);
    swap(clusters_, other.clusters_);
    swap(metric_type_, other.metric_type_);
    swap(ip_func_, other.ip_func_);
}

IVF::IVF(
    size_t num_points,
    size_t dim,
    size_t num_clusters,
    size_t total_bits,
    MetricType metric_type,
    RotatorType rotator_type,
    InitializerType initializer
)
    : initializer_type_(resolve_initializer_type(initializer, num_clusters))
    , max_elements_(num_points)
    , dim_(dim)
    , padded_dim_(dim)
    , num_clusters_(num_clusters)
    , raw_reranking_(total_bits == 32)
    , ex_bits_(total_bits == 32 ? 0 : total_bits - 1)
    , rotator_type_(rotator_type)
    , metric_type_(metric_type) {
    validate_metric_type(metric_type);
    if ((total_bits < 1 || total_bits > 9) && total_bits != 32) {
        throw std::invalid_argument("IVF bits must be in [1, 9] or 32 for raw reranking");
    };
    if (num_points == 0 || num_points > buffer::kSearchBufferMaxPointCount) {
        throw std::invalid_argument("IVF point count exceeds the supported ID range");
    }
    if (num_clusters == 0 || num_clusters > buffer::kSearchBufferMaxPointCount) {
        throw std::invalid_argument("IVF cluster count exceeds the supported ID range");
    }
    if (dim == 0 || dim > (std::numeric_limits<size_t>::max() / 32) - 64) {
        throw std::invalid_argument("IVF dimension is invalid or too large");
    }
    padded_dim_ = round_up_to_multiple(dim_, 32);
    const size_t max_size = std::numeric_limits<size_t>::max();
    if (exceeds_storage(num_points) ||
        num_clusters > max_size / sizeof(float) / padded_dim_ ||
        (rotator_type == RotatorType::MatrixRotator &&
         dim > max_size / sizeof(float) / padded_dim_)) {
        throw std::invalid_argument("IVF configuration exceeds addressable storage");
    }
    rotator_.reset(choose_rotator<float>(dim, rotator_type, padded_dim_));
    /* check size */
    assert(padded_dim_ % 32 == 0);
    assert(padded_dim_ >= dim_);
}

/**
 * @brief Construct clusters in IVF
 *
 * @param data Data objects (N*DIM)
 * @param centroids Centroid vectors (K*DIM)
 * @param cluster_ids Cluster ID for each data object
 */
void IVF::construct(
    const float* data,
    const float* centroids,
    const PID* cluster_ids,
    bool faster,
    size_t num_threads
) {
    if (max_elements_ == 0 || dim_ == 0 || num_clusters_ == 0 || rotator_ == nullptr) {
        throw std::logic_error("IVF must be configured before construction");
    }
    if (data == nullptr || centroids == nullptr || cluster_ids == nullptr) {
        throw std::invalid_argument("IVF construction inputs must not be null");
    }

    // get id list for each cluster
    std::vector<size_t> counts(num_clusters_, 0);
    std::vector<std::vector<PID>> id_lists(num_clusters_);
    for (size_t i = 0; i < max_elements_; ++i) {
        PID cid = cluster_ids[i];
        if (cid >= num_clusters_) {
            throw std::invalid_argument("Cluster ID is out of range");
        }
        id_lists[cid].push_back(static_cast<PID>(i));
        counts[cid] += 1;
    }

    allocate_memory(counts);

    // init the cluster list
    init_clusters(counts);

    // all rotated centroids
    std::vector<float> rotated_centroids(num_clusters_ * padded_dim_);

    quant::RabitqConfig config;
    if (faster) {
        config = quant::faster_config(padded_dim_, ex_bits_ + 1);
    }

    num_threads = std::min(resolve_num_threads(num_threads), num_clusters_);
    const size_t scratch_points =
        std::min(fastscan::kBatchSize, *std::max_element(counts.begin(), counts.end()));
    // Each worker reuses one batch, independently of the largest cluster size.
    // Allocate before entering OpenMP so allocation failures reach the caller.
    std::vector<std::vector<float>> rotated_blocks(num_threads);
    for (auto& block : rotated_blocks) {
        block.resize(scratch_points * padded_dim_);
    }
    /* Quantize each cluster */
    {
        index_detail::ParallelException failure;
#pragma omp parallel for schedule(dynamic) num_threads(num_threads)
        for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(num_clusters_);
             ++index) {
            if (failure.failed()) {
                continue;
            }
            try {
                const size_t i = static_cast<size_t>(index);
                const float* cur_centroid = centroids + (i * dim_);
                float* cur_rotated_c = &rotated_centroids[i * padded_dim_];
                Cluster& cp = clusters_[i];
                quantize_cluster(
                    cp,
                    id_lists[i],
                    data,
                    cur_centroid,
                    cur_rotated_c,
                    rotated_blocks[static_cast<size_t>(omp_get_thread_num())].data(),
                    config
                );
            } catch (...) { failure.capture(); }
        }
        failure.rethrow();
    }

    this->initializer_->add_vectors(rotated_centroids.data(), num_threads);
    ready_ = true;
}

void IVF::allocate_memory(const std::vector<size_t>& cluster_sizes) {
    ready_ = false;
    num_removed_ = 0;
    free_memory();
    clusters_.clear();

    if (initializer_type_ == InitializerType::Flat) {
        this->initializer_ =
            std::make_unique<FlatInitializer>(padded_dim_, num_clusters_, metric_type_);
    } else if (initializer_type_ == InitializerType::FlatRaBitQ) {
        this->initializer_ = std::make_unique<FlatRaBitQInitializer>(
            padded_dim_, num_clusters_, metric_type_
        );
    } else {
        this->initializer_ =
            std::make_unique<HNSWInitializer>(padded_dim_, num_clusters_, metric_type_);
    }
    cluster_storage_.assign(num_clusters_, ClusterStorage(raw_reranking_));
    for (size_t i = 0; i < num_clusters_; ++i) {
        resize_storage(cluster_storage_[i], cluster_sizes[i]);
    }

    this->ip_func_ = select_excode_ipfunc(ex_bits_);
}

/**
 * @brief Initialize the cluster list from each cluster's own independent storage.
 */
void IVF::init_clusters(const std::vector<size_t>& cluster_sizes) {
    this->clusters_.reserve(num_clusters_);
    for (size_t i = 0; i < num_clusters_; ++i) {
        ClusterStorage& cs = cluster_storage_[i];
        char* current_ex_data = rerank_vector_bytes() > 0 ? cs.ex_data() : nullptr;
        this->clusters_.emplace_back(
            cluster_sizes[i], cs.batch_data(), current_ex_data, cs.ids()
        );
    }
}

void IVF::quantize_cluster(
    Cluster& cp,
    const std::vector<PID>& point_ids,
    const float* data,
    const float* cur_centroid,
    float* rotated_centroid,
    float* rotated_data,
    const quant::RabitqConfig& config
) {
    size_t num_points = point_ids.size();
    if (cp.num() != num_points) {
        throw std::invalid_argument("Cluster size and ID count differ");
    }
    std::copy(point_ids.begin(), point_ids.end(), cp.ids());
    this->rotator_->rotate(cur_centroid, rotated_centroid);

    char* batch_data = cp.batch_data();
    char* ex_data = cp.ex_data();
    for (size_t i = 0; i < num_points; i += fastscan::kBatchSize) {
        size_t n = std::min(fastscan::kBatchSize, num_points - i);
        for (size_t j = 0; j < n; ++j) {
            const float* vector = data + (point_ids[i + j] * dim_);
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

void IVF::add(
    const float* data, size_t n, const PID* cluster_ids, bool faster, size_t num_threads
) {
    if (!ready_ || initializer_ == nullptr || rotator_ == nullptr ||
        clusters_.size() != num_clusters_ || cluster_storage_.size() != num_clusters_) {
        throw std::logic_error("IVF index must be constructed or loaded before add");
    }
    if (n == 0) {
        return;
    }
    if (data == nullptr) {
        throw std::invalid_argument("IVF add data must not be null");
    }
    if (n > buffer::kSearchBufferMaxPointCount - max_elements_) {
        throw std::invalid_argument("IVF point count exceeds the supported ID range");
    }
    const size_t new_num_points = max_elements_ + n;
    if (exceeds_storage(new_num_points)) {
        throw std::invalid_argument("IVF configuration exceeds addressable storage");
    }

    /* Assign every new point to a cluster */
    std::vector<PID> assigned(n);
    if (cluster_ids != nullptr) {
        for (size_t i = 0; i < n; ++i) {
            if (cluster_ids[i] >= num_clusters_) {
                throw std::invalid_argument("Cluster ID is out of range");
            }
            assigned[i] = cluster_ids[i];
        }
    } else {
        rabitqlib::index_detail::parallel_for(
            0,
            n,
            num_threads,
            [&](size_t row, size_t /*thread_id*/) {
                std::vector<float> rotated(padded_dim_);
                std::vector<AnnCandidate<float>> nearest(1);
                rotator_->rotate(data + (row * dim_), rotated.data());
                initializer_->centroids_distances(rotated.data(), 1, nearest);
                assigned[row] = nearest[0].id;
            }
        );
    }

    std::vector<std::vector<PID>> rows_of(num_clusters_);
    for (size_t i = 0; i < n; ++i) {
        rows_of[assigned[i]].push_back(static_cast<PID>(i));
    }
    std::vector<size_t> new_sizes(num_clusters_);
    std::vector<size_t> touched;
    for (size_t i = 0; i < num_clusters_; ++i) {
        new_sizes[i] = clusters_[i].num() + rows_of[i].size();
        if (!rows_of[i].empty()) {
            touched.push_back(i);
        }
    }

    quant::RabitqConfig config;
    if (faster) {
        config = quant::faster_config(padded_dim_, ex_bits_ + 1);
    }

    const size_t rerank_bytes = rerank_vector_bytes();
    const size_t batch_bytes = BatchDataMap<float>::data_bytes(padded_dim_);
    const PID first_pid = static_cast<PID>(max_elements_);

    /* Rebuild clusters without reserved capacity on the side; the rest grow in place */
    std::vector<std::optional<ClusterStorage>> staged(touched.size());
    std::vector<ByteStorage> tails(touched.size());  // repacked tails of in-place clusters
    for (size_t t = 0; t < touched.size(); ++t) {
        const size_t i = touched[t];
        if (has_capacity(cluster_storage_[i], new_sizes[i])) {
            const size_t full_batches = clusters_[i].num() / fastscan::kBatchSize;
            tails[t].resize(batch_data_bytes(new_sizes[i]) - (full_batches * batch_bytes));
        } else {
            staged[t].emplace(raw_reranking_);
            resize_storage(*staged[t], new_sizes[i]);
        }
    }
    // Growing within capacity cannot reallocate or throw, so old cluster views stay valid
    for (size_t t = 0; t < touched.size(); ++t) {
        if (!staged[t]) {
            resize_storage(cluster_storage_[touched[t]], new_sizes[touched[t]]);
        }
    }

    // An exception must not leave an OpenMP region, so keep the first one and rethrow it.
    index_detail::ParallelException failure;
#pragma omp parallel for schedule(dynamic) num_threads(resolve_num_threads(num_threads))
    for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(touched.size());
         ++index) {
        const auto t = static_cast<size_t>(index);
        const size_t i = touched[t];
        try {
            ClusterStorage& cs = staged[t] ? *staged[t] : cluster_storage_[i];
            const size_t full_batches = clusters_[i].num() / fastscan::kBatchSize;
            char* tail = staged[t] ? cs.batch_data() + (full_batches * batch_bytes)
                                   : reinterpret_cast<char*>(tails[t].data());
            extend_cluster(
                clusters_[i],
                rows_of[i],
                data,
                initializer_->centroid(static_cast<PID>(i)),
                first_pid,
                cs.batch_data(),
                rerank_bytes > 0 ? cs.ex_data() : nullptr,
                cs.ids(),
                tail,
                config
            );
        } catch (...) { failure.capture(); }
    }
    if (failure.failed()) {
        // In-place clusters were only written past their old size; shrinking drops that
        for (size_t t = 0; t < touched.size(); ++t) {
            if (!staged[t]) {
                resize_storage(cluster_storage_[touched[t]], clusters_[touched[t]].num());
            }
        }
        failure.rethrow();
    }

    /* Commit. Nothing below can throw. */
    for (size_t t = 0; t < touched.size(); ++t) {
        const size_t i = touched[t];
        if (staged[t]) {
            cluster_storage_[i] = std::move(*staged[t]);
        } else {
            const size_t full_batches = clusters_[i].num() / fastscan::kBatchSize;
            std::copy_n(
                tails[t].data(),
                tails[t].size(),
                cluster_storage_[i].batch_storage.data() + (full_batches * batch_bytes)
            );
        }
    }
    max_elements_ = new_num_points;
    clusters_.clear();
    init_clusters(new_sizes);
}

/**
 * @brief Copy one cluster into its grown storage and append new points to it.
 *
 * Full batches, rerank data and IDs are copied unchanged, unless they are already in place.
 * The old partial tail batch is unpacked and repacked together with the new points into
 * `tail_dst`, so the result is byte-identical to quantizing the same points in the same
 * order in one construct call.
 */
void IVF::extend_cluster(
    const Cluster& old_cluster,
    const std::vector<PID>& rows,
    const float* data,
    const float* rotated_centroid,
    PID first_pid,
    char* batch_dst,
    char* rerank_dst,
    PID* ids_dst,
    char* tail_dst,
    const quant::RabitqConfig& config
) const {
    assert(!rows.empty());  // add() only extends clusters that receive new points
    const size_t old_num = old_cluster.num();
    const size_t batch_bytes = BatchDataMap<float>::data_bytes(padded_dim_);
    const size_t rerank_bytes = rerank_vector_bytes();
    const size_t cols = padded_dim_ / 8;

    const size_t full_batches = old_num / fastscan::kBatchSize;
    const size_t kept = old_num - (full_batches * fastscan::kBatchSize);
    const size_t tail_num = kept + rows.size();

    if (batch_dst != old_cluster.batch_data()) {
        std::copy_n(old_cluster.batch_data(), full_batches * batch_bytes, batch_dst);
    }
    if (rerank_dst != old_cluster.ex_data()) {
        std::copy_n(old_cluster.ex_data(), old_num * rerank_bytes, rerank_dst);
    }
    if (ids_dst != old_cluster.ids()) {
        std::copy_n(old_cluster.ids(), old_num, ids_dst);
    }

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
    char* batch = tail_dst;
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

void IVF::reserve(PID cluster_id, size_t num_points) {
    if (!ready_ || clusters_.size() != num_clusters_ ||
        cluster_storage_.size() != num_clusters_) {
        throw std::logic_error("IVF index must be constructed or loaded before reserve");
    }
    if (cluster_id >= num_clusters_) {
        throw std::invalid_argument("Cluster ID is out of range");
    }
    if (num_points > buffer::kSearchBufferMaxPointCount) {
        throw std::invalid_argument("IVF point count exceeds the supported ID range");
    }
    if (exceeds_storage(num_points)) {
        throw std::invalid_argument("IVF configuration exceeds addressable storage");
    }
    ClusterStorage& cs = cluster_storage_[cluster_id];
    if (has_capacity(cs, num_points)) {
        return;
    }

    const Cluster& old_cluster = clusters_[cluster_id];
    const size_t num = old_cluster.num();
    ClusterStorage grown(raw_reranking_);
    reserve_storage(grown, num_points);
    resize_storage(grown, num);
    std::copy_n(old_cluster.batch_data(), batch_data_bytes(num), grown.batch_data());
    if (rerank_vector_bytes() > 0) {
        std::copy_n(old_cluster.ex_data(), ex_data_bytes(num), grown.ex_data());
    }
    std::copy_n(old_cluster.ids(), num, grown.ids());
    std::vector<size_t> cluster_sizes(num_clusters_);
    for (size_t i = 0; i < num_clusters_; ++i) {
        cluster_sizes[i] = clusters_[i].num();
    }

    /* Commit. Rebuilding the cluster list reuses its capacity, so it cannot throw. */
    cs = std::move(grown);
    clusters_.clear();
    init_clusters(cluster_sizes);
}

size_t IVF::remove(const PID* ids_to_remove, size_t n) {
    if (!ready_ || initializer_ == nullptr || rotator_ == nullptr ||
        clusters_.size() != num_clusters_ || cluster_storage_.size() != num_clusters_) {
        throw std::logic_error("IVF index must be constructed or loaded before remove");
    }
    if (n == 0) {
        return 0;
    }
    if (ids_to_remove == nullptr) {
        throw std::invalid_argument("IVF remove IDs must not be null");
    }

    // Validate every ID before changing anything
    std::vector<bool> marked(max_elements_, false);
    for (size_t i = 0; i < n; ++i) {
        if (ids_to_remove[i] >= max_elements_) {
            throw std::invalid_argument("IVF remove ID is out of range");
        }
        marked[ids_to_remove[i]] = true;
    }

    // A point whose f_add is +inf has +inf estimated and lower-bound distances in every
    // scan path, so it never enters the result buffer or triggers reranking.
    constexpr float kRemoved = std::numeric_limits<float>::infinity();
    const size_t batch_bytes = BatchDataMap<float>::data_bytes(padded_dim_);
    size_t removed = 0;
    for (const auto& cluster : clusters_) {
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
    num_removed_ += removed;
    return removed;
}

void IVF::save(const char* filename) const {
    if (!ready_ || initializer_ == nullptr || rotator_ == nullptr ||
        clusters_.size() != num_clusters_ || cluster_storage_.size() != num_clusters_) {
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
    output.write(reinterpret_cast<const char*>(&max_elements_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&dim_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&num_clusters_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&ex_bits_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&rotator_type_), sizeof(rotator_type_));
    output.write(reinterpret_cast<const char*>(&metric_type_), sizeof(metric_type_));

    /* Save number of vectors of each cluster */
    std::vector<size_t> cluster_sizes;
    cluster_sizes.reserve(num_clusters_);
    for (const auto& cur_cluster : clusters_) {
        cluster_sizes.push_back(cur_cluster.num());
    }
    output.write(
        reinterpret_cast<const char*>(cluster_sizes.data()),
        static_cast<std::streamsize>(sizeof(size_t) * num_clusters_)
    );

    /* Save rotator */
    this->rotator_->save(output);

    /* Save data: all batch data, then all ex data, then all IDs, each in cluster order */
    this->initializer_->save(output, filename);
    for (size_t i = 0; i < num_clusters_; ++i) {
        output.write(
            cluster_storage_[i].batch_data(),
            static_cast<std::streamsize>(batch_data_bytes(cluster_sizes[i]))
        );
    }
    if (rerank_vector_bytes() != 0) {
        for (size_t i = 0; i < num_clusters_; ++i) {
            output.write(
                cluster_storage_[i].ex_data(),
                static_cast<std::streamsize>(ex_data_bytes(cluster_sizes[i]))
            );
        }
    }
    for (size_t i = 0; i < num_clusters_; ++i) {
        output.write(
            reinterpret_cast<const char*>(cluster_storage_[i].ids()),
            static_cast<std::streamsize>(ids_bytes(cluster_sizes[i]))
        );
    }

    output.close();
}

void IVF::load(const char* filename) {
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
    input.read(reinterpret_cast<char*>(&loaded.max_elements_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&loaded.dim_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&loaded.num_clusters_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&loaded.ex_bits_), sizeof(size_t));
    input.read(
        reinterpret_cast<char*>(&loaded.rotator_type_), sizeof(loaded.rotator_type_)
    );
    input.read(reinterpret_cast<char*>(&loaded.metric_type_), sizeof(loaded.metric_type_));
    validate_metric_type(loaded.metric_type_);
    // Missing routing metadata means a historical file. Its payload layout must
    // follow the original threshold, independently of the current Auto policy.
    if (loaded.initializer_type_ == InitializerType::Auto) {
        loaded.initializer_type_ =
            loaded.num_clusters_ < 20000 ? InitializerType::Flat : InitializerType::HNSW;
    }
    if (loaded.max_elements_ == 0 ||
        loaded.max_elements_ > buffer::kSearchBufferMaxPointCount || loaded.dim_ == 0 ||
        loaded.num_clusters_ == 0 ||
        loaded.num_clusters_ > buffer::kSearchBufferMaxPointCount || loaded.ex_bits_ > 8 ||
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
    consume(loaded.num_clusters_, sizeof(size_t));
    consume(loaded.max_elements_, sizeof(PID));
    consume(loaded.max_elements_, loaded.rerank_vector_bytes());
    if (loaded.rotator_type_ == RotatorType::MatrixRotator) {
        consume(loaded.dim_, sizeof(float) * loaded.padded_dim_);
    } else if (loaded.rotator_type_ == RotatorType::FhtKacRotator) {
        consume(1, loaded.padded_dim_ / 2);
    } else {
        throw std::runtime_error("Invalid IVF rotator type");
    }
    if (loaded.initializer_type_ != InitializerType::HNSW) {
        consume(loaded.num_clusters_, sizeof(float) * loaded.padded_dim_);
    }
    if (loaded.initializer_type_ == InitializerType::FlatRaBitQ) {
        consume(loaded.padded_dim_, sizeof(float));
        consume(
            div_round_up(loaded.num_clusters_, fastscan::kBatchSize),
            BatchDataMap<float>::data_bytes(loaded.padded_dim_)
        );
    }

    /* Load number of vectors of each cluster */
    std::vector<size_t> cluster_sizes(loaded.num_clusters_, 0);
    input.read(
        reinterpret_cast<char*>(cluster_sizes.data()),
        static_cast<std::streamsize>(sizeof(size_t) * loaded.num_clusters_)
    );

    size_t total = 0;
    for (size_t size : cluster_sizes) {
        if (size > loaded.max_elements_ - total) {
            throw std::runtime_error("Invalid cluster counts in IVF index file");
        }
        total += size;
        consume(
            div_round_up(size, fastscan::kBatchSize),
            BatchDataMap<float>::data_bytes(loaded.padded_dim_)
        );
    }
    if (total != loaded.max_elements_) {
        throw std::runtime_error("Invalid cluster counts in IVF index file");
    }
    loaded.rotator_ = rotator_impl::make_rotator_for_loading(
        loaded.dim_, loaded.rotator_type_, loaded.padded_dim_
    );

    /* Load rotator */
    loaded.rotator_->load(input);

    /* Load data: all batch data, then all ex data, then all IDs, each in cluster order */
    loaded.allocate_memory(cluster_sizes);
    loaded.initializer_->load(input, filename);
    for (size_t i = 0; i < loaded.num_clusters_; ++i) {
        input.read(
            loaded.cluster_storage_[i].batch_data(),
            static_cast<std::streamsize>(loaded.batch_data_bytes(cluster_sizes[i]))
        );
    }
    if (loaded.rerank_vector_bytes() != 0) {
        for (size_t i = 0; i < loaded.num_clusters_; ++i) {
            input.read(
                loaded.cluster_storage_[i].ex_data(),
                static_cast<std::streamsize>(loaded.ex_data_bytes(cluster_sizes[i]))
            );
        }
    }
    for (size_t i = 0; i < loaded.num_clusters_; ++i) {
        input.read(
            reinterpret_cast<char*>(loaded.cluster_storage_[i].ids()),
            static_cast<std::streamsize>(ids_bytes(cluster_sizes[i]))
        );
    }
    for (size_t i = 0; i < loaded.num_clusters_; ++i) {
        for (PID id : loaded.cluster_storage_[i].id_storage) {
            if (id >= loaded.max_elements_ || id >= buffer::kSearchBufferCheckedMask) {
                throw std::runtime_error("Invalid point ID in IVF index file");
            }
        }
    }

    /* Init each cluster */
    loaded.init_clusters(cluster_sizes);
    // Removal markers are already persisted, including in historical files.
    // Count only logical rows: FastScan padding lanes are not stored points.
    const size_t batch_bytes = BatchDataMap<float>::data_bytes(loaded.padded_dim_);
    for (const auto& cluster : loaded.clusters_) {
        for (size_t first = 0; first < cluster.num(); first += fastscan::kBatchSize) {
            ConstBatchDataMap<float> batch(
                cluster.batch_data() + (first / fastscan::kBatchSize) * batch_bytes,
                loaded.padded_dim_
            );
            const size_t count = std::min(fastscan::kBatchSize, cluster.num() - first);
            for (size_t lane = 0; lane < count; ++lane) {
                loaded.num_removed_ +=
                    batch.f_add()[lane] == std::numeric_limits<float>::infinity();
            }
        }
    }
    loaded.ready_ = true;

    input.close();
    swap(loaded);
}

void IVF::search(
    const float* query, size_t k, size_t nprobe, PID* results, float* distances
) const {
    search(query, k, nprobe, results, distances, !raw_reranking_ && ex_bits_ > 2);
}

void IVF::search(
    const float* __restrict__ query,
    size_t k,
    size_t nprobe,
    PID* __restrict__ results,
    bool use_hacc
) const {
    this->search(query, k, nprobe, results, nullptr, use_hacc);
}

void IVF::search(
    const float* __restrict__ query,
    size_t k,
    size_t nprobe,
    PID* __restrict__ results,
    float* __restrict__ distances,
    bool use_hacc
) const {
    validate_search(query, k, nprobe, results);

    nprobe = std::min(nprobe, num_clusters_);  // corner case
    std::vector<float> rotated_query(padded_dim_);
    this->rotator_->rotate(query, rotated_query.data());

    // use initer to get closest nprobe centroids
    std::vector<AnnCandidate<float>> centroid_dist(nprobe);
    this->initializer_->centroids_distances(rotated_query.data(), nprobe, centroid_dist);

    buffer::SearchBuffer knns(k);

    SplitBatchQuery<float> q_obj(
        rotated_query.data(), padded_dim_, ex_bits_, metric_type_, use_hacc
    );

    for (size_t i = 0; i < nprobe; ++i) {
        PID cid = centroid_dist[i].id;
        float dist = centroid_dist[i].distance;
        const Cluster& cur_cluster = clusters_[cid];

        if (metric_type_ == METRIC_L2) {
            q_obj.set_g_add(dist);
        } else if (metric_type_ == METRIC_IP) {
            const float residual_norm = std::sqrt(euclidean_sqr(
                rotated_query.data(), initializer_->centroid(cid), padded_dim_
            ));
            auto g_add_ip = dot_product<float>(
                rotated_query.data(), initializer_->centroid(cid), padded_dim_
            );
            q_obj.set_g_add(residual_norm, g_add_ip);
        } else {
            // unsupported
            throw std::invalid_argument("Quantization only supports L2 and IP metrics");
        }
        search_cluster(cur_cluster, q_obj, knns, use_hacc, query);
    }

    const size_t found = knns.size();
    if (distances != nullptr) {
        knns.copy_results(results, distances);
        std::fill(distances + found, distances + k, std::numeric_limits<float>::infinity());
    } else {
        knns.copy_results(results);
    }
    std::fill(results + found, results + k, kPidMax);
}

void IVF::search_batch(
    const float* queries,
    size_t num_queries,
    size_t k,
    size_t nprobe,
    PID* results,
    float* distances,
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
    nprobe = std::min(nprobe, num_clusters_);
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
                distances == nullptr ? nullptr : distances + i * k,
                hacc
            );
        }
        return;
    }
    index_detail::ParallelException failure;
#pragma omp parallel for num_threads(workers) schedule(dynamic)
    for (std::ptrdiff_t query_index = 0;
         query_index < static_cast<std::ptrdiff_t>(num_queries);
         ++query_index) {
        if (failure.failed()) {
            continue;
        }
        const size_t i = static_cast<size_t>(query_index);
        try {
            search(
                queries + i * dim_,
                k,
                nprobe,
                results + i * k,
                distances == nullptr ? nullptr : distances + i * k,
                hacc
            );
        } catch (...) { failure.capture(); }
    }
    failure.rethrow();
}

inline void IVF::validate_search(
    const float* query, size_t k, size_t nprobe, const PID* results
) const {
    if (!ready_ || initializer_ == nullptr || rotator_ == nullptr ||
        clusters_.size() != num_clusters_ || cluster_storage_.size() != num_clusters_) {
        throw std::logic_error("IVF index must be constructed or loaded before search");
    }
    if (query == nullptr) {
        throw std::invalid_argument("IVF search query must not be null");
    }
    if (results == nullptr) {
        throw std::invalid_argument("IVF search results must not be null");
    }
    if (k == 0 || k > max_elements_ || k > buffer::kSearchBufferMaxPointCount) {
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
    std::array<float, fastscan::kBatchSize> est_distance;
    std::array<float, fastscan::kBatchSize> low_distance;
    std::array<float, fastscan::kBatchSize> ip_x0_qr;  // inner product of the 1st bit

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

    // Refine candidates passing the probabilistic one-bit filter. Quantized storage
    // reuses ip_x0_qr with the extra bits; raw storage scores the original vector.
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
