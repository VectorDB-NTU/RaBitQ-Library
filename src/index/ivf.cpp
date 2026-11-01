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
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "file_replacement.hpp"
#include "ivf_search.hpp"
#include "rabitqlib/clustering/qgkmeans.hpp"
#include "rabitqlib/clustering/rabitqkmeans.hpp"
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
#include "rabitqlib/index/training.hpp"
#include "rabitqlib/index/types.hpp"
#include "rabitqlib/utils/input_checks.hpp"
#include "rabitqlib/utils/path.hpp"
#include "rabitqlib/utils/space.hpp"
#include "rabitqlib/utils/tools.hpp"

namespace rabitqlib::ivf {

namespace {
// remove() marks a point by this f_add; save and load keep the marker.
constexpr float kRemovedFAdd = std::numeric_limits<float>::infinity();
// Files without routing metadata used HNSW routing from this many clusters.
constexpr size_t kHistoricalHnswMinClusters = 20000;
// An index from the point-count constructor is built by construct, never trained.
constexpr const char* kPointCountConstructOnly =
    "IVF index created with a point count must be built by construct";
}  // namespace

IVF::IVF() noexcept = default;
IVF::~IVF() = default;

bool IVF::exceeds_storage(size_t num_points) const {
    const size_t max_size = std::numeric_limits<size_t>::max();
    const size_t rerank_bytes = rerank_vector_bytes();
    return num_points > max_size / sizeof(PID) ||
           num_points > max_size / sizeof(float) / padded_dim_ ||
           num_points > max_size / batch_bytes() ||
           (rerank_bytes != 0 && num_points > max_size / rerank_bytes);
}

size_t IVF::batch_data_bytes(size_t cluster_size) const {
    return div_round_up(cluster_size, fastscan::kBatchSize) * batch_bytes();
}

// Keep the original inlining intent inside this translation unit, especially the
// per-vector stride calculation. Public consumers only need the declarations.
inline size_t IVF::batch_bytes() const {
    return BatchDataMap<float>::data_bytes(padded_dim_);
}

inline size_t IVF::rerank_vector_bytes() const {
    return raw_reranking_ ? sizeof(float) * dim_
                          : ExDataMap<float>::data_bytes(padded_dim_, ex_bits_);
}

inline size_t IVF::rerank_data_bytes(size_t cluster_size) const {
    return rerank_vector_bytes() * cluster_size;
}

void IVF::resize_storage(ClusterStorage& cs, size_t num_points) const {
    cs.batch_storage.resize(batch_data_bytes(num_points));
    if (has_rerank_data()) {
        if (raw_reranking_) {
            cs.raw_storage.resize(num_points * dim_);
        } else {
            cs.ex_storage.resize(rerank_data_bytes(num_points));
        }
    }
    cs.id_storage.resize(num_points);
}

void IVF::reserve_storage(ClusterStorage& cs, size_t num_points) const {
    cs.batch_storage.reserve(batch_data_bytes(num_points));
    if (has_rerank_data()) {
        if (raw_reranking_) {
            cs.raw_storage.reserve(num_points * dim_);
        } else {
            cs.ex_storage.reserve(rerank_data_bytes(num_points));
        }
    }
    cs.id_storage.reserve(num_points);
}

bool IVF::has_capacity(const ClusterStorage& cs, size_t num_points) const {
    if (cs.batch_storage.capacity() < batch_data_bytes(num_points) ||
        cs.id_storage.capacity() < num_points) {
        return false;
    }
    if (!has_rerank_data()) {
        return true;
    }
    return raw_reranking_ ? cs.raw_storage.capacity() >= num_points * dim_
                          : cs.ex_storage.capacity() >= rerank_data_bytes(num_points);
}

void IVF::free_storage() {
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
    InitializerType initializer_type
)
    : initializer_type_(resolve_initializer_type(initializer_type, num_clusters))
    , max_elements_(num_points)
    , dim_(dim)
    , padded_dim_(dim)
    , num_clusters_(num_clusters)
    , raw_reranking_(total_bits == 32)
    , ex_bits_(total_bits == 32 ? 0 : total_bits - 1)
    , rotator_type_(rotator_type)
    , metric_type_(metric_type) {
    configure(total_bits);
    if (num_points == 0 || num_points > buffer::kSearchBufferMaxPointCount) {
        throw std::invalid_argument("IVF point count exceeds the supported ID range");
    }
    if (exceeds_storage(num_points)) {
        throw std::invalid_argument("IVF configuration exceeds addressable storage");
    }
    rotator_.reset(choose_rotator<float>(dim, rotator_type, padded_dim_));
}

IVF::IVF(
    size_t dim,
    size_t num_clusters,
    size_t total_bits,
    MetricType metric_type,
    uint32_t seed,
    RotatorType rotator_type,
    InitializerType initializer_type
)
    : initializer_type_(resolve_initializer_type(initializer_type, num_clusters))
    , dim_(dim)
    , padded_dim_(dim)
    , num_clusters_(num_clusters)
    , raw_reranking_(total_bits == 32)
    , ex_bits_(total_bits == 32 ? 0 : total_bits - 1)
    , rotator_type_(rotator_type)
    , metric_type_(metric_type) {
    configure(total_bits);
    rotator_.reset(choose_rotator<float>(dim, rotator_type, padded_dim_, seed));
}

void IVF::configure(size_t total_bits) {
    validate_metric_type(metric_type_);
    if ((total_bits < 1 || total_bits > 9) && total_bits != 32) {
        throw std::invalid_argument("IVF bits must be in [1, 9] or 32 for raw reranking");
    }
    if (num_clusters_ == 0 || num_clusters_ > buffer::kSearchBufferMaxPointCount) {
        throw std::invalid_argument("IVF cluster count exceeds the supported ID range");
    }
    if (dim_ == 0 || dim_ > (std::numeric_limits<size_t>::max() / 32) - 64) {
        throw std::invalid_argument("IVF dimension is invalid or too large");
    }
    padded_dim_ = round_up_to_multiple(dim_, 32);
    const size_t max_size = std::numeric_limits<size_t>::max();
    if (num_clusters_ > max_size / sizeof(float) / padded_dim_ ||
        (rotator_type_ == RotatorType::MatrixRotator &&
         dim_ > max_size / sizeof(float) / padded_dim_)) {
        throw std::invalid_argument("IVF configuration exceeds addressable storage");
    }
}

std::unique_ptr<Initializer> IVF::make_initializer() const {
    if (initializer_type_ == InitializerType::Flat) {
        return std::make_unique<FlatInitializer>(padded_dim_, num_clusters_, metric_type_);
    }
    if (initializer_type_ == InitializerType::FlatRaBitQ) {
        return std::make_unique<FlatRaBitQInitializer>(
            padded_dim_, num_clusters_, metric_type_
        );
    }
    return std::make_unique<HNSWInitializer>(padded_dim_, num_clusters_, metric_type_);
}

void IVF::set_centroids(const float* centroids, size_t num_threads) {
    if (rotator_ == nullptr) {
        throw std::logic_error("IVF must be configured before setting centroids");
    }
    if (num_stored_points() != 0) {
        throw std::logic_error("IVF cannot replace the centroids of an index with points");
    }
    if (!ready_ && max_elements_ != 0) {
        throw std::logic_error(kPointCountConstructOnly);
    }
    if (centroids == nullptr) {
        throw std::invalid_argument("IVF centroids must not be null");
    }
    check_coordinates(centroids, num_clusters_ * dim_, dim_, "IVF centroids");

    // construct rotates centroids the same way, so both paths route identically.
    std::vector<float> rotated_centroids(num_clusters_ * padded_dim_);
    index_detail::parallel_for(
        0,
        num_clusters_,
        num_threads,
        [&](size_t i, size_t /*thread*/) {
            rotator_->rotate(centroids + (i * dim_), &rotated_centroids[i * padded_dim_]);
        },
        /*reuse_workers=*/true
    );
    auto initializer = make_initializer();
    initializer->add_vectors(
        rotated_centroids.data(), std::min(resolve_num_threads(num_threads), num_clusters_)
    );
    std::vector<ClusterStorage> cluster_storage(
        num_clusters_, ClusterStorage(raw_reranking_)
    );
    std::vector<Cluster> clusters;
    clusters.reserve(num_clusters_);
    for (auto& cs : cluster_storage) {
        clusters.emplace_back(
            0, cs.batch_data(), has_rerank_data() ? cs.rerank_data() : nullptr, cs.ids()
        );
    }
    const auto ip_func = select_excode_ipfunc(ex_bits_);

    // Commit. Moving the vectors keeps the empty views' storage, and nothing throws.
    initializer_ = std::move(initializer);
    cluster_storage_ = std::move(cluster_storage);
    clusters_ = std::move(clusters);
    ip_func_ = ip_func;
    ready_ = true;
}

void IVF::train(
    size_t num_points, const float* data, const rabitqkmeans::RaBitQKMeansParameters& cp
) {
    if (rotator_ == nullptr) {
        throw std::logic_error("IVF must be configured before training");
    }
    if (num_stored_points() != 0) {
        throw std::logic_error("IVF cannot retrain an index with points");
    }
    if (!ready_ && max_elements_ != 0) {
        throw std::logic_error(kPointCountConstructOnly);
    }
    if (data == nullptr) {
        throw std::invalid_argument("IVF training data must not be null");
    }
    if (num_points < num_clusters_) {
        throw std::invalid_argument("IVF training needs at least num_clusters points");
    }
    if (num_points > std::numeric_limits<size_t>::max() / sizeof(float) / dim_) {
        throw std::invalid_argument("IVF training data exceeds addressable storage");
    }
    // k-means checks the data for finite, in-range coordinates in parallel with
    // the bound check_coordinates uses; set_centroids checks its centroids.
    std::vector<float> centroids;
    if (num_clusters_ <= kMaxRaBitQKMeansClusters) {
        rabitqkmeans::RaBitQKMeans kmeans(dim_, num_clusters_, cp);
        kmeans.train_centroids(num_points, data);
        centroids = std::move(kmeans.centroids);
    } else {
        qgkmeans::QGKMeansParameters qg_cp;  // Default graph settings
        clustering::detail::copy_shared_parameters(cp, qg_cp);
        qgkmeans::QGKMeans kmeans(dim_, num_clusters_, qg_cp);
        kmeans.train_centroids(num_points, data);
        centroids = std::move(kmeans.centroids);
    }
    set_centroids(centroids.data(), cp.num_threads);
}

void IVF::construct(
    const float* data,
    const float* centroids,
    const PID* cluster_ids,
    bool fast_quantization,
    size_t num_threads
) {
    if (max_elements_ == 0 || dim_ == 0 || num_clusters_ == 0 || rotator_ == nullptr) {
        throw std::logic_error("IVF must be configured before construction");
    }
    if (data == nullptr || centroids == nullptr || cluster_ids == nullptr) {
        throw std::invalid_argument("IVF construction inputs must not be null");
    }

    // Collect the point IDs of each cluster
    std::vector<size_t> cluster_sizes(num_clusters_, 0);
    std::vector<std::vector<PID>> id_lists(num_clusters_);
    for (size_t i = 0; i < max_elements_; ++i) {
        PID cluster_id = cluster_ids[i];
        if (cluster_id >= num_clusters_) {
            throw std::invalid_argument("IVF cluster ID is out of range");
        }
        id_lists[cluster_id].push_back(static_cast<PID>(i));
        cluster_sizes[cluster_id] += 1;
    }
    check_coordinates(data, max_elements_ * dim_, dim_, "IVF construction data");
    check_coordinates(centroids, num_clusters_ * dim_, dim_, "IVF construction centroids");

    allocate_storage(cluster_sizes);
    create_cluster_views(cluster_sizes);

    // Rotated centroids, one row per cluster
    std::vector<float> rotated_centroids(num_clusters_ * padded_dim_);

    quant::RabitqConfig config;
    if (fast_quantization) {
        config = quant::faster_config(padded_dim_, ex_bits_ + 1);
    }

    const size_t num_workers = std::min(resolve_num_threads(num_threads), num_clusters_);
    const size_t scratch_points = std::min(
        fastscan::kBatchSize, *std::max_element(cluster_sizes.begin(), cluster_sizes.end())
    );
    // Each worker reuses one batch, independently of the largest cluster size.
    // Allocate before entering OpenMP so allocation failures reach the caller.
    std::vector<std::vector<float>> rotated_blocks(num_workers);
    for (auto& block : rotated_blocks) {
        block.resize(scratch_points * padded_dim_);
    }
    // Quantize each cluster
    index_detail::omp_for_dynamic(num_clusters_, num_workers, [&](size_t i, size_t thread) {
        quantize_cluster(
            clusters_[i],
            id_lists[i],
            data,
            centroids + (i * dim_),
            &rotated_centroids[i * padded_dim_],
            rotated_blocks[thread].data(),
            config
        );
    });

    initializer_->add_vectors(rotated_centroids.data(), num_workers);
    ready_ = true;
}

void IVF::allocate_storage(const std::vector<size_t>& cluster_sizes) {
    ready_ = false;
    num_removed_ = 0;
    free_storage();
    clusters_.clear();

    initializer_ = make_initializer();
    cluster_storage_.assign(num_clusters_, ClusterStorage(raw_reranking_));
    for (size_t i = 0; i < num_clusters_; ++i) {
        resize_storage(cluster_storage_[i], cluster_sizes[i]);
    }

    ip_func_ = select_excode_ipfunc(ex_bits_);
}

// Each cluster view points into that cluster's own independent storage.
void IVF::create_cluster_views(const std::vector<size_t>& cluster_sizes) {
    clusters_.reserve(num_clusters_);
    for (size_t i = 0; i < num_clusters_; ++i) {
        ClusterStorage& cs = cluster_storage_[i];
        char* rerank_data = has_rerank_data() ? cs.rerank_data() : nullptr;
        clusters_.emplace_back(cluster_sizes[i], cs.batch_data(), rerank_data, cs.ids());
    }
}

void IVF::quantize_cluster(
    Cluster& cluster,
    const std::vector<PID>& point_ids,
    const float* data,
    const float* centroid,
    float* rotated_centroid,
    float* rotated_data,
    const quant::RabitqConfig& config
) {
    size_t num_points = point_ids.size();
    assert(cluster.num() == num_points);
    std::copy(point_ids.begin(), point_ids.end(), cluster.ids());
    rotator_->rotate(centroid, rotated_centroid);

    char* batch_data = cluster.batch_data();
    char* ex_data = cluster.ex_data();
    for (size_t i = 0; i < num_points; i += fastscan::kBatchSize) {
        size_t batch_size = std::min(fastscan::kBatchSize, num_points - i);
        for (size_t j = 0; j < batch_size; ++j) {
            const float* vector = data + (point_ids[i + j] * dim_);
            rotator_->rotate(vector, rotated_data + (j * padded_dim_));
            if (raw_reranking_) {
                std::memcpy(
                    cluster.ex_data() + ((i + j) * rerank_vector_bytes()),
                    vector,
                    rerank_vector_bytes()
                );
            }
        }

        quant::quantize_split_batch(
            rotated_data,
            rotated_centroid,
            batch_size,
            padded_dim_,
            ex_bits_,
            batch_data,
            ex_data,
            metric_type_,
            config
        );

        batch_data += batch_bytes();
        if (ex_bits_ > 0) {
            ex_data += ExDataMap<float>::data_bytes(padded_dim_, ex_bits_) * batch_size;
        }
    }
}

void IVF::add(
    const float* data,
    size_t num_points,
    const PID* cluster_ids,
    bool fast_quantization,
    size_t num_threads
) {
    if (!ready_) {
        throw std::logic_error(
            "IVF index must be trained, constructed or loaded before add"
        );
    }
    if (num_points == 0) {
        return;
    }
    if (data == nullptr) {
        throw std::invalid_argument("IVF add data must not be null");
    }
    if (num_points > buffer::kSearchBufferMaxPointCount - max_elements_) {
        throw std::invalid_argument("IVF point count exceeds the supported ID range");
    }
    const size_t new_num_points = max_elements_ + num_points;
    if (exceeds_storage(new_num_points)) {
        throw std::invalid_argument("IVF configuration exceeds addressable storage");
    }
    check_coordinates(data, num_points * dim_, dim_, "IVF add data");

    // Assign every new point to a cluster
    std::vector<PID> assigned(num_points);
    if (cluster_ids != nullptr) {
        for (size_t i = 0; i < num_points; ++i) {
            if (cluster_ids[i] >= num_clusters_) {
                throw std::invalid_argument("IVF cluster ID is out of range");
            }
            assigned[i] = cluster_ids[i];
        }
    } else {
        index_detail::parallel_for(
            0, num_points, num_threads, [&](size_t row, size_t /*thread*/) {
                std::vector<float> rotated(padded_dim_);
                std::vector<AnnCandidate<float>> nearest(1);
                rotator_->rotate(data + (row * dim_), rotated.data());
                initializer_->centroids_distances(rotated.data(), 1, nearest);
                assigned[row] = nearest[0].id;
            }
        );
    }

    struct ClusterAddition {
        PID cluster_id;
        std::vector<PID> rows;
        size_t new_size;
        std::optional<ClusterStorage> staged{};
        ByteStorage tail{};  // Repacked tail for in-place growth.
    };
    std::vector<ClusterAddition> additions;
    {
        // Group only incoming rows, preserving their order within each cluster.
        // A small append must not allocate or scan num_clusters_ entries.
        std::unordered_map<PID, size_t> positions;
        for (size_t i = 0; i < num_points; ++i) {
            const auto [position, inserted] =
                positions.try_emplace(assigned[i], additions.size());
            if (inserted) {
                additions.push_back({assigned[i], {}, clusters_[assigned[i]].num()});
            }
            auto& addition = additions[position->second];
            addition.rows.push_back(static_cast<PID>(i));
            ++addition.new_size;
        }
    }

    const size_t num_workers = std::min(resolve_num_threads(num_threads), additions.size());
    static_cast<void>(num_workers);  // Consumed by the OpenMP clause below.
    quant::RabitqConfig config;
    if (fast_quantization) {
        config = quant::faster_config(padded_dim_, ex_bits_ + 1);
    }

    const PID first_pid = static_cast<PID>(max_elements_);

    // Rebuild clusters without reserved capacity on the side; the rest grow in place
    for (auto& addition : additions) {
        const PID cluster_id = addition.cluster_id;
        if (has_capacity(cluster_storage_[cluster_id], addition.new_size)) {
            const size_t full_batches = clusters_[cluster_id].num() / fastscan::kBatchSize;
            addition.tail.resize(
                batch_data_bytes(addition.new_size) - (full_batches * batch_bytes())
            );
        } else {
            addition.staged.emplace(raw_reranking_);
            resize_storage(*addition.staged, addition.new_size);
        }
    }
    // Growing within capacity cannot reallocate or throw, so old cluster views stay valid
    for (const auto& addition : additions) {
        if (!addition.staged) {
            resize_storage(cluster_storage_[addition.cluster_id], addition.new_size);
        }
    }

    // An exception must not leave an OpenMP region, so keep the first one and rethrow it.
    index_detail::ParallelException failure;
#pragma omp parallel for schedule(dynamic) num_threads(num_workers)
    for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(additions.size());
         ++index) {
        auto& addition = additions[static_cast<size_t>(index)];
        const PID cluster_id = addition.cluster_id;
        try {
            ClusterStorage& cs =
                addition.staged ? *addition.staged : cluster_storage_[cluster_id];
            const size_t full_batches = clusters_[cluster_id].num() / fastscan::kBatchSize;
            char* tail = addition.staged ? cs.batch_data() + (full_batches * batch_bytes())
                                         : reinterpret_cast<char*>(addition.tail.data());
            extend_cluster(
                clusters_[cluster_id],
                addition.rows,
                data,
                initializer_->centroid(cluster_id),
                first_pid,
                cs.batch_data(),
                has_rerank_data() ? cs.rerank_data() : nullptr,
                cs.ids(),
                tail,
                config
            );
        } catch (...) { failure.capture(); }
    }
    if (failure.failed()) {
        // In-place clusters were only written past their old size; shrinking drops that
        for (const auto& addition : additions) {
            if (!addition.staged) {
                const PID cluster_id = addition.cluster_id;
                resize_storage(cluster_storage_[cluster_id], clusters_[cluster_id].num());
            }
        }
        failure.rethrow();
    }

    // Commit. Nothing below can throw.
    for (auto& addition : additions) {
        const PID cluster_id = addition.cluster_id;
        auto& cs = cluster_storage_[cluster_id];
        if (addition.staged) {
            cs = std::move(*addition.staged);
        } else {
            const size_t full_batches = clusters_[cluster_id].num() / fastscan::kBatchSize;
            std::copy_n(
                addition.tail.data(),
                addition.tail.size(),
                cs.batch_storage.data() + (full_batches * batch_bytes())
            );
        }
        clusters_[cluster_id].reset(
            addition.new_size,
            cs.batch_data(),
            has_rerank_data() ? cs.rerank_data() : nullptr,
            cs.ids()
        );
    }
    max_elements_ = new_num_points;
}

// Copy one cluster into its grown storage and append new points to it.
//
// Full batches, rerank data and IDs are copied unchanged, unless they are already in place.
// The old partial tail batch is unpacked and repacked together with the new points into
// `tail_dst`, so the result is byte-identical to quantizing the same points in the same
// order in one construct call.
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
    const size_t old_size = old_cluster.num();
    const size_t rerank_bytes = rerank_vector_bytes();
    const size_t cols = padded_dim_ / 8;

    const size_t full_batches = old_size / fastscan::kBatchSize;
    const size_t old_tail_size = old_size - (full_batches * fastscan::kBatchSize);
    const size_t tail_size = old_tail_size + rows.size();

    if (batch_dst != old_cluster.batch_data()) {
        std::copy_n(old_cluster.batch_data(), full_batches * batch_bytes(), batch_dst);
    }
    if (rerank_dst != old_cluster.ex_data()) {
        std::copy_n(old_cluster.ex_data(), old_size * rerank_bytes, rerank_dst);
    }
    if (ids_dst != old_cluster.ids()) {
        std::copy_n(old_cluster.ids(), old_size, ids_dst);
    }

    // Codes and factors of every point in the tail, old points first
    std::vector<uint8_t> codes(tail_size * cols);
    std::vector<float> f_add(tail_size);
    std::vector<float> f_rescale(tail_size);
    std::vector<float> f_error(tail_size);
    if (old_tail_size > 0) {
        ConstBatchDataMap<float> old_tail(
            old_cluster.batch_data() + (full_batches * batch_bytes()), padded_dim_
        );
        fastscan::unpack_codes(
            padded_dim_, old_tail.bin_code(), old_tail_size, codes.data()
        );
        old_tail.f_add().copy_to(f_add.data(), old_tail_size);
        old_tail.f_rescale().copy_to(f_rescale.data(), old_tail_size);
        old_tail.f_error().copy_to(f_error.data(), old_tail_size);
    }

    std::vector<float> rotated(padded_dim_);
    for (size_t j = 0; j < rows.size(); ++j) {
        const float* vector = data + (static_cast<size_t>(rows[j]) * dim_);
        const size_t lane = old_tail_size + j;
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
        char* rerank = rerank_dst + ((old_size + j) * rerank_bytes);
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
        ids_dst[old_size + j] = first_pid + rows[j];
    }

    // Repack the tail into batches
    char* batch = tail_dst;
    for (size_t start = 0; start < tail_size; start += fastscan::kBatchSize) {
        const size_t batch_size = std::min(fastscan::kBatchSize, tail_size - start);
        BatchDataMap<float> cur_batch(batch, padded_dim_);
        fastscan::pack_codes(
            padded_dim_, codes.data() + (start * cols), batch_size, cur_batch.bin_code()
        );
        for (size_t i = 0; i < batch_size; ++i) {
            cur_batch.f_add()[i] = f_add[start + i];
            cur_batch.f_rescale()[i] = f_rescale[start + i];
            cur_batch.f_error()[i] = f_error[start + i];
        }
        batch += batch_bytes();
    }
}

void IVF::reserve(PID cluster_id, size_t num_points) {
    if (!ready_) {
        throw std::logic_error("IVF index must be constructed or loaded before reserve");
    }
    if (cluster_id >= num_clusters_) {
        throw std::invalid_argument("IVF cluster ID is out of range");
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
    const size_t old_size = old_cluster.num();
    ClusterStorage grown(raw_reranking_);
    reserve_storage(grown, num_points);
    resize_storage(grown, old_size);
    std::copy_n(old_cluster.batch_data(), batch_data_bytes(old_size), grown.batch_data());
    if (has_rerank_data()) {
        std::copy_n(
            old_cluster.ex_data(), rerank_data_bytes(old_size), grown.rerank_data()
        );
    }
    std::copy_n(old_cluster.ids(), old_size, grown.ids());

    // Commit. Only this cluster's storage moved, so only its view changes.
    cs = std::move(grown);
    clusters_[cluster_id].reset(
        old_size, cs.batch_data(), has_rerank_data() ? cs.rerank_data() : nullptr, cs.ids()
    );
}

size_t IVF::remove(const PID* ids_to_remove, size_t num_points) {
    if (!ready_) {
        throw std::logic_error("IVF index must be constructed or loaded before remove");
    }
    if (num_points == 0) {
        return 0;
    }
    if (ids_to_remove == nullptr) {
        throw std::invalid_argument("IVF remove IDs must not be null");
    }

    // Validate every ID before changing anything
    std::vector<bool> marked(max_elements_, false);
    for (size_t i = 0; i < num_points; ++i) {
        if (ids_to_remove[i] >= max_elements_) {
            throw std::invalid_argument("IVF remove ID is out of range");
        }
        marked[ids_to_remove[i]] = true;
    }

    // A point whose f_add is +inf has +inf estimated and lower-bound distances in every
    // scan path, so it never enters the result buffer or triggers reranking.
    size_t removed = 0;
    for (const auto& cluster : clusters_) {
        const PID* cluster_ids = cluster.ids();
        for (size_t pos = 0; pos < cluster.num(); ++pos) {
            if (!marked[cluster_ids[pos]]) {
                continue;
            }
            BatchDataMap<float> cur_batch(
                cluster.batch_data() + ((pos / fastscan::kBatchSize) * batch_bytes()),
                padded_dim_
            );
            const size_t lane = pos % fastscan::kBatchSize;
            if (static_cast<float>(cur_batch.f_add()[lane]) != kRemovedFAdd) {
                cur_batch.f_add()[lane] = kRemovedFAdd;
                ++removed;
            }
        }
    }
    num_removed_ += removed;
    return removed;
}

void IVF::save(const char* filename) const {
    if (!ready_) {
        throw std::logic_error("IVF index must be constructed or loaded before save");
    }
    if (max_elements_ == 0) {
        throw std::logic_error("IVF index has no points to save; add points first");
    }
    if (filename == nullptr || filename[0] == '\0') {
        throw std::invalid_argument("IVF save filename must not be empty");
    }

    // An HNSW initializer writes its sidecar next to the path it is given.
    const auto write = [this](const std::string& path) { write_file(path.c_str()); };
    if (initializer_type_ == InitializerType::HNSW) {
        index_detail::save_replacing(filename, {detail::kHnswSidecarSuffix}, write);
    } else {
        index_detail::save_replacing(filename, {}, write);
    }
}

void IVF::write_file(const char* filename) const {
    std::ofstream output(io_impl::filesystem_path(filename), std::ios::binary);
    if (!output.is_open()) {
        throw std::runtime_error("IVF cannot open index file for writing");
    }
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

    // Save metadata
    output.write(reinterpret_cast<const char*>(&max_elements_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&dim_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&num_clusters_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&ex_bits_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&rotator_type_), sizeof(rotator_type_));
    output.write(reinterpret_cast<const char*>(&metric_type_), sizeof(metric_type_));

    // Save the number of points in each cluster
    std::vector<size_t> cluster_sizes;
    cluster_sizes.reserve(num_clusters_);
    for (const auto& cluster : clusters_) {
        cluster_sizes.push_back(cluster.num());
    }
    output.write(
        reinterpret_cast<const char*>(cluster_sizes.data()),
        static_cast<std::streamsize>(sizeof(size_t) * num_clusters_)
    );

    // Save rotator
    rotator_->save(output);

    // Save data: initializer payload, then all batch data, all ex data and all IDs,
    // each in cluster order
    initializer_->save(output, filename);
    for (size_t i = 0; i < num_clusters_; ++i) {
        output.write(
            cluster_storage_[i].batch_data(),
            static_cast<std::streamsize>(batch_data_bytes(cluster_sizes[i]))
        );
    }
    if (has_rerank_data()) {
        for (size_t i = 0; i < num_clusters_; ++i) {
            output.write(
                cluster_storage_[i].rerank_data(),
                static_cast<std::streamsize>(rerank_data_bytes(cluster_sizes[i]))
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
    std::ifstream input(io_impl::filesystem_path(filename), std::ios::binary);
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

    // Load metadata
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
        loaded.initializer_type_ = loaded.num_clusters_ < kHistoricalHnswMinClusters
                                       ? InitializerType::Flat
                                       : InitializerType::HNSW;
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
            div_round_up(loaded.num_clusters_, fastscan::kBatchSize), loaded.batch_bytes()
        );
    }

    // Load the number of points in each cluster
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
        consume(div_round_up(size, fastscan::kBatchSize), loaded.batch_bytes());
    }
    if (total != loaded.max_elements_) {
        throw std::runtime_error("Invalid cluster counts in IVF index file");
    }
    loaded.rotator_ = rotator_impl::make_rotator_for_loading(
        loaded.dim_, loaded.rotator_type_, loaded.padded_dim_
    );

    // Load rotator
    loaded.rotator_->load(input);

    // Load data: initializer payload, then all batch data, all ex data and all IDs,
    // each in cluster order
    loaded.allocate_storage(cluster_sizes);
    loaded.initializer_->load(input, filename);
    for (size_t i = 0; i < loaded.num_clusters_; ++i) {
        input.read(
            loaded.cluster_storage_[i].batch_data(),
            static_cast<std::streamsize>(loaded.batch_data_bytes(cluster_sizes[i]))
        );
    }
    if (loaded.has_rerank_data()) {
        for (size_t i = 0; i < loaded.num_clusters_; ++i) {
            input.read(
                loaded.cluster_storage_[i].rerank_data(),
                static_cast<std::streamsize>(loaded.rerank_data_bytes(cluster_sizes[i]))
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

    loaded.create_cluster_views(cluster_sizes);
    // Removal markers are already persisted, including in historical files.
    // Count only logical rows: FastScan padding lanes are not stored points.
    for (const auto& cluster : loaded.clusters_) {
        for (size_t first = 0; first < cluster.num(); first += fastscan::kBatchSize) {
            ConstBatchDataMap<float> batch(
                cluster.batch_data() +
                    (first / fastscan::kBatchSize) * loaded.batch_bytes(),
                loaded.padded_dim_
            );
            const size_t batch_size = std::min(fastscan::kBatchSize, cluster.num() - first);
            for (size_t lane = 0; lane < batch_size; ++lane) {
                loaded.num_removed_ += batch.f_add()[lane] == kRemovedFAdd;
            }
        }
    }
    if (remaining != 0) {
        throw std::runtime_error("IVF index file has unexpected trailing data");
    }
    loaded.ready_ = true;

    input.close();
    swap(loaded);
}

void IVF::search(
    const float* query, size_t k, size_t nprobe, PID* result_ids, float* distances
) const {
    search(query, k, nprobe, result_ids, distances, !raw_reranking_ && ex_bits_ > 2);
}

void IVF::search(
    const float* __restrict__ query,
    size_t k,
    size_t nprobe,
    PID* __restrict__ result_ids,
    bool use_hacc
) const {
    search(query, k, nprobe, result_ids, nullptr, use_hacc);
}

void IVF::search(
    const float* __restrict__ query,
    size_t k,
    size_t nprobe,
    PID* __restrict__ result_ids,
    float* __restrict__ distances,
    bool use_hacc
) const {
    validate_search(query, k, nprobe, result_ids);
    check_coordinates(query, dim_, dim_, "IVF search query");

    nprobe = std::min(nprobe, num_clusters_);  // Probe at most every cluster
    std::vector<float> rotated_query(padded_dim_);
    rotator_->rotate(query, rotated_query.data());

    // Find the nprobe nearest centroids
    std::vector<AnnCandidate<float>> centroid_candidates(nprobe);
    initializer_->centroids_distances(rotated_query.data(), nprobe, centroid_candidates);

    buffer::SearchBuffer result_pool(k);

    SplitBatchQuery<float> query_wrapper(
        rotated_query.data(), padded_dim_, ex_bits_, metric_type_, use_hacc
    );

    for (size_t i = 0; i < nprobe; ++i) {
        PID cluster_id = centroid_candidates[i].id;
        float dist = centroid_candidates[i].distance;
        const Cluster& cluster = clusters_[cluster_id];

        if (metric_type_ == METRIC_L2) {
            query_wrapper.set_g_add(dist);
        } else {
            const float residual_norm = std::sqrt(euclidean_sqr(
                rotated_query.data(), initializer_->centroid(cluster_id), padded_dim_
            ));
            auto query_centroid_ip = dot_product<float>(
                rotated_query.data(), initializer_->centroid(cluster_id), padded_dim_
            );
            query_wrapper.set_g_add(residual_norm, query_centroid_ip);
        }
        search_cluster(cluster, query_wrapper, result_pool, use_hacc, query);
    }

    const size_t found = result_pool.size();
    if (distances != nullptr) {
        result_pool.copy_results(result_ids, distances);
        std::fill(distances + found, distances + k, std::numeric_limits<float>::infinity());
    } else {
        result_pool.copy_results(result_ids);
    }
    std::fill(result_ids + found, result_ids + k, kPidMax);
}

void IVF::search_batch(
    const float* queries,
    size_t num_queries,
    size_t k,
    size_t nprobe,
    PID* result_ids,
    float* distances,
    std::optional<bool> use_hacc,
    size_t num_threads
) const {
    if (num_queries == 0) {
        return;
    }
    validate_search(queries, k, nprobe, result_ids);
    const size_t max_values = std::numeric_limits<size_t>::max() / sizeof(float);
    if (num_queries > max_values / dim_ || num_queries > max_values / k) {
        throw std::length_error("IVF query batch is too large");
    }
    check_coordinates(queries, num_queries * dim_, dim_, "IVF search queries");
    nprobe = std::min(nprobe, num_clusters_);
    const bool hacc = use_hacc.value_or(!raw_reranking_ && ex_bits_ > 2);
    const size_t num_workers =
        num_queries == 1 || num_threads == 1
            ? 1
            : std::min(resolve_num_threads(num_threads), num_queries);
    if (num_workers == 1) {
        for (size_t i = 0; i < num_queries; ++i) {
            search(
                queries + i * dim_,
                k,
                nprobe,
                result_ids + i * k,
                distances == nullptr ? nullptr : distances + i * k,
                hacc
            );
        }
        return;
    }
    index_detail::omp_for_dynamic(
        num_queries, num_workers, [&](size_t i, size_t /*thread*/) {
            search(
                queries + i * dim_,
                k,
                nprobe,
                result_ids + i * k,
                distances == nullptr ? nullptr : distances + i * k,
                hacc
            );
        }
    );
}

inline void IVF::validate_search(
    const float* query, size_t k, size_t nprobe, const PID* result_ids
) const {
    if (!ready_) {
        throw std::logic_error(
            "IVF index must be trained, constructed or loaded before search"
        );
    }
    if (query == nullptr) {
        throw std::invalid_argument("IVF search query must not be null");
    }
    if (result_ids == nullptr) {
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
    const Cluster& cluster,
    const SplitBatchQuery<float>& query_wrapper,
    buffer::SearchBuffer<float>& result_pool,
    bool use_hacc,
    const float* query
) const {
    const size_t full_batches = cluster.num() / fastscan::kBatchSize;
    const size_t tail_size = cluster.num() - (full_batches * fastscan::kBatchSize);

    const char* batch_data = cluster.batch_data();
    const char* rerank_data = cluster.ex_data();
    const PID* ids = cluster.ids();

    // Compute distances batch by batch
    for (size_t i = 0; i < full_batches; ++i) {
        scan_one_batch(
            batch_data,
            rerank_data,
            ids,
            query_wrapper,
            result_pool,
            fastscan::kBatchSize,
            use_hacc,
            query
        );

        batch_data += batch_bytes();
        if (has_rerank_data()) {
            rerank_data += rerank_vector_bytes() * fastscan::kBatchSize;
        }
        ids += fastscan::kBatchSize;
    }

    if (tail_size > 0) {
        // Scan the partial last batch
        scan_one_batch(
            batch_data,
            rerank_data,
            ids,
            query_wrapper,
            result_pool,
            tail_size,
            use_hacc,
            query
        );
    }
}

inline void IVF::scan_one_batch(
    const char* batch_data,
    const char* rerank_data,
    const PID* ids,
    const SplitBatchQuery<float>& query_wrapper,
    buffer::SearchBuffer<float>& result_pool,
    size_t batch_size,
    bool use_hacc,
    const float* query
) const {
    std::array<float, fastscan::kBatchSize> estimated_distances;
    std::array<float, fastscan::kBatchSize> lower_bounds;
    std::array<float, fastscan::kBatchSize> ip_x0_qr;  // inner product of the 1st bit

    split_batch_estdist(
        batch_data,
        query_wrapper,
        padded_dim_,
        estimated_distances.data(),
        lower_bounds.data(),
        ip_x0_qr.data(),
        use_hacc
    );

    float dist_k = result_pool.top_dist();

    // Without reranking data, return the one-bit estimates directly.
    if (!has_rerank_data()) {
        detail::insert_candidates(result_pool, ids, estimated_distances.data(), batch_size);
        return;
    }

    // Refine candidates passing the probabilistic one-bit filter. Quantized storage
    // reuses ip_x0_qr with the extra bits; raw storage scores the original vector.
    for (size_t i = 0; i < batch_size; ++i) {
        float lower_bound = lower_bounds[i];
        if (lower_bound < dist_k) {
            PID id = ids[i];
            float refined_distance;
            if (raw_reranking_) {
                const auto* vector = reinterpret_cast<const float*>(rerank_data);
                refined_distance = metric_type_ == METRIC_L2
                                       ? euclidean_sqr(query, vector, dim_)
                                       : dot_product_dis(query, vector, dim_);
            } else {
                refined_distance = split_distance_boosting(
                    rerank_data, ip_func_, query_wrapper, padded_dim_, ex_bits_, ip_x0_qr[i]
                );
            }
            result_pool.insert(id, refined_distance);
            dist_k = result_pool.top_dist();
        }
        rerank_data += rerank_vector_bytes();
    }
}

}  // namespace rabitqlib::ivf
