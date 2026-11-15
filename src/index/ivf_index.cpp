#include "rabitqlib/index/ivf_index.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "rabitqlib/clustering/rabitqkmeans.hpp"
#include "rabitqlib/core/types.hpp"
#include "rabitqlib/index/detail/buffer.hpp"
#include "rabitqlib/index/ivf/ivf.hpp"
#include "rabitqlib/index/training.hpp"

namespace rabitqlib {

namespace {
using rabitqkmeans::RaBitQKMeansParameters;

// num_clusters / 16, rounded, and at least 1.
size_t default_nprobe(size_t num_clusters) {
    return std::max<size_t>(1, (num_clusters + 8) / 16);
}
}  // namespace

IvfIndex::IvfIndex(size_t dim, size_t num_clusters, const IvfOptions& options) {
    const uint32_t seed = options.seed.has_value() ? *options.seed : std::random_device{}();
    index_ = std::make_unique<ivf::IVF>(
        dim,
        num_clusters,
        options.nbits,
        options.metric,
        seed,
        RotatorType::FhtKacRotator,
        options.routing
    );
    cp_ = std::make_unique<RaBitQKMeansParameters>(
        default_training_parameters(options.metric, seed)
    );
}

IvfIndex::IvfIndex(
    std::unique_ptr<ivf::IVF> index, std::unique_ptr<RaBitQKMeansParameters> cp
)
    : index_(std::move(index)), cp_(std::move(cp)) {}

IvfIndex::IvfIndex(IvfIndex&& other) noexcept = default;
IvfIndex& IvfIndex::operator=(IvfIndex&& other) noexcept = default;
IvfIndex::~IvfIndex() = default;

IvfIndex IvfIndex::load(const char* filename) {
    auto index = std::make_unique<ivf::IVF>();
    index->load(filename);
    // Files from releases up to 0.8 do not record the seed.
    const auto seed = index->seed().value_or(RaBitQKMeansParameters{}.seed);
    auto cp = std::make_unique<RaBitQKMeansParameters>(
        default_training_parameters(index->metric_type(), seed)
    );
    return IvfIndex(std::move(index), std::move(cp));
}

void IvfIndex::save(const char* filename) const {
    check_trained("save");
    if (index_->num_stored_points() == 0) {
        throw std::logic_error("IvfIndex has no points to save; add points first");
    }
    index_->save(filename);
}

RaBitQKMeansParameters& IvfIndex::cp() { return *cp_; }

const RaBitQKMeansParameters& IvfIndex::cp() const { return *cp_; }

void IvfIndex::train(
    const float* data, size_t num_points, std::optional<size_t> num_threads
) {
    check_trainable();
    index_->train(num_points, data, training_parameters(num_threads));
}

void IvfIndex::set_centroids(const float* centroids, size_t num_threads) {
    check_trainable();
    index_->set_centroids(centroids, num_threads);
}

Id IvfIndex::add(const float* data, size_t num_points, const AddOptions& options) {
    const bool trains =
        !index_->is_trained() && options.cluster_ids == nullptr && num_points > 0;
    if (!trains) {
        check_trained("add");
    }
    const auto first = static_cast<Id>(index_->num_stored_points());
    const size_t num_threads = options.num_threads.value_or(0);
    if (!trains) {
        index_->add(
            data, num_points, options.cluster_ids, options.fast_quantization, num_threads
        );
        return first;
    }
    if (num_points < index_->num_clusters()) {
        throw std::invalid_argument(
            "the first add trains the index and needs at least num_clusters rows; "
            "expected at least " +
            std::to_string(index_->num_clusters()) + ", got shape (" +
            std::to_string(num_points) + ", " + std::to_string(index_->dimension()) + ")"
        );
    }
    // Train and fill a new index with the same settings and rotation seed, then move it
    // in, so that a failure leaves this index untrained and empty. Only a loaded index
    // lacks a seed, and it is trained.
    auto staged = std::make_unique<ivf::IVF>(
        index_->dimension(),
        index_->num_clusters(),
        index_->nbits(),
        index_->metric_type(),
        index_->seed().value(),
        RotatorType::FhtKacRotator,
        index_->initializer_type()
    );
    staged->train(num_points, data, training_parameters(options.num_threads));
    staged->add(data, num_points, nullptr, options.fast_quantization, num_threads);
    index_ = std::move(staged);
    return first;
}

size_t IvfIndex::remove(const Id* ids, size_t num_ids) {
    check_trained("remove");
    if (num_ids != 0 && ids == nullptr) {
        throw std::invalid_argument("IvfIndex remove ids must not be null");
    }
    const size_t num_stored = index_->num_stored_points();
    std::vector<PID> row_ids(num_ids);
    for (size_t i = 0; i < num_ids; ++i) {
        if (ids[i] < 0 || static_cast<uint64_t>(ids[i]) >= num_stored) {
            throw std::invalid_argument("ids must be in [0, num_stored_points)");
        }
        row_ids[i] = static_cast<PID>(ids[i]);
    }
    return index_->remove(row_ids.data(), row_ids.size());
}

void IvfIndex::reserve_cluster(size_t cluster_id, size_t num_points) {
    check_trained("reserve_cluster");
    if (cluster_id >= index_->num_clusters()) {
        throw std::invalid_argument("cluster_id must be in [0, num_clusters)");
    }
    index_->reserve(static_cast<PID>(cluster_id), num_points);
}

void IvfIndex::search(
    const float* queries,
    size_t num_queries,
    size_t k,
    Id* ids,
    float* distances,
    const IvfSearchOptions& options
) const {
    check_trained("search");
    if (k == 0 || k > buffer::kSearchBufferMaxPointCount) {
        throw std::invalid_argument("k must be between 1 and 2^31");
    }
    if (options.nprobe == size_t{0}) {
        throw std::invalid_argument(
            "nprobe must be at least 1; leave it unset for the default"
        );
    }
    if (num_queries == 0) {
        return;
    }
    if (ids == nullptr) {
        throw std::invalid_argument("IvfIndex search ids must not be null");
    }
    if (num_queries > std::numeric_limits<size_t>::max() / sizeof(Id) / k) {
        throw std::length_error("IvfIndex query batch is too large");
    }
    // The index writes 32-bit row numbers, which are widened afterwards.
    std::vector<PID> row_ids(num_queries * k);
    index_->search_batch(
        queries,
        num_queries,
        k,
        options.nprobe.value_or(default_nprobe(index_->num_clusters())),
        row_ids.data(),
        distances,
        options.high_accuracy,
        options.num_threads
    );
    std::transform(row_ids.begin(), row_ids.end(), ids, [](PID id) {
        return id == kPidMax ? kNoId : static_cast<Id>(id);
    });
}

size_t IvfIndex::dim() const { return index_->dimension(); }

size_t IvfIndex::num_clusters() const { return index_->num_clusters(); }

size_t IvfIndex::nbits() const { return index_->nbits(); }

MetricType IvfIndex::metric() const { return index_->metric_type(); }

ivf::Routing IvfIndex::routing() const { return index_->initializer_type(); }

std::optional<uint32_t> IvfIndex::seed() const { return index_->seed(); }

size_t IvfIndex::num_points() const { return index_->num_points(); }

size_t IvfIndex::num_stored_points() const { return index_->num_stored_points(); }

bool IvfIndex::is_trained() const { return index_->is_trained(); }

void IvfIndex::check_trained(const char* operation) const {
    if (!index_->is_trained()) {
        throw std::logic_error(
            std::string("IvfIndex must be trained or loaded before ") + operation
        );
    }
}

void IvfIndex::check_trainable() const {
    if (index_->num_stored_points() != 0) {
        throw std::logic_error("IvfIndex cannot be trained once it stores points");
    }
}

RaBitQKMeansParameters IvfIndex::training_parameters(
    std::optional<size_t> num_threads
) const {
    RaBitQKMeansParameters cp = *cp_;
    if (num_threads.has_value()) {
        cp.num_threads = static_cast<uint32_t>(
            std::min<size_t>(*num_threads, std::numeric_limits<int>::max())
        );
    }
    return cp;
}

}  // namespace rabitqlib
