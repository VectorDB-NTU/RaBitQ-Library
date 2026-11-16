#include "rabitqlib/index/hnsw_index.hpp"

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

#include "index_facade.hpp"
#include "rabitqlib/clustering/rabitqkmeans.hpp"
#include "rabitqlib/index/detail/buffer.hpp"
#include "rabitqlib/index/hnsw/hnsw.hpp"
#include "rabitqlib/index/training.hpp"
#include "rabitqlib/utils/input_checks.hpp"

namespace rabitqlib {

namespace {
using rabitqkmeans::RaBitQKMeansParameters;

// What an empty, untrained file saved by 0.7 trains, as a new index does by default.
constexpr size_t kDefaultNumClusters = 16;

// Grow the capacity to fit num_points more points, at least doubling it. The points are
// checked first, so that invalid input leaves the capacity unchanged. A count beyond the
// supported ID range is left for add to reject.
void grow_capacity(hnsw::HierarchicalNSW& index, const float* data, size_t num_points) {
    const size_t capacity = index.max_elements();
    const size_t stored = index.num_stored_points();
    if (num_points <= capacity - stored ||
        num_points > buffer::kSearchBufferMaxPointCount - stored) {
        return;
    }
    const size_t dim = index.dimension();
    check_coordinates(data, num_points * dim, dim, "HNSW add data");
    const size_t doubled =
        std::min(capacity * 2, static_cast<size_t>(buffer::kSearchBufferMaxPointCount));
    index.resize(std::max(stored + num_points, doubled));
}
}  // namespace

HnswIndex::HnswIndex(size_t dim, const HnswOptions& options)
    : num_clusters_(options.num_clusters) {
    if (options.num_clusters == 0 ||
        options.num_clusters > buffer::kSearchBufferMaxPointCount) {
        throw std::invalid_argument(
            "num_clusters must be in [1, " +
            std::to_string(buffer::kSearchBufferMaxPointCount) + "]"
        );
    }
    const uint32_t seed = options.seed.has_value() ? *options.seed : std::random_device{}();
    // A capacity of one point; add grows it and reserve preallocates.
    index_ = std::make_unique<hnsw::HierarchicalNSW>(
        1, dim, options.nbits, options.M, options.ef_construction, seed, options.metric
    );
    cp_ = std::make_unique<RaBitQKMeansParameters>(
        default_training_parameters(options.metric, seed)
    );
}

HnswIndex::HnswIndex(
    std::unique_ptr<hnsw::HierarchicalNSW> index,
    std::unique_ptr<RaBitQKMeansParameters> cp,
    size_t num_clusters
)
    : index_(std::move(index)), cp_(std::move(cp)), num_clusters_(num_clusters) {}

HnswIndex::HnswIndex(HnswIndex&& other) noexcept = default;
HnswIndex& HnswIndex::operator=(HnswIndex&& other) noexcept = default;
HnswIndex::~HnswIndex() = default;

HnswIndex HnswIndex::load(const char* filename) {
    auto index = std::make_unique<hnsw::HierarchicalNSW>();
    index->load(filename);
    const size_t num_clusters =
        index->is_trained() ? index->num_clusters() : kDefaultNumClusters;
    // Files from releases up to 0.8 do not record the seed.
    const std::optional<uint64_t> seed = index->seed();
    auto cp = std::make_unique<RaBitQKMeansParameters>(default_training_parameters(
        index->metric_type(),
        seed.has_value() ? static_cast<uint32_t>(*seed) : RaBitQKMeansParameters{}.seed
    ));
    return HnswIndex(std::move(index), std::move(cp), num_clusters);
}

void HnswIndex::save(const char* filename) const {
    check_trained("save");
    if (index_->num_stored_points() == 0) {
        throw std::logic_error("HnswIndex has no points to save; add points first");
    }
    index_->save(filename);
}

RaBitQKMeansParameters& HnswIndex::cp() { return *cp_; }

const RaBitQKMeansParameters& HnswIndex::cp() const { return *cp_; }

void HnswIndex::train(
    const float* data, size_t num_points, std::optional<size_t> num_threads
) {
    check_trainable();
    index_->train(
        num_clusters(), num_points, data, index_facade::with_num_threads(*cp_, num_threads)
    );
}

void HnswIndex::set_centroids(const float* centroids) {
    check_trainable();
    index_->set_centroids(num_clusters(), centroids);
}

Id HnswIndex::add(const float* data, size_t num_points, const AddOptions& options) {
    // Only a loaded file lacks the seed that recreates the rotation and graph levels.
    const bool trains = !index_->is_trained() && options.cluster_ids == nullptr &&
                        index_->seed().has_value() && num_points > 0;
    if (!trains) {
        check_trained("add");
    }
    if (num_points != 0 && data == nullptr) {
        throw std::invalid_argument("HnswIndex add data must not be null");
    }
    const auto first = static_cast<Id>(index_->num_stored_points());
    const size_t num_threads = options.num_threads.value_or(0);
    if (!trains) {
        // Checked before the capacity grows, as the points are.
        if (options.cluster_ids != nullptr) {
            for (size_t i = 0; i < num_points; ++i) {
                if (options.cluster_ids[i] >= index_->num_clusters()) {
                    throw std::invalid_argument("HNSW cluster ID is out of range");
                }
            }
        }
        grow_capacity(*index_, data, num_points);
        index_->add(
            data, num_points, options.cluster_ids, options.fast_quantization, num_threads
        );
        return first;
    }
    // Train and fill a new index with the same settings, seed and capacity, then move it
    // in, so that a failure leaves this index untrained and empty. Fewer points than
    // num_clusters() train one centroid per point.
    auto staged = std::make_unique<hnsw::HierarchicalNSW>(
        index_->max_elements(),
        index_->dimension(),
        index_->nbits(),
        index_->M(),
        index_->ef_construction(),
        index_->seed().value(),
        index_->metric_type()
    );
    staged->train(
        std::min(num_clusters_, num_points),
        num_points,
        data,
        index_facade::with_num_threads(*cp_, options.num_threads)
    );
    grow_capacity(*staged, data, num_points);
    staged->add(data, num_points, nullptr, options.fast_quantization, num_threads);
    index_ = std::move(staged);
    return first;
}

size_t HnswIndex::remove(const Id* ids, size_t num_ids) {
    check_trained("remove");
    if (num_ids != 0 && ids == nullptr) {
        throw std::invalid_argument("HnswIndex remove ids must not be null");
    }
    const size_t num_stored = index_->num_stored_points();
    std::vector<PID> labels(num_ids);
    for (size_t i = 0; i < num_ids; ++i) {
        if (ids[i] < 0 || static_cast<uint64_t>(ids[i]) >= num_stored) {
            throw std::invalid_argument("ids must be in [0, num_stored_points)");
        }
        labels[i] = static_cast<PID>(ids[i]);
    }
    return index_->remove(labels.data(), labels.size());
}

void HnswIndex::reserve(size_t num_points) {
    if (num_points > index_->max_elements()) {
        index_->resize(num_points);
    }
}

void HnswIndex::search(
    const float* queries,
    size_t num_queries,
    size_t k,
    Id* ids,
    float* distances,
    const GraphSearchOptions& options
) const {
    check_trained("search");
    index_facade::check_k(k);
    index_facade::check_breadth(options.ef_search, "ef_search");
    if (num_queries == 0) {
        return;
    }
    index_facade::check_outputs("HnswIndex", num_queries, k, ids);
    // Concurrent searches are safe; the graph search only reads the index.
    const auto results = index_->search(
        queries,
        num_queries,
        k,
        options.ef_search.value_or(std::max<size_t>(k, 64)),
        options.num_threads
    );
    for (size_t i = 0; i < num_queries; ++i) {
        const auto& found = results[i];
        for (size_t j = 0; j < k; ++j) {
            const bool filled = j < found.size();
            ids[(i * k) + j] = filled ? static_cast<Id>(found[j].second) : kNoId;
            if (distances != nullptr) {
                distances[(i * k) + j] =
                    filled ? found[j].first : std::numeric_limits<float>::infinity();
            }
        }
    }
}

size_t HnswIndex::dim() const { return index_->dimension(); }

size_t HnswIndex::num_clusters() const {
    return index_->is_trained() ? index_->num_clusters() : num_clusters_;
}

size_t HnswIndex::nbits() const { return index_->nbits(); }

MetricType HnswIndex::metric() const { return index_->metric_type(); }

size_t HnswIndex::M() const { return index_->M(); }

size_t HnswIndex::ef_construction() const { return index_->ef_construction(); }

std::optional<uint64_t> HnswIndex::seed() const { return index_->seed(); }

size_t HnswIndex::capacity() const { return index_->max_elements(); }

size_t HnswIndex::num_points() const { return index_->num_points(); }

size_t HnswIndex::num_stored_points() const { return index_->num_stored_points(); }

bool HnswIndex::is_trained() const { return index_->is_trained(); }

void HnswIndex::check_trained(const char* operation) const {
    if (!index_->is_trained()) {
        throw std::logic_error(
            std::string("HnswIndex must be trained or loaded before ") + operation
        );
    }
}

void HnswIndex::check_trainable() const {
    if (index_->num_stored_points() != 0) {
        throw std::logic_error("HnswIndex cannot be trained once it stores points");
    }
}

}  // namespace rabitqlib
