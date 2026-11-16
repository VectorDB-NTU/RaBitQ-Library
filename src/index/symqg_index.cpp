#include "rabitqlib/index/symqg_index.hpp"

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

#include "cosine.hpp"
#include "index_facade.hpp"
#include "rabitqlib/core/rotator.hpp"
#include "rabitqlib/core/types.hpp"
#include "rabitqlib/fastscan/pack_codes.hpp"
#include "rabitqlib/index/symqg/qg.hpp"
#include "rabitqlib/index/symqg/qg_builder.hpp"
#include "rabitqlib/utils/input_checks.hpp"

namespace rabitqlib {

namespace {
// The graph stores raw vectors as 0 bits.
size_t graph_bits(size_t nbits) { return nbits == 32 ? 0 : nbits; }
}  // namespace

SymqgIndex::SymqgIndex(size_t dim, const SymqgOptions& options)
    : dim_(dim)
    , metric_(options.metric)
    , nbits_(options.nbits)
    , max_degree_(options.max_degree)
    , ef_construction_(options.ef_construction)
    , init_(options.init) {
    validate_metric_type(index_detail::stored_metric(options.metric));
    // Checked now; the graph is created only on add.
    if (dim < 64 || dim > rotator_impl::FhtKacRotator::kMaxDim) {
        throw std::invalid_argument("Unsupported dimension for FhtKacRotator");
    }
    if (options.max_degree == 0 || options.max_degree % fastscan::kBatchSize != 0) {
        throw std::invalid_argument("max_degree must be a positive multiple of 32");
    }
    if (options.nbits != 4 && options.nbits != 8 && options.nbits != 32) {
        throw std::invalid_argument("nbits must be 4, 8, or 32");
    }
    if (options.init != symqg::QGInitialization::PiPNN &&
        options.init != symqg::QGInitialization::Random) {
        throw std::invalid_argument("init must be PiPNN or Random");
    }
    if (options.ef_construction == 0 ||
        options.ef_construction > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("ef_construction must be positive and fit uint32");
    }
    seed_ = options.seed.has_value() ? *options.seed : std::random_device{}();
}

SymqgIndex::SymqgIndex(std::unique_ptr<symqg::QuantizedGraph<float>> graph)
    : dim_(graph->dimension())
    , metric_(graph->metric_type())
    , nbits_(graph->is_quantized() ? graph->quantization_bits() : 32)
    , max_degree_(graph->degree_bound())
    , ef_construction_(graph->ef_construction())
    , init_(graph->initialization())
    , seed_(graph->seed())
    , graph_(std::move(graph)) {}

SymqgIndex::SymqgIndex(SymqgIndex&& other) noexcept = default;
SymqgIndex& SymqgIndex::operator=(SymqgIndex&& other) noexcept = default;
SymqgIndex::~SymqgIndex() = default;

SymqgIndex SymqgIndex::load(const char* filename) {
    auto graph = std::make_unique<symqg::QuantizedGraph<float>>();
    graph->load(filename);
    return SymqgIndex(std::move(graph));
}

void SymqgIndex::save(const char* filename) const {
    check_built("save");
    graph_->save(filename);
}

void SymqgIndex::train(const float* data, size_t num_points) const {
    if (num_points != 0 && data == nullptr) {
        throw std::invalid_argument("SymqgIndex train data must not be null");
    }
    check_coordinates(data, num_points * dim_, dim_, "SymqgIndex train data");
}

Id SymqgIndex::add(const float* data, size_t num_points, size_t num_threads) {
    if (graph_ != nullptr) {
        throw std::logic_error(
            "SymqgIndex does not support incremental addition; create a new index and "
            "add all vectors at once"
        );
    }
    if (num_points != 0 && data == nullptr) {
        throw std::invalid_argument("SymqgIndex add data must not be null");
    }
    // Build a new graph and move it in, so that a failure leaves this index without one.
    // Only a loaded index lacks these settings, and it already has a graph.
    const uint32_t seed = seed_.value();
    auto graph = std::make_unique<symqg::QuantizedGraph<float>>(
        num_points,
        dim_,
        max_degree_,
        metric_,
        RotatorType::FhtKacRotator,
        graph_bits(nbits_),
        seed
    );
    symqg::QGBuilder builder(
        *graph,
        static_cast<uint32_t>(ef_construction_.value()),
        data,
        num_threads,
        init_.value(),
        seed
    );
    builder.build();
    graph_ = std::move(graph);
    return 0;
}

void SymqgIndex::search(
    const float* queries,
    size_t num_queries,
    size_t k,
    Id* ids,
    float* distances,
    const GraphSearchOptions& options
) const {
    check_built("search");
    index_facade::check_k(k);
    index_facade::check_breadth(options.ef_search, "ef_search");
    if (num_queries == 0) {
        return;
    }
    index_facade::check_outputs("SymqgIndex", num_queries, k, ids);
    // The graph writes 32-bit row numbers, which are widened afterwards, and always
    // writes distances.
    std::vector<PID> row_ids(num_queries * k);
    std::vector<float> unused_distances(distances == nullptr ? num_queries * k : 0);
    // Concurrent searches are safe; a search with its own breadth only reads the graph.
    graph_->search_batch_with_ef(
        queries,
        num_queries,
        static_cast<uint32_t>(k),
        row_ids.data(),
        distances != nullptr ? distances : unused_distances.data(),
        std::max<size_t>(options.ef_search.value_or(64), k),
        options.num_threads
    );
    std::transform(row_ids.begin(), row_ids.end(), ids, [](PID id) {
        return id == kPidMax ? kNoId : static_cast<Id>(id);
    });
}

size_t SymqgIndex::dim() const { return dim_; }

size_t SymqgIndex::nbits() const { return nbits_; }

MetricType SymqgIndex::metric() const { return metric_; }

size_t SymqgIndex::max_degree() const { return max_degree_; }

std::optional<size_t> SymqgIndex::ef_construction() const { return ef_construction_; }

std::optional<symqg::QGInitialization> SymqgIndex::init() const { return init_; }

std::optional<uint32_t> SymqgIndex::seed() const { return seed_; }

size_t SymqgIndex::num_points() const {
    return graph_ != nullptr ? graph_->num_points() : 0;
}

size_t SymqgIndex::num_stored_points() const {
    return graph_ != nullptr ? graph_->num_stored_points() : 0;
}

bool SymqgIndex::is_trained() const { return true; }

void SymqgIndex::check_built(const char* operation) const {
    if (graph_ == nullptr) {
        throw std::logic_error(
            std::string("SymqgIndex has no vectors; add them or load an index before ") +
            operation
        );
    }
}

}  // namespace rabitqlib
