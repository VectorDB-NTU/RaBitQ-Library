#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include "rabitqlib/core/metric.hpp"        // IWYU pragma: export
#include "rabitqlib/index/symqg/types.hpp"  // IWYU pragma: export
#include "rabitqlib/index/types.hpp"        // IWYU pragma: export

namespace rabitqlib {
namespace symqg {
template <typename T>
class QuantizedGraph;
}

// Settings of a new SymqgIndex; set the fields by name. The metric comes first, so a lone
// positional value must be a metric: {4} does not compile.
struct SymqgOptions {
    MetricType metric = METRIC_L2;  // METRIC_L2, METRIC_IP or METRIC_COSINE
    size_t nbits = 32;              // 32 keeps raw float32 vectors; 4 or 8 RaBitQ bits
    size_t max_degree = 32;         // graph degree bound, a positive multiple of 32
    size_t ef_construction = 200;   // construction search breadth, from 1 to 2^32 - 1
    symqg::QGInitialization init = symqg::QGInitialization::PiPNN;
    std::optional<uint32_t> seed;  // rotation and construction seed; no value draws one
};

/**
 * @brief A SymphonyQG index with the workflow of the Python SymqgIndex.
 *
 * Create it, then add every vector in one call: add builds the graph, and a built
 * graph cannot be extended. Search and save need a built or loaded index. There is
 * nothing to train; train only checks its input.
 *
 * IDs are the row numbers of the added data, from 0.
 *
 * Const members may run concurrently; a call that changes the index needs exclusive
 * access. Arguments outside their documented range throw std::invalid_argument, a call
 * in the wrong state std::logic_error, and load rabitqlib::FormatError for a damaged or
 * unsupported file. The index is movable and not copyable; a moved-from index may only be
 * destroyed or assigned to.
 */
class SymqgIndex {
   public:
    /**
     * @param dim Vector dimension, from 64 to 65536
     * @param options Metric, bits, graph settings and seed
     */
    explicit SymqgIndex(size_t dim, const SymqgOptions& options = {});

    SymqgIndex(SymqgIndex&& other) noexcept;
    SymqgIndex& operator=(SymqgIndex&& other) noexcept;
    SymqgIndex(const SymqgIndex&) = delete;
    SymqgIndex& operator=(const SymqgIndex&) = delete;
    ~SymqgIndex();

    // Load an index saved by any release; it is built, so add throws.
    static SymqgIndex load(const char* filename);

    // Requires a built or loaded index. Releases up to 0.8 cannot read the file.
    void save(const char* filename) const;

    /**
     * @brief Check that the vectors are finite and within the coordinate range, as add
     * does; the index does not change. The point count is not checked.
     *
     * @param data num_points * dim values
     */
    void train(const float* data, size_t num_points) const;

    /**
     * @brief Build the graph from all vectors, once.
     *
     * Uses the constructor's metric, bits, max_degree, ef_construction, init and seed.
     * If it throws, the index stays without a graph. A built or loaded index throws
     * std::logic_error; create a new index to rebuild.
     *
     * @param data num_points * dim values
     * @param num_points More than max_degree()
     * @param num_threads Worker count; 0 selects every available CPU, capped by
     * OMP_NUM_THREADS
     * @return ID of the first point, 0; the points have consecutive IDs
     */
    Id add(const float* data, size_t num_points, size_t num_threads = 0);

    /**
     * @brief Search a batch of queries.
     *
     * Slots without a result get kNoId and +inf. Distances are squared L2, 1 - dot or
     * 1 - cos according to the metric, exact for raw vectors and estimated with 4 or 8
     * bits; lower is better.
     *
     * @param queries num_queries * dim values
     * @param k Results per query, from 1 to 2^31; may exceed the number of points
     * @param ids num_queries * k results
     * @param distances num_queries * k results, or nullptr
     */
    void search(
        const float* queries,
        size_t num_queries,
        size_t k,
        Id* ids,
        float* distances,
        const GraphSearchOptions& options = {}
    ) const;

    [[nodiscard]] size_t dim() const;
    // 32 for raw float32 vectors, or 4 or 8.
    [[nodiscard]] size_t nbits() const;
    [[nodiscard]] MetricType metric() const;
    [[nodiscard]] size_t max_degree() const;
    // No value for an index loaded from a file of a release up to 0.8.
    [[nodiscard]] std::optional<size_t> ef_construction() const;
    [[nodiscard]] std::optional<symqg::QGInitialization> init() const;
    [[nodiscard]] std::optional<uint32_t> seed() const;
    // Searchable points: 0 until add.
    [[nodiscard]] size_t num_points() const;
    // Stored points, equal to num_points(); SymphonyQG has no removal.
    [[nodiscard]] size_t num_stored_points() const;
    // Always true: SymphonyQG needs no training.
    [[nodiscard]] bool is_trained() const;

   private:
    explicit SymqgIndex(std::unique_ptr<symqg::QuantizedGraph<float>> graph);

    void check_built(const char* operation) const;

    size_t dim_;
    MetricType metric_;
    size_t nbits_;
    size_t max_degree_;
    // Settings of the current graph or of add; unknown for a file of a release up to 0.8.
    std::optional<size_t> ef_construction_;
    std::optional<symqg::QGInitialization> init_;
    std::optional<uint32_t> seed_;
    std::unique_ptr<symqg::QuantizedGraph<float>> graph_;  // null until add or load
};

}  // namespace rabitqlib
