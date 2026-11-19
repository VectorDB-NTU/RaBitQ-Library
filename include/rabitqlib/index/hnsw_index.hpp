#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include "rabitqlib/core/metric.hpp"  // IWYU pragma: export
#include "rabitqlib/index/types.hpp"  // IWYU pragma: export

namespace rabitqlib {
namespace hnsw {
class HierarchicalNSW;
}
namespace rabitqkmeans {
struct RaBitQKMeansParameters;
}

// Settings of a new HnswIndex; set the fields by name. The metric comes first, so a lone
// positional value must be a metric: {4} does not compile.
struct HnswOptions {
    MetricType metric = METRIC_L2;  // METRIC_L2, METRIC_IP or METRIC_COSINE
    size_t nbits = 7;               // 1-9 quantization bits
    // Centroids from train, set_centroids or the first add, which caps the count at its
    // number of points. From 1 to 2^31.
    size_t num_clusters = 16;
    size_t M =
        16;  // graph connectivity, at least 2; the base layer keeps up to 2 * M edges
    size_t ef_construction = 200;  // construction search breadth
    std::optional<uint32_t> seed;  // graph and rotation seed; no value draws one
};

/**
 * @brief An HNSW-RaBitQ index with the workflow of the Python HnswIndex.
 *
 * Create it, then add vectors: the first add trains an untrained index on them. Call
 * train on a sample, or set_centroids, first to choose the training data or the
 * centroids. Search, remove and save need a trained index. Capacity grows on add, at
 * least doubling; reserve preallocates it.
 *
 * IDs are the row numbers the index assigns, consecutive from 0 in the order points are
 * added. Removed points keep their IDs.
 *
 * Const members may run concurrently; a call that changes the index needs exclusive
 * access. Arguments outside their documented range throw std::invalid_argument, a call
 * in the wrong state std::logic_error, and load rabitqlib::FormatError for a damaged or
 * unsupported file. The index is movable and not copyable; a moved-from index may only be
 * destroyed or assigned to.
 */
class HnswIndex {
   public:
    /**
     * @param dim Vector dimension
     * @param options Metric, bits, cluster count, graph settings and seed
     */
    explicit HnswIndex(size_t dim, const HnswOptions& options = {});

    HnswIndex(HnswIndex&& other) noexcept;
    HnswIndex& operator=(HnswIndex&& other) noexcept;
    HnswIndex(const HnswIndex&) = delete;
    HnswIndex& operator=(const HnswIndex&) = delete;
    ~HnswIndex();

    // Load an index saved by any release. An empty, untrained file saved by 0.7 loads
    // untrained, with 16 clusters to train; without a recorded seed, it needs train or
    // set_centroids before add.
    static HnswIndex load(const char* filename);

    // Requires at least one stored point. Releases up to 0.8 cannot read the file.
    void save(const char* filename) const;

    /**
     * @brief Clustering parameters of train and of the first add.
     *
     * Defaults: niter 10, max_points_per_centroid 256, spherical for inner product and
     * cosine, and seed equal to the low 32 bits of the index seed, or the k-means default
     * for a file without one. Include rabitqlib/clustering/rabitqkmeans.hpp to read or
     * change the fields.
     */
    rabitqkmeans::RaBitQKMeansParameters& cp();
    [[nodiscard]] const rabitqkmeans::RaBitQKMeansParameters& cp() const;

    /**
     * @brief Train num_clusters() centroids on `data` and set them.
     *
     * Only centroids are trained; add assigns points. Retraining is allowed while the
     * index stores no points. The index is unchanged if this throws.
     *
     * @param data Training points (num_points * dim)
     * @param num_points At least num_clusters()
     * @param num_threads Worker count for this call; no value uses cp().num_threads
     */
    void train(
        const float* data,
        size_t num_points,
        std::optional<size_t> num_threads = std::nullopt
    );

    /**
     * @brief Use the given centroids instead of training.
     *
     * Allowed while the index stores no points. The index is unchanged if this throws.
     *
     * @param centroids num_clusters() * dim values
     */
    void set_centroids(const float* centroids);

    /**
     * @brief Add points, training an untrained index on them first.
     *
     * The first add trains min(num_clusters(), num_points) centroids on `data` as train
     * does, so it needs no cluster_ids; if it throws, the index stays untrained and
     * empty, with its capacity. On a trained index the points are quantized against the
     * existing centroids, which do not move. Capacity grows to fit the points, at least
     * doubling, after the points and cluster IDs are checked. A failure while linking may
     * leave part of the batch in the index, which stays safe to search, save and destroy.
     * An index loaded from an empty, untrained file saved before 0.8 has no seed, so
     * add does not train it; call train or set_centroids first.
     *
     * @param data New points (num_points * dim)
     * @param num_points Number of new points
     * @param options Cluster IDs, worker count and quantization speed. With more than one
     * worker the graph links vary between runs; use one, and a seed, for a reproducible
     * graph.
     * @return ID of the first new point; the batch has consecutive IDs
     */
    Id add(const float* data, size_t num_points, const AddOptions& options = {});

    /**
     * @brief Exclude points from later search results.
     *
     * Removed points keep their IDs, storage and place in the graph, and removal cannot
     * be undone.
     *
     * @param ids IDs in [0, num_stored_points()); repeats count once
     * @param num_ids Number of IDs
     * @return Number of points newly removed
     */
    size_t remove(const Id* ids, size_t num_ids);

    // Grow the capacity to num_points; a smaller value changes nothing.
    void reserve(size_t num_points);

    /**
     * @brief Search a batch of queries.
     *
     * Slots without a result get kNoId and +inf. Distances estimate squared L2, 1 - dot
     * or 1 - cos according to the metric; lower is better.
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
    // The configured count until centroids are set, then the trained count.
    [[nodiscard]] size_t num_clusters() const;
    [[nodiscard]] size_t nbits() const;
    [[nodiscard]] MetricType metric() const;
    [[nodiscard]] size_t M() const;
    [[nodiscard]] size_t ef_construction() const;
    // No value for an index loaded from a file of a release up to 0.8. A file written by
    // the C++ HierarchicalNSW class may hold a seed above 32 bits.
    [[nodiscard]] std::optional<uint64_t> seed() const;
    // Points the index can hold before it grows, including removed ones.
    [[nodiscard]] size_t capacity() const;
    // Live, searchable points.
    [[nodiscard]] size_t num_points() const;
    // Stored points, including removed ones.
    [[nodiscard]] size_t num_stored_points() const;
    // Whether centroids are set, by train, set_centroids, the first add or load.
    [[nodiscard]] bool is_trained() const;

   private:
    HnswIndex(
        std::unique_ptr<hnsw::HierarchicalNSW> index,
        std::unique_ptr<rabitqkmeans::RaBitQKMeansParameters> cp,
        size_t num_clusters
    );

    void check_trained(const char* operation) const;
    void check_trainable() const;

    std::unique_ptr<hnsw::HierarchicalNSW> index_;
    std::unique_ptr<rabitqkmeans::RaBitQKMeansParameters> cp_;
    size_t num_clusters_;  // centroid count train and set_centroids expect when untrained
};

}  // namespace rabitqlib
