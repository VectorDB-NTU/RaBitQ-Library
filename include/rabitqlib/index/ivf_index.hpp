#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include "rabitqlib/core/metric.hpp"      // IWYU pragma: export
#include "rabitqlib/index/ivf/types.hpp"  // IWYU pragma: export
#include "rabitqlib/index/types.hpp"      // IWYU pragma: export

namespace rabitqlib {
namespace ivf {
class IVF;
}
namespace rabitqkmeans {
struct RaBitQKMeansParameters;
}

// Settings of a new IvfIndex; set the fields by name. The metric comes first, so a lone
// positional value must be a metric: {4} does not compile.
struct IvfOptions {
    MetricType metric = METRIC_L2;  // METRIC_L2, METRIC_IP or METRIC_COSINE
    size_t nbits = 7;               // 1-9 quantization bits, or 32 for raw vectors
    ivf::Routing routing = ivf::Routing::Auto;
    std::optional<uint32_t> seed;  // rotation seed; no value draws one
};

// Settings of IvfIndex::search.
struct IvfSearchOptions {
    // Clusters to probe; a value above num_clusters() probes every cluster. No value
    // probes num_clusters() / 16, rounded, and at least 1; 0 throws.
    std::optional<size_t> nprobe;
    // Worker count; 0 selects every available CPU, capped by OMP_NUM_THREADS.
    size_t num_threads = 1;
    // HACC reranking. No value selects it for 4-9 bits and standard FastScan otherwise.
    std::optional<bool> high_accuracy;
};

/**
 * @brief An IVF-RaBitQ index with the workflow of the Python IvfIndex.
 *
 * Create it, then add vectors: the first add trains an untrained index on them. Call
 * train on a sample, or set_centroids, first to choose the training data or the
 * centroids. Search, remove, reserve_cluster and save need a trained index.
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
class IvfIndex {
   public:
    /**
     * @param dim Vector dimension
     * @param num_clusters Number of centroids, from train, set_centroids or the first add
     * @param options Metric, bits, routing and seed
     */
    explicit IvfIndex(size_t dim, size_t num_clusters, const IvfOptions& options = {});

    IvfIndex(IvfIndex&& other) noexcept;
    IvfIndex& operator=(IvfIndex&& other) noexcept;
    IvfIndex(const IvfIndex&) = delete;
    IvfIndex& operator=(const IvfIndex&) = delete;
    ~IvfIndex();

    // Load an index saved by any release, with its rotation; the result is trained.
    static IvfIndex load(const char* filename);

    // Requires at least one stored point. Releases up to 0.8 cannot read the file.
    void save(const char* filename) const;

    /**
     * @brief Clustering parameters of train and of the first add.
     *
     * Defaults: niter 10, max_points_per_centroid 256, spherical for inner product and
     * cosine, and seed equal to the index seed, or the k-means default for a file
     * without one. Include rabitqlib/clustering/rabitqkmeans.hpp to read or change the
     * fields.
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
     * @param num_threads Worker count for building the centroid routing
     */
    void set_centroids(const float* centroids, size_t num_threads = 0);

    /**
     * @brief Add points, training an untrained index on them first.
     *
     * The first add trains num_clusters() centroids on `data` as train does, so it needs
     * at least num_clusters() points and no cluster_ids; if it throws, the index stays
     * untrained and empty. On a trained index the points are quantized against the
     * existing centroids, which do not move, and the index is unchanged if this throws.
     *
     * @param data New points (num_points * dim)
     * @param num_points Number of new points
     * @param options Cluster IDs, worker count and quantization speed
     * @return ID of the first new point; the batch has consecutive IDs
     */
    Id add(const float* data, size_t num_points, const AddOptions& options = {});

    /**
     * @brief Exclude points from later search results.
     *
     * Removed points keep their IDs and storage, and removal cannot be undone.
     *
     * @param ids IDs in [0, num_stored_points()); repeats count once
     * @param num_ids Number of IDs
     * @return Number of points newly removed
     */
    size_t remove(const Id* ids, size_t num_ids);

    // Preallocate room for cluster_id to hold num_points points in total, so that later
    // adds grow it in place. Reserved room is not saved.
    void reserve_cluster(size_t cluster_id, size_t num_points);

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
        const IvfSearchOptions& options = {}
    ) const;

    [[nodiscard]] size_t dim() const;
    [[nodiscard]] size_t num_clusters() const;
    [[nodiscard]] size_t nbits() const;
    [[nodiscard]] MetricType metric() const;
    // Never Auto: the routing Auto resolved to.
    [[nodiscard]] ivf::Routing routing() const;
    // No value for an index loaded from a file of a release up to 0.8.
    [[nodiscard]] std::optional<uint32_t> seed() const;
    // Live, searchable points.
    [[nodiscard]] size_t num_points() const;
    // Stored points, including removed ones.
    [[nodiscard]] size_t num_stored_points() const;
    // Whether centroids are set, by train, set_centroids, the first add or load.
    [[nodiscard]] bool is_trained() const;

   private:
    IvfIndex(
        std::unique_ptr<ivf::IVF> index,
        std::unique_ptr<rabitqkmeans::RaBitQKMeansParameters> cp
    );

    void check_trained(const char* operation) const;
    void check_trainable() const;
    [[nodiscard]] rabitqkmeans::RaBitQKMeansParameters training_parameters(
        std::optional<size_t> num_threads
    ) const;

    std::unique_ptr<ivf::IVF> index_;
    std::unique_ptr<rabitqkmeans::RaBitQKMeansParameters> cp_;
};

}  // namespace rabitqlib
