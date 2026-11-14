#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/gil.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>  // IWYU pragma: keep; registers std::optional casters

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "bindings_common.hpp"
#include "rabitqlib/clustering/rabitqkmeans.hpp"
#include "rabitqlib/core/rotator.hpp"
#include "rabitqlib/index/ivf/initializer.hpp"
#include "rabitqlib/index/ivf/ivf.hpp"
#include "rabitqlib/index/training.hpp"
#include "rabitqlib/index/types.hpp"

namespace py = pybind11;

namespace rabitqlib::python_bindings {

namespace {
ivf::InitializerType routing_from_string(const std::string& value) {
    if (value == "auto") {
        return ivf::InitializerType::Auto;
    }
    if (value == "flat") {
        return ivf::InitializerType::Flat;
    }
    if (value == "flat_rabitq") {
        return ivf::InitializerType::FlatRaBitQ;
    }
    if (value == "hnsw") {
        return ivf::InitializerType::HNSW;
    }
    throw std::invalid_argument("routing must be 'auto', 'flat', 'flat_rabitq', or 'hnsw'");
}

std::string routing_to_string(ivf::InitializerType type) {
    switch (type) {
        case ivf::InitializerType::Auto:
            return "auto";
        case ivf::InitializerType::Flat:
            return "flat";
        case ivf::InitializerType::FlatRaBitQ:
            return "flat_rabitq";
        case ivf::InitializerType::HNSW:
            return "hnsw";
    }
    throw std::logic_error("Invalid IVF initializer type");
}
}  // namespace

class IvfIndex {
   public:
    IvfIndex(
        size_t dim,
        size_t num_clusters,
        size_t nbits,
        const std::string& metric,
        const std::string& routing,
        const py::object& seed
    ) {
        const auto metric_type = index_metric_from_string(metric);
        const auto initializer_type = routing_from_string(routing);
        const auto rotation_seed =
            seed.is_none() ? std::random_device{}()
                           : static_cast<uint32_t>(
                                 checked_seed(seed, std::numeric_limits<uint32_t>::max())
                             );
        index_ = std::make_unique<ivf::IVF>(
            dim,
            num_clusters,
            nbits,
            metric_type,
            rotation_seed,
            RotatorType::FhtKacRotator,
            initializer_type
        );
        seed_ = rotation_seed;
        cp_ = default_training_parameters(metric_type, rotation_seed);
    }

    void train(py::handle x, std::optional<size_t> num_threads) {
        const auto operation = access_.write();
        auto x_array = ensure_2d_array<float>(x, "x");
        check_trainable();
        ensure_dimension(
            x_array, index_->dimension(), "x dimension does not match index dim"
        );
        const auto rows = training_rows(x_array, index_->num_clusters());
        const auto cp = training_parameters(cp_, num_threads);
        const auto* x_ptr = x_array.data();
        py::gil_scoped_release release;
        index_->train(rows, x_ptr, cp);
    }

    void set_centroids(py::handle centroids, std::optional<size_t> num_threads) {
        const auto operation = access_.write();
        auto centroids_array = ensure_2d_array<float>(centroids, "centroids");
        check_trainable();
        check_centroid_shape(centroids_array, index_->num_clusters(), index_->dimension());
        const auto* centroids_ptr = centroids_array.data();
        py::gil_scoped_release release;
        index_->set_centroids(centroids_ptr, num_threads.value_or(0));
    }

    py::array_t<int64_t> add(
        py::handle data,
        const py::object& cluster_ids,
        std::optional<size_t> num_threads,
        bool fast_quantization
    ) {
        const auto operation = access_.write();
        auto data_array = ensure_2d_array<float>(data, "data");
        // The first add on an untrained index trains it on the same data. A loaded
        // index is trained, so an untrained one knows its seed.
        const bool trains =
            !index_->is_trained() && cluster_ids.is_none() && data_array.shape(0) > 0;
        if (!trains) {
            check_trained("add");
        }
        ensure_dimension(
            data_array, index_->dimension(), "data dimension does not match index dim"
        );
        const auto rows = static_cast<size_t>(data_array.shape(0));
        if (trains && rows < index_->num_clusters()) {
            throw std::invalid_argument(
                "the first add trains the index and needs at least num_clusters rows; "
                "expected at least " +
                std::to_string(index_->num_clusters()) + ", got shape " +
                array_shape(data_array)
            );
        }

        std::vector<PID> assigned;
        if (!cluster_ids.is_none()) {
            assigned = checked_cluster_ids(cluster_ids, rows, index_->num_clusters());
        }

        const size_t first = index_->num_stored_points();
        // Allocate the returned IDs before committing any changes to the index.
        auto ids = py::array_t<int64_t>(static_cast<py::ssize_t>(rows));
        auto* ids_data = ids.mutable_data();
        for (size_t i = 0; i < rows; ++i) {
            ids_data[i] = static_cast<int64_t>(first + i);
        }

        const auto* data_ptr = data_array.data();
        const size_t threads = num_threads.value_or(0);
        if (trains) {
            // Train and fill a fresh index with the same rotation, then swap it in,
            // so that a failure leaves this index untrained and empty.
            const auto cp = training_parameters(cp_, num_threads);
            auto staged = std::make_unique<ivf::IVF>(
                index_->dimension(),
                index_->num_clusters(),
                index_->nbits(),
                index_->metric_type(),
                *seed_,
                RotatorType::FhtKacRotator,
                index_->initializer_type()
            );
            {
                py::gil_scoped_release release;
                staged->train(rows, data_ptr, cp);
                staged->add(data_ptr, rows, nullptr, fast_quantization, threads);
            }
            index_.swap(staged);
            return ids;
        }
        {
            py::gil_scoped_release release;
            index_->add(
                data_ptr,
                rows,
                assigned.empty() ? nullptr : assigned.data(),
                fast_quantization,
                threads
            );
        }
        return ids;
    }

    void reserve_cluster(int64_t cluster_id, int64_t num_points) {
        const auto operation = access_.write();
        check_trained("reserve_cluster");
        if (cluster_id < 0 || static_cast<uint64_t>(cluster_id) >= index_->num_clusters()) {
            throw std::invalid_argument("cluster_id must be in [0, num_clusters)");
        }
        if (num_points < 0) {
            throw std::invalid_argument("num_points must be non-negative");
        }
        py::gil_scoped_release release;
        index_->reserve(static_cast<PID>(cluster_id), static_cast<size_t>(num_points));
    }

    size_t remove(py::handle ids) {
        const auto operation = access_.write();
        auto ids_array = ensure_1d_integer_array(ids, "ids");
        check_trained("remove");
        auto to_remove = checked_removal_ids(ids_array, index_->num_stored_points());
        py::gil_scoped_release release;
        return index_->remove(to_remove.data(), to_remove.size());
    }

    py::tuple search(
        py::handle queries,
        size_t k,
        std::optional<size_t> nprobe,
        std::optional<bool> high_accuracy,
        std::optional<size_t> num_threads
    ) {
        const auto operation = access_.read();
        auto query_array = ensure_2d_array<float>(queries, "queries");
        check_trained("search");
        ensure_dimension(
            query_array, index_->dimension(), "query dimension does not match index dim"
        );
        check_k(k);
        if (nprobe == size_t{0}) {
            throw std::invalid_argument(
                "nprobe must be at least 1; pass None for the default"
            );
        }
        // None probes num_clusters / 16 clusters, rounded, and at least one.
        const size_t num_probes =
            nprobe.value_or(std::max<size_t>(1, (index_->num_clusters() + 8) / 16));

        const size_t num_queries = static_cast<size_t>(query_array.shape(0));
        auto output = unfilled_search_results(num_queries, k);
        auto* ids_data = output.ids.data();
        auto* distances_data = output.distances.mutable_data();

        const auto* queries_data = query_array.data();
        {
            py::gil_scoped_release release;
            index_->search_batch(
                queries_data,
                num_queries,
                k,
                num_probes,
                ids_data,
                distances_data,
                high_accuracy,
                num_threads.value_or(1)
            );
        }

        return output.to_python();
    }

    void save(py::handle path) const {
        const auto filename = path_string(path);
        const auto operation = access_.read();
        check_trained("save");
        if (index_->num_stored_points() == 0) {
            throw std::runtime_error("IvfIndex has no points to save; add points first");
        }
        py::gil_scoped_release release;
        index_->save(filename.c_str());
    }

    static IvfIndex load(py::handle path) {
        const auto filename = path_string(path);
        IvfIndex wrapper;
        wrapper.index_ = std::make_unique<ivf::IVF>();
        {
            py::gil_scoped_release release;
            wrapper.index_->load(filename.c_str());
        }
        wrapper.seed_ = wrapper.index_->seed();  // recorded by format-2 files
        wrapper.cp_ = default_training_parameters(
            wrapper.index_->metric_type(),
            wrapper.seed_.value_or(rabitqkmeans::RaBitQKMeansParameters{}.seed)
        );
        return wrapper;
    }

    [[nodiscard]] std::string repr() const {
        const auto operation = access_.read();
        return "IvfIndex(dim=" + std::to_string(index_->dimension()) + ", metric='" +
               metric_to_string(index_->metric_type()) +
               "', nbits=" + std::to_string(index_->nbits()) +
               ", num_points=" + std::to_string(index_->num_points()) +
               ", num_stored_points=" + std::to_string(index_->num_stored_points()) +
               ", num_clusters=" + std::to_string(index_->num_clusters()) + ", routing='" +
               routing_to_string(index_->initializer_type()) +
               "', is_trained=" + (index_->is_trained() ? "True" : "False") + ")";
    }

    [[nodiscard]] size_t dim() const {
        const auto operation = access_.read();
        return index_->dimension();
    }
    [[nodiscard]] size_t num_points() const {
        const auto operation = access_.read();
        return index_->num_points();
    }
    [[nodiscard]] size_t num_stored_points() const {
        const auto operation = access_.read();
        return index_->num_stored_points();
    }
    [[nodiscard]] size_t num_clusters() const {
        const auto operation = access_.read();
        return index_->num_clusters();
    }
    [[nodiscard]] size_t nbits() const {
        const auto operation = access_.read();
        return index_->nbits();
    }
    [[nodiscard]] bool is_trained() const {
        const auto operation = access_.read();
        return index_->is_trained();
    }
    [[nodiscard]] std::optional<uint32_t> seed() const {
        const auto operation = access_.read();
        return seed_;
    }
    [[nodiscard]] std::string routing() const {
        const auto operation = access_.read();
        return routing_to_string(index_->initializer_type());
    }
    [[nodiscard]] std::string metric() const {
        const auto operation = access_.read();
        return metric_to_string(index_->metric_type());
    }

    // Parameters are wrapper state that train copies with the GIL held.
    rabitqkmeans::RaBitQKMeansParameters& cp() { return cp_; }
    void set_cp(const rabitqkmeans::RaBitQKMeansParameters& cp) {
        const auto operation = access_.write();
        cp_ = cp;
    }

   private:
    IvfIndex() = default;

    void check_trained(const char* operation) const {
        if (!index_->is_trained()) {
            throw std::runtime_error(
                std::string("IvfIndex must be trained or loaded before ") + operation
            );
        }
    }

    void check_trainable() const {
        if (index_->num_stored_points() != 0) {
            throw std::runtime_error("IvfIndex cannot be trained once it stores points");
        }
    }

    mutable IndexAccess access_;
    std::unique_ptr<ivf::IVF> index_;
    rabitqkmeans::RaBitQKMeansParameters cp_;
    std::optional<uint32_t> seed_;  // Unknown for files from releases up to 0.8.
};

}  // namespace rabitqlib::python_bindings

// Register IVF bindings into combined module
void register_ivf(py::module_& m) {
    using namespace rabitqlib::python_bindings;

    static constexpr char kIndexDoc[] =
        R"doc(An approximate nearest-neighbor index using IVF.

Parameters
----------
dim : int
    Original vector dimension, from 64 through 65536.
num_clusters : int
    Number of centroids, from train, set_centroids or the first add.
nbits : int, optional
    Keyword-only. Total quantization bits, 1 through 9, or 32 for owned
    float32 vectors with one-bit filtering and raw reranking. Default is 7.
metric : {'l2', 'ip', 'innerproduct', 'cosine'}, optional
    Squared L2, inner-product or cosine distance. Default is 'l2'. 'cosine'
    scales data and queries to unit length.
routing : {'auto', 'flat', 'flat_rabitq', 'hnsw'}, optional
    Centroid routing strategy. Default is 'auto'.
seed : int, optional
    Rotation seed in [0, 2**32 - 1], also the initial cp.seed. None
    (default) draws one at random; the seed property reports it.

Notes
-----
The first add trains an untrained index on the added vectors. Call train
on a sample, or set_centroids, first to choose the training data or the
centroids. Loaded indexes are already trained. Concurrent reads are
supported; an overlapping mutation raises RuntimeError.
Data, centroids and queries must be finite float32 values within the index
coordinate range; other values raise ValueError.

Examples
--------
>>> import numpy as np
>>> from rabitqlib import IvfIndex
>>> x = np.random.default_rng(42).normal(size=(80, 64)).astype(np.float32)
>>> index = IvfIndex(64, 2, seed=7)
>>> ids = index.add(x)  # trains on x first
>>> ids, distances = index.search(x[:2], k=3)
>>> ids.shape, distances.shape
((2, 3), (2, 3)))doc";

    static constexpr char kAddDoc[] = R"doc(Add vectors, training the index first if needed.

Parameters
----------
data : array_like of float, shape (n, dim)
    Vectors in the original, unpadded coordinates. Converted to contiguous
    float32; a copy may be made.
cluster_ids : array_like of int, shape (n,), optional
    Keyword-only. Existing centroid IDs. None routes each vector using the
    index's configured routing strategy, which may be approximate.
    Centroids do not move.
num_threads : int, optional
    Keyword-only. Worker count. None (default) and 0 select the available
    CPU count, capped by OMP_NUM_THREADS.
fast_quantization : bool, optional
    Keyword-only. Use faster quantizer configuration. Default is False.

Returns
-------
ids : ndarray of int64, shape (n,)
    Assigned point IDs.

Notes
-----
On an untrained index, add first trains num_clusters centroids on data,
as train(data) does, so data needs at least num_clusters rows, and an
empty first add raises RuntimeError; with
num_threads None, training uses cp.num_threads. Passing cluster_ids to an
untrained index raises RuntimeError. If the call fails, the index stays
untrained and empty. Batch additions to reduce repeated cluster copying;
reserve_cluster can preallocate room.)doc";

    py::class_<IvfIndex>(m, "IvfIndex", kIndexDoc)
        .def(
            py::init<
                size_t,
                size_t,
                size_t,
                const std::string&,
                const std::string&,
                const py::object&>(),
            py::arg("dim"),
            py::arg("num_clusters"),
            py::kw_only(),
            py::arg("nbits") = 7,
            py::arg("metric") = "l2",
            py::arg("routing") = "auto",
            py::arg("seed") = py::none(),
            kIndexDoc
        )
        .def(
            "train",
            &IvfIndex::train,
            py::arg("x"),
            py::kw_only(),
            py::arg("num_threads") = py::none(),
            R"doc(Train num_clusters centroids on vectors and set them.

Parameters
----------
x : array_like of float, shape (n, dim)
    Training vectors in the original, unpadded coordinates, with
    n >= num_clusters. Converted to contiguous float32; a copy may be made.
num_threads : int, optional
    Keyword-only. Worker count for this call. None (default) uses
    cp.num_threads; 0 selects the available CPU count, capped by OMP_NUM_THREADS.

Notes
-----
Clusters with the settings in cp: RaBitQKMeans up to 2000 clusters, and
QGKMeans with its default graph settings above that (a provisional,
unmeasured threshold). Only centroids are trained; add assigns vectors.
Retraining replaces the centroids while the index stores no vectors, and
raises RuntimeError afterwards. A failed call leaves the index unchanged.)doc"
        )
        .def(
            "set_centroids",
            &IvfIndex::set_centroids,
            py::arg("centroids"),
            py::kw_only(),
            py::arg("num_threads") = py::none(),
            R"doc(Use the given centroids instead of training.

Parameters
----------
centroids : array_like of float, shape (num_clusters, dim)
    Centroids in the original, unpadded coordinates. Converted to
    contiguous float32; a copy may be made.
num_threads : int, optional
    Keyword-only. Worker count for rotating the centroids and building the
    centroid routing. None (default) and 0 select the available CPU count,
    capped by OMP_NUM_THREADS. Only the HNSW routing graph varies with it:
    build it with 1 thread and a fixed seed for a reproducible graph.

Notes
-----
The index counts as trained afterwards. Replacing centroids is allowed
while the index stores no vectors and raises RuntimeError afterwards. A
failed call leaves the index unchanged.)doc"
        )
        .def(
            "add",
            &IvfIndex::add,
            py::arg("data"),
            py::kw_only(),
            py::arg("cluster_ids") = py::none(),
            py::arg("num_threads") = py::none(),
            py::arg("fast_quantization") = false,
            kAddDoc
        )
        .def(
            "reserve_cluster",
            &IvfIndex::reserve_cluster,
            py::arg("cluster_id"),
            py::arg("num_points"),
            R"doc(Preallocate storage for a cluster.

Parameters
----------
cluster_id : int
    Cluster ID in [0, num_clusters).
num_points : int
    Nonnegative capacity to reserve for this cluster: the total number of
    points it can hold, including its current ones.

Notes
-----
Requires a trained or loaded index. Reserved room is not saved.)doc"
        )
        .def(
            "remove",
            &IvfIndex::remove,
            py::arg("ids"),
            R"doc(Exclude point IDs from subsequent search results.

Parameters
----------
ids : array_like of int, shape (n,)
    IDs in [0, num_stored_points). Floating-point and boolean IDs are
    rejected.

Returns
-------
count : int
    Number of newly removed points; repeats do not count twice.

Notes
-----
Removal does not reclaim storage and cannot be undone.)doc"
        )
        .def(
            "search",
            &IvfIndex::search,
            py::arg("queries"),
            py::arg("k"),
            py::arg("nprobe") = py::none(),
            py::kw_only(),
            py::arg("high_accuracy") = py::none(),
            py::arg("num_threads") = py::none(),
            R"doc(Search a batch of query vectors.

Parameters
----------
queries : array_like of float, shape (n_queries, dim)
    Vectors in the original, unpadded coordinates. Converted to contiguous
    float32; a copy may be made.
k : int
    Number of neighbors requested, at least 1. It may exceed the number of
    points; slots without a neighbor are padded as described below.
nprobe : int, optional
    Number of clusters to probe, at least 1 and capped at num_clusters. None
    (default) selects num_clusters / 16, rounded to the nearest integer and
    at least 1.
high_accuracy : bool, optional
    Keyword-only. Enable HACC. None (default) selects it for 4-9 bits and
    standard FastScan otherwise. Raw storage uses standard FastScan.
num_threads : int, optional
    Keyword-only. Worker count. None (default) selects 1; 0 selects the
    available CPU count, capped by OMP_NUM_THREADS.

Returns
-------
ids : ndarray of int64, shape (n_queries, k)
    Point IDs of the selected neighbors.
distances : ndarray of float32, shape (n_queries, k)
    Lower values are better: squared L2 estimates for l2, 1 - dot for ip and
    1 - cos for cosine. These are distances, not similarity scores.

Notes
-----
Requires a trained or loaded index. Returns (ids, distances), in that
order. Missing neighbors have ID -1 and distance inf; test
np.isfinite(distances) to find them. Array shapes
and value ranges are checked at runtime, not by static type checkers.)doc"
        )
        .def(
            "save",
            &IvfIndex::save,
            py::arg("path"),
            R"doc(Save an index with stored vectors to disk.

Parameters
----------
path : str, bytes, bytearray, or os.PathLike
    Index file path. PathLike objects are resolved via __fspath__.
    Byte paths use native bytes on POSIX and UTF-8 on Windows.

Notes
-----
A trained index without vectors raises RuntimeError; add vectors first.
The one file includes HNSW centroid routing; releases up to 0.8 cannot read it.)doc"
        )
        .def_static(
            "load",
            &IvfIndex::load,
            py::arg("path"),
            R"doc(Load an index and its saved rotation from disk.

Parameters
----------
path : str, bytes, bytearray, or os.PathLike
    Index file path. PathLike objects are resolved via __fspath__.
    Byte paths use native bytes on POSIX and UTF-8 on Windows.

Returns
-------
index : IvfIndex
    A trained index ready for search.

Notes
-----
Files from releases up to 0.8 with HNSW centroid routing also need the
accompanying ``<path>.hnsw`` file.)doc"
        )
        .def(
            "__repr__",
            &IvfIndex::repr,
            "Return a one-line summary of configuration, counts and training state."
        )
        .def(
            "__len__",
            &IvfIndex::num_points,
            "Return the number of live, searchable points."
        )
        .def_property_readonly("dim", &IvfIndex::dim, R"doc(Original vector dimension.)doc")
        .def_property_readonly(
            "num_points",
            &IvfIndex::num_points,
            R"doc(Number of live, searchable points.)doc"
        )
        .def_property_readonly(
            "num_stored_points",
            &IvfIndex::num_stored_points,
            R"doc(Number of stored points, including removed points.)doc"
        )
        .def_property_readonly(
            "num_clusters",
            &IvfIndex::num_clusters,
            R"doc(Number of centroids used for quantization.)doc"
        )
        .def_property_readonly(
            "nbits", &IvfIndex::nbits, R"doc(Total storage bits per coordinate.)doc"
        )
        .def_property_readonly(
            "is_trained",
            &IvfIndex::is_trained,
            R"doc(Whether centroids are set, by train, set_centroids, the first add or load.)doc"
        )
        .def_property_readonly(
            "seed",
            &IvfIndex::seed,
            R"doc(Rotation seed, or None for an index loaded from a file from a release up to 0.8.)doc"
        )
        .def_property(
            "cp",
            &IvfIndex::cp,
            &IvfIndex::set_cp,
            py::return_value_policy::reference_internal,
            R"doc(Clustering parameters train uses, a RaBitQKMeansParameters. Defaults: niter 10, max_points_per_centroid 256, spherical for inner product and cosine, and seed equal to the index seed.)doc"
        )
        .def_property_readonly(
            "metric",
            &IvfIndex::metric,
            R"doc(Canonical metric name: 'l2', 'ip' or 'cosine'.)doc"
        )
        .def_property_readonly(
            "routing",
            &IvfIndex::routing,
            R"doc(Resolved centroid routing strategy, never 'auto'.)doc"
        );
}
