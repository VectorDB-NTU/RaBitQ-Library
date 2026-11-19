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
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bindings_common.hpp"
#include "rabitqlib/clustering/rabitqkmeans.hpp"
#include "rabitqlib/index/hnsw_index.hpp"
#include "rabitqlib/index/types.hpp"

namespace py = pybind11;

namespace rabitqlib::python_bindings {

class HnswIndex {
   public:
    HnswIndex(
        size_t dim,
        size_t nbits,
        size_t num_clusters,
        size_t M,
        size_t ef_construction,
        const std::string& metric,
        const py::object& seed
    )
        : index_(dim, index_options(nbits, num_clusters, M, ef_construction, metric, seed))
        , cp_(index_.cp()) {}

    void train(py::handle x, std::optional<size_t> num_threads) {
        const auto operation = access_.write();
        auto x_array = ensure_2d_array<float>(x, "x");
        ensure_dimension(x_array, index_.dim(), "x dimension does not match index dim");
        const auto rows = training_rows(x_array, index_.num_clusters());
        const auto* x_ptr = x_array.data();
        index_.cp() = cp_;
        py::gil_scoped_release release;
        index_.train(x_ptr, rows, num_threads);
    }

    void set_centroids(py::handle centroids) {
        const auto operation = access_.write();
        auto centroids_array = ensure_2d_array<float>(centroids, "centroids");
        check_centroid_shape(centroids_array, index_.num_clusters(), index_.dim());
        const auto* centroids_ptr = centroids_array.data();
        py::gil_scoped_release release;
        index_.set_centroids(centroids_ptr);
    }

    py::array_t<int64_t> add(
        py::handle data,
        const py::object& cluster_ids,
        std::optional<size_t> num_threads,
        bool fast_quantization
    ) {
        const auto operation = access_.write();
        auto data_array = ensure_2d_array<float>(data, "data");
        ensure_dimension(
            data_array, index_.dim(), "data dimension does not match index dim"
        );
        const auto rows = static_cast<size_t>(data_array.shape(0));
        std::vector<PID> assigned;
        AddOptions options;
        if (!cluster_ids.is_none()) {
            assigned = checked_cluster_ids(cluster_ids, rows, index_.num_clusters());
            options.cluster_ids = assigned.data();
        }
        options.num_threads = num_threads;
        options.fast_quantization = fast_quantization;
        // Allocate the returned IDs before the index changes.
        auto ids = py::array_t<int64_t>(static_cast<py::ssize_t>(rows));
        const auto* data_ptr = data_array.data();
        index_.cp() = cp_;  // the first add trains with it
        Id first = 0;
        {
            py::gil_scoped_release release;
            first = index_.add(data_ptr, rows, options);
        }
        std::iota(ids.mutable_data(), ids.mutable_data() + rows, first);
        return ids;
    }

    void reserve(int64_t num_points) {
        if (num_points < 0) {
            throw std::invalid_argument("num_points must be non-negative");
        }
        {
            // Capacity only grows, and a request within it changes nothing, so it
            // may run alongside searches.
            const auto reading = access_.read();
            if (static_cast<uint64_t>(num_points) <= index_.capacity()) {
                return;
            }
        }
        // The GIL is held from the check above to this guard, so no other call
        // can change the capacity in between.
        const auto operation = access_.write();
        py::gil_scoped_release release;
        index_.reserve(static_cast<size_t>(num_points));
    }

    size_t remove(py::handle ids) {
        const auto operation = access_.write();
        const auto to_remove = copied_ids(ensure_1d_integer_array(ids, "ids"));
        py::gil_scoped_release release;
        return index_.remove(to_remove.data(), to_remove.size());
    }

    [[nodiscard]] size_t num_points() const {
        const auto operation = access_.read();
        return index_.num_points();
    }
    [[nodiscard]] size_t num_stored_points() const {
        const auto operation = access_.read();
        return index_.num_stored_points();
    }

    py::tuple search(
        py::handle queries,
        size_t k,
        std::optional<size_t> ef_search,
        std::optional<size_t> num_threads
    ) {
        const auto operation = access_.read();
        auto query_array = ensure_2d_array<float>(queries, "queries");
        ensure_dimension(
            query_array, index_.dim(), "query dimension does not match index dim"
        );
        check_k(k);  // before the outputs are allocated
        check_ef_search(ef_search);
        GraphSearchOptions options;
        options.ef_search = ef_search;
        options.num_threads = num_threads.value_or(1);

        const auto num_queries = static_cast<size_t>(query_array.shape(0));
        const auto shape = std::vector<py::ssize_t>{
            static_cast<py::ssize_t>(num_queries), static_cast<py::ssize_t>(k)};
        auto ids = py::array_t<int64_t>(shape);
        auto distances = py::array_t<float>(shape);
        const auto* queries_data = query_array.data();
        auto* ids_data = ids.mutable_data();
        auto* distances_data = distances.mutable_data();
        {
            py::gil_scoped_release release;
            index_.search(queries_data, num_queries, k, ids_data, distances_data, options);
        }
        return py::make_tuple(ids, distances);
    }

    void save(py::handle path) const {
        const auto filename = path_string(path);
        const auto operation = access_.read();
        py::gil_scoped_release release;
        index_.save(filename.c_str());
    }

    static HnswIndex load(py::handle path) {
        const auto filename = path_string(path);
        auto index = [&filename] {
            py::gil_scoped_release release;
            return rabitqlib::HnswIndex::load(filename.c_str());
        }();
        return HnswIndex(std::move(index));
    }

    [[nodiscard]] std::string repr() const {
        const auto operation = access_.read();
        return "HnswIndex(dim=" + std::to_string(index_.dim()) + ", metric='" +
               metric_to_string(index_.metric()) +
               "', nbits=" + std::to_string(index_.nbits()) +
               ", num_points=" + std::to_string(index_.num_points()) +
               ", num_stored_points=" + std::to_string(index_.num_stored_points()) +
               ", capacity=" + std::to_string(index_.capacity()) +
               ", num_clusters=" + std::to_string(index_.num_clusters()) +
               ", is_trained=" + (index_.is_trained() ? "True" : "False") + ")";
    }

    [[nodiscard]] size_t dim() const {
        const auto operation = access_.read();
        return index_.dim();
    }
    [[nodiscard]] size_t capacity() const {
        const auto operation = access_.read();
        return index_.capacity();
    }
    [[nodiscard]] size_t nbits() const {
        const auto operation = access_.read();
        return index_.nbits();
    }
    [[nodiscard]] size_t M() const {
        const auto operation = access_.read();
        return index_.M();
    }
    [[nodiscard]] size_t ef_construction() const {
        const auto operation = access_.read();
        return index_.ef_construction();
    }
    [[nodiscard]] bool is_trained() const {
        const auto operation = access_.read();
        return index_.is_trained();
    }
    [[nodiscard]] size_t num_clusters() const {
        const auto operation = access_.read();
        return index_.num_clusters();
    }
    [[nodiscard]] std::optional<uint64_t> seed() const {
        const auto operation = access_.read();
        return index_.seed();
    }
    [[nodiscard]] std::string metric() const {
        const auto operation = access_.read();
        return metric_to_string(index_.metric());
    }

    // Python reads and writes the fields of cp_ with the GIL held but without a guard, so
    // the native index trains with a copy that train and add make with the GIL held.
    rabitqkmeans::RaBitQKMeansParameters& cp() { return cp_; }
    void set_cp(const rabitqkmeans::RaBitQKMeansParameters& cp) {
        const auto operation = access_.write();
        cp_ = cp;
    }

   private:
    explicit HnswIndex(rabitqlib::HnswIndex index)
        : index_(std::move(index)), cp_(index_.cp()) {}

    static HnswOptions index_options(
        size_t nbits,
        size_t num_clusters,
        size_t M,
        size_t ef_construction,
        const std::string& metric,
        const py::object& seed
    ) {
        HnswOptions options;
        options.metric = index_metric_from_string(metric);
        if (!seed.is_none()) {
            options.seed = static_cast<uint32_t>(
                checked_seed(seed, std::numeric_limits<uint32_t>::max())
            );
        }
        options.nbits = nbits;
        options.num_clusters = num_clusters;
        options.M = M;
        options.ef_construction = ef_construction;
        return options;
    }

    mutable IndexAccess access_;
    rabitqlib::HnswIndex index_;
    rabitqkmeans::RaBitQKMeansParameters cp_;
};

}  // namespace rabitqlib::python_bindings

// Register into combined module
void register_hnsw(py::module_& m) {
    using namespace rabitqlib::python_bindings;

    static constexpr char kIndexDoc[] =
        R"doc(An approximate nearest-neighbor index using HNSW.

Parameters
----------
dim : int
    Original vector dimension, from 64 through 65536.
nbits : int, optional
    Keyword-only. Total quantization bits, 1 through 9. Default is 7.
num_clusters : int, optional
    Keyword-only. Number of centroids train, set_centroids or the first add
    provides; the first add caps it at its row count.
    Default is 16.
M : int, optional
    Keyword-only. Graph connectivity, at least 2; the base layer keeps up to
    2 * M edges. Default is 16.
ef_construction : int, optional
    Keyword-only. Construction search breadth. Default is 200.
metric : {'l2', 'ip', 'cosine'}, optional
    Keyword-only. Squared L2, inner-product or cosine distance. Default is
    'l2'. 'cosine' scales data and queries to unit length.
seed : int, optional
    Keyword-only. Graph and rotation seed in [0, 2**32 - 1], also the initial
    cp.seed. None (default) draws one at random; the seed property reports it.

Notes
-----
The first add trains an untrained index on the added vectors. Call train
on a sample, or set_centroids, first to choose the training data or the
centroids. Loaded indexes are trained, except one loaded from a file saved
before training. Capacity grows automatically
on add; reserve can preallocate it. Concurrent reads are supported; an
overlapping mutation raises RuntimeError.
Data, centroids and queries must be finite float32 values within the index
coordinate range; other values raise ValueError.
With more than one thread, add links points in a nondeterministic order;
pass num_threads=1 and a seed for a reproducible graph.

Examples
--------
>>> import numpy as np
>>> from rabitqlib import HnswIndex
>>> x = np.random.default_rng(42).normal(size=(80, 64)).astype(np.float32)
>>> index = HnswIndex(64, num_clusters=2, seed=7)
>>> ids = index.add(x, num_threads=1)  # trains on x first
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
    Keyword-only. Existing centroid IDs. None assigns each vector to its
    nearest centroid. Centroids do not move.
num_threads : int, optional
    Keyword-only. Worker count. None (default) and 0 select the available
    CPU count, capped by OMP_NUM_THREADS; more than one thread links
    nondeterministically, so pass 1, and a seed to the constructor, for a
    reproducible graph.
fast_quantization : bool, optional
    Keyword-only. Use faster quantizer configuration. Default is False.

Returns
-------
ids : ndarray of int64, shape (n,)
    Assigned point IDs.

Notes
-----
On an untrained index, add first trains num_clusters centroids on data, as
train(data) does; with fewer rows it trains one centroid per row, and
num_clusters then reports the row count. An empty first add raises
RuntimeError. With num_threads None, training
uses cp.num_threads. Passing cluster_ids to an untrained index raises
RuntimeError, as does add on an index loaded from an untrained file;
call train or set_centroids first. If the call fails, the index stays untrained
and empty. The first
add into an empty index links vectors with their exact coordinates; later adds
link through the stored codes, so add most vectors in the first call. Capacity
grows geometrically when needed. Removed IDs are not reused.)doc";

    py::class_<HnswIndex>(m, "HnswIndex", kIndexDoc)
        .def(
            "__reduce_ex__",
            &rabitqlib::python_bindings::reduce_ex_with_protocol_2,
            py::arg("protocol"),
            rabitqlib::python_bindings::kReduceExDoc
        )
        .def(
            py::init<
                size_t,
                size_t,
                size_t,
                size_t,
                size_t,
                const std::string&,
                const py::object&>(),
            py::arg("dim"),
            py::kw_only(),
            py::arg("nbits") = 7,
            py::arg("num_clusters") = 16,
            py::arg("M") = 16,
            py::arg("ef_construction") = 200,
            py::arg("metric") = "l2",
            py::arg("seed") = py::none(),
            kIndexDoc
        )
        .def(
            "train",
            &HnswIndex::train,
            py::arg("x"),
            py::kw_only(),
            py::arg("num_threads") = py::none(),
            R"doc(Train num_clusters centroids on vectors with RaBitQKMeans and set them.

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
Clusters with the settings in cp. Only centroids are trained; add assigns
vectors. Retraining replaces the centroids while the index stores no
vectors, and raises RuntimeError afterwards. A failed call leaves the index
unchanged.)doc"
        )
        .def(
            "set_centroids",
            &HnswIndex::set_centroids,
            py::arg("centroids"),
            R"doc(Use the given centroids instead of training.

Parameters
----------
centroids : array_like of float, shape (num_clusters, dim)
    Centroids in the original, unpadded coordinates. Converted to
    contiguous float32; a copy may be made.

Notes
-----
The index counts as trained afterwards. Replacing centroids is allowed
while the index stores no vectors and raises RuntimeError afterwards. A
failed call leaves the index unchanged.)doc"
        )
        .def(
            "search",
            &HnswIndex::search,
            py::arg("queries"),
            py::arg("k"),
            py::arg("ef_search") = py::none(),
            py::kw_only(),
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
ef_search : int, optional
    Search breadth, at least 1. None (default) selects max(k, 64); values
    above the stored point count are clamped.
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
Requires a trained index. Returns (ids, distances), in that
order. Missing neighbors have ID -1 and distance inf; test
np.isfinite(distances) to find them. Array shapes
and value ranges are checked at runtime, not by static type checkers.)doc"
        )
        .def(
            "add",
            &HnswIndex::add,
            py::arg("data"),
            py::kw_only(),
            py::arg("cluster_ids") = py::none(),
            py::arg("num_threads") = py::none(),
            py::arg("fast_quantization") = false,
            kAddDoc
        )
        .def(
            "reserve",
            &HnswIndex::reserve,
            py::arg("num_points"),
            R"doc(Preallocate room for at least num_points stored points.

Parameters
----------
num_points : int
    Nonnegative number of stored points to make room for, including the
    current ones.

Notes
-----
Capacity only grows: a value at or below capacity changes nothing. add also
grows the capacity when needed, so reserve is an optimization.)doc"
        )
        .def(
            "remove",
            &HnswIndex::remove,
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
            "save",
            &HnswIndex::save,
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
The saved state includes the rotation used to build the index and its seed;
releases up to 0.8 cannot read the file.)doc"
        )
        .def_static(
            "load",
            &HnswIndex::load,
            py::arg("path"),
            R"doc(Load an index and its saved rotation from disk.

Parameters
----------
path : str, bytes, bytearray, or os.PathLike
    Index file path. PathLike objects are resolved via __fspath__.
    Byte paths use native bytes on POSIX and UTF-8 on Windows.

Returns
-------
index : HnswIndex
    The loaded index, trained unless the file was saved before training.)doc"
        )
        .def(
            "__repr__",
            &HnswIndex::repr,
            "Return a one-line summary of configuration, counts and training state."
        )
        .def(
            "__len__",
            &HnswIndex::num_points,
            "Return the number of live, searchable points."
        )
        .def_property_readonly(
            "dim", &HnswIndex::dim, R"doc(Original vector dimension.)doc"
        )
        .def_property_readonly(
            "capacity",
            &HnswIndex::capacity,
            R"doc(Current storage capacity, in stored points.)doc"
        )
        .def_property_readonly(
            "num_points",
            &HnswIndex::num_points,
            R"doc(Number of live, searchable points.)doc"
        )
        .def_property_readonly(
            "num_stored_points",
            &HnswIndex::num_stored_points,
            R"doc(Number of stored points, including removed points.)doc"
        )
        .def_property_readonly(
            "nbits", &HnswIndex::nbits, R"doc(Total storage bits per coordinate.)doc"
        )
        .def_property_readonly(
            "M",
            &HnswIndex::M,
            R"doc(Graph connectivity; the base layer keeps up to 2 * M edges.)doc"
        )
        .def_property_readonly(
            "ef_construction",
            &HnswIndex::ef_construction,
            R"doc(Construction search breadth.)doc"
        )
        .def_property_readonly(
            "num_clusters",
            &HnswIndex::num_clusters,
            R"doc(Number of centroids: the configured count until centroids are set.)doc"
        )
        .def_property_readonly(
            "is_trained",
            &HnswIndex::is_trained,
            R"doc(Whether centroids are set, by train, set_centroids, the first add or load.)doc"
        )
        .def_property_readonly(
            "seed",
            &HnswIndex::seed,
            R"doc(Graph and rotation seed, or None for an index loaded from a file from a release up to 0.8.)doc"
        )
        .def_property(
            "cp",
            &HnswIndex::cp,
            &HnswIndex::set_cp,
            py::return_value_policy::reference_internal,
            R"doc(Clustering parameters train uses, a RaBitQKMeansParameters. Defaults: niter 10, max_points_per_centroid 256, spherical for inner product and cosine, and seed equal to the index seed.)doc"
        )
        .def_property_readonly(
            "metric",
            &HnswIndex::metric,
            R"doc(Canonical metric name: 'l2', 'ip' or 'cosine'.)doc"
        );
}
