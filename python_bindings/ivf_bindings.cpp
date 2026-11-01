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
#include "rabitqlib/utils/input_checks.hpp"

namespace py = pybind11;

namespace rabitqlib::python_bindings {

namespace {
ivf::InitializerType initializer_from_string(const std::string& value) {
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
    throw std::invalid_argument(
        "initializer must be 'auto', 'flat', 'flat_rabitq', or 'hnsw'"
    );
}

std::string initializer_to_string(ivf::InitializerType type) {
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
        const std::string& initializer,
        const py::object& seed
    ) {
        const auto metric_type = metric_from_string(metric);
        const auto initializer_type = initializer_from_string(initializer);
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

    // The 0.7 point-count form, which build fills with construct.
    static IvfIndex with_point_count(
        size_t dim,
        size_t max_elements,
        size_t num_clusters,
        size_t nbits,
        const std::string& metric,
        const std::string& initializer
    ) {
        const auto metric_type = metric_from_string(metric);
        IvfIndex wrapper;
        wrapper.index_ = std::make_unique<ivf::IVF>(
            max_elements,
            dim,
            num_clusters,
            nbits,
            metric_type,
            RotatorType::FhtKacRotator,
            initializer_from_string(initializer)
        );
        wrapper.cp_ = untrainable_parameters(metric_type);
        wrapper.builds_by_construct_ = true;
        return wrapper;
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

    void set_centroids(py::handle centroids) {
        const auto operation = access_.write();
        auto centroids_array = ensure_2d_array<float>(centroids, "centroids");
        check_trainable();
        check_centroid_shape(centroids_array, index_->num_clusters(), index_->dimension());
        const auto* centroids_ptr = centroids_array.data();
        py::gil_scoped_release release;
        index_->set_centroids(centroids_ptr);
    }

    // Deprecated: the 0.7 build. A point-count index is constructed; any other
    // index sets the centroids and adds the data.
    void build(
        py::handle data,
        py::handle centroids,
        py::handle cluster_ids,
        size_t num_threads,
        bool fast_quantization
    ) {
        const auto operation = access_.write();
        auto data_array = ensure_2d_array<float>(data, "data");
        auto centroids_array = ensure_2d_array<float>(centroids, "centroids");

        ensure_dimension(
            data_array, index_->dimension(), "data dimension does not match index dim"
        );
        if (builds_by_construct_ &&
            static_cast<size_t>(data_array.shape(0)) != index_->max_elements()) {
            throw std::invalid_argument(
                "number of data rows must match index max_elements; expected " +
                std::to_string(index_->max_elements()) + ", got shape " +
                array_shape(data_array)
            );
        }
        ensure_dimension(
            centroids_array,
            index_->dimension(),
            "centroid dimension does not match index dim"
        );
        if (static_cast<size_t>(centroids_array.shape(0)) != index_->num_clusters()) {
            throw std::invalid_argument(
                "number of centroid rows must match index num_clusters; expected " +
                std::to_string(index_->num_clusters()) + ", got shape " +
                array_shape(centroids_array)
            );
        }
        const auto rows = static_cast<size_t>(data_array.shape(0));
        auto cluster_ids_vec =
            checked_cluster_ids(cluster_ids, rows, index_->num_clusters());
        if (!builds_by_construct_) {
            check_trainable();
        }

        const auto* data_ptr = data_array.data();
        const auto* centroids_ptr = centroids_array.data();
        const auto* cluster_ids_ptr = cluster_ids_vec.data();
        py::gil_scoped_release release;
        if (builds_by_construct_) {
            index_->construct(
                data_ptr, centroids_ptr, cluster_ids_ptr, fast_quantization, num_threads
            );
        } else {
            // Check the data before replacing the centroids, so that invalid input
            // leaves the index unchanged; add repeats the check.
            const size_t dim = index_->dimension();
            check_coordinates(data_ptr, rows * dim, dim, "IVF add data");
            index_->set_centroids(centroids_ptr, num_threads);
            index_->add(data_ptr, rows, cluster_ids_ptr, fast_quantization, num_threads);
        }
    }

    py::array_t<PID> add(
        py::handle data,
        const py::object& cluster_ids,
        size_t num_threads,
        bool fast_quantization
    ) {
        const auto operation = access_.write();
        auto data_array = ensure_2d_array<float>(data, "data");
        check_trained("add");
        ensure_dimension(
            data_array, index_->dimension(), "data dimension does not match index dim"
        );
        const auto rows = static_cast<size_t>(data_array.shape(0));

        std::vector<PID> assigned;
        if (!cluster_ids.is_none()) {
            assigned = checked_cluster_ids(cluster_ids, rows, index_->num_clusters());
        }

        const size_t first = index_->num_stored_points();
        // Allocate the returned IDs before committing any changes to the index.
        auto ids = py::array_t<PID>(static_cast<py::ssize_t>(rows));
        auto* ids_data = ids.mutable_data();
        for (size_t i = 0; i < rows; ++i) {
            ids_data[i] = static_cast<PID>(first + i);
        }

        const auto* data_ptr = data_array.data();
        {
            py::gil_scoped_release release;
            index_->add(
                data_ptr,
                rows,
                assigned.empty() ? nullptr : assigned.data(),
                fast_quantization,
                num_threads
            );
        }
        return ids;
    }

    void reserve(int64_t cluster_id, int64_t num_points) {
        const auto operation = access_.write();
        check_trained("reserve");
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
        size_t nprobe,
        std::optional<bool> high_accuracy,
        size_t num_threads
    ) {
        const auto operation = access_.read();
        auto query_array = ensure_2d_array<float>(queries, "queries");
        check_trained("search");
        ensure_dimension(
            query_array, index_->dimension(), "query dimension does not match index dim"
        );
        if (k == 0 || k > index_->max_elements()) {
            throw std::invalid_argument("k must be between 1 and max_elements");
        }
        if (nprobe == 0) {
            throw std::invalid_argument("nprobe must be positive");
        }

        const size_t num_queries = static_cast<size_t>(query_array.shape(0));
        auto output = unfilled_search_results(num_queries, k);
        auto* ids_data = output.ids.mutable_data();
        auto* distances_data = output.distances.mutable_data();

        const auto* queries_data = query_array.data();
        {
            py::gil_scoped_release release;
            index_->search_batch(
                queries_data,
                num_queries,
                k,
                nprobe,
                ids_data,
                distances_data,
                high_accuracy,
                num_threads
            );
        }

        return py::make_tuple(output.ids, output.distances);
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
        wrapper.cp_ = untrainable_parameters(wrapper.index_->metric_type());
        wrapper.builds_by_construct_ = true;
        return wrapper;
    }

    [[nodiscard]] std::string repr() const {
        const auto operation = access_.read();
        return "IvfIndex(dim=" + std::to_string(index_->dimension()) + ", metric='" +
               metric_to_string(index_->metric_type()) +
               "', nbits=" + std::to_string(index_->nbits()) +
               ", num_points=" + std::to_string(index_->num_points()) +
               ", num_stored_points=" + std::to_string(index_->num_stored_points()) +
               ", max_elements=" + std::to_string(index_->max_elements()) +
               ", num_clusters=" + std::to_string(index_->num_clusters()) +
               ", initializer='" + initializer_to_string(index_->initializer_type()) +
               "', is_trained=" + (index_->is_trained() ? "True" : "False") + ")";
    }

    [[nodiscard]] size_t dim() const {
        const auto operation = access_.read();
        return index_->dimension();
    }
    [[nodiscard]] size_t max_elements() const {
        const auto operation = access_.read();
        return index_->max_elements();
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
    [[nodiscard]] std::string initializer() const {
        const auto operation = access_.read();
        return initializer_to_string(index_->initializer_type());
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

    // Parameters of an index that cannot be trained; the k-means default seed.
    static rabitqkmeans::RaBitQKMeansParameters untrainable_parameters(MetricType metric) {
        return default_training_parameters(
            metric, rabitqkmeans::RaBitQKMeansParameters{}.seed
        );
    }

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
        if (builds_by_construct_) {
            throw std::runtime_error(
                "IvfIndex created with max_elements is filled by build; create "
                "IvfIndex(dim, num_clusters, nbits=...) to train or set centroids"
            );
        }
    }

    mutable IndexAccess access_;
    std::unique_ptr<ivf::IVF> index_;
    rabitqkmeans::RaBitQKMeansParameters cp_;
    std::optional<uint32_t> seed_;  // Unknown for point-count and loaded indexes.
    // Point-count and loaded indexes build by construct, never by training.
    bool builds_by_construct_ = false;
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
    Number of centroids, from train or set_centroids.
nbits : int
    Keyword-only. Total quantization bits, 1 through 9, or 32 for owned
    float32 vectors with one-bit filtering and raw reranking.
metric : {'l2', 'ip', 'innerproduct'}, optional
    Squared L2 or inner-product distance. Default is 'l2'. Vectors are not
    automatically normalized for cosine similarity.
initializer : {'auto', 'flat', 'flat_rabitq', 'hnsw'}, optional
    Centroid routing strategy. Default is 'auto'.
seed : int, optional
    Rotation seed in [0, 2**32 - 1], also the initial cp.seed. None
    (default) draws one at random; the seed property reports it.

Notes
-----
Train the index or set its centroids, then add vectors before searching or
saving; loaded indexes are already trained. Concurrent reads are supported;
an overlapping mutation raises RuntimeError.
Data, centroids and queries must be finite float32 values within the index
coordinate range; other values raise ValueError.
The 0.7 form IvfIndex(dim, max_elements, num_clusters, nbits, ...) and
build() still work in 0.8, raise DeprecationWarning, and will be removed in
0.9.

Examples
--------
>>> import numpy as np
>>> from rabitqlib import IvfIndex
>>> x = np.random.default_rng(42).normal(size=(80, 64)).astype(np.float32)
>>> index = IvfIndex(64, 2, nbits=4, seed=7)
>>> index.train(x)
>>> ids = index.add(x)
>>> ids, distances = index.search(x[:2], k=3, nprobe=2)
>>> ids.shape, distances.shape
((2, 3), (2, 3)))doc";

    static constexpr char kAddDoc[] = R"doc(Add vectors using the trained centroids.

Parameters
----------
data : array_like of float, shape (n, dim)
    Vectors in the original, unpadded coordinates. Converted to contiguous
    float32; a copy may be made.
cluster_ids : array_like of int, shape (n,), optional
    Keyword-only. Existing centroid IDs. None routes each vector using the
    index's configured centroid initializer; routing may be approximate.
    Centroids do not move.
num_threads : int, optional
    Keyword-only. Worker count. Default is 0, which selects the available
    CPU count.
fast_quantization : bool, optional
    Keyword-only. Use faster quantizer configuration. Default is False.

Returns
-------
ids : ndarray of uint32, shape (n,)
    Assigned point IDs.

Notes
-----
Requires a trained or loaded index. Batch additions to reduce repeated
cluster copying; reserve can preallocate room. Passing the options
positionally is deprecated.)doc";

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
            py::arg("nbits"),
            py::arg("metric") = "l2",
            py::arg("initializer") = "auto",
            py::arg("seed") = py::none(),
            kIndexDoc
        )
        // Private to the 0.8 compatibility layer in rabitqlib/__init__.py.
        .def(
            py::init([](size_t dim,
                        size_t num_clusters,
                        size_t nbits,
                        const std::string& metric,
                        const std::string& initializer,
                        size_t max_elements) {
                return IvfIndex::with_point_count(
                    dim, max_elements, num_clusters, nbits, metric, initializer
                );
            }),
            py::arg("dim"),
            py::arg("num_clusters"),
            py::kw_only(),
            py::arg("nbits"),
            py::arg("metric"),
            py::arg("initializer"),
            py::arg("_max_elements")
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
    cp.num_threads; 0 selects the available CPU count.

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
            "_build",
            &IvfIndex::build,
            py::arg("data"),
            py::arg("centroids"),
            py::arg("cluster_ids"),
            py::arg("num_threads"),
            py::arg("fast_quantization")
        )
        .def(
            "add",
            &IvfIndex::add,
            py::arg("data"),
            py::kw_only(),
            py::arg("cluster_ids") = py::none(),
            py::arg("num_threads") = 0,
            py::arg("fast_quantization") = false,
            kAddDoc
        )
        .def(
            "reserve",
            &IvfIndex::reserve,
            py::arg("cluster_id"),
            py::arg("num_points"),
            R"doc(Preallocate storage for a cluster.

Parameters
----------
cluster_id : int
    Cluster ID in [0, num_clusters).
num_points : int
    Nonnegative capacity to reserve for this cluster.

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
            py::arg("nprobe"),
            py::arg("high_accuracy") = py::none(),
            py::arg("num_threads") = 1,
            R"doc(Search a batch of query vectors.

Parameters
----------
queries : array_like of float, shape (n_queries, dim)
    Vectors in the original, unpadded coordinates. Converted to contiguous
    float32; a copy may be made.
k : int
    Number of neighbors requested, between 1 and max_elements.
nprobe : int
    Positive number of clusters to probe; capped at num_clusters.
high_accuracy : bool, optional
    Enable HACC. None (default) selects it for 4-9 bits and standard
    FastScan otherwise. Raw storage uses standard FastScan.
num_threads : int, optional
    Worker count. Default is 1; 0 selects the available CPU count.

Returns
-------
ids : ndarray of uint32, shape (n_queries, k)
    Point IDs of the selected neighbors.
distances : ndarray of float32, shape (n_queries, k)
    Lower values are better; squared L2 estimates for l2 and inner-product
    distance estimates for ip. These are distances, not similarity scores.

Notes
-----
Requires a trained or loaded index. Returns (ids, distances), in that
order. Missing neighbors have ID 4294967295 and distance inf. Array shapes
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
IVF with HNSW centroid routing also writes its required sidecar file.)doc"
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
    A trained index ready for search.)doc"
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
            "max_elements",
            &IvfIndex::max_elements,
            R"doc(Logical capacity: stored points, or the build row count of a 0.7-form index before build.)doc"
        )
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
            R"doc(Whether centroids are set, by train, set_centroids, build or load.)doc"
        )
        .def_property_readonly(
            "seed",
            &IvfIndex::seed,
            R"doc(Rotation seed, or None for an index loaded from disk or created in the 0.7 form.)doc"
        )
        .def_property(
            "cp",
            &IvfIndex::cp,
            &IvfIndex::set_cp,
            py::return_value_policy::reference_internal,
            R"doc(Clustering parameters train uses, a RaBitQKMeansParameters. Defaults: niter 25, max_points_per_centroid 256, spherical for inner product, and seed equal to the index seed.)doc"
        )
        .def_property_readonly(
            "metric", &IvfIndex::metric, R"doc(Canonical metric name: 'l2' or 'ip'.)doc"
        )
        .def_property_readonly(
            "initializer",
            &IvfIndex::initializer,
            R"doc(Resolved centroid routing strategy.)doc"
        );
}
