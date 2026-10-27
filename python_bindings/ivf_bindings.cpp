#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/gil.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>  // IWYU pragma: keep; registers std::optional casters

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "bindings_common.hpp"
#include "rabitqlib/core/rotator.hpp"
#include "rabitqlib/index/ivf/initializer.hpp"
#include "rabitqlib/index/ivf/ivf.hpp"
#include "rabitqlib/index/types.hpp"
#include "rabitqlib/utils/packed.hpp"

namespace py = pybind11;

namespace rabitqlib::python_bindings {

namespace {
ivf::InitializerType initializer_from_string(const std::string& value) {
    if (value == "auto")
        return ivf::InitializerType::Auto;
    if (value == "flat")
        return ivf::InitializerType::Flat;
    if (value == "flat_rabitq")
        return ivf::InitializerType::FlatRaBitQ;
    if (value == "hnsw")
        return ivf::InitializerType::HNSW;
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
        size_t max_elements,
        size_t num_clusters,
        size_t nbits,
        const std::string& metric = "l2",
        const std::string& initializer = "auto"
    ) {
        const auto metric_type = metric_from_string(metric);
        index_ = std::make_unique<rabitqlib::ivf::IVF>(
            max_elements,
            dim,
            num_clusters,
            nbits,
            metric_type,
            rabitqlib::RotatorType::FhtKacRotator,
            initializer_from_string(initializer)
        );
    }

    void build(
        py::handle data,
        py::handle centroids,
        py::handle cluster_ids,
        size_t num_threads = 1,
        bool fast_quantization = false
    ) {
        const auto operation = access_.write();
        auto data_array = ensure_2d_array<float>(data, "data");
        auto centroids_array = ensure_2d_array<float>(centroids, "centroids");

        if (static_cast<size_t>(data_array.shape(1)) != index_->dimension()) {
            throw std::invalid_argument("data dimension does not match index dim");
        }
        if (static_cast<size_t>(data_array.shape(0)) != index_->max_elements()) {
            throw std::invalid_argument(
                "number of data rows must match index max_elements"
            );
        }
        if (static_cast<size_t>(centroids_array.shape(1)) != index_->dimension()) {
            throw std::invalid_argument("centroid dimension does not match index dim");
        }
        if (static_cast<size_t>(centroids_array.shape(0)) != index_->num_clusters()) {
            throw std::invalid_argument(
                "number of centroid rows must match index num_clusters"
            );
        }
        auto cluster_ids_vec = checked_cluster_ids(
            cluster_ids, static_cast<size_t>(data_array.shape(0)), index_->num_clusters()
        );

        const auto* data_ptr = data_array.data();
        const auto* centroids_ptr = centroids_array.data();
        const auto* cluster_ids_ptr = cluster_ids_vec.data();
        {
            py::gil_scoped_release release;
            index_->construct(
                data_ptr, centroids_ptr, cluster_ids_ptr, fast_quantization, num_threads
            );
        }
        built_ = true;
    }

    py::array_t<rabitqlib::PID> add(
        py::handle data,
        const py::object& cluster_ids,
        size_t num_threads = 1,
        bool fast_quantization = false
    ) {
        const auto operation = access_.write();
        auto data_array = ensure_2d_array<float>(data, "data");
        if (!built_) {
            throw std::runtime_error("IvfIndex must be built or loaded before add");
        }
        if (static_cast<size_t>(data_array.shape(1)) != index_->dimension()) {
            throw std::invalid_argument("data dimension does not match index dim");
        }
        const auto rows = static_cast<size_t>(data_array.shape(0));

        std::vector<rabitqlib::PID> assigned;
        if (!cluster_ids.is_none()) {
            assigned = checked_cluster_ids(cluster_ids, rows, index_->num_clusters());
        }

        const size_t first = index_->num_stored_points();
        // Allocate the returned IDs before committing any changes to the index.
        auto ids = py::array_t<rabitqlib::PID>(static_cast<py::ssize_t>(rows));
        auto* ids_data = ids.mutable_data();
        for (size_t i = 0; i < rows; ++i) {
            ids_data[i] = static_cast<rabitqlib::PID>(first + i);
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
        if (!built_) {
            throw std::runtime_error("IvfIndex must be built or loaded before reserve");
        }
        if (cluster_id < 0 || static_cast<uint64_t>(cluster_id) >= index_->num_clusters()) {
            throw std::invalid_argument("cluster_id must be in [0, num_clusters)");
        }
        if (num_points < 0) {
            throw std::invalid_argument("num_points must be non-negative");
        }
        py::gil_scoped_release release;
        index_->reserve(
            static_cast<rabitqlib::PID>(cluster_id), static_cast<size_t>(num_points)
        );
    }

    size_t remove(py::handle ids) {
        const auto operation = access_.write();
        auto ids_array = ensure_1d_integer_array(ids, "ids");
        if (!built_) {
            throw std::runtime_error("IvfIndex must be built or loaded before remove");
        }
        std::vector<rabitqlib::PID> to_remove;
        to_remove.reserve(static_cast<size_t>(ids_array.shape(0)));
        const rabitqlib::detail::ConstPackedArrayView<int64_t> values(
            reinterpret_cast<const char*>(ids_array.data())
        );
        for (py::ssize_t i = 0; i < ids_array.shape(0); ++i) {
            const int64_t id = values[static_cast<size_t>(i)];
            if (id < 0 || static_cast<uint64_t>(id) >= index_->num_stored_points()) {
                throw std::invalid_argument("ids must be in [0, num_stored_points)");
            }
            to_remove.push_back(static_cast<rabitqlib::PID>(id));
        }
        py::gil_scoped_release release;
        return index_->remove(to_remove.data(), to_remove.size());
    }

    py::tuple search(
        py::handle queries,
        size_t k,
        size_t nprobe,
        std::optional<bool> high_accuracy = std::nullopt,
        size_t num_threads = 1
    ) {
        const auto operation = access_.read();
        auto query_array = ensure_2d_array<float>(queries, "queries");
        if (!built_) {
            throw std::runtime_error("IvfIndex must be built or loaded before search");
        }
        if (static_cast<size_t>(query_array.shape(1)) != index_->dimension()) {
            throw std::invalid_argument("query dimension does not match index dim");
        }
        if (k == 0 || k > index_->max_elements()) {
            throw std::invalid_argument("k must be between 1 and max_elements");
        }
        if (nprobe == 0) {
            throw std::invalid_argument("nprobe must be positive");
        }

        const size_t nq = static_cast<size_t>(query_array.shape(0));
        const auto shape = std::vector<py::ssize_t>{
            static_cast<py::ssize_t>(nq), static_cast<py::ssize_t>(k)};
        auto ids = py::array_t<rabitqlib::PID>(shape);
        auto dists = py::array_t<float>(shape);
        auto* ids_data = ids.mutable_data();
        auto* dists_data = dists.mutable_data();
        std::fill(ids_data, ids_data + ids.size(), rabitqlib::kPidMax);
        std::fill(
            dists_data, dists_data + dists.size(), std::numeric_limits<float>::infinity()
        );

        const auto* queries_data = query_array.data();
        {
            py::gil_scoped_release release;
            index_->search_batch(
                queries_data,
                nq,
                k,
                nprobe,
                ids_data,
                dists_data,
                high_accuracy,
                num_threads
            );
        }

        return py::make_tuple(ids, dists);
    }

    void save(const std::string& path) const {
        const auto operation = access_.read();
        if (!built_) {
            throw std::runtime_error("IvfIndex must be built or loaded before save");
        }
        py::gil_scoped_release release;
        index_->save(path.c_str());
    }

    static IvfIndex load(const std::string& path) {
        IvfIndex wrapper;
        wrapper.index_ = std::make_unique<rabitqlib::ivf::IVF>();
        {
            py::gil_scoped_release release;
            wrapper.index_->load(path.c_str());
        }
        wrapper.built_ = true;
        return wrapper;
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
    [[nodiscard]] bool is_built() const {
        const auto operation = access_.read();
        return built_;
    }
    [[nodiscard]] std::string initializer() const {
        const auto operation = access_.read();
        return initializer_to_string(index_->initializer_type());
    }
    [[nodiscard]] std::string metric() const {
        const auto operation = access_.read();
        return metric_to_string(index_->metric_type());
    }

   private:
    IvfIndex() = default;

    bool built_ = false;
    mutable IndexAccess access_;
    std::unique_ptr<rabitqlib::ivf::IVF> index_;
};

}  // namespace rabitqlib::python_bindings

// Register IVF bindings into combined module
void register_ivf(py::module_& m) {
    using namespace rabitqlib::python_bindings;

    py::class_<IvfIndex>(
        m, "IvfIndex", R"doc(An approximate nearest-neighbor index using IVF.

Parameters
----------
dim : int
    Original vector dimension, from 64 through 65536.
max_elements : int
    Number of vectors required by the initial build. Later adds grow
    storage.
num_clusters : int
    Number of supplied centroids.
nbits : int
    Total quantization bits, 1 through 9, or 32 for owned float32 vectors
    with one-bit filtering and raw reranking.
metric : {'l2', 'ip', 'innerproduct'}, optional
    Squared L2 or inner-product distance. Default is 'l2'. Vectors are not
    automatically normalized for cosine similarity.
initializer : {'auto', 'flat', 'flat_rabitq', 'hnsw'}, optional
    Centroid routing strategy. Default is 'auto'.

Notes
-----
Build or load the index before searching or saving. Concurrent reads are
supported; an overlapping mutation raises RuntimeError.

Examples
--------
>>> import numpy as np
>>> from rabitqlib import IvfIndex
>>> x = np.random.default_rng(42).normal(size=(80, 64)).astype(np.float32)
>>> index = IvfIndex(64, len(x), num_clusters=1, nbits=4)
>>> index.build(x, x.mean(axis=0, keepdims=True), np.zeros(len(x), dtype=np.uint32))
>>> ids, distances = index.search(x[:2], k=3, nprobe=1)
>>> ids.shape, distances.shape
((2, 3), (2, 3)))doc"
    )
        .def(
            py::init<
                size_t,
                size_t,
                size_t,
                size_t,
                const std::string&,
                const std::string&>(),
            py::arg("dim"),
            py::arg("max_elements"),
            py::arg("num_clusters"),
            py::arg("nbits"),
            py::arg("metric") = "l2",
            py::arg("initializer") = "auto",
            R"doc(An approximate nearest-neighbor index using IVF.

Parameters
----------
dim : int
    Original vector dimension, from 64 through 65536.
max_elements : int
    Number of vectors required by the initial build. Later adds grow
    storage.
num_clusters : int
    Number of supplied centroids.
nbits : int
    Total quantization bits, 1 through 9, or 32 for owned float32 vectors
    with one-bit filtering and raw reranking.
metric : {'l2', 'ip', 'innerproduct'}, optional
    Squared L2 or inner-product distance. Default is 'l2'. Vectors are not
    automatically normalized for cosine similarity.
initializer : {'auto', 'flat', 'flat_rabitq', 'hnsw'}, optional
    Centroid routing strategy. Default is 'auto'.

Notes
-----
Build or load the index before searching or saving. Concurrent reads are
supported; an overlapping mutation raises RuntimeError.

Examples
--------
>>> import numpy as np
>>> from rabitqlib import IvfIndex
>>> x = np.random.default_rng(42).normal(size=(80, 64)).astype(np.float32)
>>> index = IvfIndex(64, len(x), num_clusters=1, nbits=4)
>>> index.build(x, x.mean(axis=0, keepdims=True), np.zeros(len(x), dtype=np.uint32))
>>> ids, distances = index.search(x[:2], k=3, nprobe=1)
>>> ids.shape, distances.shape
((2, 3), (2, 3)))doc"
        )
        .def(
            "build",
            &IvfIndex::build,
            py::arg("data"),
            py::arg("centroids"),
            py::arg("cluster_ids"),
            py::arg("num_threads") = 1,
            py::arg("fast_quantization") = false,
            R"doc(Build the index from vectors and supplied centroid assignments.

Parameters
----------
data : array_like of float, shape (n, dim)
    Vectors in the original, unpadded coordinates. Converted to contiguous
    float32; a copy may be made.
centroids : array_like of float, shape (num_clusters, dim)
    Vectors in the original, unpadded coordinates. Converted to contiguous
    float32; a copy may be made.
cluster_ids : array_like of int, shape (n,)
    Centroid ID per vector, in [0, num_clusters). Floating-point and
    boolean IDs are rejected.
num_threads : int, optional
    Worker count. Default is 1; 0 selects the available CPU count.
fast_quantization : bool, optional
    Use faster quantizer configuration. Default is False.

Notes
-----
The initial data must satisfy n == max_elements. Centroids and data use
original coordinates. The input arrays need not be retained after this
synchronous call.)doc"
        )
        .def(
            "add",
            &IvfIndex::add,
            py::arg("data"),
            py::arg("cluster_ids") = py::none(),
            py::arg("num_threads") = 1,
            py::arg("fast_quantization") = false,
            R"doc(Append vectors using the existing centroids.

Parameters
----------
data : array_like of float, shape (n, dim)
    Vectors in the original, unpadded coordinates. Converted to contiguous
    float32; a copy may be made.
cluster_ids : array_like of int, shape (n,), optional
    Existing centroid IDs. None assigns each vector to its nearest
    centroid. Centroids do not move.
num_threads : int, optional
    Worker count. Default is 1; 0 selects the available CPU count.
fast_quantization : bool, optional
    Use faster quantizer configuration. Default is False.

Returns
-------
ids : ndarray of uint32, shape (n,)
    Assigned point IDs.

Notes
-----
Batch additions to reduce repeated cluster copying; reserve can preallocate
room.)doc"
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
Requires a built index. Reserved room is not saved.)doc"
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
Returns (ids, distances), in that order. Missing neighbors have ID
4294967295 and distance inf. Array shapes and value ranges are checked at
runtime, not by static type checkers.)doc"
        )
        .def(
            "save",
            &IvfIndex::save,
            py::arg("path"),
            R"doc(Save a built index to disk.

Parameters
----------
path : str
    Index file path. Pass str(path) for pathlib.Path objects.

Notes
-----
IVF with HNSW centroid routing also writes its required sidecar file.)doc"
        )
        .def_static(
            "load",
            &IvfIndex::load,
            py::arg("path"),
            R"doc(Load an index and its saved rotation from disk.

Parameters
----------
path : str
    Index file path. Pass str(path) for pathlib.Path objects.

Returns
-------
index : IvfIndex
    A built index ready for search.)doc"
        )
        .def_property_readonly("dim", &IvfIndex::dim, R"doc(Original vector dimension.)doc")
        .def_property_readonly(
            "max_elements", &IvfIndex::max_elements, R"doc(Current storage capacity.)doc"
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
            "is_built",
            &IvfIndex::is_built,
            R"doc(Whether the index has been built or loaded.)doc"
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
