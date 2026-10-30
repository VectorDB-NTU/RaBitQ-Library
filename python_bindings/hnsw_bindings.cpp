#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/gil.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bindings_common.hpp"
#include "rabitqlib/index/hnsw/hnsw.hpp"
#include "rabitqlib/index/types.hpp"

namespace py = pybind11;

namespace rabitqlib::python_bindings {

class HnswIndex {
   public:
    HnswIndex(
        size_t dim,
        size_t max_elements,
        size_t M,
        size_t ef_construction,
        size_t nbits,
        const std::string& metric,
        size_t random_seed
    )
        : index_(
              std::make_unique<rabitqlib::hnsw::HierarchicalNSW>(
                  max_elements,
                  dim,
                  nbits,
                  M,
                  ef_construction,
                  random_seed,
                  metric_from_string(metric)
              )
          ) {}

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
        if (data_array.shape(0) == 0 ||
            static_cast<size_t>(data_array.shape(0)) > index_->max_elements()) {
            throw std::invalid_argument(
                "number of data rows must be between 1 and max_elements; expected 1 to " +
                std::to_string(index_->max_elements()) + ", got shape " +
                array_shape(data_array)
            );
        }
        ensure_dimension(
            centroids_array,
            index_->dimension(),
            "centroid dimension does not match index dim"
        );
        const size_t num_clusters = static_cast<size_t>(centroids_array.shape(0));
        if (num_clusters == 0) {
            throw std::invalid_argument("at least one centroid is required");
        }
        auto cluster_ids_vec = checked_cluster_ids(
            cluster_ids,
            static_cast<size_t>(data_array.shape(0)),
            num_clusters,
            "cluster_ids contains an out-of-range value"
        );

        const auto* data_ptr = data_array.data();
        const auto* centroids_ptr = centroids_array.data();
        const auto rows = static_cast<size_t>(data_array.shape(0));
        {
            py::gil_scoped_release release;
            index_->construct(
                num_clusters,
                centroids_ptr,
                rows,
                data_ptr,
                cluster_ids_vec.data(),
                num_threads,
                fast_quantization
            );
        }
        built_ = true;
    }

    py::array_t<rabitqlib::PID> add(
        py::handle data,
        const py::object& cluster_ids,
        size_t num_threads,
        bool fast_quantization
    ) {
        const auto operation = access_.write();
        auto data_array = ensure_2d_array<float>(data, "data");
        if (!built_) {
            throw std::runtime_error("HnswIndex must be built or loaded before add");
        }
        ensure_dimension(
            data_array, index_->dimension(), "data dimension does not match index dim"
        );
        const auto rows = static_cast<size_t>(data_array.shape(0));

        std::vector<rabitqlib::PID> assigned;
        if (!cluster_ids.is_none()) {
            assigned = checked_cluster_ids(cluster_ids, rows, index_->num_clusters());
        }

        // Allocated before the index is touched.
        auto ids = py::array_t<rabitqlib::PID>(static_cast<py::ssize_t>(rows));
        const auto* data_ptr = data_array.data();
        const auto labels = [&] {
            py::gil_scoped_release release;
            return index_->add(
                data_ptr,
                rows,
                assigned.empty() ? nullptr : assigned.data(),
                fast_quantization,
                num_threads
            );
        }();
        std::copy(labels.begin(), labels.end(), ids.mutable_data());
        return ids;
    }

    void resize(size_t new_max_elements) {
        const auto operation = access_.write();
        if (!built_) {
            throw std::runtime_error("HnswIndex must be built or loaded before resize");
        }
        py::gil_scoped_release release;
        index_->resize(new_max_elements);
    }

    size_t remove(py::handle ids) {
        const auto operation = access_.write();
        auto ids_array = ensure_1d_integer_array(ids, "ids");
        if (!built_) {
            throw std::runtime_error("HnswIndex must be built or loaded before remove");
        }
        auto to_remove = checked_removal_ids(ids_array, index_->num_stored_points());
        py::gil_scoped_release release;
        return index_->remove(to_remove.data(), to_remove.size());
    }

    [[nodiscard]] size_t num_points() const {
        const auto operation = access_.read();
        return built_ ? index_->num_points() : 0;
    }
    [[nodiscard]] size_t num_stored_points() const {
        const auto operation = access_.read();
        return built_ ? index_->num_stored_points() : 0;
    }

    py::tuple search(py::handle queries, size_t k, size_t ef, size_t num_threads) {
        const auto operation = access_.read();
        auto query_array = ensure_2d_array<float>(queries, "queries");
        if (!built_) {
            throw std::runtime_error("HnswIndex must be built or loaded before search");
        }
        if (index_->dimension() != 0) {
            ensure_dimension(
                query_array, index_->dimension(), "query dimension does not match index dim"
            );
        }
        if (ef == 0) {
            ef = std::max<size_t>(k, 10);
        }
        if (k == 0 || k > index_->max_elements()) {
            throw std::invalid_argument("k must be between 1 and max_elements");
        }

        auto output = unfilled_search_results(static_cast<size_t>(query_array.shape(0)), k);
        auto result_ids_view = output.ids.mutable_unchecked<2>();
        auto distances_view = output.distances.mutable_unchecked<2>();

        const auto* queries_data = query_array.data();
        const auto num_queries = static_cast<size_t>(query_array.shape(0));
        std::vector<std::vector<std::pair<float, rabitqlib::PID>>> results;
        {
            py::gil_scoped_release release;
            results = index_->search(queries_data, num_queries, k, ef, num_threads);
        }

        for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(results.size()); ++i) {
            for (py::ssize_t j = 0; j < static_cast<py::ssize_t>(std::min<size_t>(
                                            k, results[static_cast<size_t>(i)].size()
                                        ));
                 ++j) {
                result_ids_view(i, j) =
                    results[static_cast<size_t>(i)][static_cast<size_t>(j)].second;
                distances_view(i, j) =
                    results[static_cast<size_t>(i)][static_cast<size_t>(j)].first;
            }
        }
        return py::make_tuple(output.ids, output.distances);
    }

    void save(py::handle path) const {
        const auto filename = path_string(path);
        const auto operation = access_.read();
        if (!built_) {
            throw std::runtime_error("HnswIndex must be built or loaded before save");
        }
        py::gil_scoped_release release;
        index_->save(filename.c_str());
    }

    static HnswIndex load(py::handle path) {
        const auto filename = path_string(path);
        HnswIndex wrapper;
        wrapper.index_ = std::make_unique<rabitqlib::hnsw::HierarchicalNSW>();
        {
            py::gil_scoped_release release;
            wrapper.index_->load(filename.c_str());
        }
        wrapper.built_ = true;
        return wrapper;
    }

    [[nodiscard]] std::string repr() const {
        const auto operation = access_.read();
        return "HnswIndex(dim=" + std::to_string(index_->dimension()) + ", metric='" +
               metric_to_string(index_->metric_type()) +
               "', nbits=" + std::to_string(index_->nbits()) +
               ", num_points=" + std::to_string(built_ ? index_->num_points() : 0) +
               ", num_stored_points=" +
               std::to_string(built_ ? index_->num_stored_points() : 0) +
               ", max_elements=" + std::to_string(index_->max_elements()) +
               ", num_clusters=" + std::to_string(index_->num_clusters()) +
               ", is_built=" + (built_ ? "True" : "False") + ")";
    }

    [[nodiscard]] size_t dim() const {
        const auto operation = access_.read();
        return index_->dimension();
    }
    [[nodiscard]] size_t max_elements() const {
        const auto operation = access_.read();
        return index_->max_elements();
    }
    [[nodiscard]] size_t nbits() const {
        const auto operation = access_.read();
        return index_->nbits();
    }
    [[nodiscard]] bool is_built() const {
        const auto operation = access_.read();
        return built_;
    }
    [[nodiscard]] size_t num_clusters() const {
        const auto operation = access_.read();
        return index_->num_clusters();
    }
    [[nodiscard]] std::string metric() const {
        const auto operation = access_.read();
        return metric_to_string(index_->metric_type());
    }

   private:
    HnswIndex() = default;

    bool built_ = false;
    mutable IndexAccess access_;
    std::unique_ptr<rabitqlib::hnsw::HierarchicalNSW> index_;
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
max_elements : int
    Storage capacity, including removed points. Use resize to grow it.
M : int, optional
    Graph connectivity, at least 2. Default is 16.
ef_construction : int, optional
    Construction search breadth. Default is 200.
nbits : int, optional
    Total quantization bits, 1 through 9. Default is 8.
metric : {'l2', 'ip', 'innerproduct'}, optional
    Squared L2 or inner-product distance. Default is 'l2'. Vectors are not
    automatically normalized for cosine similarity.
random_seed : int, optional
    Graph random seed. Default is 100.

Notes
-----
Build or load the index before searching or saving. Concurrent reads are
supported; an overlapping mutation raises RuntimeError.)doc";

    py::class_<HnswIndex>(m, "HnswIndex", kIndexDoc)
        .def(
            py::init<size_t, size_t, size_t, size_t, size_t, const std::string&, size_t>(),
            py::arg("dim"),
            py::arg("max_elements"),
            py::arg("M") = 16,
            py::arg("ef_construction") = 200,
            py::arg("nbits") = 8,
            py::arg("metric") = "l2",
            py::arg("random_seed") = 100,
            kIndexDoc
        )
        .def(
            "build",
            &HnswIndex::build,
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
The initial data must satisfy 1 <= n <= max_elements. Centroids and data
use original coordinates. The input arrays need not be retained after this
synchronous call.)doc"
        )
        .def(
            "search",
            &HnswIndex::search,
            py::arg("queries"),
            py::arg("k"),
            py::arg("ef") = 0,
            py::arg("num_threads") = 1,
            R"doc(Search a batch of query vectors.

Parameters
----------
queries : array_like of float, shape (n_queries, dim)
    Vectors in the original, unpadded coordinates. Converted to contiguous
    float32; a copy may be made.
k : int
    Number of neighbors requested, between 1 and max_elements.
ef : int, optional
    Search breadth. 0 (default) selects max(k, 10).
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
            "add",
            &HnswIndex::add,
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
The stored point count plus n must fit max_elements; resize first if
needed. Removed IDs are not reused.)doc"
        )
        .def(
            "resize",
            &HnswIndex::resize,
            py::arg("max_elements"),
            R"doc(Resize the storage capacity of a built index.

Parameters
----------
max_elements : int
    New capacity; must be at least num_stored_points.)doc"
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
            R"doc(Save a built index to disk.

Parameters
----------
path : str, bytes, bytearray, or os.PathLike
    Index file path. PathLike objects are resolved via __fspath__.
    Byte paths use native bytes on POSIX and UTF-8 on Windows.

Notes
-----
The saved state includes the rotation used to build the index.)doc"
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
    A built index ready for search.)doc"
        )
        .def(
            "__repr__",
            &HnswIndex::repr,
            "Return a one-line summary of configuration, counts and build state."
        )
        .def_property_readonly(
            "dim", &HnswIndex::dim, R"doc(Original vector dimension.)doc"
        )
        .def_property_readonly(
            "max_elements", &HnswIndex::max_elements, R"doc(Current storage capacity.)doc"
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
            "num_clusters",
            &HnswIndex::num_clusters,
            R"doc(Number of centroids used for quantization.)doc"
        )
        .def_property_readonly(
            "is_built",
            &HnswIndex::is_built,
            R"doc(Whether the index has been built or loaded.)doc"
        )
        .def_property_readonly(
            "metric", &HnswIndex::metric, R"doc(Canonical metric name: 'l2' or 'ip'.)doc"
        );
}
