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
#include <utility>
#include <vector>

#include "bindings_common.hpp"
#include "rabitqlib/clustering/rabitqkmeans.hpp"
#include "rabitqlib/index/detail/buffer.hpp"
#include "rabitqlib/index/hnsw/hnsw.hpp"
#include "rabitqlib/index/training.hpp"
#include "rabitqlib/index/types.hpp"
#include "rabitqlib/utils/input_checks.hpp"

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
    ) {
        if (num_clusters == 0 || num_clusters > buffer::kSearchBufferMaxPointCount) {
            throw std::invalid_argument(
                "num_clusters must be in [1, " +
                std::to_string(buffer::kSearchBufferMaxPointCount) + "]"
            );
        }
        const auto metric_type = metric_from_string(metric);
        const auto graph_seed = checked_seed(seed, std::numeric_limits<size_t>::max());
        // Capacity grows on add; resize preallocates.
        index_ = std::make_unique<hnsw::HierarchicalNSW>(
            1, dim, nbits, M, ef_construction, graph_seed, metric_type
        );
        num_clusters_ = num_clusters;
        seed_ = graph_seed;
        cp_ = default_training_parameters(metric_type, static_cast<uint32_t>(graph_seed));
    }

    // The 0.7 form, whose max_elements becomes the initial capacity.
    static HnswIndex with_point_count(
        size_t dim,
        size_t max_elements,
        size_t M,
        size_t ef_construction,
        size_t nbits,
        const std::string& metric,
        const py::object& random_seed
    ) {
        const auto metric_type = metric_from_string(metric);
        const auto graph_seed =
            checked_seed(random_seed, std::numeric_limits<size_t>::max());
        HnswIndex wrapper;
        wrapper.index_ = std::make_unique<hnsw::HierarchicalNSW>(
            max_elements, dim, nbits, M, ef_construction, graph_seed, metric_type
        );
        wrapper.num_clusters_ = kDefaultNumClusters;
        wrapper.seed_ = graph_seed;
        wrapper.cp_ =
            default_training_parameters(metric_type, static_cast<uint32_t>(graph_seed));
        return wrapper;
    }

    void train(py::handle x, std::optional<size_t> num_threads) {
        const auto operation = access_.write();
        auto x_array = ensure_2d_array<float>(x, "x");
        check_trainable();
        ensure_dimension(
            x_array, index_->dimension(), "x dimension does not match index dim"
        );
        const auto rows = training_rows(x_array, num_clusters_);
        const auto cp = training_parameters(cp_, num_threads);
        const auto* x_ptr = x_array.data();
        const size_t num_clusters = num_clusters_;
        py::gil_scoped_release release;
        index_->train(num_clusters, rows, x_ptr, cp);
    }

    void set_centroids(py::handle centroids) {
        const auto operation = access_.write();
        auto centroids_array = ensure_2d_array<float>(centroids, "centroids");
        check_trainable();
        check_centroid_shape(centroids_array, num_clusters_, index_->dimension());
        const auto* centroids_ptr = centroids_array.data();
        const size_t num_clusters = num_clusters_;
        py::gil_scoped_release release;
        index_->set_centroids(num_clusters, centroids_ptr);
    }

    // Deprecated: the 0.7 build, which takes any number of centroids. An untrained
    // index is constructed; a trained one without points takes the centroids and
    // adds the data, which links it the same way.
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
        if (data_array.shape(0) == 0) {
            throw std::invalid_argument(
                "number of data rows must be at least 1; got shape " +
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
        const auto rows = static_cast<size_t>(data_array.shape(0));
        auto cluster_ids_vec = checked_cluster_ids(
            cluster_ids, rows, num_clusters, "cluster_ids contains an out-of-range value"
        );

        const auto* data_ptr = data_array.data();
        const auto* centroids_ptr = centroids_array.data();
        const size_t dim = index_->dimension();
        {
            py::gil_scoped_release release;
            const bool empty = index_->num_stored_points() == 0;
            if (empty) {
                // Invalid centroids must not grow or retrain the index first.
                check_coordinates(
                    centroids_ptr, num_clusters * dim, dim, "HNSW construction centroids"
                );
                grow_capacity(rows, data_ptr, "HNSW construction data");
            }
            if (empty && index_->is_trained()) {
                index_->set_centroids(num_clusters, centroids_ptr);
                index_->add(
                    data_ptr, rows, cluster_ids_vec.data(), fast_quantization, num_threads
                );
            } else {
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
        }
        num_clusters_ = num_clusters;
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

        // Allocated before the index is touched.
        auto ids = py::array_t<PID>(static_cast<py::ssize_t>(rows));
        const auto* data_ptr = data_array.data();
        const auto labels = [&] {
            py::gil_scoped_release release;
            grow_capacity(rows, data_ptr, "HNSW add data");
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
        py::gil_scoped_release release;
        index_->resize(new_max_elements);
    }

    size_t remove(py::handle ids) {
        const auto operation = access_.write();
        auto ids_array = ensure_1d_integer_array(ids, "ids");
        check_trained("remove");
        auto to_remove = checked_removal_ids(ids_array, index_->num_stored_points());
        py::gil_scoped_release release;
        return index_->remove(to_remove.data(), to_remove.size());
    }

    [[nodiscard]] size_t num_points() const {
        const auto operation = access_.read();
        return index_->num_points();
    }
    [[nodiscard]] size_t num_stored_points() const {
        const auto operation = access_.read();
        return index_->num_stored_points();
    }

    py::tuple search(
        py::handle queries, size_t k, std::optional<size_t> ef, size_t num_threads
    ) {
        const auto operation = access_.read();
        auto query_array = ensure_2d_array<float>(queries, "queries");
        check_trained("search");
        ensure_dimension(
            query_array, index_->dimension(), "query dimension does not match index dim"
        );
        // None and 0 select max(k, 10), as SymqgIndex does.
        const size_t ef_search = ef.value_or(0) == 0 ? std::max<size_t>(k, 10) : *ef;
        // Capacity grows geometrically, so bound k by what is stored, as IVF does.
        if (k == 0 || k > index_->num_stored_points()) {
            throw std::invalid_argument("k must be between 1 and num_stored_points");
        }

        auto output = unfilled_search_results(static_cast<size_t>(query_array.shape(0)), k);
        auto result_ids_view = output.ids.mutable_unchecked<2>();
        auto distances_view = output.distances.mutable_unchecked<2>();

        const auto* queries_data = query_array.data();
        const auto num_queries = static_cast<size_t>(query_array.shape(0));
        std::vector<std::vector<std::pair<float, PID>>> results;
        {
            py::gil_scoped_release release;
            results = index_->search(queries_data, num_queries, k, ef_search, num_threads);
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
        check_trained("save");
        if (index_->num_stored_points() == 0) {
            throw std::runtime_error("HnswIndex has no points to save; add points first");
        }
        py::gil_scoped_release release;
        index_->save(filename.c_str());
    }

    static HnswIndex load(py::handle path) {
        const auto filename = path_string(path);
        HnswIndex wrapper;
        wrapper.index_ = std::make_unique<hnsw::HierarchicalNSW>();
        {
            py::gil_scoped_release release;
            wrapper.index_->load(filename.c_str());
        }
        // Untrained, empty files from 0.7 have no clusters; train them with the default.
        wrapper.num_clusters_ = wrapper.index_->is_trained()
                                    ? wrapper.index_->num_clusters()
                                    : kDefaultNumClusters;
        wrapper.cp_ = default_training_parameters(
            wrapper.index_->metric_type(), rabitqkmeans::RaBitQKMeansParameters{}.seed
        );
        return wrapper;
    }

    [[nodiscard]] std::string repr() const {
        const auto operation = access_.read();
        return "HnswIndex(dim=" + std::to_string(index_->dimension()) + ", metric='" +
               metric_to_string(index_->metric_type()) +
               "', nbits=" + std::to_string(index_->nbits()) +
               ", num_points=" + std::to_string(index_->num_points()) +
               ", num_stored_points=" + std::to_string(index_->num_stored_points()) +
               ", max_elements=" + std::to_string(index_->max_elements()) +
               ", num_clusters=" + std::to_string(current_num_clusters()) +
               ", is_trained=" + (index_->is_trained() ? "True" : "False") + ")";
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
    [[nodiscard]] bool is_trained() const {
        const auto operation = access_.read();
        return index_->is_trained();
    }
    [[nodiscard]] size_t num_clusters() const {
        const auto operation = access_.read();
        return current_num_clusters();
    }
    [[nodiscard]] std::optional<size_t> seed() const {
        const auto operation = access_.read();
        return seed_;
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
    static constexpr size_t kDefaultNumClusters = 16;

    HnswIndex() = default;

    // The configured count until centroids are set, then the trained count.
    [[nodiscard]] size_t current_num_clusters() const {
        return index_->is_trained() ? index_->num_clusters() : num_clusters_;
    }

    void check_trained(const char* operation) const {
        if (!index_->is_trained()) {
            throw std::runtime_error(
                std::string("HnswIndex must be trained or loaded before ") + operation
            );
        }
    }

    void check_trainable() const {
        if (index_->num_stored_points() != 0) {
            throw std::runtime_error("HnswIndex cannot be trained once it stores points");
        }
    }

    // Grow capacity geometrically to fit `rows` more points from `data`, checked
    // first so that invalid input leaves the capacity unchanged. Leaves IDs beyond
    // the supported range for add to reject. Called without the GIL.
    void grow_capacity(size_t rows, const float* data, const char* what) {
        const size_t capacity = index_->max_elements();
        const size_t stored = index_->num_stored_points();
        if (rows <= capacity - stored ||
            rows > buffer::kSearchBufferMaxPointCount - stored) {
            return;
        }
        const size_t dim = index_->dimension();
        check_coordinates(data, rows * dim, dim, what);
        const size_t needed = stored + rows;
        const size_t doubled =
            std::min(capacity * 2, static_cast<size_t>(buffer::kSearchBufferMaxPointCount));
        index_->resize(std::max(needed, doubled));
    }

    mutable IndexAccess access_;
    std::unique_ptr<hnsw::HierarchicalNSW> index_;
    rabitqkmeans::RaBitQKMeansParameters cp_;
    size_t num_clusters_ = 0;     // Centroid count train and set_centroids expect.
    std::optional<size_t> seed_;  // Unknown for loaded indexes.
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
nbits : int
    Keyword-only. Total quantization bits, 1 through 9.
num_clusters : int, optional
    Keyword-only. Number of centroids train or set_centroids provides.
    Default is 16.
M : int, optional
    Keyword-only. Graph connectivity, at least 2; the base layer keeps up to
    2 * M edges. Default is 16.
ef_construction : int, optional
    Keyword-only. Construction search breadth. Default is 200.
metric : {'l2', 'ip', 'innerproduct'}, optional
    Keyword-only. Squared L2 or inner-product distance. Default is 'l2'.
    Vectors are not automatically normalized for cosine similarity.
seed : int, optional
    Keyword-only. Graph and rotation seed, also the initial cp.seed (its low
    32 bits). Default is 100.

Notes
-----
Train the index or set its centroids, then add vectors before searching or
saving; loaded indexes are already trained. Capacity grows automatically
on add; resize can preallocate it. Concurrent reads are supported; an
overlapping mutation raises RuntimeError.
Data, centroids and queries must be finite float32 values within the index
coordinate range; other values raise ValueError.
With more than one thread, add links points in a nondeterministic order;
pass num_threads=1 for a reproducible graph.
The 0.7 forms HnswIndex(dim, max_elements, ...), random_seed= and build()
still work in 0.8, raise DeprecationWarning, and will be removed in 0.9.

Examples
--------
>>> import numpy as np
>>> from rabitqlib import HnswIndex
>>> x = np.random.default_rng(42).normal(size=(80, 64)).astype(np.float32)
>>> index = HnswIndex(64, nbits=4, num_clusters=2)
>>> index.train(x)
>>> ids = index.add(x, num_threads=1)
>>> ids, distances = index.search(x[:2], k=3)
>>> ids.shape, distances.shape
((2, 3), (2, 3)))doc";

    static constexpr char kAddDoc[] = R"doc(Add vectors using the trained centroids.

Parameters
----------
data : array_like of float, shape (n, dim)
    Vectors in the original, unpadded coordinates. Converted to contiguous
    float32; a copy may be made.
cluster_ids : array_like of int, shape (n,), optional
    Keyword-only. Existing centroid IDs. None assigns each vector to its
    nearest centroid. Centroids do not move.
num_threads : int, optional
    Keyword-only. Worker count. Default is 0, which selects the available
    CPU count; more than one thread links nondeterministically, so pass 1
    for a reproducible graph.
fast_quantization : bool, optional
    Keyword-only. Use faster quantizer configuration. Default is False.

Returns
-------
ids : ndarray of uint32, shape (n,)
    Assigned point IDs.

Notes
-----
Requires a trained or loaded index. The first add into an empty index links
vectors with their exact coordinates; later adds link through the stored
codes. Capacity grows geometrically when needed. Removed IDs are not
reused. Passing the options positionally is deprecated.)doc";

    py::class_<HnswIndex>(m, "HnswIndex", kIndexDoc)
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
            py::arg("nbits"),
            py::arg("num_clusters") = 16,
            py::arg("M") = 16,
            py::arg("ef_construction") = 200,
            py::arg("metric") = "l2",
            py::arg("seed") = 100,
            kIndexDoc
        )
        // Private to the 0.8 compatibility layer in rabitqlib/__init__.py.
        .def(
            py::init(&HnswIndex::with_point_count),
            py::arg("dim"),
            py::kw_only(),
            py::arg("_max_elements"),
            py::arg("M"),
            py::arg("ef_construction"),
            py::arg("nbits"),
            py::arg("metric"),
            py::arg("random_seed")
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
    cp.num_threads; 0 selects the available CPU count.

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
            "_build",
            &HnswIndex::build,
            py::arg("data"),
            py::arg("centroids"),
            py::arg("cluster_ids"),
            py::arg("num_threads"),
            py::arg("fast_quantization")
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
    Number of neighbors requested, between 1 and num_stored_points.
ef : int, optional
    Search breadth. None or 0 (default) selects max(k, 10); values above the stored
    point count are clamped.
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
            "add",
            &HnswIndex::add,
            py::arg("data"),
            py::kw_only(),
            py::arg("cluster_ids") = py::none(),
            py::arg("num_threads") = 0,
            py::arg("fast_quantization") = false,
            kAddDoc
        )
        .def(
            "resize",
            &HnswIndex::resize,
            py::arg("max_elements"),
            R"doc(Change the storage capacity, for example to preallocate before add.

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
            R"doc(Save an index with stored vectors to disk.

Parameters
----------
path : str, bytes, bytearray, or os.PathLike
    Index file path. PathLike objects are resolved via __fspath__.
    Byte paths use native bytes on POSIX and UTF-8 on Windows.

Notes
-----
A trained index without vectors raises RuntimeError; add vectors first.
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
    A trained index ready for search.)doc"
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
            R"doc(Number of centroids: the configured count until centroids are set.)doc"
        )
        .def_property_readonly(
            "is_trained",
            &HnswIndex::is_trained,
            R"doc(Whether centroids are set, by train, set_centroids, build or load.)doc"
        )
        .def_property_readonly(
            "seed",
            &HnswIndex::seed,
            R"doc(Graph and rotation seed, or None for an index loaded from disk.)doc"
        )
        .def_property(
            "cp",
            &HnswIndex::cp,
            &HnswIndex::set_cp,
            py::return_value_policy::reference_internal,
            R"doc(Clustering parameters train uses, a RaBitQKMeansParameters. Defaults: niter 25, max_points_per_centroid 256, spherical for inner product, and seed equal to the low 32 bits of the index seed.)doc"
        )
        .def_property_readonly(
            "metric", &HnswIndex::metric, R"doc(Canonical metric name: 'l2' or 'ip'.)doc"
        );
}
