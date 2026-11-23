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
#include "rabitqlib/index/symqg_index.hpp"
#include "rabitqlib/index/types.hpp"
#include "rabitqlib/utils/input_checks.hpp"

namespace py = pybind11;

namespace rabitqlib::python_bindings {

class SymqgIndex {
   public:
    SymqgIndex(
        size_t dim,
        size_t nbits,
        size_t max_degree,
        size_t ef_construction,
        const std::string& init,
        const std::string& metric,
        const py::object& seed
    )
        : index_(
              dim, index_options(nbits, max_degree, ef_construction, init, metric, seed)
          ) {}

    // SymphonyQG learns nothing before add: train only checks the coordinates of x as
    // add does, so the index is unchanged and a shared guard suffices. The check names
    // x, so it stays here rather than in the native train.
    void train(py::handle x, std::optional<size_t> /*num_threads*/) {
        const auto operation = access_.read();
        auto x_array = ensure_2d_array<float>(x, "x");
        const size_t dim = index_.dim();
        ensure_dimension(x_array, dim, "x dimension does not match index dim");
        const auto* x_ptr = x_array.data();
        const size_t count = static_cast<size_t>(x_array.size());
        py::gil_scoped_release release;
        check_coordinates(x_ptr, count, dim, "x");
    }

    py::array_t<int64_t> add(py::handle data, std::optional<size_t> num_threads) {
        const auto operation = access_.write();
        auto data_array = ensure_2d_array<float>(data, "data");
        ensure_dimension(
            data_array, index_.dim(), "data dimension does not match index dim"
        );
        const auto rows = static_cast<size_t>(data_array.shape(0));
        // Allocate the returned IDs before the index changes.
        auto ids = py::array_t<int64_t>(static_cast<py::ssize_t>(rows));
        const auto* data_ptr = data_array.data();
        Id first = 0;
        {
            py::gil_scoped_release release;
            first = index_.add(data_ptr, rows, num_threads.value_or(0));
        }
        std::iota(ids.mutable_data(), ids.mutable_data() + rows, first);
        return ids;
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

    static SymqgIndex load(py::handle path) {
        const auto filename = path_string(path);
        auto index = [&filename] {
            py::gil_scoped_release release;
            return rabitqlib::SymqgIndex::load(filename.c_str());
        }();
        return SymqgIndex(std::move(index));
    }

    [[nodiscard]] std::string repr() const {
        const auto operation = access_.read();
        return "SymqgIndex(dim=" + std::to_string(index_.dim()) + ", metric='" +
               metric_to_string(index_.metric()) +
               "', nbits=" + std::to_string(index_.nbits()) +
               ", num_points=" + std::to_string(index_.num_points()) +
               ", max_degree=" + std::to_string(index_.max_degree()) + ")";
    }

    [[nodiscard]] size_t dim() const {
        const auto operation = access_.read();
        return index_.dim();
    }
    [[nodiscard]] size_t max_degree() const {
        const auto operation = access_.read();
        return index_.max_degree();
    }
    [[nodiscard]] std::optional<size_t> ef_construction() const {
        const auto operation = access_.read();
        return index_.ef_construction();
    }
    [[nodiscard]] std::optional<std::string> init() const {
        const auto operation = access_.read();
        const auto init = index_.init();
        if (!init.has_value()) {
            return std::nullopt;
        }
        return *init == symqg::QGInitialization::PiPNN ? "pipnn" : "random";
    }
    [[nodiscard]] size_t num_points() const {
        const auto operation = access_.read();
        return index_.num_points();
    }
    [[nodiscard]] size_t num_stored_points() const {
        const auto operation = access_.read();
        return index_.num_stored_points();
    }
    // Always true and reads no state, so it needs no guard.
    [[nodiscard]] bool is_trained() const { return index_.is_trained(); }
    [[nodiscard]] std::string metric() const {
        const auto operation = access_.read();
        return metric_to_string(index_.metric());
    }
    [[nodiscard]] size_t nbits() const {
        const auto operation = access_.read();
        return index_.nbits();
    }
    [[nodiscard]] std::optional<uint32_t> seed() const {
        const auto operation = access_.read();
        return index_.seed();
    }

   private:
    explicit SymqgIndex(rabitqlib::SymqgIndex index) : index_(std::move(index)) {}

    static SymqgOptions index_options(
        size_t nbits,
        size_t max_degree,
        size_t ef_construction,
        const std::string& init,
        const std::string& metric,
        const py::object& seed
    ) {
        SymqgOptions options;
        options.metric = index_metric_from_string(metric);
        if (init == "pipnn") {
            options.init = symqg::QGInitialization::PiPNN;
        } else if (init == "random") {
            options.init = symqg::QGInitialization::Random;
        } else {
            throw std::invalid_argument("init must be 'random' or 'pipnn'");
        }
        if (!seed.is_none()) {
            options.seed = static_cast<uint32_t>(
                checked_seed(seed, std::numeric_limits<uint32_t>::max())
            );
        }
        options.nbits = nbits;
        options.max_degree = max_degree;
        options.ef_construction = ef_construction;
        return options;
    }

    rabitqlib::SymqgIndex index_;
    mutable IndexAccess access_;
};

}  // namespace rabitqlib::python_bindings

// Register Symqg bindings into combined module
void register_symqg(py::module_& m) {
    using namespace rabitqlib::python_bindings;

    static constexpr char kIndexDoc[] =
        R"doc(An approximate nearest-neighbor index using SymphonyQG.

Parameters
----------
dim : int
    Original vector dimension, from 64 through 65536.
nbits : {4, 8, 32}, optional
    Keyword-only. Storage bits: 32 (default) keeps raw float32 vectors, and 4
    or 8 store centroid-relative RaBitQ codes.
max_degree : int, optional
    Keyword-only. Graph degree bound; a positive multiple of 32, smaller than
    the number of vectors added. Default is 32.
ef_construction : int, optional
    Keyword-only. Positive construction search breadth fitting uint32.
    Default is 200.
init : {'pipnn', 'random'}, optional
    Keyword-only. Graph initialization method. Default is 'pipnn'.
metric : {'l2', 'ip', 'cosine'}, optional
    Keyword-only. Squared L2, inner-product or cosine distance. Default is
    'l2'. 'cosine' scales data and queries to unit length.
seed : int, optional
    Keyword-only. Rotation and graph-construction seed in [0, 2**32 - 1].
    None (default) draws one at random; the seed property reports it.

Notes
-----
There is nothing to train: train only checks its input, and is_trained is
always True. add builds the graph from all vectors at once and may be called
only once; to add more vectors, create a new index. Load an index or add
vectors before searching or saving. Concurrent reads are supported; an
overlapping mutation raises RuntimeError.
Data and queries must be finite float32 values within the index coordinate
range; other values raise ValueError.

Examples
--------
>>> import numpy as np
>>> from rabitqlib import SymqgIndex
>>> x = np.random.default_rng(42).normal(size=(80, 64)).astype(np.float32)
>>> index = SymqgIndex(64, ef_construction=40, seed=7)
>>> ids = index.add(x, num_threads=1)
>>> ids, distances = index.search(x[:2], k=3)
>>> ids.shape, distances.shape
((2, 3), (2, 3)))doc";

    static constexpr char kSearchDoc[] = R"doc(Search a batch of query vectors.

Parameters
----------
queries : array_like of float, shape (n_queries, dim)
    Vectors in the original, unpadded coordinates. Converted to contiguous
    float32; a copy may be made.
k : int
    Number of neighbors requested, at least 1. It may exceed the number of
    points; slots without a neighbor are padded as described below.
ef_search : int, optional
    Search breadth, at least 1. None (default) selects max(k, 64); smaller
    values are raised to k.
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
Returns (ids, distances), in that order. When the graph reaches fewer than k
points, the remaining neighbors have ID -1 and distance inf; test
np.isfinite(distances) to find them. Array shapes and value ranges are
checked at runtime, not by static type checkers.)doc";

    py::class_<SymqgIndex>(m, "SymqgIndex", kIndexDoc)
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
                const std::string&,
                const std::string&,
                const py::object&>(),
            py::arg("dim"),
            py::kw_only(),
            py::arg("nbits") = 32,
            py::arg("max_degree") = 32,
            py::arg("ef_construction") = 200,
            py::arg("init") = "pipnn",
            py::arg("metric") = "l2",
            py::arg("seed") = py::none(),
            kIndexDoc
        )
        .def(
            "train",
            &SymqgIndex::train,
            py::arg("x"),
            py::kw_only(),
            py::arg("num_threads") = py::none(),
            R"doc(Check training vectors; SymphonyQG has nothing to train.

Parameters
----------
x : array_like of float, shape (n, dim)
    Vectors in the original, unpadded coordinates, checked like the data of
    add.
num_threads : int, optional
    Keyword-only. Accepted for interface consistency with the other indexes.
    Default is None.

Notes
-----
Provided so that SymphonyQG follows the same train/add/search workflow as
the other indexes. It does not change the index, so it may run alongside
searches.)doc"
        )
        .def(
            "add",
            &SymqgIndex::add,
            py::arg("data"),
            py::kw_only(),
            py::arg("num_threads") = py::none(),
            R"doc(Build the graph from all vectors.

Parameters
----------
data : array_like of float, shape (n, dim)
    Vectors in the original, unpadded coordinates, with n > max_degree.
    Converted to contiguous float32; a copy may be made.
num_threads : int, optional
    Keyword-only. Worker count. None (default) and 0 select the available
    CPU count, capped by OMP_NUM_THREADS.

Returns
-------
ids : ndarray of int64, shape (n,)
    Point IDs, the row numbers of data.

Notes
-----
Uses the constructor's ef_construction, init and seed. The graph cannot be
extended: a second call raises RuntimeError, so add every vector in one
call, or create a new index to rebuild. Point IDs are the row numbers of
data. Input arrays need not be retained after this synchronous call.)doc"
        )
        .def(
            "search",
            &SymqgIndex::search,
            py::arg("queries"),
            py::arg("k"),
            py::arg("ef_search") = py::none(),
            py::kw_only(),
            py::arg("num_threads") = py::none(),
            kSearchDoc
        )
        .def(
            "save",
            &SymqgIndex::save,
            py::arg("path"),
            R"doc(Save an index with vectors to disk.

Parameters
----------
path : str, bytes, bytearray, or os.PathLike
    Index file path. PathLike objects are resolved via __fspath__.
    Byte paths use native bytes on POSIX and UTF-8 on Windows.

Notes
-----
The saved state includes the rotation used to build the index and its seed;
releases up to 0.8 cannot read the file.)doc"
        )
        .def_static(
            "load",
            &SymqgIndex::load,
            py::arg("path"),
            R"doc(Load an index and its saved rotation from disk.

Parameters
----------
path : str, bytes, bytearray, or os.PathLike
    Index file path. PathLike objects are resolved via __fspath__.
    Byte paths use native bytes on POSIX and UTF-8 on Windows.

Returns
-------
index : SymqgIndex
    An index ready for search; add raises because it already has a graph.)doc"
        )
        .def(
            "__repr__",
            &SymqgIndex::repr,
            "Return a one-line summary of configuration and counts."
        )
        .def(
            "__len__",
            &SymqgIndex::num_points,
            "Return the number of live, searchable points."
        )
        .def_property_readonly(
            "dim", &SymqgIndex::dim, R"doc(Original vector dimension.)doc"
        )
        .def_property_readonly(
            "max_degree", &SymqgIndex::max_degree, R"doc(Graph degree bound.)doc"
        )
        .def_property_readonly(
            "ef_construction",
            &SymqgIndex::ef_construction,
            R"doc(Construction search breadth of the current graph or the next add, or None for an index loaded from a file from a release up to 0.8.)doc"
        )
        .def_property_readonly(
            "init",
            &SymqgIndex::init,
            R"doc(Graph initialization method of the current graph or the next add, or None for an index loaded from a file from a release up to 0.8.)doc"
        )
        .def_property_readonly(
            "num_points",
            &SymqgIndex::num_points,
            R"doc(Number of live, searchable points.)doc"
        )
        .def_property_readonly(
            "num_stored_points",
            &SymqgIndex::num_stored_points,
            R"doc(Number of stored points, including removed points.)doc"
        )
        .def_property_readonly(
            "is_trained",
            &SymqgIndex::is_trained,
            R"doc(Always True: SymphonyQG needs no training.)doc"
        )
        .def_property_readonly(
            "metric",
            &SymqgIndex::metric,
            R"doc(Canonical metric name: 'l2', 'ip' or 'cosine'.)doc"
        )
        .def_property_readonly(
            "nbits",
            &SymqgIndex::nbits,
            R"doc(Storage bits: 32 for raw float32, or 4 or 8 for RaBitQ.)doc"
        )
        .def_property_readonly(
            "seed",
            &SymqgIndex::seed,
            R"doc(Rotation and graph-construction seed, or None for an index loaded from a file from a release up to 0.8.)doc"
        );
}
