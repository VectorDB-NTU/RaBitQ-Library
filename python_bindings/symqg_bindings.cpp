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
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bindings_common.hpp"
#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/core/rotator.hpp"
#include "rabitqlib/fastscan/pack_codes.hpp"
#include "rabitqlib/index/symqg/qg.hpp"
#include "rabitqlib/index/symqg/qg_builder.hpp"
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
        : dim_(dim)
        , max_degree_(max_degree)
        , metric_type_(metric_from_string(metric))
        , ef_construction_(ef_construction)
        , init_(init) {
        // Reject now what add would reject only after converting all the data.
        if (dim < 64 || dim > rotator_impl::FhtKacRotator::kMaxDim) {
            throw std::invalid_argument("Unsupported dimension for FhtKacRotator");
        }
        check_max_degree(max_degree);
        if (nbits != 4 && nbits != 8 && nbits != 32) {
            throw std::invalid_argument("nbits must be 4, 8, or 32");
        }
        quantization_bits_ = nbits == 32 ? 0 : nbits;
        check_construction(ef_construction, init);
        seed_given_ = !seed.is_none();
        seed_ = seed_given_ ? static_cast<uint32_t>(
                                  checked_seed(seed, std::numeric_limits<uint32_t>::max())
                              )
                            : random_seed();
    }

    // SymphonyQG learns nothing before add: train only checks x as add would, so
    // the index is unchanged and a shared guard suffices.
    void train(py::handle x, std::optional<size_t> /*num_threads*/) {
        const auto operation = access_.read();
        auto x_array = ensure_2d_array<float>(x, "x");
        ensure_dimension(x_array, dim_, "x dimension does not match index dim");
        const auto* x_ptr = x_array.data();
        const size_t count = static_cast<size_t>(x_array.size());
        py::gil_scoped_release release;
        check_coordinates(x_ptr, count, dim_, "x");
    }

    void add(py::handle data, size_t num_threads) {
        const auto operation = access_.write();
        if (built()) {
            throw std::runtime_error(
                "SymqgIndex does not support incremental addition; create a new index "
                "and add all vectors at once"
            );
        }
        // Only a loaded index lacks these settings, and it is already built.
        build_graph(data, *ef_construction_, num_threads, *init_, *seed_);
    }

    // Deprecated: the 0.7 build, which replaces any existing graph. Without a
    // user seed, each build draws a new one, as 0.7 did.
    void build(
        py::handle data, size_t ef_construction, size_t num_threads, const std::string& init
    ) {
        const auto operation = access_.write();
        check_construction(ef_construction, init);
        const uint32_t seed = seed_given_ ? *seed_ : random_seed();
        build_graph(data, ef_construction, num_threads, init, seed);
        ef_construction_ = ef_construction;
        init_ = init;
        seed_ = seed;
    }

    py::tuple search(
        py::handle queries, size_t k, std::optional<size_t> ef, size_t num_threads
    ) {
        const auto operation = access_.read();
        auto query_array = ensure_2d_array<float>(queries, "queries");
        check_built("search");
        if (k == 0 || k > index_->num_points()) {
            throw std::invalid_argument("k must be between 1 and num_points");
        }
        // As in HnswIndex: no or zero ef selects max(k, 10), and ef is at least k.
        const size_t ef_search =
            std::max(ef.value_or(0) == 0 ? std::max<size_t>(k, 10) : *ef, k);
        ensure_dimension(query_array, dim_, "query dimension does not match index dim");

        const size_t num_queries = static_cast<size_t>(query_array.shape(0));
        const auto shape = std::vector<py::ssize_t>{
            static_cast<py::ssize_t>(num_queries), static_cast<py::ssize_t>(k)};
        auto result_ids = py::array_t<PID>(shape);
        auto distances = py::array_t<float>(shape);
        auto* ids_data = result_ids.mutable_data();
        auto* distances_data = distances.mutable_data();
        const auto* queries_data = query_array.data();
        {
            py::gil_scoped_release release;
            index_->search_batch_with_ef(
                queries_data,
                num_queries,
                static_cast<uint32_t>(k),
                ids_data,
                distances_data,
                ef_search,
                num_threads
            );
        }

        return py::make_tuple(result_ids, distances);
    }

    void save(py::handle path) const {
        const auto filename = path_string(path);
        const auto operation = access_.read();
        check_built("save");
        py::gil_scoped_release release;
        index_->save(filename.c_str());
    }

    static SymqgIndex load(py::handle path) {
        const auto filename = path_string(path);
        SymqgIndex wrapper;
        wrapper.index_ = std::make_unique<symqg::QuantizedGraph<float>>();
        {
            py::gil_scoped_release release;
            wrapper.index_->load(filename.c_str());
        }
        wrapper.dim_ = wrapper.index_->dimension();
        wrapper.max_degree_ = wrapper.index_->degree_bound();
        wrapper.metric_type_ = wrapper.index_->metric_type();
        wrapper.quantization_bits_ = wrapper.index_->quantization_bits();
        // Files store neither the seed nor the construction settings.
        wrapper.ef_construction_.reset();
        wrapper.init_.reset();
        return wrapper;
    }

    [[nodiscard]] std::string repr() const {
        const auto operation = access_.read();
        return "SymqgIndex(dim=" + std::to_string(dim_) + ", metric='" +
               metric_to_string(metric_type_) + "', nbits=" + std::to_string(nbits()) +
               ", num_points=" + std::to_string(built() ? index_->num_points() : 0) +
               ", max_degree=" + std::to_string(max_degree_) + ")";
    }

    [[nodiscard]] size_t dim() const {
        const auto operation = access_.read();
        return dim_;
    }
    [[nodiscard]] size_t max_degree() const {
        const auto operation = access_.read();
        return max_degree_;
    }
    [[nodiscard]] std::optional<size_t> ef_construction() const {
        const auto operation = access_.read();
        return ef_construction_;
    }
    [[nodiscard]] std::optional<std::string> init() const {
        const auto operation = access_.read();
        return init_;
    }
    [[nodiscard]] size_t num_points() const {
        const auto operation = access_.read();
        return built() ? index_->num_points() : 0;
    }
    [[nodiscard]] size_t num_stored_points() const {
        const auto operation = access_.read();
        return built() ? index_->num_stored_points() : 0;
    }
    [[nodiscard]] size_t max_elements() const {
        const auto operation = access_.read();
        return built() ? index_->max_elements() : 0;
    }
    [[nodiscard]] std::string metric() const {
        const auto operation = access_.read();
        return metric_to_string(metric_type_);
    }
    [[nodiscard]] size_t nbits_property() const {
        const auto operation = access_.read();
        return nbits();
    }
    [[nodiscard]] size_t quantization_bits() const {
        const auto operation = access_.read();
        return quantization_bits_;
    }
    [[nodiscard]] std::optional<uint32_t> seed() const {
        const auto operation = access_.read();
        return seed_;
    }

   private:
    static constexpr size_t kDefaultEfConstruction = 200;

    SymqgIndex() = default;

    static uint32_t random_seed() { return std::random_device{}(); }

    static void check_max_degree(size_t max_degree) {
        if (max_degree == 0 || max_degree % fastscan::kBatchSize != 0) {
            throw std::invalid_argument("max_degree must be a positive multiple of 32");
        }
    }

    static void check_construction(size_t ef_construction, const std::string& init) {
        if (init != "random" && init != "pipnn") {
            throw std::invalid_argument("init must be 'random' or 'pipnn'");
        }
        if (ef_construction == 0 ||
            ef_construction > std::numeric_limits<uint32_t>::max()) {
            throw std::invalid_argument("ef_construction must be positive and fit uint32");
        }
    }

    void check_built(const char* operation) const {
        if (!built()) {
            throw std::runtime_error(
                std::string(
                    "SymqgIndex has no vectors; add them or load an index before "
                ) +
                operation
            );
        }
    }

    // Build a new graph from data; a failure keeps the previous one. Requires the
    // write guard.
    void build_graph(
        py::handle data,
        size_t ef_construction,
        size_t num_threads,
        const std::string& init,
        uint32_t seed
    ) {
        auto data_array = ensure_2d_array<float>(data, "data");
        ensure_dimension(data_array, dim_, "data dimension does not match index dim");

        const size_t num_points = static_cast<size_t>(data_array.shape(0));
        auto index = std::make_unique<symqg::QuantizedGraph<float>>(
            num_points,
            dim_,
            max_degree_,
            metric_type_,
            RotatorType::FhtKacRotator,
            quantization_bits_,
            seed
        );

        const auto initialization = init == "pipnn" ? symqg::QGInitialization::PiPNN
                                                    : symqg::QGInitialization::Random;
        const auto* data_ptr = data_array.data();
        {
            py::gil_scoped_release release;
            symqg::QGBuilder builder(
                *index,
                static_cast<uint32_t>(ef_construction),
                data_ptr,
                num_threads,
                initialization,
                seed
            );
            builder.build();
        }
        index_ = std::move(index);
    }

    [[nodiscard]] size_t nbits() const {
        return quantization_bits_ == 0 ? 32 : quantization_bits_;
    }

    [[nodiscard]] bool built() const { return index_ != nullptr; }

    size_t dim_ = 0;
    size_t max_degree_ = 0;
    MetricType metric_type_ = METRIC_L2;
    size_t quantization_bits_ = 0;  // 0 stores raw vectors.
    // Settings of the current graph, or of the next add; unknown after load.
    std::optional<size_t> ef_construction_ = kDefaultEfConstruction;
    std::optional<std::string> init_ = "pipnn";
    std::optional<uint32_t> seed_;  // Unknown for loaded indexes.
    bool seed_given_ = false;       // The user chose the seed.
    mutable IndexAccess access_;
    std::unique_ptr<symqg::QuantizedGraph<float>> index_;
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
metric : {'l2', 'ip', 'innerproduct'}, optional
    Keyword-only. Squared L2 or inner-product distance. Default is 'l2'.
    Vectors are not automatically normalized for cosine similarity.
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
The 0.7 forms SymqgIndex(dim, max_degree, metric, quantization_bits),
quantization_bits=, build() and search_batch() still work in 0.8, raise
DeprecationWarning, and will be removed in 0.9.

Examples
--------
>>> import numpy as np
>>> from rabitqlib import SymqgIndex
>>> x = np.random.default_rng(42).normal(size=(80, 64)).astype(np.float32)
>>> index = SymqgIndex(64, ef_construction=40, seed=7)
>>> index.add(x, num_threads=1)
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
    Number of neighbors requested, between 1 and num_points.
ef : int, optional
    Search breadth. None or 0 (default) selects max(k, 10); smaller values
    are raised to k.
num_threads : int, optional
    Worker count. Default is 1; 0 selects the available CPU count,
    capped by OMP_NUM_THREADS.

Returns
-------
ids : ndarray of uint32, shape (n_queries, k)
    Point IDs of the selected neighbors.
distances : ndarray of float32, shape (n_queries, k)
    Lower values are better; squared L2 estimates for l2 and inner-product
    distance estimates for ip. These are distances, not similarity scores.

Notes
-----
Returns (ids, distances), in that order. Array shapes and value ranges are
checked at runtime, not by static type checkers.)doc";

    py::class_<SymqgIndex>(m, "SymqgIndex", kIndexDoc)
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
            py::arg("num_threads") = 0,
            R"doc(Build the graph from all vectors.

Parameters
----------
data : array_like of float, shape (n, dim)
    Vectors in the original, unpadded coordinates, with n > max_degree.
    Converted to contiguous float32; a copy may be made.
num_threads : int, optional
    Keyword-only. Worker count. Default is 0, which selects the available
    CPU count.

Notes
-----
Uses the constructor's ef_construction, init and seed. The graph cannot be
extended: a second call raises RuntimeError, so add every vector in one
call, or create a new index to rebuild. Point IDs are the row numbers of
data. Input arrays need not be retained after this synchronous call.)doc"
        )
        .def(
            "_build",
            &SymqgIndex::build,
            py::arg("data"),
            py::arg("ef_construction"),
            py::arg("num_threads"),
            py::arg("init")
        )
        .def(
            "search",
            &SymqgIndex::search,
            py::arg("queries"),
            py::arg("k"),
            py::arg("ef") = py::none(),
            py::arg("num_threads") = 1,
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
The saved state includes the rotation used to build the index.)doc"
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
            R"doc(Construction search breadth of the current graph or the next add, or None for an index loaded from disk.)doc"
        )
        .def_property_readonly(
            "init",
            &SymqgIndex::init,
            R"doc(Graph initialization method of the current graph or the next add, or None for an index loaded from disk.)doc"
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
            "max_elements", &SymqgIndex::max_elements, R"doc(Current storage capacity.)doc"
        )
        .def_property_readonly(
            "is_trained",
            [](const SymqgIndex&) { return true; },
            R"doc(Always True: SymphonyQG needs no training.)doc"
        )
        .def_property_readonly(
            "metric", &SymqgIndex::metric, R"doc(Canonical metric name: 'l2' or 'ip'.)doc"
        )
        .def_property_readonly(
            "nbits",
            &SymqgIndex::nbits_property,
            R"doc(Storage bits: 32 for raw float32, or 4 or 8 for RaBitQ.)doc"
        )
        .def_property_readonly("_quantization_bits", &SymqgIndex::quantization_bits)
        .def_property_readonly(
            "seed",
            &SymqgIndex::seed,
            R"doc(Rotation and graph-construction seed, or None for an index loaded from disk.)doc"
        );
}
