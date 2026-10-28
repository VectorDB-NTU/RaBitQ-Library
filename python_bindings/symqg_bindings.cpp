#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/gil.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
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

namespace py = pybind11;

namespace rabitqlib::python_bindings {

class SymqgIndex {
   public:
    SymqgIndex(
        size_t dim,
        size_t max_degree,
        const std::string& metric = "l2",
        size_t quantization_bits = 0
    )
        : dim_(dim)
        , max_degree_(max_degree)
        , metric_type_(metric_from_string(metric))
        , quantization_bits_(quantization_bits) {
        if (max_degree == 0 || max_degree % rabitqlib::fastscan::kBatchSize != 0) {
            throw std::invalid_argument("max_degree must be a positive multiple of 32");
        }
        if (quantization_bits != 0 && quantization_bits != 4 && quantization_bits != 8) {
            throw std::invalid_argument("quantization_bits must be 0, 4, or 8");
        }
    }

    void build(
        py::handle data,
        size_t ef_construction,
        size_t num_threads = 1,
        const std::string& init = "pipnn"
    ) {
        const auto operation = access_.write();
        if (init != "random" && init != "pipnn") {
            throw std::invalid_argument("init must be 'random' or 'pipnn'");
        }
        if (ef_construction == 0 ||
            ef_construction > std::numeric_limits<uint32_t>::max()) {
            throw std::invalid_argument("ef_construction must be positive and fit uint32");
        }
        auto data_array = ensure_2d_array<float>(data, "data");
        ensure_dimension(data_array, dim_, "data dimension does not match index dim");

        const size_t num_points = static_cast<size_t>(data_array.shape(0));
        auto index = std::make_unique<rabitqlib::symqg::QuantizedGraph<float>>(
            num_points,
            dim_,
            max_degree_,
            metric_type_,
            rabitqlib::RotatorType::FhtKacRotator,
            quantization_bits_
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
                initialization
            );
            builder.build();
        }
        index_ = std::move(index);
        built_ = true;
    }

    py::tuple search(py::handle queries, size_t k, size_t ef, size_t num_threads = 1) {
        const auto operation = access_.read();
        auto query_array = ensure_2d_array<float>(queries, "queries");
        if (!built_) {
            throw std::runtime_error("SymqgIndex must be built or loaded before search");
        }
        if (k == 0 || k > index_->num_points()) {
            throw std::invalid_argument("k must be between 1 and num_points");
        }
        if (ef == 0) {
            throw std::invalid_argument("ef must be positive");
        }
        if (ef < k) {
            throw std::invalid_argument("ef must be at least k");
        }
        ensure_dimension(query_array, dim_, "query dimension does not match index dim");

        const size_t num_queries = static_cast<size_t>(query_array.shape(0));
        const auto shape = std::vector<py::ssize_t>{
            static_cast<py::ssize_t>(num_queries), static_cast<py::ssize_t>(k)};
        auto result_ids = py::array_t<rabitqlib::PID>(shape);
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
                ef,
                num_threads
            );
        }

        return py::make_tuple(result_ids, distances);
    }

    void save(py::handle path) const {
        const auto filename = path_string(path);
        const auto operation = access_.read();
        if (!built_) {
            throw std::runtime_error("SymqgIndex must be built before save");
        }
        py::gil_scoped_release release;
        index_->save(filename.c_str());
    }

    static SymqgIndex load(py::handle path) {
        const auto filename = path_string(path);
        SymqgIndex wrapper;
        wrapper.index_ = std::make_unique<rabitqlib::symqg::QuantizedGraph<float>>();
        {
            py::gil_scoped_release release;
            wrapper.index_->load(filename.c_str());
        }
        wrapper.dim_ = wrapper.index_->dimension();
        wrapper.max_degree_ = wrapper.index_->degree_bound();
        wrapper.metric_type_ = wrapper.index_->metric_type();
        wrapper.quantization_bits_ = wrapper.index_->quantization_bits();
        wrapper.built_ = true;
        return wrapper;
    }

    [[nodiscard]] std::string repr() const {
        const auto operation = access_.read();
        return "SymqgIndex(dim=" + std::to_string(dim_) + ", metric='" +
               metric_to_string(metric_type_) +
               "', quantization_bits=" + std::to_string(quantization_bits_) +
               ", num_points=" + std::to_string(index_ ? index_->num_points() : 0) +
               ", max_degree=" + std::to_string(max_degree_) +
               ", is_built=" + (built_ ? "True" : "False") + ")";
    }

    [[nodiscard]] size_t dim() const {
        const auto operation = access_.read();
        return dim_;
    }
    [[nodiscard]] size_t max_degree() const {
        const auto operation = access_.read();
        return max_degree_;
    }
    [[nodiscard]] size_t num_points() const {
        const auto operation = access_.read();
        return index_ ? index_->num_points() : 0;
    }
    [[nodiscard]] size_t num_stored_points() const {
        const auto operation = access_.read();
        return index_ ? index_->num_stored_points() : 0;
    }
    [[nodiscard]] size_t max_elements() const {
        const auto operation = access_.read();
        return index_ ? index_->max_elements() : 0;
    }
    [[nodiscard]] bool is_built() const {
        const auto operation = access_.read();
        return built_;
    }
    [[nodiscard]] std::string metric() const {
        const auto operation = access_.read();
        return metric_to_string(metric_type_);
    }
    [[nodiscard]] size_t quantization_bits() const {
        const auto operation = access_.read();
        return quantization_bits_;
    }

   private:
    SymqgIndex() = default;

    size_t dim_ = 0;
    size_t max_degree_ = 0;
    rabitqlib::MetricType metric_type_ = rabitqlib::METRIC_L2;
    size_t quantization_bits_ = 0;
    bool built_ = false;
    mutable IndexAccess access_;
    std::unique_ptr<rabitqlib::symqg::QuantizedGraph<float>> index_;
};

}  // namespace rabitqlib::python_bindings

// Register Symqg bindings into combined module
void register_symqg(py::module_& m) {
    using namespace rabitqlib::python_bindings;

    py::class_<SymqgIndex>(
        m, "SymqgIndex", R"doc(An approximate nearest-neighbor index using SymphonyQG.

Parameters
----------
dim : int
    Original vector dimension, from 64 through 65536.
max_degree : int
    Graph degree bound; a positive multiple of 32.
metric : {'l2', 'ip', 'innerproduct'}, optional
    Squared L2 or inner-product distance. Default is 'l2'. Vectors are not
    automatically normalized for cosine similarity.
quantization_bits : {0, 4, 8}, optional
    Centroid-relative RaBitQ storage at 4 or 8 bits, or raw float32 at 0
    (default).

Notes
-----
Build or load the index before searching or saving. Concurrent reads are
supported; an overlapping mutation raises RuntimeError.)doc"
    )
        .def(
            py::init<size_t, size_t, const std::string&, size_t>(),
            py::arg("dim"),
            py::arg("max_degree"),
            py::arg("metric") = "l2",
            py::arg("quantization_bits") = 0,
            R"doc(An approximate nearest-neighbor index using SymphonyQG.

Parameters
----------
dim : int
    Original vector dimension, from 64 through 65536.
max_degree : int
    Graph degree bound; a positive multiple of 32.
metric : {'l2', 'ip', 'innerproduct'}, optional
    Squared L2 or inner-product distance. Default is 'l2'. Vectors are not
    automatically normalized for cosine similarity.
quantization_bits : {0, 4, 8}, optional
    Centroid-relative RaBitQ storage at 4 or 8 bits, or raw float32 at 0
    (default).

Notes
-----
Build or load the index before searching or saving. Concurrent reads are
supported; an overlapping mutation raises RuntimeError.)doc"
        )
        .def(
            "build",
            &SymqgIndex::build,
            py::arg("data"),
            py::arg("ef_construction"),
            py::arg("num_threads") = 1,
            py::arg("init") = "pipnn",
            R"doc(Build the graph from input vectors.

Parameters
----------
data : array_like of float, shape (n, dim)
    Vectors in the original, unpadded coordinates. Converted to contiguous
    float32; a copy may be made.
ef_construction : int
    Positive construction search breadth fitting uint32.
num_threads : int, optional
    Worker count. Default is 1; 0 selects the available CPU count.
init : {'pipnn', 'random'}, optional
    Graph initialization method. Default is 'pipnn'.

Notes
-----
Requires n > max_degree. Input arrays need not be retained after this
synchronous call.)doc"
        )
        .def(
            "search",
            &SymqgIndex::search,
            py::arg("queries"),
            py::arg("k"),
            py::arg("ef"),
            py::arg("num_threads") = 1,
            R"doc(Search a batch of query vectors.

Parameters
----------
queries : array_like of float, shape (n_queries, dim)
    Vectors in the original, unpadded coordinates. Converted to contiguous
    float32; a copy may be made.
k : int
    Number of neighbors requested, between 1 and num_points.
ef : int
    Search breadth. Must be positive and at least k.
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
Returns (ids, distances), in that order. Array shapes and value ranges are
checked at runtime, not by static type checkers.)doc"
        )
        .def(
            "search_batch",
            &SymqgIndex::search,
            py::arg("queries"),
            py::arg("k"),
            py::arg("ef"),
            py::arg("num_threads") = 1,
            R"doc(Search a batch of query vectors.

Parameters
----------
queries : array_like of float, shape (n_queries, dim)
    Vectors in the original, unpadded coordinates. Converted to contiguous
    float32; a copy may be made.
k : int
    Number of neighbors requested, between 1 and num_points.
ef : int
    Search breadth. Must be positive and at least k.
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
Returns (ids, distances), in that order. Array shapes and value ranges are
checked at runtime, not by static type checkers.)doc"
        )
        .def(
            "save",
            &SymqgIndex::save,
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
    A built index ready for search.)doc"
        )
        .def(
            "__repr__",
            &SymqgIndex::repr,
            "Return a one-line summary of configuration, counts and build state."
        )
        .def_property_readonly(
            "dim", &SymqgIndex::dim, R"doc(Original vector dimension.)doc"
        )
        .def_property_readonly(
            "max_degree", &SymqgIndex::max_degree, R"doc(Graph degree bound.)doc"
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
            "is_built",
            &SymqgIndex::is_built,
            R"doc(Whether the index has been built or loaded.)doc"
        )
        .def_property_readonly(
            "metric", &SymqgIndex::metric, R"doc(Canonical metric name: 'l2' or 'ip'.)doc"
        )
        .def_property_readonly(
            "quantization_bits",
            &SymqgIndex::quantization_bits,
            R"doc(Storage bits: 0 for raw float32, or 4 or 8 for RaBitQ.)doc"
        );
}
