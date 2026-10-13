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
            throw std::invalid_argument("number of data rows must match index max_elements"
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

    py::class_<IvfIndex>(m, "IvfIndex")
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
            "Create IVF with 1-9 quantization bits, or nbits=32 for owned raw-vector "
            "reranking. initializer selects auto, flat, flat_rabitq, or hnsw centroid "
            "routing."
        )
        .def(
            "build",
            &IvfIndex::build,
            py::arg("data"),
            py::arg("centroids"),
            py::arg("cluster_ids"),
            py::arg("num_threads") = 1,
            py::arg("fast_quantization") = false
        )
        .def(
            "add",
            &IvfIndex::add,
            py::arg("data"),
            py::arg("cluster_ids") = py::none(),
            py::arg("num_threads") = 1,
            py::arg("fast_quantization") = false,
            "Append vectors without the original data and return their ids. Vectors "
            "are quantized against the existing centroids, which do not move. Without "
            "cluster_ids each vector goes to its nearest centroid. Each call copies "
            "the whole index, so add many vectors per call rather than one at a time."
        )
        .def(
            "remove",
            &IvfIndex::remove,
            py::arg("ids"),
            "Exclude ids from later search results and return how many were newly "
            "removed. Removed vectors keep their storage and cannot be restored."
        )
        .def(
            "search",
            &IvfIndex::search,
            py::arg("queries"),
            py::arg("k"),
            py::arg("nprobe"),
            py::arg("high_accuracy") = py::none(),
            py::arg("num_threads") = 1
        )
        .def("save", &IvfIndex::save, py::arg("path"))
        .def_static("load", &IvfIndex::load, py::arg("path"))
        .def_property_readonly("dim", &IvfIndex::dim)
        .def_property_readonly("max_elements", &IvfIndex::max_elements)
        .def_property_readonly("num_points", &IvfIndex::num_points)
        .def_property_readonly("num_stored_points", &IvfIndex::num_stored_points)
        .def_property_readonly("num_clusters", &IvfIndex::num_clusters)
        .def_property_readonly("nbits", &IvfIndex::nbits)
        .def_property_readonly("is_built", &IvfIndex::is_built)
        .def_property_readonly("metric", &IvfIndex::metric)
        .def_property_readonly("initializer", &IvfIndex::initializer);
}
