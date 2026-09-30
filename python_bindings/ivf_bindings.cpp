#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
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
#include "rabitqlib/defines.hpp"
#include "rabitqlib/index/ivf/initializer.hpp"
#include "rabitqlib/index/ivf/ivf.hpp"
#include "rabitqlib/utils/rotator.hpp"

namespace py = pybind11;

namespace rabitqlib::python_bindings {

class IvfIndex {
   public:
    IvfIndex(
        size_t dim,
        size_t max_elements,
        size_t num_clusters,
        size_t nbits,
        const std::string& metric = "l2"
    )
        : dim_(dim)
        , max_elements_(max_elements)
        , num_clusters_(num_clusters)
        , nbits_(nbits)
        , metric_(metric_from_string(metric))
        , index_(std::make_unique<rabitqlib::ivf::IVF>(
              max_elements,
              dim,
              num_clusters,
              nbits,
              metric_,
              rabitqlib::RotatorType::FhtKacRotator
          )) {}

    void build(
        py::handle data,
        py::handle centroids,
        py::handle cluster_ids,
        size_t num_threads = 1,
        bool fast_quantization = false
    ) {
        auto data_array = ensure_2d_array<float>(data, "data");
        auto centroids_array = ensure_2d_array<float>(centroids, "centroids");
        auto cluster_ids_array =
            ensure_1d_array<rabitqlib::PID>(cluster_ids, "cluster_ids");

        if (static_cast<size_t>(data_array.shape(1)) != dim_) {
            throw std::invalid_argument("data dimension does not match index dim");
        }
        if (static_cast<size_t>(data_array.shape(0)) != max_elements_) {
            throw std::invalid_argument("number of data rows must match index max_elements"
            );
        }
        if (static_cast<size_t>(centroids_array.shape(1)) != dim_) {
            throw std::invalid_argument("centroid dimension does not match index dim");
        }
        if (static_cast<size_t>(centroids_array.shape(0)) != num_clusters_) {
            throw std::invalid_argument(
                "number of centroid rows must match index num_clusters"
            );
        }
        if (static_cast<size_t>(cluster_ids_array.shape(0)) !=
            static_cast<size_t>(data_array.shape(0))) {
            throw std::invalid_argument(
                "cluster_ids length must match number of rows in data"
            );
        }

        index_->construct(
            data_array.data(),
            centroids_array.data(),
            cluster_ids_array.data(),
            fast_quantization,
            num_threads
        );
        built_ = true;
    }

    py::array_t<rabitqlib::PID> add(
        py::handle data,
        const py::object& cluster_ids,
        size_t num_threads = 1,
        bool fast_quantization = false
    ) {
        auto data_array = ensure_2d_array<float>(data, "data");
        if (!built_) {
            throw std::runtime_error("IvfIndex must be built or loaded before add");
        }
        if (static_cast<size_t>(data_array.shape(1)) != dim_) {
            throw std::invalid_argument("data dimension does not match index dim");
        }
        const auto rows = static_cast<size_t>(data_array.shape(0));

        // Validate cluster IDs as int64 first: casting straight to uint32 would wrap
        // negative or oversized values onto valid clusters.
        std::vector<rabitqlib::PID> assigned;
        if (!cluster_ids.is_none()) {
            auto cluster_ids_array = ensure_1d_integer_array(cluster_ids, "cluster_ids");
            if (static_cast<size_t>(cluster_ids_array.shape(0)) != rows) {
                throw std::invalid_argument(
                    "cluster_ids length must match number of rows in data"
                );
            }
            assigned.reserve(rows);
            for (py::ssize_t i = 0; i < cluster_ids_array.shape(0); ++i) {
                const int64_t id = cluster_ids_array.data()[i];
                if (id < 0 || static_cast<uint64_t>(id) >= num_clusters_) {
                    throw std::invalid_argument("cluster_ids must be in [0, num_clusters)");
                }
                assigned.push_back(static_cast<rabitqlib::PID>(id));
            }
        }

        const size_t first = index_->max_elements();
        // Allocate the returned IDs before committing any changes to the index.
        auto ids = py::array_t<rabitqlib::PID>(static_cast<py::ssize_t>(rows));
        auto* ids_data = ids.mutable_data();
        for (size_t i = 0; i < rows; ++i) {
            ids_data[i] = static_cast<rabitqlib::PID>(first + i);
        }

        index_->add(
            data_array.data(),
            rows,
            assigned.empty() ? nullptr : assigned.data(),
            fast_quantization,
            num_threads
        );
        max_elements_ = index_->max_elements();
        return ids;
    }

    size_t remove(py::handle ids) {
        auto ids_array = ensure_1d_integer_array(ids, "ids");
        if (!built_) {
            throw std::runtime_error("IvfIndex must be built or loaded before remove");
        }
        std::vector<rabitqlib::PID> to_remove;
        to_remove.reserve(static_cast<size_t>(ids_array.shape(0)));
        for (py::ssize_t i = 0; i < ids_array.shape(0); ++i) {
            const int64_t id = ids_array.data()[i];
            if (id < 0 || static_cast<uint64_t>(id) >= max_elements_) {
                throw std::invalid_argument("ids must be in [0, max_elements)");
            }
            to_remove.push_back(static_cast<rabitqlib::PID>(id));
        }
        return index_->remove(to_remove.data(), to_remove.size());
    }

    py::tuple search(
        py::handle queries,
        size_t k,
        size_t nprobe,
        std::optional<bool> high_accuracy = std::nullopt,
        size_t num_threads = 1
    ) {
        auto query_array = ensure_2d_array<float>(queries, "queries");
        if (!built_) {
            throw std::runtime_error("IvfIndex must be built or loaded before search");
        }
        if (static_cast<size_t>(query_array.shape(1)) != dim_) {
            throw std::invalid_argument("query dimension does not match index dim");
        }
        if (k == 0 || k > max_elements_) {
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

        rabitqlib::ivf::parallel_for(
            0,
            nq,
            num_threads,
            [&](size_t idx, size_t /*threadId*/) {
                const float* query = query_array.data() + (idx * dim_);
                auto* result_ids = ids_data + (idx * k);
                auto* result_dists = dists_data + (idx * k);
                if (high_accuracy.has_value()) {
                    index_->search(
                        query, k, nprobe, result_ids, result_dists, *high_accuracy
                    );
                } else {
                    index_->search(query, k, nprobe, result_ids, result_dists);
                }
            }
        );

        return py::make_tuple(ids, dists);
    }

    void save(const std::string& path) const {
        if (!built_) {
            throw std::runtime_error("IvfIndex must be built or loaded before save");
        }
        index_->save(path.c_str());
    }

    static IvfIndex load(const std::string& path) {
        IvfIndex wrapper;
        wrapper.index_ = std::make_unique<rabitqlib::ivf::IVF>();
        wrapper.index_->load(path.c_str());
        wrapper.dim_ = wrapper.index_->dimension();
        wrapper.max_elements_ = wrapper.index_->max_elements();
        wrapper.num_clusters_ = wrapper.index_->num_clusters();
        wrapper.nbits_ = wrapper.index_->nbits();
        wrapper.metric_ = wrapper.index_->metric_type();
        wrapper.built_ = true;
        return wrapper;
    }

    [[nodiscard]] size_t dim() const { return dim_; }
    [[nodiscard]] size_t max_elements() const { return max_elements_; }
    [[nodiscard]] size_t num_clusters() const { return num_clusters_; }
    [[nodiscard]] size_t nbits() const { return nbits_; }
    [[nodiscard]] bool is_built() const { return built_; }
    [[nodiscard]] std::string metric() const { return metric_to_string(metric_); }

   private:
    IvfIndex() = default;

    size_t dim_ = 0;
    size_t max_elements_ = 0;
    size_t num_clusters_ = 0;
    size_t nbits_ = 0;
    rabitqlib::MetricType metric_ = rabitqlib::METRIC_L2;
    bool built_ = false;
    std::unique_ptr<rabitqlib::ivf::IVF> index_;
};

}  // namespace rabitqlib::python_bindings

// Register IVF bindings into combined module
void register_ivf(py::module_& m) {
    using namespace rabitqlib::python_bindings;

    py::class_<IvfIndex>(m, "IvfIndex")
        .def(
            py::init<size_t, size_t, size_t, size_t, const std::string&>(),
            py::arg("dim"),
            py::arg("max_elements"),
            py::arg("num_clusters"),
            py::arg("nbits"),
            py::arg("metric") = "l2",
            "Create IVF with 1-9 quantization bits, or nbits=32 for owned raw-vector "
            "reranking."
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
        .def_property_readonly("num_clusters", &IvfIndex::num_clusters)
        .def_property_readonly("nbits", &IvfIndex::nbits)
        .def_property_readonly("is_built", &IvfIndex::is_built)
        .def_property_readonly("metric", &IvfIndex::metric);
}
