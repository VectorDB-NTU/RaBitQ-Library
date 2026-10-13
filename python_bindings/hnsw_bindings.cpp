#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/gil.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bindings_common.hpp"
#include "rabitqlib/index/hnsw/hnsw.hpp"
#include "rabitqlib/index/types.hpp"
#include "rabitqlib/utils/packed.hpp"

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
        const std::string& metric = "l2",
        size_t random_seed = 100
    )
        : index_(std::make_unique<rabitqlib::hnsw::HierarchicalNSW>(
              max_elements,
              dim,
              nbits,
              M,
              ef_construction,
              random_seed,
              metric_from_string(metric)
          )) {}

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
        if (data_array.shape(0) == 0 ||
            static_cast<size_t>(data_array.shape(0)) > index_->max_elements()) {
            throw std::invalid_argument(
                "number of data rows must be between 1 and max_elements"
            );
        }
        if (static_cast<size_t>(centroids_array.shape(1)) != index_->dimension()) {
            throw std::invalid_argument("centroid dimension does not match index dim");
        }
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
        size_t num_threads = 1,
        bool fast_quantization = false
    ) {
        const auto operation = access_.write();
        auto data_array = ensure_2d_array<float>(data, "data");
        if (!built_) {
            throw std::runtime_error("HnswIndex must be built or loaded before add");
        }
        if (static_cast<size_t>(data_array.shape(1)) != index_->dimension()) {
            throw std::invalid_argument("data dimension does not match index dim");
        }
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
        // Removed labels remain allocated; validate against the full stored ID range.
        const size_t count = index_->num_stored_points();
        std::vector<rabitqlib::PID> to_remove;
        to_remove.reserve(static_cast<size_t>(ids_array.shape(0)));
        const rabitqlib::detail::ConstPackedArrayView<int64_t> values(
            reinterpret_cast<const char*>(ids_array.data())
        );
        for (py::ssize_t i = 0; i < ids_array.shape(0); ++i) {
            const int64_t id = values[static_cast<size_t>(i)];
            if (id < 0 || static_cast<uint64_t>(id) >= count) {
                throw std::invalid_argument("ids must be in [0, num_stored_points)");
            }
            to_remove.push_back(static_cast<rabitqlib::PID>(id));
        }
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

    py::tuple search(py::handle queries, size_t k, size_t ef = 0, size_t num_threads = 1) {
        const auto operation = access_.read();
        auto query_array = ensure_2d_array<float>(queries, "queries");
        if (!built_) {
            throw std::runtime_error("HnswIndex must be built or loaded before search");
        }
        if (index_->dimension() != 0 &&
            static_cast<size_t>(query_array.shape(1)) != index_->dimension()) {
            throw std::invalid_argument("query dimension does not match index dim");
        }
        if (ef == 0) {
            ef = std::max<size_t>(k, 10);
        }
        if (k == 0 || k > index_->max_elements()) {
            throw std::invalid_argument("k must be between 1 and max_elements");
        }

        const auto shape = std::vector<py::ssize_t>{
            static_cast<py::ssize_t>(query_array.shape(0)), static_cast<py::ssize_t>(k)};
        auto ids = py::array_t<rabitqlib::PID>(shape);
        auto dists = py::array_t<float>(shape);
        std::fill(ids.mutable_data(), ids.mutable_data() + ids.size(), rabitqlib::kPidMax);
        std::fill(
            dists.mutable_data(),
            dists.mutable_data() + dists.size(),
            std::numeric_limits<float>::infinity()
        );
        auto ids_buf = ids.mutable_unchecked<2>();
        auto dists_buf = dists.mutable_unchecked<2>();

        const auto* queries_data = query_array.data();
        const auto nq = static_cast<size_t>(query_array.shape(0));
        std::vector<std::vector<std::pair<float, rabitqlib::PID>>> results;
        {
            py::gil_scoped_release release;
            results = index_->search(queries_data, nq, k, ef, num_threads);
        }

        for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(results.size()); ++i) {
            for (py::ssize_t j = 0; j < static_cast<py::ssize_t>(std::min<size_t>(
                                            k, results[static_cast<size_t>(i)].size()
                                        ));
                 ++j) {
                ids_buf(i, j) =
                    results[static_cast<size_t>(i)][static_cast<size_t>(j)].second;
                dists_buf(i, j) =
                    results[static_cast<size_t>(i)][static_cast<size_t>(j)].first;
            }
        }
        return py::make_tuple(ids, dists);
    }

    void save(const std::string& path) const {
        const auto operation = access_.read();
        if (!built_) {
            throw std::runtime_error("HnswIndex must be built or loaded before save");
        }
        py::gil_scoped_release release;
        index_->save(path.c_str());
    }

    static HnswIndex load(const std::string& path) {
        HnswIndex wrapper;
        wrapper.index_ = std::make_unique<rabitqlib::hnsw::HierarchicalNSW>();
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

    py::class_<HnswIndex>(m, "HnswIndex")
        .def(
            py::init<size_t, size_t, size_t, size_t, size_t, const std::string&, size_t>(),
            py::arg("dim"),
            py::arg("max_elements"),
            py::arg("M") = 16,
            py::arg("ef_construction") = 200,
            py::arg("nbits") = 8,
            py::arg("metric") = "l2",
            py::arg("random_seed") = 100
        )
        .def(
            "build",
            &HnswIndex::build,
            py::arg("data"),
            py::arg("centroids"),
            py::arg("cluster_ids"),
            py::arg("num_threads") = 1,
            py::arg("fast_quantization") = false
        )
        .def(
            "search",
            &HnswIndex::search,
            py::arg("queries"),
            py::arg("k"),
            py::arg("ef") = 0,
            py::arg("num_threads") = 1
        )
        .def(
            "add",
            &HnswIndex::add,
            py::arg("data"),
            py::arg("cluster_ids") = py::none(),
            py::arg("num_threads") = 1,
            py::arg("fast_quantization") = false
        )
        .def("resize", &HnswIndex::resize, py::arg("max_elements"))
        .def(
            "remove",
            &HnswIndex::remove,
            py::arg("ids"),
            "Exclude ids from later search results and return how many were newly "
            "removed. Removed points stay in the graph and in num_stored_points, and "
            "cannot "
            "be restored."
        )
        .def("save", &HnswIndex::save, py::arg("path"))
        .def_static("load", &HnswIndex::load, py::arg("path"))
        .def_property_readonly("dim", &HnswIndex::dim)
        .def_property_readonly("max_elements", &HnswIndex::max_elements)
        .def_property_readonly("num_points", &HnswIndex::num_points)
        .def_property_readonly("num_stored_points", &HnswIndex::num_stored_points)
        .def_property_readonly("nbits", &HnswIndex::nbits)
        .def_property_readonly("num_clusters", &HnswIndex::num_clusters)
        .def_property_readonly("is_built", &HnswIndex::is_built)
        .def_property_readonly("metric", &HnswIndex::metric);
}
