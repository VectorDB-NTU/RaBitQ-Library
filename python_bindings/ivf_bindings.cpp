#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include <pybind11/stl.h>

#include "bindings_common.hpp"
#include "rabitqlib/index/ivf/ivf.hpp"

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
        bool fast_quantization = false
    ) {
        auto data_array = ensure_2d_array<float>(data, "data");
        auto centroids_array = ensure_2d_array<float>(centroids, "centroids");
        auto cluster_ids_array = ensure_1d_array<rabitqlib::PID>(cluster_ids, "cluster_ids");

        if (static_cast<size_t>(data_array.shape(1)) != dim_) {
            throw std::invalid_argument("data dimension does not match index dim");
        }
        if (static_cast<size_t>(centroids_array.shape(1)) != dim_) {
            throw std::invalid_argument("centroid dimension does not match index dim");
        }
        if (static_cast<size_t>(cluster_ids_array.shape(0)) != static_cast<size_t>(data_array.shape(0))) {
            throw std::invalid_argument("cluster_ids length must match number of rows in data");
        }

        py::gil_scoped_release release;
        index_->construct(data_array.data(), centroids_array.data(), cluster_ids_array.data(), fast_quantization);
        built_ = true;
    }

    py::tuple search(
        py::handle queries,
        size_t k,
        size_t nprobe,
        bool high_accuracy = true,
        size_t num_threads = 1
    ) {
        auto query_array = ensure_2d_array<float>(queries, "queries");
        if (static_cast<size_t>(query_array.shape(1)) != dim_) {
            throw std::invalid_argument("query dimension does not match index dim");
        }

        const size_t nq = static_cast<size_t>(query_array.shape(0));
        const auto shape = std::vector<ssize_t>{static_cast<ssize_t>(nq), static_cast<ssize_t>(k)};
        auto ids = py::array_t<rabitqlib::PID>(shape);
        auto dists = py::array_t<float>(shape);

        {
            py::gil_scoped_release release;
            auto ids_buf = ids.mutable_unchecked<2>();
            auto dists_buf = dists.mutable_unchecked<2>();

            rabitqlib::ivf::parallel_for(
                0,
                nq,
                num_threads,
                [&](size_t idx, size_t /*threadId*/) {
                    std::vector<rabitqlib::PID> row_ids(k, 0);
                    std::vector<float> row_dists(k, 0.0F);
                    
                    index_->search(
                        query_array.data() + (idx * dim_),
                        k,
                        nprobe,
                        row_ids.data(),
                        row_dists.data(),
                        high_accuracy
                    );
                    
                    for (size_t j = 0; j < k; ++j) {
                        ids_buf(static_cast<ssize_t>(idx), static_cast<ssize_t>(j)) = row_ids[j];
                        dists_buf(static_cast<ssize_t>(idx), static_cast<ssize_t>(j)) = row_dists[j];
                    }
                }
            );
        }

        return py::make_tuple(ids, dists);
    }

    void save(const std::string& path) const {
        if (!built_) {
            throw std::runtime_error("IvfIndex must be built or loaded before save");
        }
        py::gil_scoped_release release;
        index_->save(path.c_str());
    }

    static IvfIndex load(const std::string& path) {
        IvfIndex wrapper;
        wrapper.index_ = std::make_unique<rabitqlib::ivf::IVF>();
        py::gil_scoped_release release;
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
void register_ivf(py::module_ &m) {
    using namespace rabitqlib::python_bindings;

    py::class_<IvfIndex>(m, "IvfIndex")
       .def(py::init<size_t, size_t, size_t, size_t, const std::string&>(),
           py::arg("dim"),
           py::arg("max_elements"),
           py::arg("num_clusters"),
           py::arg("nbits"),
           py::arg("metric") = "l2")
       .def("build", &IvfIndex::build,
           py::arg("data"),
           py::arg("centroids"),
           py::arg("cluster_ids"),
           py::arg("fast_quantization") = false)
       .def("search", &IvfIndex::search,
           py::arg("queries"),
           py::arg("k"),
           py::arg("nprobe"),
           py::arg("high_accuracy") = true,
           py::arg("num_threads") = 1)
       .def("save", &IvfIndex::save, py::arg("path"))
       .def_static("load", &IvfIndex::load, py::arg("path"))
       .def_property_readonly("dim", &IvfIndex::dim)
       .def_property_readonly("max_elements", &IvfIndex::max_elements)
       .def_property_readonly("num_clusters", &IvfIndex::num_clusters)
       .def_property_readonly("nbits", &IvfIndex::nbits)
       .def_property_readonly("is_built", &IvfIndex::is_built)
       .def_property_readonly("metric", &IvfIndex::metric);
}