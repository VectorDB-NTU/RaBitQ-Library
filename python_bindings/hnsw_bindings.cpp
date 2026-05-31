#include <algorithm>
#include <memory>
#include <string>
#include <cstring>
#include <vector>

#include <pybind11/stl.h>

#include "bindings_common.hpp"
#include "rabitqlib/index/hnsw/hnsw.hpp"

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
        : dim_(dim)
        , max_elements_(max_elements)
        , M_(M)
        , ef_construction_(ef_construction)
        , nbits_(nbits)
        , metric_(metric_from_string(metric))
        , random_seed_(random_seed)
        , index_(std::make_unique<rabitqlib::hnsw::HierarchicalNSW>(
              max_elements,
              dim,
              nbits,
              M,
              ef_construction,
              random_seed,
              metric_
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

        const size_t num_clusters = static_cast<size_t>(centroids_array.shape(0));
        num_clusters_ = num_clusters;

        // Ensure cluster_ids are writable for the C++ API by making a copy
        std::vector<rabitqlib::PID> cluster_ids_vec(static_cast<size_t>(cluster_ids_array.shape(0)));
        std::memcpy(cluster_ids_vec.data(), cluster_ids_array.data(), cluster_ids_vec.size() * sizeof(rabitqlib::PID));

        py::gil_scoped_release release;
        index_->construct(
            num_clusters,
            centroids_array.data(),
            static_cast<size_t>(data_array.shape(0)),
            data_array.data(),
            cluster_ids_vec.data(),
            num_threads,
            fast_quantization
        );
        built_ = true;
    }

    py::tuple search(py::handle queries, size_t k, size_t ef = 0, size_t num_threads = 1) {
        auto query_array = ensure_2d_array<float>(queries, "queries");
        if (dim_ != 0 && static_cast<size_t>(query_array.shape(1)) != dim_) {
            throw std::invalid_argument("query dimension does not match index dim");
        }
        if (ef == 0) {
            ef = std::max<size_t>(k, 10);
        }

        const auto shape = std::vector<ssize_t>{
            static_cast<ssize_t>(query_array.shape(0)), static_cast<ssize_t>(k)};
        auto ids = py::array_t<rabitqlib::PID>(shape);
        auto dists = py::array_t<float>(shape);
        auto ids_buf = ids.mutable_unchecked<2>();
        auto dists_buf = dists.mutable_unchecked<2>();
        std::vector<std::vector<std::pair<float, rabitqlib::PID>>> results;
        {
            py::gil_scoped_release release;
            results = index_->search(
                query_array.data(),
                static_cast<size_t>(query_array.shape(0)),
                k,
                ef,
                num_threads
            );
        }


        for (ssize_t i = 0; i < static_cast<ssize_t>(results.size()); ++i) {
            for (
                ssize_t j = 0;
                j < static_cast<ssize_t>(std::min<size_t>(k, results[static_cast<size_t>(i)].size()));
                ++j
            ) {
                ids_buf(i, j) = results[static_cast<size_t>(i)][static_cast<size_t>(j)].second;
                dists_buf(i, j) = results[static_cast<size_t>(i)][static_cast<size_t>(j)].first;
            }
        }
        return py::make_tuple(ids, dists);
    }

    void save(const std::string& path) const {
        py::gil_scoped_release release;
        index_->save(path.c_str());
    }

    static HnswIndex load(const std::string& path) {
        HnswIndex wrapper;
        wrapper.index_ = std::make_unique<rabitqlib::hnsw::HierarchicalNSW>();
        py::gil_scoped_release release;
        wrapper.index_->load(path.c_str());
        wrapper.dim_ = wrapper.index_->dimension();
        wrapper.max_elements_ = wrapper.index_->max_elements();
        wrapper.M_ = wrapper.index_->M();
        wrapper.ef_construction_ = wrapper.index_->ef_construction();
        wrapper.nbits_ = wrapper.index_->nbits();
        wrapper.num_clusters_ = wrapper.index_->num_clusters();
        wrapper.metric_ = wrapper.index_->metric_type();
        wrapper.built_ = true;
        return wrapper;
    }

    [[nodiscard]] size_t dim() const { return dim_; }
    [[nodiscard]] size_t max_elements() const { return max_elements_; }
    [[nodiscard]] size_t nbits() const { return nbits_; }
    [[nodiscard]] bool is_built() const { return built_; }
    [[nodiscard]] size_t num_clusters() const { return num_clusters_; }

   private:
    HnswIndex() = default;

    size_t dim_ = 0;
    size_t max_elements_ = 0;
    size_t M_ = 0;
    size_t ef_construction_ = 0;
    size_t nbits_ = 0;
    rabitqlib::MetricType metric_ = rabitqlib::METRIC_L2;
    size_t random_seed_ = 100;
    size_t num_clusters_ = 0;
    bool built_ = false;
    std::unique_ptr<rabitqlib::hnsw::HierarchicalNSW> index_;
};

}  // namespace rabitqlib::python_bindings

// Register into combined module
void register_hnsw(py::module_ &m) {
    using namespace rabitqlib::python_bindings;

    py::class_<HnswIndex>(m, "HnswIndex")
        .def(py::init<size_t, size_t, size_t, size_t, size_t, const std::string&, size_t>(),
             py::arg("dim"),
             py::arg("max_elements"),
             py::arg("M") = 16,
             py::arg("ef_construction") = 200,
             py::arg("nbits") = 8,
             py::arg("metric") = "l2",
             py::arg("random_seed") = 100)
        .def("build", &HnswIndex::build,
             py::arg("data"),
             py::arg("centroids"),
             py::arg("cluster_ids"),
             py::arg("num_threads") = 1,
             py::arg("fast_quantization") = false)
        .def("search", &HnswIndex::search,
             py::arg("queries"),
             py::arg("k"),
             py::arg("ef") = 0,
             py::arg("num_threads") = 1)
        .def("save", &HnswIndex::save, py::arg("path"))
        .def_static("load", &HnswIndex::load, py::arg("path"))
        .def_property_readonly("dim", &HnswIndex::dim)
        .def_property_readonly("max_elements", &HnswIndex::max_elements)
        .def_property_readonly("nbits", &HnswIndex::nbits)
        .def_property_readonly("num_clusters", &HnswIndex::num_clusters)
        .def_property_readonly("is_built", &HnswIndex::is_built);
}
