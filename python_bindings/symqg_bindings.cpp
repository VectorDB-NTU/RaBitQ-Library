#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include <pybind11/stl.h>

#include "bindings_common.hpp"
#include "rabitqlib/index/ivf/initializer.hpp"
#include "rabitqlib/index/symqg/qg.hpp"
#include "rabitqlib/index/symqg/qg_builder.hpp"

namespace py = pybind11;

namespace rabitqlib::python_bindings {

class SymqgIndex {
   public:
    SymqgIndex(size_t dim, size_t max_degree, const std::string& metric = "l2")
        : dim_(dim)
        , max_degree_(max_degree)
        , metric_(metric_from_string(metric)) {}

    void build(py::handle data, size_t ef_construction, size_t num_threads = 1) {
        auto data_array = ensure_2d_array<float>(data, "data");
        if (static_cast<size_t>(data_array.shape(1)) != dim_) {
            throw std::invalid_argument("data dimension does not match index dim");
        }

        num_points_ = static_cast<size_t>(data_array.shape(0));
        index_ = std::make_unique<rabitqlib::symqg::QuantizedGraph<float>>(
            num_points_, dim_, max_degree_, metric_, rabitqlib::RotatorType::FhtKacRotator
        );

        py::gil_scoped_release release;
        rabitqlib::symqg::QGBuilder builder(*index_, ef_construction, data_array.data(), num_threads);
        builder.build();
        built_ = true;
    }

    py::tuple search(py::handle queries, size_t k, size_t ef, size_t num_threads = 1) {
        auto query_array = ensure_2d_array<float>(queries, "queries");
        if (!built_) {
            throw std::runtime_error("SymqgIndex must be built or loaded before search");
        }
        if (static_cast<size_t>(query_array.shape(1)) != dim_) {
            throw std::invalid_argument("query dimension does not match index dim");
        }

        const size_t nq = static_cast<size_t>(query_array.shape(0));
        const auto shape = std::vector<ssize_t>{static_cast<ssize_t>(nq), static_cast<ssize_t>(k)};
        auto ids = py::array_t<rabitqlib::PID>(shape);
        auto dists = py::array_t<float>(shape);
        index_->set_ef(ef);

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
                    index_->search(query_array.data() + (idx * dim_), static_cast<uint32_t>(k), row_ids.data(), row_dists.data());
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
            throw std::runtime_error("SymqgIndex must be built before save");
        }
        py::gil_scoped_release release;
        index_->save(path.c_str());
    }

    static SymqgIndex load(const std::string& path) {
        SymqgIndex wrapper;
        wrapper.index_ = std::make_unique<rabitqlib::symqg::QuantizedGraph<float>>();
        py::gil_scoped_release release;
        wrapper.index_->load(path.c_str());
        wrapper.num_points_ = wrapper.index_->num_vertices();
        wrapper.dim_ = wrapper.index_->dimension();
        wrapper.max_degree_ = wrapper.index_->degree_bound();
        wrapper.metric_ = wrapper.index_->metric_type();
        wrapper.built_ = true;
        return wrapper;
    }

    [[nodiscard]] size_t dim() const { return dim_; }
    [[nodiscard]] size_t max_degree() const { return max_degree_; }
    [[nodiscard]] size_t num_points() const { return num_points_; }
    [[nodiscard]] bool is_built() const { return built_; }
    [[nodiscard]] std::string metric() const { return metric_to_string(metric_); }

   private:
    SymqgIndex() = default;

    size_t dim_ = 0;
    size_t max_degree_ = 0;
    size_t num_points_ = 0;
    rabitqlib::MetricType metric_ = rabitqlib::METRIC_L2;
    bool built_ = false;
    std::unique_ptr<rabitqlib::symqg::QuantizedGraph<float>> index_;
};

}  // namespace rabitqlib::python_bindings

// Register Symqg bindings into combined module
void register_symqg(py::module_ &m) {
    using namespace rabitqlib::python_bindings;

    py::class_<SymqgIndex>(m, "SymqgIndex")
       .def(py::init<size_t, size_t, const std::string&>(),
           py::arg("dim"),
           py::arg("max_degree"),
           py::arg("metric") = "l2")
       .def("build", &SymqgIndex::build,
           py::arg("data"),
           py::arg("ef_construction"),
           py::arg("num_threads") = 1)
       .def("search", &SymqgIndex::search,
           py::arg("queries"),
           py::arg("k"),
           py::arg("ef"),
           py::arg("num_threads") = 1)
       .def("save", &SymqgIndex::save, py::arg("path"))
       .def_static("load", &SymqgIndex::load, py::arg("path"))
       .def_property_readonly("dim", &SymqgIndex::dim)
       .def_property_readonly("max_degree", &SymqgIndex::max_degree)
       .def_property_readonly("num_points", &SymqgIndex::num_points)
       .def_property_readonly("is_built", &SymqgIndex::is_built)
       .def_property_readonly("metric", &SymqgIndex::metric);
}