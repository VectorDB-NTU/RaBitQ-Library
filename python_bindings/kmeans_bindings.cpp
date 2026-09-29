#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/gil.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>  // IWYU pragma: keep (vector conversion for iteration_stats)

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "bindings_common.hpp"
#include "rabitqlib/clustering/qgkmeans.hpp"
#include "rabitqlib/clustering/rabitqkmeans.hpp"

namespace py = pybind11;

namespace rabitqlib::python_bindings {

template <typename T>
py::array_t<T> vector_to_array(const std::vector<T>& values) {
    py::array_t<T> result(static_cast<py::ssize_t>(values.size()));
    if (!values.empty()) {
        std::memcpy(result.mutable_data(), values.data(), values.size() * sizeof(T));
    }
    return result;
}

template <typename Clustering>
py::array_t<float> centroids_to_array(const Clustering& clustering, const char* name) {
    if (clustering.centroids.empty()) {
        throw std::logic_error(std::string(name) + " has not been trained");
    }
    const auto shape = std::vector<py::ssize_t>{
        static_cast<py::ssize_t>(clustering.k), static_cast<py::ssize_t>(clustering.d)};
    py::array_t<float> result(shape);
    std::memcpy(
        result.mutable_data(),
        clustering.centroids.data(),
        clustering.centroids.size() * sizeof(float)
    );
    return result;
}

template <typename Clustering>
double train(Clustering& clustering, py::handle x) {
    auto x_array = ensure_2d_array<float>(x, "x");
    if (static_cast<size_t>(x_array.shape(1)) != clustering.d) {
        throw std::invalid_argument("x dimension does not match d");
    }

    {
        py::gil_scoped_release release;
        clustering.train(static_cast<size_t>(x_array.shape(0)), x_array.data());
    }
    return clustering.iteration_stats.back().obj;
}

template <typename Parameters>
void bind_training_parameters(py::class_<Parameters>& binding) {
    binding.def(py::init<>())
        .def_readwrite("niter", &Parameters::niter)
        .def_readwrite("verbose", &Parameters::verbose)
        .def_readwrite("spherical", &Parameters::spherical)
        .def_readwrite("seed", &Parameters::seed)
        .def_readwrite("min_points_per_centroid", &Parameters::min_points_per_centroid)
        .def_readwrite("early_stop_threshold", &Parameters::early_stop_threshold)
        .def_readwrite("num_threads", &Parameters::num_threads)
        .def_readwrite("final_assignment", &Parameters::final_assignment);
}

template <typename Clustering, typename Parameters>
void bind_clustering(py::module_& m, const char* native_name, const char* public_name) {
    py::class_<Clustering>(m, native_name)
        .def(py::init<size_t, size_t, const Parameters&>())
        .def("train", &train<Clustering>, py::arg("x"))
        .def_readonly("d", &Clustering::d)
        .def_readonly("k", &Clustering::k)
        .def_readonly("final_obj", &Clustering::final_obj)
        .def_property_readonly(
            "centroids",
            [public_name](const Clustering& clustering) {
                return centroids_to_array(clustering, public_name);
            }
        )
        .def_property_readonly(
            "assignments",
            [](const Clustering& clustering) {
                return vector_to_array(clustering.assignments);
            }
        )
        .def_property_readonly(
            "distances",
            [](const Clustering& clustering) {
                return vector_to_array(clustering.distances);
            }
        )
        .def_property_readonly(
            "obj",
            [](const Clustering& clustering) {
                std::vector<double> values;
                values.reserve(clustering.iteration_stats.size());
                for (const auto& stats : clustering.iteration_stats) {
                    values.push_back(stats.obj);
                }
                return vector_to_array(values);
            }
        )
        .def_readonly("iteration_stats", &Clustering::iteration_stats);
}

}  // namespace rabitqlib::python_bindings

void register_kmeans(py::module_& m) {
    using namespace rabitqlib::python_bindings;
    using rabitqlib::qgkmeans::FinalAssignmentMode;
    using rabitqlib::qgkmeans::QGKMeans;
    using rabitqlib::qgkmeans::QGKMeansIterationStats;
    using rabitqlib::qgkmeans::QGKMeansParameters;
    using rabitqlib::rabitqkmeans::RaBitQKMeans;
    using rabitqlib::rabitqkmeans::RaBitQKMeansParameters;

    py::enum_<FinalAssignmentMode>(m, "FinalAssignmentMode")
        .value("Approximate", FinalAssignmentMode::Approximate)
        .value("SymphonyQG", FinalAssignmentMode::SymphonyQG)
        .value("Exact", FinalAssignmentMode::Exact);

    py::class_<QGKMeansParameters> qg_parameters(m, "QGKMeansParameters");
    bind_training_parameters(qg_parameters);
    qg_parameters.def_readwrite("graph_degree", &QGKMeansParameters::graph_degree)
        .def_readwrite("ef_build", &QGKMeansParameters::ef_build)
        .def_readwrite("ef_search", &QGKMeansParameters::ef_search)
        .def_readwrite(
            "graph_build_iterations", &QGKMeansParameters::graph_build_iterations
        )
        .def_readwrite("quantization_bits", &QGKMeansParameters::quantization_bits);

    py::class_<RaBitQKMeansParameters> flat_parameters(m, "RaBitQKMeansParameters");
    bind_training_parameters(flat_parameters);

    py::class_<QGKMeansIterationStats>(m, "QGKMeansIterationStats")
        .def_readonly("iteration", &QGKMeansIterationStats::iteration)
        .def_readonly("obj", &QGKMeansIterationStats::obj)
        .def_readonly("shift", &QGKMeansIterationStats::shift)
        .def_readonly("nsplit", &QGKMeansIterationStats::nsplit);
    m.attr("RaBitQKMeansIterationStats") = m.attr("QGKMeansIterationStats");

    bind_clustering<QGKMeans, QGKMeansParameters>(m, "_QGKMeans", "QGKMeans");
    bind_clustering<RaBitQKMeans, RaBitQKMeansParameters>(
        m, "_RaBitQKMeans", "RaBitQKMeans"
    );
}
