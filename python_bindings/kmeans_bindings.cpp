#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/gil.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>  // IWYU pragma: keep (vector conversion for iteration_stats)

#include <stdexcept>
#include <string>
#include <vector>

#include "bindings_common.hpp"
#include "rabitqlib/clustering/qgkmeans.hpp"
#include "rabitqlib/clustering/rabitqkmeans.hpp"

namespace py = pybind11;

namespace rabitqlib::python_bindings {
namespace {

// Without a base object, pybind11 copies the data into a new array.
template <typename T>
py::array_t<T> vector_to_array(const std::vector<T>& values) {
    return py::array_t<T>(static_cast<py::ssize_t>(values.size()), values.data());
}

template <typename Clustering>
py::array_t<float> centroids_to_array(const Clustering& clustering, const char* name) {
    if (clustering.centroids.empty()) {
        throw std::runtime_error(std::string(name) + " has not been trained");
    }
    const auto shape = std::vector<py::ssize_t>{
        static_cast<py::ssize_t>(clustering.k), static_cast<py::ssize_t>(clustering.d)};
    return py::array_t<float>(shape, clustering.centroids.data());
}

template <typename Clustering>
double train(Clustering& clustering, py::handle x) {
    auto x_array = ensure_2d_array<float>(x, "x");
    ensure_dimension(x_array, clustering.d, "x dimension does not match d");

    {
        py::gil_scoped_release release;
        clustering.train(static_cast<size_t>(x_array.shape(0)), x_array.data());
    }
    return clustering.iteration_stats.back().obj;
}

template <typename Parameters>
void bind_training_parameters(py::class_<Parameters>& binding) {
    binding.def(py::init<>(), "Create parameters with their documented defaults.")
        .def_readwrite(
            "niter",
            &Parameters::niter,
            R"doc(Maximum number of clustering iterations. Default is 25.)doc"
        )
        .def_readwrite(
            "verbose",
            &Parameters::verbose,
            R"doc(Print training progress when True. Default is False.)doc"
        )
        .def_readwrite(
            "spherical",
            &Parameters::spherical,
            R"doc(Normalize centroids after each update and use inner-product assignment. Normalize input vectors yourself for cosine clustering. Default is False.)doc"
        )
        .def_readwrite("seed", &Parameters::seed, R"doc(Random seed. Default is 42.)doc")
        .def_readwrite(
            "min_points_per_centroid",
            &Parameters::min_points_per_centroid,
            R"doc(Warn below this many training vectors per centroid. Default is 39.)doc"
        )
        .def_readwrite(
            "early_stop_threshold",
            &Parameters::early_stop_threshold,
            R"doc(Stop when the absolute relative objective change is at most this threshold. Default is 0.0.)doc"
        )
        .def_readwrite(
            "num_threads",
            &Parameters::num_threads,
            R"doc(Worker count; 0 selects the available CPU count. Default is 0.)doc"
        )
        .def_readwrite(
            "final_assignment",
            &Parameters::final_assignment,
            R"doc(Approximate uses the training assignment method; Exact uses exhaustive assignment against final centroids. Default is FinalAssignmentMode.Approximate.)doc"
        );
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
                py::array_t<double> values(
                    static_cast<py::ssize_t>(clustering.iteration_stats.size())
                );
                double* out = values.mutable_data();
                for (const auto& stats : clustering.iteration_stats) {
                    *out++ = stats.obj;
                }
                return values;
            }
        )
        .def_readonly("iteration_stats", &Clustering::iteration_stats);
}

}  // namespace
}  // namespace rabitqlib::python_bindings

void register_kmeans(py::module_& m) {
    using namespace rabitqlib::python_bindings;
    using rabitqlib::qgkmeans::FinalAssignmentMode;
    using rabitqlib::qgkmeans::QGKMeans;
    using rabitqlib::qgkmeans::QGKMeansIterationStats;
    using rabitqlib::qgkmeans::QGKMeansParameters;
    using rabitqlib::rabitqkmeans::RaBitQKMeans;
    using rabitqlib::rabitqkmeans::RaBitQKMeansParameters;

    py::enum_<FinalAssignmentMode>(
        m, "FinalAssignmentMode", R"doc(Final centroid assignment policy.

Approximate uses the selected clustering algorithm. SymphonyQG is a
compatibility alias for Approximate. Exact performs exhaustive assignment.)doc"
    )
        .value("Approximate", FinalAssignmentMode::Approximate)
        .value("SymphonyQG", FinalAssignmentMode::SymphonyQG)
        .value("Exact", FinalAssignmentMode::Exact);

    py::class_<QGKMeansParameters> qg_parameters(
        m, "QGKMeansParameters", R"doc(Mutable training settings for QGKMeans.

Notes
-----
Create with no arguments; set attributes directly or use keyword arguments
on QGKMeans.)doc"
    );
    bind_training_parameters(qg_parameters);
    qg_parameters
        .def_readwrite(
            "graph_degree",
            &QGKMeansParameters::graph_degree,
            R"doc(Centroid graph degree; a positive multiple of 32. Default is 32.)doc"
        )
        .def_readwrite(
            "ef_build",
            &QGKMeansParameters::ef_build,
            R"doc(Positive centroid-graph construction search breadth. Default is 240.)doc"
        )
        .def_readwrite(
            "ef_search",
            &QGKMeansParameters::ef_search,
            R"doc(Positive centroid-graph assignment search breadth. Default is 16.)doc"
        )
        .def_readwrite(
            "graph_build_iterations",
            &QGKMeansParameters::graph_build_iterations,
            R"doc(Positive number of graph-building passes after initialization. Default is 1.)doc"
        )
        .def_readwrite(
            "quantization_bits",
            &QGKMeansParameters::quantization_bits,
            R"doc(Centroid storage: 0 for raw float32, or 4 or 8 for RaBitQ. Default is 0.)doc"
        );

    py::class_<RaBitQKMeansParameters> flat_parameters(
        m, "RaBitQKMeansParameters", R"doc(Mutable training settings for RaBitQKMeans.

Notes
-----
Create with no arguments; set attributes directly or use keyword arguments
on RaBitQKMeans.)doc"
    );
    bind_training_parameters(flat_parameters);

    py::class_<QGKMeansIterationStats>(
        m,
        "QGKMeansIterationStats",
        R"doc(Read-only statistics for one completed clustering iteration.)doc"
    )
        .def_readonly(
            "iteration",
            &QGKMeansIterationStats::iteration,
            R"doc(Iteration number, starting at 1.)doc"
        )
        .def_readonly(
            "obj",
            &QGKMeansIterationStats::obj,
            R"doc(Objective measured before the centroid update.)doc"
        )
        .def_readonly(
            "shift",
            &QGKMeansIterationStats::shift,
            R"doc(Total squared centroid displacement during the update.)doc"
        )
        .def_readonly(
            "nsplit",
            &QGKMeansIterationStats::nsplit,
            R"doc(Number of empty centroids repaired by splitting.)doc"
        );
    m.attr("RaBitQKMeansIterationStats") = m.attr("QGKMeansIterationStats");

    bind_clustering<QGKMeans, QGKMeansParameters>(m, "_QGKMeans", "QGKMeans");
    bind_clustering<RaBitQKMeans, RaBitQKMeansParameters>(
        m, "_RaBitQKMeans", "RaBitQKMeans"
    );
}
