#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/utils/tools.hpp"

namespace py = pybind11;

// Forward declarations of registration functions implemented in other cpp files
void register_hnsw(py::module_& m);
void register_ivf(py::module_& m);
void register_symqg(py::module_& m);
void register_kmeans(py::module_& m);

PYBIND11_MODULE(_rabitqlib, m) {
    m.doc() = "RabitQ Python bindings combined module";

    m.def("_available_cpu_count", &rabitqlib::total_threads);

    // Register shared enums once
    py::enum_<rabitqlib::MetricType>(m, "MetricType")
        .value("L2", rabitqlib::METRIC_L2)
        .value("IP", rabitqlib::METRIC_IP)
        .export_values();

    py::enum_<rabitqlib::RotatorType>(m, "RotatorType")
        .value("FhtKacRotator", rabitqlib::RotatorType::FhtKacRotator)
        .value("MatrixRotator", rabitqlib::RotatorType::MatrixRotator)
        .export_values();

    // Register each index's bindings into the same module
    register_hnsw(m);
    register_ivf(m);
    register_symqg(m);
    register_kmeans(m);
}
