#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>

#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/core/rotator.hpp"
#include "rabitqlib/index/format_error.hpp"
#include "rabitqlib/utils/tools.hpp"

namespace py = pybind11;

namespace {
// rabitqlib.FormatError; the module keeps it alive for the interpreter's lifetime.
PyObject* format_error_type = nullptr;

constexpr const char* kFormatErrorDoc =
    "A damaged or mismatched index file or serialized byte string, or one this release "
    "cannot read. A subclass of both ValueError and RuntimeError.";
}  // namespace

// Forward declarations of registration functions implemented in other cpp files
void register_hnsw(py::module_& m);
void register_ivf(py::module_& m);
void register_symqg(py::module_& m);
void register_kmeans(py::module_& m);
void register_quant(py::module_& m);

PYBIND11_MODULE(_rabitqlib, m) {
    m.doc() = "RabitQ Python bindings combined module";

    m.attr("__version__") = RABITQ_VERSION;

    // One exception for every rejected file or byte string, registered before the
    // modules that raise it.
    {
        const py::object builtins = py::module_::import("builtins");
        const py::tuple bases =
            py::make_tuple(builtins.attr("ValueError"), builtins.attr("RuntimeError"));
        PyObject* type = PyErr_NewExceptionWithDoc(
            "rabitqlib.FormatError", kFormatErrorDoc, bases.ptr(), nullptr
        );
        if (type == nullptr) {
            throw py::error_already_set();
        }
        m.attr("FormatError") = py::reinterpret_steal<py::object>(type);
        format_error_type = type;
    }
    // pybind11 translators take std::exception_ptr by value.
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    py::register_exception_translator([](std::exception_ptr exception) {
        try {
            if (exception) {
                std::rethrow_exception(exception);
            }
        } catch (const rabitqlib::FormatError& error) {
            PyErr_SetString(format_error_type, error.what());
        }
    });
    m.def("_available_cpu_count", &rabitqlib::total_threads);
    m.def("_default_thread_count", &rabitqlib::default_threads);

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
    register_quant(m);
}
