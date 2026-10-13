#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <cstddef>
#include <stdexcept>

#include "bindings_common.hpp"

PYBIND11_MODULE(rabitq_ids_benchmark, module) {
    module.def("convert", [](const pybind11::object& input, size_t clusters) {
        const auto array = pybind11::array::ensure(input);
        if (!array || array.ndim() != 1 || array.size() == 0) {
            throw std::invalid_argument("Benchmark requires a nonempty 1D array");
        }
        const auto result = rabitqlib::python_bindings::checked_cluster_ids(
            input, static_cast<size_t>(array.size()), clusters
        );
        return result.back();
    });
}
