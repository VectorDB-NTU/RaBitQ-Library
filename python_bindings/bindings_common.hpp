#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/utils/rotator.hpp"

namespace py = pybind11;

namespace rabitqlib::python_bindings {

inline rabitqlib::MetricType metric_from_string(const std::string& metric) {
    if (metric == "l2") {
        return rabitqlib::METRIC_L2;
    }
    if (metric == "ip" || metric == "innerproduct") {
        return rabitqlib::METRIC_IP;
    }
    throw std::invalid_argument("Unsupported metric. Use 'l2' or 'ip'.");
}

inline std::string metric_to_string(rabitqlib::MetricType metric) {
    return metric == rabitqlib::METRIC_IP ? "ip" : "l2";
}

inline rabitqlib::RotatorType rotator_from_string(const std::string& method) {
    if (method == "matrix") {
        return rabitqlib::RotatorType::MatrixRotator;
    }
    if (method == "fht_kac" || method == "fht") {
        return rabitqlib::RotatorType::FhtKacRotator;
    }
    throw std::invalid_argument("Unsupported rotator method. Use 'fht_kac' or 'matrix'.");
}

template <typename T>
inline py::array_t<T, py::array::c_style | py::array::forcecast> ensure_2d_array(
    py::handle value,
    const char* name
) {
    auto array = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(value);
    if (!array) {
        throw std::invalid_argument(std::string(name) + " must be a NumPy array");
    }
    if (array.ndim() != 2) {
        throw std::invalid_argument(std::string(name) + " must be a 2D NumPy array");
    }
    return array;
}

template <typename T>
inline py::array_t<T, py::array::c_style | py::array::forcecast> ensure_1d_array(
    py::handle value,
    const char* name
) {
    auto array = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(value);
    if (!array) {
        throw std::invalid_argument(std::string(name) + " must be a NumPy array");
    }
    if (array.ndim() != 1) {
        throw std::invalid_argument(std::string(name) + " must be a 1D NumPy array");
    }
    return array;
}

}  // namespace rabitqlib::python_bindings