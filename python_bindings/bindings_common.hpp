#pragma once

#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/core/rotator.hpp"
#include "rabitqlib/index/types.hpp"
#include "rabitqlib/utils/packed.hpp"

namespace py = pybind11;

namespace rabitqlib::python_bindings {

// All guard acquisition/destruction happens with the GIL held. Native work may
// release the GIL inside the guard's lifetime, but must reacquire it first on
// exit (including exceptions). This makes conflicts fail immediately without
// blocking the GIL or introducing a lock-order dependency.
class IndexAccess {
   private:
    size_t readers_ = 0;
    bool writing_ = false;

   public:
    class Guard {
       private:
        IndexAccess& access_;
        bool writing_;

       public:
        Guard(IndexAccess& access, bool writing) : access_(access), writing_(writing) {
            if (access.writing_ || (writing && access.readers_ != 0)) {
                throw std::runtime_error(
                    "Index is busy: conflicting operation in progress"
                );
            }
            if (writing) {
                access.writing_ = true;
            } else {
                ++access.readers_;
            }
        }
        Guard(const Guard&) = delete;
        Guard& operator=(const Guard&) = delete;
        ~Guard() {
            if (writing_) {
                access_.writing_ = false;
            } else {
                --access_.readers_;
            }
        }
    };

    [[nodiscard]] Guard read() { return Guard(*this, false); }
    [[nodiscard]] Guard write() { return Guard(*this, true); }
};

// Resolve Python callbacks before index guards and own bytes across GIL release.
inline std::string path_string(py::handle value) {
    // The old std::string caster also accepted bytearray directly.
    auto path = PyByteArray_Check(value.ptr())
                    ? py::reinterpret_borrow<py::object>(value)
                    : py::reinterpret_steal<py::object>(PyOS_FSPath(value.ptr()));
    if (!path) {
        throw py::error_already_set();
    }
    if (PyUnicode_Check(path.ptr())) {
#if defined(_WIN32)
        path = py::reinterpret_steal<py::object>(PyUnicode_AsUTF8String(path.ptr()));
#else
        path = py::reinterpret_steal<py::object>(PyUnicode_EncodeFSDefault(path.ptr()));
#endif
        if (!path) {
            throw py::error_already_set();
        }
    }
    auto result = py::cast<std::string>(path);
    if (result.find('\0') != std::string::npos) {
        throw std::invalid_argument("path must not contain embedded null bytes");
    }
    return result;
}

inline std::string array_shape(const py::array& array) {
    std::string result = "(";
    for (py::ssize_t axis = 0; axis < array.ndim(); ++axis) {
        if (axis != 0) {
            result += ", ";
        }
        result += std::to_string(array.shape(axis));
    }
    if (array.ndim() == 1) {
        result += ',';
    }
    return result + ')';
}

inline std::string array_input_type(py::handle value) {
    auto result = py::str(py::type::of(value).attr("__name__")).cast<std::string>();
    if (py::isinstance<py::array>(value)) {
        result +=
            " with dtype " +
            py::str(py::reinterpret_borrow<py::array>(value).dtype()).cast<std::string>();
    }
    return result;
}

// Call only after rank validation. Formatting is confined to the error path.
inline void ensure_dimension(const py::array& array, size_t expected, const char* message) {
    if (static_cast<size_t>(array.shape(1)) != expected) {
        throw std::invalid_argument(
            std::string(message) + "; expected last dimension " + std::to_string(expected) +
            ", got shape " + array_shape(array)
        );
    }
}

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
    py::handle value, const char* name
) {
    auto array = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(value);
    if (!array) {
        throw std::invalid_argument(
            std::string(name) + " must be convertible to a NumPy array of dtype " +
            py::str(py::dtype::of<T>()).cast<std::string>() + "; got " +
            array_input_type(value)
        );
    }
    if (array.ndim() != 2) {
        throw std::invalid_argument(
            std::string(name) + " must be a 2D NumPy array; got shape " + array_shape(array)
        );
    }
    // C-contiguous NumPy views may start at an unaligned byte offset. Native
    // scalar accesses require alignof(T), independently of SIMD load alignment.
    if (reinterpret_cast<std::uintptr_t>(array.data()) % alignof(T) != 0) {
        py::array_t<T, py::array::c_style | py::array::forcecast> aligned(
            {array.shape(0), array.shape(1)}
        );
        if (array.size() != 0) {
            std::memcpy(
                aligned.mutable_data(), array.data(), static_cast<size_t>(array.nbytes())
            );
        }
        return aligned;
    }
    return array;
}

template <typename T>
inline py::array_t<T, py::array::c_style | py::array::forcecast> ensure_1d_array(
    py::handle value, const char* name
) {
    auto array = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(value);
    if (!array) {
        throw std::invalid_argument(
            std::string(name) + " must be convertible to a NumPy array of dtype " +
            py::str(py::dtype::of<T>()).cast<std::string>() + "; got " +
            array_input_type(value)
        );
    }
    if (array.ndim() != 1) {
        throw std::invalid_argument(
            std::string(name) + " must be a 1D NumPy array; got shape " + array_shape(array)
        );
    }
    return array;
}

// Like ensure_1d_array<int64_t>, but refuses input whose dtype is not an integer. forcecast
// would otherwise truncate floats and turn booleans or numeric strings into ids, and
// callers use these ids to remove points permanently. An empty sequence has no dtype to
// check.
inline py::array_t<int64_t, py::array::c_style | py::array::forcecast>
ensure_1d_integer_array(py::handle value, const char* name) {
    auto typed = py::array::ensure(value);
    if (!typed) {
        throw std::invalid_argument(
            std::string(name) + " must be convertible to an integer NumPy array; got " +
            array_input_type(value)
        );
    }
    const char kind = typed.dtype().kind();
    if (typed.size() > 0 && kind != 'i' && kind != 'u') {
        throw std::invalid_argument(
            std::string(name) + " must contain integers; got dtype " +
            py::str(typed.dtype()).cast<std::string>()
        );
    }
    return ensure_1d_array<int64_t>(value, name);
}

// Validate before narrowing, and own the result for APIs that mutate assignments.
inline std::vector<PID> checked_cluster_ids(
    py::handle value,
    size_t rows,
    size_t clusters,
    const char* range_error = "cluster_ids must be in [0, num_clusters)"
) {
    const auto checked_copy = [&](const auto& ids) {
        if (static_cast<size_t>(ids.shape(0)) != rows) {
            throw std::invalid_argument(
                "cluster_ids length must match number of rows in data; expected " +
                std::to_string(rows) + " IDs, got " + std::to_string(ids.shape(0))
            );
        }
        std::vector<PID> result(rows);
        using Value = typename std::decay_t<decltype(ids)>::value_type;
        // C-contiguous NumPy arrays can still have an unaligned data pointer.
        const rabitqlib::detail::ConstPackedArrayView<Value> values(
            reinterpret_cast<const char*>(ids.data())
        );
        for (size_t i = 0; i < rows; ++i) {
            const int64_t id = values[i];
            if (id < 0 || static_cast<uint64_t>(id) >= clusters ||
                static_cast<uint64_t>(id) > std::numeric_limits<PID>::max()) {
                throw std::invalid_argument(
                    std::string(range_error) + "; expected [0, " +
                    std::to_string(clusters) + "), invalid entry at index " +
                    std::to_string(i)
                );
            }
            result[i] = static_cast<PID>(id);
        }
        return result;
    };
    // Already-native IDs need no intermediate int64 array. Keep the owned copy
    // required by construction APIs, and validate it just like wider integers.
    if (py::isinstance<py::array_t<PID>>(value)) {
        return checked_copy(ensure_1d_array<PID>(value, "cluster_ids"));
    }
    return checked_copy(ensure_1d_integer_array(value, "cluster_ids"));
}

}  // namespace rabitqlib::python_bindings
