#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>

#include <cstddef>
#include <cstdint>
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
        throw std::invalid_argument(std::string(name) + " must be a NumPy array");
    }
    if (array.ndim() != 2) {
        throw std::invalid_argument(std::string(name) + " must be a 2D NumPy array");
    }
    return array;
}

template <typename T>
inline py::array_t<T, py::array::c_style | py::array::forcecast> ensure_1d_array(
    py::handle value, const char* name
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

// Like ensure_1d_array<int64_t>, but refuses input whose dtype is not an integer. forcecast
// would otherwise truncate floats and turn booleans or numeric strings into ids, and
// callers use these ids to remove points permanently. An empty sequence has no dtype to
// check.
inline py::array_t<int64_t, py::array::c_style | py::array::forcecast>
ensure_1d_integer_array(py::handle value, const char* name) {
    auto typed = py::array::ensure(value);
    if (!typed) {
        throw std::invalid_argument(std::string(name) + " must be a NumPy array");
    }
    const char kind = typed.dtype().kind();
    if (typed.size() > 0 && kind != 'i' && kind != 'u') {
        throw std::invalid_argument(std::string(name) + " must contain integers");
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
                "cluster_ids length must match number of rows in data"
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
                throw std::invalid_argument(range_error);
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
