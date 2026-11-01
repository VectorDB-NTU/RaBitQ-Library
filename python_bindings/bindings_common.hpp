#pragma once

#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "rabitqlib/core/metric.hpp"
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

// A seed is an integer in [0, max_seed]; bool is rejected like other integer options.
inline uint64_t checked_seed(py::handle value, uint64_t max_seed) {
    const std::string range =
        "seed must be an integer in [0, " + std::to_string(max_seed) + "]";
    if (PyBool_Check(value.ptr())) {
        throw py::type_error(range);
    }
    auto integer = py::reinterpret_steal<py::object>(PyNumber_Index(value.ptr()));
    if (!integer) {
        PyErr_Clear();
        throw py::type_error(range);
    }
    const unsigned long long seed = PyLong_AsUnsignedLongLong(integer.ptr());
    if (PyErr_Occurred() != nullptr) {
        PyErr_Clear();
        throw std::invalid_argument(range);
    }
    if (seed > max_seed) {
        throw std::invalid_argument(range);
    }
    return seed;
}

// Call only after rank validation.
inline void check_centroid_shape(
    const py::array& centroids, size_t num_clusters, size_t dim
) {
    if (static_cast<size_t>(centroids.shape(0)) != num_clusters ||
        static_cast<size_t>(centroids.shape(1)) != dim) {
        throw std::invalid_argument(
            "centroids must have shape (num_clusters, dim); expected (" +
            std::to_string(num_clusters) + ", " + std::to_string(dim) + "), got shape " +
            array_shape(centroids)
        );
    }
}

// Training input rows, which must cover num_clusters centroids. Call only after
// rank validation.
inline size_t training_rows(const py::array& x, size_t num_clusters) {
    const auto rows = static_cast<size_t>(x.shape(0));
    if (rows < num_clusters) {
        throw std::invalid_argument(
            "x must have at least num_clusters rows; expected at least " +
            std::to_string(num_clusters) + ", got shape " + array_shape(x)
        );
    }
    return rows;
}

// The parameters for one train call: an explicit num_threads overrides cp's.
template <typename Parameters>
Parameters training_parameters(Parameters cp, std::optional<size_t> num_threads) {
    if (num_threads.has_value()) {
        cp.num_threads = static_cast<uint32_t>(
            std::min<size_t>(*num_threads, std::numeric_limits<int>::max())
        );
    }
    return cp;
}

inline MetricType metric_from_string(const std::string& metric) {
    if (metric == "l2") {
        return METRIC_L2;
    }
    if (metric == "ip" || metric == "innerproduct") {
        return METRIC_IP;
    }
    throw std::invalid_argument("metric must be 'l2' or 'ip'");
}

inline std::string metric_to_string(MetricType metric) {
    return metric == METRIC_IP ? "ip" : "l2";
}

template <typename T>
inline py::array_t<T, py::array::c_style | py::array::forcecast> ensure_array_rank(
    py::handle value, const char* name, py::ssize_t rank
) {
    auto array = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(value);
    if (!array) {
        throw std::invalid_argument(
            std::string(name) + " must be convertible to a NumPy array of dtype " +
            py::str(py::dtype::of<T>()).cast<std::string>() + "; got " +
            array_input_type(value)
        );
    }
    if (array.ndim() != rank) {
        throw std::invalid_argument(
            std::string(name) + " must be a " + std::to_string(rank) +
            "D NumPy array; got shape " + array_shape(array)
        );
    }
    return array;
}

template <typename T>
inline py::array_t<T, py::array::c_style | py::array::forcecast> ensure_2d_array(
    py::handle value, const char* name
) {
    auto array = ensure_array_rank<T>(value, name, 2);
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
    return ensure_array_rank<T>(value, name, 1);
}

struct SearchResults {
    py::array_t<PID> ids;
    py::array_t<float> distances;
};

// (num_queries, k) outputs. Slots without a neighbor keep kPidMax and +inf.
inline SearchResults unfilled_search_results(size_t num_queries, size_t k) {
    const auto shape = std::vector<py::ssize_t>{
        static_cast<py::ssize_t>(num_queries), static_cast<py::ssize_t>(k)};
    SearchResults results{py::array_t<PID>(shape), py::array_t<float>(shape)};
    std::fill(
        results.ids.mutable_data(), results.ids.mutable_data() + results.ids.size(), kPidMax
    );
    std::fill(
        results.distances.mutable_data(),
        results.distances.mutable_data() + results.distances.size(),
        std::numeric_limits<float>::infinity()
    );
    return results;
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

// The caller has already validated rank/dtype. Removed points still occupy IDs.
inline std::vector<PID> checked_removal_ids(
    const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& ids,
    size_t num_stored_points
) {
    std::vector<PID> result;
    result.reserve(static_cast<size_t>(ids.shape(0)));
    const detail::ConstPackedArrayView<int64_t> values(
        reinterpret_cast<const char*>(ids.data())
    );
    for (py::ssize_t i = 0; i < ids.shape(0); ++i) {
        const int64_t id = values[static_cast<size_t>(i)];
        if (id < 0 || static_cast<uint64_t>(id) >= num_stored_points) {
            throw std::invalid_argument("ids must be in [0, num_stored_points)");
        }
        result.push_back(static_cast<PID>(id));
    }
    return result;
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
        const detail::ConstPackedArrayView<Value> values(
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
