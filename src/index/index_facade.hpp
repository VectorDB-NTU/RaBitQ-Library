#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

#include "rabitqlib/clustering/rabitqkmeans.hpp"
#include "rabitqlib/index/detail/buffer.hpp"
#include "rabitqlib/index/types.hpp"

// Checks and settings that the IvfIndex, HnswIndex and SymqgIndex classes share. Their
// messages are the ones the Python classes raise.
namespace rabitqlib::index_facade {

// The parameters of one training call: an explicit num_threads overrides cp's.
inline rabitqkmeans::RaBitQKMeansParameters with_num_threads(
    rabitqkmeans::RaBitQKMeansParameters cp, std::optional<size_t> num_threads
) {
    if (num_threads.has_value()) {
        cp.num_threads = static_cast<uint32_t>(
            std::min<size_t>(*num_threads, std::numeric_limits<int>::max())
        );
    }
    return cp;
}

inline void check_k(size_t k) {
    if (k == 0 || k > buffer::kSearchBufferMaxPointCount) {
        throw std::invalid_argument("k must be between 1 and 2^31");
    }
}

// A search breadth (nprobe or ef_search) is either unset, for the default, or at least 1.
inline void check_breadth(std::optional<size_t> breadth, const char* name) {
    if (breadth == size_t{0}) {
        throw std::invalid_argument(
            std::string(name) + " must be at least 1; leave it unset for the default"
        );
    }
}

// The output of a nonempty batch: ids is required and num_queries * k results fit.
inline void check_outputs(
    const char* class_name, size_t num_queries, size_t k, const Id* ids
) {
    if (ids == nullptr) {
        throw std::invalid_argument(
            std::string(class_name) + " search ids must not be null"
        );
    }
    if (num_queries > std::numeric_limits<size_t>::max() / sizeof(Id) / k) {
        throw std::length_error(std::string(class_name) + " query batch is too large");
    }
}

}  // namespace rabitqlib::index_facade
