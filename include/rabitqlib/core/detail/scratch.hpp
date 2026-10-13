#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <memory_resource>
#include <stdexcept>
#include <utility>
#include <vector>

#include "rabitqlib/core/detail/eigen.hpp"

namespace rabitqlib::quant::rabitq_impl {
// Every scratch array has one entry per coordinate. Walk the same declarations
// either to derive a reusable size budget or to bind a caller's bounded resource.
class ScratchAllocator {
    std::pmr::memory_resource* resource_ = nullptr;
    size_t dim_ = 0;
    size_t bytes_per_coordinate_ = 0;
    size_t alignment_padding_ = 0;

    template <typename T>
    void measure() {
        static_assert(alignof(T) <= alignof(std::max_align_t));
        const size_t maximum = std::numeric_limits<size_t>::max();
        if (sizeof(T) > maximum - bytes_per_coordinate_ ||
            alignof(T) - 1 > maximum - alignment_padding_) {
            throw std::length_error("Quantization workspace size overflows size_t");
        }
        bytes_per_coordinate_ += sizeof(T);
        // Each allocation may need at most alignof(T)-1 bytes before its data.
        alignment_padding_ += alignof(T) - 1;
    }

   public:
    ScratchAllocator() = default;
    ScratchAllocator(std::pmr::memory_resource* resource, size_t dim)
        : resource_(resource), dim_(dim) {}

    [[nodiscard]] size_t bytes(size_t dim) const {
        if (bytes_per_coordinate_ != 0 &&
            dim > (std::numeric_limits<size_t>::max() - alignment_padding_) /
                      bytes_per_coordinate_) {
            throw std::length_error("Quantization workspace size overflows size_t");
        }
        return dim * bytes_per_coordinate_ + alignment_padding_;
    }
    [[nodiscard]] std::pmr::memory_resource* resource() const { return resource_; }

    template <typename T>
    T* array() {
        if (resource_ == nullptr) {
            measure<T>();
            return nullptr;
        }
        auto* result = std::pmr::polymorphic_allocator<T>(resource_).allocate(dim_);
        // Establish C++17 object lifetimes without initializing scalar values.
        std::uninitialized_default_construct_n(result, dim_);
        return result;
    }

    template <typename T>
    std::pmr::vector<T> vector() {
        std::pmr::vector<T> result(
            resource_ == nullptr ? std::pmr::null_memory_resource() : resource_
        );
        if (resource_ == nullptr) {
            measure<T>();
        } else {
            result.reserve(dim_);
        }
        return result;
    }
};

using RescaleEvent = std::pair<double, size_t>;

// This declaration supplies both requirements()'s fallback budget and the
// actual event-sweep containers. Sizing returns empty, non-allocating vectors.
inline auto event_sweep_buffers(ScratchAllocator& allocator) {
    return std::pair{allocator.vector<RescaleEvent>(), allocator.vector<int>()};
}

// Internal non-owning scratch; never part of the stable public API.
template <typename T>
struct Scratch {
    T* residual;
    T* absolute;
    T* centered;
    int* binary;
    int* total;
    double* magnitudes;
    double* reciprocals;
    std::pmr::memory_resource* resource;
};

template <typename T>
inline RowMajorArrayMap<T> scratch_array(T* external, size_t dim, RowMajorArray<T>& owned) {
    if (external == nullptr) {
        owned.resize(1, static_cast<long>(dim));
        external = owned.data();
    }
    return RowMajorArrayMap<T>(external, 1, static_cast<long>(dim));
}
}  // namespace rabitqlib::quant::rabitq_impl
