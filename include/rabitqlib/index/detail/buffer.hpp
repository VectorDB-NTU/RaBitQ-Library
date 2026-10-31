#pragma once

#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "rabitqlib/index/types.hpp"
#include "rabitqlib/utils/memory.hpp"

namespace rabitqlib::buffer {

inline constexpr PID kSearchBufferCheckedMask = PID{1}
                                                << (std::numeric_limits<PID>::digits - 1);
inline constexpr size_t kSearchBufferMaxPointCount =
    static_cast<size_t>(kSearchBufferCheckedMask);

// Sorted linear buffer, used as the beam set for graph-based ANN search. In SymphonyQG,
// the search buffer may contain duplicate IDs with different distances.
template <typename T = float>
class SearchBuffer {
   private:
    using Storage = std::vector<AnnCandidate<T>, memory::AlignedAllocator<AnnCandidate<T>>>;

    Storage data_;
    size_t size_ = 0;
    size_t cur_ = 0;
    size_t capacity_ = 0;

    [[nodiscard]] static size_t storage_size(size_t capacity) {
        if (capacity == std::numeric_limits<size_t>::max()) {
            throw std::length_error("SearchBuffer capacity is too large");
        }
        return capacity + 1;
    }

    [[nodiscard]] size_t binary_search(T dist) const {
        size_t lo = 0;
        size_t len = size_;
        while (len > 1) {
            const size_t half = len >> 1;
            len -= half;
            lo += static_cast<size_t>(data_[lo + half - 1].distance < dist) * half;
        }
        return (lo < size_ && data_[lo].distance < dist) ? lo + 1 : lo;
    }

    // Mark a candidate as checked by setting the ID's top bit.
    static void set_checked(PID& internal_id) { internal_id |= kSearchBufferCheckedMask; }

    [[nodiscard]] static bool is_checked(PID internal_id) {
        return (internal_id & kSearchBufferCheckedMask) != 0;
    }

   public:
    SearchBuffer() = default;

    explicit SearchBuffer(size_t capacity)
        : data_(storage_size(capacity)), capacity_(capacity) {}

    // Insert a data point into the buffer.
    void insert(PID internal_id, T dist) {
        if (is_full(dist)) {
            return;
        }

        size_t lo = binary_search(dist);
        std::memmove(&data_[lo + 1], &data_[lo], (size_ - lo) * sizeof(AnnCandidate<T>));
        data_[lo] = AnnCandidate<T>(internal_id, dist);
        size_ += static_cast<size_t>(size_ < capacity_);
        cur_ = lo < cur_ ? lo : cur_;
    }

    // Get the unchecked candidate with the minimum distance.
    PID pop() {
        const PID internal_id = data_[cur_].id;
        set_checked(data_[cur_].id);
        ++cur_;
        while (cur_ < size_ && is_checked(data_[cur_].id)) {
            ++cur_;
        }
        return internal_id;
    }

    void clear() {
        size_ = 0;
        cur_ = 0;
    }

    // Return the candidate ID for the next pop().
    [[nodiscard]] PID next_id() const { return data_[cur_].id; }

    [[nodiscard]] bool has_next() const { return cur_ < size_; }

    void resize(size_t new_size) {
        data_ = Storage(storage_size(new_size));
        capacity_ = new_size;
        clear();
    }

    [[nodiscard]] size_t size() const { return size_; }

    void copy_results(PID* result_ids) const {
        for (size_t i = 0; i < size_; ++i) {
            result_ids[i] = data_[i].id;
        }
    }

    void copy_results(PID* result_ids, T* distances) const {
        for (size_t i = 0; i < size_; ++i) {
            result_ids[i] = data_[i].id;
            distances[i] = data_[i].distance;
        }
    }

    [[nodiscard]] T top_dist() const {
        if (capacity_ == 0) {
            return std::numeric_limits<T>::lowest();
        }
        return is_full() ? data_[size_ - 1].distance : std::numeric_limits<T>::max();
    }

    [[nodiscard]] bool is_full() const { return size_ == capacity_; }

    // Return whether dist cannot be inserted into the buffer.
    [[nodiscard]] bool is_full(T dist) const { return capacity_ == 0 || dist > top_dist(); }

    [[nodiscard]] const Storage& data() const { return data_; }
};
}  // namespace rabitqlib::buffer
