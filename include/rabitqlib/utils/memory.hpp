#pragma once

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#include <sys/mman.h>

#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <type_traits>

namespace rabitqlib::memory {
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))

namespace detail {
template <size_t Alignment>
[[nodiscard]] constexpr size_t aligned_allocation_size(size_t bytes) {
    static_assert(Alignment >= alignof(void*));
    static_assert((Alignment & (Alignment - 1)) == 0, "Alignment must be a power of two");

    if (bytes == 0) {
        return 0;
    }
    if (bytes > std::numeric_limits<size_t>::max() - (Alignment - 1)) {
        throw std::length_error("aligned allocation size overflow");
    }
    return (bytes + (Alignment - 1)) & ~(Alignment - 1);
}
}  // namespace detail

template <typename T, size_t Alignment = 64, bool HugePage = false>
class AlignedAllocator {
   private:
    static_assert(Alignment >= alignof(T));
    static_assert(Alignment >= alignof(void*));
    static_assert((Alignment & (Alignment - 1)) == 0, "Alignment must be a power of two");

   public:
    using value_type = T;

    template <class U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment, HugePage>;
    };

    using is_always_equal = std::true_type;

    constexpr AlignedAllocator() noexcept = default;

    constexpr AlignedAllocator(const AlignedAllocator&) noexcept = default;

    template <typename U>
    constexpr explicit AlignedAllocator(AlignedAllocator<
                                        U,
                                        Alignment,
                                        HugePage> const&) noexcept {}

    template <typename U>
    [[nodiscard]] constexpr bool operator==(const AlignedAllocator<U, Alignment, HugePage>&)
        const noexcept {
        return true;
    }

    template <typename U>
    [[nodiscard]] constexpr bool operator!=(const AlignedAllocator<U, Alignment, HugePage>&)
        const noexcept {
        return false;
    }

    [[nodiscard]] T* allocate(std::size_t n) {
        if (n == 0) {
            return nullptr;
        }
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length();
        }

        const size_t nbytes = detail::aligned_allocation_size<Alignment>(n * sizeof(T));
        auto* ptr = std::aligned_alloc(Alignment, nbytes);
        if (ptr == nullptr) {
            throw std::bad_alloc();
        }
        if constexpr (HugePage) {
            static_cast<void>(madvise(ptr, nbytes, MADV_HUGEPAGE));
        }
        return reinterpret_cast<T*>(ptr);
    }

    void deallocate(T* ptr, [[maybe_unused]] std::size_t n) noexcept { std::free(ptr); }
};

template <typename T>
class Allocator {
   public:
    using value_type = T;
    using is_always_equal = std::true_type;

    template <typename U>
    struct rebind {
        using other = Allocator<U>;
    };

    constexpr Allocator() noexcept = default;

    template <typename U>
    constexpr Allocator(const Allocator<U>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t n) {
        if (n == 0) {
            return nullptr;
        }
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length();
        }
        return std::allocator<T>{}.allocate(n);
    }

    void deallocate(T* ptr, std::size_t n) noexcept {
        if (ptr != nullptr) {
            std::allocator<T>{}.deallocate(ptr, n);
        }
    }

    template <typename U>
    [[nodiscard]] constexpr bool operator==(const Allocator<U>&) const noexcept {
        return true;
    }

    template <typename U>
    [[nodiscard]] constexpr bool operator!=(const Allocator<U>&) const noexcept {
        return false;
    }
};

template <size_t Alignment, typename T, bool HugePage = false>
inline T* align_allocate(size_t nbytes) {
    static_assert(Alignment >= alignof(T));
    const size_t size = detail::aligned_allocation_size<Alignment>(nbytes);
    if (size == 0) {
        return nullptr;
    }
    void* ptr = std::aligned_alloc(Alignment, size);
    if (ptr == nullptr) {
        throw std::bad_alloc();
    }
    if constexpr (HugePage) {
        static_cast<void>(madvise(ptr, size, MADV_HUGEPAGE));
    }
    return static_cast<T*>(ptr);
}

static inline void prefetch_l1(const void* addr) {
#if defined(__SSE2__)
    _mm_prefetch(addr, _MM_HINT_T0);
#else
    __builtin_prefetch(addr, 0, 3);
#endif
}

static inline void prefetch_l2(const void* addr) {
#if defined(__SSE2__)
    _mm_prefetch((const char*)addr, _MM_HINT_T1);
#else
    __builtin_prefetch(addr, 0, 2);
#endif
}

inline void mem_prefetch_l1(const char* ptr, size_t num_lines) {
    // The repeated fallthrough branches intentionally unroll up to 20 prefetches.
    // NOLINTBEGIN(bugprone-branch-clone)
    switch (num_lines) {
        default:
            [[fallthrough]];
        case 20:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 19:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 18:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 17:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 16:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 15:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 14:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 13:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 12:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 11:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 10:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 9:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 8:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 7:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 6:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 5:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 4:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 3:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 2:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 1:
            prefetch_l1(ptr);
            [[fallthrough]];
        case 0:
            break;
    }
    // NOLINTEND(bugprone-branch-clone)
}

inline void mem_prefetch_l2(const char* ptr, size_t num_lines) {
    // The repeated fallthrough branches intentionally unroll up to 20 prefetches.
    // NOLINTBEGIN(bugprone-branch-clone)
    switch (num_lines) {
        default:
            [[fallthrough]];
        case 20:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 19:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 18:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 17:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 16:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 15:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 14:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 13:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 12:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 11:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 10:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 9:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 8:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 7:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 6:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 5:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 4:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 3:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 2:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 1:
            prefetch_l2(ptr);
            [[fallthrough]];
        case 0:
            break;
    }
    // NOLINTEND(bugprone-branch-clone)
}
}  // namespace rabitqlib::memory
