#pragma once

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#include <sys/mman.h>

#include <cstddef>
#include <cstdlib>
#include <limits>
#include <new>
#include <type_traits>

namespace rabitqlib::memory {
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))

template <typename T, size_t Alignment = 64, bool HugePage = false>
class AlignedAllocator {
   private:
    static_assert(Alignment >= alignof(T));
    static_assert((Alignment & (Alignment - 1)) == 0, "Alignment must be a power of two");

    template <typename U>
    using ReboundAllocator = AlignedAllocator<U, Alignment, HugePage>;

    [[nodiscard]] static constexpr size_t aligned_size(size_t nbytes) {
        const size_t remainder = nbytes % Alignment;
        if (remainder == 0) {
            return nbytes;
        }
        const size_t padding = Alignment - remainder;
        if (nbytes > std::numeric_limits<size_t>::max() - padding) {
            throw std::bad_array_new_length();
        }
        return nbytes + padding;
    }

   public:
    using value_type = T;
    using is_always_equal = std::true_type;

    template <class U>
    struct rebind {
        using other = ReboundAllocator<U>;
    };

    constexpr AlignedAllocator() noexcept = default;

    constexpr AlignedAllocator(const AlignedAllocator&) noexcept = default;

    template <typename U>
    constexpr explicit AlignedAllocator(const ReboundAllocator<U>&) noexcept {}

    friend constexpr bool
    operator==(const AlignedAllocator&, const AlignedAllocator&) noexcept {
        return true;
    }

    friend constexpr bool
    operator!=(const AlignedAllocator&, const AlignedAllocator&) noexcept {
        return false;
    }

    [[nodiscard]] T* allocate(std::size_t n) {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length();
        }

        if (n == 0) {
            return nullptr;
        }

        const auto nbytes = aligned_size(n * sizeof(T));
        auto* ptr = std::aligned_alloc(Alignment, nbytes);
        if (ptr == nullptr) {
            throw std::bad_alloc();
        }
        if (HugePage) {
            madvise(ptr, nbytes, MADV_HUGEPAGE);
        }
        return reinterpret_cast<T*>(ptr);
    }

    void deallocate(T* ptr, [[maybe_unused]] std::size_t n) { std::free(ptr); }
};

template <typename T>
struct Allocator {
   public:
    using value_type = T;

    constexpr Allocator() noexcept = default;

    template <typename U>
    explicit constexpr Allocator(const Allocator<U>&) noexcept {}

    [[nodiscard]] constexpr T* allocate(std::size_t n) { return ::new T[n]; }

    constexpr void deallocate(T* ptr, [[maybe_unused]] size_t n) noexcept {
        ::delete[] ptr;
    }

    // Intercept zero-argument construction to do default initialization.
    template <typename U>
    void construct(U* ptr) noexcept(std::is_nothrow_default_constructible_v<U>) {
        ::new (static_cast<void*>(ptr)) U;
    }
};

template <size_t Alignment, typename T, bool HugePage = false>
inline T* align_allocate(size_t nbytes) {
    static_assert(Alignment >= alignof(T));
    static_assert((Alignment & (Alignment - 1)) == 0, "Alignment must be a power of two");

    if (nbytes == 0) {
        return nullptr;
    }

    const size_t remainder = nbytes % Alignment;
    const size_t padding = remainder == 0 ? 0 : Alignment - remainder;
    if (nbytes > std::numeric_limits<size_t>::max() - padding) {
        throw std::bad_array_new_length();
    }
    const size_t size = nbytes + padding;
    void* ptr = std::aligned_alloc(Alignment, size);
    if (ptr == nullptr) {
        throw std::bad_alloc();
    }
    if (HugePage) {
        madvise(ptr, size, MADV_HUGEPAGE);
    }
    return static_cast<T*>(ptr);
}

inline constexpr size_t kHugePageSize = 2UL << 20;

// Allocate a large buffer backed by transparent huge pages.
template <typename T>
inline T* huge_page_allocate(size_t nbytes) {
    return align_allocate<kHugePageSize, T, true>(nbytes);
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
