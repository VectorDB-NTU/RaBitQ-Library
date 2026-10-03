#pragma once

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#endif
#if defined(__linux__)
#include <sys/mman.h>
#endif
#if defined(_MSC_VER)
#include <malloc.h>
#endif

#include <cstddef>
#include <cstdlib>
#include <limits>
#include <new>
#include <type_traits>

namespace rabitqlib::memory {
inline void* aligned_allocate_bytes(size_t alignment, size_t size) {
#if defined(_MSC_VER)
    return _aligned_malloc(size, alignment);
#else
    return std::aligned_alloc(alignment, size);
#endif
}

inline void aligned_deallocate(void* ptr) noexcept {
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

inline void advise_huge_pages(void* ptr, size_t size) {
#if defined(__linux__)
    madvise(ptr, size, MADV_HUGEPAGE);
#else
    (void)ptr;
    (void)size;
#endif
}

inline constexpr size_t kHugePageSize = 2UL << 20;

// Huge-page advice is opt-in and based on requested bytes, before rounding.
// Small buffers retain their required alignment; Linux buffers of at least
// 2 MiB also receive huge-page alignment. Other platforms avoid that padding.
template <size_t Alignment, typename T, bool HugePage = false>
inline T* align_allocate(size_t nbytes) {
    static_assert(Alignment >= alignof(T));
    static_assert((Alignment & (Alignment - 1)) == 0, "Alignment must be a power of two");

    if (nbytes == 0) {
        return nullptr;
    }

    const bool huge_pages = HugePage && nbytes >= kHugePageSize;
    size_t alignment = Alignment;
#if defined(__linux__)
    if (huge_pages && alignment < kHugePageSize) {
        alignment = kHugePageSize;
    }
#endif
    const size_t remainder = nbytes % alignment;
    const size_t padding = remainder == 0 ? 0 : alignment - remainder;
    if (nbytes > std::numeric_limits<size_t>::max() - padding) {
        throw std::bad_array_new_length();
    }
    const size_t size = nbytes + padding;
    void* ptr = aligned_allocate_bytes(alignment, size);
    if (ptr == nullptr) {
        throw std::bad_alloc();
    }
    if (huge_pages) {
        advise_huge_pages(ptr, size);
    }
    return static_cast<T*>(ptr);
}

template <typename T, size_t Alignment = 64, bool HugePage = false>
class AlignedAllocator {
   private:
    static_assert(Alignment >= alignof(T));
    static_assert((Alignment & (Alignment - 1)) == 0, "Alignment must be a power of two");

    template <typename U>
    using ReboundAllocator = AlignedAllocator<U, Alignment, HugePage>;

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

        return align_allocate<Alignment, T, HugePage>(n * sizeof(T));
    }

    void deallocate(T* ptr, [[maybe_unused]] std::size_t n) { aligned_deallocate(ptr); }
};

// Retain alignment and object lifetimes without zero-filling scalar storage.
// Callers must initialize every element before reading it.
template <typename T, size_t Alignment = 64, bool HugePage = false>
class DefaultInitAlignedAllocator : public AlignedAllocator<T, Alignment, HugePage> {
    template <typename U>
    using ReboundAllocator = DefaultInitAlignedAllocator<U, Alignment, HugePage>;

   public:
    template <typename U>
    struct rebind {
        using other = ReboundAllocator<U>;
    };

    constexpr DefaultInitAlignedAllocator() noexcept = default;

    template <typename U>
    constexpr explicit DefaultInitAlignedAllocator(const ReboundAllocator<U>&) noexcept {}

    template <typename U>
    void construct(U* ptr) noexcept(std::is_nothrow_default_constructible_v<U>) {
        ::new (static_cast<void*>(ptr)) U;
    }
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

// Use the same size-aware policy as huge-page-enabled STL storage. Advice is
// best-effort; actual page sizes remain under the operating system's control.
template <typename T>
inline T* huge_page_allocate(size_t nbytes) {
    constexpr size_t kAlignment = alignof(T) > 64 ? alignof(T) : 64;
    return align_allocate<kAlignment, T, true>(nbytes);
}

static inline void prefetch_l1(const void* addr) {
#if defined(__SSE2__) || defined(_M_X64) || defined(_M_IX86)
    _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_T0);
#else
    __builtin_prefetch(addr, 0, 3);
#endif
}

static inline void prefetch_l2(const void* addr) {
#if defined(__SSE2__) || defined(_M_X64) || defined(_M_IX86)
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
