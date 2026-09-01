#pragma once

#include <cstddef>
#include <type_traits>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/utils/visited_set_epoch.hpp"
#include "rabitqlib/utils/visited_set_hash.hpp"

namespace rabitqlib {
namespace detail {
template <typename T>
struct CheckVisitedSet {
    static_assert(
        std::is_constructible<T, size_t>::value,
        "visited set must be constructible from a size_t id space"
    );
    static_assert(
        std::is_constructible<T, size_t, size_t>::value,
        "visited set must be constructible from (num_elements, size_hint)"
    );
    static_assert(
        std::is_same<decltype(&T::initialize), void (T::*)(size_t, size_t)>::value,
        "visited set must declare: void initialize(size_t num_elements, size_t size_hint)"
    );
    static_assert(
        std::is_same<decltype(&T::clear), void (T::*)()>::value,
        "visited set must declare: void clear()"
    );
    static_assert(
        std::is_same<decltype(&T::get), bool (T::*)(PID) const>::value,
        "visited set must declare: bool get(PID) const"
    );
    static_assert(
        std::is_same<decltype(&T::set), void (T::*)(PID)>::value,
        "visited set must declare: void set(PID)"
    );
    static constexpr bool kValue = true;
};
}  // namespace detail

static_assert(detail::CheckVisitedSet<EpochBasedVisitedSet>::kValue, "");
static_assert(detail::CheckVisitedSet<HashBasedVisitedSet>::kValue, "");

using VisitedSet = HashBasedVisitedSet;
}  // namespace rabitqlib
