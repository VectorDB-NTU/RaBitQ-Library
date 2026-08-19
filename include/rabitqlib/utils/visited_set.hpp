#pragma once

#include <cstddef>
#include <type_traits>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/utils/visited_set_epoch.hpp"
#include "rabitqlib/utils/visited_set_hash.hpp"

namespace rabitqlib {
/**
 * @brief Selects the visited-set implementation used across the library.
 *
 * A visited set records which vertices a single graph search has already
 * touched. Every implementation is a plain class -- there is no virtual base --
 * so the choice is made at compile time and the search path keeps direct calls.
 * An implementation must provide:
 *
 *   explicit Impl(size_t num_elements);   // num_elements is the ID SPACE
 *   void initialize(size_t num_elements); // (re)size, leaving the set empty
 *   void clear();                         // forget every mark
 *   bool get(PID data_id) const;          // has data_id been marked?
 *   void set(PID data_id);                // mark data_id
 *
 * num_elements is always the id space -- the number of vertices, such that
 * every id passed to get()/set() is < num_elements. An implementation that
 * wants a smaller table derives it internally (HashBasedVisitedSet does).
 *
 * Available implementations:
 *   EpochBasedVisitedSet (visited_set_epoch.hpp) -- default. O(1) clear, no
 *       allocation while searching, 2 bytes per id resident.
 *   HashBasedVisitedSet (visited_set_hash.hpp) -- sublinear memory, but
 *       allocates on collisions and clear() walks the table.
 *
 * To switch the whole library, change this alias -- it is the single place the
 * implementation is named.
 */
namespace detail {
/**
 * @brief Compile-time enforcement of the interface described above.
 *
 * Each assert compares the member's type exactly, so an implementation that
 * takes a PID& or forgets the const on get() fails here, naming the member,
 * rather than at some call site or -- worse -- silently, by binding to a
 * conversion. Instantiating the struct is what fires the asserts.
 *
 * This is a check, not a base class: the implementations stay unrelated
 * concrete types with non-virtual get()/set() that inline into the search
 * loops. A virtual base would enforce the same thing through the type system,
 * but at the price of an indirect call per visited vertex.
 */
template <typename T>
struct check_visited_set {
    static_assert(
        std::is_constructible<T, size_t>::value,
        "visited set must be constructible from a size_t element count"
    );
    static_assert(
        std::is_same<decltype(&T::initialize), void (T::*)(size_t)>::value,
        "visited set must declare: void initialize(size_t)"
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
    static constexpr bool value = true;
};
}  // namespace detail

static_assert(detail::check_visited_set<EpochBasedVisitedSet>::value, "");
static_assert(detail::check_visited_set<HashBasedVisitedSet>::value, "");

using VisitedSet = HashBasedBooleanSet;
}  // namespace rabitqlib
