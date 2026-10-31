#include "rabitqlib/index/detail/hnsw_dispatch.hpp"

#include <cstddef>
#include <utility>
#include <vector>

#include "../simd/dispatch_resolver.hpp"
#include "rabitqlib/index/types.hpp"
#include "rabitqlib/utils/cpu_features.hpp"

namespace rabitqlib::hnsw::detail {
namespace {
using SearchKnnFn =
    std::vector<std::pair<float, PID>> (*)(HierarchicalNSW&, const float*, size_t, size_t);
}  // namespace
std::vector<std::pair<float, PID>> search_knn_sorted(
    HierarchicalNSW& index, const float* rotated_query, size_t k, size_t ef_search
) {
#if defined(__x86_64__) || defined(_M_X64)
    // The core variant uses AVX2 warmup; the popcount variant has its own stricter tier.
    static const SearchKnnFn kSearchKnnFn =
        cpu::has_avx512_popcnt() ? search_knn_sorted_avx512_popcnt
                                 : RABITQ_RESOLVE(
                                       search_knn_sorted_avx512_core,
                                       search_knn_sorted_avx2,
                                       search_knn_sorted_generic,
                                       cpu::has_avx512_core() && cpu::has_avx2()
                                   );
#else
    static const SearchKnnFn kSearchKnnFn = simd::resolve_optional_kernel(
        search_knn_sorted_neon, search_knn_sorted_generic, cpu::has_neon()
    );
#endif

    return kSearchKnnFn(index, rotated_query, k, ef_search);
}
}  // namespace rabitqlib::hnsw::detail
