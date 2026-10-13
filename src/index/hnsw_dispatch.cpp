#include "rabitqlib/index/detail/hnsw_dispatch.hpp"

#include <cstddef>
#include <queue>
#include <utility>
#include <vector>

#include "../simd/dispatch_resolver.hpp"
#include "rabitqlib/index/types.hpp"
#include "rabitqlib/utils/cpu_features.hpp"

namespace rabitqlib::hnsw::detail {
namespace {
using SearchKnnFn =
    std::vector<std::pair<float, PID>> (*)(HierarchicalNSW&, const float*, size_t, size_t);
// The core variant uses AVX2 warmup; the popcount variant has its own stricter tier.
}  // namespace
std::vector<std::pair<float, PID>> search_knn_sorted(
    HierarchicalNSW& index, const float* query, size_t topk, size_t ef
) {
#if defined(__x86_64__) || defined(_M_X64)
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

    return kSearchKnnFn(index, query, topk, ef);
}
std::priority_queue<std::pair<float, PID>> search_knn(
    HierarchicalNSW& index, const float* query, size_t topk, size_t ef
) {
    auto results = search_knn_sorted(index, query, topk, ef);
    return {results.begin(), results.end()};
}
}  // namespace rabitqlib::hnsw::detail
