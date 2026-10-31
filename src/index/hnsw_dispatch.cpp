#include "rabitqlib/index/detail/hnsw_dispatch.hpp"

#include <cstddef>
#include <utility>
#include <vector>

#include "../simd/dispatch_resolver.hpp"
#include "hnsw_search_impl.hpp"
#include "rabitqlib/index/types.hpp"
#include "rabitqlib/utils/cpu_features.hpp"

namespace rabitqlib::hnsw::detail {
std::vector<std::pair<float, PID>> search_knn_sorted(
    HierarchicalNSW& index, const float* rotated_query, size_t k, size_t ef_search
) {
#if defined(__x86_64__) || defined(_M_X64)
    // The core variant uses AVX2 warmup; the popcount variant has its own stricter tier.
    static const SearchLayersFn kSearchLayersFn =
        cpu::has_avx512_popcnt() ? search_layers_avx512_popcnt
                                 : RABITQ_RESOLVE(
                                       search_layers_avx512_core,
                                       search_layers_avx2,
                                       search_layers_generic,
                                       cpu::has_avx512_core() && cpu::has_avx2()
                                   );
#else
    static const SearchLayersFn kSearchLayersFn = simd::resolve_optional_kernel(
        search_layers_neon, search_layers_generic, cpu::has_neon()
    );
#endif

    return HnswSearchAccess::search(index, kSearchLayersFn, rotated_query, k, ef_search);
}

std::vector<std::pair<float, PID>> search_knn_sorted_generic(
    HierarchicalNSW& index, const float* rotated_query, size_t k, size_t ef_search
) {
    return HnswSearchAccess::search(
        index, search_layers_generic, rotated_query, k, ef_search
    );
}

#if defined(__x86_64__) || defined(_M_X64)
std::vector<std::pair<float, PID>> search_knn_sorted_avx2(
    HierarchicalNSW& index, const float* rotated_query, size_t k, size_t ef_search
) {
    return HnswSearchAccess::search(index, search_layers_avx2, rotated_query, k, ef_search);
}

std::vector<std::pair<float, PID>> search_knn_sorted_avx512_core(
    HierarchicalNSW& index, const float* rotated_query, size_t k, size_t ef_search
) {
    return HnswSearchAccess::search(
        index, search_layers_avx512_core, rotated_query, k, ef_search
    );
}

std::vector<std::pair<float, PID>> search_knn_sorted_avx512_popcnt(
    HierarchicalNSW& index, const float* rotated_query, size_t k, size_t ef_search
) {
    return HnswSearchAccess::search(
        index, search_layers_avx512_popcnt, rotated_query, k, ef_search
    );
}
#else
std::vector<std::pair<float, PID>> search_knn_sorted_neon(
    HierarchicalNSW& index, const float* rotated_query, size_t k, size_t ef_search
) {
    return HnswSearchAccess::search(index, search_layers_neon, rotated_query, k, ef_search);
}
#endif
}  // namespace rabitqlib::hnsw::detail
