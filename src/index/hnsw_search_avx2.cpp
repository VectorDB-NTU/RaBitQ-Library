#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "hnsw_search_avx2_kernels.hpp"
#include "hnsw_search_impl.hpp"  // IWYU pragma: keep (backend template definitions)
#include "rabitqlib/index/hnsw/hnsw.hpp"
#include "rabitqlib/index/types.hpp"

namespace rabitqlib::hnsw::detail {

struct HnswAvx2Kernel {
    static inline float warmup_ip_x0_q_512(
        const uint64_t* data,
        const uint64_t* query,
        float delta,
        float vl,
        size_t padded_dim,
        size_t b_query
    ) {
        return hnsw_warmup_ip_x0_q_512_avx2(data, query, delta, vl, padded_dim, b_query);
    }

    static inline float mask_ip_x0_q(
        const float* query, const uint64_t* data, size_t padded_dim
    ) {
        return hnsw_mask_ip_x0_q_avx2(query, data, padded_dim);
    }
};

std::vector<std::pair<float, PID>> search_knn_sorted_avx2(
    HierarchicalNSW& index, const float* rotated_query, size_t k, size_t ef_search
) {
    return index.search_knn_sorted_direct<HnswAvx2Kernel>(rotated_query, k, ef_search);
}

}  // namespace rabitqlib::hnsw::detail
