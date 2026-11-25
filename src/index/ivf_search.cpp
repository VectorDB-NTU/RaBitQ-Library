#include "ivf_search.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>

#include "rabitqlib/clustering/detail/kmeans.hpp"
#include "rabitqlib/core/detail/simd/matrix_dispatch.hpp"
#include "rabitqlib/index/detail/buffer.hpp"
#include "rabitqlib/index/detail/candidate.hpp"
#include "rabitqlib/index/types.hpp"
#include "rabitqlib/utils/space.hpp"

namespace rabitqlib::ivf::detail {
// Keep the scalar top-k loop out of the large inlined IVF search body. This avoids
// register spills in wheel builds without changing candidate order or tie handling.
void insert_candidates(
    buffer::SearchBuffer<float>& result_pool,
    const PID* ids,
    const float* distances,
    size_t count
) {
    for (size_t i = 0; i < count; ++i) {
        result_pool.insert(ids[i], distances[i]);
    }
}

void exact_assign_topn(
    const float* x,
    const float* centroids,
    size_t n,
    size_t d,
    size_t k,
    size_t nprobe,
    bool spherical,
    const double* centroid_norms,
    AnnCandidate<float>* candidates,
    TopNAssignmentScratch& scratch
) {
    assert(nprobe > 0 && nprobe <= k);
    constexpr size_t kCentroidBlock = 256;
    const index_detail::CandidateDistanceIdLess less;
    scratch.products.resize(n * std::min(k, kCentroidBlock));
    scratch.point_norms.resize(n);
    const auto distance = [d, spherical](const float* vector, const float* centroid) {
        return spherical ? dot_product_dis(vector, centroid, d)
                         : std::sqrt(euclidean_sqr(vector, centroid, d));
    };
    const auto consider = [&](AnnCandidate<float>* heap, size_t cluster, float dist) {
        // Centroid IDs increase, so an equal returned distance cannot improve the heap.
        if (dist < heap[0].distance) {
            std::pop_heap(heap, heap + nprobe, less);
            heap[nprobe - 1] = AnnCandidate<float>(static_cast<PID>(cluster), dist);
            std::push_heap(heap, heap + nprobe, less);
        }
    };
    for (size_t row = 0; row < n; ++row) {
        const float* vector = x + row * d;
        auto* heap = candidates + row * nprobe;
        scratch.point_norms[row] = clustering::detail::l2norm_sqr_f64(vector, d);
        for (size_t cluster = 0; cluster < nprobe; ++cluster) {
            heap[cluster].id = static_cast<PID>(cluster);
            heap[cluster].distance = distance(vector, centroids + cluster * d);
        }
        std::make_heap(heap, heap + nprobe, less);
    }

    // For validated d <= 65536, this conservatively covers the GEMM dot product,
    // float distance accumulation, and sqrt rounding when the cutoff is squared.
    // Both metrics need an underflow allowance. Only IP subtracts the dot from
    // one and can incur absolute rounding error near one even for tiny norms.
    const double error_scale = 8.0 * static_cast<double>(d) *
                               static_cast<double>(std::numeric_limits<float>::epsilon());
    const double error_floor =
        8.0 * static_cast<double>(d) * std::numeric_limits<float>::min() +
        (spherical ? 8.0 * std::numeric_limits<float>::epsilon() : 0.0);
    for (size_t first = nprobe; first < k; first += kCentroidBlock) {
        const size_t cols = std::min(kCentroidBlock, k - first);
        simd::matrix_product_transposed(
            x, centroids + first * d, scratch.products.data(), n, d, cols
        );
        bool use_direct_l2 = !spherical;
        for (size_t row = 0; row < n; ++row) {
            const float* vector = x + row * d;
            auto* heap = candidates + row * nprobe;
            for (size_t col = 0; col < cols; ++col) {
                const size_t cluster = first + col;
                const double norm_sum = scratch.point_norms[row] + centroid_norms[cluster];
                const double dot = scratch.products[row * cols + col];
                const double estimate = spherical ? 1.0 - dot : norm_sum - 2.0 * dot;
                const double worst = heap[0].distance;
                const double cutoff = spherical ? worst : worst * worst;
                if (estimate > cutoff + error_scale * norm_sum + error_floor) {
                    use_direct_l2 = false;
                    continue;
                }
                consider(heap, cluster, distance(vector, centroids + cluster * d));
            }
            // Switch only when the tile rejected nothing and every query's own
            // rounding allowance is already larger than its squared cutoff.
            const double worst = heap[0].distance;
            use_direct_l2 =
                use_direct_l2 && error_scale * scratch.point_norms[row] > worst * worst;
        }
        if (use_direct_l2) {
            // With a large common offset, GEMM and exact scoring would visit the
            // same pairs. Keep the current heaps and score the remaining centroids once.
            for (size_t row = 0; row < n; ++row) {
                auto* heap = candidates + row * nprobe;
                const float* vector = x + row * d;
                for (size_t cluster = first + cols; cluster < k; ++cluster) {
                    const float squared = euclidean_sqr(vector, centroids + cluster * d, d);
                    const double worst = heap[0].distance;
                    if (squared < worst * worst) {
                        consider(heap, cluster, std::sqrt(squared));
                    }
                }
            }
            break;
        }
    }
    for (size_t row = 0; row < n; ++row) {
        auto* heap = candidates + row * nprobe;
        std::sort_heap(heap, heap + nprobe, less);
    }
}
}  // namespace rabitqlib::ivf::detail
