#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <utility>
#include <vector>

#include "rabitqlib/core/estimator.hpp"
#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/core/query.hpp"
#include "rabitqlib/index/detail/buffer.hpp"
#include "rabitqlib/index/detail/hnsw_dispatch.hpp"
#include "rabitqlib/index/detail/visited_set.hpp"
#include "rabitqlib/index/hnsw/hnsw.hpp"
#include "rabitqlib/index/types.hpp"
#include "rabitqlib/utils/memory.hpp"

namespace rabitqlib::hnsw {

namespace detail {
// Grants the dispatch and per-ISA translation units access to the search members.
struct HnswSearchAccess {
    static std::vector<std::pair<float, PID>> search(
        const HierarchicalNSW& index,
        SearchLayersFn search_layers,
        const float* rotated_query,
        size_t k,
        size_t ef_search
    ) {
        return index.search_knn_sorted_with(search_layers, rotated_query, k, ef_search);
    }

    template <class Kernel>
    static size_t search_layers(
        const HierarchicalNSW& index,
        const SplitSingleQuery<float>& query_wrapper,
        const float* q_to_centroids,
        VisitedSet& visited,
        buffer::SearchBuffer<float>& search_pool,
        size_t k,
        AnnCandidate<float>* nearest
    ) {
        return index.search_layers_direct<Kernel>(
            query_wrapper, q_to_centroids, visited, search_pool, k, nearest
        );
    }
};
}  // namespace detail

template <class Kernel>
inline void HierarchicalNSW::estimate_bin_direct(
    const float* q_to_centroids,
    const SplitSingleQuery<float>& query_wrapper,
    PID candidate_id,
    DistanceEstimate& estimate
) const {
    if (metric_type_ == METRIC_IP) {
        const float query_centroid_ip = q_to_centroids[cluster_id_of(candidate_id)];
        const float residual_norm =
            q_to_centroids[cluster_id_of(candidate_id) + num_clusters_];
        estimate = estimate_split_single_direct<Kernel>(
            bin_data(candidate_id),
            query_wrapper,
            padded_dim_,
            -query_centroid_ip,
            residual_norm
        );
    } else {
        // L2 distance
        const float residual_norm = q_to_centroids[cluster_id_of(candidate_id)];
        estimate = estimate_split_single_direct<Kernel>(
            bin_data(candidate_id),
            query_wrapper,
            padded_dim_,
            residual_norm * residual_norm,
            residual_norm
        );
    }
}

template <class Kernel>
inline void HierarchicalNSW::estimate_full_direct(
    const float* q_to_centroids,
    const SplitSingleQuery<float>& query_wrapper,
    PID candidate_id,
    DistanceEstimate& estimate
) const {
    if (metric_type_ == METRIC_IP) {
        const float query_centroid_ip = q_to_centroids[cluster_id_of(candidate_id)];
        const float residual_norm =
            q_to_centroids[cluster_id_of(candidate_id) + num_clusters_];
        estimate = estimate_split_single_full_direct<Kernel>(
            bin_data(candidate_id),
            ex_data(candidate_id),
            ip_func_,
            query_wrapper,
            padded_dim_,
            ex_bits_,
            -query_centroid_ip,
            residual_norm
        );
    } else {
        // L2 distance
        const float residual_norm = q_to_centroids[cluster_id_of(candidate_id)];
        estimate = estimate_split_single_full_direct<Kernel>(
            bin_data(candidate_id),
            ex_data(candidate_id),
            ip_func_,
            query_wrapper,
            padded_dim_,
            ex_bits_,
            residual_norm * residual_norm,
            residual_norm
        );
    }
}

template <class Kernel>
inline size_t HierarchicalNSW::search_layers_direct(
    const SplitSingleQuery<float>& query_wrapper,
    const float* q_to_centroids,
    VisitedSet& visited,
    buffer::SearchBuffer<float>& search_pool,
    size_t k,
    AnnCandidate<float>* nearest
) const {
    PID current_id = entry_point_id_;
    DistanceEstimate current_estimate;

    estimate_bin_direct<Kernel>(
        q_to_centroids, query_wrapper, current_id, current_estimate
    );

    for (int level = max_level_; level > 0; --level) {
        bool changed = true;
        while (changed) {
            changed = false;
            const PID* links = upper_layer_links(current_id, level);
            const size_t num_neighbors = link_count(links);

            const PID* neighbors = links + 1;
            for (size_t i = 0; i < num_neighbors; ++i) {
                PID candidate_id = neighbors[i];
                assert(candidate_id < num_stored_points_);

                DistanceEstimate candidate_estimate;
                estimate_bin_direct<Kernel>(
                    q_to_centroids, query_wrapper, candidate_id, candidate_estimate
                );

                if (candidate_estimate.distance < current_estimate.distance) {
                    current_estimate = candidate_estimate;
                    current_id = candidate_id;
                    changed = true;
                }
            }
        }
    }

    // Bounded priority queue implemented as a sorted vector. A local class has no
    // linkage, so the vector code it instantiates is never shared between ISA builds.
    class BoundedKNN {
       public:
        struct Entry {
            PID id;
            float distance;
        };

        explicit BoundedKNN(size_t capacity) : capacity_(capacity) {}

        // Insert a candidate in sorted order (ascending by distance), after equal ones.
        void insert(PID id, float distance) {
            auto it = std::upper_bound(
                queue_.begin(),
                queue_.end(),
                distance,
                [](float value, const Entry& entry) { return value < entry.distance; }
            );
            queue_.insert(it, Entry{id, distance});
            // If we exceed capacity, drop the worst candidate (largest distance).
            if (queue_.size() > capacity_) {
                queue_.pop_back();
            }
        }

        // Returns the worst (largest distance) candidate.
        [[nodiscard]] const Entry& worst() const { return queue_.back(); }

        [[nodiscard]] size_t size() const { return queue_.size(); }

        [[nodiscard]] const std::vector<Entry>& candidates() const { return queue_; }

       private:
        size_t capacity_;
        // Sorted in ascending order by distance so that the worst is at the back.
        std::vector<Entry> queue_;
    };

    BoundedKNN nearest_candidates(k);
    search_base_layer_direct<Kernel>(
        current_id,
        k,
        query_wrapper,
        q_to_centroids,
        visited,
        search_pool,
        nearest_candidates
    );
    size_t count = 0;
    for (const auto& candidate : nearest_candidates.candidates()) {
        nearest[count++] = AnnCandidate<float>(candidate.id, candidate.distance);
    }
    return count;
}

template <class Kernel, class Results>
inline void HierarchicalNSW::search_base_layer_direct(
    PID entry_point_id,
    size_t k,
    const SplitSingleQuery<float>& query_wrapper,
    const float* q_to_centroids,
    VisitedSet& visited,
    buffer::SearchBuffer<float>& search_pool,
    Results& nearest_candidates
) const {
    float dist_k = 1e10F;

    DistanceEstimate start_estimate;
    if (ex_bits_ > 0) {
        estimate_full_direct<Kernel>(
            q_to_centroids, query_wrapper, entry_point_id, start_estimate
        );
    } else {
        estimate_bin_direct<Kernel>(
            q_to_centroids, query_wrapper, entry_point_id, start_estimate
        );
    }
    float estimated_distance = start_estimate.distance;

    // Insert initial candidate. A removed point is still explored, never returned.
    if (!is_removed(entry_point_id)) {
        nearest_candidates.insert(entry_point_id, estimated_distance);
        dist_k = estimated_distance;
    }
    search_pool.insert(entry_point_id, estimated_distance);

    visited.set(entry_point_id);

    const size_t prefetch_size = (((padded_dim_ / 8) + 63) / 64) + 1;
    constexpr size_t kPrefetchLookahead = 4;  // Number of neighbors to prefetch in advance.

    while (search_pool.has_next()) {
        // Step 1 - get the next node to explore.
        PID current_node_id = search_pool.pop();
        const PID* links = level0_links(current_node_id);
        const size_t num_neighbors = link_count(links);

        for (size_t p = 0; p < kPrefetchLookahead; ++p) {
            memory::mem_prefetch_l1(bin_data(*(links + 1 + p)), prefetch_size);
        }
        // Iterate over neighbors. (List starts at index 1.)
        for (size_t j = 1; j <= num_neighbors; ++j) {
            const PID candidate_id = *(links + j);

            if (j + kPrefetchLookahead <= num_neighbors) {
                memory::mem_prefetch_l1(
                    bin_data(*(links + j + kPrefetchLookahead)), prefetch_size
                );
            }

            if (visited.get(candidate_id)) {
                continue;
            }
            visited.set(candidate_id);

            DistanceEstimate candidate_estimate;
            estimate_bin_direct<Kernel>(
                q_to_centroids, query_wrapper, candidate_id, candidate_estimate
            );

            // A removed neighbor keeps its binary estimate and stays in search_pool.
            bool update_results =
                !is_removed(candidate_id) &&
                (nearest_candidates.size() < k || candidate_estimate.lower_bound < dist_k);

            if (update_results) {
                // Compute the full estimate if promising.
                if (ex_bits_ > 0) {
                    estimate_full_direct<Kernel>(
                        q_to_centroids, query_wrapper, candidate_id, candidate_estimate
                    );
                }
                nearest_candidates.insert(candidate_id, candidate_estimate.distance);
                dist_k = nearest_candidates.worst().distance;
            }

            if (!search_pool.is_full(candidate_estimate.distance)) {
                search_pool.insert(candidate_id, candidate_estimate.distance);
            }

            memory::mem_prefetch_l2(
                reinterpret_cast<const char*>(level0_links(search_pool.next_id())), 2
            );
        }
    }
}

}  // namespace rabitqlib::hnsw
