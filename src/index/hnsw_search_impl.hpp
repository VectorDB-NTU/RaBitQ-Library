#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "rabitqlib/core/estimator.hpp"
#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/core/query.hpp"
#include "rabitqlib/index/detail/buffer.hpp"
#include "rabitqlib/index/detail/visited_pool.hpp"
#include "rabitqlib/index/hnsw/hnsw.hpp"
#include "rabitqlib/index/types.hpp"
#include "rabitqlib/utils/memory.hpp"
#include "rabitqlib/utils/space.hpp"

namespace rabitqlib::hnsw {

template <class Kernel>
inline void HierarchicalNSW::get_bin_est_direct(
    const std::vector<float>& q_to_centroids,
    const SplitSingleQuery<float>& query_wrapper,
    PID candidate_id,
    HierarchicalNSW::EstimateRecord& estimate
) const {
    if (metric_type_ == METRIC_IP) {
        const float query_centroid_ip =
            q_to_centroids[get_clusterid_by_internalid(candidate_id)];
        const float residual_norm =
            q_to_centroids[get_clusterid_by_internalid(candidate_id) + num_clusters_];
        estimate = estimate_split_single_direct<Kernel>(
            get_bindata_by_internalid(candidate_id),
            query_wrapper,
            padded_dim_,
            -query_centroid_ip,
            residual_norm
        );
    } else {
        // L2 distance
        const float residual_norm =
            q_to_centroids[get_clusterid_by_internalid(candidate_id)];
        estimate = estimate_split_single_direct<Kernel>(
            get_bindata_by_internalid(candidate_id),
            query_wrapper,
            padded_dim_,
            residual_norm * residual_norm,
            residual_norm
        );
    }
}

template <class Kernel>
inline void HierarchicalNSW::get_full_est_direct(
    const std::vector<float>& q_to_centroids,
    const SplitSingleQuery<float>& query_wrapper,
    PID candidate_id,
    HierarchicalNSW::EstimateRecord& estimate
) const {
    if (metric_type_ == METRIC_IP) {
        const float query_centroid_ip =
            q_to_centroids[get_clusterid_by_internalid(candidate_id)];
        const float residual_norm =
            q_to_centroids[get_clusterid_by_internalid(candidate_id) + num_clusters_];
        estimate = estimate_split_single_full_direct<Kernel>(
            get_bindata_by_internalid(candidate_id),
            get_exdata_by_internalid(candidate_id),
            ip_func_,
            query_wrapper,
            padded_dim_,
            ex_bits_,
            -query_centroid_ip,
            residual_norm
        );
    } else {
        // L2 distance
        const float residual_norm =
            q_to_centroids[get_clusterid_by_internalid(candidate_id)];
        estimate = estimate_split_single_full_direct<Kernel>(
            get_bindata_by_internalid(candidate_id),
            get_exdata_by_internalid(candidate_id),
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
inline std::vector<std::pair<float, PID>> HierarchicalNSW::search_knn_sorted_direct(
    const float* rotated_query, size_t k, size_t ef_search
) const {
    std::vector<std::pair<float, PID>> result;
    if (num_stored_points_ == 0) {
        return result;
    }

    SplitSingleQuery<float> query_wrapper(
        rotated_query, padded_dim_, ex_bits_, query_config_, metric_type_
    );

    // Preprocess - get the distance from query to all centroids
    std::vector<float> q_to_centroids(num_clusters_);

    if (metric_type_ == METRIC_L2) {
        for (size_t i = 0; i < num_clusters_; i++) {
            q_to_centroids[i] = std::sqrt(raw_dist_func_(
                rotated_query,
                reinterpret_cast<const float*>(centroids_memory_.get()) + (i * padded_dim_),
                padded_dim_
            ));
        }
    } else if (metric_type_ == METRIC_IP) {
        q_to_centroids.resize(2 * num_clusters_);
        // First half stores dot(query, centroid); second half stores ||query - centroid||.
        for (size_t i = 0; i < num_clusters_; i++) {
            q_to_centroids[i] = dot_product(
                rotated_query,
                reinterpret_cast<const float*>(centroids_memory_.get()) + (i * padded_dim_),
                padded_dim_
            );
            q_to_centroids[i + num_clusters_] = std::sqrt(euclidean_sqr(
                rotated_query,
                reinterpret_cast<const float*>(centroids_memory_.get()) + (i * padded_dim_),
                padded_dim_
            ));
        }
    }

    PID current_id = entry_point_id_;
    EstimateRecord current_estimate;

    get_bin_est_direct<Kernel>(q_to_centroids, query_wrapper, current_id, current_estimate);

    for (int level = max_level_; level > 0; level--) {
        bool changed = true;
        while (changed) {
            changed = false;
            const PID* data = get_linklist(current_id, level);
            const size_t size = get_list_count(data);

            const PID* datal = data + 1;
            for (size_t i = 0; i < size; i++) {
                PID cand = datal[i];
                if (cand > max_elements_) {
                    throw std::runtime_error(
                        "HNSW search upper-layer traversal: invalid candidate ID " +
                        std::to_string(cand) + " at level " + std::to_string(level) +
                        " (capacity=" + std::to_string(max_elements_) + ")"
                    );
                }

                EstimateRecord candidate_estimate;
                get_bin_est_direct<Kernel>(
                    q_to_centroids, query_wrapper, cand, candidate_estimate
                );

                if (candidate_estimate.distance < current_estimate.distance) {
                    current_estimate = candidate_estimate;
                    current_id = cand;
                    changed = true;
                }
            }
        }
    }

    BoundedKNN nearest_candidates(k);
    search_base_layer_direct<Kernel>(
        current_id,
        std::max(ef_search, k),
        k,
        query_wrapper,
        q_to_centroids,
        nearest_candidates
    );
    result.reserve(nearest_candidates.size());
    for (const auto& candidate : nearest_candidates.candidates()) {
        result.emplace_back(candidate.est_dist, get_external_label(candidate.id));
    }
    // Candidate selection uses distance only. Preserve that selection, then match
    // the former heap's (distance, external label) ordering within each tie group.
    for (auto first = result.begin(); first != result.end();) {
        auto last = first + 1;
        while (last != result.end() && last->first == first->first) {
            ++last;
        }
        std::sort(first, last);
        first = last;
    }
    return result;
}

template <class Kernel>
inline void HierarchicalNSW::search_base_layer_direct(
    PID entry_point_id,
    size_t ef_search,
    size_t k,
    const SplitSingleQuery<float>& query_wrapper,
    const std::vector<float>& q_to_centroids,
    BoundedKNN& nearest_candidates
) const {
    auto visited = visited_list_pool_->get_free_vislist();

    // Use our bounded priority queue instead of the maxheap.
    buffer::SearchBuffer<float> search_pool(ef_search);

    float dist_k = 1e10;

    EstimateRecord start_estimate_record;
    if (ex_bits_ > 0) {
        get_full_est_direct<Kernel>(
            q_to_centroids, query_wrapper, entry_point_id, start_estimate_record
        );
    } else {
        get_bin_est_direct<Kernel>(
            q_to_centroids, query_wrapper, entry_point_id, start_estimate_record
        );
    }
    float est_dist = start_estimate_record.distance;

    // Insert initial candidate. A removed point is still explored, never returned.
    if (!is_removed(entry_point_id)) {
        nearest_candidates.insert({est_dist, entry_point_id});
        dist_k = est_dist;
    }
    search_pool.insert(entry_point_id, est_dist);

    visited->set(entry_point_id);

    const size_t prefetch_size = (((padded_dim_ / 8) + 63) / 64) + 1;
    const size_t prefetch_lookahead = 4;  // Number of neighbors to prefetch in advance.

    while (search_pool.has_next()) {
        // Step 1 - get the next node to explore.
        PID current_node_id = search_pool.pop();
        const PID* data = get_linklist0(current_node_id);
        const size_t size = get_list_count(data);

        for (size_t p = 0; p < prefetch_lookahead; ++p) {
            rabitqlib::memory::mem_prefetch_l1(
                get_bindata_by_internalid(*(data + 1 + p)), prefetch_size
            );
        }
        // Iterate over neighbors. (List starts at index 1.)
        for (size_t j = 1; j <= size; j++) {
            const PID candidate_id = *(data + j);

            if (j + prefetch_lookahead <= size) {
                rabitqlib::memory::mem_prefetch_l1(
                    get_bindata_by_internalid(*(data + j + prefetch_lookahead)),
                    prefetch_size
                );
            }

            if (visited->get(candidate_id)) {
                continue;
            }
            visited->set(candidate_id);

            EstimateRecord candidate_estimate;
            get_bin_est_direct<Kernel>(
                q_to_centroids, query_wrapper, candidate_id, candidate_estimate
            );

            // A removed neighbor keeps its binary estimate and stays in search_pool.
            bool update_results =
                !is_removed(candidate_id) &&
                (nearest_candidates.size() < k || candidate_estimate.lower_bound < dist_k);

            if (update_results) {
                // Compute the full estimate if promising.
                if (ex_bits_ > 0) {
                    get_full_est_direct<Kernel>(
                        q_to_centroids, query_wrapper, candidate_id, candidate_estimate
                    );
                }
                Candidate cand{candidate_estimate.distance, candidate_id};
                nearest_candidates.insert(cand);
                dist_k = nearest_candidates.worst().est_dist;
            }

            if (!search_pool.is_full(candidate_estimate.distance)) {
                search_pool.insert(candidate_id, candidate_estimate.distance);
            }

            rabitqlib::memory::mem_prefetch_l2(
                reinterpret_cast<const char*>(get_linklist0(search_pool.next_id())), 2
            );
        }
    }

    visited_list_pool_->release_vis_list(std::move(visited));
}

}  // namespace rabitqlib::hnsw
