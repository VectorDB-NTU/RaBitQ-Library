#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
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
    std::vector<float>& q_to_centroids,
    SplitSingleQuery<float>& query_wrapper,
    PID candidate_id,
    HierarchicalNSW::EstimateRecord& res
) {
    if (metric_type_ == METRIC_IP) {
        float norm = q_to_centroids[get_clusterid_by_internalid(candidate_id)];
        float error =
            q_to_centroids[get_clusterid_by_internalid(candidate_id) + num_clusters_];
        res = estimate_split_single_direct<Kernel>(
            get_bindata_by_internalid(candidate_id),
            query_wrapper,
            padded_dim_,
            -norm,
            error
        );
    } else {
        // L2 distance
        float norm = q_to_centroids[get_clusterid_by_internalid(candidate_id)];
        res = estimate_split_single_direct<Kernel>(
            get_bindata_by_internalid(candidate_id),
            query_wrapper,
            padded_dim_,
            norm * norm,
            norm
        );
    }
}

template <class Kernel>
inline void HierarchicalNSW::get_full_est_direct(
    std::vector<float>& q_to_centroids,
    SplitSingleQuery<float>& query_wrapper,
    PID candidate_id,
    HierarchicalNSW::EstimateRecord& res
) const {
    if (metric_type_ == METRIC_IP) {
        float norm = q_to_centroids[get_clusterid_by_internalid(candidate_id)];
        float error =
            q_to_centroids[get_clusterid_by_internalid(candidate_id) + num_clusters_];
        res = estimate_split_single_full_direct<Kernel>(
            get_bindata_by_internalid(candidate_id),
            get_exdata_by_internalid(candidate_id),
            ip_func_,
            query_wrapper,
            padded_dim_,
            ex_bits_,
            -norm,
            error
        );
    } else {
        // L2 distance
        float norm = q_to_centroids[get_clusterid_by_internalid(candidate_id)];
        res = estimate_split_single_full_direct<Kernel>(
            get_bindata_by_internalid(candidate_id),
            get_exdata_by_internalid(candidate_id),
            ip_func_,
            query_wrapper,
            padded_dim_,
            ex_bits_,
            norm * norm,
            norm
        );
    }
}

template <class Kernel>
inline std::vector<std::pair<float, PID>> HierarchicalNSW::search_knn_sorted_direct(
    const float* rotated_query, size_t k, size_t ef
) {
    std::vector<std::pair<float, PID>> result;
    if (cur_element_count_ == 0) {
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
                reinterpret_cast<float*>(centroids_memory_.get()) + (i * padded_dim_),
                padded_dim_
            ));
        }
    } else if (metric_type_ == METRIC_IP) {
        q_to_centroids.resize(2 * num_clusters_);
        // first half as g_add, second half as g_error
        for (size_t i = 0; i < num_clusters_; i++) {
            q_to_centroids[i] = dot_product(
                rotated_query,
                reinterpret_cast<float*>(centroids_memory_.get()) + (i * padded_dim_),
                padded_dim_
            );
            q_to_centroids[i + num_clusters_] = std::sqrt(euclidean_sqr(
                rotated_query,
                reinterpret_cast<float*>(centroids_memory_.get()) + (i * padded_dim_),
                padded_dim_
            ));
        }
    }

    PID current_id = entry_point_id_;
    EstimateRecord curest;

    get_bin_est_direct<Kernel>(q_to_centroids, query_wrapper, current_id, curest);

    for (int level = max_level_; level > 0; level--) {
        bool changed = true;
        while (changed) {
            changed = false;
            unsigned int* data;

            data = static_cast<unsigned int*>(get_linklist(current_id, level));
            int size = get_list_count(data);

            PID* datal = static_cast<PID*>(data + 1);
            for (int i = 0; i < size; i++) {
                PID cand = datal[i];
                if (cand > max_elements_) {
                    throw std::runtime_error("cand error");
                }

                EstimateRecord candest;
                get_bin_est_direct<Kernel>(q_to_centroids, query_wrapper, cand, candest);

                if (candest.distance < curest.distance) {
                    curest = candest;
                    current_id = cand;
                    changed = true;
                }
            }
        }
    }

    BoundedKNN nearest_candidates(k);
    search_base_layer_direct<Kernel>(
        current_id,
        std::max(ef, k),
        k,
        query_wrapper,
        q_to_centroids,
        rotated_query,
        nearest_candidates
    );
    result.reserve(nearest_candidates.size());
    for (const auto& candidate : nearest_candidates.candidates()) {
        result.emplace_back(candidate.record.est_dist, get_external_label(candidate.id));
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
    PID ep_id,
    size_t ef,
    size_t k,
    SplitSingleQuery<float>& query_wrapper,
    std::vector<float>& q_to_centroids,
    [[maybe_unused]] const float* query,
    BoundedKNN& nearest_candidates
) {
    auto vl = visited_list_pool_->get_free_vislist();

    // Use our bounded priority queue instead of the maxheap.
    buffer::SearchBuffer<float> candidate_set(ef);

    float distk = 1e10;

    EstimateRecord start_estimate_record;
    if (ex_bits_ > 0) {
        get_full_est_direct<Kernel>(
            q_to_centroids, query_wrapper, ep_id, start_estimate_record
        );
    } else {
        get_bin_est_direct<Kernel>(
            q_to_centroids, query_wrapper, ep_id, start_estimate_record
        );
    }
    float est_dist = start_estimate_record.distance;
    float low_dist = start_estimate_record.lower_bound;

    // Insert initial candidate. A removed point is still explored, never returned.
    if (!is_removed(ep_id)) {
        nearest_candidates.insert({ResultRecord(est_dist, low_dist), ep_id});
        distk = est_dist;
    }
    candidate_set.insert(ep_id, est_dist);

    vl->set(ep_id);

    const size_t prefetch_size = (((padded_dim_ / 8) + 63) / 64) + 1;
    const size_t prefetch_lookahead = 4;  // Number of neighbors to prefetch in advance.

    while (candidate_set.has_next()) {
        // Step 1 - get the next node to explore.
        PID current_node_id = candidate_set.pop();
        int* data = (int*)get_linklist0(current_node_id);
        size_t size = get_list_count((PID*)data);

        for (size_t p = 0; p < prefetch_lookahead; ++p) {
            rabitqlib::memory::mem_prefetch_l1(
                get_bindata_by_internalid(*(data + 1 + p)), prefetch_size
            );
        }
        // Iterate over neighbors. (List starts at index 1.)
        for (size_t j = 1; j <= size; j++) {
            int candidate_id = *(data + j);

            if (j + prefetch_lookahead <= size) {
                rabitqlib::memory::mem_prefetch_l1(
                    get_bindata_by_internalid(*(data + j + prefetch_lookahead)),
                    prefetch_size
                );
            }

            if (vl->get(candidate_id)) {
                continue;
            }
            vl->set(candidate_id);

            EstimateRecord candest;
            get_bin_est_direct<Kernel>(
                q_to_centroids, query_wrapper, candidate_id, candest
            );

            // A removed neighbor keeps its binary estimate and stays in candidate_set.
            bool update_results =
                !is_removed(static_cast<PID>(candidate_id)) &&
                (nearest_candidates.size() < k || candest.lower_bound < distk);

            if (update_results) {
                // Compute the full estimate if promising.
                if (ex_bits_ > 0) {
                    get_full_est_direct<Kernel>(
                        q_to_centroids, query_wrapper, candidate_id, candest
                    );
                }
                Candidate cand{
                    ResultRecord(candest.distance, candest.lower_bound),
                    static_cast<PID>(candidate_id)};
                nearest_candidates.insert(cand);
                distk = nearest_candidates.worst().record.est_dist;
            }

            if (!candidate_set.is_full(candest.distance)) {
                candidate_set.insert(candidate_id, candest.distance);
            }

            rabitqlib::memory::mem_prefetch_l2(
                (char*)get_linklist0(candidate_set.next_id()), 2
            );
        }
    }

    visited_list_pool_->release_vis_list(std::move(vl));
}

}  // namespace rabitqlib::hnsw
