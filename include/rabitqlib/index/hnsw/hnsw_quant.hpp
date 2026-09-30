// The construct path with get_quant_dist in place of get_data_dist: rawDataPtr_
// dangles once construct returns, so these link using the stored codes.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/index/hnsw/hnsw.hpp"
#include "rabitqlib/quantization/data_layout.hpp"
#include "rabitqlib/quantization/pack_excode.hpp"
#include "rabitqlib/quantization/rabitq.hpp"

namespace rabitqlib::hnsw {

namespace detail {
// The reconstructed query of get_quant_dist, reused while the query does not
// move. Cleared at the start of every insert so it cannot outlive its index.
struct QuantQuery {
    PID id = kPidMax;
    std::vector<float> rotated;
    std::optional<SplitSingleQuery<float>> prepared;

    void clear() {
        id = kPidMax;
        prepared.reset();
    }
};

inline QuantQuery& quant_query() {
    thread_local QuantQuery query;
    return query;
}
}  // namespace detail

// Code per dimension is (sign << ex_bits) + ex, up to nine bits, scaled by
// f_rescale_ex.
inline void HierarchicalNSW::reconstruct_rotated(PID id, float* out) const {
    thread_local std::vector<uint16_t> combined;
    thread_local std::vector<uint8_t> ex_codes;
    combined.resize(padded_dim_);

    const ConstBinDataMap<float> bin(get_bindata_by_internalid(id), padded_dim_);
    const uint8_t* sign_code = bin.bin_code();
    for (size_t word = 0; word * 64 < padded_dim_; ++word) {
        uint64_t bits = 0;
        std::memcpy(&bits, sign_code + (word * sizeof(uint64_t)), sizeof(bits));
        const size_t base = word * 64;
        const size_t span = std::min<size_t>(64, padded_dim_ - base);
        for (size_t j = 0; j < span; ++j) {
            combined[base + j] = static_cast<uint16_t>((bits >> (63 - j)) & 1ULL);
        }
    }

    const auto* centroid = reinterpret_cast<const float*>(centroids_memory_) +
                           (get_clusterid_by_internalid(id) * padded_dim_);

    if (ex_bits_ == 0) {
        quant::reconstruct_full_vec<float, uint16_t>(
            combined.data(), centroid, padded_dim_, 1, bin.f_rescale(), out, metric_type_
        );
        return;
    }

    ex_codes.resize(padded_dim_);
    const ConstExDataMap<float> ex(get_exdata_by_internalid(id), padded_dim_, ex_bits_);
    quant::rabitq_impl::ex_bits::unpacking_rabitqplus_code(
        ex.ex_code(), ex_codes.data(), padded_dim_, ex_bits_
    );
    for (size_t d = 0; d < padded_dim_; ++d) {
        combined[d] = static_cast<uint16_t>((combined[d] << ex_bits_) + ex_codes[d]);
    }

    quant::reconstruct_full_vec<float, uint16_t>(
        combined.data(),
        centroid,
        padded_dim_,
        ex_bits_ + 1,
        ex.f_rescale_ex(),
        out,
        metric_type_
    );
}

// Only `query` is reconstructed; `target` is scored from its codes, as QGBuilder
// does. Callers hold `query` fixed across a loop, so it is prepared once.
inline float HierarchicalNSW::get_quant_dist(PID target, PID query) const {
    detail::QuantQuery& cached = detail::quant_query();
    if (cached.id != query || !cached.prepared.has_value()) {
        cached.rotated.resize(padded_dim_);
        reconstruct_rotated(query, cached.rotated.data());
        cached.prepared.emplace(
            cached.rotated.data(), padded_dim_, ex_bits_, query_config_, metric_type_
        );
        cached.id = query;
    }

    const auto* centroid = reinterpret_cast<const float*>(centroids_memory_) +
                           (get_clusterid_by_internalid(target) * padded_dim_);
    const float dist_sqr = euclidean_sqr(cached.rotated.data(), centroid, padded_dim_);
    const float g_error = std::sqrt(dist_sqr);
    const float g_add = (metric_type_ == METRIC_IP)
                            ? -dot_product(cached.rotated.data(), centroid, padded_dim_)
                            : dist_sqr;

    float est_dist = 0;
    float low_dist = 0;
    float ip_x0_qr = 0;
    if (ex_bits_ == 0) {
        split_single_estdist(
            get_bindata_by_internalid(target),
            *cached.prepared,
            padded_dim_,
            ip_x0_qr,
            est_dist,
            low_dist,
            g_add,
            g_error
        );
    } else {
        split_single_fulldist(
            get_bindata_by_internalid(target),
            get_exdata_by_internalid(target),
            ip_func_,
            *cached.prepared,
            padded_dim_,
            ex_bits_,
            est_dist,
            low_dist,
            ip_x0_qr,
            g_add,
            g_error
        );
    }
    return est_dist;
}

inline maxheap<std::pair<float, PID>> HierarchicalNSW::search_base_layer_quant(
    PID ep_id, PID cur_c, int layer
) {
    VisitedSet* vl = visited_list_pool_->get_free_vislist();

    maxheap<std::pair<float, PID>> top_candidates;
    minheap<std::pair<float, PID>> candidate_set;

    float lower_bound = get_quant_dist(ep_id, cur_c);
    top_candidates.emplace(lower_bound, ep_id);
    candidate_set.emplace(lower_bound, ep_id);
    vl->set(ep_id);

    while (!candidate_set.empty()) {
        std::pair<float, PID> curr_el_pair = candidate_set.top();
        if (curr_el_pair.first > lower_bound && top_candidates.size() == ef_construction_) {
            break;
        }
        candidate_set.pop();

        PID cur_node_num = curr_el_pair.second;
        std::unique_lock<std::mutex> lock(link_list_locks_[cur_node_num]);

        PID* data =
            (layer == 0) ? get_linklist0(cur_node_num) : get_linklist(cur_node_num, layer);
        size_t size = get_list_count(data);
        auto* datal = data + 1;

        for (size_t j = 0; j < size; j++) {
            PID candidate_id = datal[j];
            if (candidate_id >= cur_element_count_) {
                throw std::runtime_error("cand error");
            }
            if (vl->get(candidate_id)) {
                continue;
            }
            vl->set(candidate_id);

            float dist1 = get_quant_dist(candidate_id, cur_c);
            if (top_candidates.size() < ef_construction_ || lower_bound > dist1) {
                candidate_set.emplace(dist1, candidate_id);
                top_candidates.emplace(dist1, candidate_id);
                if (top_candidates.size() > ef_construction_) {
                    top_candidates.pop();
                }
                if (!top_candidates.empty()) {
                    lower_bound = top_candidates.top().first;
                }
            }
        }
    }
    visited_list_pool_->release_vis_list(vl);
    return top_candidates;
}

inline void HierarchicalNSW::get_neighbors_by_heuristic2_quant(
    maxheap<std::pair<float, PID>>& top_candidates, size_t M
) {
    if (top_candidates.size() < M) {
        return;
    }

    minheap<std::pair<float, PID>> queue_closest;
    std::vector<std::pair<float, PID>> return_list;
    while (top_candidates.size() > 0) {
        queue_closest.emplace(top_candidates.top());
        top_candidates.pop();
    }

    while (queue_closest.size() > 0) {
        if (return_list.size() >= M) {
            break;
        }
        std::pair<float, PID> current_pair = queue_closest.top();
        float dist_to_query = current_pair.first;
        queue_closest.pop();
        bool good = true;

        for (std::pair<float, PID> second_pair : return_list) {
            float curdist = get_quant_dist(second_pair.second, current_pair.second);
            if (curdist < dist_to_query) {
                good = false;
                break;
            }
        }
        if (good) {
            return_list.push_back(current_pair);
        }
    }

    for (std::pair<float, PID> current_pair : return_list) {
        top_candidates.emplace(current_pair);
    }
}

inline PID HierarchicalNSW::mutually_connect_quant(
    PID cur_c, maxheap<std::pair<float, PID>>& top_candidates, int level
) {
    size_t max_m = level > 0 ? maxM_ : maxM0_;
    get_neighbors_by_heuristic2_quant(top_candidates, M_);
    if (top_candidates.size() > M_) {
        throw std::runtime_error(
            "Should be not be more than M_ candidates returned by the heuristic"
        );
    }

    std::vector<PID> selected_neighbors;
    selected_neighbors.reserve(M_);
    while (top_candidates.size() > 0) {
        selected_neighbors.push_back(top_candidates.top().second);
        top_candidates.pop();
    }
    if (selected_neighbors.empty()) {
        throw std::runtime_error("No neighbors selected for the new element");
    }
    PID next_closest_entry_point = selected_neighbors.back();

    {
        PID* ll_cur = (level == 0) ? get_linklist0(cur_c) : get_linklist(cur_c, level);
        if (*ll_cur > 0) {
            throw std::runtime_error(
                "The newly inserted element should have blank link list"
            );
        }

        set_list_count(ll_cur, selected_neighbors.size());
        auto* data = ll_cur + 1;
        for (size_t idx = 0; idx < selected_neighbors.size(); idx++) {
            if (data[idx] != 0) {
                throw std::runtime_error("Possible memory corruption");
            }
            if (level > element_levels_[selected_neighbors[idx]]) {
                throw std::runtime_error("Trying to make a link on a non-existent level");
            }
            data[idx] = selected_neighbors[idx];
        }
    }

    for (auto selected_neighbor : selected_neighbors) {
        std::unique_lock<std::mutex> lock(link_list_locks_[selected_neighbor]);

        PID* ll_other = (level == 0) ? get_linklist0(selected_neighbor)
                                     : get_linklist(selected_neighbor, level);
        size_t sz_link_list_other = get_list_count(ll_other);

        if (sz_link_list_other > max_m) {
            throw std::runtime_error("Bad value of sz_link_list_other");
        }
        if (selected_neighbor == cur_c) {
            throw std::runtime_error("Trying to connect an element to itself");
        }
        if (level > element_levels_[selected_neighbor]) {
            throw std::runtime_error("Trying to make a link on a non-existent level");
        }

        auto* data = ll_other + 1;

        bool is_cur_c_present = false;
        for (size_t j = 0; j < sz_link_list_other; j++) {
            if (data[j] == cur_c) {
                is_cur_c_present = true;
                break;
            }
        }

        if (!is_cur_c_present) {
            if (sz_link_list_other < max_m) {
                data[sz_link_list_other] = cur_c;
                set_list_count(ll_other, sz_link_list_other + 1);
            } else {
                float d_max = get_quant_dist(selected_neighbor, cur_c);
                maxheap<std::pair<float, PID>> candidates;
                candidates.emplace(d_max, cur_c);
                for (size_t j = 0; j < sz_link_list_other; j++) {
                    candidates.emplace(get_quant_dist(data[j], selected_neighbor), data[j]);
                }

                get_neighbors_by_heuristic2_quant(candidates, max_m);

                int indx = 0;
                while (candidates.size() > 0) {
                    data[indx] = candidates.top().second;
                    candidates.pop();
                    indx++;
                }
                set_list_count(ll_other, indx);
            }
        }
    }

    return next_closest_entry_point;
}

inline PID HierarchicalNSW::add_point_quant(
    const float* vec, PID cluster_id, const quant::RabitqConfig& config
) {
    detail::quant_query().clear();
    int curlevel = get_random_level(mult_);

    // Allocated before the slot is claimed so a failure leaves the index untouched.
    std::unique_ptr<char, void (*)(void*)> link_list(nullptr, std::free);
    if (curlevel > 0) {
        auto* raw =
            static_cast<char*>(std::calloc(1, (size_links_per_element_ * curlevel) + 1));
        if (raw == nullptr) {
            throw std::runtime_error(
                "Not enough memory: add_point_quant failed to allocate linklist"
            );
        }
        link_list.reset(raw);
    }

    PID cur_c = 0;
    PID label = 0;
    {
        std::unique_lock<std::mutex> lock_table(label_lookup_lock_);
        if (cur_element_count_ >= max_elements_) {
            throw std::runtime_error("The number of elements exceeds the specified limit");
        }
        cur_c = static_cast<PID>(cur_element_count_.load());
        label = cur_c;
        if (label_lookup_.find(label) != label_lookup_.end()) {
            throw std::runtime_error("Label already present");
        }
        cur_element_count_++;
        label_lookup_[label] = cur_c;
    }

    std::unique_lock<std::mutex> lock_el(link_list_locks_[cur_c]);
    element_levels_[cur_c] = curlevel;

    std::unique_lock<std::mutex> templock(global_);
    int maxlevelcopy = maxlevel_;
    if (curlevel <= maxlevelcopy) {
        templock.unlock();
    }
    PID curr_obj = enterpoint_node_;

    std::memset(
        data_level0_memory_ + (cur_c * size_data_per_element_), 0, size_data_per_element_
    );
    std::memcpy(get_external_label_pt(cur_c), &label, sizeof(PID));
    std::memcpy(get_clusterid_pt(cur_c), &cluster_id, sizeof(PID));

    std::vector<float> rotated_data(padded_dim_);
    rotator_->rotate(vec, rotated_data.data());
    quant::quantize_split_single(
        rotated_data.data(),
        reinterpret_cast<float*>(centroids_memory_) + (cluster_id * padded_dim_),
        padded_dim_,
        ex_bits_,
        get_bindata_by_internalid(cur_c),
        get_exdata_by_internalid(cur_c),
        metric_type_,
        config
    );

    if (curlevel > 0) {
        linkLists_[cur_c] = link_list.release();
    }

    if (static_cast<signed>(curr_obj) != -1) {
        if (curlevel < maxlevelcopy) {
            float curdist = get_quant_dist(curr_obj, cur_c);
            for (int level = maxlevelcopy; level > curlevel; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    std::unique_lock<std::mutex> lock(link_list_locks_[curr_obj]);
                    PID* data = get_linklist(curr_obj, level);
                    int size = get_list_count(data);
                    auto* datal = data + 1;
                    for (int i = 0; i < size; i++) {
                        PID cand = datal[i];
                        if (cand >= cur_element_count_) {
                            throw std::runtime_error("cand error");
                        }
                        float d = get_quant_dist(cand, cur_c);
                        if (d < curdist) {
                            curdist = d;
                            curr_obj = cand;
                            changed = true;
                        }
                    }
                }
            }
        }

        for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
            maxheap<std::pair<float, PID>> top_candidates =
                search_base_layer_quant(curr_obj, cur_c, level);
            curr_obj = mutually_connect_quant(cur_c, top_candidates, level);
        }
    } else {
        enterpoint_node_ = cur_c;
        maxlevel_ = curlevel;
    }

    if (curlevel > maxlevelcopy) {
        enterpoint_node_ = cur_c;
        maxlevel_ = curlevel;
    }
    return label;
}

inline std::vector<PID> HierarchicalNSW::add(
    const float* data, size_t n, const PID* cluster_ids, bool faster
) {
    if (data_level0_memory_ == nullptr || centroids_memory_ == nullptr ||
        rotator_ == nullptr || num_cluster_ == 0) {
        throw std::logic_error("HNSW index must be constructed or loaded before add");
    }
    if (n == 0) {
        return {};
    }
    if (data == nullptr || cluster_ids == nullptr) {
        throw std::invalid_argument("HNSW add inputs must not be null");
    }
    if (n > max_elements_ - cur_element_count_) {
        throw std::invalid_argument("HNSW add: not enough capacity");
    }
    if (n > buffer::kSearchBufferMaxPointCount - cur_element_count_) {
        throw std::invalid_argument("HNSW add: point count exceeds the supported ID range");
    }
    for (size_t i = 0; i < n; ++i) {
        if (cluster_ids[i] >= num_cluster_) {
            throw std::invalid_argument("HNSW cluster ID is out of range");
        }
    }

    quant::RabitqConfig config;
    if (faster) {
        config = quant::faster_config(padded_dim_, ex_bits_ + 1);
    }

    std::vector<PID> labels;
    labels.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        labels.push_back(add_point_quant(data + (i * dim_), cluster_ids[i], config));
    }
    return labels;
}

}  // namespace rabitqlib::hnsw
