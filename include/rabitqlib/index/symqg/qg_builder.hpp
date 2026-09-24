#pragma once

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/index/symqg/detail/pipnn.hpp"
#include "rabitqlib/index/symqg/qg.hpp"
#include "rabitqlib/utils/space.hpp"
#include "rabitqlib/utils/tools.hpp"
#include "rabitqlib/utils/visited_set.hpp"

namespace rabitqlib::symqg {
constexpr size_t kMaxBsIter = 5;  // max iter for binary search of pruning bar
using CandidateList = std::vector<AnnCandidate<float>>;
enum class QGInitialization { PiPNN, Random };

// Owns temporary state for SymphonyQG initialization and refinement.
class QGBuilder {
    friend struct QGConstructionTestAccess;

   private:
    QuantizedGraph<float>& qg_;
    size_t ef_build_;      // size of search pool for indexing
    size_t num_threads_;   // number of threads used for indexing
    size_t num_nodes_;     // num of data points
    size_t dim_;           // dimension of data
    size_t degree_bound_;  // degree bound for qg, multiple of 32
    bool seeded_ = true;
    static constexpr size_t kMaxCandidatePoolSize =
        750;  // max num of candidates for indexing
    static constexpr size_t kMaxPrunedSize =
        300;                                    // max number of recorded pruned candidates
    std::vector<CandidateList> new_neighbors_;  // new neighbors for current iteration
    std::vector<CandidateList> pruned_neighbors_;  // recorded pruned neighbors
    std::vector<VisitedSet> visited_list_;         // per-thread visited sets
    std::vector<uint32_t> degrees_;                // record degree of qg
    void random_init();
    void search_new_neighbors(bool refine);
    void heuristic_prune(PID, CandidateList&, CandidateList&, bool);
    void add_reverse_edges(bool);
    void add_pruned_edges(
        const CandidateList&, const CandidateList&, CandidateList&, float
    );
    void graph_refine();
    void iter(bool);

    void initialize_storage(const float* data);

    QGBuilder(QuantizedGraph<float>& index, uint32_t ef_build, size_t num_threads)
        : qg_{index}
        , ef_build_{ef_build}
        , num_threads_{resolve_num_threads(num_threads)}
        , num_nodes_{qg_.num_vertices()}
        , dim_{qg_.dimension()}
        , degree_bound_(qg_.degree_bound()) {}

   public:
    explicit QGBuilder(
        QuantizedGraph<float>& index,
        uint32_t ef_build,
        const float* data,
        size_t num_threads = std::numeric_limits<size_t>::max(),
        QGInitialization init = QGInitialization::PiPNN
    )
        : QGBuilder(index, ef_build, num_threads) {
        if (data == nullptr) {
            throw std::invalid_argument("QGBuilder data must not be null");
        }
        if (init != QGInitialization::PiPNN && init != QGInitialization::Random) {
            throw std::invalid_argument("Unknown QG initialization");
        }
        if (init == QGInitialization::PiPNN) {
            auto seed = detail::build_initial_graph(
                data, num_nodes_, dim_, degree_bound_, qg_.metric_type_, num_threads_
            );
            initialize_seed(data, seed.offsets, seed.neighbors);
        } else if (init == QGInitialization::Random) {
            seeded_ = false;
            initialize_storage(data);
            random_init();
        }
    }

   private:
    void initialize_seed(
        const float* data,
        const std::vector<size_t>& offsets,
        const std::vector<PID>& neighbors
    ) {
        if (offsets.size() != num_nodes_ + 1 || offsets.front() != 0 ||
            offsets.back() != neighbors.size()) {
            throw std::invalid_argument("Seed graph offsets must delimit every vertex");
        }
        for (size_t i = 0; i < num_nodes_; ++i) {
            if (offsets[i] > offsets[i + 1] || offsets[i + 1] > neighbors.size() ||
                offsets[i + 1] - offsets[i] > degree_bound_) {
                throw std::invalid_argument(
                    "Seed graph row exceeds degree bound or has invalid offsets"
                );
            }
            for (size_t j = offsets[i]; j < offsets[i + 1]; ++j) {
                if (neighbors[j] >= num_nodes_ || neighbors[j] == i) {
                    throw std::invalid_argument(
                        "Seed graph IDs must be in range and exclude self"
                    );
                }
            }
        }
        initialize_storage(data);
#pragma omp parallel num_threads(num_threads_)
        {
            CandidateList row;
            row.reserve(degree_bound_);
            std::vector<PID> ids;
            ids.reserve(degree_bound_);
#pragma omp for schedule(dynamic)
            for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(num_nodes_);
                 ++index) {
                const size_t i = static_cast<size_t>(index);
                row.clear();
                ids.assign(
                    neighbors.begin() + static_cast<ptrdiff_t>(offsets[i]),
                    neighbors.begin() + static_cast<ptrdiff_t>(offsets[i + 1])
                );
                std::sort(ids.begin(), ids.end());
                ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
                for (PID id : ids) {
                    // Encoding only consumes IDs. Score surviving seed candidates
                    // during search instead of retaining a second graph of scores.
                    row.emplace_back(id, 0.0F);
                }
                degrees_[i] = row.size();
                qg_.update_qg(i, row);
            }
        }
    }

   public:
    // One complete search/prune/reverse-edge/degree-completion iteration. Call
    // before serving a seeded graph: search() requires full final rows.
    void refine() {
        qg_.ready_ = false;
        iter(true);
        qg_.ready_ = true;
    }

    // One refinement for PiPNN initialization, three passes for random init.
    void build() {
        if (seeded_) {
            refine();
        } else {
            build(3);
        }
    }

    void build(size_t num_iter) {
        if (num_iter < 2) {
            throw std::invalid_argument(
                "The number of QG build iterations must be at least 2"
            );
        }
        qg_.ready_ = false;
        // for first iterations, we do not need to refine the graph structure
        for (size_t i = 0; i < num_iter - 1; ++i) {
            iter(false);
        }
        iter(true);
        qg_.ready_ = true;
    }

    [[nodiscard]] bool check_dup() const {
        std::atomic<bool> flag(false);
#pragma omp parallel for num_threads(num_threads_)
        for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(num_nodes_);
             ++index) {
            const size_t i = static_cast<size_t>(index);
            std::unordered_set<PID> edges;
            for (auto nei : new_neighbors_[i]) {
                if (edges.find(nei.id) != edges.end()) {
                    flag = true;
                }
                edges.emplace(nei.id);
            }
        }
        return flag;
    }

    [[nodiscard]] float avg_degree() const {
        size_t degrees = std::accumulate(degrees_.begin(), degrees_.end(), 0U);
        return static_cast<float>(degrees) / static_cast<float>(num_nodes_);
    }
};

inline void QGBuilder::initialize_storage(const float* data) {
    if (data == nullptr) {
        throw std::invalid_argument("QGBuilder data must not be null");
    }
    // Allocate refinement scratch only after PiPNN's dense workspace is released.
    new_neighbors_.resize(num_nodes_);
    pruned_neighbors_.resize(num_nodes_);
    visited_list_ = std::vector<VisitedSet>(
        num_threads_,
        VisitedSet(num_nodes_, std::min(ef_build_ * ef_build_, num_nodes_ / 10))
    );
    degrees_.assign(num_nodes_, degree_bound_);
    std::vector<float> centroid = compute_centroid(data, num_nodes_, dim_, num_threads_);

    qg_.ready_ = false;
    qg_.set_quantization_centroid(centroid.data());
    qg_.copy_vectors(data, num_threads_);

    PID entry_point = 0;
    if (qg_.is_quantized()) {
        QuantizedQuery query(
            qg_.centroid_.data(), qg_.centroid_.data(), qg_.padded_dim_, qg_.metric_type_
        );
        float best = std::numeric_limits<float>::max();
        for (PID id = 0; id < num_nodes_; ++id) {
            const float distance = qg_.quantized_distance(query, id);
            if (distance < best) {
                best = distance;
                entry_point = id;
            }
        }
    } else {
        entry_point = exact_nn(
            data, centroid.data(), num_nodes_, dim_, num_threads_, qg_.raw_dist_func_
        );
    }

    qg_.set_ep(entry_point);
}

inline void QGBuilder::add_pruned_edges(
    const CandidateList& result,
    const CandidateList& pruned_list,
    CandidateList& new_result,
    float threshold
) {
    size_t start = 0;
    new_result.clear();
    new_result = result;

    std::unordered_set<PID> nei_set;
    nei_set.reserve(degree_bound_);
    for (const auto& nei : result) {
        nei_set.emplace(nei.id);
    }

    std::vector<float> reconstructed;
    std::optional<QuantizedQuery> prepared;
    while (new_result.size() < degree_bound_ && start < pruned_list.size()) {
        const auto& cur = pruned_list[start];
        bool occlude = false;
        const float* cur_data = qg_.prepare_build_query(cur.id, reconstructed, prepared);
        float dik_sqr = cur.distance;

        if (nei_set.find(cur.id) != nei_set.end()) {
            break;
        }

        for (auto& nei : new_result) {
            float dij_sqr = nei.distance;
            if (dij_sqr > dik_sqr) {
                break;
            }
            float djk_sqr =
                qg_.point_distance(cur_data, prepared ? &*prepared : nullptr, nei.id);
            float cosine =
                (dik_sqr + dij_sqr - djk_sqr) / (2 * std::sqrt(dij_sqr * dik_sqr));
            if (cosine > threshold) {
                occlude = true;
                break;
            }
        }

        if (!occlude) {
            new_result.emplace_back(cur);
            nei_set.emplace(cur.id);
            std::sort(new_result.begin(), new_result.end());
        }

        ++start;
    }
}

inline void QGBuilder::heuristic_prune(
    PID cur_id, CandidateList& pool, CandidateList& pruned_results, bool refine
) {
    if (pool.empty()) {
        return;
    }
    pruned_results.clear();
    size_t poolsize = pool.size();

    // if we dont have enough candidates, just keep all neighbors
    if (poolsize <= degree_bound_) {
        pruned_results = pool;
        return;
    }

    std::vector<bool> pruned(
        poolsize, false
    );                 // bool vector to record if this neighbor is pruned
    size_t start = 0;  // start position

    std::vector<float> reconstructed;
    std::optional<QuantizedQuery> prepared;
    while (pruned_results.size() < degree_bound_ && start < poolsize) {
        auto candidate_id = pool[start].id;

        // if already pruned, move to next
        if (pruned[start]) {
            ++start;
            continue;
        }

        pruned_results.emplace_back(pool[start]);  // add current candidate to result
        const float* data_j =
            qg_.prepare_build_query(candidate_id, reconstructed, prepared);

        // i : current vertex
        // j : neighbor added in this iter
        // k : remained unpruned candidate neighbor
        for (size_t k = start + 1; k < poolsize; ++k) {
            if (pruned[k]) {
                continue;
            }
            float dik = pool[k].distance;
            auto djk =
                qg_.point_distance(data_j, prepared ? &*prepared : nullptr, pool[k].id);

            if (djk < dik) {
                if (refine && pruned_neighbors_[cur_id].size() < kMaxPrunedSize) {
                    pruned_neighbors_[cur_id].emplace_back(pool[k]);
                }
                pruned[k] = true;
            }
        }

        ++start;
    }
}

/**
 * @brief search for new neighbor in qg
 *
 * @param refine refine = true means recording pruned candidates
 */
inline void QGBuilder::search_new_neighbors(bool refine) {
#pragma omp parallel for schedule(dynamic) num_threads(num_threads_)
    for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(num_nodes_);
         ++index) {
        const size_t i = static_cast<size_t>(index);
        PID cur_id = i;
        auto tid = omp_get_thread_num();
        CandidateList candidates;
        VisitedSet& vis = visited_list_[tid];
        candidates.reserve(2 * kMaxCandidatePoolSize);
        vis.clear();
        qg_.find_candidates(cur_id, ef_build_, candidates, vis, degrees_);

        // Seeded construction keeps its initial edges only in QG. Materialize
        // their scores on demand, after the caller can release the input/CSR.
        if (new_neighbors_[cur_id].empty() && degrees_[cur_id] != 0) {
            std::vector<float> reconstructed;
            std::optional<QuantizedQuery> prepared;
            const float* source = qg_.prepare_build_query(cur_id, reconstructed, prepared);
            const auto ids = qg_.get_neighbors(cur_id);
            for (size_t j = 0; j < degrees_[cur_id]; ++j) {
                if (ids[j] != cur_id && !vis.get(ids[j])) {
                    candidates.emplace_back(
                        ids[j],
                        qg_.point_distance(source, prepared ? &*prepared : nullptr, ids[j])
                    );
                }
            }
        }

        // Add current neighbors retained by preceding iterations.
        for (auto& nei : new_neighbors_[cur_id]) {
            auto neighbor_id = nei.id;
            if (neighbor_id != cur_id && !vis.get(neighbor_id)) {
                candidates.emplace_back(nei);
            }
        }

        size_t min_size = std::min(candidates.size(), kMaxCandidatePoolSize);
        std::partial_sort(
            candidates.begin(),
            candidates.begin() + static_cast<long>(min_size),
            candidates.end()
        );
        candidates.resize(min_size);

        // prune and update qg
        new_neighbors_[cur_id].reserve(degree_bound_);
        heuristic_prune(cur_id, candidates, new_neighbors_[cur_id], refine);
    }
}

inline void QGBuilder::add_reverse_edges(bool refine) {
    std::vector<std::mutex> locks(num_nodes_);
    std::vector<CandidateList> reverse_buffer(num_nodes_);

    // Keep new_neighbors_ read-only while reverse candidates are collected. Mutating a
    // destination row here races with another worker reading that row as its source.
#pragma omp parallel for schedule(dynamic) num_threads(num_threads_)
    for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(num_nodes_);
         ++index) {
        const PID data_id = static_cast<PID>(index);
        for (const auto& nei : new_neighbors_[data_id]) {
            const PID destination = nei.id;
            const CandidateList& destination_neighbors = new_neighbors_[destination];
            const bool reciprocal = std::any_of(
                destination_neighbors.begin(),
                destination_neighbors.end(),
                [&](const auto& destination_neighbor) {
                    return destination_neighbor.id == data_id;
                }
            );
            if (reciprocal) {
                continue;
            }

            std::lock_guard lock(locks[destination]);
            const size_t missing_slots =
                degree_bound_ - std::min(degree_bound_, destination_neighbors.size());
            if (reverse_buffer[destination].size() <
                kMaxCandidatePoolSize + missing_slots) {
                reverse_buffer[destination].emplace_back(data_id, nei.distance);
            }
        }
    }

#pragma omp parallel for schedule(dynamic) num_threads(num_threads_)
    for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(num_nodes_);
         ++index) {
        const PID data_id = static_cast<PID>(index);
        CandidateList& tmp_pool = reverse_buffer[data_id];
        if (qg_.is_quantized() && !tmp_pool.empty()) {
            // RaBitQ estimates are directional: score destination -> source afresh.
            std::vector<float> reconstructed;
            std::optional<QuantizedQuery> prepared;
            qg_.prepare_build_query(data_id, reconstructed, prepared);
            for (auto& candidate : tmp_pool) {
                candidate.distance = qg_.quantized_distance(*prepared, candidate.id);
            }
        }
        tmp_pool.reserve(tmp_pool.size() + degree_bound_);
        tmp_pool.insert(
            tmp_pool.end(), new_neighbors_[data_id].begin(), new_neighbors_[data_id].end()
        );
        std::sort(tmp_pool.begin(), tmp_pool.end());
        heuristic_prune(data_id, tmp_pool, new_neighbors_[data_id], refine);
    }
}

inline void QGBuilder::random_init() {
    const PID min_id = 0;
    const PID max_id = num_nodes_ - 1;
#pragma omp parallel for schedule(dynamic) num_threads(num_threads_)
    for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(num_nodes_);
         ++index) {
        const size_t i = static_cast<size_t>(index);
        std::unordered_set<PID> neighbor_set;
        neighbor_set.reserve(degree_bound_);
        while (neighbor_set.size() < degree_bound_) {
            PID rand_id = rand_integer<PID>(min_id, max_id);
            if (rand_id != i) {
                neighbor_set.emplace(rand_id);
            }
        }

        std::vector<float> reconstructed;
        std::optional<QuantizedQuery> prepared;
        const float* cur_data = qg_.prepare_build_query(i, reconstructed, prepared);
        new_neighbors_[i].reserve(degree_bound_);
        for (PID cur_neigh : neighbor_set) {
            new_neighbors_[i].emplace_back(
                cur_neigh,
                qg_.point_distance(cur_data, prepared ? &*prepared : nullptr, cur_neigh)
            );
        }

        degrees_[i] = new_neighbors_[i].size();
        qg_.update_qg(i, new_neighbors_[i]);
    }
}

/**
 * @brief refine the graph structure, make sure the degree for each vertex in qg equals the
 * degree bound (multiple of 32)
 *
 */
inline void QGBuilder::graph_refine() {
#pragma omp parallel for schedule(dynamic) num_threads(num_threads_)
    for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(num_nodes_);
         ++index) {
        const size_t i = static_cast<size_t>(index);
        CandidateList& cur_neighbors = new_neighbors_[i];
        size_t cur_degree = cur_neighbors.size();

        // skip vertices with enough neighbors
        if (cur_degree >= degree_bound_) {
            continue;
        }

        CandidateList& pruned_list = pruned_neighbors_[i];
        CandidateList new_result;
        new_result.reserve(degree_bound_);

        std::sort(pruned_list.begin(), pruned_list.end());

        // use binary search to get refined results
        float left = 0.5;
        float right = 1.0;
        size_t iter = 0;
        while (iter++ < kMaxBsIter) {
            float mid = (left + right) / 2;
            add_pruned_edges(cur_neighbors, pruned_list, new_result, mid);
            if (new_result.size() < degree_bound_) {
                left = mid;
            } else {
                right = mid;
            }
        }

        // update neighbors with larger cosine value since we want to retain more edges
        add_pruned_edges(cur_neighbors, pruned_list, new_result, right);

        // if the vertex still doesn't have enough neighbors, use random vertices
        if (new_result.size() < degree_bound_) {
            std::unordered_set<PID> ids;
            ids.reserve(degree_bound_);
            for (auto& neighbor : new_result) {
                ids.emplace(neighbor.id);
            }
            std::vector<float> reconstructed;
            std::optional<QuantizedQuery> prepared;
            const float* source = qg_.prepare_build_query(i, reconstructed, prepared);
            while (new_result.size() < degree_bound_) {
                PID rand_id = rand_integer<PID>(0, static_cast<PID>(num_nodes_) - 1);
                if (rand_id != static_cast<PID>(i) && ids.find(rand_id) == ids.end()) {
                    new_result.emplace_back(
                        rand_id,
                        qg_.point_distance(source, prepared ? &*prepared : nullptr, rand_id)
                    );
                    ids.emplace(rand_id);
                }
            }
        }

        cur_neighbors = new_result;
    }
}

inline void QGBuilder::iter(bool refine) {
    if (refine) {
        for (size_t i = 0; i < num_nodes_; ++i) {
            pruned_neighbors_[i].clear();
            pruned_neighbors_[i].reserve(kMaxPrunedSize);
        }
    }

    search_new_neighbors(refine);

    add_reverse_edges(refine);

    // Use pruned edges to refine graph
    if (refine) {
        graph_refine();
    }

    // update qg
#pragma omp parallel for schedule(dynamic) num_threads(num_threads_)
    for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(num_nodes_);
         ++index) {
        const size_t i = static_cast<size_t>(index);
        qg_.update_qg(i, new_neighbors_[i]);
        degrees_[i] = new_neighbors_[i].size();
    }
}
}  // namespace rabitqlib::symqg
