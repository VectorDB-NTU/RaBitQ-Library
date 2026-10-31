#include "rabitqlib/index/symqg/qg_builder.hpp"

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "rabitqlib/core/rotator.hpp"
#include "rabitqlib/fastscan/query.hpp"
#include "rabitqlib/index/detail/buffer.hpp"
#include "rabitqlib/index/detail/exact_nn.hpp"
#include "rabitqlib/index/detail/parallel_exception.hpp"
#include "rabitqlib/index/detail/parallel_for.hpp"
#include "rabitqlib/index/detail/visited_set.hpp"
#include "rabitqlib/index/symqg/detail/pipnn.hpp"
#include "rabitqlib/index/symqg/qg.hpp"
#include "rabitqlib/index/types.hpp"
#include "rabitqlib/utils/space.hpp"
#include "rabitqlib/utils/tools.hpp"

namespace rabitqlib::symqg {

QGBuilder::BuildQuery::BuildQuery(
    const QuantizedGraph<float>& graph, const BuildCache* cache
)
    : graph_(graph), cache_(cache) {}

QGBuilder::BuildWorker::BuildWorker(size_t num_points, size_t visited_capacity)
    : visited(num_points, visited_capacity) {}

const QGBuilder::BuildCache* QGBuilder::build_cache() const {
    return cache_ ? &*cache_ : nullptr;
}

QGBuilder::QGBuilder(
    QuantizedGraph<float>& index,
    uint32_t ef_construction,
    size_t num_threads,
    uint32_t seed,
    bool cache_vectors
)
    : qg_(index)
    , ef_construction_(ef_construction)
    , num_workers_(resolve_num_threads(num_threads))
    , num_points_(qg_.num_vertices())
    , dim_(qg_.dimension())
    , max_degree_(qg_.degree_bound())
    , seed_(seed) {
    if (cache_vectors) {
        cache_.emplace();
    }
}

QGBuilder::QGBuilder(
    QuantizedGraph<float>& index,
    uint32_t ef_construction,
    const float* data,
    size_t num_threads,
    QGInitialization init,
    uint32_t seed,
    bool cache_vectors
)
    : QGBuilder(index, ef_construction, num_threads, seed, cache_vectors) {
    if (init != QGInitialization::PiPNN && init != QGInitialization::Random) {
        throw std::invalid_argument("QuantizedGraph initialization is invalid");
    }
    initialization_ = init;
    reset(data);
}

QGBuilder::QGBuilder(
    QuantizedGraph<float>& index,
    uint32_t ef_construction,
    const float* data,
    size_t num_threads,
    uint32_t seed,
    bool cache_vectors
)
    : QGBuilder(
          index,
          ef_construction,
          data,
          num_threads,
          QGInitialization::Random,
          seed,
          cache_vectors
      ) {}

void QGBuilder::reset(const float* data) {
    if (data == nullptr) {
        throw std::invalid_argument("QuantizedGraph build data must not be null");
    }
    if (initialization_ == QGInitialization::PiPNN) {
        auto initial = detail::build_initial_graph(
            data, num_points_, dim_, max_degree_, qg_.metric_type_, num_workers_
        );
        initialize_seed(data, initial.offsets, initial.neighbors);
    } else {
        initialize_storage(data);
        random_init();
    }
}

void QGBuilder::initialize_seed(
    const float* data, const std::vector<size_t>& offsets, const std::vector<PID>& neighbors
) {
    if (offsets.size() != num_points_ + 1 || offsets.front() != 0 ||
        offsets.back() != neighbors.size()) {
        throw std::invalid_argument(
            "QuantizedGraph seed graph offsets must delimit every vertex"
        );
    }
    for (size_t i = 0; i < num_points_; ++i) {
        if (offsets[i] > offsets[i + 1] || offsets[i + 1] > neighbors.size() ||
            offsets[i + 1] - offsets[i] > max_degree_) {
            throw std::invalid_argument(
                "QuantizedGraph seed graph row exceeds degree bound or has invalid offsets"
            );
        }
        for (size_t j = offsets[i]; j < offsets[i + 1]; ++j) {
            if (neighbors[j] >= num_points_ || neighbors[j] == i) {
                throw std::invalid_argument(
                    "QuantizedGraph seed graph IDs must be in range and exclude self"
                );
            }
        }
    }
    initialize_storage(data);
    index_detail::ParallelException failure;
#pragma omp parallel num_threads(num_workers_)
    {
        auto& worker = workers_[static_cast<size_t>(omp_get_thread_num())];
        auto& ids = worker.neighbor_ids;
        try {
            ids.reserve(max_degree_);
        } catch (...) { failure.capture(); }
#pragma omp for schedule(dynamic)
        for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(num_points_);
             ++index) {
            if (failure.failed()) {
                continue;
            }
            try {
                const size_t i = static_cast<size_t>(index);
                ids.assign(
                    neighbors.begin() + static_cast<std::ptrdiff_t>(offsets[i]),
                    neighbors.begin() + static_cast<std::ptrdiff_t>(offsets[i + 1])
                );
                std::sort(ids.begin(), ids.end());
                ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
                // Seed rows need only IDs; scores are materialized during search.
                write_row(static_cast<PID>(i), worker);
            } catch (...) { failure.capture(); }
        }
    }
    failure.rethrow();
}

void QGBuilder::refine() { run_passes(1); }

void QGBuilder::build() { run_passes(initialization_ == QGInitialization::PiPNN ? 1 : 3); }

void QGBuilder::build(size_t num_iter) {
    if (num_iter < 2) {
        throw std::invalid_argument("QuantizedGraph build iterations must be at least 2");
    }
    run_passes(num_iter);
}

bool QGBuilder::check_dup() const {
    std::atomic<bool> flag(false);
    index_detail::ParallelException failure;
#pragma omp parallel for num_threads(num_workers_)
    for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(num_points_);
         ++index) {
        if (failure.failed()) {
            continue;
        }
        try {
            const size_t i = static_cast<size_t>(index);
            std::unordered_set<PID> edges;
            for (const auto& neighbor : new_neighbors_[i]) {
                if (edges.find(neighbor.id) != edges.end()) {
                    flag = true;
                }
                edges.emplace(neighbor.id);
            }
        } catch (...) { failure.capture(); }
    }
    failure.rethrow();
    return flag;
}

float QGBuilder::avg_degree() const {
    // A size_t initial value keeps the sum from wrapping at 2^32 on large graphs.
    size_t degrees = std::accumulate(degrees_.begin(), degrees_.end(), size_t{0});
    return static_cast<float>(degrees) / static_cast<float>(num_points_);
}

void QGBuilder::run_passes(size_t num_iter) {
    qg_.ready_ = false;
    // Entry points guarantee a positive count; only the last pass completes degrees.
    for (size_t i = 1; i < num_iter; ++i) {
        run_pass(/*final_pass=*/false);
    }
    run_pass(/*final_pass=*/true);
    qg_.ready_ = true;
}

void QGBuilder::write_row(PID internal_id, BuildWorker& worker) {
    if (!worker.neighbor_ids.empty()) {
        const auto& new_neighbors = worker.neighbor_ids;
        const size_t num_neighbors = new_neighbors.size();
        const auto* cache = build_cache();
        const size_t padded_dim = qg_.padded_dim_;
        auto& rotated_data = worker.rotated_neighbors;
        rotated_data.resize((num_neighbors + 1) * padded_dim);
        float* rotated_source = rotated_data.data() + num_neighbors * padded_dim;
        // Cached rows, rotated raw vectors or reconstructed codes, as configured.
        const auto rotated_row = [&](PID source, float* row) {
            if (cache != nullptr) {
                std::copy_n(
                    cache->rotated_vectors.data() + source * padded_dim, padded_dim, row
                );
            } else if (qg_.quantization_bits_ == 0) {
                qg_.rotator_->rotate(qg_.get_vector(source), row);
            } else {
                qg_.reconstruct_quantized_vector(source, row);
            }
        };
        for (size_t i = 0; i < num_neighbors; ++i) {
            rotated_row(new_neighbors[i], rotated_data.data() + i * padded_dim);
        }
        rotated_row(internal_id, rotated_source);
        qg_.encode_neighbors(
            internal_id, new_neighbors, rotated_data.data(), rotated_source
        );
    }
    degrees_[internal_id] = worker.neighbor_ids.size();
}

void QGBuilder::write_row(
    PID internal_id, const CandidateList& neighbors, BuildWorker& worker
) {
    worker.neighbor_ids.resize(neighbors.size());
    for (size_t i = 0; i < neighbors.size(); ++i) {
        worker.neighbor_ids[i] = neighbors[i].id;
    }
    write_row(internal_id, worker);
}

void QGBuilder::initialize_storage(const float* data) {
    if (data == nullptr) {
        throw std::invalid_argument("QuantizedGraph build data must not be null");
    }
    // Allocate refinement scratch after PiPNN releases its dense workspace.
    // Clear old graph-dependent state on reset while retaining vector capacities.
    new_neighbors_.resize(num_points_);
    reverse_buffer_.resize(num_points_);
    for (size_t i = 0; i < num_points_; ++i) {
        new_neighbors_[i].clear();
        reverse_buffer_[i].clear();
    }
    if (workers_.empty()) {
        workers_ = std::vector<BuildWorker>(
            num_workers_,
            BuildWorker(
                num_points_, std::min(ef_construction_ * ef_construction_, num_points_ / 10)
            )
        );
    }
    degrees_.assign(num_points_, max_degree_);
    std::vector<float> centroid = compute_centroid(data, num_points_, dim_, num_workers_);

    qg_.ready_ = false;
    qg_.set_quantization_centroid(centroid.data());
    qg_.copy_vectors(data, num_workers_);

    if (cache_) {
        const size_t padded_dim = qg_.padded_dim_;
        cache_->rotated_vectors.resize(num_points_ * padded_dim);
        index_detail::ParallelException failure;
#pragma omp parallel for num_threads(num_workers_) schedule(static)
        for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(num_points_);
             ++index) {
            if (failure.failed()) {
                continue;
            }
            try {
                const auto internal_id = static_cast<PID>(index);
                float* row = cache_->rotated_vectors.data() + internal_id * padded_dim;
                if (qg_.is_quantized()) {
                    qg_.reconstruct_quantized_vector(internal_id, row);
                } else {
                    qg_.rotator_->rotate(qg_.get_vector(internal_id), row);
                }
            } catch (...) { failure.capture(); }
        }
        failure.rethrow();
        cache_->queries.clear();
        if (qg_.is_quantized()) {
            cache_->queries.reserve(num_points_);
            for (size_t internal_id = 0; internal_id < num_points_; ++internal_id) {
                cache_->queries.emplace_back(
                    cache_->rotated_vectors.data() + internal_id * padded_dim,
                    qg_.centroid_.data(),
                    padded_dim,
                    qg_.metric_type_
                );
            }
        }
    }

    PID entry_point = 0;
    if (qg_.is_quantized()) {
        QuantizedQuery query(
            qg_.centroid_.data(), qg_.centroid_.data(), qg_.padded_dim_, qg_.metric_type_
        );
        float best = std::numeric_limits<float>::max();
        for (PID internal_id = 0; internal_id < num_points_; ++internal_id) {
            const float distance = qg_.quantized_distance(query, internal_id);
            if (distance < best) {
                best = distance;
                entry_point = internal_id;
            }
        }
    } else {
        entry_point = exact_nn(
            data, centroid.data(), num_points_, dim_, num_workers_, qg_.raw_dist_func_
        );
    }

    qg_.set_ep(entry_point);
}

void QGBuilder::add_pruned_edges(
    const CandidateList& result,
    const CandidateList& pruned_list,
    CandidateList& new_result,
    float threshold
) {
    size_t start = 0;
    new_result.clear();
    new_result = result;

    std::unordered_set<PID> neighbor_ids;
    neighbor_ids.reserve(max_degree_);
    for (const auto& neighbor : result) {
        neighbor_ids.emplace(neighbor.id);
    }

    BuildQuery query(qg_, build_cache());
    while (new_result.size() < max_degree_ && start < pruned_list.size()) {
        const auto& candidate = pruned_list[start];
        bool occlude = false;
        query.reset(candidate.id);
        float dik_sqr = candidate.distance;

        if (neighbor_ids.find(candidate.id) != neighbor_ids.end()) {
            break;
        }

        for (const auto& neighbor : new_result) {
            float dij_sqr = neighbor.distance;
            if (dij_sqr > dik_sqr) {
                break;
            }
            float djk_sqr = query.distance_to(neighbor.id);
            float cosine =
                (dik_sqr + dij_sqr - djk_sqr) / (2 * std::sqrt(dij_sqr * dik_sqr));
            if (cosine > threshold) {
                occlude = true;
                break;
            }
        }

        if (!occlude) {
            new_result.emplace_back(candidate);
            neighbor_ids.emplace(candidate.id);
            std::sort(new_result.begin(), new_result.end());
        }

        ++start;
    }
}

void QGBuilder::heuristic_prune(
    CandidateList& pool, CandidateList& selected, CandidateList* recorded
) {
    if (pool.empty()) {
        return;
    }
    selected.clear();
    const size_t pool_size = pool.size();
    if (pool_size <= max_degree_) {
        selected = pool;
        return;
    }

    std::vector<bool> pruned(pool_size, false);
    size_t start = 0;

    BuildQuery query(qg_, build_cache());
    while (selected.size() < max_degree_ && start < pool_size) {
        auto candidate_id = pool[start].id;
        if (pruned[start]) {
            ++start;
            continue;
        }

        selected.emplace_back(pool[start]);
        query.reset(candidate_id);

        // i : current vertex
        // j : neighbor added in this iter
        // k : remained unpruned candidate neighbor
        for (size_t k = start + 1; k < pool_size; ++k) {
            if (pruned[k]) {
                continue;
            }
            float dik = pool[k].distance;
            auto djk = query.distance_to(pool[k].id);

            if (djk < dik) {
                if (recorded != nullptr && recorded->size() < kMaxPrunedSize) {
                    recorded->emplace_back(pool[k]);
                }
                pruned[k] = true;
            }
        }

        ++start;
    }
}

// Search for new neighbors in the graph. Search pruning does not record candidates
// for final degree completion.
void QGBuilder::search_new_neighbors() {
    index_detail::omp_for_dynamic(num_points_, num_workers_, [&](size_t i, size_t thread) {
        const auto internal_id = static_cast<PID>(i);
        auto& worker = workers_[thread];
        CandidateList& candidates = worker.candidates;
        candidates.clear();
        VisitedSet& visited = worker.visited;
        candidates.reserve(2 * kMaxCandidatePoolSize);
        visited.clear();
        find_candidates(internal_id, candidates, visited);

        // Seeded construction keeps its initial edges only in QG. Materialize
        // their scores on demand, after the caller can release the input/CSR.
        if (new_neighbors_[internal_id].empty() && degrees_[internal_id] != 0) {
            BuildQuery query(qg_, build_cache());
            query.reset(internal_id);
            const auto ids = qg_.get_neighbors(internal_id);
            for (size_t j = 0; j < degrees_[internal_id]; ++j) {
                if (ids[j] != internal_id && !visited.get(ids[j])) {
                    candidates.emplace_back(ids[j], query.distance_to(ids[j]));
                }
            }
        }

        // Add current neighbors retained by preceding iterations.
        for (const auto& neighbor : new_neighbors_[internal_id]) {
            const PID neighbor_id = neighbor.id;
            if (neighbor_id != internal_id && !visited.get(neighbor_id)) {
                candidates.emplace_back(neighbor);
            }
        }

        const size_t min_size = std::min(candidates.size(), kMaxCandidatePoolSize);
        std::partial_sort(
            candidates.begin(),
            candidates.begin() + static_cast<std::ptrdiff_t>(min_size),
            candidates.end()
        );
        candidates.resize(min_size);

        // Prune and update the graph.
        new_neighbors_[internal_id].reserve(max_degree_);
        heuristic_prune(candidates, new_neighbors_[internal_id], /*recorded=*/nullptr);
    });
}

void QGBuilder::add_reverse_edges(bool final_pass) {
    std::vector<std::mutex> locks(std::min(num_points_, kLockStripes));
    for (auto& row : reverse_buffer_) {
        row.clear();
    }

    // Keep new_neighbors_ read-only while reverse candidates are collected. Mutating a
    // destination row here races with another worker reading that row as its source.
    index_detail::omp_for_dynamic(
        num_points_, num_workers_, [&](size_t i, size_t /*thread*/) {
            const auto source = static_cast<PID>(i);
            for (const auto& neighbor : new_neighbors_[source]) {
                const PID destination = neighbor.id;
                const CandidateList& destination_neighbors = new_neighbors_[destination];
                const bool reciprocal = std::any_of(
                    destination_neighbors.begin(),
                    destination_neighbors.end(),
                    [&](const auto& destination_neighbor) {
                        return destination_neighbor.id == source;
                    }
                );
                if (reciprocal) {
                    continue;
                }

                std::lock_guard lock(locks[destination % locks.size()]);
                const size_t missing_slots =
                    max_degree_ - std::min(max_degree_, destination_neighbors.size());
                if (reverse_buffer_[destination].size() <
                    kMaxCandidatePoolSize + missing_slots) {
                    reverse_buffer_[destination].emplace_back(source, neighbor.distance);
                }
            }
        }
    );

    index_detail::omp_for_dynamic(num_points_, num_workers_, [&](size_t i, size_t thread) {
        const auto internal_id = static_cast<PID>(i);
        CandidateList& candidate_pool = reverse_buffer_[internal_id];
        if (qg_.is_quantized() && !candidate_pool.empty()) {
            // RaBitQ estimates are directional: score destination -> source afresh.
            BuildQuery query(qg_, build_cache());
            query.reset(internal_id);
            for (auto& candidate : candidate_pool) {
                candidate.distance = query.distance_to(candidate.id);
            }
        }
        candidate_pool.reserve(candidate_pool.size() + max_degree_);
        candidate_pool.insert(
            candidate_pool.end(),
            new_neighbors_[internal_id].begin(),
            new_neighbors_[internal_id].end()
        );
        std::sort(candidate_pool.begin(), candidate_pool.end());
        auto& worker = workers_[thread];
        worker.pruned.clear();
        if (final_pass) {
            worker.pruned.reserve(kMaxPrunedSize);
        }
        heuristic_prune(
            candidate_pool,
            new_neighbors_[internal_id],
            final_pass ? &worker.pruned : nullptr
        );
        // Reverse collection has finished. Complete this row against the old
        // graph before reusing its worker scratch; graph writes stay deferred.
        if (final_pass) {
            complete_degree(internal_id, worker);
        }
    });
}

void QGBuilder::random_init() {
    const PID min_id = 0;
    const PID max_id = num_points_ - 1;
    index_detail::omp_for_dynamic(num_points_, num_workers_, [&](size_t i, size_t thread) {
        new_neighbors_[i].clear();
        std::mt19937 rng(seed_ ^ static_cast<uint32_t>(i));
        std::uniform_int_distribution<PID> distribution(min_id, max_id);
        std::unordered_set<PID> neighbor_set;
        neighbor_set.reserve(max_degree_);
        while (neighbor_set.size() < max_degree_) {
            const PID rand_id = distribution(rng);
            if (rand_id != i) {
                neighbor_set.emplace(rand_id);
            }
        }

        BuildQuery query(qg_, build_cache());
        query.reset(i);
        new_neighbors_[i].reserve(max_degree_);
        for (PID neighbor_id : neighbor_set) {
            new_neighbors_[i].emplace_back(neighbor_id, query.distance_to(neighbor_id));
        }

        write_row(static_cast<PID>(i), new_neighbors_[i], workers_[thread]);
    });
}

// Complete one final-pass row using only this worker's reverse-prune candidates.
void QGBuilder::complete_degree(PID internal_id, BuildWorker& worker) {
    CandidateList& current_neighbors = new_neighbors_[internal_id];
    const size_t num_neighbors = current_neighbors.size();

    // Skip vertices with enough neighbors.
    if (num_neighbors >= max_degree_) {
        return;
    }

    CandidateList& pruned_list = worker.pruned;
    CandidateList& new_result = worker.completion_result;
    new_result.reserve(max_degree_);

    std::sort(pruned_list.begin(), pruned_list.end());

    // Use binary search to get refined results.
    CandidateList& successful_result = worker.successful_result;
    successful_result.clear();
    successful_result.reserve(max_degree_);
    float left = 0.5F;
    float right = 1.0F;
    size_t iter = 0;
    while (iter++ < kMaxBsIter) {
        float mid = (left + right) / 2;
        add_pruned_edges(current_neighbors, pruned_list, new_result, mid);
        if (new_result.size() < max_degree_) {
            left = mid;
        } else {
            right = mid;
            successful_result = new_result;
        }
    }

    // Every successful right endpoint was already evaluated at that exact
    // threshold.
    if (!successful_result.empty()) {
        new_result.swap(successful_result);
    } else {
        add_pruned_edges(current_neighbors, pruned_list, new_result, right);
    }

    // If the vertex still does not have enough neighbors, use random vertices.
    if (new_result.size() < max_degree_) {
        std::mt19937 rng(seed_ ^ static_cast<uint32_t>(internal_id) ^ 0x80000000U);
        std::uniform_int_distribution<PID> distribution(
            0, static_cast<PID>(num_points_) - 1
        );
        std::unordered_set<PID> ids;
        ids.reserve(max_degree_);
        for (auto& neighbor : new_result) {
            ids.emplace(neighbor.id);
        }
        BuildQuery query(qg_, build_cache());
        query.reset(internal_id);
        while (new_result.size() < max_degree_) {
            const PID rand_id = distribution(rng);
            if (rand_id != internal_id && ids.find(rand_id) == ids.end()) {
                new_result.emplace_back(rand_id, query.distance_to(rand_id));
                ids.emplace(rand_id);
            }
        }
    }

    current_neighbors = new_result;
}

void QGBuilder::run_pass(bool final_pass) {
    search_new_neighbors();
    add_reverse_edges(final_pass);

    // Update the graph.
    index_detail::omp_for_dynamic(num_points_, num_workers_, [&](size_t i, size_t thread) {
        write_row(static_cast<PID>(i), new_neighbors_[i], workers_[thread]);
    });
}

void QGBuilder::BuildQuery::reset(PID internal_id) {
    raw_ = nullptr;
    rotated_ = nullptr;
    prepared_.reset();
    if (cache_ != nullptr) {
        rotated_ = cache_->rotated_vectors.data() + internal_id * graph_.padded_dim_;
    }
    if (graph_.is_quantized()) {
        if (cache_ != nullptr) {
            prepared_.emplace(rotated_, cache_->queries[internal_id]);
        } else {
            storage_.resize(graph_.padded_dim_);
            // Reconstructed vectors are already rotated; do not rotate them again.
            graph_.reconstruct_quantized_vector(internal_id, storage_.data());
            rotated_ = storage_.data();
            prepared_.emplace(
                rotated_, graph_.centroid_.data(), graph_.padded_dim_, graph_.metric_type_
            );
        }
    } else {
        raw_ = graph_.get_vector(internal_id);
    }
}

float QGBuilder::BuildQuery::distance_to(PID target) const {
    return graph_.point_distance(raw_, prepared_ ? &*prepared_ : nullptr, target);
}

const float* QGBuilder::BuildQuery::rotated_query() {
    if (rotated_ == nullptr) {
        storage_.resize(graph_.padded_dim_);
        graph_.rotator_->rotate(raw_, storage_.data());
        rotated_ = storage_.data();
    }
    return rotated_;
}

// Find candidate neighbors for internal_id, excluding the vertex itself.
void QGBuilder::find_candidates(
    PID internal_id, CandidateList& results, VisitedSet& visited
) const {
    BuildQuery query(qg_, build_cache());
    query.reset(internal_id);
    BatchQuery<float> query_wrapper(
        query.rotated_query(), qg_.padded_dim_, qg_.metric_type_
    );
    buffer::SearchBuffer candidate_set(ef_construction_);
    candidate_set.insert(qg_.entry_point_id_, std::numeric_limits<float>::max());

    std::vector<float> estimated_distances(max_degree_);
    while (candidate_set.has_next()) {
        auto candidate_id = candidate_set.pop();
        if (visited.get(candidate_id)) {
            continue;
        }
        visited.set(candidate_id);
        auto num_neighbors = degrees_[candidate_id];
        const float vertex_distance = query.distance_to(candidate_id);
        query_wrapper.set_g_add(vertex_distance);
        qg_.scan_neighbors(
            query_wrapper,
            candidate_id,
            estimated_distances.data(),
            candidate_set,
            visited,
            num_neighbors
        );
        if (candidate_id != internal_id) {
            results.emplace_back(candidate_id, vertex_distance);
        }
    }
}

}  // namespace rabitqlib::symqg
