#include "rabitqlib/index/symqg/detail/pipnn.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "pipnn_internal.hpp"
#include "rabitqlib/core/detail/eigen.hpp"
#include "rabitqlib/core/detail/simd/matrix_dispatch.hpp"
#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/index/detail/buffer.hpp"
#include "rabitqlib/index/detail/parallel_exception.hpp"
#include "rabitqlib/index/types.hpp"
#include "rabitqlib/utils/space.hpp"
#include "rabitqlib/utils/tools.hpp"

namespace rabitqlib::symqg::detail {

namespace pipnn_impl {
constexpr size_t kTileSize = 2048;
constexpr size_t kMinTileSize = 64;
constexpr size_t kHashBits = 12;
constexpr size_t kLocalDegree = 2;
constexpr float kAlpha = 1.1F;

struct Candidate {
    PID id;
    float distance;
    uint16_t hash;
};

// Collection maintains hash-sorted, bounded rows under their own locks. After
// collection finishes, each row is pruned once and reused to hold only selected IDs.
class CandidateTable {
   public:
    CandidateTable(size_t count, size_t capacity)
        : capacity_(capacity)
        , table_(count * capacity)
        , sizes_(count, 0)
        , worst_(count, 0)
        , locks_(count) {}

    void merge(
        PID source, PID target, float distance, const RowMajorMatrix<float>& sketches
    ) {
        uint16_t hash = 0;
        for (size_t bit = 0; bit < kHashBits; ++bit) {
            hash = static_cast<uint16_t>(
                (hash << 1) | (sketches(target, static_cast<Eigen::Index>(bit)) >
                               sketches(source, static_cast<Eigen::Index>(bit)))
            );
        }
        std::lock_guard lock(locks_[source]);
        auto* row = table_.data() + source * capacity_;
        auto& size = sizes_[source];
        auto& furthest = worst_[source];
        if (size == capacity_ && distance >= row[furthest].distance) {
            return;
        }
        auto* pos = std::lower_bound(
            row, row + size, hash, [](const Candidate& candidate, uint16_t key) {
                return candidate.hash < key;
            }
        );
        const size_t slot = static_cast<size_t>(pos - row);
        if (slot < size && pos->hash == hash) {
            if (distance >= pos->distance) {
                return;
            }
            *pos = {target, distance, hash};
        } else if (size < capacity_) {
            std::move_backward(pos, row + size, row + size + 1);
            *pos = {target, distance, hash};
            ++size;
        } else {
            const size_t insertion = slot > furthest ? slot - 1 : slot;
            if (slot > furthest) {
                std::move(row + furthest + 1, row + slot, row + furthest);
            } else {
                std::move_backward(row + slot, row + furthest, row + furthest + 1);
            }
            row[insertion] = {target, distance, hash};
        }
        // Bounded rows keep this scan small; retain full-precision scores.
        furthest = static_cast<size_t>(
            std::max_element(
                row,
                row + size,
                [](const Candidate& a, const Candidate& b) {
                    return a.distance < b.distance;
                }
            ) -
            row
        );
    }

    // Pruning runs after all merges, with one worker owning each row.
    void copy_candidates(size_t id, std::vector<Candidate>& candidates) const {
        const auto* row = table_.data() + id * capacity_;
        candidates.assign(row, row + sizes_[id]);
    }

    void retain_ids(size_t id, const Bucket& selected) {
        auto* row = table_.data() + id * capacity_;
        for (size_t j = 0; j < selected.size(); ++j) {
            row[j].id = selected[j];
        }
        sizes_[id] = selected.size();
    }

    // Export only after every row has been pruned.
    InitialGraph export_graph(size_t num_workers) const {
        InitialGraph result;
        result.offsets.resize(sizes_.size() + 1, 0);
        for (size_t i = 0; i < sizes_.size(); ++i) {
            result.offsets[i + 1] = result.offsets[i] + sizes_[i];
        }
        result.neighbors.resize(result.offsets.back());
#pragma omp parallel for num_threads(num_workers)
        for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(sizes_.size());
             ++index) {
            const size_t i = static_cast<size_t>(index);
            const auto* row = table_.data() + i * capacity_;
            for (size_t j = 0; j < sizes_[i]; ++j) {
                result.neighbors[result.offsets[i] + j] = row[j].id;
            }
        }
        return result;
    }

   private:
    size_t capacity_;
    std::vector<Candidate> table_;
    std::vector<size_t> sizes_;
    std::vector<size_t> worst_;
    std::vector<std::mutex> locks_;
};

struct BucketJob {
    Bucket ids;
    size_t depth = 0;
    uint32_t seed = 555;
};

ScratchPool::ScratchPool(size_t count, size_t dim, size_t num_workers)
    : points_buffer_size(std::min(count, kTileSize) * dim)
    , distances_buffer_size(
          std::max(
              std::min(count, kTileSize) * std::min(count, size_t{512}),
              std::min(count, kLeafSize) * std::min(count, kLeafSize)
          )
      )
    , worker_buffer_size(
          points_buffer_size + distances_buffer_size + std::min(count, kTileSize)
      )
    // Preserve uninitialized allocation and worker first-touch placement.
    , storage(new float[num_workers * worker_buffer_size]) {}

Workspace ScratchPool::worker(size_t id) {
    float* base = storage.get() + id * worker_buffer_size;
    return {
        base,
        base + points_buffer_size,
        base + points_buffer_size + distances_buffer_size,
        {}};
}

void gather(const float* data, size_t dim, const Bucket& ids, Workspace& work) {
    for (size_t i = 0; i < ids.size(); ++i) {
        std::copy_n(data + ids[i] * dim, dim, work.points + i * dim);
    }
}

void pairwise(Workspace& work, size_t size, size_t dim, MetricType metric_type) {
    simd::pairwise_distances_lower(
        work.points, work.distances, work.norms, size, dim, metric_type == METRIC_IP
    );
}

void select_local_neighbors(const Bucket& ids, Workspace& work) {
    static_assert(kLocalDegree == 2);
    work.candidates.assign(
        ids.size() * kLocalDegree, {std::numeric_limits<float>::infinity(), kPidMax}
    );
    const auto keep_nearest = [&](size_t row, float distance, PID target) {
        auto* best = work.candidates.data() + row * kLocalDegree;
        const std::pair<float, PID> candidate(distance, target);
        if (candidate < best[1]) {
            best[1] = candidate;
            if (best[1] < best[0]) {
                std::swap(best[0], best[1]);
            }
        }
    };
    // Both metrics are symmetric. Read each stored pair once, contiguously,
    // instead of gathering the upper half through a strided column walk.
    for (size_t i = 1; i < ids.size(); ++i) {
        const float* distances = work.distances + i * ids.size();
        for (size_t j = 0; j < i; ++j) {
            const float distance = distances[j];
            keep_nearest(i, distance, ids[j]);
            keep_nearest(j, distance, ids[i]);
        }
    }
}

std::vector<Bucket> partition(
    const float* data,
    size_t dim,
    const BucketJob& job,
    MetricType metric_type,
    size_t num_workers,
    ScratchPool& scratch
) {
    const auto& ids = job.ids;
    // Small root partitions need enough tiles to occupy the requested team.
    // Deeper levels already distribute independent buckets across workers.
    const size_t tile_size =
        job.depth == 0
            ? std::min(
                  kTileSize,
                  round_up_to_multiple(div_round_up(ids.size(), num_workers), kMinTileSize)
              )
            : kTileSize;
    const size_t leader_count = std::min(
        ids.size(), std::clamp<size_t>(job.depth == 0 ? 512 : ids.size() / 200, 3, 512)
    );
    const size_t fanout = std::min(
        leader_count,
        job.depth == 0   ? size_t{10}
        : job.depth == 1 ? size_t{3}
                         : size_t{1}
    );
    std::mt19937 random(job.seed);
    Bucket leaders;
    leaders.reserve(leader_count);
    std::sample(ids.begin(), ids.end(), std::back_inserter(leaders), leader_count, random);
    RowMajorMatrix<float> leader_data(leader_count, dim);
    for (size_t i = 0; i < leader_count; ++i) {
        std::copy_n(data + leaders[i] * dim, dim, leader_data.data() + i * dim);
    }
    Vector<float> leader_norms(leader_count);
    simd::row_norms(leader_data.data(), leader_norms.data(), leader_count, dim);
    std::vector<PID> assignments(ids.size() * fanout);
    const auto assign_tile = [&](size_t begin, Workspace& work) {
        // Both callers run inside a PiPNN OpenMP task, including one-thread teams.
        // Keep Eigen within that worker without changing the caller's thread budget.
        omp_set_num_threads(1);
        const size_t count = std::min(tile_size, ids.size() - begin);
        RowMajorMatrixMap<float> points(
            work.points, static_cast<Eigen::Index>(count), static_cast<Eigen::Index>(dim)
        );
        RowMajorMatrixMap<float> distances(
            work.distances,
            static_cast<Eigen::Index>(count),
            static_cast<Eigen::Index>(leader_count)
        );
        VectorMap<float> norms(work.norms, static_cast<Eigen::Index>(count));
        for (size_t i = 0; i < count; ++i) {
            std::copy_n(data + ids[begin + i] * dim, dim, points.data() + i * dim);
        }
        simd::matrix_product_transposed(
            points.data(), leader_data.data(), distances.data(), count, dim, leader_count
        );
        simd::row_norms(points.data(), norms.data(), count, dim);
        work.candidates.resize(leader_count);
        for (size_t i = 0; i < count; ++i) {
            for (size_t j = 0; j < leader_count; ++j) {
                const float dot =
                    distances(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
                const float distance =
                    metric_type == METRIC_L2
                        ? std::max(0.0F, norms.data()[i] + leader_norms.data()[j] - 2 * dot)
                        : -dot;
                work.candidates[j] = {distance, static_cast<PID>(j)};
            }
            std::partial_sort(
                work.candidates.begin(),
                work.candidates.begin() + static_cast<ptrdiff_t>(fanout),
                work.candidates.end()
            );
            for (size_t j = 0; j < fanout; ++j) {
                assignments[(begin + i) * fanout + j] = work.candidates[j].second;
            }
        }
    };
    if (job.depth == 0) {
        {
            index_detail::ParallelException failure;
#pragma omp parallel num_threads(std::min(num_workers, div_round_up(ids.size(), tile_size)))
            {
                auto work = scratch.worker(static_cast<size_t>(omp_get_thread_num()));
#pragma omp for schedule(static)
                for (std::ptrdiff_t offset = 0;
                     offset < static_cast<std::ptrdiff_t>(ids.size());
                     offset += static_cast<std::ptrdiff_t>(tile_size)) {
                    if (failure.failed()) {
                        continue;
                    }
                    try {
                        const size_t begin = static_cast<size_t>(offset);
                        assign_tile(begin, work);
                    } catch (...) { failure.capture(); }
                }
            }
            failure.rethrow();
        }
    } else {
        // Already inside the bucket-level team. Eigen must see that team rather
        // than a serialized nested region when deciding its own parallelism.
        auto work = scratch.worker(static_cast<size_t>(omp_get_thread_num()));
        for (size_t begin = 0; begin < ids.size(); begin += tile_size) {
            assign_tile(begin, work);
        }
    }
    std::vector<Bucket> children(leader_count);
    for (size_t i = 0; i < ids.size(); ++i) {
        for (size_t j = 0; j < fanout; ++j) {
            children[assignments[i * fanout + j]].push_back(ids[i]);
        }
    }
    return children;
}

std::vector<Bucket> cluster(
    const float* data,
    size_t count,
    size_t dim,
    MetricType metric_type,
    size_t num_workers,
    ScratchPool& scratch
) {
    Bucket all(count);
    std::iota(all.begin(), all.end(), PID{0});
    // The root parallelizes tiles; deeper levels parallelize independent buckets.
    std::vector<BucketJob> active;
    active.push_back({std::move(all), 0, 555});
    std::vector<Bucket> leaves;
    while (!active.empty()) {
        std::vector<std::vector<BucketJob>> next(active.size());
        std::vector<std::vector<Bucket>> finished(active.size());
        const auto process_bucket = [&](size_t i) {
            auto& job = active[i];
            if (job.ids.size() <= kLeafSize) {
                finished[i].push_back(std::move(job.ids));
                return;
            }
            auto children = partition(data, dim, job, metric_type, num_workers, scratch);
            std::sort(children.begin(), children.end(), [](const auto& a, const auto& b) {
                return a.size() < b.size();
            });
            Bucket small;
            for (size_t j = 0; j < children.size(); ++j) {
                auto& child = children[j];
                if (child.empty()) {
                    continue;
                }
                if (child.size() < kLeafSize / 10) {
                    small.insert(small.end(), child.begin(), child.end());
                    if (small.size() < kLeafSize / 10) {
                        continue;
                    }
                    std::sort(small.begin(), small.end());
                    small.erase(std::unique(small.begin(), small.end()), small.end());
                    finished[i].push_back(std::move(small));
                    small = Bucket();
                } else if (child.size() <= kLeafSize) {
                    finished[i].push_back(std::move(child));
                } else if (
                    job.depth >= 4 ||
                    (job.depth >= 2 && child.size() * 5 > job.ids.size() * 4)
                ) {
                    // Duplicates must not cause unbounded recursion/replication.
                    std::mt19937 random(job.seed + static_cast<uint32_t>(j));
                    std::shuffle(child.begin(), child.end(), random);
                    for (size_t begin = 0; begin < child.size(); begin += kLeafSize) {
                        finished[i].emplace_back(
                            child.begin() + static_cast<ptrdiff_t>(begin),
                            child.begin() + static_cast<ptrdiff_t>(
                                                std::min(begin + kLeafSize, child.size())
                                            )
                        );
                    }
                } else {
                    next[i].push_back(
                        {std::move(child),
                         job.depth + 1,
                         job.seed * 1664525U + static_cast<uint32_t>(j) + 1013904223U}
                    );
                }
            }
            if (!small.empty()) {
                std::sort(small.begin(), small.end());
                small.erase(std::unique(small.begin(), small.end()), small.end());
                finished[i].push_back(std::move(small));
            }
            Bucket().swap(job.ids);
        };
        if (active.front().depth == 0) {
            // A serialized outer OpenMP region would make the root's tile team
            // nested, forcing the runtime to create workers again on every reset.
            process_bucket(0);
        } else {
            {
                index_detail::ParallelException failure;
#pragma omp parallel for num_threads(std::min(num_workers, active.size())) schedule(dynamic)
                for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(active.size());
                     ++i) {
                    if (failure.failed()) {
                        continue;
                    }
                    try {
                        process_bucket(static_cast<size_t>(i));
                    } catch (...) { failure.capture(); }
                }
                failure.rethrow();
            }
        }
        std::vector<BucketJob> new_active;
        for (size_t i = 0; i < active.size(); ++i) {
            for (auto& leaf : finished[i]) {
                leaves.push_back(std::move(leaf));
            }
            for (auto& job : next[i]) {
                new_active.push_back(std::move(job));
            }
        }
        active = std::move(new_active);
    }
    return leaves;
}
}  // namespace pipnn_impl

// Native PiPNN-style seeding: overlapping leader partitions, local dense kNN,
// directional hash reservoirs, and alpha pruning. Uses existing Eigen/OpenMP.
InitialGraph build_initial_graph(
    const float* data,
    size_t count,
    size_t dim,
    size_t degree,
    MetricType metric_type,
    size_t num_threads
) {
    validate_metric_type(metric_type);
    if (data == nullptr || dim == 0 || degree == 0 || degree >= count ||
        count > buffer::kSearchBufferMaxPointCount || degree % 32 != 0) {
        throw std::invalid_argument(
            "PiPNN seed requires data, positive dim, and a degree multiple of 32 below "
            "count"
        );
    }
    using namespace pipnn_impl;
    const size_t num_workers = resolve_num_threads(num_threads);
    ScratchPool scratch(count, dim, num_workers);
    auto leaves = cluster(data, count, dim, metric_type, num_workers, scratch);
    RowMajorMatrix<float> projections(dim, kHashBits);
    std::mt19937 random(555);
    std::normal_distribution<float> normal;
    for (Eigen::Index i = 0; i < projections.size(); ++i) {
        projections.data()[i] = normal(random);
    }
    RowMajorMatrix<float> sketches(count, kHashBits);
    const size_t sketch_tile = std::min(
        kTileSize, round_up_to_multiple(div_round_up(count, num_workers), kMinTileSize)
    );
    {
        index_detail::ParallelException failure;
#pragma omp parallel for num_threads(                           \
        std::min(num_workers, div_round_up(count, sketch_tile)) \
) schedule(static)
        for (std::ptrdiff_t offset = 0; offset < static_cast<std::ptrdiff_t>(count);
             offset += static_cast<std::ptrdiff_t>(sketch_tile)) {
            if (failure.failed()) {
                continue;
            }
            try {
                const size_t begin = static_cast<size_t>(offset);
                const size_t rows = std::min(sketch_tile, count - begin);
                omp_set_num_threads(1);
                simd::matrix_product(
                    data + begin * dim,
                    projections.data(),
                    sketches.data() + begin * kHashBits,
                    rows,
                    dim,
                    kHashBits
                );
            } catch (...) { failure.capture(); }
        }
        failure.rethrow();
    }
    CandidateTable table(count, degree * 5 / 2);
    {
        index_detail::ParallelException failure;
#pragma omp parallel num_threads(std::min(num_workers, leaves.size()))
        {
            auto work = scratch.worker(static_cast<size_t>(omp_get_thread_num()));
#pragma omp for schedule(dynamic)
            for (std::ptrdiff_t index = 0;
                 index < static_cast<std::ptrdiff_t>(leaves.size());
                 ++index) {
                if (failure.failed()) {
                    continue;
                }
                try {
                    const size_t leaf = static_cast<size_t>(index);
                    const auto& ids = leaves[leaf];
                    if (ids.size() < 2) {
                        continue;
                    }
                    gather(data, dim, ids, work);
                    pairwise(work, ids.size(), dim, metric_type);
                    select_local_neighbors(ids, work);
                    const size_t keep = std::min(kLocalDegree, ids.size() - 1);
                    // Retain the original point order and (distance, ID) tie order
                    // when publishing forward and reverse proposals.
                    for (size_t i = 0; i < ids.size(); ++i) {
                        for (size_t j = 0; j < keep; ++j) {
                            const auto [distance, target] =
                                work.candidates[i * kLocalDegree + j];
                            table.merge(ids[i], target, distance, sketches);
                            table.merge(target, ids[i], distance, sketches);
                        }
                    }
                } catch (...) { failure.capture(); }
            }
        }
        failure.rethrow();
    }
    std::vector<Bucket>().swap(leaves);
    scratch.storage.reset();
    sketches.resize(0, 0);
    // Prune in-place: retain selected IDs in the front of each reservoir row.
    {
        index_detail::ParallelException failure;
#pragma omp parallel num_threads(num_workers)
        {
            std::vector<Candidate> candidates;
            Bucket selected;
#pragma omp for schedule(dynamic)
            for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(count);
                 ++index) {
                if (failure.failed()) {
                    continue;
                }
                try {
                    const size_t i = static_cast<size_t>(index);
                    table.copy_candidates(i, candidates);
                    std::sort(
                        candidates.begin(),
                        candidates.end(),
                        [](const Candidate& a, const Candidate& b) {
                            return a.distance < b.distance ||
                                   (a.distance == b.distance && a.id < b.id);
                        }
                    );
                    selected.clear();
                    for (const auto& candidate : candidates) {
                        bool occluded = false;
                        for (PID accepted : selected) {
                            const float distance = metric_type == METRIC_L2
                                                       ? euclidean_sqr<float>(
                                                             data + accepted * dim,
                                                             data + candidate.id * dim,
                                                             dim
                                                         )
                                                       : dot_product_dis<float>(
                                                             data + accepted * dim,
                                                             data + candidate.id * dim,
                                                             dim
                                                         ) - 1.0F;
                            if (distance <= candidate.distance / kAlpha) {
                                occluded = true;
                                break;
                            }
                        }
                        if (!occluded) {
                            selected.push_back(candidate.id);
                        }
                        if (selected.size() == degree) {
                            break;
                        }
                    }
                    table.retain_ids(i, selected);
                } catch (...) { failure.capture(); }
            }
        }
        failure.rethrow();
    }
    return table.export_graph(num_workers);
}
}  // namespace rabitqlib::symqg::detail
