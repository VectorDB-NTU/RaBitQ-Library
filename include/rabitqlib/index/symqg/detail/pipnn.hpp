#pragma once

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/simd/matrix_dispatch.hpp"
#include "rabitqlib/utils/buffer.hpp"
#include "rabitqlib/utils/space.hpp"
#include "rabitqlib/utils/tools.hpp"

namespace rabitqlib::symqg::detail {

// A temporary graph. No vectors, distances, or clustering scratch survive export.
struct InitialGraph {
    std::vector<size_t> offsets;
    std::vector<PID> neighbors;
};

namespace pipnn_impl {
constexpr size_t kLeafSize = 1024;
constexpr size_t kTileSize = 2048;
constexpr size_t kHashBits = 12;
constexpr size_t kLocalDegree = 2;
constexpr float kAlpha = 1.1F;
using Bucket = std::vector<PID>;

struct Candidate {
    PID id;
    float distance;
    uint16_t hash;
};

struct BucketJob {
    Bucket ids;
    size_t depth = 0;
    uint32_t seed = 555;
};

// Non-owning slices of a single build allocation. Keeping the large matrices out
// of worker heaps avoids retaining their high-water marks after seeding finishes.
struct Workspace {
    float* points;
    float* distances;
    float* norms;
    std::vector<std::pair<float, PID>> candidates;
};

struct ScratchPool {
    size_t point_size;
    size_t distance_size;
    size_t stride;
    std::unique_ptr<float[]> storage;

    ScratchPool(size_t count, size_t dim, size_t threads)
        : point_size(std::min(count, kTileSize) * dim)
        , distance_size(std::max(
              std::min(count, kTileSize) * std::min(count, size_t{512}),
              std::min(count, kLeafSize) * std::min(count, kLeafSize)
          ))
        , stride(point_size + distance_size + std::min(count, kTileSize))
        // Workers initialize their own slices before reading, preserving NUMA
        // first-touch placement without a serial zero-fill of the entire pool.
        , storage(new float[threads * stride]) {}

    Workspace worker(size_t id) {
        float* base = storage.get() + id * stride;
        return {base, base + point_size, base + point_size + distance_size, {}};
    }
};

inline void gather(const float* data, size_t dim, const Bucket& ids, Workspace& work) {
    for (size_t i = 0; i < ids.size(); ++i) {
        std::copy_n(data + ids[i] * dim, dim, work.points + i * dim);
    }
}

inline void pairwise(Workspace& work, size_t size, size_t dim, MetricType metric) {
    simd::pairwise_distances_lower(
        work.points, work.distances, work.norms, size, dim, metric == METRIC_IP
    );
}

inline std::vector<Bucket> partition(
    const float* data,
    size_t dim,
    const BucketJob& job,
    MetricType metric,
    size_t threads,
    ScratchPool& scratch
) {
    const auto& ids = job.ids;
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
        const size_t count = std::min(kTileSize, ids.size() - begin);
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
                    metric == METRIC_L2
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
#pragma omp parallel num_threads(threads)
        {
            auto work = scratch.worker(static_cast<size_t>(omp_get_thread_num()));
#pragma omp for schedule(static)
            for (std::ptrdiff_t offset = 0;
                 offset < static_cast<std::ptrdiff_t>(ids.size());
                 offset += static_cast<std::ptrdiff_t>(kTileSize)) {
                const size_t begin = static_cast<size_t>(offset);
                assign_tile(begin, work);
            }
        }
    } else {
        // Already inside the bucket-level team. Eigen must see that team rather
        // than a serialized nested region when deciding its own parallelism.
        auto work = scratch.worker(static_cast<size_t>(omp_get_thread_num()));
        for (size_t begin = 0; begin < ids.size(); begin += kTileSize) {
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

inline std::vector<Bucket> cluster(
    const float* data,
    size_t count,
    size_t dim,
    MetricType metric,
    size_t threads,
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
#pragma omp parallel for num_threads(active.front().depth == 0 ? 1 : threads) \
    schedule(dynamic)
        for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(active.size());
             ++index) {
            const size_t i = static_cast<size_t>(index);
            auto& job = active[i];
            if (job.ids.size() <= kLeafSize) {
                finished[i].push_back(std::move(job.ids));
                continue;
            }
            auto children = partition(data, dim, job, metric, threads, scratch);
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
                } else if (job.depth >= 4 || (job.depth >= 2 && child.size() * 5 > job.ids.size() * 4)) {
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
inline InitialGraph build_initial_graph(
    const float* data,
    size_t count,
    size_t dim,
    size_t degree,
    MetricType metric = METRIC_L2,
    size_t num_threads = std::numeric_limits<size_t>::max()
) {
    validate_metric_type(metric);
    if (data == nullptr || dim == 0 || degree == 0 || degree >= count ||
        count >= buffer::kSearchBufferMaxPointCount || degree % 32 != 0) {
        throw std::invalid_argument(
            "PiPNN seed requires data, positive dim, and a degree multiple of 32 below "
            "count"
        );
    }
    using namespace pipnn_impl;
    const size_t threads = resolve_num_threads(num_threads);
    ScratchPool scratch(count, dim, threads);
    auto leaves = cluster(data, count, dim, metric, threads, scratch);
    RowMajorMatrix<float> projections(dim, kHashBits);
    std::mt19937 random(555);
    std::normal_distribution<float> normal;
    for (Eigen::Index i = 0; i < projections.size(); ++i) {
        projections.data()[i] = normal(random);
    }
    RowMajorMatrix<float> sketches(count, kHashBits);
#pragma omp parallel for num_threads(threads) schedule(static)
    for (std::ptrdiff_t offset = 0; offset < static_cast<std::ptrdiff_t>(count);
         offset += static_cast<std::ptrdiff_t>(kTileSize)) {
        const size_t begin = static_cast<size_t>(offset);
        const size_t rows = std::min(kTileSize, count - begin);
        simd::matrix_product(
            data + begin * dim,
            projections.data(),
            sketches.data() + begin * kHashBits,
            rows,
            dim,
            kHashBits
        );
    }
    const size_t capacity = degree * 5 / 2;
    std::vector<Candidate> table(count * capacity);
    std::vector<size_t> sizes(count, 0), worst(count, 0);
    std::vector<std::mutex> locks(count);
    const auto merge = [&](PID source, PID target, float distance) {
        uint16_t hash = 0;
        for (size_t bit = 0; bit < kHashBits; ++bit) {
            hash = static_cast<uint16_t>(
                (hash << 1) | (sketches(target, static_cast<Eigen::Index>(bit)) >
                               sketches(source, static_cast<Eigen::Index>(bit)))
            );
        }
        std::lock_guard lock(locks[source]);
        auto* row = table.data() + source * capacity;
        auto& size = sizes[source];
        auto& furthest = worst[source];
        if (size == capacity && distance >= row[furthest].distance) {
            return;
        }
        auto* pos = std::lower_bound(
            row,
            row + size,
            hash,
            [](const Candidate& candidate, uint16_t key) { return candidate.hash < key; }
        );
        const size_t slot = static_cast<size_t>(pos - row);
        if (slot < size && pos->hash == hash) {
            if (distance >= pos->distance) {
                return;
            }
            *pos = {target, distance, hash};
        } else if (size < capacity) {
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
    };
#pragma omp parallel num_threads(threads)
    {
        auto work = scratch.worker(static_cast<size_t>(omp_get_thread_num()));
#pragma omp for schedule(dynamic)
        for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(leaves.size());
             ++index) {
            const size_t leaf = static_cast<size_t>(index);
            const auto& ids = leaves[leaf];
            if (ids.size() < 2) {
                continue;
            }
            gather(data, dim, ids, work);
            pairwise(work, ids.size(), dim, metric);
            for (size_t i = 0; i < ids.size(); ++i) {
                work.candidates.clear();
                for (size_t j = 0; j < ids.size(); ++j) {
                    if (i != j) {
                        work.candidates.emplace_back(
                            work.distances[std::max(i, j) * ids.size() + std::min(i, j)],
                            ids[j]
                        );
                    }
                }
                const size_t keep = std::min(kLocalDegree, work.candidates.size());
                std::partial_sort(
                    work.candidates.begin(),
                    work.candidates.begin() + static_cast<ptrdiff_t>(keep),
                    work.candidates.end()
                );
                for (size_t j = 0; j < keep; ++j) {
                    const auto [distance, target] = work.candidates[j];
                    merge(ids[i], target, distance);
                    merge(target, ids[i], distance);
                }
            }
        }
    }
    std::vector<Bucket>().swap(leaves);
    scratch.storage.reset();
    sketches.resize(0, 0);
    InitialGraph result;
    result.offsets.resize(count + 1, 0);
    // Prune in-place: retain selected IDs in the front of each reservoir row.
#pragma omp parallel num_threads(threads)
    {
        std::vector<Candidate> candidates;
        Bucket selected;
#pragma omp for schedule(dynamic)
        for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(count);
             ++index) {
            const size_t i = static_cast<size_t>(index);
            const auto* row = table.data() + i * capacity;
            candidates.assign(row, row + sizes[i]);
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
                    const float distance =
                        metric == METRIC_L2
                            ? euclidean_sqr<float>(
                                  data + accepted * dim, data + candidate.id * dim, dim
                              )
                            : dot_product_dis<float>(
                                  data + accepted * dim, data + candidate.id * dim, dim
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
            for (size_t j = 0; j < selected.size(); ++j) {
                table[i * capacity + j].id = selected[j];
            }
            result.offsets[i + 1] = selected.size();
        }
    }
    std::partial_sum(result.offsets.begin(), result.offsets.end(), result.offsets.begin());
    result.neighbors.resize(result.offsets.back());
#pragma omp parallel for num_threads(threads)
    for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(count); ++index) {
        const size_t i = static_cast<size_t>(index);
        for (size_t j = 0; j < result.offsets[i + 1] - result.offsets[i]; ++j) {
            result.neighbors[result.offsets[i] + j] = table[i * capacity + j].id;
        }
    }
    return result;
}
}  // namespace rabitqlib::symqg::detail
