#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ios>
#include <istream>
#include <memory>
#include <ostream>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/index/detail/parallel_for.hpp"
#include "rabitqlib/index/ivf/detail/hnsw_sidecar.hpp"
#include "rabitqlib/index/ivf/types.hpp"  // IWYU pragma: export
#include "rabitqlib/index/types.hpp"
#include "rabitqlib/third/hnswlib/hnswlib.h"
#include "rabitqlib/utils/memory.hpp"
#include "rabitqlib/utils/space.hpp"

namespace rabitqlib::ivf {
inline InitializerType resolve_initializer_type(InitializerType type, size_t num_clusters) {
    // Auto routes through the cheapest initializer that stays fast at this cluster count.
    constexpr size_t kFlatMaxClusters = 5000;
    constexpr size_t kFlatRaBitQMaxClusters = 60000;
    switch (type) {
        case InitializerType::Auto:
            if (num_clusters < kFlatMaxClusters) {
                return InitializerType::Flat;
            }
            return num_clusters < kFlatRaBitQMaxClusters ? InitializerType::FlatRaBitQ
                                                         : InitializerType::HNSW;
        case InitializerType::Flat:
        case InitializerType::FlatRaBitQ:
        case InitializerType::HNSW:
            return type;
    }
    throw std::invalid_argument("Invalid IVF initializer type");
}

/**
 * @brief For IVF centroids, an initializer finds the candidate clusters.
 */
class Initializer {
   protected:
    size_t dim_;
    size_t num_cluster_;

   public:
    explicit Initializer(size_t dim, size_t num_clusters)
        : dim_(dim), num_cluster_(num_clusters) {}
    virtual ~Initializer() = 0;
    [[nodiscard]] virtual const float* centroid(PID id) const = 0;
    virtual void add_vectors(const float* centroids, size_t num_threads) = 0;
    // Resize the output to min(nprobe, number of clusters) and fill it with the
    // nearest centroids in no particular order; the caller need not presize it.
    virtual void centroids_distances(
        const float* query, size_t nprobe, std::vector<AnnCandidate<float>>& candidates
    ) const = 0;
    // Write the routing data that read() restores.
    virtual void write(std::ostream& output) const = 0;
    // Read `bytes` bytes that write() wrote, from the current position of `input`.
    // Data that does not fit this initializer raises std::runtime_error.
    virtual void read(std::istream& input, uint64_t bytes) = 0;
};
inline Initializer::~Initializer() = default;

class FlatInitializer : public Initializer {
   private:
    std::vector<float> centroids_;
    MetricType metric_type_;

   public:
    explicit FlatInitializer(
        size_t dim, size_t num_clusters, MetricType metric_type = METRIC_L2
    )
        : Initializer(dim, num_clusters)
        , centroids_(num_cluster_ * dim_)
        , metric_type_(metric_type) {
        validate_metric_type(metric_type_);
    }

    ~FlatInitializer() override = default;

    [[nodiscard]] const float* centroid(PID id) const override {
        return &centroids_[id * dim_];
    }

    void add_vectors(const float* centroids, size_t /*num_threads*/) override {
        std::memcpy(centroids_.data(), centroids, sizeof(float) * num_cluster_ * dim_);
    }

    void centroids_distances(
        const float* query, size_t nprobe, std::vector<AnnCandidate<float>>& candidates
    ) const override {
        std::vector<AnnCandidate<float>> centroid_candidates(num_cluster_);
        for (PID i = 0; i < num_cluster_; ++i) {
            centroid_candidates[i].id = i;
            centroid_candidates[i].distance =
                metric_type_ == METRIC_IP
                    ? dot_product_dis(query, centroid(i), dim_)
                    : std::sqrt(euclidean_sqr(query, centroid(i), dim_));
        }
        const size_t num_probes = std::min(nprobe, num_cluster_);
        std::partial_sort(
            centroid_candidates.begin(),
            centroid_candidates.begin() + static_cast<std::ptrdiff_t>(num_probes),
            centroid_candidates.end()
        );

        candidates.resize(num_probes);
        std::copy_n(centroid_candidates.begin(), num_probes, candidates.begin());
    }

    // The flat initializer stores only its centroids.
    void write(std::ostream& output) const override {
        output.write(
            reinterpret_cast<const char*>(centroids_.data()),
            static_cast<std::streamsize>(sizeof(float) * dim_ * num_cluster_)
        );
        if (!output) {
            throw std::runtime_error("IVF Flat initializer cannot be written");
        }
    }

    void read(std::istream& input, uint64_t bytes) override {
        if (bytes != sizeof(float) * dim_ * num_cluster_) {
            throw std::runtime_error("IVF Flat initializer data has the wrong size");
        }
        input.read(
            reinterpret_cast<char*>(centroids_.data()), static_cast<std::streamsize>(bytes)
        );
        if (!input) {
            throw std::runtime_error("IVF Flat initializer cannot be read");
        }
    }
};

// Inputs are already in IVF's rotated, padded domain. Persist the codes instead
// of regenerating them on load, so routing keeps the same quantization state.
class FlatRaBitQInitializer : public Initializer {
   private:
    std::vector<float> centroids_;
    std::vector<float> global_centroid_;
    std::vector<std::byte, memory::AlignedAllocator<std::byte, 64, true>> batch_data_;
    MetricType metric_type_;

   public:
    explicit FlatRaBitQInitializer(
        size_t dim, size_t num_clusters, MetricType metric_type = METRIC_L2
    );
    [[nodiscard]] const float* centroid(PID id) const override;
    void add_vectors(const float* centroids, size_t num_threads) override;
    void centroids_distances(
        const float* query, size_t nprobe, std::vector<AnnCandidate<float>>& candidates
    ) const override;
    void write(std::ostream& output) const override;
    void read(std::istream& input, uint64_t bytes) override;
};

// Keep centroid routing on the same runtime-selected distance kernels as flat IVF.
class CentroidL2Space : public hnswlib::SpaceInterface<float> {
   private:
    size_t dim_;

    static float distance(const void* a, const void* b, const void* dim) {
        return euclidean_sqr(
            static_cast<const float*>(a),
            static_cast<const float*>(b),
            *static_cast<const size_t*>(dim)
        );
    }

   public:
    explicit CentroidL2Space(size_t dim) : dim_(dim) {}

    size_t get_data_size() override { return dim_ * sizeof(float); }
    hnswlib::DISTFUNC<float> get_dist_func() override { return distance; }
    void* get_dist_func_param() override { return &dim_; }
};

class CentroidIPSpace : public hnswlib::SpaceInterface<float> {
   private:
    size_t dim_;

    static float distance(const void* a, const void* b, const void* dim) {
        return dot_product_dis(
            static_cast<const float*>(a),
            static_cast<const float*>(b),
            *static_cast<const size_t*>(dim)
        );
    }

   public:
    explicit CentroidIPSpace(size_t dim) : dim_(dim) {}

    size_t get_data_size() override { return dim_ * sizeof(float); }
    hnswlib::DISTFUNC<float> get_dist_func() override { return distance; }
    void* get_dist_func_param() override { return &dim_; }
};

class HNSWInitializer : public Initializer {
   private:
    // Fixed hnswlib graph parameters for centroid routing.
    static constexpr size_t kM = 16;
    static constexpr size_t kEfConstruction = 400;
    static constexpr size_t kMinSearchEf = 768;

    std::unique_ptr<hnswlib::HierarchicalNSW<float>> alg_hnsw_;
    CentroidL2Space l2_space_;
    CentroidIPSpace ip_space_;
    hnswlib::SpaceInterface<float>* space_;
    MetricType metric_type_;
    std::vector<hnswlib::tableint> label_to_internal_;

    void rebuild_label_lookup() {
        if (alg_hnsw_->cur_element_count != num_cluster_) {
            throw std::runtime_error(
                "HNSW centroid count does not match IVF cluster count"
            );
        }
        std::vector<hnswlib::tableint> label_to_internal(num_cluster_);
        std::vector<bool> seen(num_cluster_, false);
        for (size_t internal_id = 0; internal_id < num_cluster_; ++internal_id) {
            const auto label =
                alg_hnsw_->getExternalLabel(static_cast<hnswlib::tableint>(internal_id));
            if (label >= num_cluster_ || seen[label]) {
                throw std::runtime_error("Invalid HNSW centroid label");
            }
            label_to_internal[label] = static_cast<hnswlib::tableint>(internal_id);
            seen[label] = true;
        }
        label_to_internal_ = std::move(label_to_internal);
    }

   public:
    explicit HNSWInitializer(
        size_t dim, size_t num_clusters, MetricType metric_type = METRIC_L2
    )
        : Initializer(dim, num_clusters)
        , l2_space_(dim)
        , ip_space_(dim)
        , space_(
              metric_type == METRIC_IP
                  ? static_cast<hnswlib::SpaceInterface<float>*>(&ip_space_)
                  : static_cast<hnswlib::SpaceInterface<float>*>(&l2_space_)
          )
        , metric_type_(metric_type) {
        validate_metric_type(metric_type_);
        alg_hnsw_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(
            space_, num_cluster_, kM, kEfConstruction
        );
    }

    void add_vectors(const float* centroids, size_t num_threads) override {
        index_detail::parallel_for(
            0, num_cluster_, num_threads, [&](size_t row, size_t /*thread*/) {
                alg_hnsw_->addPoint(centroids + (row * dim_), row);
            }
        );
        rebuild_label_lookup();
    }

    [[nodiscard]] const float* centroid(PID id) const override {
        return reinterpret_cast<const float*>(
            alg_hnsw_->getDataByInternalId(label_to_internal_.at(id))
        );
    }

    void centroids_distances(
        const float* query, size_t nprobe, std::vector<AnnCandidate<float>>& candidates
    ) const override {
        const size_t num_probes = std::min(nprobe, num_cluster_);
        candidates.resize(num_probes);
        if (num_probes == 0) {
            return;
        }
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result =
            alg_hnsw_->searchKnnWithEf(
                query, num_probes, std::max(kMinSearchEf, 2 * num_probes)
            );
        if (result.size() != num_probes) {
            throw std::runtime_error("HNSW centroid graph returned too few candidates");
        }

        for (size_t i = 0; i < num_probes; ++i) {
            candidates[i].distance = metric_type_ == METRIC_IP
                                         ? result.top().first
                                         : std::sqrt(result.top().first);
            candidates[i].id = result.top().second;
            result.pop();
        }
    }

    // The HNSW initializer stores its graph in hnswlib's own layout, which earlier
    // releases wrote to the sidecar file <index>.hnsw.
    void write(std::ostream& output) const override { alg_hnsw_->saveIndex(output); }

    void read(std::istream& input, uint64_t bytes) override {
        detail::validate_hnsw_graph(input, bytes, num_cluster_, space_->get_data_size());
        alg_hnsw_->loadIndex(input, static_cast<size_t>(bytes), space_, num_cluster_);
        const auto& graph = *alg_hnsw_;
        // validate_hnsw_graph already checked these header fields, and loadIndex reads
        // them again. Recheck the loaded values so that data changed in between cannot
        // give centroid() an out-of-record offset.
        if (graph.offsetData_ > graph.label_offset_ ||
            graph.label_offset_ > graph.size_data_per_element_ ||
            graph.label_offset_ - graph.offsetData_ != space_->get_data_size() ||
            graph.size_data_per_element_ - graph.label_offset_ !=
                sizeof(hnswlib::labeltype)) {
            throw std::runtime_error("HNSW centroid graph changed while loading");
        }
        detail::validate_hnsw_centroid_graph(graph);
        rebuild_label_lookup();
    }

    ~HNSWInitializer() override = default;
};
}  // namespace rabitqlib::ivf
