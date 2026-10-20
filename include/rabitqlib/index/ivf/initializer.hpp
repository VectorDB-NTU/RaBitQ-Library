#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <ios>
#include <memory>
#include <mutex>
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
inline InitializerType resolve_initializer_type(InitializerType type, size_t clusters) {
    switch (type) {
        case InitializerType::Auto:
            if (clusters < 5000) {
                return InitializerType::Flat;
            }
            return clusters < 60000 ? InitializerType::FlatRaBitQ : InitializerType::HNSW;
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
    explicit Initializer(size_t d, size_t k) : dim_(d), num_cluster_(k) {}
    virtual ~Initializer() = 0;
    [[nodiscard]] virtual const float* centroid(PID) const = 0;
    virtual void add_vectors(const float*, size_t) = 0;
    virtual void centroids_distances(
        const float*, size_t, std::vector<AnnCandidate<float>>&
    ) const = 0;
    virtual void load(std::ifstream&, const char*) = 0;
    virtual void save(std::ofstream&, const char*) const = 0;
};
inline Initializer::~Initializer() {}

class FlatInitializer : public Initializer {
   private:
    std::vector<float> centroids_;
    MetricType metric_type_;

   public:
    explicit FlatInitializer(
        size_t d, size_t k, MetricType metric_type = MetricType::METRIC_L2
    )
        : Initializer(d, k), centroids_(num_cluster_ * dim_), metric_type_(metric_type) {
        validate_metric_type(metric_type_);
    }

    ~FlatInitializer() override = default;

    [[nodiscard]] const float* centroid(PID id) const override {
        return &centroids_[id * dim_];
    }

    void add_vectors(const float* cent, size_t) override {
        std::memcpy(centroids_.data(), cent, sizeof(float) * num_cluster_ * dim_);
    }

    void centroids_distances(
        const float* query, size_t nprobe, std::vector<AnnCandidate<float>>& candidates
    ) const override {
        std::vector<AnnCandidate<float>> centroid_dist(this->num_cluster_);
        for (PID i = 0; i < num_cluster_; ++i) {
            centroid_dist[i].id = i;
            centroid_dist[i].distance =
                metric_type_ == METRIC_IP
                    ? dot_product_dis(query, centroid(i), dim_)
                    : std::sqrt(euclidean_sqr(query, centroid(i), dim_));
        }
        std::partial_sort(
            centroid_dist.begin(),
            centroid_dist.begin() + static_cast<long>(nprobe),
            centroid_dist.end()
        );

        std::memcpy(
            candidates.data(), centroid_dist.data(), sizeof(AnnCandidate<float>) * nprobe
        );
    }

    // for flat initer, we save & load into the ifstream
    void save(std::ofstream& output, const char*) const override {
        output.write(
            reinterpret_cast<const char*>(centroids_.data()),
            static_cast<std::streamsize>(sizeof(float) * dim_ * num_cluster_)
        );
    }

    void load(std::ifstream& input, const char*) override {
        input.read(
            reinterpret_cast<char*>(centroids_.data()),
            static_cast<std::streamsize>(sizeof(float) * dim_ * num_cluster_)
        );
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
    explicit FlatRaBitQInitializer(size_t d, size_t k, MetricType metric = METRIC_L2);
    [[nodiscard]] const float* centroid(PID id) const override;
    void add_vectors(const float* centroids, size_t num_threads) override;
    void centroids_distances(
        const float* query, size_t nprobe, std::vector<AnnCandidate<float>>& candidates
    ) const override;
    void save(std::ofstream& output, const char*) const override;
    void load(std::ifstream& input, const char*) override;
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
    int M_ = 16;
    int ef_construction_ = 400;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> alg_hnsw_;
    CentroidL2Space l2_space_;
    CentroidIPSpace ip_space_;
    hnswlib::SpaceInterface<float>* space_;
    MetricType metric_type_;
    std::vector<hnswlib::tableint> label_to_internal_;
    mutable std::mutex search_mutex_;

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
        size_t d, size_t k, MetricType metric_type = MetricType::METRIC_L2
    )
        : Initializer(d, k)
        , l2_space_(d)
        , ip_space_(d)
        , space_(
              metric_type == METRIC_IP
                  ? static_cast<hnswlib::SpaceInterface<float>*>(&ip_space_)
                  : static_cast<hnswlib::SpaceInterface<float>*>(&l2_space_)
          )
        , metric_type_(metric_type) {
        validate_metric_type(metric_type_);
        alg_hnsw_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(
            space_, num_cluster_, M_, ef_construction_
        );
    }

    void add_vectors(const float* cent, size_t num_threads) override {
        size_t start = 0;
        size_t rows = num_cluster_;
        rabitqlib::index_detail::parallel_for(
            start, rows, num_threads, [&](size_t row, size_t /*thread_id*/) {
                alg_hnsw_->addPoint(cent + (row * dim_), row);
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
        std::lock_guard<std::mutex> lock(search_mutex_);
        alg_hnsw_->setEf(std::max(size_t{768}, 2 * nprobe));
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result =
            alg_hnsw_->searchKnn(query, nprobe);
        if (result.size() != nprobe) {
            throw std::runtime_error("HNSW centroid graph returned too few candidates");
        }

        for (size_t i = 0; i < nprobe; ++i) {
            candidates[i].distance = metric_type_ == METRIC_IP
                                         ? result.top().first
                                         : std::sqrt(result.top().first);
            candidates[i].id = result.top().second;
            result.pop();
        }
    }

    // for hnsw initer, we save & load into a separate file by hnswlib
    void save(std::ofstream&, const char* filename) const override {
        std::string hnsw(filename);
        hnsw += ".hnsw";
        alg_hnsw_->saveIndex(hnsw);
    }

    void load(std::ifstream&, const char* filename) override {
        std::string hnsw(filename);
        hnsw += ".hnsw";
        detail::validate_hnsw_sidecar_file(hnsw, num_cluster_, space_->get_data_size());
        alg_hnsw_->loadIndex(hnsw, space_, num_cluster_);
        const auto& graph = *alg_hnsw_;
        if (graph.offsetData_ > graph.label_offset_ ||
            graph.label_offset_ > graph.size_data_per_element_ ||
            graph.label_offset_ - graph.offsetData_ != space_->get_data_size() ||
            graph.size_data_per_element_ - graph.label_offset_ !=
                sizeof(hnswlib::labeltype)) {
            throw std::runtime_error("HNSW centroid layout does not match IVF dimension");
        }
        detail::validate_hnsw_centroid_graph(graph);
        rebuild_label_lookup();
    }

    ~HNSWInitializer() override = default;
};
}  // namespace rabitqlib::ivf
