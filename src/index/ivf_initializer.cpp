#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <ios>
#include <limits>
#include <stdexcept>
#include <vector>

#include "rabitqlib/core/metric.hpp"
#include "rabitqlib/fastscan/data_layout.hpp"
#include "rabitqlib/fastscan/encoder.hpp"
#include "rabitqlib/fastscan/estimator.hpp"
#include "rabitqlib/fastscan/pack_codes.hpp"
#include "rabitqlib/fastscan/query.hpp"
#include "rabitqlib/index/detail/buffer.hpp"
#include "rabitqlib/index/detail/parallel_for.hpp"
#include "rabitqlib/index/ivf/initializer.hpp"
#include "rabitqlib/index/types.hpp"
#include "rabitqlib/utils/space.hpp"
#include "rabitqlib/utils/tools.hpp"

namespace rabitqlib::ivf {
FlatRaBitQInitializer::FlatRaBitQInitializer(
    size_t dim, size_t num_clusters, MetricType metric_type
)
    : Initializer(dim, num_clusters), metric_type_(metric_type) {
    validate_metric_type(metric_type);
    const size_t max_size = std::numeric_limits<size_t>::max();
    if (dim == 0 || dim % 32 != 0 || dim > max_size / 32 || num_clusters == 0 ||
        num_clusters > buffer::kSearchBufferMaxPointCount ||
        num_clusters > max_size / sizeof(float) / dim) {
        throw std::invalid_argument("IVF Flat RaBitQ initializer dimensions are invalid");
    }
    const size_t num_batches = div_round_up(num_clusters, fastscan::kBatchSize);
    const size_t batch_bytes = BatchDataMap<float>::data_bytes(dim);
    if (num_batches > max_size / batch_bytes) {
        throw std::invalid_argument(
            "IVF Flat RaBitQ initializer storage exceeds addressable memory"
        );
    }
    centroids_.resize(num_clusters * dim);
    global_centroid_.resize(dim);
    batch_data_.resize(num_batches * batch_bytes);
}

const float* FlatRaBitQInitializer::centroid(PID id) const {
    return centroids_.data() + static_cast<size_t>(id) * dim_;
}

void FlatRaBitQInitializer::add_vectors(const float* centroids, size_t num_threads) {
    std::memcpy(centroids_.data(), centroids, centroids_.size() * sizeof(float));
    for (size_t d = 0; d < dim_; ++d) {
        double sum = 0;
        for (size_t i = 0; i < num_cluster_; ++i) {
            sum += centroids_[i * dim_ + d];
        }
        global_centroid_[d] = static_cast<float>(sum / static_cast<double>(num_cluster_));
    }
    const size_t batch_bytes = BatchDataMap<float>::data_bytes(dim_);
    index_detail::parallel_for(
        0,
        div_round_up(num_cluster_, fastscan::kBatchSize),
        num_threads,
        [&](size_t batch, size_t /*thread*/) {
            const size_t first = batch * fastscan::kBatchSize;
            quant::quantize_split_batch(
                centroids_.data() + first * dim_,
                global_centroid_.data(),
                std::min(fastscan::kBatchSize, num_cluster_ - first),
                dim_,
                0,
                reinterpret_cast<char*>(batch_data_.data()) + batch * batch_bytes,
                nullptr,
                metric_type_
            );
        }
    );
}

void FlatRaBitQInitializer::centroids_distances(
    const float* query, size_t nprobe, std::vector<AnnCandidate<float>>& candidates
) const {
    const size_t num_probes = std::min(nprobe, num_cluster_);
    candidates.resize(num_probes);
    if (num_probes == 0) {
        return;
    }
    const auto exact_distance = [&](PID id) {
        return metric_type_ == METRIC_L2 ? euclidean_sqr(query, centroid(id), dim_)
                                         : dot_product_dis(query, centroid(id), dim_);
    };
    if (num_probes == num_cluster_) {
        for (size_t i = 0; i < num_probes; ++i) {
            candidates[i] = AnnCandidate<float>(static_cast<PID>(i), exact_distance(i));
        }
        std::sort(candidates.begin(), candidates.end());
    } else {
        SplitBatchQuery<float> query_wrapper(query, dim_, 0, metric_type_, false);
        const float norm = std::sqrt(euclidean_sqr(query, global_centroid_.data(), dim_));
        query_wrapper.set_g_add(
            norm,
            metric_type_ == METRIC_IP ? dot_product(query, global_centroid_.data(), dim_)
                                      : 0.0F
        );
        // Use twice the standard RaBitQ error margin for centroid routing:
        // missing a centroid also skips all vectors in its IVF list.
        constexpr float kRoutingErrorScale = 2.0F;
        std::vector<float> cluster_lower_bounds(num_cluster_);
        std::vector<AnnCandidate<float>> cluster_upper_bounds(num_cluster_);
        std::array<float, fastscan::kBatchSize> estimated_distances{}, lower_bounds{},
            ip_x0_qr{};
        const size_t batch_bytes = BatchDataMap<float>::data_bytes(dim_);
        for (size_t first = 0; first < num_cluster_; first += fastscan::kBatchSize) {
            split_batch_estdist(
                reinterpret_cast<const char*>(batch_data_.data()) +
                    (first / fastscan::kBatchSize) * batch_bytes,
                query_wrapper,
                dim_,
                estimated_distances.data(),
                lower_bounds.data(),
                ip_x0_qr.data(),
                false
            );
            const size_t batch_size = std::min(fastscan::kBatchSize, num_cluster_ - first);
            for (size_t j = 0; j < batch_size; ++j) {
                cluster_lower_bounds[first + j] =
                    estimated_distances[j] -
                    kRoutingErrorScale * (estimated_distances[j] - lower_bounds[j]);
                cluster_upper_bounds[first + j] = AnnCandidate<float>(
                    static_cast<PID>(first + j),
                    2 * estimated_distances[j] - cluster_lower_bounds[first + j]
                );
            }
        }
        std::nth_element(
            cluster_upper_bounds.begin(),
            cluster_upper_bounds.begin() + static_cast<std::ptrdiff_t>(num_probes),
            cluster_upper_bounds.end()
        );
        buffer::SearchBuffer<float> result_pool(num_probes);
        // Seed with exact distances: an estimated upper bound need not hold for
        // every vector. Always fill the result before pruning other candidates.
        for (size_t i = 0; i < num_probes; ++i) {
            result_pool.insert(
                cluster_upper_bounds[i].id, exact_distance(cluster_upper_bounds[i].id)
            );
        }
        for (size_t i = num_probes; i < num_cluster_; ++i) {
            const PID id = cluster_upper_bounds[i].id;
            if (cluster_lower_bounds[id] <= result_pool.top_dist()) {
                result_pool.insert(id, exact_distance(id));
            }
        }
        std::copy_n(result_pool.data().begin(), num_probes, candidates.begin());
    }
    if (metric_type_ == METRIC_L2) {
        for (auto& candidate : candidates) {
            candidate.distance = std::sqrt(candidate.distance);
        }
    }
}

void FlatRaBitQInitializer::save(std::ofstream& output, const char* /*filename*/) const {
    output.write(
        reinterpret_cast<const char*>(centroids_.data()),
        static_cast<std::streamsize>(centroids_.size() * sizeof(float))
    );
    output.write(
        reinterpret_cast<const char*>(global_centroid_.data()),
        static_cast<std::streamsize>(global_centroid_.size() * sizeof(float))
    );
    output.write(
        reinterpret_cast<const char*>(batch_data_.data()),
        static_cast<std::streamsize>(batch_data_.size())
    );
    if (!output) {
        throw std::runtime_error("IVF Flat RaBitQ initializer cannot be written");
    }
}

void FlatRaBitQInitializer::load(std::ifstream& input, const char* /*filename*/) {
    input.read(
        reinterpret_cast<char*>(centroids_.data()),
        static_cast<std::streamsize>(centroids_.size() * sizeof(float))
    );
    input.read(
        reinterpret_cast<char*>(global_centroid_.data()),
        static_cast<std::streamsize>(global_centroid_.size() * sizeof(float))
    );
    input.read(
        reinterpret_cast<char*>(batch_data_.data()),
        static_cast<std::streamsize>(batch_data_.size())
    );
    if (!input) {
        throw std::runtime_error("IVF Flat RaBitQ initializer cannot be read");
    }
}
}  // namespace rabitqlib::ivf
