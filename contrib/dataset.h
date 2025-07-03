#pragma once

#include <string>
#include <vector>
#include <random>
#include <memory>
#include <unordered_map>
#include <set>
#include <rabitqlib/defines.hpp>
#include <rabitqlib/utils/space.hpp>

namespace rabitqlib {

/**
 * @brief calculate L2 distance between two vectors, copied from hnswlib/space_l2.h
 * @param pVect1 pointer to the first vector
 * @param pVect2 pointer to the second vector
 * @param qty_ptr pointer to the dimension of the vectors
 * @return L2 distance
 */
static float
L2Sqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    float res = 0;
    for (size_t i = 0; i < qty; i++) {
        float t = *pVect1 - *pVect2;
        pVect1++;
        pVect2++;
        res += t * t;
    }
    return (res);
}

namespace test {

using data_type = rabitqlib::RowMajorArray<float>;


/**
 * @brief normalize a vector
 * @param vec vector to normalize
 * @param dim dimension of the vector
 */
void normalize_vector(float* vec, size_t dim) {
    float norm = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        norm += vec[i] * vec[i];
    }
    norm = std::sqrt(norm);
    if (norm > 1e-6f) {
        for (size_t i = 0; i < dim; ++i) {
            vec[i] /= norm;
        }
    }
}

class Dataset {
public:
    enum class DatasetType {
        Random,
        GaussianMixture,
        SphericalCluster
    };

    Dataset(
        size_t num_points,
        size_t dim,
        size_t num_clusters,
        size_t num_queries,
        DatasetType type = DatasetType::Random,
        uint32_t seed = 42
    ) 
    : data(num_points * dim)
    , centroids(num_clusters * dim)
    , cluster_ids(num_points)
    , queries(num_queries * dim)
    , num_points(num_points)
    , dim(dim)
    , num_clusters(num_clusters)
    , num_queries(num_queries)
    , seed(seed) {
        switch (type) {
            case DatasetType::Random:
                generate_random_dataset();
                break;
            case DatasetType::GaussianMixture:
                generate_gaussian_mixture_dataset();
                break;
            case DatasetType::SphericalCluster:
                generate_spherical_cluster_dataset();
                break;
            default:
                throw std::invalid_argument("Invalid dataset type");
        }
    }
    ~Dataset() = default;

    /**
     * @brief get number of points
     * @return number of points
     */
    size_t get_num_points() {
        return num_points;
    }

    /**
     * @brief get dimension
     * @return dimension
     */
    size_t get_dim() {
        return dim;
    }

    /**
     * @brief get number of clusters
     * @return number of clusters
     */
    size_t get_num_clusters() {
        return num_clusters;
    }

    /**
     * @brief get data pointer
     * @return data pointer
     */
    const float* get_data_ptr() {
        return data.data();
    }

    /**
     * @brief get centroids pointer
     * @return centroids pointer
     */
    const float* get_centroids_ptr() {
        return centroids.data();
    }

    /**
     * @brief get cluster ids pointer
     * @return cluster ids pointer
     */
    const PID* get_cluster_ids_ptr() {
        return cluster_ids.data();
    }

    /**
     * @brief get queries pointer
     * @return queries pointer
     */
    const float* get_queries_ptr() {
        return queries.data();
    }

    /**
     * @brief get number of queries
     * @return number of queries
     */
    size_t get_num_queries() {
        return num_queries;
    }

    /**
     * @brief validate dataset quality
     * @return whether valid
     */
    bool validate_dataset() {
        if (data.empty() || centroids.empty() || cluster_ids.empty()) {
            return false;
        }
        
        if (data.size() != num_points * dim) {
            return false;
        }
        
        if (centroids.size() != num_clusters * dim) {
            return false;
        }
        
        if (cluster_ids.size() != num_points) {
            return false;
        }

        if (queries.size() != num_queries * dim) {
            return false;
        }
        
        // check if cluster ids are in valid range
        for (PID cluster_id : cluster_ids) {
            if (cluster_id >= static_cast<PID>(num_clusters)) {
                return false;
            }
        }
        
        return true;
    }

    /**
     * @brief calculate dataset statistics
     * @return statistics string
     */
    std::string get_dataset_stats() {
        std::string stats = "Dataset Statistics:\n";
        stats += "  Points: " + std::to_string(num_points) + "\n";
        stats += "  Dimension: " + std::to_string(dim) + "\n";
        stats += "  Clusters: " + std::to_string(num_clusters) + "\n";
        
        // calculate number of points per cluster
        std::vector<size_t> cluster_counts(num_clusters, 0);
        for (PID cluster_id : cluster_ids) {
            cluster_counts[cluster_id]++;
        }
        
        stats += "  Points per cluster:\n";
        for (size_t i = 0; i < num_clusters; ++i) {
            stats += "    Cluster " + std::to_string(i) + ": " + 
                     std::to_string(cluster_counts[i]) + "\n";
        }
        
        return stats;
    }

    /**
     * @brief get results
     * @param metric_type metric type, currently only support L2
     * @param topk top k results
     * @return results
     */
    std::unordered_map<PID, std::set<PID>> get_results(MetricType metric_type = METRIC_L2, size_t topk = 10) {
        auto cmp = [](const std::pair<PID, float>& a, const std::pair<PID, float>& b) {
            return a.second < b.second;
        };
        std::priority_queue<std::pair<PID, float>, std::vector<std::pair<PID, float>>, decltype(cmp)> pq(cmp);
        std::unordered_map<PID, std::set<PID>> results;
        for (size_t i = 0; i < num_queries; i++) {
            for (size_t j = 0; j < num_points; j++) {
                float distance = L2Sqr(data.data() + j * dim, queries.data() + i * dim, &dim);
                if (pq.size() < topk) {
                    pq.push(std::make_pair(j, distance));
                } else {
                    if (pq.top().second > distance) {
                        pq.pop();
                        pq.push(std::make_pair(j, distance));
                    }
                }
            }
            auto candidate = std::set<PID>();
            while (!pq.empty()) {
                candidate.insert(pq.top().first);
                pq.pop();
            }
            results.insert(std::make_pair(i, candidate));
        }
        return results;
    }

private:
    /**
     * @brief generate random dataset
     */
    void generate_random_dataset() {
        if (num_clusters == 0) {
            num_clusters = std::max(1UL, num_points / 1000); // default number of clusters
        }
        
        std::mt19937 gen(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        // generate centroids
        centroids.resize(num_clusters * dim);
        for (size_t i = 0; i < num_clusters; ++i) {
            for (size_t j = 0; j < dim; ++j) {
                centroids[i * dim + j] = dist(gen) * 2.0f; // centroids distribution
            }
        }
        
        // generate data points
        data.resize(num_points * dim);
        cluster_ids.resize(num_points);
        
        std::uniform_int_distribution<size_t> cluster_dist(0, num_clusters - 1);
        std::normal_distribution<float> noise_dist(0.0f, 0.3f); // add noise

        for (size_t i = 0; i < num_points; ++i) {
            size_t cluster_id = cluster_dist(gen);
            cluster_ids[i] = static_cast<PID>(cluster_id);
            
            // generate data points based on centroids
            for (size_t j = 0; j < dim; ++j) {
                float centroid_val = centroids[cluster_id * dim + j];
                data[i * dim + j] = centroid_val + noise_dist(gen);
            }
            
            // normalize the vector
            float* vec_ptr = &data[i * dim];
            normalize_vector(vec_ptr, dim);
        }
        
        // normalize centroids
        for (size_t i = 0; i < num_clusters; ++i) {
            float* centroid_ptr = &centroids[i * dim];
            normalize_vector(centroid_ptr, dim);
        }
        
        // generate batch queries (10% of num_points, at least 1)
        num_queries = std::max<size_t>(1, num_points / 10);
        queries.resize(num_queries * dim);
        std::normal_distribution<float> query_dist(0.0f, 1.0f);
        for (size_t i = 0; i < num_queries; ++i) {
            float* qptr = &queries[i * dim];
            for (size_t j = 0; j < dim; ++j) {
                qptr[j] = query_dist(gen);
            }
            normalize_vector(qptr, dim);
        }
        
    }

    /**
     * @brief generate gaussian mixture dataset
     */
    void generate_gaussian_mixture_dataset() {
        if (num_clusters == 0) {
            num_clusters = std::max(1UL, num_points / 1000);
        }
        
        std::mt19937 gen(seed);
        
        // generate centroids (uniformly distributed in hypercube)
        centroids.resize(num_clusters * dim);
        std::uniform_real_distribution<float> centroid_dist(-5.0f, 5.0f);
        
        for (size_t i = 0; i < num_clusters; ++i) {
            for (size_t j = 0; j < dim; ++j) {
                centroids[i * dim + j] = centroid_dist(gen);
            }
        }
        
        // set different standard deviations for each cluster
        std::vector<float> cluster_std(num_clusters);
        std::uniform_real_distribution<float> std_dist(0.5f, 2.0f);
        for (size_t i = 0; i < num_clusters; ++i) {
            cluster_std[i] = std_dist(gen);
        }
        
        // generate data points
        data.resize(num_points * dim);
        cluster_ids.resize(num_points);
        
        // number of points per cluster
        std::vector<size_t> points_per_cluster(num_clusters, num_points / num_clusters);
        size_t remaining = num_points % num_clusters;
        for (size_t i = 0; i < remaining; ++i) {
            points_per_cluster[i]++;
        }
        
        size_t point_idx = 0;
        for (size_t cluster_id = 0; cluster_id < num_clusters; ++cluster_id) {
            std::normal_distribution<float> noise_dist(0.0f, cluster_std[cluster_id]);
            
            for (size_t p = 0; p < points_per_cluster[cluster_id]; ++p) {
                cluster_ids[point_idx] = static_cast<PID>(cluster_id);
                
                for (size_t j = 0; j < dim; ++j) {
                    float centroid_val = centroids[cluster_id * dim + j];
                    data[point_idx * dim + j] = centroid_val + noise_dist(gen);
                }
                
                // normalize the vector
                float* vec_ptr = &data[point_idx * dim];
                normalize_vector(vec_ptr, dim);
                
                point_idx++;
            }
        }
        
        // normalize centroids
        for (size_t i = 0; i < num_clusters; ++i) {
            float* centroid_ptr = &centroids[i * dim];
            normalize_vector(centroid_ptr, dim);
        }
        
        // generate batch queries (10% of num_points, at least 1)
        num_queries = std::max<size_t>(1, num_points / 10);
        queries.resize(num_queries * dim);
        std::normal_distribution<float> query_dist(0.0f, 1.0f);
        for (size_t i = 0; i < num_queries; ++i) {
            float* qptr = &queries[i * dim];
            for (size_t j = 0; j < dim; ++j) {
                qptr[j] = query_dist(gen);
            }
            normalize_vector(qptr, dim);
        }
    }

    /**
     * @brief generate spherical cluster dataset
     */
    void generate_spherical_cluster_dataset() {
        if (num_clusters == 0) {
            num_clusters = std::max(1UL, num_points / 1000);
        }
        
        std::mt19937 gen(seed);
        
        // generate centroids
        centroids.resize(num_clusters * dim);
        std::uniform_real_distribution<float> centroid_dist(-10.0f, 10.0f);
        
        for (size_t i = 0; i < num_clusters; ++i) {
            for (size_t j = 0; j < dim; ++j) {
                centroids[i * dim + j] = centroid_dist(gen);
            }
        }
        
        // generate data points
        data.resize(num_points * dim);
        cluster_ids.resize(num_points);
        
        std::vector<size_t> points_per_cluster(num_clusters, num_points / num_clusters);
        size_t remaining = num_points % num_clusters;
        for (size_t i = 0; i < remaining; ++i) {
            points_per_cluster[i]++;
        }
        
        size_t point_idx = 0;
        for (size_t cluster_id = 0; cluster_id < num_clusters; ++cluster_id) {
            float radius = 1.0f + (cluster_id % 3) * 0.5f; // different cluster radii
            
            for (size_t p = 0; p < points_per_cluster[cluster_id]; ++p) {
                cluster_ids[point_idx] = static_cast<PID>(cluster_id);
                
                // generate random points on the sphere
                std::vector<float> direction(dim);
                float norm = 0.0f;
                
                std::normal_distribution<float> normal_dist(0.0f, 1.0f);
                for (size_t j = 0; j < dim; ++j) {
                    direction[j] = normal_dist(gen);
                    norm += direction[j] * direction[j];
                }
                norm = std::sqrt(norm);
                
                // normalize and add radius
                std::uniform_real_distribution<float> radius_dist(0.0f, radius);
                float r = radius_dist(gen);
                
                for (size_t j = 0; j < dim; ++j) {
                    float centroid_val = centroids[cluster_id * dim + j];
                    data[point_idx * dim + j] = centroid_val + (direction[j] / norm) * r;
                }
                
                // normalize the vector
                float* vec_ptr = &data[point_idx * dim];
                normalize_vector(vec_ptr, dim);
                
                point_idx++;
            }
        }
        
        // normalize centroids
        for (size_t i = 0; i < num_clusters; ++i) {
            float* centroid_ptr = &centroids[i * dim];
            normalize_vector(centroid_ptr, dim);
        }
        
        // generate batch queries (10% of num_points, at least 1)
        num_queries = std::max<size_t>(1, num_points / 10);
        queries.resize(num_queries * dim);
        std::normal_distribution<float> query_dist(0.0f, 1.0f);
        for (size_t i = 0; i < num_queries; ++i) {
            float* qptr = &queries[i * dim];
            for (size_t j = 0; j < dim; ++j) {
                qptr[j] = query_dist(gen);
            }
            normalize_vector(qptr, dim);
        }
    }


private:
    std::vector<float> data;           // raw data
    std::vector<float> centroids;      // centroids
    std::vector<PID> cluster_ids;      // cluster ids
    std::vector<float> queries;        // batch queries
    size_t num_points;                 // number of points
    size_t dim;                        // dimension
    size_t num_clusters;               // number of clusters
    size_t num_queries;                // number of queries
    uint32_t seed;                     // random seed
};

} // namespace test

} // namespace rabitqlib