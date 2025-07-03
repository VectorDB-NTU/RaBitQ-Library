#include "dataset.h"
#include <iostream>
#include <rabitqlib/index/ivf/ivf.hpp>
#include <rabitqlib/index/ivf/initializer.hpp>

using namespace rabitqlib::test;
using namespace rabitqlib;

int main() {
    // case1: generate random dataset
    std::cout << "=== generate random dataset ===" << std::endl;
    auto random_dataset = Dataset(10000, 128, 10, 42, Dataset::DatasetType::Random);
    
    if (random_dataset.validate_dataset()) {
        std::cout << "random dataset generated successfully!" << std::endl;
        std::cout << random_dataset.get_dataset_stats() << std::endl;
    }
    
    // case2: generate gaussian mixture dataset
    std::cout << "\n=== generate gaussian mixture dataset ===" << std::endl;
    auto gmm_dataset = Dataset(15000, 256, 15, 123, Dataset::DatasetType::GaussianMixture);
    
    if (gmm_dataset.validate_dataset()) {
        std::cout << "gaussian mixture dataset generated successfully!" << std::endl;
        std::cout << gmm_dataset.get_dataset_stats() << std::endl;
    }
    
    // case3: generate spherical cluster dataset
    std::cout << "\n=== generate spherical cluster dataset ===" << std::endl;
    auto spherical_dataset = Dataset(8000, 64, 8, 456, Dataset::DatasetType::SphericalCluster);
    
    if (spherical_dataset.validate_dataset()) {
        std::cout << "spherical cluster dataset generated successfully!" << std::endl;
        std::cout << spherical_dataset.get_dataset_stats() << std::endl;
    }
    
    // case4: use IVF::construct function
    std::cout << "\n=== use IVF::construct function ===" << std::endl;
    
    // generate test dataset
    auto test_dataset = Dataset(5000, 128, 5, 789, Dataset::DatasetType::GaussianMixture);
    
    // get data pointer
    const float* data_ptr = test_dataset.get_data_ptr();
    const float* centroids_ptr = test_dataset.get_centroids_ptr();
    const PID* cluster_ids_ptr = test_dataset.get_cluster_ids_ptr();
    
    std::cout << "dataset info:" << std::endl;
    std::cout << "  num_points: " << test_dataset.get_num_points() << std::endl;
    std::cout << "  dim: " << test_dataset.get_dim() << std::endl;
    std::cout << "  num_clusters: " << test_dataset.get_num_clusters() << std::endl;
    
    // IVF::construct(data_ptr, test_dataset.num_points, test_dataset.dim, ...);
    
         // case5: validate data quality and normalization
     std::cout << "\n=== validate data quality and normalization ===" << std::endl;
     
     // check data range
     float min_val = data_ptr[0], max_val = data_ptr[0];
     for (size_t i = 1; i < test_dataset.get_num_points() * test_dataset.get_dim(); ++i) {
         min_val = std::min(min_val, data_ptr[i]);
         max_val = std::max(max_val, data_ptr[i]);
     }
     
     std::cout << "data range: [" << min_val << ", " << max_val << "]" << std::endl;
     
     // verify normalization
     std::cout << "verifying normalization..." << std::endl;
     bool all_normalized = true;
     for (size_t i = 0; i < test_dataset.get_num_points(); ++i) {
         float norm = 0.0f;
         for (size_t j = 0; j < test_dataset.get_dim(); ++j) {
             norm += data_ptr[i * test_dataset.get_dim() + j] * data_ptr[i * test_dataset.get_dim() + j];
         }
         norm = std::sqrt(norm);
         
         if (std::abs(norm - 1.0f) > 1e-5f) {
             std::cout << "Warning: Vector " << i << " not normalized, norm = " << norm << std::endl;
             all_normalized = false;
         }
     }
     
     if (all_normalized) {
         std::cout << "All vectors are properly normalized!" << std::endl;
     }
     
     // verify centroid normalization
     std::cout << "verifying centroid normalization..." << std::endl;
     bool centroids_normalized = true;
     for (size_t i = 0; i < test_dataset.get_num_clusters(); ++i) {
         float norm = 0.0f;
         for (size_t j = 0; j < test_dataset.get_dim(); ++j) {
             norm += centroids_ptr[i * test_dataset.get_dim() + j] * centroids_ptr[i * test_dataset.get_dim() + j];
         }
         norm = std::sqrt(norm);
         
         if (std::abs(norm - 1.0f) > 1e-5f) {
             std::cout << "Warning: Centroid " << i << " not normalized, norm = " << norm << std::endl;
             centroids_normalized = false;
         }
     }
     
     if (centroids_normalized) {
         std::cout << "All centroids are properly normalized!" << std::endl;
     }
    
    // check cluster distribution
    std::vector<size_t> cluster_counts(test_dataset.get_num_clusters(), 0);
    for (size_t i = 0; i < test_dataset.get_num_points(); ++i) {
        cluster_counts[cluster_ids_ptr[i]]++;
    }
    
    std::cout << "cluster distribution:" << std::endl;
    for (size_t i = 0; i < test_dataset.get_num_clusters(); ++i) {
        std::cout << "  cluster " << i << ": " << cluster_counts[i] << " points" << std::endl;
    }
    
    return 0;
} 