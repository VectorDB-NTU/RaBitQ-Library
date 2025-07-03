#include "dataset.h"
#include <iostream>
#include <cmath>

using namespace rabitqlib::test;

bool verify_normalization(const float* data, size_t num_points, size_t dim, const std::string& name) {
    std::cout << "Verifying " << name << " normalization..." << std::endl;
    
    bool all_normalized = true;
    float min_norm = 1.0f, max_norm = 1.0f;
    
    for (size_t i = 0; i < num_points; ++i) {
        float norm = 0.0f;
        for (size_t j = 0; j < dim; ++j) {
            norm += data[i * dim + j] * data[i * dim + j];
        }
        norm = std::sqrt(norm);
        
        min_norm = std::min(min_norm, norm);
        max_norm = std::max(max_norm, norm);
        
        if (std::abs(norm - 1.0f) > 1e-5f) {
            std::cout << "  Warning: Vector " << i << " not normalized, norm = " << norm << std::endl;
            all_normalized = false;
        }
    }
    
    std::cout << "  Norm range: [" << min_norm << ", " << max_norm << "]" << std::endl;
    
    if (all_normalized) {
        std::cout << "  ✓ All vectors are properly normalized!" << std::endl;
    } else {
        std::cout << "  ✗ Some vectors are not normalized!" << std::endl;
    }
    
    return all_normalized;
}

int main() {
    std::cout << "=== Testing Dataset Normalization ===" << std::endl;
    
    // Test 1: Random dataset
    std::cout << "\n1. Testing random dataset..." << std::endl;
    auto random_dataset = Dataset::generate_random_dataset(1000, 128, 5, 42);
    const float* random_data = Dataset::get_data_ptr(random_dataset);
    const float* random_centroids = Dataset::get_centroids_ptr(random_dataset);
    
    bool random_ok = verify_normalization(random_data, random_dataset.num_points, random_dataset.dim, "random data");
    bool random_centroids_ok = verify_normalization(random_centroids, random_dataset.num_clusters, random_dataset.dim, "random centroids");
    
    // Test 2: Gaussian mixture dataset
    std::cout << "\n2. Testing gaussian mixture dataset..." << std::endl;
    auto gmm_dataset = Dataset::generate_gaussian_mixture_dataset(1000, 256, 8, 123);
    const float* gmm_data = Dataset::get_data_ptr(gmm_dataset);
    const float* gmm_centroids = Dataset::get_centroids_ptr(gmm_dataset);
    
    bool gmm_ok = verify_normalization(gmm_data, gmm_dataset.num_points, gmm_dataset.dim, "GMM data");
    bool gmm_centroids_ok = verify_normalization(gmm_centroids, gmm_dataset.num_clusters, gmm_dataset.dim, "GMM centroids");
    
    // Test 3: Spherical cluster dataset
    std::cout << "\n3. Testing spherical cluster dataset..." << std::endl;
    auto spherical_dataset = Dataset::generate_spherical_cluster_dataset(1000, 64, 6, 456);
    const float* spherical_data = Dataset::get_data_ptr(spherical_dataset);
    const float* spherical_centroids = Dataset::get_centroids_ptr(spherical_dataset);
    
    bool spherical_ok = verify_normalization(spherical_data, spherical_dataset.num_points, spherical_dataset.dim, "spherical data");
    bool spherical_centroids_ok = verify_normalization(spherical_centroids, spherical_dataset.num_clusters, spherical_dataset.dim, "spherical centroids");
    
    // Summary
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Random dataset: " << (random_ok && random_centroids_ok ? "✓ PASS" : "✗ FAIL") << std::endl;
    std::cout << "GMM dataset: " << (gmm_ok && gmm_centroids_ok ? "✓ PASS" : "✗ FAIL") << std::endl;
    std::cout << "Spherical dataset: " << (spherical_ok && spherical_centroids_ok ? "✓ PASS" : "✗ FAIL") << std::endl;
    
    bool all_passed = random_ok && random_centroids_ok && 
                     gmm_ok && gmm_centroids_ok && 
                     spherical_ok && spherical_centroids_ok;
    
    std::cout << "\nOverall result: " << (all_passed ? "✓ ALL TESTS PASSED" : "✗ SOME TESTS FAILED") << std::endl;
    
    return all_passed ? 0 : 1;
} 