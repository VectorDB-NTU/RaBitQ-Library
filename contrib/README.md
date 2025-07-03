# Vector Dataset Generator

This module provides a fully functional vector dataset generator specifically designed for building IVF indexes in the RaBitQ-Library.

## Features

- **Multiple Dataset Types**: Supports various generation methods such as random datasets, Gaussian mixture models, spherical clusters, etc.
- **Flexible Parameter Configuration**: Customizable parameters including number of data points, dimension, number of clusters, number of queries, etc.
- **Data Quality Assurance**: Built-in data validation and statistics computation
- **Batch Query Generation**: Automatically generates 10% of data points as normalized query vectors
- **Easy Integration**: Simple interfaces for seamless integration with the `IVF::construct` function

## Main Class and Methods

### `Dataset` Class

The `Dataset` class is now an object-oriented design that encapsulates all dataset information and provides convenient access methods.

```cpp
class Dataset {
public:
    enum class DatasetType {
        Random,
        GaussianMixture,
        SphericalCluster
    };
    
    // Constructor
    Dataset(
        size_t num_points,      // Number of data points
        size_t dim,             // Vector dimension
        size_t num_clusters,    // Number of clusters
        size_t num_queries,     // Number of query vectors
        DatasetType type,       // Dataset generation type
        uint32_t seed = 42      // Random seed
    );
};
```

### Dataset Generation Types

#### 1. Random Dataset (`DatasetType::Random`)

```cpp
auto dataset = Dataset(
    10000,                          // num_points
    128,                            // dim
    10,                             // num_clusters
    1000,                           // num_queries
    Dataset::DatasetType::Random,   // type
    42                              // seed
);
```

**Features**: 
- Cluster centers are randomly distributed in space
- Data points are generated around centroids with Gaussian noise
- Suitable for basic functionality testing

#### 2. Gaussian Mixture Model Dataset (`DatasetType::GaussianMixture`)

```cpp
auto dataset = Dataset(
    50000,                              // num_points
    256,                                // dim
    20,                                 // num_clusters
    5000,                               // num_queries
    Dataset::DatasetType::GaussianMixture,  // type
    42                                  // seed
);
```

**Features**:
- Centroids uniformly distributed in a hypercube
- Each cluster has its own standard deviation
- Data points are evenly distributed across clusters
- Closer to real-world cluster distribution

#### 3. Spherical Cluster Dataset (`DatasetType::SphericalCluster`)

```cpp
auto dataset = Dataset(
    8000,                               // num_points
    64,                                 // dim
    8,                                  // num_clusters
    800,                                // num_queries
    Dataset::DatasetType::SphericalCluster,  // type
    42                                  // seed
);
```

**Features**:
- Cluster centers randomly distributed in space
- Data points are located in spherical regions around centroids
- Different clusters may have different radii
- Useful for testing spatial distribution algorithms

### Data Access Methods

#### Basic Information

```cpp
size_t num_points = dataset.get_num_points();
size_t dim = dataset.get_dim();
size_t num_clusters = dataset.get_num_clusters();
```

#### Data Pointers

```cpp
const float* data_ptr = dataset.get_data_ptr();
const float* centroids_ptr = dataset.get_centroids_ptr();
const PID* cluster_ids_ptr = dataset.get_cluster_ids_ptr();
const float* queries_ptr = dataset.get_queries_ptr();
```

#### Data Validation

```cpp
bool is_valid = dataset.validate_dataset();
```

#### Dataset Statistics

```cpp
std::string stats = dataset.get_dataset_stats();
std::cout << stats << std::endl;
```

## Usage Example

### Basic Usage

```cpp
#include "dataset.h"

using namespace rabitqlib::test;

int main() {
    // Generate dataset with 10% queries
    Dataset dataset(10000, 128, 10, 1000, Dataset::DatasetType::GaussianMixture);
    
    // Validate dataset
    if (!dataset.validate_dataset()) {
        std::cerr << "Dataset generation failed!" << std::endl;
        return -1;
    }
    
    // Access data
    const float* data = dataset.get_data_ptr();
    const float* centroids = dataset.get_centroids_ptr();
    const PID* cluster_ids = dataset.get_cluster_ids_ptr();
    const float* queries = dataset.get_queries_ptr();
    
    // Print statistics
    std::cout << dataset.get_dataset_stats() << std::endl;
    
    return 0;
}
```

### Usage with IVF Index

```cpp
#include "dataset.h"
#include <rabitqlib/index/ivf/ivf.hpp>

using namespace rabitqlib::test;

int main() {
    // Generate test dataset
    Dataset dataset(50000, 256, 20, 5000, Dataset::DatasetType::GaussianMixture);
    
    // Access data pointer
    const float* data_ptr = dataset.get_data_ptr();
    const float* queries_ptr = dataset.get_queries_ptr();
    
    // Create IVF index
    rabitqlib::index::ivf::IVF<float> ivf_index;
    
    // Build the index (adjust according to the actual IVF::construct interface)
    // ivf_index.construct(data_ptr, dataset.get_num_points(), dataset.get_dim(), ...);
    
    // Test queries
    for (size_t i = 0; i < dataset.get_num_queries(); ++i) {
        const float* query = queries_ptr + i * dataset.get_dim();
        // Perform search with query
        // auto results = ivf_index.search(query, k);
    }
    
    return 0;
}
```

### Batch Query Testing

```cpp
#include "dataset.h"

using namespace rabitqlib::test;

int main() {
    Dataset dataset(10000, 128, 10, 1000, Dataset::DatasetType::SphericalCluster);
    
    const float* queries = dataset.get_queries_ptr();
    size_t num_queries = dataset.get_num_queries();
    size_t dim = dataset.get_dim();
    
    // Test each query vector
    for (size_t i = 0; i < num_queries; ++i) {
        const float* query = queries + i * dim;
        
        // Verify query vector is normalized
        float norm = 0.0f;
        for (size_t j = 0; j < dim; ++j) {
            norm += query[j] * query[j];
        }
        norm = std::sqrt(norm);
        
        std::cout << "Query " << i << " norm: " << norm << std::endl;
    }
    
    return 0;
}
```

### Data Quality Analysis

```cpp
auto dataset = Dataset(8000, 64, 8, 800, Dataset::DatasetType::SphericalCluster);

// Analyze value range
const float* data = dataset.get_data_ptr();
float min_val = data[0], max_val = data[0];
for (size_t i = 1; i < dataset.get_num_points() * dataset.get_dim(); ++i) {
    min_val = std::min(min_val, data[i]);
    max_val = std::max(max_val, data[i]);
}

std::cout << "Value range: [" << min_val << ", " << max_val << "]" << std::endl;

// Analyze cluster distribution
std::vector<size_t> cluster_counts(dataset.get_num_clusters(), 0);
const PID* cluster_ids = dataset.get_cluster_ids_ptr();
for (size_t i = 0; i < dataset.get_num_points(); ++i) {
    cluster_counts[cluster_ids[i]]++;
}

for (size_t i = 0; i < dataset.get_num_clusters(); ++i) {
    std::cout << "Cluster " << i << ": " << cluster_counts[i] << " points" << std::endl;
}

// Analyze query distribution
const float* queries = dataset.get_queries_ptr();
std::cout << "Number of queries: " << dataset.get_num_queries() << std::endl;
```

## Parameter Recommendations

### Number of Data Points
- **Small-scale test**: 1,000 - 10,000 points
- **Medium-scale**: 10,000 - 100,000 points
- **Large-scale test**: 100,000+ points

### Vector Dimension
- **Low-dimensional**: 64 - 128 dimensions
- **Medium-dimensional**: 128 - 512 dimensions
- **High-dimensional**: 512+ dimensions

### Number of Clusters
- **Default**: `num_points / 1000`
- **Dense clustering**: `num_points / 500`
- **Sparse clustering**: `num_points / 2000`

### Number of Queries
- **Recommended**: `num_points / 10` (10% of data points)
- **Light testing**: `num_points / 20` (5% of data points)
- **Heavy testing**: `num_points / 5` (20% of data points)

## Notes

1. **Memory Usage**: Large datasets can consume a lot of memory. Ensure your system has enough RAM.
2. **Random Seed**: Use a fixed seed for reproducible results.
3. **Data Validation**: Always call `validate_dataset()` before using the data.
4. **Cluster IDs**: Cluster IDs start from 0, in the range [0, num_clusters - 1]
5. **Query Vectors**: All query vectors are automatically normalized to unit length
6. **Data Layout**: All vectors (data, centroids, queries) are stored in row-major format

## Compilation and Execution

Ensure your project has correctly configured dependencies for RaBitQ-Library. Then compile and run the example:

```bash
g++ -std=c++17 -O3 -mavx2 dataset_example.cpp -o dataset_example
./dataset_example
```

## Extension Ideas

You can extend this dataset generator with the following features:

1. **Add New Dataset Types**: Implement new generation methods by adding new enum values and corresponding generation functions
2. **Custom Distributions**: Modify distribution parameters in existing methods
3. **Data Export**: Add functionality to save datasets to files
4. **Visualization**: Add visualization tools (for low-dimensional data)
5. **Query Generation Strategies**: Implement different query generation strategies (e.g., based on cluster centroids, edge cases, etc.)
