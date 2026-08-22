# IVF + RaBitQ

[IVF](https://dl.acm.org/doi/10.1109/TPAMI.2010.57) is a classical
clustering-based ANN method. This implementation stores RaBitQ codes and
vector IDs rather than retaining the raw dataset after construction, and uses
[FastScan](https://dl.acm.org/doi/abs/10.1145/3078971.3078992) to estimate a
batch of distances. Its actual memory, latency, and recall depend on the bit
width, number of clusters, and `nprobe`.

The algorithm includes two phases: indexing and querying.

## Index Construction
The first step is to run a clustering algorithm to partition raw data vectors (`*.fvecs` format) into different buckets.
The algorithm performs KMeans clustering on raw vectors based on [Faiss](https://github.com/facebookresearch/faiss) (see `python/ivf.py`).
To run the algorithm, you need to execute the command in `shell`:

```shell
python python/ivf.py  /path/to/raw/data \
                      number_of_clusters \
                      /path/to/output/centroids \
                      /path/to/output/cluster_ids \
                      distance_metric
```
For example, the following command splits the sift vector data into 4096 clusters using Euclidean (l2) distance:
```shell
python python/ivf.py /data/sift/sift_base.fvecs \
                     4096 \
                     /data/sift/sift_centroids_4096_l2.fvecs \
                     /data/sift/sift_clusterids_4096_l2.ivecs \
                     l2
```

After files are prepared, you need to load them into memory:
```c++
using data_type = rabitqlib::RowMajorArray<float>;
using gt_type = rabitqlib::RowMajorArray<uint32_t>;

data_type data;
data_type centroids;
gt_type cids;

rabitqlib::load_vecs<float, data_type>(data_file, data);
rabitqlib::load_vecs<float, data_type>(centroids_file, centroids);
rabitqlib::load_vecs<PID, gt_type>(cids_file, cids);
```

Then initialize an IVF object with the number of data points, vector
dimension, number of clusters, and total bits per dimension. The C++ index
accepts total bit widths from 1 through 9.

```c++
using index_type = rabitqlib::ivf::IVF;

size_t num_points = data.rows();
size_t dim = data.cols();
size_t k = centroids.rows();

index_type ivf(num_points, dim, k, total_bits);
```

Finally, call the construct API:
```c++
void IVF::construct(
    const float* data, 
    const float* centroids, 
    const PID* cluster_ids, 
    bool faster = false,
    size_t num_threads = std::numeric_limits<size_t>::max()
);
```

- **data**: Pointer to the raw data vectors.
- **centroids**: Centroids computed by K-means clustering on the raw data vectors (we recommend to tune cluster_num around 4 * the square root of the dataset following Faiss).
- **cluster_ids**: Array of length `data_num`; every entry must be in the range `[0, cluster_num)`.
- **faster**: If true, enable fast implementations for RaBitQ (By default, it is set as `false` to pursue better accuracy.).
- **num_threads**: Maximum number of OpenMP threads used to quantize clusters.

For example:
```c++
ivf.construct(data.data(), centroids.data(), cids.data(), true);
```

During the construction phase, we quantize each cluster in parallel. For each cluster, we first rotate the centroid and
vectors in this cluster using a random matrix, then compute the 1-bit codes and (total_bits - 1)-bit ex codes along with
corresponding factors.
The raw `data` input is used during construction but is not stored in the
finished IVF index.

After construction, you can directly save the index file to disk:
```c++
ivf.save(outoput_index_file);
```
### Data Layout
The main data layout for our IVF is organized as follows:
```c++
[batch data]    // 1-bit code and factors
[ex_data]       // code for remaining bits
[ids]           // PID of vectors (organized by clusters)
[cluster_lst]   // List of clusters' metadata in IVF
```

## Querying
Currently, querying requires the index to be loaded in memory. If you want to use a previously saved index on the disk,  firstly load it into memory:

```c++
using index_type = rabitqlib::ivf::IVF;
index_type ivf;
ivf.load(index_file);
```
Once the index is loaded, you can call the search function for queries:
```c++
void IVF::search(
    const float* __restrict__ query, 
    size_t k, 
    size_t nprobe, 
    PID* __restrict__ results,
    bool use_hacc
) const;
```

- **query**: Query vector.
- **k**: Top-k.
- **nprobe**: The number of closest clusters to search.
- **results**: Result buffer, size of k.
- **use_hacc**: If use high accuracy FastScan, true by default. For data quantized by high number of bits (e.g., >3), we recommend to use high accuracy FastScan to reduce the error caused by FastScan. Also, user may disable it to improve the query efficiency.

During the search phase, we first rotate the query vector and compute distances between the query vector and the clusters' centroids. Then, we select the n (nprobe) clusters with the smallest distances for search. For each cluster, we first use FastScan to get the coarse distance. Then, if the accuracy of the coarse distance is insufficient, we access the remaining ex bits to boost the accuracy. The search terminates when all selected clusters are scanned and returns the top k nearest neighbours for the given query.
