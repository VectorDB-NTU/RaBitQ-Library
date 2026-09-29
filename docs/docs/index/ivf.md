# IVF + RaBitQ

[IVF](https://dl.acm.org/doi/10.1109/TPAMI.2010.57) is a classical
clustering-based ANN method. This implementation stores RaBitQ codes and
vector IDs, optionally retaining raw vectors for reranking, and uses
[FastScan](https://dl.acm.org/doi/abs/10.1145/3078971.3078992) to estimate a
batch of distances. Its actual memory, latency, and recall depend on the bit
width, number of clusters, and `nprobe`.

The algorithm includes two phases: indexing and querying.

## Index Construction

Use [QGKMeans](../clustering.md#qgkmeans) to partition the input vectors (`.fvecs`).
Build the C++ examples, then run `bin/qgkmeans` to save centroids and cluster IDs.
For small cluster counts, use `bin/rabitqkmeans` for
[RaBitQKMeans](../clustering.md#rabitqkmeans) flat assignment. Both use exact final
assignment. The optional metric defaults to `l2`; append `ip` for normalized input.

For example, split SIFT into 4,096 clusters using squared L2 distance:

```shell
./bin/qgkmeans /data/sift/sift_base.fvecs 4096 \
    /data/sift/sift_centroids_4096_l2.fvecs \
    /data/sift/sift_clusterids_4096_l2.ivecs
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
accepts total quantized bit widths from 1 through 9, or `total_bits=32`
(Python `nbits=32`) to store raw float32 vectors for reranking. Raw storage
and automatic FastScan precision selection are available starting with version 0.3.3.

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
- **centroids**: Clustering centroids; tune `cluster_num` for the dataset and search budget.
- **cluster_ids**: Array of length `data_num`; every entry must be in the range `[0, cluster_num)`.
- **faster**: If true, enable fast implementations for RaBitQ (By default, it is set as `false` to pursue better accuracy.).
- **num_threads**: Maximum number of OpenMP threads used to quantize clusters.

For example:
```c++
ivf.construct(data.data(), centroids.data(), cids.data(), true);
```

During construction, we process clusters in parallel, rotate their centroids and
vectors, and compute one-bit codes and factors. Quantized mode also computes
`total_bits - 1` extra-bit codes and factors; raw mode stores the original
float32 vectors instead. In raw mode, the index owns a copy of the input, so
the original data can be released after construction.

After construction, you can directly save the index file to disk:
```c++
ivf.save(outoput_index_file);
```
Raw-mode files use a magic/version header and include both the original vectors
and the rotation state. Loading restores the mode automatically, so querying
does not require an external dataset. Existing quantized files retain their
format and remain compatible; versions before 0.3.3 cannot load raw-mode files.

### Data Layout
The main data layout for our IVF is organized as follows:
```c++
[batch data]    // 1-bit code and factors
[ex_data]       // code for remaining bits, or original float32 vectors
[ids]           // PID of vectors (organized by clusters)
[cluster_lst]   // List of clusters' metadata in IVF
```
In raw mode, `ex_data` holds `4 * num_points * dim` bytes in cluster order,
without padding or extra-bit factors. The one-bit filtering codes and factors
are additional storage. `nbits()` (Python `nbits`) reports 32 in raw mode;
the one-bit filtering code is not included in that value.

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
    float* dists = nullptr
) const;
```

- **query**: Query vector.
- **k**: Top-k.
- **nprobe**: The number of closest clusters to search.
- **results**: Result buffer, size of k.
- **dists**: Optional distance buffer, size of `k`.

FastScan precision is selected automatically: HACC for 4–9 quantized bits, and
standard FastScan for 1–3 bits or raw vectors (`32`). HACC reduces lookup-table
error carried into extra-bit distance estimation; raw reranking computes its
final distances directly from floats. Existing C++ overloads with `use_hacc`
and Python's `high_accuracy=True/False` still allow explicit overrides. Python
uses automatic selection when `high_accuracy` is omitted or `None`.

```cpp
ivf.search(query, k, nprobe, ids, distances);         // Automatic
ivf.search(query, k, nprobe, ids, distances, true);   // Force HACC
ivf.search(query, k, nprobe, ids, distances, false);  // Force standard FastScan
```

The IDs-only overloads accept the same override without the `distances`
argument. In Python, call `index.search(queries, k, nprobe)` for automatic
selection, or pass `high_accuracy=True` / `False` to force either mode.
The querying examples also default to automatic selection; Python's
`--use-hacc` flag forces HACC, and the C++ example accepts an optional final
`true` or `false` argument.

During search, we rotate the query and select the `nprobe` closest centroids.
FastScan filters candidates using one-bit codes, then reranking uses either the
remaining quantized bits or the stored raw vectors. Raw reranking computes
squared L2 or `1 - dot(query, vector)` in the original coordinates; the search
API is unchanged. Cluster selection and filtering remain approximate in both
modes. Search returns the top `k` results after scanning the selected clusters.
