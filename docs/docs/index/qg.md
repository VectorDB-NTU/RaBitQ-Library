# QG + RaBitQ (SymphonyQG)

[QG](https://medium.com/@masajiro.iwasaki/fusion-of-graph-based-indexing-and-product-quantization-for-ann-search-7d1f0336d0d0)
is a graph-based index originating from the
[NGT library](https://github.com/yahoojapan/NGT). This implementation comes
from the [SymphonyQG](https://dl.acm.org/doi/abs/10.1145/3709730) project. For
each vertex it stores the raw vector, a fixed-size neighbor list, and batched
one-bit RaBitQ data for those neighbors. This layout uses more memory than the
raw vectors alone, but lets graph traversal estimate a group of neighbor
distances with FastScan while computing exact distances for visited vertices.

Memory and performance depend on the dimension, degree, build window, and
search window. See `sample/cpp/symqg_indexing.cpp` and
`sample/cpp/symqg_querying.cpp` for complete programs.

## Index Construction

We build the QG by iteratively refining the graph structure.
Since the QG is more complicated than other indices, we need a QGBuilder to help us construct the index.

At the beginning, we need to intialize a QG and a QGBuilder by following construtor.
```cpp
QuantizedGraph::QuantizedGraph(
        size_t num,
        size_t dim,
        size_t max_deg,
        MetricType metric_type = METRIC_L2,
        RotatorType rotator_type = RotatorType::FhtKacRotator
    );

QGBuilder::QGBuilder(
        QuantizedGraph<float>& index,
        uint32_t ef_build,
        const float* data,
        size_t num_threads = std::numeric_limits<size_t>::max()
    )
```
- **num**: Number of vertices (vectors) in the dataset.  
- **dim**: Dimension of the dataset.  
- **max_deg**: Degree bound of QG, must be a multiple of 32.  
- **index**: Previously initialized QG.  
- **ef_build**: Search window size during indexing.  
- **data**: Pointer to the dataset, size of num * dim.  
- **num_threads**: Number of threads to use (default: std::numeric_limits<size_t>::max(), which auto-selects).  
```cpp
size_t rows = 1000000;
size_t cols = 128;
size_t degree = 32;
size_t ef = 200;

std::vector<float> data(rows * cols); // populate with the dataset

QuantizedGraph<float> qg(rows, cols, degree);

QGBuilder builder(qg, ef, data.data());
```

Then, we can use the builder to construct the index. Then we can save the index.
```cpp
builder.build();    // build index interatively

const char* index_file = "./qg_example.index";
qg.save(index_file);    // save index
```

### Data Layout

Each indexed element is stored in the following layout.
```
[Raw data vector]
[Batch data for QG]
[Edges]
```

`Batch data for QG` contains one-bit codes and estimator factors for the
element's neighbors, organized in FastScan batches of 32. Consequently,
`max_deg` must be a multiple of 32.

## Querying

For querying, code is pretty simple.
```cpp
void QuantizedGraph::search(
    const T* __restrict__ query, 
    uint32_t k, 
    uint32_t* __restrict__ results);
```
- **query**: Query vector.  
- **k**: Top-k.  
- **results**: Result buffer, size of k.  
Then we can use a pre-constructed index to search.
```cpp
QuantizedGraph<float> qg;
qg.load("./qg_example.index"); // load pre-constructed index


size_t ef = 100;
size_t topk = 10;
std::vector<PID> results(topk); // result buffer
std::vector<float> query(cols); // populate with a query vector

qg.set_ef(ef);  // set search window size
qg.search(query.data(), topk, results.data());
```
