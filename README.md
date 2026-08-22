# RaBitQ Library

[![C++ tests](https://github.com/VectorDB-NTU/RaBitQ-Library/actions/workflows/test.yaml/badge.svg)](https://github.com/VectorDB-NTU/RaBitQ-Library/actions/workflows/test.yaml)
[![Python tests](https://github.com/VectorDB-NTU/RaBitQ-Library/actions/workflows/python.yml/badge.svg)](https://github.com/VectorDB-NTU/RaBitQ-Library/actions/workflows/python.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

RaBitQ Library is a C++17 library with Python bindings for compact, accurate
vector quantization and approximate nearest-neighbor search. It provides:

- the [1-bit](https://arxiv.org/abs/2405.12497) and
  [multi-bit](https://arxiv.org/abs/2409.09913) RaBitQ quantizers;
- IVF, HNSW, and [SymphonyQG](https://dl.acm.org/doi/abs/10.1145/3709730)
  indexes powered by RaBitQ;
- Euclidean distance, inner product, and cosine similarity estimation; and
- optimized AVX2 and AVX-512 kernels with runtime CPU dispatch.

RaBitQ is developed by the
[VectorDB group](https://vectordb-ntu.github.io/) at Nanyang Technological
University, Singapore. A GPU implementation is also available in
[cuvs_rabitq](https://github.com/Stardust-SJF/cuvs_rabitq/tree/cuvs_ivf_rabitq).

## Quick start

### Python

#### Requirements

- Python 3.9 or newer
- a C++17 compiler
- CMake 3.15 or newer
- OpenMP
- an x86-64 CPU with AVX2 and FMA support

AVX2 and FMA are the baseline SIMD requirements. On CPUs that also support
AVX-512F, AVX-512BW, and AVX-512DQ, RaBitQ selects its AVX-512 kernels at
runtime. AVX-512 VPOPCNTDQ provides additional acceleration when available.

On Ubuntu or Debian, install the system build tools and then install RaBitQ
from the repository:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libomp-dev

git clone https://github.com/VectorDB-NTU/RaBitQ-Library.git
cd RaBitQ-Library
python -m pip install .
```

The following complete example builds a small IVF index and searches it. It
uses deterministic synthetic data, so no dataset download is required.

```python
import numpy as np
from rabitqlib import IvfIndex

rng = np.random.default_rng(42)
data = rng.standard_normal((500, 64)).astype(np.float32)
queries = rng.standard_normal((5, 64)).astype(np.float32)

# Assign vectors to five clusters and calculate their centroids.
cluster_ids = (np.arange(len(data)) % 5).astype(np.uint32)
centroids = np.stack(
    [data[cluster_ids == cluster].mean(axis=0) for cluster in range(5)]
).astype(np.float32)

index = IvfIndex(
    dim=64,
    max_elements=len(data),
    num_clusters=5,
    nbits=4,
    metric="l2",
)
index.build(data, centroids, cluster_ids)

ids, distances = index.search(queries, k=10, nprobe=5)
print(ids.shape, distances.shape)  # (5, 10) (5, 10)
print(ids[0])
```

Python bindings are also available for `HnswIndex` and `SymqgIndex`. See the
[Python examples](sample/python/) for index construction, querying, and index
persistence.

### C++

#### Requirements

- CMake 3.10 or newer
- a C++17 compiler with OpenMP support
- an x86-64 CPU with AVX2 and FMA support

AVX2 and FMA are the baseline SIMD requirements. On CPUs that also support
AVX-512F, AVX-512BW, and AVX-512DQ, RaBitQ selects its AVX-512 kernels at
runtime. AVX-512 VPOPCNTDQ provides additional acceleration when available.

Clone and build the library and example programs:

```bash
git clone https://github.com/VectorDB-NTU/RaBitQ-Library.git
cd RaBitQ-Library

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

The index example executables are written to `bin/`. Their source code shows
the complete indexing and querying workflows:

- [IVF + RaBitQ](sample/cpp/ivf_rabitq_indexing.cpp)
- [HNSW + RaBitQ](sample/cpp/hnsw_rabitq_indexing.cpp)
- [SymphonyQG](sample/cpp/symqg_indexing.cpp)

A separate [RaBitQ quantization example](sample/cpp/quantizer.cpp) demonstrates
the lower-level quantizer API; it is provided as source and is not currently a
CMake target.

To build and run the C++ test suite:

```bash
cmake -S . -B build -DRABITQ_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

GoogleTest is downloaded during test configuration. For a full benchmark on
the GIST dataset, see [`example.sh`](example.sh). More detailed API and
algorithm guidance is available in the [documentation](docs/docs/index.md).

## Contributing

Contributions are welcome. See the [contributing guide](CONTRIBUTING.md) for
the build, formatting, pre-commit, and static-analysis workflows.

## Why RaBitQ?

- **High accuracy with tiny codes.** RaBitQ provides state-of-the-art
  similarity estimation across different bit widths and remains effective at
  one bit per dimension.
- **Fast distance estimation.** Its kernels use bitwise operations and
  [FastScan](https://arxiv.org/abs/1704.07355) for efficient search.
- **Theoretical error bounds.** RaBitQ provides an asymptotically optimal error
  bound that can support reliable ordering and reranking.
- **Multiple index trade-offs.** IVF and HNSW prioritize memory efficiency,
  while SymphonyQG uses additional memory to optimize query performance.

In typical workloads, 4-bit, 5-bit, and 7-bit quantization can achieve roughly
90%, 95%, and 99% recall, respectively, without reranking. Actual results
depend on the dataset, index configuration, and search parameters.

## RaBitQ in industry

RaBitQ has been adopted by vector databases, search engines, and libraries:

- [Milvus](https://github.com/milvus-io/milvus) — IVF + RaBitQ (C++)
- [Faiss](https://github.com/facebookresearch/faiss) — IVF + RaBitQ (C++)
- [VSAG](https://github.com/antgroup/vsag) — HGraph + RaBitQ (C++)
- [VectorChord](https://github.com/tensorchord/VectorChord) — IVF + RaBitQ (Rust)
- [Volcengine OpenSearch](https://www.volcengine.com/docs/6465/1553583) — DiskANN + RaBitQ
- [CockroachDB](https://github.com/cockroachdb/cockroach) — CSPANN + RaBitQ (Go)
- [Elasticsearch](https://github.com/elastic/elasticsearch) — HNSW + BBQ, a modified RaBitQ implementation (Java)
- [Lucene](https://github.com/apache/lucene) — HNSW + BBQ, a modified RaBitQ implementation (Java)
- [turbopuffer](https://turbopuffer.com/blog/ann-v3#:~:text=ANN%20v3%20employs%20the%20RaBitQ) — SPFresh + RaBitQ
- [Zvec](https://github.com/alibaba/zvec) — HNSW/IVF + RaBitQ (C++)

## Citation

If RaBitQ helps your research or system, please cite:

> Jianyang Gao, Yutong Gou, Yuexuan Xu, Yongyi Yang, Cheng Long, and Raymond
> Chi-Wing Wong. “Practical and Asymptotically Optimal Quantization of
> High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor
> Search.” SIGMOD 2025. [arXiv:2409.09913](https://arxiv.org/abs/2409.09913).

## Acknowledgements

RaBitQ Library is developed by Yutong Gou, Jianyang Gao, Yuexuan Xu, Jifan Shi,
and Zhonghao Yang. We thank Alexandr Guzhva, Li Liu, Chao Gao, Silu Huang,
Jiabao Jin, Xiaoyao Zhong, and Jinjing Zhou for their valuable feedback.

## License

RaBitQ Library is available under the [Apache License 2.0](LICENSE).
