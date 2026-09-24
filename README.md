<div align="center">

<h1>RaBitQ Library</h1>

<h3>Compact vectors. Accurate distances. Fast ANN search.</h3>

<p>
  A research-backed C++17 library with Python bindings for 1-bit and multi-bit<br>
  vector quantization, IVF, HNSW, and SymphonyQG.
</p>

<p>
  <a href="https://pypi.org/project/rabitqlib/"><img alt="PyPI" src="https://img.shields.io/pypi/v/rabitqlib.svg?cacheSeconds=300"></a>
  <a href="https://pypi.org/project/rabitqlib/"><img alt="Python versions" src="https://img.shields.io/badge/python-3.11--3.14-3776AB.svg?logo=python&amp;logoColor=white"></a>
  <a href="https://vectordb-ntu.github.io/RaBitQ-Library/"><img alt="Documentation" src="https://github.com/VectorDB-NTU/RaBitQ-Library/actions/workflows/docs.yml/badge.svg"></a>
  <a href="https://doi.org/10.1145/3725413"><img alt="Paper DOI" src="https://img.shields.io/badge/DOI-10.1145%2F3725413-blue"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue.svg"></a>
</p>

<p>
  <a href="https://vectordb-ntu.github.io/RaBitQ-Library/">Documentation</a> ·
  <a href="https://pypi.org/project/rabitqlib/">Python package</a> ·
  <a href="https://doi.org/10.1145/3725413">Paper</a> ·
  <a href="https://github.com/VectorDB-NTU/RaBitQ-Library/releases">Releases</a> ·
  <a href="ROADMAP.md">Maintenance</a>
</p>

</div>

> **Contributors welcome!** Help shape RaBitQ by reporting bugs, asking questions,
> suggesting features, or contributing code, tests, documentation, and examples.
> First-time contributors are welcome—[open an issue](https://github.com/VectorDB-NTU/RaBitQ-Library/issues/new/choose)
> or start with our [contribution guide](CONTRIBUTING.md#your-first-contribution)
> and [starter tasks](CONTRIBUTING.md#starter-tasks).

## News

- **September 2026 — Windows x86-64 support:** C++ and Python source builds now
  support MSVC, runtime AVX2/AVX-512 dispatch, and Unicode index paths. Windows
  wheels are available for CPython 3.11–3.14. For source builds, see the
  [Windows build instructions](tests/README.md#prerequisites).

- **September 2026 — IVF raw-vector reranking:** Use `nbits=32` for float32
  reranking. Quantized IVF automatically selects HACC for 4–9-bit codes.
  See the [IVF documentation](docs/docs/index/ivf.md).

- **September 2026 — Quantized SymphonyQG:** Set `quantization_bits=4` or `8`
  for compact vector storage; raw vectors remain the default.
  See the [SymphonyQG documentation](docs/docs/index/qg.md).

## Install

```bash
python -m pip install --upgrade rabitqlib
```

Wheels: CPython 3.11–3.14 on Linux and Windows x86-64, and macOS 14+ ARM64
(Apple Silicon). x86-64 uses AVX2/FMA with optional AVX-512 acceleration;
ARM64 uses NEON and portable scalar kernels. macOS wheels bundle OpenMP;
Intel Mac and universal2 wheels are not provided.

## Python quick start

Build and search a small IVF index using synthetic data:

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

For all three indexes, `build` and `search` interpret `num_threads=0` as the
detected hardware thread count. Larger requests are capped at that count;
smaller positive requests are respected. Operations may use fewer workers when
there are fewer work items. If hardware detection is unavailable, one thread is
used. Omitting `num_threads` in Python still defaults to one thread.

Python bindings are also available for `HnswIndex` and `SymqgIndex`. See the
[Python examples](sample/python/README.md) for index construction, querying, and
index persistence. IVF and HNSW examples run Faiss clustering in a separate process,
then pass saved clusters to RaBitQ indexing so their OpenMP runtimes stay separate.

Index save/load paths are UTF-8 strings on Windows and native path bytes on POSIX
in C++; Python paths are Unicode strings on all platforms.

<details>
<summary>Build the Python bindings from source</summary>

Source builds require a C++17 compiler, CMake 3.20 or newer, and OpenMP. On
Windows, install Visual Studio 2026 with the Desktop development with C++
workload, then run `python -m pip install .` from the repository root.
On Ubuntu or Debian:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libomp-dev
git clone https://github.com/VectorDB-NTU/RaBitQ-Library.git
cd RaBitQ-Library
python -m pip install .
```

</details>

## Choose the right building block

| Component | Best fit | Storage and search profile |
| --- | --- | --- |
| **Quantizer** | Integrating RaBitQ into an existing system | Low-level 1-bit or multi-bit encoding and distance estimation. |
| **IVF** | Memory-efficient partitioned search | Stores quantized codes, or one-bit codes plus raw vectors for reranking. |
| **HNSW** | Graph search with compact vectors | Adds graph links and searches directly from quantized codes. |
| **SymphonyQG** | Fast graph search with a configurable memory/accuracy tradeoff | Uses raw vectors by default, or optional packed 4-bit/8-bit RaBitQ vectors, alongside per-neighborhood quantization data. |

IVF and SymphonyQG use [FastScan](https://arxiv.org/abs/1704.07355) for batched
estimates, while HNSW uses single-code kernels selected for the target architecture.

In typical workloads, 4-bit, 5-bit, and 7-bit quantization can achieve roughly
90%, 95%, and 99% recall, respectively, without reranking. Actual results
depend on the dataset, index configuration, and search parameters.

## Why RaBitQ?

| | |
| --- | --- |
| **Compact by design** | Choose [1-bit](https://doi.org/10.1145/3654970) or [multi-bit](https://doi.org/10.1145/3725413) codes to match your memory and accuracy target. |
| **Accurate estimates** | An asymptotically optimal theoretical error bound supports reliable ordering and reranking. |
| **Native CPU backends** | Runtime AVX2/AVX-512 selection on x86-64; NEON distance, packed-code, FastScan, rotation, and HNSW search kernels on ARM64, with scalar packing and query transposition. |
| **Ready for ANN search** | Use the quantizer directly or build complete IVF, HNSW, and [SymphonyQG](https://dl.acm.org/doi/abs/10.1145/3709730) indexes. |

The library supports Euclidean distance and inner product. Cosine search is
available by normalizing vectors before using inner product.

RaBitQ is developed by the
[VectorDB group](https://vectordb-ntu.github.io/) at Nanyang Technological
University, Singapore. A GPU implementation is also available in
[cuvs_rabitq](https://github.com/Stardust-SJF/cuvs_rabitq/tree/cuvs_ivf_rabitq).

## RaBitQ across the vector-search ecosystem

The projects below illustrate adoption of RaBitQ techniques across vector
search; this is not a list of direct dependencies on RaBitQ-Library.

**Integration story:** [How zvec integrates RaBitQ-Library](docs/docs/integrations/zvec.md)
traces its use of the library's quantizers and estimators inside zvec's IVF
and HNSW implementations, with links to the source code.

<table>
  <tr>
    <td align="center" width="20%">
      <a href="https://github.com/milvus-io/milvus"><img src="https://github.com/milvus-io.png?size=96" width="64" height="64" alt="Milvus logo"><br><strong>Milvus</strong></a>
    </td>
    <td align="center" width="20%">
      <a href="https://github.com/facebookresearch/faiss"><img src="https://github.com/facebookresearch.png?size=96" width="64" height="64" alt="Faiss logo"><br><strong>Faiss</strong></a>
    </td>
    <td align="center" width="20%">
      <a href="https://github.com/NVIDIA/cuvs"><img src="https://github.com/NVIDIA.png?size=96" width="64" height="64" alt="NVIDIA cuVS logo"><br><strong>NVIDIA cuVS</strong></a>
    </td>
    <td align="center" width="20%">
      <a href="https://github.com/microsoft/DiskANN/blob/main/diskann-quantization/src/lib.rs"><img src="https://github.com/microsoft.png?size=96" width="64" height="64" alt="Microsoft DiskANN logo"><br><strong>Microsoft DiskANN</strong></a>
    </td>
    <td align="center" width="20%">
      <a href="https://github.com/antgroup/vsag"><img src="https://github.com/antgroup.png?size=96" width="64" height="64" alt="VSAG logo"><br><strong>VSAG</strong></a>
    </td>
  </tr>
  <tr>
    <td align="center" width="20%">
      <a href="https://github.com/tensorchord/VectorChord"><img src="https://github.com/tensorchord.png?size=96" width="64" height="64" alt="VectorChord logo"><br><strong>VectorChord</strong></a>
    </td>
    <td align="center" width="20%">
      <a href="https://www.volcengine.com/docs/6465/1553583"><img src="https://github.com/volcengine.png?size=96" width="64" height="64" alt="Volcengine OpenSearch logo"><br><strong>Volcengine OpenSearch</strong></a>
    </td>
    <td align="center" width="20%">
      <a href="https://github.com/cockroachdb/cockroach"><img src="https://github.com/cockroachdb.png?size=96" width="64" height="64" alt="CockroachDB logo"><br><strong>CockroachDB</strong></a>
    </td>
    <td align="center" width="20%">
      <a href="https://github.com/elastic/elasticsearch"><img src="https://github.com/elastic.png?size=96" width="64" height="64" alt="Elasticsearch logo"><br><strong>Elasticsearch</strong></a>
    </td>
    <td align="center" width="20%">
      <a href="https://github.com/apache/lucene"><img src="https://github.com/apache.png?size=96" width="64" height="64" alt="Apache Lucene logo"><br><strong>Apache Lucene</strong></a>
    </td>
  </tr>
  <tr>
    <td align="center" width="20%">
      <a href="https://turbopuffer.com/blog/ann-v3#:~:text=ANN%20v3%20employs%20the%20RaBitQ"><img src="https://github.com/turbopuffer.png?size=96" width="64" height="64" alt="turbopuffer logo"><br><strong>turbopuffer</strong></a>
    </td>
    <td align="center" width="20%">
      <a href="https://github.com/alibaba/zvec"><img src="https://github.com/alibaba.png?size=96" width="64" height="64" alt="Zvec logo"><br><strong>Zvec</strong></a>
    </td>
    <td align="center" width="20%">
      <a href="https://docs.lancedb.com/indexing/quantization#rabitq-quantization"><img src="https://github.com/lancedb.png?size=96" width="64" height="64" alt="LanceDB logo"><br><strong>LanceDB</strong></a>
    </td>
    <td align="center" width="20%">
      <a href="https://docs.databricks.com/aws/en/oltp/projects/lakebase-vector"><img src="https://github.com/databricks.png?size=96" width="64" height="64" alt="Databricks logo"><br><strong>Databricks</strong></a>
    </td>
    <td align="center" width="20%">
      <a href="https://clickhouse.com/docs/engines/table-engines/mergetree-family/annindexes#quantized-codecs-methods"><img src="https://github.com/ClickHouse.png?size=96" width="64" height="64" alt="ClickHouse logo"><br><strong>ClickHouse</strong></a>
    </td>
  </tr>
  <tr>
    <td align="center" width="20%">
      <a href="https://qdrant.tech/articles/turboquant-quantization/#1-bit-rabitq-bit-plane-scoring"><img src="https://github.com/qdrant.png?size=96" width="64" height="64" alt="Qdrant logo"><br><strong>Qdrant</strong></a>
    </td>
    <td align="center" width="20%">
      <a href="https://docs.weaviate.io/weaviate/concepts/vector-quantization#rotational-quantization"><img src="https://github.com/weaviate.png?size=96" width="64" height="64" alt="Weaviate logo"><br><strong>Weaviate</strong></a>
    </td>
  </tr>
</table>

## C++ quick start

### Requirements

- CMake 3.20 or newer
- a C++17 compiler with OpenMP support
- an x86-64 CPU with AVX2 and FMA, or an Apple Silicon Mac

Clone and build the library and example programs:

```bash
git clone https://github.com/VectorDB-NTU/RaBitQ-Library.git
cd RaBitQ-Library

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

For MSVC, follow the [Windows build instructions](tests/README.md#prerequisites).
For Apple Silicon source builds, see the [macOS ARM64 instructions](tests/README.md#macos-arm64).
The ARM kernel sources use portable AArch64 intrinsics, but macOS tests do not
establish Linux ARM64 support.
Local GCC/Clang builds enable `-march=native` by default; set
`-DRABITQ_ENABLE_NATIVE_OPTIMIZATION=OFF` for portable binaries, as release
wheels do. See [CPU dispatch details](DEVELOPMENT.md#dispatch-conventions-and-coverage)
for backend requirements and fallbacks.

### Use RaBitQ-Library in another C++ project

The C++ API and ABI are still evolving. For reproducible builds, pin a release
or commit and include RaBitQ-Library as a Git submodule:

```bash
git submodule add https://github.com/VectorDB-NTU/RaBitQ-Library.git third_party/rabitqlib
git submodule update --init --recursive
```

Add the library and link its namespaced target in the consuming project's
`CMakeLists.txt`:

```cmake
set(RABITQ_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/rabitqlib)

target_link_libraries(my_program PRIVATE rabitqlib::rabitqlib)
```

Update the pinned revision deliberately when you are ready to adopt upstream
changes:

```bash
git -C third_party/rabitqlib fetch
git -C third_party/rabitqlib checkout <release-or-commit>
git add third_party/rabitqlib
```

<details>
<summary>Optional: install the C++ library</summary>

Installation is useful for package managers, container images, and shared
server environments. Disable native optimization when the installed library
may run on a different CPU from the build machine:

```bash
cmake -S . -B build \
  -DRABITQ_BUILD_SAMPLES=OFF \
  -DRABITQ_ENABLE_NATIVE_OPTIMIZATION=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build build --parallel
cmake --install build
```

Consume the installed package with:

```cmake
find_package(rabitqlib CONFIG REQUIRED)
target_link_libraries(my_program PRIVATE rabitqlib::rabitqlib)
```

For a non-system prefix, point CMake to the installation when configuring the
consumer:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH="$HOME/.local"
cmake --build build --parallel
```

The [downstream consumer test](tests/consumer/) provides a minimal complete
example of the installed-package workflow.

</details>

Both integration methods require OpenMP on the consuming system.

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

## Citation

If RaBitQ helps your research or system, please cite:

> Jianyang Gao, Yutong Gou, Yuexuan Xu, Yongyi Yang, Cheng Long, and Raymond
> Chi-Wing Wong. “Practical and Asymptotically Optimal Quantization of
> High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor
> Search.” *Proceedings of the ACM on Management of Data* 3, 3, Article 202
> (June 2025), 26 pages. [https://doi.org/10.1145/3725413](https://doi.org/10.1145/3725413).

> Yutong Gou, Jianyang Gao, Yuexuan Xu, and Cheng Long. “SymphonyQG: Towards
> Symphonious Integration of Quantization and Graph for Approximate Nearest
> Neighbor Search.” *Proceedings of the ACM on Management of Data* 3, 1,
> Article 80 (February 2025), 26 pages.
> [https://doi.org/10.1145/3709730](https://doi.org/10.1145/3709730).

> Jianyang Gao and Cheng Long. “RaBitQ: Quantizing High-Dimensional Vectors
> with a Theoretical Error Bound for Approximate Nearest Neighbor Search.”
> *Proceedings of the ACM on Management of Data* 2, 3, Article 167 (May 2024),
> 27 pages. [https://doi.org/10.1145/3654970](https://doi.org/10.1145/3654970).

## Contributing

Contributions are welcome, including documentation and examples. Start with
[your first contribution](CONTRIBUTING.md#your-first-contribution) or choose a
[small starter task](CONTRIBUTING.md#starter-tasks). The guide explains which
build, test, and formatting checks apply to your change.

See [maintenance and feedback](ROADMAP.md) for the current maintainer. Use
[GitHub Issues](https://github.com/VectorDB-NTU/RaBitQ-Library/issues/new/choose)
for bugs, feature requests, and usage or contribution questions.

## Acknowledgements

RaBitQ Library is developed by Yutong Gou, Jianyang Gao, Yuexuan Xu, Jifan Shi,
and Zhonghao Yang. We thank Alexandr Guzhva, Li Liu, Chao Gao, Silu Huang,
Jiabao Jin, Xiaoyao Zhong, and Jinjing Zhou for their valuable feedback.

## License

RaBitQ Library is available under the [Apache License 2.0](LICENSE).
