<div align="center">

<h1>RaBitQ Library</h1>

<h3>Compact vectors. Accurate distances. Fast search.</h3>

<p>
  A research-backed C++17 library with Python bindings for RaBitQ<br>
  vector quantization, clustering, and indexes including IVF, HNSW and SymphonyQG.
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

## News

- **October 2026 — v0.5.2:** [HNSW add, resize, and remove](docs/docs/index/hnsw.md#updating-an-index),
  batch search for IVF and SymphonyQG, and clustering and allocation optimizations.
- **September 2026 — Cross-platform support:** C++ builds and CPython 3.11–3.14
  wheels for Linux x86-64/ARM64, Windows x86-64, and macOS 14+ ARM64.

## Install

```bash
python -m pip install --upgrade "rabitqlib>=0.5.2"
```

Wheels cover the platforms above. x86-64 requires AVX2/FMA and optionally uses
AVX-512; ARM64 uses NEON. See [platform requirements and source installation](docs/docs/quick_start.md#requirements).

## Python quick start

Build and search a small IVF index using synthetic data:

```python
import numpy as np
from rabitqlib import FinalAssignmentMode, IvfIndex, RaBitQKMeans

rng = np.random.default_rng(42)
data = rng.standard_normal((500, 64)).astype(np.float32)
queries = rng.standard_normal((5, 64)).astype(np.float32)

clustering = RaBitQKMeans(
    64, 5, num_threads=2, final_assignment=FinalAssignmentMode.Exact
)
clustering.train(data)

index = IvfIndex(
    dim=64,
    max_elements=len(data),
    num_clusters=5,
    nbits=4,
    metric="l2",
)
index.build(data, clustering.centroids, clustering.assignments)

ids, distances = index.search(queries, k=10, nprobe=5)
print(ids.shape, distances.shape)  # (5, 10) (5, 10)
print(ids[0])
```

See the [quick start](docs/docs/quick_start.md) for index updates and
[Python examples](sample/python/README.md) for all three indexes. Native
[clustering](docs/docs/clustering.md) needs no external k-means package.
See [threading and save/load paths](docs/docs/quick_start.md#threading-and-file-paths)
for runtime conventions.

## Choose the right building block

| Component | Use case |
| --- | --- |
| [Quantizer](docs/docs/rabitq/quantizer.md) | Integrate 1-bit or multi-bit encoding and distance estimation into your system. |
| [IVF](docs/docs/index/ivf.md) | Memory-efficient partitioned search, with optional raw-vector reranking. |
| [HNSW](docs/docs/index/hnsw.md) | Graph search directly over compact quantized vectors. |
| [SymphonyQG](docs/docs/index/qg.md) | Fast graph search with raw or packed 4-bit/8-bit vector storage. |

IVF and HNSW support adding and removing vectors; HNSW also supports explicit
capacity resizing. See their guides for update costs and file compatibility.
The library supports L2 and inner product; normalize vectors for cosine search.

## Why RaBitQ?

- **Compact codes:** Choose 1-bit or multi-bit quantization for your memory and accuracy needs.
- **Accurate estimates:** An asymptotically optimal error bound supports distance estimation.
- **Native CPU acceleration:** Runtime AVX2/AVX-512 dispatch on x86-64 and NEON on ARM64.

Developed by the [VectorDB group](https://vectordb-ntu.github.io/) at Nanyang
Technological University. For GPU support, see [cuvs_rabitq](https://github.com/Stardust-SJF/cuvs_rabitq/tree/cuvs_ivf_rabitq).

## RaBitQ across the vector-search ecosystem

The projects below illustrate adoption of RaBitQ techniques across vector
search; this is not a list of direct dependencies on RaBitQ-Library.

[txtai](https://github.com/neuml/txtai) uses the library as an ANN backend
([configuration](https://github.com/neuml/txtai/blob/master/docs/embeddings/configuration/ann.md#rabitq)).
Read [how zvec integrates RaBitQ-Library](docs/docs/integrations/zvec.md) for
another integration example.

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

Requires CMake 3.20+, a C++17 compiler with OpenMP, and an x86-64 CPU with
AVX2/FMA or an ARM64 CPU. Build the library and examples:

```bash
git clone https://github.com/VectorDB-NTU/RaBitQ-Library.git
cd RaBitQ-Library
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Set `-DRABITQ_ENABLE_NATIVE_OPTIMIZATION=OFF` for binaries that will run on
other CPUs. See [platform-specific build instructions](tests/README.md#prerequisites).

### Use RaBitQ-Library in another C++ project

Add the repository as a submodule at `third_party/rabitqlib`, pin a release or
commit, and link its CMake target:

```cmake
set(RABITQ_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/rabitqlib)
target_link_libraries(my_program PRIVATE rabitqlib::rabitqlib)
```

The C++ API and ABI are evolving; update the pinned revision deliberately.
For installed packages, use `find_package(rabitqlib CONFIG REQUIRED)` and the
same target. Both approaches require OpenMP.

See [submodule setup and version pinning](docs/docs/quick_start.md#use-in-another-c-project),
[C++ installation](docs/docs/quick_start.md#install-the-c-library), and the
[build and test guide](tests/README.md) for complete workflows.
Examples: [C++ indexes and quantization](docs/docs/quick_start.md#c-examples).
Benchmarks: [FAISS clustering comparison](docs/docs/clustering.md#compare-with-faiss)
and [GIST workflow](example.sh).

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

Start with the [contribution guide](CONTRIBUTING.md#your-first-contribution)
or [starter tasks](CONTRIBUTING.md#starter-tasks). Report bugs and request
features through [GitHub Issues](https://github.com/VectorDB-NTU/RaBitQ-Library/issues/new/choose).
See [maintenance and feedback](ROADMAP.md) for maintainer information.

## Acknowledgements

RaBitQ Library is developed by Yutong Gou, Jianyang Gao, Yuexuan Xu, Jifan Shi,
and Zhonghao Yang. We thank Alexandr Guzhva, Li Liu, Chao Gao, Silu Huang,
Jiabao Jin, Xiaoyao Zhong, and Jinjing Zhou for their valuable feedback.

## License

RaBitQ Library is available under the [Apache License 2.0](LICENSE).
