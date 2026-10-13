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

- **v0.7.0:** Modular C++ components, stable core API v1, optional
  OpenMP for numerical components, and unified index counts. This documentation
  targets 0.7.0; see the [upgrade notes](docs/docs/releases/0.7.0.md).
- **v0.6.0:** Concurrent Python index searches, configurable IVF centroid routing,
  32-coordinate padding, and Eigen 5.0.1. See the
  [upgrade notes](docs/docs/releases/0.6.0.md).
- **October 2026 — v0.5.2:** [HNSW add, resize, and remove](docs/docs/index/hnsw.md#updating-an-index),
  batch search for IVF and SymphonyQG, and clustering and allocation optimizations.
- **September 2026 — Cross-platform support:** C++ builds and CPython 3.11–3.14
  wheels for Linux x86-64/ARM64, Windows x86-64, and macOS 14+ ARM64.

## Install

```bash
python -m pip install --upgrade "rabitqlib>=0.7.0"
```

Wheels cover the platforms above. x86-64 requires AVX2/FMA and optionally uses
AVX-512; ARM64 uses NEON. See [platform requirements and source installation](docs/docs/installation.md#requirements).

## Python quick start

Follow the [complete Python quick start](docs/docs/quick_start.md) to build and
search an IVF index, interpret its results, and save/load it using synthetic data.
Native [clustering](docs/docs/clustering.md) needs no external k-means package.

Then see the [IVF](docs/docs/index/ivf.md), [HNSW](docs/docs/index/hnsw.md), and
[SymphonyQG](docs/docs/index/qg.md) guides, or the
[dataset-based Python examples](sample/python/README.md).
See [threading and save/load paths](docs/docs/index/usage.md#threading-and-file-paths)
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

C++ systems with their own indexes can link `rabitqlib::core` for encoding,
single-vector estimation and rotation, or `rabitqlib::fastscan` for batch scoring.
Configure `RABITQ_BUILD_INDEX=OFF` to omit ANN implementations; also disable
`RABITQ_BUILD_FASTSCAN` for pure core. Set `RABITQ_USE_OPENMP=OFF` for core/FastScan
without an OpenMP dependency; ANN builds retain OpenMP. `rabitqlib::index`
provides indexes and clustering. The 0.7.0 refactor removes the former
`rabitqlib::rabitqlib` target and forwarding headers; see the
[migration guide](docs/docs/modular_migration.md). Optional Python bindings retain
`import rabitqlib`.
For a source-stable, storage-neutral interface with caller-backed workspace, use
[`rabitqlib/core/api.hpp`](docs/docs/rabitq/core_api.md). Its v1 entry points make no heap
allocations on valid calls and keep code and factors separate.
See [modular integration](docs/docs/cpp_integration.md#core-only-c-integration), the
[core example](sample/cpp/core/) and [FastScan example](sample/cpp/fastscan/).
The [core benchmark](benchmarks/core/README.md) runs independently of ANN and Python.

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
AVX2/FMA or an ARM64 CPU. First [get the 0.7.0 source](docs/docs/installation.md#get-the-source),
then build the library and examples from the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Set `-DRABITQ_ENABLE_NATIVE_OPTIMIZATION=OFF` for binaries that will run on
other CPUs. See [platform-specific build instructions](tests/README.md#prerequisites).

### Use RaBitQ-Library in another C++ project

The component targets below are introduced in 0.7.0. Use a matching checkout,
as described in the [migration guide](docs/docs/modular_migration.md). Add that
checkout as a submodule at `third_party/rabitqlib`, pin its revision, and link the
component you need:

```cmake
set(RABITQ_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/rabitqlib)
target_link_libraries(my_program PRIVATE rabitqlib::index)
```

The target propagates C++17, OpenMP, and required public-header definitions.
Optimization and native-tuning options apply only to RaBitQ's own targets;
configure your application's optimization flags separately. Explicit sanitizer
builds propagate instrumentation and runtime linkage for header compatibility.

The [core v1 API](docs/docs/rabitq/core_api.md) has a source-compatibility contract;
other C++ interfaces may evolve. No cross-version C++ binary ABI is promised, so
rebuild consumers when updating the library. For an installed ANN package, use
`find_package(rabitqlib CONFIG REQUIRED COMPONENTS index)` and the same target.
ANN requires OpenMP; core/FastScan can be built with `RABITQ_USE_OPENMP=OFF`.

See [submodule setup and version pinning](docs/docs/cpp_integration.md#use-in-another-c-project),
[C++ installation](docs/docs/cpp_integration.md#install-the-c-library), and the
[build and test guide](tests/README.md) for complete workflows.
Examples: [C++ indexes and quantization](docs/docs/cpp_integration.md#c-examples).
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
