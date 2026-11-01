<div align="center">

<p><img src="https://vectordb-ntu.github.io/RaBitQ-Library/assets/logo.svg" alt="RaBitQ Library logo" width="800"></p>

<p>
  <a href="https://pypi.org/project/rabitqlib/"><img alt="PyPI" src="https://img.shields.io/pypi/v/rabitqlib.svg?cacheSeconds=300"></a>
  <a href="https://pypi.org/project/rabitqlib/"><img alt="Python versions" src="https://img.shields.io/badge/python-3.11--3.14-3776AB.svg?logo=python&amp;logoColor=white"></a>
  <a href="https://vectordb-ntu.github.io/RaBitQ-Library/"><img alt="Documentation" src="https://github.com/VectorDB-NTU/RaBitQ-Library/actions/workflows/docs.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue.svg"></a>
</p>

</div>

RaBitQ-Library is a research-backed library for vector quantization and
approximate nearest-neighbor search. See the [research papers](https://vectordb-ntu.github.io/RaBitQ-Library/papers/).

## News

- **v0.7.2:** Python type hints, expanded API documentation, path-like inputs,
  clearer input errors, and concise index representations. Internal quantization
  refactoring and streamlined regression checks preserve existing APIs and index formats.
  See the [release notes](https://vectordb-ntu.github.io/RaBitQ-Library/releases/0.7.2/).
- **v0.7.0:** Modular C++ components, stable core API v1, optional
  OpenMP for numerical components, and unified index counts. See the
  [upgrade notes](https://vectordb-ntu.github.io/RaBitQ-Library/releases/0.7.0/).

## Vector indexes

Use the included indexes when you want construction, search, and save/load in one
library. All three support L2 and inner product; normalize vectors for cosine
search. Their structures, storage costs, and search parameters offer different
QPS–recall tradeoffs.

| Index | Search approach | Storage and updates |
| --- | --- | --- |
| [IVF-RaBitQ](https://vectordb-ntu.github.io/RaBitQ-Library/index/ivf/) | Partitioned search with batch scoring. | 1–9-bit codes or raw-vector reranking; add/remove. |
| [HNSW-RaBitQ](https://vectordb-ntu.github.io/RaBitQ-Library/index/hnsw/) | Graph traversal over quantized vectors. | 1–9-bit codes plus graph edges; add/remove and resize. |
| [SymphonyQG](https://vectordb-ntu.github.io/RaBitQ-Library/index/qg/) | Graph search with batched distance estimation. | Raw or 4/8-bit storage; updates require rebuilding. |

[Native clustering methods](https://vectordb-ntu.github.io/RaBitQ-Library/clustering/) prepare centroids and
assignments for IVF-RaBitQ and HNSW-RaBitQ.

See the [benchmarks](https://vectordb-ntu.github.io/RaBitQ-Library/benchmarks/bench/) for measured results,
settings, and limitations.

## Quick start

### Python

```bash
python -m pip install --upgrade "rabitqlib>=0.7.2"
```

Wheels support CPython 3.11–3.14 on Linux x86-64/ARM64, Windows x86-64,
and macOS 14+ ARM64. x86-64 requires AVX2/FMA and optionally uses
AVX-512; ARM64 uses NEON. See [platform requirements and source installation](https://vectordb-ntu.github.io/RaBitQ-Library/installation/#requirements).

Build and search a small IVF-RaBitQ index with synthetic data; no dataset download is
required. This example demonstrates the API, not performance.

```python
import numpy as np
from rabitqlib import IvfIndex

rng = np.random.default_rng(42)
data = rng.standard_normal((500, 64)).astype(np.float32)
queries = rng.standard_normal((5, 64)).astype(np.float32)

index = IvfIndex(64, 5, nbits=4, metric="l2")
index.train(data)
index.add(data)
ids, distances = index.search(queries, k=10, nprobe=5)
print(ids.shape, distances.shape)  # (5, 10) (5, 10)
```

`train` learns the centroids, and `add` quantizes the vectors into their
clusters. IDs identify added rows; distances here estimate squared L2 distance,
with smaller values first. Quantization remains approximate even when all clusters are probed.
The [complete quick start](https://vectordb-ntu.github.io/RaBitQ-Library/quick_start/) explains result handling,
measures recall while varying `nprobe`, and shows save/load. Code written for
0.7 still runs in 0.8 with deprecation warnings; see the
[migration guide](https://vectordb-ntu.github.io/RaBitQ-Library/python_migration/).

For other indexes, see the [HNSW-RaBitQ](https://vectordb-ntu.github.io/RaBitQ-Library/index/hnsw/#minimal-workflow) and
[SymphonyQG](https://vectordb-ntu.github.io/RaBitQ-Library/index/qg/#minimal-workflow) examples.

### C++

Use the [C++ integration guide](https://vectordb-ntu.github.io/RaBitQ-Library/cpp_integration/) for source builds,
component linking, and installation. Start with the
[C++ index examples](https://vectordb-ntu.github.io/RaBitQ-Library/cpp_integration/#c-examples), or use the
quantization components below in your own engine.

## Quantization components

Integrate encoding and distance estimation into your own storage and search
engine, independently of the included indexes.

| Component | What it provides | Start here |
| --- | --- | --- |
| `rabitqlib::core` | Rotation, per-vector encoding, query preparation, and distance estimation. | [Core example](https://github.com/VectorDB-NTU/RaBitQ-Library/tree/main/sample/cpp/core) · [Stable v1 API](https://vectordb-ntu.github.io/RaBitQ-Library/rabitq/core_api/) |
| `rabitqlib::fastscan` | Batch encoding and lookup-table scoring over 32-vector layouts, built on core. | [FastScan example](https://github.com/VectorDB-NTU/RaBitQ-Library/tree/main/sample/cpp/fastscan) |

See [component integration](https://vectordb-ntu.github.io/RaBitQ-Library/cpp_integration/#core-only-c-integration)
for build options and compatibility, the [quantizer guide](https://vectordb-ntu.github.io/RaBitQ-Library/rabitq/quantizer/)
for encoding choices, and the [core benchmark](https://github.com/VectorDB-NTU/RaBitQ-Library/blob/main/benchmarks/core/README.md) for
component throughput and estimation error.

## Adoption and resources

### Built with RaBitQ-Library

Two examples of direct library integration:

| Project | Integration |
| --- | --- |
| [txtai](https://vectordb-ntu.github.io/RaBitQ-Library/integrations/txtai/) | Uses the Python IVF-RaBitQ and HNSW-RaBitQ indexes as an embedding-search backend. |
| [zvec](https://vectordb-ntu.github.io/RaBitQ-Library/integrations/zvec/) | Uses core quantization in its own IVF-RaBitQ and HNSW-RaBitQ indexes, with FastScan batch scoring in IVF-RaBitQ. |

The linked walkthroughs describe integration boundaries and provide versioned
source references.

### RaBitQ across the vector-search ecosystem

The projects below illustrate adoption of RaBitQ techniques across vector search.
They do not necessarily depend on RaBitQ-Library; direct library integrations are
listed above.

<table>
  <tr>
    <td align="center" width="16%">
      <a href="https://github.com/milvus-io/milvus"><img src="https://github.com/milvus-io.png?size=96" width="40" height="40" alt="Milvus logo"><br>Milvus</a>
    </td>
    <td align="center" width="16%">
      <a href="https://github.com/facebookresearch/faiss"><img src="https://github.com/facebookresearch.png?size=96" width="40" height="40" alt="Faiss logo"><br>Faiss</a>
    </td>
    <td align="center" width="16%">
      <a href="https://github.com/NVIDIA/cuvs"><img src="https://github.com/NVIDIA.png?size=96" width="40" height="40" alt="NVIDIA cuVS logo"><br>NVIDIA cuVS</a>
    </td>
    <td align="center" width="16%">
      <a href="https://github.com/microsoft/DiskANN/blob/main/diskann-quantization/src/lib.rs"><img src="https://github.com/microsoft.png?size=96" width="40" height="40" alt="Microsoft DiskANN logo"><br>Microsoft DiskANN</a>
    </td>
    <td align="center" width="16%">
      <a href="https://github.com/antgroup/vsag"><img src="https://github.com/antgroup.png?size=96" width="40" height="40" alt="VSAG logo"><br>VSAG</a>
    </td>
    <td align="center" width="16%">
      <a href="https://github.com/tensorchord/VectorChord"><img src="https://github.com/tensorchord.png?size=96" width="40" height="40" alt="VectorChord logo"><br>VectorChord</a>
    </td>
  </tr>
  <tr>
    <td align="center" width="16%">
      <a href="https://www.volcengine.com/docs/6465/1553583"><img src="https://github.com/volcengine.png?size=96" width="40" height="40" alt="Volcengine OpenSearch logo"><br>Volcengine OpenSearch</a>
    </td>
    <td align="center" width="16%">
      <a href="https://github.com/cockroachdb/cockroach"><img src="https://github.com/cockroachdb.png?size=96" width="40" height="40" alt="CockroachDB logo"><br>CockroachDB</a>
    </td>
    <td align="center" width="16%">
      <a href="https://github.com/elastic/elasticsearch"><img src="https://github.com/elastic.png?size=96" width="40" height="40" alt="Elasticsearch logo"><br>Elasticsearch</a>
    </td>
    <td align="center" width="16%">
      <a href="https://github.com/apache/lucene"><img src="https://github.com/apache.png?size=96" width="40" height="40" alt="Apache Lucene logo"><br>Apache Lucene</a>
    </td>
    <td align="center" width="16%">
      <a href="https://turbopuffer.com/blog/ann-v3#:~:text=ANN%20v3%20employs%20the%20RaBitQ"><img src="https://github.com/turbopuffer.png?size=96" width="40" height="40" alt="turbopuffer logo"><br>turbopuffer</a>
    </td>
    <td align="center" width="16%">
      <a href="https://github.com/alibaba/zvec"><img src="https://github.com/alibaba.png?size=96" width="40" height="40" alt="Zvec logo"><br>Zvec</a>
    </td>
  </tr>
  <tr>
    <td align="center" width="16%">
      <a href="https://docs.lancedb.com/indexing/quantization#rabitq-quantization"><img src="https://github.com/lancedb.png?size=96" width="40" height="40" alt="LanceDB logo"><br>LanceDB</a>
    </td>
    <td align="center" width="16%">
      <a href="https://docs.databricks.com/aws/en/oltp/projects/lakebase-vector"><img src="https://github.com/databricks.png?size=96" width="40" height="40" alt="Databricks logo"><br>Databricks</a>
    </td>
    <td align="center" width="16%">
      <a href="https://clickhouse.com/docs/engines/table-engines/mergetree-family/annindexes#quantized-codecs-methods"><img src="https://github.com/ClickHouse.png?size=96" width="40" height="40" alt="ClickHouse logo"><br>ClickHouse</a>
    </td>
    <td align="center" width="16%">
      <a href="https://qdrant.tech/articles/turboquant-quantization/#1-bit-rabitq-bit-plane-scoring"><img src="https://github.com/qdrant.png?size=96" width="40" height="40" alt="Qdrant logo"><br>Qdrant</a>
    </td>
    <td align="center" width="16%">
      <a href="https://docs.weaviate.io/weaviate/concepts/vector-quantization#rotational-quantization"><img src="https://github.com/weaviate.png?size=96" width="40" height="40" alt="Weaviate logo"><br>Weaviate</a>
    </td>
    <td align="center" width="16%">
      <a href="https://github.com/pinecone-io/pg_pinecone"><img src="https://github.com/pinecone-io.png?size=96" width="40" height="40" alt="pg_pinecone logo"><br>pg_pinecone</a>
    </td>
  </tr>
</table>

### Further reading

- [Documentation](https://vectordb-ntu.github.io/RaBitQ-Library/) and [algorithm overview](https://vectordb-ntu.github.io/RaBitQ-Library/rabitq/rabitq/).
- [Papers and citations](https://vectordb-ntu.github.io/RaBitQ-Library/papers/) for the research behind the library.
- [Release notes](https://github.com/VectorDB-NTU/RaBitQ-Library/releases) and [0.7 migration guide](https://vectordb-ntu.github.io/RaBitQ-Library/modular_migration/).
- [GPU implementation](https://github.com/Stardust-SJF/cuvs_rabitq/tree/cuvs_ivf_rabitq) in cuVS.

## Contributing

Start with the [contribution guide](https://github.com/VectorDB-NTU/RaBitQ-Library/blob/main/.github/CONTRIBUTING.md#your-first-contribution)
or [starter tasks](https://github.com/VectorDB-NTU/RaBitQ-Library/blob/main/.github/CONTRIBUTING.md#starter-tasks). Report bugs and request
features through [GitHub Issues](https://github.com/VectorDB-NTU/RaBitQ-Library/issues/new/choose).

## Authors and acknowledgements

RaBitQ Library is led and maintained by [Yutong Gou](https://github.com/gouyt13),
and developed with [Jianyang Gao](https://github.com/gaoj0017),
[Yuexuan Xu](https://github.com/YuexuanXu7), [Jifan Shi](https://github.com/Stardust-SJF),
[Zhonghao Yang](https://github.com/zhonghao-yang), [Tamim Ehsan](https://github.com/TamimEhsan),
and [Cheng Long](https://github.com/novaxlong). We thank Alexandr Guzhva, Li Liu, Chao Gao, Silu Huang,
Jiabao Jin, Xiaoyao Zhong, and Jinjing Zhou for their valuable feedback.

## License

RaBitQ Library is available under the [Apache License 2.0](https://github.com/VectorDB-NTU/RaBitQ-Library/blob/main/LICENSE).
