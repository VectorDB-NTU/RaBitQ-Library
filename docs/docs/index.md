# Compact vector search with RaBitQ

<div class="hero" markdown>

**RaBitQ Library** is a C++17 library with Python bindings for compact,
accurate vector quantization and approximate nearest-neighbor search.

Build with the low-level quantizer or use complete IVF, HNSW, and SymphonyQG
indexes backed by optimized AVX2 and AVX-512 kernels.

[Get started](quick_start.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/VectorDB-NTU/RaBitQ-Library){ .md-button }
[Install from PyPI](https://pypi.org/project/rabitqlib/){ .md-button }

</div>

<div class="feature-grid">
  <div class="feature-card">
    <h2>Compact by design</h2>
    <p>Use RaBitQ as an alternative to binary or scalar quantization, with
    useful estimates from a one-bit code per padded dimension plus a small
    set of per-vector factors.</p>
  </div>
  <div class="feature-card">
    <h2>Fast on modern CPUs</h2>
    <p>Runtime dispatch selects optimized AVX2 or AVX-512 kernels. IVF and
    SymphonyQG use FastScan for batched distance estimation.</p>
  </div>
  <div class="feature-card">
    <h2>Ready for vector search</h2>
    <p>Choose IVF, HNSW, or SymphonyQG to balance memory, indexing cost,
    latency, and recall for your workload.</p>
  </div>
</div>

## Accuracy at a glance

![RaBitQ estimation error benchmark across MSong, YouTube, OpenAI embeddings, Word2Vec, and GIST](assets/img/acc_bench.png)

*Average and maximum relative estimation error across six datasets; lower is
better. Results from the
[SIGMOD camera-ready paper](https://doi.org/10.1145/3725413).*

## Start with Python

Install the latest release from PyPI:

```bash
python -m pip install rabitqlib
```

Build an IVF index and search a batch of queries:

```python
import numpy as np
from rabitqlib import IvfIndex

rng = np.random.default_rng(42)
data = rng.standard_normal((500, 64)).astype(np.float32)
queries = rng.standard_normal((5, 64)).astype(np.float32)

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
```

[Continue to the complete quick start](quick_start.md){ .md-button .md-button--primary }

## Choose an index

| Index | Best fit | Typical relative memory | Main search control |
| --- | --- | --- | --- |
| [IVF + RaBitQ](index/ivf.md) | Large datasets and predictable memory use | Lowest | Number of probed clusters |
| [HNSW + RaBitQ](index/hnsw.md) | General-purpose graph search | Moderate | Search candidate list size |
| [SymphonyQG](index/qg.md) | Latency-focused graph search | Highest | Search window size |

IVF and HNSW store quantized vectors instead of accessing raw vectors during
search. SymphonyQG uses additional memory and multiple codes per vector to
optimize its access pattern.

These are typical relative profiles, not fixed guarantees. Actual memory,
latency, and recall depend on vector dimension, quantization width, graph
degree, and search parameters.

## Why RaBitQ?

- **High accuracy with tiny codes.** RaBitQ provides strong similarity
  estimates across different bit widths and remains effective with a one-bit
  code per padded dimension plus per-vector factors.
- **Fast distance estimation.** IVF and SymphonyQG use
  [FastScan](https://arxiv.org/abs/1704.07355) for batched estimates; HNSW
  uses single-code AVX kernels.
- **Theoretical error bounds.** An asymptotically optimal error bound supports
  reliable ordering and reranking.
- **Multiple integration points.** Use the quantizer directly or select a
  complete vector-search index.

The library supports Euclidean distance and inner product. Cosine similarity
can be implemented by normalizing vectors and using inner product.
It implements the [1-bit RaBitQ](https://arxiv.org/abs/2405.12497) and
[multi-bit RaBitQ](https://doi.org/10.1145/3725413) research from the
[VectorDB Group](https://vectordb-ntu.github.io/) at Nanyang Technological
University.

## Used across the vector-search ecosystem

RaBitQ has been adopted by projects including
[Milvus](https://github.com/milvus-io/milvus),
[Faiss](https://github.com/facebookresearch/faiss),
[VSAG](https://github.com/antgroup/vsag),
[VectorChord](https://github.com/tensorchord/VectorChord),
[Volcengine OpenSearch](https://www.volcengine.com/docs/6465/1553583),
[CockroachDB](https://github.com/cockroachdb/cockroach),
[Elasticsearch](https://github.com/elastic/elasticsearch),
[Lucene](https://github.com/apache/lucene),
[turbopuffer](https://turbopuffer.com/blog/ann-v3), and
[Zvec](https://github.com/alibaba/zvec).

## Citation

If RaBitQ helps your research or system, please cite:

> Jianyang Gao, Yutong Gou, Yuexuan Xu, Yongyi Yang, Cheng Long, and Raymond
> Chi-Wing Wong. “Practical and Asymptotically Optimal Quantization of
> High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor
> Search.” *Proceedings of the ACM on Management of Data* 3, 3, Article 202
> (June 2025), 26 pages. [https://doi.org/10.1145/3725413](https://doi.org/10.1145/3725413).
