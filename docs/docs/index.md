---
title: RaBitQ Library
hide:
  - toc
---

<div class="hero" markdown>

<p class="eyebrow">RaBitQ Library · C++17 &amp; Python</p>

# Compact vectors.<br>Accurate search. { #compact-vector-search-with-rabitq }

<p class="hero-description">Vector quantization and approximate nearest-neighbor
search. Start with a quantizer, or build with complete IVF, HNSW, and SymphonyQG indexes.</p>

<div class="hero-actions" markdown>

[Get started](quick_start.md){ .md-button .md-button--primary }
[Explore C++ integration &rarr;](cpp_integration.md)

</div>

<p class="hero-platforms">Linux x86-64 &amp; ARM64 · Windows x86-64 · macOS ARM64</p>

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
    <p>Use AVX2/AVX-512 kernels on x86-64 and NEON on ARM64. IVF and
    SymphonyQG use FastScan for batched distance estimation.</p>
  </div>
  <div class="feature-card">
    <h2>Ready for vector search</h2>
    <p>Choose IVF, HNSW, or SymphonyQG to balance memory, indexing cost,
    latency, and recall for your workload.</p>
  </div>
</div>

## New in 0.7.0

- **Modular C++ integration:** Link core, FastScan, or indexes independently.
- **Stable core API v1:** Encode and estimate distances with caller-owned buffers
  and workspace, with no heap allocation on valid calls.
- **Optional OpenMP:** Build core and FastScan without an OpenMP dependency.
- **Unified index counts:** Distinguish capacity, stored records, and live points
  consistently in C++ and Python.

This documentation describes 0.7.0. Read the
[0.7.0 upgrade notes](releases/0.7.0.md) before updating an existing application.

## Start with Python

See [installation options](quick_start.md#install) for supported platforms and source builds.

```bash
python -m pip install "rabitqlib>=0.7.0"
```

The [complete quick start](quick_start.md) builds and searches an IVF index with
synthetic data, explains the results, and demonstrates save/load. No external
dataset or clustering package is needed.

[Start the Python example](quick_start.md#build-and-search-an-ivf-index){ .md-button .md-button--primary }

## C++ components

The [0.7.0 component model](modular_migration.md) provides independent core and
FastScan targets alongside the index library, plus a [stable core v1 API](rabitq/core_api.md).
Use a matching source checkout or 0.7.0 installation for these interfaces.

## Choose an index

| Index | Best fit | Typical relative memory | Main search control |
| --- | --- | --- | --- |
| [IVF RaBitQ](index/ivf.md) | Large datasets and predictable memory use | Lowest | Number of probed clusters |
| [HNSW RaBitQ](index/hnsw.md) | General-purpose graph search | Moderate | Search candidate list size |
| [SymphonyQG](index/qg.md) | Latency-focused graph search | Highest | Search window size |

IVF stores quantized vectors by default, or retains original float32 vectors
for reranking with `nbits=32`. HNSW searches quantized
vectors instead of accessing raw vectors during search. SymphonyQG uses additional memory and multiple codes per vector to
optimize its access pattern.

These are typical relative profiles, not fixed guarantees. Actual memory,
latency, and recall depend on vector dimension, quantization width, graph
degree, and search parameters.
See [memory estimates](index/usage.md#estimate-storage) for the main storage costs,
including SymphonyQG's copies of neighbor codes.

## Why RaBitQ?

RaBitQ combines compact codes with distance estimates and probabilistic error
intervals. The [algorithm overview](rabitq/rabitq.md) explains the rotation,
encoding and scoring workflow; the [benchmarks](benchmarks/bench.md) report
measured memory, recall and throughput for specific configurations.

The library supports squared L2 and inner-product distance (`1 - dot`). For cosine
search, normalize data and queries before using inner product.

## Used across the vector-search ecosystem

[txtai](https://github.com/neuml/txtai) uses `rabitqlib` directly as an ANN
backend with IVF and HNSW modes. See its
[RaBitQ configuration](https://github.com/neuml/txtai/blob/master/docs/embeddings/configuration/ann.md#rabitq).

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
