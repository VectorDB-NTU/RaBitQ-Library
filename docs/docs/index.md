---
title: RaBitQ Library
hide:
  - toc
---

<div class="hero" markdown>

![RaBitQ Library logo](assets/logo.svg){ .hero-logo }

<p class="eyebrow">RaBitQ Library · C++17 &amp; Python</p>

# Vector search with<br>compact RaBitQ codes. { #compact-vector-search-with-rabitq }

<p class="hero-description">Compress vectors and search for nearest neighbors.
Use ready-made indexes in Python or C++, or integrate quantization and distance
estimation into your own C++ search engine.</p>

<div class="hero-actions" id="start-here" markdown>

[Python quick start](quick_start.md){ .md-button .md-button--primary }
[C++ integration](cpp_integration.md){ .md-button }

</div>

```bash
python -m pip install "rabitqlib>=0.7.2"
```

[Installation and supported platforms](installation.md) ·
[How RaBitQ works](rabitq/rabitq.md)

</div>

## Choose your index

All three indexes support L2 and inner-product search, with save/load support.
Start with IVF-RaBitQ for the complete introductory example, then compare on your own data.

<div class="feature-grid" markdown>

<div class="feature-card" markdown>

### [IVF-RaBitQ](index/ivf.md)

Search selected clusters of compressed vectors. Tune the bit width and number
of clusters probed to balance storage, recall, and query time.

**Updates:** append and remove vectors.

</div>

<div class="feature-card" markdown>

### [HNSW-RaBitQ](index/hnsw.md)

Traverse a graph over compressed vectors. Graph links add storage; tune the
search window and graph degree alongside the code precision.

**Updates:** append and remove vectors.

</div>

<div class="feature-card" markdown>

### [SymphonyQG](index/qg.md)

Build a graph directly from vectors, without a separate clustering step.
Combine batched neighbor scoring with raw or 4/8-bit vector storage.

**Updates:** rebuild to change the dataset.

</div>

</div>

For input shapes, distance conventions, memory estimates, and threading, see
[shared index contracts](index/usage.md).

## Build on the library

- **Bring your own storage:** use the [stable C++ core API](rabitq/core_api.md)
  to encode vectors and estimate distances in caller-owned buffers. Add
  [FastScan](cpp_integration.md#fastscan-integration) for batch scoring.
- **Train clusters:** use [RaBitQKMeans or QGKMeans](clustering.md) to prepare
  centroids and assignments for IVF-RaBitQ or HNSW-RaBitQ.

## Compare measured performance

See the [benchmarks](benchmarks/bench.md) for datasets, settings, measured
results, and limitations. Evaluate recall and query time on your own dataset
when choosing an index.

## Built with RaBitQ-Library

- [txtai](integrations/txtai.md) uses the Python IVF-RaBitQ and HNSW-RaBitQ indexes as an
  embedding-search backend. See how to select the backend and its search settings.
- [zvec](integrations/zvec.md) uses core quantization in its IVF-RaBitQ and
  HNSW-RaBitQ implementations, with FastScan batch scoring in IVF-RaBitQ. See the
  integration walkthrough for storage, scoring, and platform boundaries.

## Explore further

- [Algorithm overview](rabitq/rabitq.md) for the rotation, encoding, and scoring workflow.
- [RaBitQ Library 0.7.2 release notes](releases/0.7.2.md) for the latest changes and upgrade guidance.
- [Papers and citations](papers.md) for the original RaBitQ method, its multi-bit extension, and SymphonyQG.
