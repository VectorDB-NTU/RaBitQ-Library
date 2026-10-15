---
title: RaBitQ Library
hide:
  - toc
---

<div class="hero" markdown>

<p class="eyebrow">RaBitQ Library · C++17 &amp; Python</p>

# Compact vectors.<br>Accurate search. { #compact-vector-search-with-rabitq }

<p class="hero-description">A C++17 and Python library for vector search,
from compact quantization to complete indexes.</p>

<div class="hero-actions" markdown>

[Get started with Python](quick_start.md){ .md-button .md-button--primary }
[Use C++ &rarr;](cpp_integration.md)

</div>

</div>

## Start here

<div class="feature-grid">
  <div class="feature-card">
    <h3><a href="quick_start/">Python quick start &rarr;</a></h3>
    <p>Install the package, build an IVF index, search it, and save it.</p>
  </div>
  <div class="feature-card">
    <h3><a href="cpp_integration/">C++ integration &rarr;</a></h3>
    <p>Choose core, FastScan, or index components for your application.</p>
  </div>
  <div class="feature-card">
    <h3><a href="rabitq/rabitq/">How RaBitQ works &rarr;</a></h3>
    <p>Understand rotation, compact codes, and distance estimation.</p>
  </div>
</div>

## Choose an index

| Index | Best fit | Typical relative memory |
| --- | --- | --- |
| [IVF RaBitQ](index/ivf.md) | Large datasets and predictable storage | Lowest |
| [HNSW RaBitQ](index/hnsw.md) | General-purpose graph search | Moderate |
| [SymphonyQG](index/qg.md) | Latency-focused graph search | Highest |

Memory, latency, and recall depend on the dataset and search settings. See the
[index usage guide](index/usage.md#estimate-storage) for storage estimates and the
[benchmarks](benchmarks/bench.md) for measured workloads.

## Explore further

- [Algorithm overview](rabitq/rabitq.md) for the rotation, encoding, and scoring workflow.
- [RaBitQ Library 0.7.0 release notes](releases/0.7.0.md) for new APIs and upgrade guidance.

## Research papers

- [1-bit RaBitQ](https://doi.org/10.1145/3654970) · 2024 · Quantization with a theoretical error bound.
- [SymphonyQG](https://doi.org/10.1145/3709730) · 2025 · Quantization and graph search.
- [Multi-bit RaBitQ](https://doi.org/10.1145/3725413) · 2025 · Higher-bit quantization.
