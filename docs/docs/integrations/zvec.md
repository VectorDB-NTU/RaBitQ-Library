# How zvec integrates RaBitQ-Library

Zvec uses RaBitQ-Library for rotation, quantization, and distance estimation
inside its IVF-RaBitQ and HNSW-RaBitQ implementations. Zvec manages storage, search traversal,
filtering, and result collection. This is a component integration: zvec links
core/FastScan functionality while keeping its own index implementations.

Checked on October 21, 2026 against the merged RaBitQ 0.7 migration at commit
[`c5496863ee98`](https://github.com/alibaba/zvec/tree/c5496863ee98bca155abca1e9e4aadf46e9ebed1).
This identifies the inspected source, not a released zvec package version.

## Where the components fit

| Index | How it uses RaBitQ-Library |
| --- | --- |
| **IVF-RaBitQ** | The [reformer][ivf-reformer] quantizes vectors in batches and prepares query lookup tables. The [search code][ivf-entity] uses FastScan estimates and extra-bit refinement for selected candidates. |
| **HNSW-RaBitQ** | The [reformer][hnsw-reformer] quantizes individual vectors. The [query algorithm][hnsw-query] uses core per-vector estimators during graph traversal; it does not use the IVF-RaBitQ FastScan batch-scoring path. |

The inspected reformers serialize centroids, padded dimensions, and rotator
state, then restore them when loading. Their query preprocessing uses the stored
padded dimension and rotation. Preserving this state is a key integration
contract; this source walkthrough is not a complete persistence audit.

## A maintained integration

[PR #814](https://github.com/alibaba/zvec/pull/814), merged on October 19, 2026,
upgraded zvec's bundled RaBitQ-Library from 0.3.8 to 0.7.0. It replaced custom
SIMD source lists and a local lookup-table reuse patch with upstream component
targets and `SplitBatchQuery::reset`.

When `RABITQ_SUPPORTED` is enabled, the [dependency configuration][dependency]
links `rabitqlib::fastscan`, which also links core,
and disables RaBitQ's standalone indexes, Python bindings, OpenMP, and native
tuning. The [IVF-RaBitQ query preparation][ivf-reformer] passes lookup-table buffers owned by
its query state to `SplitBatchQuery::reset`. This permits reuse when that state
is reused; it does not imply that every search is allocation-free.

This change illustrates a practical integration benefit of the component split:
zvec can consume upstream quantization and scoring implementations while retaining
control of its indexes and threading. The PR description reports compatibility checks and
Linux x86-64 validation, along with untested platforms and incomplete full-database
sanitizer checks. Those checks were not rerun for this walkthrough; it
is not evidence of production deployment scale or a measured speedup.

## Platform and API boundaries

At this snapshot, zvec's [build gate][platform] enables RaBitQ only for Linux
x86-64 or Windows x86-64 with MSVC, and only if its compiler checks pass.
Other platforms use stub implementations. RaBitQ-Library's own ARM64 support
does not make these zvec integrations available on ARM64 or macOS.

The code uses the advanced encoder, query, estimator, and record-layout headers.
It does not use the stable `rabitqlib::quant::v1` API. The recommendation below is
an entry point for new integrations, not a description of zvec's API usage.

## Use this pattern in your own engine

For a new per-vector integration, start with the
[stable core API v1](../rabitq/core_api.md#minimal-example). For FastScan or existing
record layouts like those used here, consult the advanced
[quantizer](../rabitq/quantizer.md) and [estimator](../rabitq/estimator.md) references.
For a complete standalone index, see the [quick start](../quick_start.md) or
the [txtai integration](txtai.md).

[ivf-reformer]: https://github.com/alibaba/zvec/blob/c5496863ee98bca155abca1e9e4aadf46e9ebed1/src/core/algorithm/ivf_rabitq/ivf_rabitq_reformer.cc
[ivf-entity]: https://github.com/alibaba/zvec/blob/c5496863ee98bca155abca1e9e4aadf46e9ebed1/src/core/algorithm/ivf_rabitq/ivf_rabitq_entity.cc
[hnsw-reformer]: https://github.com/alibaba/zvec/blob/c5496863ee98bca155abca1e9e4aadf46e9ebed1/src/core/algorithm/hnsw_rabitq/rabitq_reformer.cc
[hnsw-query]: https://github.com/alibaba/zvec/blob/c5496863ee98bca155abca1e9e4aadf46e9ebed1/src/core/algorithm/hnsw_rabitq/hnsw_rabitq_query_algorithm.cc
[dependency]: https://github.com/alibaba/zvec/blob/c5496863ee98bca155abca1e9e4aadf46e9ebed1/thirdparty/RaBitQ-Library/CMakeLists.txt

[platform]: https://github.com/alibaba/zvec/blob/c5496863ee98bca155abca1e9e4aadf46e9ebed1/CMakeLists.txt
