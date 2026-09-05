# How zvec integrates RaBitQ-Library

Zvec uses RaBitQ-Library for rotation, quantization, and distance estimation
inside its own IVF and HNSW indexes. Zvec manages storage, search traversal,
filtering, and result collection.

| Index | How it uses RaBitQ-Library |
| --- | --- |
| **IVF** | The [reformer][ivf-reformer] quantizes vectors in batches and prepares query lookup tables. The [search code][ivf-entity] uses FastScan estimates and extra-bit refinement for selected candidates. |
| **HNSW** | The [reformer][hnsw-reformer] quantizes individual vectors. The [query algorithm][hnsw-query] estimates distances from those codes during graph traversal. |

Both integrations preserve centroids and rotator state with the index and use
padded dimensions consistently for rotated vectors and queries. These are
key contracts when embedding the quantizer in another system.

This overview follows zvec commit
[`44d8bc6f7c3d`](https://github.com/alibaba/zvec/tree/44d8bc6f7c3d656efde2641c9410432ea14c1657).
Its [bundled dependency][dependency] includes local patches for lookup-table
reuse and SIMD error handling; check the pinned version before adapting its code.

To integrate RaBitQ-Library, start with the [quantizer](../rabitq/quantizer.md)
and [estimator](../rabitq/estimator.md) guides. For a complete standalone index,
see the [quick start](../quick_start.md).

[ivf-reformer]: https://github.com/alibaba/zvec/blob/44d8bc6f7c3d656efde2641c9410432ea14c1657/src/core/algorithm/ivf_rabitq/ivf_rabitq_reformer.cc
[ivf-entity]: https://github.com/alibaba/zvec/blob/44d8bc6f7c3d656efde2641c9410432ea14c1657/src/core/algorithm/ivf_rabitq/ivf_rabitq_entity.cc
[hnsw-reformer]: https://github.com/alibaba/zvec/blob/44d8bc6f7c3d656efde2641c9410432ea14c1657/src/core/algorithm/hnsw_rabitq/rabitq_reformer.cc
[hnsw-query]: https://github.com/alibaba/zvec/blob/44d8bc6f7c3d656efde2641c9410432ea14c1657/src/core/algorithm/hnsw_rabitq/hnsw_rabitq_query_algorithm.cc
[dependency]: https://github.com/alibaba/zvec/blob/44d8bc6f7c3d656efde2641c9410432ea14c1657/thirdparty/RaBitQ-Library/CMakeLists.txt
