# Quick start

Build, search, save and reload a small IVF index with Python. This guide describes
**0.7.0** and uses synthetic data, so no dataset download is required.
For C++, go directly to [C++ integration](cpp_integration.md).

## Install

```bash
python -m pip install "rabitqlib>=0.7.0"
```

Wheels support CPython 3.11–3.14 on Linux x86-64/ARM64, Windows x86-64 and
macOS 14+ ARM64. x86-64 requires AVX2/FMA; ARM64 uses NEON.
See [installation](installation.md) for platform details and source builds.

Supply data and queries with shape `(count, dim)` in the original dimension.
Python indexes convert them to contiguous float32 and accept `64 <= dim <= 65536`;
they rotate and pad internally. See [input requirements](index/usage.md#input-requirements).

## Build and search an IVF index

The following complete example uses deterministic synthetic data and does not
require a dataset download:

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

# IVF requires exactly max_elements rows when building.
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

## Read the results

Both outputs have shape `(number_of_queries, k)`. IDs are zero-based row numbers
in the data supplied to `build`; added vectors receive new IDs without renumbering
existing ones. Results are ordered by increasing distance: `metric="l2"` estimates
squared Euclidean distance, while `metric="ip"` estimates `1 - dot(data, query)`.
Smaller values are better for both metrics. Quantized distances are estimates,
not exact distances to the original vectors.

IVF and HNSW may return fewer than `k` valid neighbors, for example after removals
or when the probed IVF clusters contain too few points. Unfilled slots contain
ID `2**32 - 1` and distance `+inf`. Filter them with
`valid = ids != np.iinfo(np.uint32).max` before using IDs to index your data.

The `metric` argument accepts `"l2"` and `"ip"` (also `"innerproduct"`). For
cosine search, normalize database and query vectors first and use `metric="ip"`.

## Measure recall

For this small L2 example, compute exact neighbors from the original vectors,
then compare several probe counts on the same built index. Recall@10 is the mean
fraction of exact top-10 IDs returned for each query.

```python
k = 10
# Float64 exhaustive distances are practical for this tiny dataset only.
delta = queries.astype(np.float64)[:, None, :] - data.astype(np.float64)[None, :, :]
exact_distances = np.sum(delta * delta, axis=2)
exact_ids = np.argsort(exact_distances, axis=1, kind="stable")[:, :k]

for nprobe in (1, 2, 5):
    candidate_ids, _ = index.search(queries, k=k, nprobe=nprobe)
    recall = np.mean([
        np.intersect1d(found, expected).size / k
        for found, expected in zip(candidate_ids, exact_ids)
    ])
    print(f"nprobe={nprobe}: Recall@{k}={recall:.3f}")
```

Missing-result IDs cannot match any ground-truth ID and count as misses.
Probing more clusters searches more candidates, but even probing all five does
not guarantee recall of 1: quantized scoring and filtering remain approximate.

This small example demonstrates the calculation. For tuning, use representative
held-out queries and compare recall with latency; see the [benchmark guide](benchmarks/bench.md).
The exhaustive array above is only suitable for small datasets. IP ground truth
must rank `1 - dot` instead of squared L2.

## Save and load

Continue with the same `index` and `queries`. This example uses a temporary
directory that is removed when the block finishes; use your own persistent path
when keeping an index.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

with TemporaryDirectory() as directory:
    path = str(Path(directory) / "ivf.index")
    index.save(path)
    loaded = IvfIndex.load(path)
    loaded_ids, loaded_distances = loaded.search(queries, k=10, nprobe=5)
    np.testing.assert_array_equal(loaded_ids, ids)
    np.testing.assert_allclose(loaded_distances, distances)
```

The loaded index owns its data and restores the saved rotation. IVF indexes using
HNSW centroid routing also need the accompanying `.hnsw` file; keep it with the
main index. This small example uses Flat routing and has no sidecar.
See [file compatibility](index/usage.md#index-file-compatibility) before downgrading.

## Next steps

- [IVF](index/ivf.md): raw-vector reranking, precision, centroid routing and updates.
- [HNSW](index/hnsw.md) and [SymphonyQG](index/qg.md): graph-index workflows and parameters.
- [Index usage contracts](index/usage.md): counts, threading, input lifetimes and file paths.
- [Clustering](clustering.md): choose flat RaBitQKMeans or graph-based QGKMeans.
- [C++ integration](cpp_integration.md): builds, installation and component selection.
- [Stable core API](rabitq/core_api.md): integrate quantization into your own storage.
