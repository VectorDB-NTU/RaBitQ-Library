# txtai: using the complete indexes

[txtai](https://github.com/neuml/txtai) exposes RaBitQ-Library through its
`rabitq` ANN backend. Applications can select an IVF-RaBitQ or HNSW-RaBitQ index within txtai's
embedding-search workflow, while txtai handles the surrounding application API
and result conversion.

This walkthrough was checked on November 20, 2026 against txtai commit
[`86f3bad`][snapshot], which moved the adapter to forms that rabitqlib 0.8.5 and 0.9
accept ([neuml/txtai#1616](https://github.com/neuml/txtai/pull/1616)). It describes
the downstream adapter at that revision; it is a source snapshot, not a verification
of every published txtai package. Consult the [current configuration reference][current-config]
for later changes.

rabitqlib 0.9 needs txtai 9.15.0 or later. txtai 9.14.0, the first release with
the backend, uses call forms that 0.9 removes and fails with it; rabitqlib 0.8
still accepts them, with deprecation warnings. Its `ann` extra only declares
`rabitqlib>=0.5.1`, so with txtai 9.14.0 keep `rabitqlib<0.9`.

## Select the backend

In an existing txtai embeddings configuration, select `backend: rabitq` and add
the corresponding settings. This fragment uses four-bit IVF-RaBitQ:

```yaml
backend: rabitq
rabitq:
  mode: ivf
  nbits: 4
```

Keep your existing embedding model and other application settings. Install
optional dependencies according to [txtai's installation guide][install]; its
adapter imports both `rabitqlib` and scikit-learn. The inspected
[package configuration][dependencies] declares `rabitqlib>=0.8.5` in its `ann` extra. The fragment above selects
the backend, rather than providing a complete model-and-data example.

`clusters` controls the KMeans centroid count in both modes: IVF-RaBitQ uses these
centroids for partitioning, while HNSW-RaBitQ uses them for residual quantization.
`nprobe` controls how many IVF-RaBitQ clusters are searched. For HNSW-RaBitQ, select `mode: hnsw` and use `m`, `efconstruction`, and
`efsearch`. Both modes accept 1–9 quantization bits; IVF-RaBitQ also accepts `nbits: 32`
for original-vector reranking. Measure recall and latency on representative
queries when choosing precision and search parameters.

## How the integration works

The [adapter source][adapter] shows the boundary between the projects:

| Stage | txtai's role | RaBitQ-Library's role |
| --- | --- | --- |
| Build | Converts embeddings to contiguous float32 and uses scikit-learn KMeans to produce centroids and assignments. | Creates `IvfIndex` or `HnswIndex` with the inner-product metric, takes the centroids with `set_centroids`, and adds the embeddings with `add(cluster_ids=...)`. |
| Search | Selects `nprobe` or `efsearch`, drops unfilled result slots by their infinite distance, and converts returned distances to scores with `1 - distance`. | Searches the native index and returns IDs and distances; quantized modes return estimates, while IVF-RaBitQ raw mode uses original-vector reranking. |
| Persistence | Calls the backend's save/load methods. | Stores and restores the native index state. |
| Updates | Routes IVF-RaBitQ append/delete requests to the backend. | Adds vectors or removes them from search. |

This is an example of integrating complete indexes: txtai calls the Python index
APIs without implementing RaBitQ encoding or graph traversal itself.

## Boundaries to keep in mind

At the inspected revision, txtai enables append/delete only for IVF-RaBitQ. Its HNSW-RaBitQ
adapter raises `NotImplementedError` for append/delete through the
[ANN base class][base], even though RaBitQ-Library itself supports HNSW-RaBitQ updates.
IVF-RaBitQ additions reuse the original clusters, and deletions retain their storage;
txtai recommends rebuilding when new data differs enough to affect recall.
See its [versioned configuration documentation][config] for these limits.

The adapter requests inner-product search and converts its `1 - dot` distance
convention back to scores. For normalized database and query embeddings, these are cosine-similarity
estimates in quantized modes; normalization does not remove quantization error.

The first build adds with every available CPU, capped by `OMP_NUM_THREADS`, so an
HNSW-RaBitQ graph built with several threads differs between runs even with the same
`randomseed`. An index saved with rabitqlib 0.9 uses index format 2, which rabitqlib 0.8
cannot read.

txtai's RaBitQ tests pass with this adapter on rabitqlib 0.8.5 and on a 0.9 build.
That establishes the integration path, not production deployment scale or a
performance gain.

For maintainer checks against future library versions, see the
[downstream compatibility procedure](https://github.com/VectorDB-NTU/RaBitQ-Library/blob/main/docs/development.md#downstream-compatibility-validation).

To use the same indexes directly, follow the library's [Python quick start](../quick_start.md)
and [HNSW-RaBitQ guide](../index/hnsw.md). To integrate only quantization into your own
index, see the [zvec walkthrough](zvec.md).

[snapshot]: https://github.com/neuml/txtai/tree/86f3bad49de8fe3f1b2cb4b0f3e3166afb84820b
[adapter]: https://github.com/neuml/txtai/blob/86f3bad49de8fe3f1b2cb4b0f3e3166afb84820b/src/python/txtai/ann/dense/rabitq.py
[config]: https://github.com/neuml/txtai/blob/86f3bad49de8fe3f1b2cb4b0f3e3166afb84820b/docs/embeddings/configuration/ann.md#rabitq
[current-config]: https://github.com/neuml/txtai/blob/master/docs/embeddings/configuration/ann.md#rabitq
[install]: https://neuml.github.io/txtai/install/#ann

[dependencies]: https://github.com/neuml/txtai/blob/86f3bad49de8fe3f1b2cb4b0f3e3166afb84820b/setup.py
[base]: https://github.com/neuml/txtai/blob/86f3bad49de8fe3f1b2cb4b0f3e3166afb84820b/src/python/txtai/ann/base.py
