# txtai: using the complete indexes

[txtai](https://github.com/neuml/txtai) exposes RaBitQ-Library through its
`rabitq` ANN backend. Applications can select an IVF or HNSW index within txtai's
embedding-search workflow, while txtai handles the surrounding application API
and result conversion.

This walkthrough was checked on October 21, 2026 against txtai commit
[`0ea7cb0d8234`][snapshot]. It describes the downstream adapter at that revision;
it is a source snapshot, not a verification of every published txtai package.
Consult the [current configuration reference][current-config] for later changes.

## Select the backend

In an existing txtai embeddings configuration, select `backend: rabitq` and add
the corresponding settings. This fragment uses four-bit IVF:

```yaml
backend: rabitq
rabitq:
  mode: ivf
  nbits: 4
```

Keep your existing embedding model and other application settings. Install
optional dependencies according to [txtai's installation guide][install]; its
adapter imports both `rabitqlib` and scikit-learn. The inspected
[package configuration][dependencies] declares `rabitqlib>=0.7.0` in its `ann` extra. The fragment above selects
the backend, rather than providing a complete model-and-data example.

`clusters` controls the KMeans centroid count in both modes: IVF uses these
centroids for partitioning, while HNSW uses them for residual quantization.
`nprobe` controls how many IVF clusters are searched. For HNSW, select `mode: hnsw` and use `m`, `efconstruction`, and
`efsearch`. Both modes accept 1–9 quantization bits; IVF also accepts `nbits: 32`
for original-vector reranking. Measure recall and latency on representative
queries when choosing precision and search parameters.

## How the integration works

The [adapter source][adapter] shows the boundary between the projects:

| Stage | txtai's role | RaBitQ-Library's role |
| --- | --- | --- |
| Build | Converts embeddings to contiguous float32 and uses scikit-learn KMeans to produce centroids and assignments. | Builds `IvfIndex` or `HnswIndex` with the inner-product metric. |
| Search | Selects `nprobe` or `ef`, drops missing-result IDs, and converts returned distances to scores with `1 - distance`. | Searches the native index and returns IDs and distances; quantized modes return estimates, while IVF raw mode uses original-vector reranking. |
| Persistence | Calls the backend's save/load methods. | Stores and restores the native index state. |
| Updates | Routes IVF append/delete requests to the backend. | Adds vectors or removes them from search. |

This is an example of integrating complete indexes: txtai calls the Python index
APIs without implementing RaBitQ encoding or graph traversal itself.

## Boundaries to keep in mind

At the inspected revision, txtai enables append/delete only for IVF. Its HNSW
adapter raises `NotImplementedError` for append/delete through the
[ANN base class][base], even though RaBitQ-Library itself supports HNSW updates.
IVF additions reuse the original clusters, and deletions retain their storage;
txtai recommends rebuilding when new data differs enough to affect recall.
See its [versioned configuration documentation][config] for these limits.

The adapter requests inner-product search and converts its `1 - dot` distance
convention back to scores. For normalized database and query embeddings, these are cosine-similarity
estimates in quantized modes; normalization does not remove quantization error.

The source inspection establishes the integration path. It does not establish
production deployment scale or a performance gain, and this configuration has
not been exercised end to end in this documentation update.

To use the same indexes directly, follow the library's [Python quick start](../quick_start.md)
and [HNSW guide](../index/hnsw.md). To integrate only quantization into your own
index, see the [zvec walkthrough](zvec.md).

[snapshot]: https://github.com/neuml/txtai/tree/0ea7cb0d8234497e8ea9ed1a0f2f58492e00460b
[adapter]: https://github.com/neuml/txtai/blob/0ea7cb0d8234497e8ea9ed1a0f2f58492e00460b/src/python/txtai/ann/dense/rabitq.py
[config]: https://github.com/neuml/txtai/blob/0ea7cb0d8234497e8ea9ed1a0f2f58492e00460b/docs/embeddings/configuration/ann.md#rabitq
[current-config]: https://github.com/neuml/txtai/blob/master/docs/embeddings/configuration/ann.md#rabitq
[install]: https://neuml.github.io/txtai/install/#ann

[dependencies]: https://github.com/neuml/txtai/blob/0ea7cb0d8234497e8ea9ed1a0f2f58492e00460b/setup.py
[base]: https://github.com/neuml/txtai/blob/0ea7cb0d8234497e8ea9ed1a0f2f58492e00460b/src/python/txtai/ann/base.py
