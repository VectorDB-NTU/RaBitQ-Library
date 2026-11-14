# Unreleased

## Index files

- IVF-RaBitQ, HNSW-RaBitQ and SymphonyQG `save` write [index format 2](https://github.com/VectorDB-NTU/RaBitQ-Library/blob/main/docs/formats.md#index-format-2), a single file with a header, a section table and checksummed metadata.
- Releases up to 0.8 cannot read these files. Keep a copy saved by 0.8 if you may downgrade.
- Files written by every earlier release still load, including the `.hnsw` sidecars of HNSW-routed IVF files. Saving one writes format 2 with the same rotation, codes and results.
- IVF-RaBitQ stores HNSW centroid routing inside the file, so `save` no longer writes a `<file>.hnsw` sidecar. A sidecar left by an earlier save is not removed.
- The saved bytes now include that routing graph. It is reproducible only when `train`, `set_centroids` or the first `add` builds it with `num_threads=1`. Saving the same index twice still gives identical files.
- `save` rejects IVF-RaBitQ indexes of more than 65,536 dimensions.
- A format-2 file that uses read-only-compatible features unknown to this release loads for search only: operations that change or save the index raise. Rebuilding with IVF-RaBitQ `construct`, or a new SymphonyQG build, makes it writable again.

## Removed and renamed Python forms

0.9 removes every Python form that 0.8 deprecated. The [migration guide](../python_migration.md) lists each replacement; all of them work in 0.8.5, where the old forms warn.

- The 0.7 call forms, deprecated since 0.8.0 (#213):
  - `IvfIndex(dim, max_elements, num_clusters, nbits, ...)`, `HnswIndex(dim, max_elements, ...)` and its `random_seed=`, and `SymqgIndex(dim, max_degree, metric, quantization_bits)` and its `quantization_bits=`;
  - `build()` and `is_built` on every index, and `SymqgIndex.search_batch()` and `SymqgIndex.quantization_bits`;
  - `cluster_ids`, `num_threads` and `fast_quantization` passed to `add` by position.
- The 0.8 index spellings, deprecated since 0.8.5 (#232):
  - `max_elements`: read `capacity` on `HnswIndex` and `num_stored_points` on `IvfIndex` and `SymqgIndex`;
  - `HnswIndex.resize(n)`: call `reserve(n)`, which only grows the capacity;
  - `IvfIndex.reserve(cluster_id, num_points)`: call `reserve_cluster`;
  - `IvfIndex(initializer=...)` and `index.initializer`: use `routing`;
  - `search(queries, k, ef=...)` on `HnswIndex` and `SymqgIndex`: use `ef_search=`;
  - `high_accuracy` and `num_threads` passed to `search` by position: they are keyword-only, and only `nprobe` or `ef_search` stays positional;
  - `nprobe=0` and `ef_search=0` for the default: they raise `ValueError`, and `None` selects the default. `num_threads=0` still means every available CPU;
  - `HnswIndex` seeds above `2**32 - 1`: they raise `ValueError`, as on the other two classes.
- The 0.8 k-means spellings, deprecated since 0.8.5 (#233):
  - `QGKMeans` and `QGKMeansParameters` take `nbits` (32, 4 or 8), `max_degree` and `ef_construction` instead of `quantization_bits` (0, 4 or 8), `graph_degree` and `ef_build`;
  - `final_assignment` is `'approximate'` or `'exact'`, also on `index.cp`, and reads back as that string. `FinalAssignmentMode` is no longer in the package;
  - `KMeansIterationStats` is the only name of the statistics class; `QGKMeansIterationStats` and `RaBitQKMeansIterationStats` are no longer in the package.
- A removed call form of an index raises `TypeError`, and a removed attribute or name `AttributeError` (`ImportError` when imported), as does a removed `QGKMeans` keyword, like any unknown keyword there; none is read as a different call. For example, `IvfIndex(dim, 1000, 64)` is an error, not an index with 1000 clusters.
- Changes that 0.8.5 could not announce with a warning:
  - `HnswIndex()` without `seed=` draws a seed, as `IvfIndex` and `SymqgIndex` do, instead of using 100. `index.seed` reports it; pass `seed=`, and `num_threads=1` to `add`, for a reproducible build.
  - `repr(index)` prints `capacity=` for `HnswIndex` and `routing=` for `IvfIndex`, and no longer prints `max_elements=` for `IvfIndex`.
- The index classes are now the extension's own, without replaced methods:
  - A wrong argument raises pybind11's `TypeError`, which lists the supported signature. Messages such as `IvfIndex() got an unexpected keyword argument 'x'` are gone.
  - `inspect.signature()` no longer works on the index constructors and methods; `help()` and the type stubs describe them.
- k-means messages name the new parameters: `QGKMeans nbits must be 4, 8, or 32`, `QGKMeans max_degree must be a positive multiple of 32`, `QGKMeans requires more centroids than max_degree` and `QGKMeans ef_construction and ef_search must be positive`. As for the other parameters, `train` reports an invalid `nbits`; 0.8.5 rejected it when it was set.
- txtai 9.14.0 and earlier use the 0.7 forms and fail with this release. They also filter results with `uid != 0xFFFFFFFF`, which no longer matches a missing result (see `int64` IDs below). See #213 for the first compatible txtai release.

## Python changes

- New `metric="cosine"` for `IvfIndex`, `HnswIndex` and `SymqgIndex` (#212). The index scales data and queries to unit length and returns `1 - cos`; results equal those of a `metric="ip"` index built from the scaled vectors. A zero vector stays zero, so its true distance to every vector is 1. The saved file records the metric, so a loaded index keeps scaling. `index.metric` can now return `"cosine"`, and an invalid metric raises `metric must be 'l2', 'ip' or 'cosine'`. See [the cosine metric](../index/usage.md#the-cosine-metric).
- Point IDs are `int64` at the Python boundary (#211):
  - `search` returns `int64` IDs, and slots without a neighbor hold `-1` instead of `2**32 - 1`; their distance is still `inf`. Code that compares IDs with `2**32 - 1`, `0xFFFFFFFF` or `np.iinfo(np.uint32).max` must change; `np.isfinite(distances)`, recommended since 0.8.2, works before and after.
  - `add` returns `int64` IDs. `remove` already took any integer IDs and still rejects `-1` and other out-of-range values without removing anything.
  - IDs remain the row numbers the index assigns, and C++ point IDs remain 32-bit `PID` values with `kPidMax` for a missing result.

- Loading a format-2 file keeps the `seed` of every index, and `ef_construction` and `init` of SymphonyQG. They were `None` after `load`, and still are for files from releases up to 0.8.
- With a recorded seed, `cp.seed` after `load` follows it as for a new index; for an HNSW-RaBitQ file whose seed exceeds 32 bits, which only C++ can write, it is the low 32 bits. It was the k-means default.
- A new exception, `rabitqlib.FormatError`, subclasses both `ValueError` and `RuntimeError`:
  - Every index `load` raises it for a truncated, corrupt, mismatched or unsupported file, in format 2 and in every earlier layout. These were `RuntimeError`, which still catches them. A file that cannot be opened still raises a plain `RuntimeError`.
  - `quant.FormatError` is now the same class. It was a subclass of `ValueError` only, which still catches it.
- Messages for damaged or mismatched files change:
  - A format-2 file names the problem, for example `IVF index file is 6916 bytes, but its sections end at 6852` for trailing bytes.
  - Errors in an HNSW routing graph name the graph rather than the sidecar: `Truncated HNSW centroid graph` and `HNSW centroid graph changed while loading` replace their `sidecar` forms.
  - A missing sidecar of an earlier file still raises `Cannot open HNSW centroid sidecar`.
  - A failed HNSW-RaBitQ write raises `HNSW index file cannot be written` instead of `HNSW failed to write index file`.

## C++ changes

- `qgkmeans::QGKMeansParameters` uses the Python names (#233): `graph_degree` becomes `max_degree`, `ef_build` becomes `ef_construction`, and `quantization_bits` (0, 4 or 8) becomes `nbits` (32, 4 or 8, with 32 for raw float32). The fields keep their order and types. An `nbits` of 0 is rejected, so code that initialized the field positionally with 0 must pass 32. `QGAssigner` keeps its constructor, including `quantization_bits` with 0 for raw storage.
- Messages follow the names: `QGKMeans nbits must be 4, 8, or 32`, `QGKMeans max_degree must be a positive multiple of 32`, `QGKMeans requires more centroids than max_degree` and `QGKMeans ef_construction and ef_search must be positive`.
- `FinalAssignmentMode::SymphonyQG`, an alias of `Approximate`, is removed.
- The aliases `qgkmeans::QGKMeansIterationStats` and `rabitqkmeans::RaBitQKMeansIterationStats` are removed; use `clustering::KMeansIterationStats`.

- `MetricType` gains `METRIC_COSINE`, which `IVF`, `HierarchicalNSW`, `QuantizedGraph<float>` and `default_training_parameters` accept. The indexes normalize their inputs and store inner-product codes; `metric_type()` returns `METRIC_COSINE`. Entry points that validate the metric reject it with their existing messages: the stable core API v1, factor computation in the encoders, the query classes, FastScan queries, clustering and the IVF initializers. Code with an exhaustive `switch` over `MetricType` may warn about the new value.

- Loading an index throws `rabitqlib::FormatError`, a subclass of `std::runtime_error` in the new header `rabitqlib/index/format_error.hpp`, for a damaged, mismatched or unsupported file. Failing to open the file still throws `std::runtime_error`. Files of earlier layouts that threw other exceptions for invalid contents now throw `FormatError` too: an IVF-RaBitQ file with an invalid metric or a dimension its rotation does not support, which threw `std::invalid_argument`, and a SymphonyQG file with invalid settings or dimensions, which threw `std::invalid_argument` or `std::length_error`.

- `ivf::Initializer` and its subclasses:
  - `save(std::ofstream&, const char*)` becomes `write(std::ostream&)`, and `load(std::ifstream&, const char*)` becomes `read(std::istream&, uint64_t bytes)`.
  - The HNSW initializer no longer writes a sidecar itself.
- New accessors report what format-2 files record, or no value when unknown:
  - `IVF::seed()`;
  - `HierarchicalNSW::seed()`;
  - `QuantizedGraph<float>::seed()`, `ef_construction()` and `initialization()`.
- `symqg::QGInitialization` moves from `qg_builder.hpp` to `qg.hpp`, which `qg_builder.hpp` includes.
- The vendored hnswlib adds `saveIndex(std::ostream&)` and `loadIndex(std::istream&, size_t bytes, ...)`; the path overloads are unchanged.
