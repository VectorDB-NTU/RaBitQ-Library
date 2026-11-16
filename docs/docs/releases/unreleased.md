# Unreleased

## Index files

- IVF-RaBitQ, HNSW-RaBitQ and SymphonyQG `save` write [index format 2](https://github.com/VectorDB-NTU/RaBitQ-Library/blob/main/docs/formats.md#index-format-2), a single file with a header, a section table and checksummed metadata.
- Releases up to 0.8 cannot read these files. Keep a copy saved by 0.8 if you may downgrade.
- Files written by every earlier release still load, including the `.hnsw` sidecars of HNSW-routed IVF files. Saving one writes format 2 with the same rotation, codes and results.
- IVF-RaBitQ stores HNSW centroid routing inside the file, so `save` no longer writes a `<file>.hnsw` sidecar. A sidecar left by an earlier save is not removed.
- The saved bytes now include that routing graph. It is reproducible only when `train`, `set_centroids` or the first `add` builds it with `num_threads=1`. Saving the same index twice still gives identical files.
- `save` rejects IVF-RaBitQ indexes of more than 65,536 dimensions.
- A format-2 file that uses read-only-compatible features unknown to this release loads for search only: operations that change or save the index raise. A new index built from the original data is writable.

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
  - `metric="innerproduct"` raises `ValueError` on the index classes and in `rabitqlib.quant`. Use `"ip"`, which every release accepts and the `metric` properties return.
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
  - IDs remain the row numbers the index assigns. The C++ `IvfIndex`, `HnswIndex` and `SymqgIndex` use 64-bit `rabitqlib::Id` values with `kNoId` (-1) for a missing result, as Python does. The indexes still hold 32-bit row numbers inside, so they keep their limit of 2^31 points; the unsupported `ivf::IVF`, `hnsw::HierarchicalNSW` and `symqg::QuantizedGraph` keep `PID` and `kPidMax`.

- An `IvfIndex`, `HnswIndex` or `SymqgIndex` call with both an invalid argument and the wrong state may now report the argument first, for example `ValueError` for `nprobe=0` on an untrained index, which raised `RuntimeError`, or for data of the wrong dimension passed to `SymqgIndex.add` after the graph is built. `HnswIndex()` with an invalid `metric` or `seed` and an out-of-range `num_clusters` now reports the metric or seed. `SymqgIndex()` with an invalid `init` and an invalid `dim`, `nbits` or `max_degree` now reports the `init`, and with an invalid `seed` and an invalid `dim`, `nbits`, `max_degree` or `ef_construction` the seed.

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
- `pickle.dumps(obj, protocol=0)` or `protocol=1` on a `rabitqlib` object no longer aborts the Python process. `quant.Rotator` and `quant.Query` now pickle with every protocol; the index, k-means, parameter and iteration-statistics classes raise `TypeError: cannot pickle ...`, as they already did with protocol 2 and higher, which include the default. Protocols 0 and 1 cannot reduce pybind11 objects ([pybind11#3745](https://github.com/pybind/pybind11/issues/3745)), so they now use the protocol-2 reduction.

## C++ changes

- New `rabitqlib::IvfIndex` in `rabitqlib/index/ivf_index.hpp` (#234), with the workflow and defaults of the Python class, which now wraps it, and its messages where they do not name Python values:
  - `IvfIndex(dim, num_clusters, options)` takes an `IvfOptions` struct with an `ivf::Routing`, an alias of `ivf::InitializerType`. A static `load` returns a trained index, and `train`, `set_centroids`, `add`, `remove`, `reserve_cluster`, `search` and `save` follow the Python methods.
  - The first `add` trains an untrained index and leaves it untrained and empty if it fails. `add` takes an `AddOptions` struct and returns the first new ID.
  - IDs are `rabitqlib::Id` (`int64_t`). A missing search result is `kNoId` (-1) with distance `+inf`.
  - `search` takes a batch of queries and an `IvfSearchOptions` struct. Without an `nprobe` it probes `num_clusters / 16` clusters, rounded and at least 1, and an explicit 0 throws.
  - A seed is drawn unless given, and `num_threads == 0` means every available CPU, capped by `OMP_NUM_THREADS`.
  - Arguments out of range throw `std::invalid_argument`, and calls in the wrong state `std::logic_error`.
- New `rabitqlib::HnswIndex` in `rabitqlib/index/hnsw_index.hpp` (#234), built the same way, which the Python `HnswIndex` now wraps:
  - `HnswIndex(dim, options)` takes an `HnswOptions` struct (metric, `nbits`, `num_clusters`, `M`, `ef_construction`, seed). `train`, `set_centroids`, `add`, `remove`, `reserve`, `search`, `save` and the static `load` follow the Python methods.
  - The first `add` trains `min(num_clusters, number of points)` centroids and leaves the index untrained, empty and with its capacity if it fails.
  - Capacity starts at one point and grows on `add`, at least doubling, after the points and cluster IDs are checked; `reserve` only grows it.
  - `add` uses every available CPU unless `num_threads` is set, whereas `hnsw::HierarchicalNSW::add` uses one; pass `num_threads = 1` and a seed for a reproducible graph.
  - `search` takes a `GraphSearchOptions` struct. Without an `ef_search` it uses `max(k, 64)`, and an explicit 0 throws.
  - `seed()` is a `std::optional<uint64_t>`, because a file written by `hnsw::HierarchicalNSW` may record a seed above 32 bits.
- New `rabitqlib::SymqgIndex` in `rabitqlib/index/symqg_index.hpp` (#234), which the Python `SymqgIndex` now wraps:
  - `SymqgIndex(dim, options)` takes a `SymqgOptions` struct (metric, `nbits` of 32, 4 or 8 with 32 for raw float32, `max_degree`, `ef_construction`, `init`, seed). `add` builds the graph from every vector, once, and a built or loaded index throws `std::logic_error` on `add`. `train` only checks its input. `search`, `save` and the static `load` follow the Python methods.
  - A failed `add` leaves the index without a graph. `add` takes a worker count, with 0, the default, for every available CPU.
  - `search` takes a `GraphSearchOptions` struct. Without an `ef_search` it uses `max(k, 64)`, a smaller value is raised to `k`, and an explicit 0 throws; `symqg::QuantizedGraph` rejects an `ef_search` below `k`.
  - `seed()`, `ef_construction()` and `init()` have no value for a file of a release up to 0.8.
- `IvfIndex`, `HnswIndex` and `SymqgIndex` are the supported and documented C++ index API from 0.9. `ivf::IVF`, `hnsw::HierarchicalNSW`, `symqg::QuantizedGraph` and `symqg::QGBuilder` stay installed at their paths and keep their interfaces apart from the changes listed below, but they leave the documentation and carry no compatibility promise.

Move to the new classes as follows:

| 0.8 | 0.9 |
| --- | --- |
| `IVF(num_points, dim, num_clusters, bits, ...)`, then `construct(data, centroids, cluster_ids)` | `IvfIndex(dim, num_clusters, options)`, `set_centroids(centroids)`, then `add(data, num_points, add_options)` with `add_options.cluster_ids` |
| `HierarchicalNSW(max_elements, dim, bits, M, ef_construction, seed, metric)`, then `construct(...)` | `HnswIndex(dim, options)` with `options.num_clusters`, then the same two calls; capacity grows on `add` |
| `QuantizedGraph(num_points, dim, max_degree, ...)` with `quantization_bits` 0 for raw vectors, then `QGBuilder(graph, ef_construction, data).build()` | `SymqgIndex(dim, options)` with `options.nbits` 32 for raw vectors, then `add(data, num_points)` |
| `IVF::search`, `IVF::search_batch`, `HierarchicalNSW::search` (nested vectors), `set_ef` with `QuantizedGraph::search_batch` | `search(queries, num_queries, k, ids, distances, search_options)` |
| `Index index; index.load(filename);` | `auto index = IvfIndex::load(filename);`, and likewise `HnswIndex::load` and `SymqgIndex::load` |
| `PID` results, `kPidMax` for a missing IVF or SymphonyQG result | `Id` results, `kNoId` for a missing result |
| `dimension()`, `metric_type()`, `initializer_type()`, `max_elements()` (HNSW), `degree_bound()`, `quantization_bits()` (0 for raw), `num_vertices()`; `IVF::reserve`, `HierarchicalNSW::resize` | `dim()`, `metric()`, `routing()`, `capacity()`, `max_degree()`, `nbits()` (32 for raw), `num_points()`; `reserve_cluster`, `reserve` |
| `IVF::train(num_points, data, cp)`, `HierarchicalNSW::train(num_clusters, num_points, data, cp)`, `HierarchicalNSW::set_centroids(num_clusters, centroids)`; `add` returning `void` or every label | `train(data, num_points)` with `cp()`, `set_centroids(centroids)`, `add` returning the first `Id` |

Four differences are not renames: `HnswIndex::add` uses every CPU unless `num_threads` is set; `HnswIndex::search` writes all `k` slots, padding with `kNoId` and `+inf`, where `HierarchicalNSW::search` returns only the pairs found; a seed is drawn unless given, where `HierarchicalNSW` defaults to 100; and default thread counts are capped by `OMP_NUM_THREADS`. The new classes do not offer the matrix rotator, rebuilding with `IVF::construct`, shrinking with `HierarchicalNSW::resize`, `HierarchicalNSW::cluster_id_of`, `QuantizedGraph::search_with_scratch` with its warm-start hint, `QGBuilder`'s multi-pass `build(n)`, `reset`, `refine`, `avg_degree`, `check_dup` and `cache_vectors`, or `padded_dim()`; use the existing classes for them.

- The C++ examples in `sample/cpp` and the IVF `add` hot-path benchmark use the new classes. `symqg_indexing` takes `nbits` (32 for raw vectors, 4 or 8, default 32) and still reads 0 as raw vectors; it no longer prints stage times or the average degree. Its optional thread count 0 selects every available CPU. The default thread counts of `symqg_indexing` and `ivf_rabitq_indexing` are now capped by `OMP_NUM_THREADS`, which they ignored.

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
- `symqg::QGInitialization` moves from `qg_builder.hpp` to `rabitqlib/index/symqg/types.hpp`, which `qg.hpp`, and through it `qg_builder.hpp`, includes.
- The vendored hnswlib adds `saveIndex(std::ostream&)` and `loadIndex(std::istream&, size_t bytes, ...)`; the path overloads are unchanged.
