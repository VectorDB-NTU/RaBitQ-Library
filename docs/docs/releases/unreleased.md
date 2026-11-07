# Unreleased

## Index files

- IVF-RaBitQ, HNSW-RaBitQ and SymphonyQG `save` write [index format 2](https://github.com/VectorDB-NTU/RaBitQ-Library/blob/main/docs/formats.md#index-format-2), a single file with a header, a section table and checksummed metadata.
- Releases up to 0.8 cannot read these files. Keep a copy saved by 0.8 if you may downgrade.
- Files written by every earlier release still load, including the `.hnsw` sidecars of HNSW-routed IVF files. Saving one writes format 2 with the same rotation, codes and results.
- IVF-RaBitQ stores HNSW centroid routing inside the file, so `save` no longer writes a `<file>.hnsw` sidecar. A sidecar left by an earlier save is not removed.
- The saved bytes now include that routing graph. It is reproducible only when `train`, `set_centroids` or the first `add` builds it with `num_threads=1`. Saving the same index twice still gives identical files.
- `save` rejects IVF-RaBitQ indexes of more than 65,536 dimensions.
- A format-2 file that uses read-only-compatible features unknown to this release loads for search only: operations that change or save the index raise. Rebuilding with IVF-RaBitQ `construct`, or a new SymphonyQG build, makes it writable again.

## Python changes

- Loading a format-2 file keeps the `seed` of every index, and `ef_construction` and `init` of SymphonyQG. They were `None` after `load`, and still are for files from releases up to 0.8.
- With a recorded seed, `cp.seed` after `load` follows it as for a new index: the seed for IVF-RaBitQ, its low 32 bits for HNSW-RaBitQ. It was the k-means default.
- A new exception, `rabitqlib.FormatError`, subclasses both `ValueError` and `RuntimeError`:
  - Every index `load` raises it for a truncated, corrupt, mismatched or unsupported file, in format 2 and in every earlier layout. These were `RuntimeError`, which still catches them. A file that cannot be opened still raises a plain `RuntimeError`.
  - `quant.FormatError` is now the same class. It was a subclass of `ValueError` only, which still catches it.
- Messages for damaged or mismatched files change:
  - A format-2 file names the problem, for example `IVF index file is 6916 bytes, but its sections end at 6852` for trailing bytes.
  - Errors in an HNSW routing graph name the graph rather than the sidecar: `Truncated HNSW centroid graph` and `HNSW centroid graph changed while loading` replace their `sidecar` forms.
  - A missing sidecar of an earlier file still raises `Cannot open HNSW centroid sidecar`.
  - A failed HNSW-RaBitQ write raises `HNSW index file cannot be written` instead of `HNSW failed to write index file`.

## C++ changes

- Loading an index throws `rabitqlib::FormatError`, a subclass of `std::runtime_error` in the new header `rabitqlib/index/format_error.hpp`, for a damaged, mismatched or unsupported file. Failing to open the file still throws `std::runtime_error`. A SymphonyQG file of an earlier layout with invalid settings threw `std::invalid_argument`; it now throws `FormatError`.

- `ivf::Initializer` and its subclasses:
  - `save(std::ofstream&, const char*)` becomes `write(std::ostream&)`, and `load(std::ifstream&, const char*)` becomes `read(std::istream&, uint64_t bytes)`.
  - The HNSW initializer no longer writes a sidecar itself.
- New accessors report what format-2 files record, or no value when unknown:
  - `IVF::seed()`;
  - `HierarchicalNSW::seed()`;
  - `QuantizedGraph<float>::seed()`, `ef_construction()` and `initialization()`.
- `symqg::QGInitialization` moves from `qg_builder.hpp` to `qg.hpp`, which `qg_builder.hpp` includes.
- The vendored hnswlib adds `saveIndex(std::ostream&)` and `loadIndex(std::istream&, size_t bytes, ...)`; the path overloads are unchanged.
