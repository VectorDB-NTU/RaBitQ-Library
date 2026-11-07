# Unreleased

## Index files

- IVF-RaBitQ `save` writes [index format 2](https://github.com/VectorDB-NTU/RaBitQ-Library/blob/main/docs/formats.md#index-format-2), a single file with a header, a section table and checksummed metadata. HNSW centroid routing is stored inside it, so `save` no longer writes a `<file>.hnsw` sidecar. A sidecar left by an earlier save is not removed.
- Releases up to 0.8 cannot read these files. Keep a copy saved by 0.8 if you may downgrade.
- The saved bytes now include the HNSW routing graph, which is reproducible only when `train`, `set_centroids` or the first `add` builds it with `num_threads=1`; saving the same index twice still gives identical files.
- Files written by every earlier release still load, including the `.hnsw` sidecars of HNSW-routed files; saving one writes format 2 with the same rotation, codes and results.
- Loading a damaged or mismatched format-2 file raises `RuntimeError` with a message naming the problem, for example `IVF index file is 6916 bytes, but its sections end at 6852` for trailing bytes.
- `save` rejects IVF-RaBitQ indexes of more than 65,536 dimensions.
- Errors in an HNSW routing graph name the graph rather than the sidecar: `Truncated HNSW centroid graph` and `HNSW centroid graph changed while loading` replace their `sidecar` forms. A missing sidecar of an earlier file still raises `Cannot open HNSW centroid sidecar`.
- A format-2 file that uses read-only-compatible features unknown to this release loads for search only: `add`, `reserve`, `remove` and `save` raise until `construct` rebuilds the index.

## C++ changes

- `ivf::Initializer` and its subclasses replace `save(std::ofstream&, const char*)` and `load(std::ifstream&, const char*)` with `write(std::ostream&)` and `read(std::istream&, uint64_t bytes)`. The HNSW initializer no longer writes a sidecar itself.
- The vendored hnswlib adds `saveIndex(std::ostream&)` and `loadIndex(std::istream&, size_t bytes, ...)`; the path overloads are unchanged.
