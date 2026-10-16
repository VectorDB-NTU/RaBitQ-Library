# IVF snapshot immediately before the per-cluster-storage refactor

These four files were written by `rabitqlib::ivf::IVF::save()` on `main` at commit
`64fc943e0e92fa7e18eec1477050b23342a976f8`, before `ivf.hpp`/`src/index/ivf.cpp` were changed
to give each cluster its own independently-growable storage instead of one combined buffer
across all clusters. Unlike `../v0.5.2/`, which proves format stability across a historical
release, this snapshot proves this specific refactor didn't change a single byte of the format
it reads or writes: it is pinned to the implementation immediately prior to that change, not a
distant past release.

Generated with the same construction as `../generate.cpp`'s IVF cases (`kCount=33`,
`kDim=65`, `kQueries=3`, `kTopK=5`, the same LCG-seeded deterministic data, one cluster, point 0
removed after construction), covering `ivf4`/`ivf32` x `l2`/`ip`. The one-off generator used to
produce them was not committed — it is a near-identical copy of `../generate.cpp` restricted to
the IVF cases and updated to this repository's current (post-0.7.0-refactor) include paths, run
once and discarded; `manifest.json` in this directory records the producer and SHA-256 hashes of
every file, the same way `../manifest.json` does for `v0.5.2`.
