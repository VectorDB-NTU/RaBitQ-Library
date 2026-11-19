# Index fixture from v0.7.2

This file was written by release **v0.7.2**, commit
`655c9c6656318e9726e3a3908d0abe0563f7630c`, using [generate.cpp](generate.cpp).
[manifest.json](manifest.json) records the producer and SHA-256 hash.
Keep it unchanged; tests never regenerate it with the current writer.

`hnsw4_empty_l2.index` is an HNSW-RaBitQ index that was constructed and saved without
points or training: capacity 130, 65 original dimensions, 4 bits, L2, M = 8,
`ef_construction` 64, seed 42. The C++ API of releases before 0.8 allowed this; 0.8
and later releases refuse to save an empty index. Such a file has no
centroids and does not record its seed, so it loads untrained.

## Deliberate regeneration

Build the generator only against an isolated copy of the release:

```bash
mkdir -p build/fixtures/v0.7.2
git archive v0.7.2 | tar -x -C build/fixtures/v0.7.2
cmake -S tests/fixtures/persistence/v0.7.2 -B build/fixtures/v0.7.2-generator \
  -DRABITQ_HISTORICAL_SOURCE="$PWD/build/fixtures/v0.7.2" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/fixtures/v0.7.2-generator --target generate_fixtures
```

Run it with a new output directory. The file has no points, and its rotation is seeded
with 42, so the generator writes the same bytes on every run with the same toolchain
(the manifest records it); record a new hash in the manifest before adopting a
different file.
