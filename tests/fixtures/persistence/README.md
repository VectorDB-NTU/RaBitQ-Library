# Historical persistence fixtures

These files were written by release **v0.5.2**, commit
`d929e30dbb7ded52db1830736a2356fbc7dba99a`, using [generate.cpp](generate.cpp).
Tests load committed bytes; they do not generate indexes with the current writer.
[manifest.json](manifest.json) records the producer and SHA-256 hashes of every
index and its reference search results. Keep existing fixtures unchanged when
adding a new format or release.

The twelve cases cover L2 and inner product for:

- IVF four-bit legacy files without a header, and raw-vector IVF v1 files;
- HNSW four-bit files, including persisted removal of point 0;
- SymphonyQG legacy raw files and versioned four/eight-bit files.

IVF also persists removal of point 0. Each file contains 33 points in 65 original
dimensions and the historical **128-dimensional padded domain**. This differs
from current construction's 96 dimensions and detects accidental regeneration
of rotation or padding on load. The `.expected` files hold the point count,
dimension, query count, k, original queries, and ordered ID/distance pairs from
the historical implementation.

Tests require identical result IDs and allow distance error of
`2e-4 * max(1, abs(reference))` in C++, or `rtol=atol=2e-4` in Python, to account
for compiler and SIMD reductions. The native formats contain `size_t`: these
fixtures target the supported little-endian, 64-bit platforms, not a portable
interchange format for arbitrary architectures. Python also tests synthetic IVF
v1 headers around the frozen historical payload, unknown versions, and invalid
padding. That header test is distinct from release-produced fixtures.

The small IVF fixtures use Flat centroid routing. Current round-trip and failure
tests exercise explicit HNSW routing and its sidecar; historical auto-selected
HNSW routing at 20,000 clusters is not represented by these small fixtures.

## Deliberate regeneration

Use an isolated copy of the pinned release. Do not point this build at the
current source tree. For example, from the repository root:

```bash
mkdir -p build/fixtures/v0.5.2
git archive v0.5.2 | tar -x -C build/fixtures/v0.5.2
cmake -S tests/fixtures/persistence -B build/fixtures/generator \
  -DRABITQ_HISTORICAL_SOURCE="$PWD/build/fixtures/v0.5.2" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/fixtures/generator --config Release --parallel 2
```

Run the resulting `generate_fixtures` executable with a **new output directory**.
On Windows it is under `build/fixtures/generator/Release/`. Review the new
artifacts and record their hashes and compiler/ISA in a separate manifest before
adopting them. Some historical constructors seed rotation from `random_device`,
so rebuilding does not promise byte-identical files; each committed index and its
reference results form a fixed pair. `.gitattributes` prevents newline conversion
from changing fixture hashes on Windows.
