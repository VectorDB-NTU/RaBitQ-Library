# IVF-RaBitQ fixtures from before per-cluster storage

These files were written by `main` at commit
`64fc943e0e92fa7e18eec1477050b23342a976f8`, the last commit where IVF-RaBitQ kept every
cluster in one shared buffer, using [generate.cpp](generate.cpp).
[manifest.json](manifest.json) records the producer and SHA-256 hashes.
Keep these files unchanged; tests never regenerate them with the current writer.

The two cases cover four-bit and raw-vector (`nbits=32`) IVF-RaBitQ for L2.
Each index holds 130 points in 65 original dimensions (96 padded) across
four clusters of 0, 1, 32 and 97 points, so the file has an empty cluster, a
single-point cluster, a full FastScan batch and a partial tail batch. Point 0 is
removed. The `.expected` files hold reference results of a search that probes
every cluster.

The tests load each file, compare search results, and require that saving it
again reproduces the original bytes. Because the clusters differ in size, a
reader or writer that orders the batch data, extra-bit data and IDs differently
from the shared-buffer layout fails these checks.

## Deliberate regeneration

Build the generator only against an isolated copy of the pinned commit:

```bash
mkdir -p build/fixtures/pre-ivf-per-cluster-storage
git archive 64fc943 | tar -x -C build/fixtures/pre-ivf-per-cluster-storage
cmake -S tests/fixtures/persistence/pre-ivf-per-cluster-storage \
  -B build/fixtures/pre-ivf-generator \
  -DRABITQ_HISTORICAL_SOURCE="$PWD/build/fixtures/pre-ivf-per-cluster-storage" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/fixtures/pre-ivf-generator --target generate_fixtures
```

Run `generate_fixtures` with a new output directory. Rotation is seeded from
`random_device`, so the bytes differ on every run; each index and its `.expected`
file form a fixed pair. Record new hashes in the manifest before adopting them.
