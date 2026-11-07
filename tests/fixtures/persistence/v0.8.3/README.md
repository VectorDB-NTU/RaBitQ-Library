# IVF-RaBitQ fixtures from v0.8.3

These files were written by release **v0.8.3**, commit
`dabfdb8c200a6f155df4edd0b9c825e6728b14a1`, using [generate.cpp](generate.cpp).
[manifest.json](manifest.json) records the producer and SHA-256 hashes.
Keep these files unchanged; tests never regenerate them with the current writer.

They freeze the last IVF-RaBitQ layout before index format 2: `RABQIDX1` version 2,
which stores the centroid-routing type, and the hnswlib sidecar `<file>.hnsw` that
HNSW routing writes beside the index. The five cases are:

| File | Bits | Metric | Rotator | Routing |
| --- | --- | --- | --- | --- |
| `ivf4_flat_l2` | 4 | L2 | FHT/Kac | Flat |
| `ivf7_flatrabitq_ip` | 7 | inner product | FHT/Kac | FlatRaBitQ |
| `ivf4_hnsw_l2` | 4 | L2 | FHT/Kac | HNSW, with sidecar |
| `ivf32_hnsw_ip` | raw (`nbits=32`) | inner product | FHT/Kac | HNSW, with sidecar |
| `ivf1_flat_matrix_l2` | 1 | L2 | matrix | Flat |

Each index holds 130 points in 65 original dimensions (96 padded) across four clusters
of 0, 1, 32 and 97 points, with point 0 removed, as in
[pre-ivf-per-cluster-storage](../pre-ivf-per-cluster-storage/README.md). The
`.expected` files hold reference results of a search that probes every cluster.

## Deliberate regeneration

Build the generator only against an isolated copy of the release:

```bash
mkdir -p build/fixtures/v0.8.3
git archive v0.8.3 | tar -x -C build/fixtures/v0.8.3
cmake -S tests/fixtures/persistence/v0.8.3 -B build/fixtures/v0.8.3-generator \
  -DRABITQ_HISTORICAL_SOURCE="$PWD/build/fixtures/v0.8.3" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/fixtures/v0.8.3-generator --target generate_fixtures
```

Run `generate_fixtures` with a new output directory and `OMP_NUM_THREADS=1`.
Rotation is seeded from `random_device`, so the bytes differ on every run; each index,
its sidecar and its `.expected` file form a fixed set. Record new hashes in the manifest
before adopting them.
