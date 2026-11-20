# Index fixtures from v0.9.0

These files were written by release **v0.9.0** using [generate.cpp](generate.cpp), built
from commit `9c7c4a5` on the release branch, whose sources the `v0.9.0` tag contains.
[manifest.json](manifest.json) records the producer and SHA-256 hashes.
Keep these files unchanged; tests never regenerate them with the current writer.

They freeze [index format 2](../../../../docs/formats.md#index-files-format-2) as its
first release wrote it: each file records the seed (42) and the writer version, and
together they use every section kind 0.9 writes and each metric, including cosine,
for every index type. `generate.cpp` writes them through `IvfIndex`, `HnswIndex` and
`SymqgIndex`, and through `ivf::IVF` and `symqg::QuantizedGraph` for the matrix rotator,
which the new classes do not offer.

| File | Bits | Metric | Rotator | Notes |
| --- | --- | --- | --- | --- |
| `ivf4_flat_l2` | 4 | L2 | FHT/Kac | Flat routing |
| `ivf7_flatrabitq_cosine` | 7 | cosine | FHT/Kac | FlatRaBitQ routing |
| `ivf32_hnsw_ip` | raw (`nbits=32`) | inner product | FHT/Kac | HNSW routing graph stored in the file |
| `ivf1_flat_matrix_l2` | 1 | L2 | matrix | Flat routing, written by `ivf::IVF` |
| `hnsw1_l2` | 1 | L2 | FHT/Kac | M = 8, `ef_construction` 64 |
| `hnsw5_cosine` | 5 | cosine | FHT/Kac | As above |
| `hnsw9_ip` | 9 | inner product | FHT/Kac | As above |
| `qg32_ip` | raw (`nbits=32`) | inner product | FHT/Kac | Degree 32, PiPNN, `ef_construction` 64 |
| `qg4_cosine` | 4 | cosine | FHT/Kac | As above |
| `qg8_l2` | 8 | L2 | FHT/Kac | As above |
| `qg4_matrix_l2` | 4 | L2 | matrix | As above, written by `symqg::QuantizedGraph` |

Each index holds the 130 points in 65 original dimensions (96 padded) of the
[v0.8.3](../v0.8.3/README.md) fixtures. The IVF-RaBitQ and HNSW-RaBitQ files use its four
clusters of 0, 1, 32 and 97 points, with points 0 and 32 removed: point 0 is the only
point of its cluster, and point 32 is the nearest neighbour of the third query, so the
SymphonyQG results, which have no removal, include it. The cosine indexes get the
centroids scaled to unit length, as their data is. The `.expected` files hold
reference results: IVF-RaBitQ probes every cluster, and the graph indexes search with
`ef_search` 64.

## Deliberate regeneration

Build the generator only against an isolated copy of the release:

```bash
mkdir -p build/fixtures/v0.9.0
git archive v0.9.0 | tar -x -C build/fixtures/v0.9.0
cmake -S tests/fixtures/persistence/v0.9.0 -B build/fixtures/v0.9.0-generator \
  -DRABITQ_HISTORICAL_SOURCE="$PWD/build/fixtures/v0.9.0" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/fixtures/v0.9.0-generator --target generate_fixtures
```

Run it with a new output directory and `OMP_NUM_THREADS=1`. Every index is seeded and
built with one thread, so two runs on the producer in the manifest write identical
bytes; another compiler or ISA may change the results. Each index and its `.expected`
file form a fixed pair. Record new hashes in the manifest before adopting them.
