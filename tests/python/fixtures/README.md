`ivf_legacy_4bit.index` was saved by the IVF implementation before raw-vector
storage was introduced (legacy, unversioned x86-64 format). It contains three
vectors: `[0, 0, 0]`, `[1, 2, 3]`, and `[-1, -2, -3]`, each zero-padded to
64 dimensions, in one cluster with a zero centroid, four-bit L2 quantization,
and FHT/Kac rotation.
The zero query has squared distances 0, 14, and 14. This fixture verifies legacy
loading and migration to the current versioned format without changing search results.

`ivf_legacy_padding65_1bit.index` and `ivf_raw_v1_padding65.index` were saved before
the adaptive-padding change, using the historical unversioned quantized and raw v1
formats respectively. They have original dimension 65, padded dimension 128, one
zero centroid, and the same three vectors above with zeros in all remaining
coordinates. Their widths are 1 and 32 bits, with L2 distance and FHT/Kac rotation.
They verify that loading does not substitute the new 96-coordinate default.

The IVF tests also query `q[i] = (i % 9 - 4) / 4` before and after migration.
The expected IDs and distances were recorded using the pad64 implementation at
commit `a065785`, covering nonzero query rotation and quantized distance estimation
as well as raw-vector reranking.

`hnsw_legacy_padding65_4bit.index` was written before HNSW switched to 32-coordinate
padding: dimension 65, padded dimension 128, 4 total bits, L2, capacity 8, M=2,
ef_construction=8. Its three vectors and zero centroid match the dimension-65 IVF
fixtures above. A zero query returns IDs 0, 1, 2 with distances 0, 14, 14. Tests
check loading, preserving the old padded domain through save/load, and insertion.
