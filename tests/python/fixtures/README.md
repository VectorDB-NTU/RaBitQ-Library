`ivf_legacy_4bit.index` was saved by the IVF implementation before raw-vector
storage was introduced (legacy, unversioned x86-64 format). It contains three
vectors: `[0, 0, 0]`, `[1, 2, 3]`, and `[-1, -2, -3]`, each zero-padded to
64 dimensions, in one cluster with a zero centroid, four-bit L2 quantization,
and FHT/Kac rotation.
The zero query has squared distances 0, 14, and 14. This fixture verifies that
adding versioned raw-vector storage preserves legacy loading and serialization.
