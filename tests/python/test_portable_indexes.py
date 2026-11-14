"""Portable kernels preserve metric, padded-domain, and persistence contracts."""

import format2_file
import numpy as np
import pytest
from rabitqlib import HnswIndex, IvfIndex, SymqgIndex


def check_index_round_trip(tmp_path, metric, dim, kind, bits):
    rng = np.random.default_rng(764)
    data = rng.standard_normal((96, dim)).astype(np.float32)
    # Unit vectors make self-query the exact winner for both metrics.
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    if kind == "ivf":
        index = IvfIndex(dim, 1, nbits=bits, metric=metric)
        search_args = {"nprobe": 1}
    elif kind == "hnsw":
        index = HnswIndex(
            dim,
            num_clusters=1,
            M=16,
            ef_construction=96,
            nbits=bits,
            metric=metric,
            seed=42,
        )
        search_args = {"ef_search": 96}
    else:
        # Seeded and single-threaded, so every self-query below must succeed (#216).
        index = SymqgIndex(
            dim,
            max_degree=32,
            nbits=bits or 32,
            ef_construction=96,
            metric=metric,
            seed=42,
        )
        search_args = {"ef_search": 96}
    if kind == "qg":
        index.add(data, num_threads=1)
    else:
        index.set_centroids(data.mean(axis=0, keepdims=True))
        index.add(data, cluster_ids=np.zeros(len(data), np.uint32), num_threads=1)
    queries = data[:5].copy()
    ids, distances = index.search(queries, k=5, **search_args)
    assert ids.shape == distances.shape == (5, 5)
    assert np.isfinite(distances).all()
    assert ((ids >= 0) & (ids < len(data))).all()
    np.testing.assert_array_equal(ids[:, 0], np.arange(5))
    assert (np.diff(distances, axis=1) >= -1e-6).all()
    if (kind == "ivf" and bits == 32) or (kind == "qg" and bits == 0):
        selected = data[ids]
        expected = (
            np.sum((queries[:, None] - selected) ** 2, axis=2)
            if metric == "l2"
            else 1 - np.einsum("qd,qkd->qk", queries, selected)
        )
        np.testing.assert_allclose(distances, expected, atol=2e-5)
    path = tmp_path / "portable.index"
    index.save(str(path))
    payload = path.read_bytes()
    assert payload[:8] == format2_file.MAGIC
    assert format2_file.padded_dim(payload) == (dim + 31) // 32 * 32
    loaded = type(index).load(str(path))
    assert loaded.dim == dim
    assert loaded.metric == metric
    assert loaded.nbits == (bits or 32 if kind == "qg" else bits)
    actual_ids, actual_distances = loaded.search(queries, k=5, **search_args)
    np.testing.assert_array_equal(actual_ids, ids)
    np.testing.assert_array_equal(actual_distances, distances)


@pytest.mark.parametrize("metric", ["l2", "ip"])
@pytest.mark.parametrize(
    "kind,bits",
    [(kind, bits) for kind in ("ivf", "hnsw") for bits in range(1, 10)]
    + [("ivf", 32)]
    + [("qg", bits) for bits in (0, 4, 8)],
)
def test_supported_storage_round_trip(tmp_path, metric, kind, bits):
    # Every public width crosses the Python boundary once per metric, with padding.
    # C++ owns the width x dimension x backend numerical reference matrices.
    check_index_round_trip(tmp_path, metric, 65, kind, bits)


@pytest.mark.parametrize("metric", ["l2", "ip"])
@pytest.mark.parametrize(
    "kind,bits,dim",
    [
        ("ivf", 1, 64),
        ("ivf", 3, 511),
        ("ivf", 4, 513),
        ("ivf", 9, 97),
        ("ivf", 32, 960),
        ("hnsw", 1, 64),
        ("hnsw", 5, 511),
        ("hnsw", 5, 513),
        ("hnsw", 9, 960),
        ("qg", 0, 511),
        ("qg", 0, 513),
        ("qg", 4, 97),
        ("qg", 8, 193),
    ],
)
def test_padded_domain_boundaries(tmp_path, metric, dim, kind, bits):
    # Representative integration paths: aligned input, compact tails, both sides
    # of a 512-coordinate query block, and multi-block rotation/raw reranking.
    check_index_round_trip(tmp_path, metric, dim, kind, bits)


@pytest.mark.parametrize("kind,bits", [("ivf", 4), ("hnsw", 4), ("qg", 4)])
def test_rejects_development_only_padding(tmp_path, kind, bits):
    check_index_round_trip(tmp_path, "l2", 65, kind, bits)
    path = tmp_path / "portable.index"
    payload = bytearray(path.read_bytes())
    # 80 is a valid pad16 dimension but unsupported by the pad32 kernels.
    payload[56:64] = (80).to_bytes(8, "little")
    format2_file.reseal(payload)
    path.write_bytes(payload)
    cls = {"ivf": IvfIndex, "hnsw": HnswIndex, "qg": SymqgIndex}[kind]
    component = {"ivf": "IVF", "hnsw": "HNSW", "qg": "QuantizedGraph"}[kind]
    message = (
        f"{component} index file has padded dimension 80, which must be a multiple of "
        "32 from the dimension 65 to 65536"
    )
    with pytest.raises(RuntimeError, match=f"^{message}$"):
        cls.load(str(path))
