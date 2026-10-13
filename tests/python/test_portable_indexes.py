"""Portable kernels preserve metric, padded-domain, and persistence contracts."""

import numpy as np
import pytest
from rabitqlib import HnswIndex, IvfIndex, SymqgIndex


def check_index_round_trip(tmp_path, metric, dim, kind, bits):
    rng = np.random.default_rng(764)
    data = rng.standard_normal((96, dim)).astype(np.float32)
    # Unit vectors make self-query the exact winner for both metrics.
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    if kind == "ivf":
        index = IvfIndex(dim, len(data), 1, nbits=bits, metric=metric)
        search_args = {"nprobe": 1}
    elif kind == "hnsw":
        index = HnswIndex(
            dim,
            len(data),
            M=16,
            ef_construction=96,
            nbits=bits,
            metric=metric,
            random_seed=42,
        )
        search_args = {"ef": 96}
    else:
        index = SymqgIndex(dim, max_degree=32, quantization_bits=bits, metric=metric)
        search_args = {"ef": 96}
    if kind == "qg":
        index.build(data, ef_construction=96)
    else:
        index.build(
            data, data.mean(axis=0, keepdims=True), np.zeros(len(data), np.uint32)
        )
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
    if kind == "ivf":
        assert payload[:8] == b"RABQIDX1"
        multiple = 32
        stored_padded = int.from_bytes(payload[16:24], "little")
    elif kind == "qg":
        multiple = 32
        offset = (12 if bits else 0) + 3 * np.dtype(np.uintp).itemsize
        stored_padded = int.from_bytes(payload[offset : offset + 8], "little")
    else:
        multiple = 32
        stored_padded = int.from_bytes(payload[24:32], "little")
    assert stored_padded == (dim + multiple - 1) // multiple * multiple
    loaded = type(index).load(str(path))
    assert loaded.dim == dim
    assert loaded.metric == metric
    assert (loaded.quantization_bits if kind == "qg" else loaded.nbits) == bits
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
    offset = {"ivf": 16, "hnsw": 24, "qg": 12 + 3 * np.dtype(np.uintp).itemsize}[kind]
    # 80 is a valid pad16 dimension but unsupported by the pad32 kernels.
    payload[offset : offset + 8] = (80).to_bytes(8, "little")
    path.write_bytes(payload)
    cls = {"ivf": IvfIndex, "hnsw": HnswIndex, "qg": SymqgIndex}[kind]
    message = {
        "ivf": "Invalid padded dimension in IVF index file",
        "hnsw": "HNSW: invalid or truncated index file",
        "qg": "Invalid padded dimension in quantized graph file",
    }[kind]
    with pytest.raises(RuntimeError, match=f"^{message}$"):
        cls.load(str(path))
