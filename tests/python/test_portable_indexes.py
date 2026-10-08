"""Portable kernels preserve metric, padded-domain, and persistence contracts."""

import numpy as np
import pytest
from rabitqlib import HnswIndex, IvfIndex, SymqgIndex


@pytest.mark.parametrize("metric", ["l2", "ip"])
@pytest.mark.parametrize("dim", [64, 65, 81, 97])
@pytest.mark.parametrize(
    "kind,bits",
    [(kind, bits) for kind in ("ivf", "hnsw") for bits in range(1, 10)]
    + [("ivf", 32)]
    + [("qg", bits) for bits in (0, 4, 8)],
)
def test_portable_index_round_trip(tmp_path, metric, dim, kind, bits):
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
    actual_ids, actual_distances = loaded.search(queries, k=5, **search_args)
    np.testing.assert_array_equal(actual_ids, ids)
    np.testing.assert_array_equal(actual_distances, distances)


@pytest.mark.parametrize("metric", ["l2", "ip"])
@pytest.mark.parametrize("bits", range(1, 10))
def test_hnsw_multi_block_kernel_round_trip(tmp_path, metric, bits):
    # 960 coordinates exercise a full 512-bit query block and its 448-bit tail.
    test_portable_index_round_trip(tmp_path, metric, 960, "hnsw", bits)


@pytest.mark.parametrize("metric", ["l2", "ip"])
@pytest.mark.parametrize("dim", [193, 960])
@pytest.mark.parametrize(
    "kind,bits",
    [("ivf", 5), ("ivf", 32), ("hnsw", 5), ("qg", 0), ("qg", 4), ("qg", 8)],
)
def test_rotation_padded_domain_round_trip(tmp_path, metric, dim, kind, bits):
    test_portable_index_round_trip(tmp_path, metric, dim, kind, bits)


@pytest.mark.parametrize("metric", ["l2", "ip"])
@pytest.mark.parametrize("dim", [511, 513])
@pytest.mark.parametrize("kind,bits", [("ivf", 3), ("ivf", 5), ("hnsw", 5), ("qg", 0)])
def test_query_preparation_block_boundaries(tmp_path, metric, dim, kind, bits):
    # Exercise both sides of the 512-coordinate transpose block, standard/HACC
    # LUT quantization, and persistence with original dimensions requiring padding.
    test_portable_index_round_trip(tmp_path, metric, dim, kind, bits)


@pytest.mark.parametrize("kind,bits", [("ivf", 4), ("hnsw", 4), ("qg", 4)])
def test_rejects_development_only_padding(tmp_path, kind, bits):
    test_portable_index_round_trip(tmp_path, "l2", 65, kind, bits)
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
