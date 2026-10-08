"""Explicit centroid routing, persistence, and legacy initializer inference."""

from pathlib import Path

import numpy as np
import pytest
from rabitqlib import IvfIndex


@pytest.mark.parametrize("metric", ["l2", "ip"])
@pytest.mark.parametrize("bits", list(range(1, 10)) + [32])
@pytest.mark.parametrize("initializer", ["auto", "flat", "flat_rabitq", "hnsw"])
def test_initializer_roundtrip_and_updates(tmp_path, metric, bits, initializer):
    rng = np.random.default_rng(773)
    dim, count = 65, 33
    data = rng.standard_normal((count + 1, dim)).astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    index = IvfIndex(dim, count, count, bits, metric, initializer=initializer)
    resolved = "flat" if initializer == "auto" else initializer
    assert index.initializer == resolved
    index.build(data[:count], data[:count], np.arange(count, dtype=np.uint32), 2)
    queries = data[[0, 17, 32]]
    expected = index.search(queries, 3, 7, num_threads=2)
    np.testing.assert_array_equal(expected[0][:, 0], [0, 17, 32])
    path = tmp_path / "索引.index"
    index.save(str(path))
    payload = path.read_bytes()
    assert int.from_bytes(payload[8:12], "little") == 2
    assert (
        int.from_bytes(payload[24:28], "little")
        == {
            "flat": 1,
            "flat_rabitq": 2,
            "hnsw": 3,
        }[resolved]
    )
    assert Path(str(path) + ".hnsw").exists() == (resolved == "hnsw")
    loaded = IvfIndex.load(str(path))
    assert loaded.initializer == resolved
    actual = loaded.search(queries, 3, 7, num_threads=2)
    for left, right in zip(expected, actual):
        np.testing.assert_array_equal(left, right)
    loaded.save(str(path))
    assert path.read_bytes() == payload
    for obj in (index, loaded):
        obj.add(data[count:], num_threads=2)
        obj.remove(np.array([0], dtype=np.uint32))
        assert obj.initializer == resolved
    for left, right in zip(index.search(queries, 3, 7), loaded.search(queries, 3, 7)):
        np.testing.assert_array_equal(left, right)
    loaded.save(str(path))
    assert IvfIndex.load(str(path)).initializer == resolved


@pytest.mark.parametrize(
    "clusters,expected",
    [
        (4999, "flat"),
        (5000, "flat_rabitq"),
        (19999, "flat_rabitq"),
        (20000, "flat_rabitq"),
        (59999, "flat_rabitq"),
        (60000, "hnsw"),
    ],
)
def test_auto_initializer_thresholds(clusters, expected):
    assert IvfIndex(64, 1, clusters, 1).initializer == expected
    for mode in ("flat", "flat_rabitq", "hnsw"):
        assert IvfIndex(64, 1, clusters, 1, initializer=mode).initializer == mode


@pytest.mark.parametrize("bits", [1, 32])
@pytest.mark.parametrize("clusters", [3, 5000, 19999, 20000])
@pytest.mark.parametrize("format_version", ["v1", "legacy"])
def test_legacy_formats_infer_historical_initializer(
    tmp_path, bits, clusters, format_version
):
    rng = np.random.default_rng(43)
    centroids = rng.standard_normal((clusters, 64)).astype(np.float32)
    historical = "flat" if clusters < 20000 else "hnsw"
    index = IvfIndex(64, 3, clusters, bits, initializer=historical)
    index.build(centroids[:3], centroids, np.arange(3, dtype=np.uint32), 2)
    path = tmp_path / "v1.index"
    index.save(str(path))
    # v2 must honor the stored type even when today's Auto would choose differently.
    assert IvfIndex.load(str(path)).initializer == historical
    payload = bytearray(path.read_bytes())
    if format_version == "v1":
        # v1 has the same payload but no initializer field after padded_dim.
        del payload[24:28]
        payload[8:12] = (1).to_bytes(4, "little")
    else:
        # Dimension 64 also matches the padding inferred by both older formats.
        payload = payload[28:]
        if bits == 32:
            payload = b"RBFQRAW1" + (1).to_bytes(4, "little") + payload
    path.write_bytes(payload)
    loaded = IvfIndex.load(str(path))
    assert loaded.initializer == index.initializer
    for left, right in zip(
        index.search(centroids[:3], 1, 2), loaded.search(centroids[:3], 1, 2)
    ):
        np.testing.assert_array_equal(left, right)


def test_initializer_rejects_invalid_argument():
    with pytest.raises(ValueError, match="initializer must be"):
        IvfIndex(64, 3, 1, 1, initializer="unknown")


@pytest.mark.parametrize("initializer", ["flat", "flat_rabitq"])
def test_explicit_flat_routing_above_historical_threshold(tmp_path, initializer):
    centroids = np.zeros((20000, 64), dtype=np.float32)
    centroids[17, 0] = 1
    index = IvfIndex(64, 1, len(centroids), 32, initializer=initializer)
    index.build(centroids[17:18], centroids, np.array([17], dtype=np.uint32), 2)
    path = tmp_path / "large.index"
    index.save(str(path))
    assert not Path(str(path) + ".hnsw").exists()
    for obj in (index, IvfIndex.load(str(path))):
        assert obj.initializer == initializer
        ids, distances = obj.search(centroids[17:18], 1, 1)
        np.testing.assert_array_equal(ids, [[0]])
        np.testing.assert_array_equal(distances, [[0]])


@pytest.mark.parametrize("damage", ["auto", "unknown", "truncated_initializer"])
def test_initializer_rejects_invalid_persistence(tmp_path, damage):
    data = np.zeros((3, 65), dtype=np.float32)
    index = IvfIndex(65, 3, 1, 1, initializer="flat_rabitq")
    index.build(data, data[:1], np.zeros(3, dtype=np.uint32))
    path = tmp_path / "bad.index"
    index.save(str(path))
    payload = bytearray(path.read_bytes())
    if damage in ("auto", "unknown"):
        payload[24:28] = (0 if damage == "auto" else 99).to_bytes(4, "little")
    else:
        # The rotator is followed by raw centroids, mean, and the physical code batch.
        payload = payload[: 28 + 4 * 8 + 2 + 8 + 96 // 2 + 96 * 4 + 1]
    path.write_bytes(payload)
    with pytest.raises(RuntimeError):
        IvfIndex.load(str(path))


@pytest.mark.parametrize("clusters", [5000, 20000, 59999])
def test_auto_flat_rabitq_persists_actual_routing(tmp_path, clusters):
    centroids = np.zeros((clusters, 64), dtype=np.float32)
    centroids[17, 0] = 1
    index = IvfIndex(64, 1, clusters, 32)
    index.build(centroids[17:18], centroids, np.array([17], dtype=np.uint32), 2)
    path = tmp_path / "auto.index"
    index.save(str(path))
    assert not Path(str(path) + ".hnsw").exists()
    for obj in (index, IvfIndex.load(str(path))):
        assert obj.initializer == "flat_rabitq"
        ids, distances = obj.search(centroids[17:18], 1, 1)
        np.testing.assert_array_equal(ids, [[0]])
        np.testing.assert_array_equal(distances, [[0]])
