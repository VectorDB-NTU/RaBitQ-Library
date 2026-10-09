"""Frozen historical writer outputs must remain searchable with current wheels."""

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest
from rabitqlib import HnswIndex, IvfIndex, SymqgIndex

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "persistence"
NAMES = [
    f"{kind}_{metric}"
    for kind in ("ivf4", "ivf32", "hnsw4", "qg0", "qg4", "qg8")
    for metric in ("l2", "ip")
]


def reference(name):
    path = FIXTURES / "v0.5.2" / f"{name}.index"
    lines = path.with_suffix(".expected").read_text().splitlines()
    count, dim, nq, k = map(int, lines[0].split())
    queries = np.fromstring(lines[1], sep=" ", dtype=np.float32).reshape(nq, dim)
    results = np.loadtxt(lines[2:])
    ids = results[:, 0].astype(np.uint32).reshape(nq, k)
    distances = results[:, 1].astype(np.float32).reshape(nq, k)
    cls = (
        IvfIndex
        if name.startswith("ivf")
        else (HnswIndex if name.startswith("hnsw") else SymqgIndex)
    )
    options = {"nprobe": 1} if name.startswith("ivf") else {"ef": 64}
    return path, cls, count, queries, ids, distances, options


def test_fixture_hashes_match_pinned_manifest():
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    for filename, digest in manifest["sha256"].items():
        assert hashlib.sha256((FIXTURES / filename).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("name", NAMES)
def test_historical_load_search_and_resave(name, tmp_path):
    path, cls, count, queries, ids, distances, options = reference(name)
    original = cls.load(str(path))
    destination = tmp_path / "migrated.index"
    original.save(str(destination))
    for index in (original, cls.load(str(destination))):
        assert index.dim == 65
        assert index.metric == name.rsplit("_", 1)[1]
        assert (index.max_elements if cls is IvfIndex else index.num_points) == count
        actual_ids, actual_distances = index.search(queries, 5, **options)
        np.testing.assert_array_equal(actual_ids, ids)
        # Distances may differ across AVX2, AVX-512, NEON, and compiler reductions.
        np.testing.assert_allclose(actual_distances, distances, rtol=2e-4, atol=2e-4)
        if cls in (IvfIndex, HnswIndex):
            assert not np.any(actual_ids == 0)  # Historical removal markers survive.
    damaged = tmp_path / "truncated.index"
    damaged.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(RuntimeError):
        cls.load(str(damaged))


@pytest.mark.parametrize("name", ["ivf32_l2", "qg4_l2", "qg8_l2"])
def test_unknown_historical_version_is_rejected(name, tmp_path):
    path, cls, *_ = reference(name)
    damaged = bytearray(path.read_bytes())
    struct.pack_into("<I", damaged, 8, 0xFFFFFFFF)
    destination = tmp_path / "unknown-version.index"
    destination.write_bytes(damaged)
    message = (
        "Unsupported raw IVF index version"
        if cls is IvfIndex
        else "Unsupported QuantizedGraph file version"
    )
    with pytest.raises(RuntimeError, match=f"^{message}$"):
        cls.load(str(destination))


@pytest.mark.parametrize("name", ["ivf4_l2", "ivf4_ip", "ivf32_l2", "ivf32_ip"])
def test_ivf_v1_header_uses_historical_payload_and_padding(name, tmp_path):
    path, cls, _, queries, ids, distances, options = reference(name)
    raw = name.startswith("ivf32")
    # The intermediate IVF v1 layout prepends explicit padding to the same
    # historical payload. This synthetic header test complements frozen files;
    # it never calls the current writer to manufacture its input.
    payload = path.read_bytes()[12:] if raw else path.read_bytes()
    header = struct.pack("<QIIQ", 0x3158444951424152, 1, int(raw), 128)
    destination = tmp_path / "ivf-v1.index"
    destination.write_bytes(header + payload)
    index = cls.load(str(destination))
    assert index.initializer == "flat"
    actual_ids, actual_distances = index.search(queries, 5, **options)
    np.testing.assert_array_equal(actual_ids, ids)
    np.testing.assert_allclose(actual_distances, distances, rtol=2e-4, atol=2e-4)
    damaged = bytearray(header + payload)
    struct.pack_into("<I", damaged, 8, 99)
    destination.write_bytes(damaged)
    with pytest.raises(RuntimeError, match="^Unsupported IVF index version$"):
        cls.load(str(destination))
    damaged = bytearray(header + payload)
    struct.pack_into("<Q", damaged, 16, 80)
    destination.write_bytes(damaged)
    with pytest.raises(
        RuntimeError, match="^Invalid padded dimension in IVF index file$"
    ):
        cls.load(str(destination))
