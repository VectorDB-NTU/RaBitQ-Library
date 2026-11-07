"""Frozen historical writer outputs must remain searchable with current wheels."""

import hashlib
import json
import struct
from pathlib import Path

import format2_file
import numpy as np
import pytest
import rabitqlib
from rabitqlib import HnswIndex, IvfIndex, SymqgIndex, quant

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
    registered = set()
    for path in sorted(FIXTURES.rglob("manifest.json")):
        manifest = json.loads(path.read_text())
        assert manifest["sha256"], path
        for filename, digest in manifest["sha256"].items():
            assert filename not in registered, filename
            registered.add(filename)
            assert (
                hashlib.sha256((FIXTURES / filename).read_bytes()).hexdigest() == digest
            ), filename
    expected = {
        path.relative_to(FIXTURES).as_posix()
        for suffix in ("*.index", "*.hnsw", "*.expected")
        for path in FIXTURES.rglob(suffix)
    }
    assert expected
    assert registered == expected


@pytest.mark.parametrize("name", NAMES)
def test_historical_load_search_and_resave(name, tmp_path):
    path, cls, count, queries, ids, distances, options = reference(name)
    original = cls.load(str(path))
    destination = tmp_path / "migrated.index"
    original.save(str(destination))
    for index in (original, cls.load(str(destination))):
        assert index.dim == 65
        assert index.metric == name.rsplit("_", 1)[1]
        assert index.num_stored_points == count
        assert index.num_points == count - (0 if cls is SymqgIndex else 1)
        assert index.max_elements >= index.num_stored_points
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
        else "QuantizedGraph file version is unsupported"
    )
    with pytest.raises(RuntimeError, match=f"^{message}$"):
        cls.load(str(destination))


@pytest.mark.parametrize("name", ["ivf4_l2", "ivf4_ip", "ivf32_l2", "ivf32_ip"])
def test_ivf_v1_header_uses_historical_payload_and_padding(name, tmp_path):
    path, cls, count, queries, ids, distances, options = reference(name)
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
    assert index.num_stored_points == count
    assert index.num_points == count - 1
    assert index.max_elements == count
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


# The last IVF layout before format 2: RABQIDX1 version 2 and the HNSW sidecar.
V083_IVF = [
    ("ivf4_flat_l2", "flat"),
    ("ivf7_flatrabitq_ip", "flat_rabitq"),
    ("ivf4_hnsw_l2", "hnsw"),
    ("ivf32_hnsw_ip", "hnsw"),
    ("ivf1_flat_matrix_l2", "flat"),
]


@pytest.mark.parametrize("name,initializer", V083_IVF)
def test_v083_ivf_loads_and_migrates_without_loss(name, initializer, tmp_path):
    path = FIXTURES / "v0.8.3" / f"{name}.index"
    lines = path.with_suffix(".expected").read_text().splitlines()
    count, dim, nq, k = map(int, lines[0].split())
    queries = np.fromstring(lines[1], sep=" ", dtype=np.float32).reshape(nq, dim)
    results = np.loadtxt(lines[2:])
    ids = results[:, 0].astype(np.uint32).reshape(nq, k)
    distances = results[:, 1].astype(np.float32).reshape(nq, k)

    index = IvfIndex.load(str(path))
    assert index.initializer == initializer
    assert index.num_stored_points == count
    assert index.num_points == count - 1
    destination = tmp_path / "migrated.index"
    index.save(str(destination))
    assert not Path(str(destination) + ".hnsw").exists()
    for loaded in (index, IvfIndex.load(str(destination))):
        actual_ids, actual_distances = loaded.search(queries, k, nprobe=4)
        np.testing.assert_array_equal(actual_ids, ids)
        np.testing.assert_allclose(actual_distances, distances, rtol=2e-4, atol=2e-4)

    # Format 2 keeps every byte: converting back reproduces the 0.8.3 files, which
    # also checks the converter that other tests use to build earlier layouts.
    main, sidecar = format2_file.to_rabqidx1(destination.read_bytes())
    assert main == path.read_bytes()
    original_sidecar = Path(str(path) + ".hnsw")
    if initializer == "hnsw":
        assert sidecar == original_sidecar.read_bytes()
    else:
        assert sidecar is None and not original_sidecar.exists()


# The last HNSW-RaBitQ and SymphonyQG layouts before format 2.
V083_GRAPHS = ["hnsw1_l2", "hnsw5_ip", "hnsw9_l2", "qg0_ip", "qg4_l2", "qg8_ip"]
V083_GRAPHS += ["qg4_matrix_l2"]


@pytest.mark.parametrize("name", V083_GRAPHS)
def test_v083_graphs_load_and_migrate(name, tmp_path):
    path = FIXTURES / "v0.8.3" / f"{name}.index"
    lines = path.with_suffix(".expected").read_text().splitlines()
    count, dim, nq, k = map(int, lines[0].split())
    queries = np.fromstring(lines[1], sep=" ", dtype=np.float32).reshape(nq, dim)
    results = np.loadtxt(lines[2:])
    ids = results[:, 0].astype(np.uint32).reshape(nq, k)
    distances = results[:, 1].astype(np.float32).reshape(nq, k)
    cls = HnswIndex if name.startswith("hnsw") else SymqgIndex

    original = cls.load(str(path))
    assert original.seed is None  # earlier layouts do not record it
    if cls is SymqgIndex:
        assert (original.ef_construction, original.init) == (None, None)
    destination = tmp_path / "migrated.index"
    original.save(str(destination))
    assert destination.read_bytes()[:8] == format2_file.MAGIC
    for index in (original, cls.load(str(destination))):
        assert index.num_stored_points == count
        assert index.num_points == count - (1 if cls is HnswIndex else 0)
        assert index.seed is None
        actual_ids, actual_distances = index.search(queries, k, ef=64)
        np.testing.assert_array_equal(actual_ids, ids)
        np.testing.assert_allclose(actual_distances, distances, rtol=2e-4, atol=2e-4)


def test_rejected_files_raise_format_error(tmp_path):
    assert issubclass(rabitqlib.FormatError, ValueError)
    assert issubclass(rabitqlib.FormatError, RuntimeError)
    assert rabitqlib.FormatError.__module__ == "rabitqlib"
    assert quant.FormatError is rabitqlib.FormatError
    for name, cls in [
        ("ivf4_flat_l2", IvfIndex),
        ("hnsw1_l2", HnswIndex),
        ("qg4_l2", SymqgIndex),
    ]:
        legacy = (FIXTURES / "v0.8.3" / f"{name}.index").read_bytes()
        damaged = tmp_path / f"{name}.index"
        damaged.write_bytes(legacy[: len(legacy) // 2])
        with pytest.raises(rabitqlib.FormatError):
            cls.load(str(damaged))
        migrated = tmp_path / f"{name}-format2.index"
        cls.load(str(FIXTURES / "v0.8.3" / f"{name}.index")).save(str(migrated))
        payload = migrated.read_bytes()
        migrated.write_bytes(payload[:-1])
        with pytest.raises(rabitqlib.FormatError, match="index file"):
            cls.load(str(migrated))
    # Failing to open a file is not a format error.
    with pytest.raises(RuntimeError) as missing:
        IvfIndex.load(str(tmp_path / "missing.index"))
    assert not isinstance(missing.value, rabitqlib.FormatError)
