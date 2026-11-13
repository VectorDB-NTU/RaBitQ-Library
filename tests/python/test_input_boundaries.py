"""Public input diagnostics and filesystem-path conversion contracts."""

import errno
import os
from pathlib import Path

import numpy as np
import pytest
from rabitqlib import HnswIndex, IvfIndex, QGKMeans, RaBitQKMeans, SymqgIndex


class FilesystemPath:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def __fspath__(self):
        self.calls += 1
        return self.value


@pytest.fixture(scope="module", params=["ivf", "ivf_hnsw", "hnsw", "symqg"])
def built_index(request):
    data = np.random.default_rng(42).normal(size=(80, 64)).astype(np.float32)
    if request.param == "symqg":
        index = SymqgIndex(64, max_degree=32, ef_construction=40)
        index.add(data, num_threads=1)
    else:
        if request.param.startswith("ivf"):
            initializer = "hnsw" if request.param == "ivf_hnsw" else "flat"
            index = IvfIndex(64, 1, nbits=4, initializer=initializer)
        else:
            index = HnswIndex(64, nbits=4, num_clusters=1)
        index.set_centroids(data.mean(axis=0, keepdims=True))
        index.add(data, cluster_ids=np.zeros(80, dtype=np.uint32), num_threads=1)
    return index, data[:2]


@pytest.mark.parametrize(
    "kind", ["str", "path", "fspath_str", "bytes", "bytearray", "fspath_bytes"]
)
def test_path_roundtrip(built_index, tmp_path, kind):
    index, queries = built_index
    filename = tmp_path / "索引 café 🐇.index"
    value = {
        "str": str(filename),
        "path": filename,
        "fspath_str": FilesystemPath(str(filename)),
        "bytes": os.fsencode(filename),
        "bytearray": bytearray(os.fsencode(filename)),
        "fspath_bytes": FilesystemPath(os.fsencode(filename)),
    }[kind]
    index.save(value)
    assert filename.is_file()
    if isinstance(value, FilesystemPath):
        assert value.calls == 1
    if isinstance(index, IvfIndex) and index.initializer == "hnsw":
        assert Path(str(filename) + ".hnsw").is_file()
    restored = type(index).load(value)
    if isinstance(value, FilesystemPath):
        assert value.calls == 2
    expected = index.search(queries, 3, 32)
    actual = restored.search(queries, 3, 32)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-6, atol=1e-6)


def test_save_replaces_existing_files_without_leftovers(built_index, tmp_path):
    index, queries = built_index
    filename = tmp_path / "replaced.index"
    filename.write_bytes(b"stale")
    expected_files = {filename.name}
    if isinstance(index, IvfIndex) and index.initializer == "hnsw":
        Path(str(filename) + ".hnsw").write_bytes(b"stale")
        expected_files.add(filename.name + ".hnsw")
    for _ in range(2):
        index.save(filename)
        assert {path.name for path in tmp_path.iterdir()} == expected_files
    restored = type(index).load(filename)
    expected = index.search(queries, 3, 32)
    actual = restored.search(queries, 3, 32)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(os.name != "posix", reason="POSIX native path bytes")
def test_path_preserves_undecodable_bytes(built_index, tmp_path):
    index, queries = built_index
    raw = os.fsencode(tmp_path) + b"/index-\xff.bin"
    # POSIX does not guarantee that the filesystem accepts non-UTF-8 names.
    # Probe independently so binding failures below cannot become skips.
    try:
        descriptor = os.open(raw, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except OSError as error:
        if error.errno in (errno.EINVAL, errno.EILSEQ):
            pytest.skip(f"filesystem rejects non-UTF-8 filenames (errno {error.errno})")
        raise
    os.close(descriptor)
    os.unlink(raw)
    filename = Path(os.fsdecode(raw))
    index.save(filename)
    assert os.path.isfile(raw)
    restored = type(index).load(filename)
    np.testing.assert_array_equal(
        restored.search(queries, 3, 32)[0], index.search(queries, 3, 32)[0]
    )
    if isinstance(index, IvfIndex) and index.initializer == "hnsw":
        assert os.path.isfile(raw + b".hnsw")


@pytest.mark.parametrize("built_index", ["ivf_hnsw"], indirect=True)
@pytest.mark.parametrize(
    "value", [object(), FilesystemPath(123), FilesystemPath(bytearray(b"index"))]
)
def test_invalid_paths_raise_type_error(built_index, value):
    index, _ = built_index
    with pytest.raises(TypeError):
        index.save(value)
    with pytest.raises(TypeError):
        type(index).load(value)


@pytest.mark.parametrize(
    "built_index,kind",
    [
        ("ivf_hnsw", "str"),
        ("ivf_hnsw", "bytes"),
        ("ivf_hnsw", "bytearray"),
        ("ivf_hnsw", "pathlike"),
        ("hnsw", "str"),
        ("symqg", "str"),
    ],
    indirect=["built_index"],
)
def test_embedded_null_does_not_truncate_path(built_index, tmp_path, kind):
    index, _ = built_index
    filename = tmp_path / "untouched.index"
    filename.write_bytes(b"unchanged")
    value = str(filename) + "\0suffix"
    if kind == "bytes":
        value = os.fsencode(value)
    elif kind == "bytearray":
        value = bytearray(os.fsencode(value))
    elif kind == "pathlike":
        value = FilesystemPath(value)
    for operation in (index.save, type(index).load):
        with pytest.raises(
            ValueError, match="^path must not contain embedded null bytes$"
        ):
            operation(value)
        assert filename.read_bytes() == b"unchanged"
        assert not Path(str(filename) + ".hnsw").exists()


@pytest.mark.parametrize("built_index", ["ivf_hnsw", "hnsw", "symqg"], indirect=True)
def test_fspath_exception_is_preserved(built_index):
    class BrokenPath:
        def __fspath__(self):
            raise RuntimeError("fspath failed")

    index, _ = built_index
    for operation in (index.save, type(index).load):
        with pytest.raises(RuntimeError, match="^fspath failed$"):
            operation(BrokenPath())


@pytest.mark.parametrize("model", [QGKMeans, RaBitQKMeans])
def test_unknown_keyword_names_public_constructor(model):
    with pytest.raises(AttributeError) as error:
        model(64, 40, niterr=2)
    assert (
        str(error.value)
        == f"{model.__name__}() got an unexpected keyword argument 'niterr'"
    )


@pytest.mark.parametrize("model", [QGKMeans, RaBitQKMeans])
@pytest.mark.parametrize("value", ["two", -1])
def test_keyword_conversion_names_public_parameter(model, value):
    with pytest.raises(TypeError) as error:
        model(64, 40, niter=value)
    assert (
        str(error.value)
        == f"{model.__name__}() could not convert parameter 'niter' from {type(value).__name__}"
    )


@pytest.mark.parametrize("built_index", ["ivf_hnsw"], indirect=True)
def test_integer_id_dtype_diagnostic(built_index):
    index, _ = built_index
    assert isinstance(index, IvfIndex)
    with pytest.raises(ValueError) as error:
        index.remove(np.array([1.5], dtype=np.float32))  # type: ignore[arg-type]
    assert str(error.value) == "ids must contain integers; got dtype float32"


@pytest.mark.parametrize("built_index", ["ivf_hnsw"], indirect=True)
@pytest.mark.parametrize("shape", [(), (64,), (2, 1, 64)])
def test_query_rank_diagnostics(built_index, shape):
    index, _ = built_index
    with pytest.raises(ValueError) as error:
        index.search(np.zeros(shape, dtype=np.float32), 1, 32)
    assert str(error.value) == f"queries must be a 2D NumPy array; got shape {shape}"


@pytest.mark.parametrize("built_index", ["ivf_hnsw", "hnsw", "symqg"], indirect=True)
def test_path_callback_runs_before_access_guard(built_index, tmp_path):
    index, queries = built_index

    class UpdatingPath:
        def __fspath__(self):
            if isinstance(index, IvfIndex):
                index.reserve(0, index.num_stored_points)
            elif isinstance(index, HnswIndex):
                index.resize(index.max_elements)
            else:
                # add is SymphonyQG's only write. It takes the write guard and then
                # refuses a built graph; a guard still held would raise "busy".
                with pytest.raises(RuntimeError, match="incremental addition"):
                    index.add(np.zeros((1, 64), dtype=np.float32))
            return str(tmp_path / "callback.index")

    index.save(UpdatingPath())
    expected = index.search(queries, 3, 32)
    restored = type(index).load(tmp_path / "callback.index")
    np.testing.assert_array_equal(restored.search(queries, 3, 32)[0], expected[0])


@pytest.mark.parametrize("index_kind,dtype", [("ivf", np.uint32), ("hnsw", np.int64)])
def test_cluster_id_error_identifies_position(
    index_kind, dtype: type[np.int64] | type[np.uint32]
):
    if index_kind == "ivf":
        index = IvfIndex(64, 1, nbits=4)
    else:
        index = HnswIndex(64, nbits=8, num_clusters=1)
    index.set_centroids(np.zeros((1, 64)))
    ids = np.zeros(80, dtype=dtype)
    ids[-1] = 1
    with pytest.raises(ValueError) as error:
        index.add(np.zeros((80, 64)), cluster_ids=ids, num_threads=1)
    assert str(error.value) == (
        "cluster_ids must be in [0, num_clusters); expected [0, 1), "
        "invalid entry at index 79"
    )


@pytest.mark.parametrize("built_index", ["ivf_hnsw"], indirect=True)
@pytest.mark.parametrize(
    "value,description",
    [
        (np.array([["bad"]]), "ndarray with dtype <U3"),
        ([["bad"]], "list"),
        (object(), "object"),
    ],
)
def test_array_conversion_reports_target_dtype(built_index, value, description):
    index, _ = built_index
    with pytest.raises(ValueError) as error:
        index.search(value, 1, 32)
    assert (
        str(error.value)
        == "queries must be convertible to a NumPy array of dtype float32; got "
        + description
    )


@pytest.mark.parametrize(
    "value,message",
    [
        (np.nan, "must contain only finite values"),
        (np.inf, "must contain only finite values"),
        (-np.inf, "must contain only finite values"),
        (1e30, "exceeds the safe float32 coordinate range"),
    ],
)
def test_nonfinite_or_oversized_coordinates_are_rejected(built_index, value, message):
    index, queries = built_index
    before = index.num_points
    bad_queries = np.array(queries, copy=True)
    bad_queries[1, 3] = value
    with pytest.raises(ValueError, match=message):
        index.search(bad_queries, 3, 32)
    extra = np.random.default_rng(7).normal(size=(2, 64)).astype(np.float32)
    extra[0, 0] = value
    if isinstance(index, (IvfIndex, HnswIndex)):
        with pytest.raises(ValueError, match=message):
            index.add(extra, num_threads=1)
    assert index.num_points == before
    ids, distances = index.search(queries, 3, 32)
    assert ids.shape == (2, 3) and np.all(np.isfinite(distances))


@pytest.mark.parametrize("index_kind", ["ivf", "hnsw", "symqg"])
@pytest.mark.parametrize("value", [np.nan, np.inf, 1e30])
def test_build_rejects_bad_coordinates_without_building(index_kind, value):
    data = np.random.default_rng(3).normal(size=(80, 64)).astype(np.float32)
    bad_data = data.copy()
    bad_data[5, 3] = value
    if index_kind == "symqg":
        index = SymqgIndex(64, max_degree=32, ef_construction=40)
        with pytest.raises(ValueError):
            index.add(bad_data, num_threads=1)
    else:
        centroid = data.mean(axis=0, keepdims=True)
        ids = np.zeros(80, dtype=np.uint32)
        index = (
            IvfIndex(64, 1, nbits=4)
            if index_kind == "ivf"
            else HnswIndex(64, nbits=8, num_clusters=1)
        )
        bad_centroid = centroid.copy()
        bad_centroid[0, 0] = value
        with pytest.raises(ValueError):
            index.set_centroids(bad_centroid)
        index.set_centroids(centroid)
        with pytest.raises(ValueError):
            index.add(bad_data, cluster_ids=ids, num_threads=1)
    assert len(index) == 0
