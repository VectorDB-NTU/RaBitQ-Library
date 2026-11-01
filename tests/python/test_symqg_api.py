"""The 0.8 train / add / search workflow of SymqgIndex."""

import inspect
import warnings

import numpy as np
import pytest
from conftest import DIM, N_VECTORS
from rabitqlib import SymqgIndex


@pytest.fixture(autouse=True)
def no_deprecations():
    # The 0.8 API must never go through the compatibility layer.
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        yield


def saved_bytes(index, path):
    index.save(path)
    return path.read_bytes()


def test_constructor_options_are_keyword_only_with_defaults():
    parameters = inspect.signature(SymqgIndex.__init__).parameters
    keyword = inspect.Parameter.KEYWORD_ONLY
    assert [p for p in parameters if parameters[p].kind is not keyword] == [
        "self",
        "dim",
    ]
    defaults = {name: p.default for name, p in parameters.items() if name != "self"}
    assert defaults == {
        "dim": inspect.Parameter.empty,
        "nbits": 32,
        "max_degree": 32,
        "ef_construction": 200,
        "init": "pipnn",
        "metric": "l2",
        "seed": None,
    }
    index = SymqgIndex(DIM)
    assert (index.nbits, index.max_degree, index.ef_construction, index.init) == (
        32,
        32,
        200,
        "pipnn",
    )
    assert index.is_trained
    assert len(index) == 0


@pytest.mark.parametrize(
    "make,error,message",
    [
        (
            lambda: SymqgIndex(10),
            ValueError,
            "^Unsupported dimension for FhtKacRotator$",
        ),
        (
            lambda: SymqgIndex(65537),
            ValueError,
            "^Unsupported dimension for FhtKacRotator$",
        ),
        (lambda: SymqgIndex(DIM, nbits=0), ValueError, "^nbits must be 4, 8, or 32$"),
        (lambda: SymqgIndex(DIM, nbits=16), ValueError, "^nbits must be 4, 8, or 32$"),
        (
            lambda: SymqgIndex(DIM, max_degree=48),
            ValueError,
            "^max_degree must be a positive multiple of 32$",
        ),
        (
            lambda: SymqgIndex(DIM, ef_construction=0),
            ValueError,
            "^ef_construction must be positive and fit uint32$",
        ),
        (
            lambda: SymqgIndex(DIM, init="pippn"),
            ValueError,
            "^init must be 'random' or 'pipnn'$",
        ),
        (lambda: SymqgIndex(DIM, seed=-1), ValueError, "seed must be an integer in"),
        (lambda: SymqgIndex(DIM, seed=2**32), ValueError, r"\[0, 4294967295\]"),
    ],
)
def test_constructor_argument_errors(make, error, message):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(error, match=message):
            make()


def test_seed_is_drawn_once_and_reproduces_the_index(base_data, tmp_path):
    drawn = [SymqgIndex(DIM).seed for _ in range(4)]
    assert all(isinstance(seed, int) and 0 <= seed < 2**32 for seed in drawn)
    assert len(set(drawn)) > 1
    first = SymqgIndex(DIM, ef_construction=50)
    second = SymqgIndex(DIM, ef_construction=50, seed=first.seed)
    for index in (first, second):
        index.add(base_data, num_threads=1)
    assert saved_bytes(first, tmp_path / "a.index") == saved_bytes(
        second, tmp_path / "b.index"
    )


def test_train_only_checks_its_input(base_data):
    index = SymqgIndex(DIM)
    index.train(base_data)
    index.train(base_data, num_threads=1)
    index.train(base_data, num_threads=None)
    assert len(index) == 0
    bad = base_data.copy()
    bad[3, 3] = np.nan
    with pytest.raises(ValueError, match="^x must contain only finite values$"):
        index.train(bad)
    with pytest.raises(ValueError, match="x dimension does not match index dim"):
        index.train(base_data[:, :-1])
    with pytest.raises(ValueError, match="x must be a 2D NumPy array"):
        index.train(base_data[0])


@pytest.mark.parametrize("nbits", [4, 8, 32])
def test_add_builds_once_and_search_finds_the_vectors(nbits, base_data, tmp_path):
    index = SymqgIndex(DIM, nbits=nbits, ef_construction=60, seed=5)
    assert index.add(base_data) is None
    assert len(index) == index.num_points == N_VECTORS
    ids, distances = index.search(base_data[:20], 1)
    assert np.mean(ids[:, 0] == np.arange(20)) >= 0.9
    message = "^SymqgIndex does not support incremental addition; create a new index and add all vectors at once$"
    with pytest.raises(RuntimeError, match=message):
        index.add(base_data[:40])
    assert len(index) == N_VECTORS
    index.save(tmp_path / "graph.index")
    loaded = SymqgIndex.load(tmp_path / "graph.index")
    assert loaded.nbits == nbits
    # Files store neither the seed nor the construction settings.
    assert (loaded.seed, loaded.ef_construction, loaded.init) == (None, None, None)
    assert loaded.is_trained
    with pytest.raises(RuntimeError, match=message):
        loaded.add(base_data)


def test_add_uses_the_constructor_settings(base_data, tmp_path):
    for init in ("pipnn", "random"):
        added = SymqgIndex(DIM, ef_construction=60, init=init, seed=9)
        added.add(base_data, num_threads=1)
        with pytest.warns(DeprecationWarning):
            built = SymqgIndex(DIM, seed=9)
            built.build(base_data, 60, 1, init)
        assert saved_bytes(added, tmp_path / "a.index") == saved_bytes(
            built, tmp_path / "b.index"
        )


def test_search_ef_defaults_to_max_of_k_and_10_and_is_raised_to_k(base_data):
    index = SymqgIndex(DIM, ef_construction=60, seed=5)
    index.add(base_data)
    queries = base_data[:10]
    for k, implied_ef in ((3, 10), (20, 20)):
        expected = index.search(queries, k, implied_ef)
        for ef in (None, 0):
            actual = index.search(queries, k, ef)
            np.testing.assert_array_equal(actual[0], expected[0])
            np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_array_equal(
        index.search(queries, 5, 2)[0], index.search(queries, 5, 5)[0]
    )


def test_operations_before_add_raise(tmp_path, base_data):
    index = SymqgIndex(DIM)
    for operation, call in (
        ("search", lambda: index.search(base_data[:1], 1)),
        ("save", lambda: index.save(tmp_path / "empty.index")),
    ):
        with pytest.raises(
            RuntimeError,
            match=f"^SymqgIndex has no vectors; add them or load an index before {operation}$",
        ):
            call()
    assert not (tmp_path / "empty.index").exists()


def test_type_errors_list_only_the_public_constructor():
    with pytest.raises(TypeError) as error:
        SymqgIndex(DIM, nbits="four")
    assert "_quantization_bits" not in str(error.value)
