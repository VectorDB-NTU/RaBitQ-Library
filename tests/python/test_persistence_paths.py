"""Index persistence uses UTF-8 paths, including on Windows."""

import numpy as np
import pytest
from rabitqlib import HnswIndex, IvfIndex, SymqgIndex


@pytest.mark.parametrize("kind", ["ivf", "ivf_raw", "hnsw", "qg", "qg4", "qg8"])
@pytest.mark.parametrize("unicode_directory", [False, True])
def test_unicode_path_round_trip(tmp_path, kind, unicode_directory):
    data = np.random.default_rng(42).standard_normal((64, 64)).astype(np.float32)
    if kind.startswith("ivf"):
        index = IvfIndex(64, 64, 1, nbits=32 if kind == "ivf_raw" else 4)
    elif kind == "hnsw":
        index = HnswIndex(64, 64, M=4, nbits=4)
    else:
        bits = {"qg": 0, "qg4": 4, "qg8": 8}[kind]
        index = SymqgIndex(64, max_degree=32, quantization_bits=bits)

    if kind.startswith("qg"):
        index.build(data, ef_construction=64)
        search_args = {"ef": 64}
    else:
        index.build(data, data.mean(axis=0, keepdims=True), np.zeros(64, np.uint32))
        search_args = {"nprobe": 1} if kind.startswith("ivf") else {"ef": 64}

    directory = tmp_path / "\u6d4b\u8bd5_\U0001f680" if unicode_directory else tmp_path
    directory.mkdir(exist_ok=True)
    path = directory / "\u7d22\u5f15_\U0001f680.index"
    expected_ids, expected_distances = index.search(data[:3], k=5, **search_args)
    index.save(str(path))
    assert path.is_file()  # Catch successful writes to a mojibake filename.

    # Rename through Python's native Unicode path API before loading.
    renamed = directory / "\u91cd\u547d\u540d.index"
    path.rename(renamed)
    loaded = type(index).load(str(renamed))
    ids, distances = loaded.search(data[:3], k=5, **search_args)
    np.testing.assert_array_equal(ids, expected_ids)
    np.testing.assert_array_equal(distances, expected_distances)
