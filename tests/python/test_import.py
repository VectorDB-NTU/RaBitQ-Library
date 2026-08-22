"""Tests for module import and class construction."""

import pytest
from rabitqlib import HnswIndex, IvfIndex, SymqgIndex

# ── import ────────────────────────────────────────────────────────────────────


def test_all_classes_importable():
    assert HnswIndex is not None
    assert IvfIndex is not None
    assert SymqgIndex is not None


# ── valid construction ────────────────────────────────────────────────────────


def test_ivf_construct_l2():
    idx = IvfIndex(64, 500, 5, nbits=4, metric="l2")
    assert idx.dim == 64
    assert idx.num_clusters == 5
    assert idx.nbits == 4
    assert idx.metric == "l2"
    assert not idx.is_built


def test_ivf_construct_ip():
    idx = IvfIndex(64, 500, 5, nbits=4, metric="ip")
    assert idx.metric == "ip"


def test_hnsw_construct_defaults():
    idx = HnswIndex(64, 500)
    assert idx.dim == 64
    assert not idx.is_built


def test_hnsw_construct_explicit():
    idx = HnswIndex(
        64, 500, M=8, ef_construction=50, nbits=4, metric="ip", random_seed=7
    )
    assert idx.metric == "ip"
    assert idx.nbits == 4


def test_symqg_construct():
    idx = SymqgIndex(64, max_degree=16)
    assert idx.dim == 64
    assert idx.max_degree == 16
    assert not idx.is_built


# ── invalid metric ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "cls,args",
    [
        (IvfIndex, (64, 500, 5, 4)),
        (HnswIndex, (64, 500)),
        (SymqgIndex, (64, 16)),
    ],
)
def test_invalid_metric_raises(cls, args):
    with pytest.raises(Exception, match="[Uu]nsupported metric|[Ii]nvalid"):
        cls(*args, metric="cosine")
