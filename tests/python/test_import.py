"""Tests for module import and class construction."""

import pytest
from rabitqlib import HnswIndex, IvfIndex, SymqgIndex


@pytest.mark.parametrize(
    "cls,args,kwargs,expected",
    [
        (
            IvfIndex,
            (64, 5),
            {"nbits": 4, "metric": "l2"},
            {"num_clusters": 5, "nbits": 4, "metric": "l2"},
        ),
        (IvfIndex, (64, 5), {"nbits": 4, "metric": "ip"}, {"metric": "ip"}),
        (HnswIndex, (64,), {}, {}),
        (
            HnswIndex,
            (64,),
            {
                "M": 8,
                "ef_construction": 50,
                "nbits": 4,
                "metric": "ip",
                "seed": 7,
            },
            {"metric": "ip", "nbits": 4},
        ),
        (SymqgIndex, (64,), {"max_degree": 32}, {"max_degree": 32}),
    ],
    ids=["ivf-l2", "ivf-ip", "hnsw-default", "hnsw-explicit", "symqg"],
)
def test_constructor_contract(cls, args, kwargs, expected):
    idx = cls(*args, **kwargs)
    assert idx.dim == 64
    assert len(idx) == 0
    for name, value in expected.items():
        assert getattr(idx, name) == value


@pytest.mark.parametrize(
    "cls,args",
    [
        (IvfIndex, (64, 5)),
        (HnswIndex, (64,)),
        (SymqgIndex, (64,)),
    ],
)
def test_invalid_metric_raises(cls, args):
    with pytest.raises(ValueError, match=r"^metric must be 'l2' or 'ip'$"):
        cls(*args, metric="cosine")
