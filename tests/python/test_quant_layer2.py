"""Python boundary coverage for quant layer 2: Query, estimates and refinement."""

import copy
import gc
import pickle
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np
import pytest
from rabitqlib import quant

BUSY = "Query is busy: set_centroid cannot overlap another call"


def rows(n, d, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, d)) * scale).astype(np.float32)


def setup(metric, nbits, *, n=40, d=96, centroid=True, seed=0):
    """Codes, factors and a query for vectors x quantized around centroid c."""
    x = rows(n, d, seed)
    c = rows(1, d, seed + 1, 0.3)[0] if centroid else None
    residuals = x - (c if centroid else 0)
    codes = quant.quantize(residuals, nbits=nbits)
    factors = quant.code_factors(
        residuals, codes, nbits=nbits, metric=metric, centroid=c
    )
    q = rows(1, d, seed + 2)[0]
    return x, c, residuals, codes, factors, q


def oracle(q, c, codes, factors, nbits, metric):
    """The reference formula in float64."""
    q64 = q.astype(np.float64)
    c64 = np.zeros_like(q64) if c is None else c.astype(np.float64)
    ip = codes.astype(np.float64) @ q64
    g_add = ((q64 - c64) ** 2).sum() if metric == "l2" else -(q64 @ c64)
    g_error = np.sqrt(((q64 - c64) ** 2).sum())
    k = -0.5 * q64.sum() * (2**nbits - 1)
    distance = factors["add"] + g_add + factors["rescale"] * (ip + k)
    return distance, factors["error"] * g_error, ip


def close(actual, expected, scale):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-5 * scale)


@pytest.mark.parametrize("metric", ["l2", "ip"])
@pytest.mark.parametrize("nbits", range(1, 10))
@pytest.mark.parametrize("centroid", [True, False])
def test_estimate_codes_matches_the_formula(metric, nbits, centroid):
    _, c, _, codes, factors, q = setup(metric, nbits, centroid=centroid)
    query = quant.Query(q, metric=metric, centroid=c)
    estimates = query.estimate_codes(codes, factors, nbits=nbits)
    assert estimates.dtype == quant.ESTIMATE_DTYPE and estimates.shape == (40,)
    distance, error, ip = oracle(q, c, codes, factors, nbits, metric)
    scale = np.abs(factors["rescale"]).max() * np.abs(q).sum() * 2**nbits
    close(estimates["distance"], distance, scale)
    close(estimates["refine_input"], ip, np.abs(q).sum() * 2**nbits)
    np.testing.assert_allclose(estimates["error"], error, rtol=1e-5)
    terms = quant.query_terms(q, metric=metric, centroid=c)
    assert isinstance(terms, np.void) and terms.dtype == quant.QUERY_TERMS_DTYPE
    from_ip = quant.estimate_from_ip(
        estimates["refine_input"], factors, terms, nbits=nbits
    )
    np.testing.assert_array_equal(from_ip, estimates)


@pytest.mark.parametrize("metric", ["l2", "ip"])
def test_estimates_track_true_distances(metric):
    x, c, _, codes, factors, q = setup(metric, 7, n=200, d=128)
    estimates = quant.Query(q, metric=metric, centroid=c).estimate_codes(
        codes, factors, nbits=7
    )
    # IP folds the 1 of 1 - <q, x> into the factors, not into g_add.
    true = ((x - q) ** 2).sum(axis=1) if metric == "l2" else 1 - x @ q
    assert np.corrcoef(estimates["distance"], true)[0, 1] > 0.999
    # The estimate is unbiased: a missing or doubled 1 would shift it by 1.
    scale = np.linalg.norm(q) * np.linalg.norm(x - c, axis=1).mean()
    difference = estimates["distance"] - true
    assert abs(difference.mean()) < 1e-3 * scale
    assert np.abs(difference).mean() < 4e-3 * scale
    terms = quant.query_terms(q, metric=metric, centroid=c)
    expected_add = ((q - c) ** 2).sum() if metric == "l2" else -(q @ c)
    np.testing.assert_allclose(terms["g_add"], expected_add, rtol=1e-5)
    np.testing.assert_allclose(terms["g_error"], np.linalg.norm(q - c), rtol=1e-5)
    np.testing.assert_allclose(terms["g_k1xsumq"], -0.5 * q.sum(), rtol=1e-5)


@pytest.mark.parametrize("metric", ["l2", "ip"])
@pytest.mark.parametrize("prefix_bits", [1, 2])
def test_two_stage_scoring_with_one_query(metric, prefix_bits):
    nbits = 5
    _, c, residuals, codes, f_full, q = setup(metric, nbits)
    prefix, ex = quant.split(codes, nbits=nbits, prefix_bits=prefix_bits)
    ex_bits = nbits - prefix_bits
    f_prefix = quant.code_factors(
        residuals, prefix, nbits=prefix_bits, metric=metric, centroid=c
    )
    query = quant.Query(q, metric=metric, centroid=c)
    first = query.estimate_codes(prefix, f_prefix, nbits=prefix_bits)
    full = query.estimate_codes(codes, f_full, nbits=nbits)
    refined = query.refine(ex, f_full, first, ex_bits=ex_bits, prefix_bits=prefix_bits)
    packed = query.refine_packed(
        quant.pack_codes(ex, nbits=ex_bits),
        f_full,
        first,
        ex_bits=ex_bits,
        prefix_bits=prefix_bits,
    )
    assert refined.dtype == np.float32 and refined.shape == (40,)
    scale = np.abs(f_full["rescale"]).max() * np.abs(q).sum() * 2**nbits
    close(refined, full["distance"], scale)
    close(packed, refined, scale)
    if prefix_bits == 1:
        bitset = query.estimate_bitset(quant.pack_codes(prefix, nbits=1), f_prefix)
        close(bitset["distance"], first["distance"], scale)
        np.testing.assert_array_equal(bitset["error"], first["error"])
    # Scoring several widths in turn does not change the query.
    fresh = quant.Query(q, metric=metric, centroid=c)
    np.testing.assert_array_equal(
        fresh.estimate_codes(codes, f_full, nbits=nbits), full
    )
    np.testing.assert_array_equal(
        query.estimate_codes(prefix, f_prefix, nbits=prefix_bits), first
    )


def test_indices_read_rows_in_place():
    _, c, residuals, codes, factors, q = setup("l2", 4)
    prefix, ex = quant.split(codes, nbits=4, prefix_bits=1)
    f_prefix = quant.code_factors(residuals, prefix, nbits=1, metric="l2", centroid=c)
    bitsets = quant.pack_codes(prefix, nbits=1)
    query = quant.Query(q, metric="l2", centroid=c)
    everything = query.estimate_bitset(bitsets, f_prefix)
    chosen = np.array([7, 0, 39, 7], dtype=np.uint32)
    some = query.estimate_bitset(bitsets, f_prefix, indices=chosen)
    np.testing.assert_array_equal(some, everything[chosen])
    np.testing.assert_array_equal(
        query.estimate_codes(codes, factors, nbits=4, indices=[3, 5]),
        query.estimate_codes(codes, factors, nbits=4)[[3, 5]],
    )
    refined_all = query.refine(ex, factors, everything, ex_bits=3)
    refined = query.refine(ex, factors, everything[chosen], ex_bits=3, indices=chosen)
    np.testing.assert_array_equal(refined, refined_all[chosen])
    packed = quant.pack_codes(ex, nbits=3)
    np.testing.assert_array_equal(
        query.refine_packed(
            packed, factors, everything[chosen], ex_bits=3, indices=chosen
        ),
        query.refine_packed(packed, factors, everything, ex_bits=3)[chosen],
    )
    assert query.estimate_bitset(bitsets, f_prefix, indices=[]).shape == (0,)
    for bad, message in (
        (np.array([True, False]), "indices must contain integers"),
        (np.array([1.0, 2.0]), "indices must contain integers"),
        ([40], r"indices must be in \[0, 40\); got 40 at position 0"),
        ([0, -1], r"indices must be in \[0, 40\); got -1 at position 1"),
    ):
        with pytest.raises(ValueError, match=message):
            query.estimate_bitset(bitsets, f_prefix, indices=bad)
    with pytest.raises(ValueError, match="first must have one record per entry"):
        query.refine(ex, factors, everything, ex_bits=3, indices=chosen)


def test_set_centroid_copy_and_pickle():
    _, c, _, codes, factors, q = setup("ip", 4)
    query = quant.Query(q, metric="ip")
    assert repr(query) == "Query(dim=96, metric='ip', centroid=None)"
    assert (query.dim, query.metric) == (96, "ip")
    query.set_centroid(c)
    assert repr(query) == "Query(dim=96, metric='ip', centroid=set)"
    expected = quant.Query(q, metric="ip", centroid=c).estimate_codes(
        codes, factors, nbits=4
    )
    np.testing.assert_array_equal(
        query.estimate_codes(codes, factors, nbits=4), expected
    )
    copies = [
        query.copy(),
        copy.copy(query),
        copy.deepcopy(query),
        pickle.loads(pickle.dumps(query)),
    ]
    assert all(other is not query for other in copies)
    del query
    gc.collect()
    for other in copies:
        np.testing.assert_array_equal(
            other.estimate_codes(codes, factors, nbits=4), expected
        )
    # Copies are independent.
    copies[0].set_centroid(None)
    np.testing.assert_array_equal(
        copies[1].estimate_codes(codes, factors, nbits=4), expected
    )
    np.testing.assert_array_equal(
        copies[0].estimate_codes(codes, factors, nbits=4),
        quant.Query(q, metric="ip").estimate_codes(codes, factors, nbits=4),
    )
    # The caller's vectors are copied, not referenced.
    q_mutable = q.copy()
    owned = quant.Query(q_mutable, metric="ip", centroid=c)
    q_mutable[:] = 0
    np.testing.assert_array_equal(
        owned.estimate_codes(codes, factors, nbits=4), expected
    )


class PausedArray:
    """Hold an operation open deterministically during Python input conversion."""

    def __init__(self, array):
        self.array = array
        self.entered = Event()
        self.resume = Event()

    def __array__(self, dtype=None, copy=None):
        self.entered.set()
        if not self.resume.wait(timeout=15):
            raise RuntimeError("test did not release input conversion")
        return np.array(self.array, dtype=dtype, copy=True)


def test_set_centroid_during_scoring_is_rejected():
    _, c, _, codes, factors, q = setup("l2", 4)
    query = quant.Query(q, metric="l2", centroid=c)
    expected = query.estimate_codes(codes, factors, nbits=4)
    paused = PausedArray(codes)
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(query.estimate_codes, paused, factors, nbits=4)
        try:
            assert paused.entered.wait(timeout=10)
            np.testing.assert_array_equal(
                query.estimate_codes(codes, factors, nbits=4), expected
            )
            assert repr(query.copy()) == repr(query)
            with pytest.raises(RuntimeError, match=f"^{BUSY}$"):
                query.set_centroid(None)
        finally:
            paused.resume.set()
        np.testing.assert_array_equal(running.result(timeout=10), expected)
    query.set_centroid(None)


def test_threads_out_and_empty_inputs():
    _, c, _, codes, factors, q = setup("l2", 8, n=300)
    query = quant.Query(q, metric="l2", centroid=c)
    single = query.estimate_codes(codes, factors, nbits=8)
    np.testing.assert_array_equal(
        query.estimate_codes(codes, factors, nbits=8, num_threads=4), single
    )
    out = np.empty(300, quant.ESTIMATE_DTYPE)
    assert query.estimate_codes(codes, factors, nbits=8, out=out) is out
    np.testing.assert_array_equal(out, single)
    prefix, ex = quant.split(codes, nbits=8, prefix_bits=1)
    distances = np.empty(300, np.float32)
    assert query.refine(ex, factors, single, ex_bits=7, out=distances) is distances
    terms = quant.query_terms(q, metric="l2", centroid=c)
    estimates = np.empty(300, quant.ESTIMATE_DTYPE)
    assert (
        quant.estimate_from_ip(
            single["refine_input"], factors, terms, nbits=8, out=estimates
        )
        is estimates
    )
    # Strided factors are read correctly.
    doubled = np.repeat(factors, 2)[::2]
    np.testing.assert_array_equal(query.estimate_codes(codes, doubled, nbits=8), single)
    empty_codes = np.empty((0, 96), np.uint8)
    empty_factors = np.empty(0, quant.FACTORS_DTYPE)
    assert query.estimate_codes(empty_codes, empty_factors, nbits=8).shape == (0,)
    assert query.estimate_bitset(np.empty((0, 12), np.uint8), empty_factors).shape == (
        0,
    )
    assert query.refine(
        empty_codes, empty_factors, np.empty(0, quant.ESTIMATE_DTYPE), ex_bits=7
    ).shape == (0,)


def test_rejected_arguments():
    _, c, _, codes, factors, q = setup("l2", 4)
    prefix, ex = quant.split(codes, nbits=4, prefix_bits=1)
    query = quant.Query(q, metric="l2", centroid=c)
    first = query.estimate_codes(codes, factors, nbits=4)
    cases = [
        (lambda: quant.Query(q[:40], metric="l2"), "rotated_query must have a length"),
        (lambda: quant.Query(q, metric="cosine"), "metric"),
        (
            lambda: quant.Query(q, metric="l2", centroid=c[:32]),
            "centroid must have shape",
        ),
        (
            lambda: quant.Query(np.full(96, np.nan, np.float32), metric="l2"),
            "rotated_query must contain only finite values",
        ),
        (
            lambda: query.estimate_codes(codes, factors, nbits=10),
            r"nbits must be in \[1, 9\]",
        ),
        (
            lambda: query.estimate_codes(codes[:, :64], factors, nbits=4),
            "codes must have d = 96 columns",
        ),
        (
            lambda: query.estimate_codes(codes.astype(np.uint16), factors, nbits=4),
            "codes must have dtype uint8",
        ),
        (
            lambda: query.estimate_codes(codes, factors, nbits=3),
            r"codes values must be below 2\*\*3",
        ),
        (
            lambda: query.estimate_codes(codes, factors[:5], nbits=4),
            "factors must have one record per row of codes, 40",
        ),
        (
            lambda: query.estimate_codes(codes, first, nbits=4),
            "factors must be a 1D NumPy array of dtype quant.FACTORS_DTYPE",
        ),
        (
            lambda: query.estimate_codes(
                codes, factors.view(np.float32).reshape(-1, 3), nbits=4
            ),
            "factors must be a 1D NumPy array of dtype quant.FACTORS_DTYPE",
        ),
        (
            lambda: query.estimate_bitset(codes, factors),
            "bitsets must have d // 8 = 12 columns",
        ),
        (
            lambda: query.refine(ex, factors, first, ex_bits=2),
            r"ex values must be below 2\*\*2",
        ),
        (
            lambda: query.refine(ex, factors, first, ex_bits=9),
            r"ex_bits must be in \[1, 8\]",
        ),
        (
            lambda: query.refine(ex, factors, first, ex_bits=3, prefix_bits=0),
            r"prefix_bits must be in \[1, 8\]",
        ),
        (
            lambda: query.refine(ex, factors, first[:3], ex_bits=3),
            "first must have one record per row of ex, 40",
        ),
        (
            lambda: query.refine(ex, factors, factors, ex_bits=3),
            "first must be a 1D NumPy array of dtype quant.ESTIMATE_DTYPE",
        ),
        (
            lambda: query.refine_packed(ex, factors, first, ex_bits=3),
            r"packed must have d \* ex_bits // 8 = 36 columns",
        ),
        (
            lambda: query.estimate_codes(
                codes, factors, nbits=4, out=np.empty(40, quant.FACTORS_DTYPE)
            ),
            "out must be a writable, C-contiguous NumPy array of dtype "
            r"quant.ESTIMATE_DTYPE and shape \(40,\)",
        ),
        (
            lambda: query.refine(ex, factors, first, ex_bits=3, out=np.empty(40)),
            "out must be a writable, C-contiguous NumPy array of dtype float32",
        ),
        (
            lambda: quant.estimate_from_ip(
                np.zeros(40, np.float32), factors, first[0], nbits=4
            ),
            "terms must be one record of dtype quant.QUERY_TERMS_DTYPE",
        ),
        (
            lambda: quant.estimate_from_ip(
                np.full(40, np.inf, np.float32),
                factors,
                quant.query_terms(q, metric="l2"),
                nbits=4,
            ),
            "ip must contain only finite values",
        ),
    ]
    for call, message in cases:
        with pytest.raises(ValueError, match=message):
            call()
    broken = factors.copy()
    broken["rescale"][3] = np.nan
    with pytest.raises(ValueError, match="factors must contain only finite values"):
        query.estimate_codes(codes, broken, nbits=4)
    # A non-finite factor outside the selected rows is not read.
    query.estimate_codes(codes, broken, nbits=4, indices=[0, 1])


def test_out_overlapping_an_input_is_rejected():
    _, c, _, codes, factors, q = setup("l2", 4)
    prefix, ex = quant.split(codes, nbits=4, prefix_bits=1)
    query = quant.Query(q, metric="l2", centroid=c)
    shared = np.zeros(80, quant.ESTIMATE_DTYPE)
    shared_factors = shared.view(quant.FACTORS_DTYPE)[:40]
    shared_factors[:] = factors
    with pytest.raises(ValueError, match="out must not overlap factors"):
        query.estimate_codes(codes, shared_factors, nbits=4, out=shared[20:60])
    first = query.estimate_codes(codes, factors, nbits=4)
    with pytest.raises(ValueError, match="out must not overlap first"):
        query.refine(ex, factors, first, ex_bits=3, out=first.view(np.float32)[:40])
    terms = quant.query_terms(q, metric="l2", centroid=c)
    shared_ip = np.zeros(3 * 40, np.float32)
    ip = shared_ip[:40]
    ip[:] = first["refine_input"]
    with pytest.raises(ValueError, match="out must not overlap ip"):
        quant.estimate_from_ip(
            ip, factors, terms, nbits=4, out=shared_ip.view(quant.ESTIMATE_DTYPE)
        )


def test_corrupt_pickle_state_is_rejected():
    query = quant.Query.__new__(quant.Query)
    with pytest.raises(ValueError, match="Invalid Query pickle state"):
        query.__setstate__((np.zeros(64, np.float32), 5, None))
