"""Python boundary coverage for quant layer 3: FastScan blocks and estimates."""

import copy
import gc
import pickle
from pathlib import Path

import numpy as np
import pytest
from rabitqlib import quant

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "quant"
PERM = [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15]


def reference_blocks(bitsets, factors):
    """The FastScan blocks version 1 layout of docs/formats.md, in NumPy."""
    n, cols = bitsets.shape
    msb_first = np.packbits(
        np.unpackbits(bitsets, axis=1, bitorder="little"), axis=1, bitorder="big"
    )
    blocks = []
    for b in range(-(-n // 32)):
        count = min(32, n - 32 * b)
        lanes = np.zeros((32, cols), np.uint8)
        lanes[:count] = msb_first[32 * b : 32 * b + count]
        codes = np.zeros((cols, 32), np.uint8)
        for j, p in enumerate(PERM):
            a = lanes[p].astype(np.uint16)
            c = lanes[p + 16].astype(np.uint16)
            codes[:, j] = (a >> 4) | (c & 0xF0)
            codes[:, j + 16] = ((a & 0x0F) | (c << 4)) & 0xFF
        record = np.zeros((3, 32), "<f4")
        for k, name in enumerate(("add", "rescale", "error")):
            record[k, :count] = factors[name][32 * b : 32 * b + count]
        blocks.append(codes.tobytes() + record.tobytes())
    return b"".join(blocks)


def setup(metric, n, d=96, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)).astype(np.float32)
    c = (rng.standard_normal(d) * 0.3).astype(np.float32)
    prefix = quant.quantize(x - c, nbits=1)
    factors = quant.code_factors(x - c, prefix, nbits=1, metric=metric, centroid=c)
    q = rng.standard_normal(d).astype(np.float32)
    return c, quant.pack_codes(prefix, nbits=1), factors, q


@pytest.mark.parametrize("metric", ["l2", "ip"])
@pytest.mark.parametrize("n", [0, 1, 31, 32, 33, 100])
def test_fastscan_matches_estimate_bitset(metric, n):
    c, bitsets, factors, q = setup(metric, n)
    blocks = quant.pack_fastscan_blocks(bitsets, factors)
    assert blocks.dtype == np.uint8 and blocks.shape == (-(-n // 32), 96 * 4 + 384)
    assert blocks.tobytes() == reference_blocks(bitsets, factors)
    query = quant.Query(q, metric=metric, centroid=c)
    fastscan = query.estimate_fastscan(blocks, num_points=n)
    assert fastscan.dtype == quant.ESTIMATE_DTYPE and fastscan.shape == (n,)
    exact = query.estimate_bitset(bitsets, factors)
    # The 4-bit query table quantizes the query: a separate, looser tolerance.
    scale = np.abs(factors["rescale"]).max(initial=0) * np.abs(q).sum()
    np.testing.assert_allclose(
        fastscan["distance"], exact["distance"], rtol=0, atol=5e-5 * scale
    )
    np.testing.assert_allclose(
        fastscan["refine_input"],
        exact["refine_input"],
        rtol=0,
        atol=5e-5 * np.abs(q).sum(),
    )
    np.testing.assert_array_equal(fastscan["error"], exact["error"])
    np.testing.assert_array_equal(
        query.estimate_fastscan(blocks, num_points=n, num_threads=3), fastscan
    )
    np.testing.assert_array_equal(
        quant.pack_fastscan_blocks(bitsets, factors, num_threads=3), blocks
    )


def test_set_centroid_and_copies_rebuild_their_own_table():
    c, bitsets, factors, q = setup("ip", 70)
    blocks = quant.pack_fastscan_blocks(bitsets, factors)
    query = quant.Query(q, metric="ip")
    before = query.estimate_fastscan(blocks, num_points=70)
    query.set_centroid(c)
    expected = quant.Query(q, metric="ip", centroid=c).estimate_fastscan(
        blocks, num_points=70
    )
    np.testing.assert_array_equal(
        query.estimate_fastscan(blocks, num_points=70), expected
    )
    assert not np.array_equal(before, expected)
    copies = [
        query.copy(),
        copy.copy(query),
        copy.deepcopy(query),
        pickle.loads(pickle.dumps(query)),
    ]
    del query
    gc.collect()
    for other in copies:
        np.testing.assert_array_equal(
            other.estimate_fastscan(blocks, num_points=70), expected
        )
    out = np.empty(70, quant.ESTIMATE_DTYPE)
    assert copies[0].estimate_fastscan(blocks, num_points=70, out=out) is out
    np.testing.assert_array_equal(out, expected)


def test_golden_blocks():
    """FastScan blocks version 1 is frozen; see tests/python/fixtures/quant/README.md."""
    assert quant.FASTSCAN_BLOCKS_VERSION == 1
    golden = (FIXTURES / "fastscan_blocks_n33_d64.bin").read_bytes()
    i, byte = np.arange(33)[:, None], np.arange(8)[None, :]
    bitsets = ((i * 37 + byte * 11 + 3) % 256).astype(np.uint8)
    factors = np.zeros(33, quant.FACTORS_DTYPE)
    factors["add"] = np.arange(33) * 0.5
    factors["rescale"] = -(np.arange(33) + 1) * 0.25
    factors["error"] = np.arange(33) / 8
    blocks = quant.pack_fastscan_blocks(bitsets, factors)
    assert blocks.tobytes() == golden
    assert golden == reference_blocks(bitsets, factors)
    out = np.empty((2, 640), np.uint8)
    assert quant.pack_fastscan_blocks(bitsets, factors, out=out) is out
    assert out.tobytes() == golden


def test_rejected_arguments():
    c, bitsets, factors, q = setup("l2", 40)
    blocks = quant.pack_fastscan_blocks(bitsets, factors)
    query = quant.Query(q, metric="l2", centroid=c)
    broken = factors.copy()
    broken["add"][0] = np.inf
    cases = [
        (
            lambda: quant.pack_fastscan_blocks(bitsets[:, :5], factors),
            "bitsets must have d // 8 bytes per row",
        ),
        (
            lambda: quant.pack_fastscan_blocks(bitsets, factors[:3]),
            "factors must have one record per row of bitsets, 40",
        ),
        (
            lambda: quant.pack_fastscan_blocks(bitsets, broken),
            "factors must contain only finite values",
        ),
        (
            lambda: quant.pack_fastscan_blocks(
                bitsets, factors, out=np.empty((2, 100), np.uint8)
            ),
            "out must be a writable NumPy array of dtype uint8",
        ),
        (
            lambda: query.estimate_fastscan(blocks, num_points=70),
            r"blocks must have shape \(3, 768\) for num_points 70",
        ),
        (
            lambda: query.estimate_fastscan(blocks[:, :700], num_points=40),
            r"blocks must have shape \(2, 768\)",
        ),
        (
            lambda: quant.Query(q[:64], metric="l2").estimate_fastscan(
                blocks, num_points=40
            ),
            r"blocks must have shape \(2, 640\) for num_points 40 and d 64",
        ),
        (
            lambda: query.estimate_fastscan(blocks.view(np.int8), num_points=40),
            "blocks must have dtype uint8",
        ),
    ]
    for call, message in cases:
        with pytest.raises(ValueError, match=message):
            call()


def test_hand_built_blocks_with_non_finite_factors_are_rejected():
    c, bitsets, factors, q = setup("l2", 40)
    query = quant.Query(q, metric="l2", centroid=c)
    nan = np.frombuffer(np.float32(np.nan).tobytes(), np.uint8)
    add = 96 * 4  # Block 1 holds vectors 32-39 in lanes 0-7.
    blocks = quant.pack_fastscan_blocks(bitsets, factors)
    blocks[1, add : add + 4] = nan
    with pytest.raises(ValueError, match="blocks must contain only finite factors"):
        query.estimate_fastscan(blocks, num_points=40)
    # Padding lanes are not read.
    blocks = quant.pack_fastscan_blocks(bitsets, factors)
    blocks[1, add + 4 * 31 : add + 4 * 32] = nan
    query.estimate_fastscan(blocks, num_points=40)


def test_out_overlapping_an_input_is_rejected():
    _, bitsets, factors, _ = setup("l2", 40)
    shared = np.zeros(2 * 768, np.uint8)
    inputs = shared[: 40 * 12].reshape(40, 12)
    inputs[:] = bitsets
    with pytest.raises(ValueError, match="out must not overlap bitsets"):
        quant.pack_fastscan_blocks(inputs, factors, out=shared.reshape(2, 768))
