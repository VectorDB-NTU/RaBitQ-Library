"""Python boundary coverage for rabitqlib.quant layer 1: quantize, split, factors, packing."""

from pathlib import Path

import numpy as np
import pytest
from rabitqlib import quant

D = 96  # one full 64-coordinate block plus a 32-coordinate tail
EPSILON = 1.9  # folded into the error factor (kConstEpsilon)


def residuals(n=8, d=D, seed=0):
    return np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)


def code_dtype(nbits):
    return np.uint16 if nbits == 9 else np.uint8


def reference_factors(r, codes, nbits, metric, centroid):
    """The published factor formulas (docs/docs/rabitq/estimator.md) in float64."""
    r = r.astype(np.float64)
    c = np.zeros(r.shape[1]) if centroid is None else centroid.astype(np.float64)
    xu = codes.astype(np.float64) - (2**nbits - 1) / 2
    l2 = np.sum(r * r, axis=1)
    ip_r = np.sum(r * xu, axis=1)
    ip_c = xu @ c
    # error = scale * sqrt((ratio - 1) / (d - 1)) with ratio = |r|^2 |xu|^2 / <r, xu>^2.
    ratio = l2 * np.sum(xu * xu, axis=1) / ip_r**2
    scale = (2 if metric == "l2" else 1) * EPSILON * np.sqrt(l2)
    if metric == "l2":
        return l2 + 2 * l2 * ip_c / ip_r, -2 * l2 / ip_r, ratio, scale
    return 1 - r @ c + l2 * ip_c / ip_r, -l2 / ip_r, ratio, scale


def test_padded_dim():
    assert [quant.padded_dim(d) for d in (1, 32, 33, 100, 128)] == [
        32,
        32,
        64,
        128,
        128,
    ]
    with pytest.raises(ValueError, match="dim must be positive"):
        quant.padded_dim(0)


@pytest.mark.parametrize("nbits", range(1, 10))
def test_quantize_codes_and_factors(nbits):
    r = residuals()
    codes = quant.quantize(r, nbits=nbits)
    assert codes.dtype == code_dtype(nbits) and codes.shape == r.shape
    assert codes.max() < 2**nbits
    # The top bit of each code is the sign of its residual.
    np.testing.assert_array_equal(codes >> (nbits - 1), (r > 0).astype(codes.dtype))
    centroid = residuals(1, seed=1)[0]
    for metric in ("l2", "ip"):
        factors = quant.code_factors(
            r, codes, nbits=nbits, metric=metric, centroid=centroid
        )
        assert factors.dtype == quant.FACTORS_DTYPE and factors.shape == (len(r),)
        add, rescale, ratio, scale = reference_factors(
            r, codes, nbits, metric, centroid
        )
        np.testing.assert_allclose(factors["add"], add, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(factors["rescale"], rescale, rtol=1e-4)
        # The float32 ratio - 1 cancels at high widths, so compare the ratio itself.
        error = factors["error"].astype(np.float64)
        np.testing.assert_allclose((error / scale) ** 2 * (D - 1) + 1, ratio, rtol=1e-5)


def test_fast_quantization_and_thread_counts():
    r = residuals(n=40)
    exact = quant.quantize(r, nbits=5, num_threads=1)
    np.testing.assert_array_equal(quant.quantize(r, nbits=5, num_threads=0), exact)
    fast = quant.quantize(r, nbits=5, fast_quantization=True)
    assert fast.dtype == np.uint8 and fast.max() < 32
    # The sign bit does not depend on the scale.
    np.testing.assert_array_equal(fast >> 4, exact >> 4)
    np.testing.assert_array_equal(
        quant.quantize(r, nbits=5, fast_quantization=True, num_threads=3), fast
    )


@pytest.mark.parametrize("nbits", [2, 4, 9])
def test_split_prefix_factors(nbits):
    r = residuals()
    codes = quant.quantize(r, nbits=nbits)
    for prefix_bits in range(1, nbits):
        prefix, ex = quant.split(codes, nbits=nbits, prefix_bits=prefix_bits)
        ex_bits = nbits - prefix_bits
        assert prefix.dtype == ex.dtype == np.uint8
        np.testing.assert_array_equal(prefix, codes >> ex_bits)
        np.testing.assert_array_equal(ex, codes & (2**ex_bits - 1))
        # A prefix is itself a code of the prefix width for the same residual.
        quant.code_factors(r, prefix, nbits=prefix_bits, metric="l2")


@pytest.mark.parametrize("nbits", [4, 9])
def test_split_threads_and_out(nbits):
    # 300 rows span several parallel ranges.
    codes = quant.quantize(residuals(300, seed=12), nbits=nbits)
    prefix, ex = quant.split(codes, nbits=nbits, prefix_bits=2, num_threads=1)
    for threads in (0, 4):
        parts = quant.split(codes, nbits=nbits, prefix_bits=2, num_threads=threads)
        np.testing.assert_array_equal(parts[0], prefix)
        np.testing.assert_array_equal(parts[1], ex)
    out = (np.empty((300, D), np.uint8), np.empty((300, D), np.uint8))
    result = quant.split(codes, nbits=nbits, prefix_bits=2, num_threads=4, out=out)
    assert result[0] is out[0] and result[1] is out[1]
    np.testing.assert_array_equal(out[0], prefix)
    np.testing.assert_array_equal(out[1], ex)
    # Strided output rows are accepted, as for the other out= arguments. Overlap is
    # judged by byte extent, so the two parts come from separate buffers.
    wide = (np.zeros((300, 2 * D), np.uint8), np.zeros((300, 2 * D), np.uint8))
    quant.split(codes, nbits=nbits, prefix_bits=2, out=(wide[0][:, :D], wide[1][:, D:]))
    np.testing.assert_array_equal(wide[0][:, :D], prefix)
    np.testing.assert_array_equal(wide[1][:, D:], ex)
    assert not wide[0][:, D:].any() and not wide[1][:, :D].any()


def test_split_rejects_bad_out_and_leaves_out_unchanged():
    codes = quant.quantize(residuals(300, seed=13), nbits=4)
    good = np.zeros((300, D), np.uint8)
    cases = [
        (np.zeros((300, D), np.uint8), r"out must be a \(prefix, ex\) tuple"),
        ((good,), r"out must be a \(prefix, ex\) tuple"),
        ([good, good.copy()], r"out must be a \(prefix, ex\) tuple"),
        (
            (np.zeros((300, D), np.int8), good),
            r"out\[0\] must be a writable NumPy array of dtype uint8 and shape \(300, 96\)",
        ),
        (
            (good, np.zeros((299, D), np.uint8)),
            r"out\[1\] must be a writable NumPy array of dtype uint8",
        ),
        ((good, good), r"out\[1\] must not overlap out\[0\]"),
        ((good, good[:, ::-1]), r"out\[1\] must be a writable NumPy array"),
    ]
    for out, message in cases:
        with pytest.raises(ValueError, match=message):
            quant.split(codes, nbits=4, prefix_bits=1, out=out)
    shared = np.zeros((300, 2 * D), np.uint8)
    view = shared[:, :D]
    view[:] = codes
    with pytest.raises(ValueError, match=r"out\[0\] must not overlap codes"):
        quant.split(
            view, nbits=4, prefix_bits=1, out=(shared[:, D // 2 : D + D // 2], good)
        )
    # Codes are checked in a pass of their own, before out= is written.
    bad = codes.copy()
    bad[250, 3] = 16
    out = (np.zeros((300, D), np.uint8), np.zeros((300, D), np.uint8))
    with pytest.raises(ValueError, match=r"got 16 at \(250, 3\)$"):
        quant.split(bad, nbits=4, prefix_bits=1, num_threads=4, out=out)
    assert not out[0].any() and not out[1].any()


@pytest.mark.parametrize("nbits", range(1, 9))
def test_pack_round_trip(nbits):
    codes = np.random.default_rng(nbits).integers(0, 2**nbits, (5, D), dtype=np.uint8)
    packed = quant.pack_codes(codes, nbits=nbits)
    assert packed.dtype == np.uint8 and packed.shape == (5, D * nbits // 8)
    np.testing.assert_array_equal(quant.unpack_codes(packed, nbits=nbits), codes)
    if nbits == 1:
        np.testing.assert_array_equal(
            packed, np.packbits(codes, axis=1, bitorder="little")
        )


def test_empty_inputs():
    r = np.empty((0, D), dtype=np.float32)
    codes = quant.quantize(r, nbits=4)
    assert codes.shape == (0, D)
    assert quant.code_factors(r, codes, nbits=4, metric="ip").shape == (0,)
    assert quant.pack_codes(codes, nbits=4).shape == (0, D // 2)
    assert quant.unpack_codes(np.empty((0, D // 2), np.uint8), nbits=4).shape == (0, D)
    prefix, ex = quant.split(codes, nbits=4, prefix_bits=1)
    assert prefix.shape == ex.shape == (0, D)


def test_strided_noncontiguous_misaligned_and_memmapped_codes(tmp_path):
    codes = np.random.default_rng(3).integers(0, 16, (6, 128), dtype=np.uint8)
    expected = quant.pack_codes(codes, nbits=4)
    # A column slice of a wider record buffer is read through its row stride.
    records = np.zeros((6, 200), dtype=np.uint8)
    records[:, 9 : 9 + 128] = codes
    np.testing.assert_array_equal(
        quant.pack_codes(records[:, 9 : 9 + 128], nbits=4), expected
    )
    # Rows that are not contiguous are copied.
    np.testing.assert_array_equal(
        quant.pack_codes(codes[:, ::2], nbits=4),
        quant.pack_codes(np.ascontiguousarray(codes[:, ::2]), nbits=4),
    )
    np.testing.assert_array_equal(
        quant.pack_codes(codes[::-1], nbits=4), expected[::-1]
    )
    mapped = np.memmap(
        tmp_path / "codes.bin", dtype=np.uint8, mode="w+", shape=codes.shape
    )
    mapped[:] = codes
    np.testing.assert_array_equal(quant.pack_codes(mapped, nbits=4), expected)
    # Misaligned uint16 codes are copied before typed reads.
    r = residuals()
    wide = quant.quantize(r, nbits=9)
    storage = np.zeros(wide.nbytes + 1, dtype=np.uint8)
    misaligned = np.frombuffer(storage[1:].data, dtype=np.uint16).reshape(wide.shape)
    misaligned[:] = wide
    np.testing.assert_array_equal(
        quant.code_factors(r, misaligned, nbits=9, metric="l2"),
        quant.code_factors(r, wide, nbits=9, metric="l2"),
    )


def test_out_arguments():
    r = residuals()
    out = np.empty(r.shape, dtype=np.uint8)
    assert quant.quantize(r, nbits=4, out=out) is out
    np.testing.assert_array_equal(out, quant.quantize(r, nbits=4))
    factors = np.empty(len(r), dtype=quant.FACTORS_DTYPE)
    assert quant.code_factors(r, out, nbits=4, metric="l2", out=factors) is factors
    packed = np.empty((len(r), D // 2), dtype=np.uint8)
    assert quant.pack_codes(out, nbits=4, out=packed) is packed
    unpacked = np.empty_like(out)
    assert quant.unpack_codes(packed, nbits=4, out=unpacked) is unpacked
    np.testing.assert_array_equal(unpacked, out)
    # A row-strided output is written through its stride.
    wide = np.zeros((len(r), 2 * D), dtype=np.uint8)
    quant.quantize(r, nbits=4, out=wide[:, :D])
    np.testing.assert_array_equal(wide[:, :D], out)

    readonly = np.empty_like(out)
    readonly.flags.writeable = False
    for bad in (
        np.empty(r.shape, dtype=np.uint16),
        np.empty((len(r), D + 32), dtype=np.uint8),
        readonly,
        np.empty((len(r), 2 * D), dtype=np.uint8)[:, ::2],
        out.tolist(),
    ):
        with pytest.raises(ValueError, match="out must be a writable NumPy array"):
            quant.quantize(r, nbits=4, out=bad)
    with pytest.raises(ValueError, match="out must be a writable, C-contiguous"):
        quant.code_factors(
            r, out, nbits=4, metric="l2", out=np.empty(len(r), np.float32)
        )


def test_rejected_arguments():
    r = residuals()
    codes = quant.quantize(r, nbits=4)
    with pytest.raises(ValueError, match=r"nbits must be in \[1, 9\]; got 0"):
        quant.quantize(r, nbits=0)
    with pytest.raises(ValueError, match="positive multiple of 32"):
        quant.quantize(r[:, :40], nbits=4)
    for bad in (np.nan, np.inf):
        nonfinite = r.copy()
        nonfinite[2, 5] = bad
        with pytest.raises(
            ValueError, match="residuals must contain only finite values"
        ):
            quant.quantize(nonfinite, nbits=4)
    with pytest.raises(
        ValueError, match="residuals exceeds the safe float32 coordinate range"
    ):
        quant.quantize(np.full((1, D), 1e30, np.float32), nbits=4)
    with pytest.raises(ValueError, match="metric must be 'l2' or 'ip'"):
        quant.code_factors(r, codes, nbits=4, metric="cosine")
    with pytest.raises(TypeError):
        quant.code_factors(r, codes, nbits=4)  # metric is required
    with pytest.raises(
        ValueError, match="codes must have dtype uint8; got dtype int64"
    ):
        quant.code_factors(r, codes.astype(np.int64), nbits=4, metric="l2")
    with pytest.raises(ValueError, match="codes must have dtype uint8; got dtype bool"):
        quant.pack_codes(codes > 7, nbits=1)
    with pytest.raises(ValueError, match="codes must have dtype uint16"):
        quant.code_factors(r, codes, nbits=9, metric="l2")
    with pytest.raises(ValueError, match="codes must have the shape of residuals"):
        quant.code_factors(r, codes[:4], nbits=4, metric="l2")
    with pytest.raises(
        ValueError,
        match="codes row 0 for residuals row 0: Unpacked code exceeds the configured",
    ):
        quant.code_factors(r, codes, nbits=3, metric="l2")
    with pytest.raises(ValueError, match=r"codes values must be below 2\*\*3 = 8"):
        quant.pack_codes(codes, nbits=3)
    with pytest.raises(
        ValueError, match="codes row 0 for residuals row 0: code does not"
    ):
        quant.code_factors(-r, codes, nbits=4, metric="l2")
    with pytest.raises(ValueError, match=r"centroid must have shape \(96,\)"):
        quant.code_factors(r, codes, nbits=4, metric="l2", centroid=np.zeros(64))
    with pytest.raises(ValueError, match="centroid must contain only finite values"):
        quant.code_factors(r, codes, nbits=4, metric="l2", centroid=np.full(D, np.nan))
    with pytest.raises(ValueError, match=r"prefix_bits must be in \[1, 3\]"):
        quant.split(codes, nbits=4, prefix_bits=4)
    with pytest.raises(ValueError, match=r"nbits must be in \[1, 8\]; got 9"):
        quant.pack_codes(codes, nbits=9)
    with pytest.raises(
        ValueError, match="packed must have d \\* nbits / 8 bytes per row"
    ):
        quant.unpack_codes(np.zeros((2, 10), np.uint8), nbits=3)
    with pytest.raises(
        ValueError, match="fast_quantization requires a working dimension"
    ):
        quant.quantize(
            np.zeros((1, 65568), np.float32), nbits=2, fast_quantization=True
        )


@pytest.mark.parametrize("nbits", range(1, 9))
def test_golden_packed_codes(nbits):
    """Packed-codes version 1 is frozen; see tests/python/fixtures/quant/README.md."""
    assert quant.PACKED_CODES_VERSION == 1
    fixtures = Path(__file__).resolve().parent / "fixtures" / "quant"
    golden = (fixtures / f"packed_codes_nbits{nbits}_n3_d96.bin").read_bytes()
    i, j = np.meshgrid(np.arange(3), np.arange(96), indexing="ij")
    codes = ((i * 37 + j * 11 + (j // 32) * 5) % (1 << nbits)).astype(np.uint8)
    assert quant.pack_codes(codes, nbits=nbits).tobytes() == golden
    unpacked = quant.unpack_codes(
        np.frombuffer(golden, np.uint8).reshape(3, -1), nbits=nbits
    )
    np.testing.assert_array_equal(unpacked, codes)


def test_out_overlapping_an_input_is_rejected():
    rng = np.random.default_rng(7)
    buffer = rng.standard_normal((5, 64)).astype(np.float32)
    with pytest.raises(ValueError, match="out must not overlap residuals"):
        quant.quantize(buffer[:4], nbits=2, out=buffer.view(np.uint8)[1:, :64])
    codes = quant.quantize(buffer, nbits=4)
    with pytest.raises(ValueError, match="out must not overlap codes"):
        quant.pack_codes(codes[:4], nbits=4, out=codes[1:, :32])
    packed = np.zeros((4, 64), np.uint8)
    with pytest.raises(ValueError, match="out must not overlap packed"):
        quant.unpack_codes(packed[:2, :32], nbits=4, out=packed[1:3])
    shared = np.zeros(5 * 64 + 15, np.float32)
    x = shared[: 5 * 64].reshape(5, 64)
    x[:] = buffer
    out = shared[5 * 64 - 15 : 5 * 64].view(quant.FACTORS_DTYPE)
    with pytest.raises(ValueError, match="out must not overlap residuals"):
        quant.code_factors(x, codes, nbits=4, metric="l2", out=out)
    # Disjoint caller buffers are accepted.
    assert quant.quantize(buffer, nbits=4, out=np.empty_like(codes)) is not None


def test_code_factors_leaves_out_unchanged_on_rejection():
    rng = np.random.default_rng(8)
    x = rng.standard_normal((6, 64)).astype(np.float32)
    codes = quant.quantize(x, nbits=4)
    codes[4] = 15 - codes[4]
    out = np.zeros(6, quant.FACTORS_DTYPE)
    with pytest.raises(ValueError, match="codes row 4 for residuals row 4"):
        quant.code_factors(x, codes, nbits=4, metric="l2", out=out)
    assert not out.view(np.float32).any()


def test_rejected_codes_leave_pack_out_unchanged():
    # The codes are checked in a pass of their own, before out= is written.
    codes = quant.quantize(residuals(300, seed=11), nbits=4)
    codes[200, 7] = 16
    out = np.zeros((300, D * 4 // 8), np.uint8)
    with pytest.raises(ValueError, match=r"got 16 at \(200, 7\)$"):
        quant.pack_codes(codes, nbits=4, num_threads=4, out=out)
    assert not out.any()


def test_code_factors_names_the_lowest_rejected_row_across_threads():
    # 1000 rows span several parallel ranges; the lowest rejected row is reported.
    x = residuals(1000, seed=10)
    codes = quant.quantize(x, nbits=4)
    codes[[700, 300]] = 15 - codes[[700, 300]]
    for threads in (1, 4):
        with pytest.raises(
            ValueError, match="^codes row 300 for residuals row 300: code does not"
        ):
            quant.code_factors(x, codes, nbits=4, metric="l2", num_threads=threads)


@pytest.mark.parametrize("metric", ["l2", "ip"])
def test_code_factors_reports_the_overflowing_row(metric):
    # With every code 1, <residual, centered code> is half the residual's sum. The
    # +-2**25 at coordinates 0 and 64 share an accumulator slot in every dot-product
    # backend and cancel exactly, and coordinate 65 keeps its tiny value apart, so the
    # sum is a positive 1e-30 and the rescale factor overflows float32.
    x = residuals(6, seed=9)
    codes = quant.quantize(x, nbits=1)
    for row in (3, 5):
        x[row] = 0
        x[row, [0, 64, 65]] = [2.0**25, -(2.0**25), 1e-30]
        codes[row] = 1
    out = np.zeros(6, quant.FACTORS_DTYPE)
    with pytest.raises(
        OverflowError,
        match="^codes row 3 for residuals row 3: Quantization factors are not "
        "representable in float32$",
    ):
        quant.code_factors(x, codes, nbits=1, metric=metric, num_threads=4, out=out)
