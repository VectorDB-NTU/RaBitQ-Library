"""Python boundary coverage for quant.Rotator and the rotator blob format."""

import copy
import pickle
import zlib
from pathlib import Path

import numpy as np
import pytest
from rabitqlib import quant

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "quant"


def vectors(n, dim, seed=0):
    return np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)


@pytest.mark.parametrize("method", ["fht_kac", "matrix"])
def test_rotation_preserves_norms_and_round_trips(method):
    rotator = quant.Rotator(100, method=method, seed=5)
    assert (rotator.dim, rotator.padded_dim, rotator.method, rotator.seed) == (
        100,
        128,
        method,
        5,
    )
    x = vectors(7, 100)
    y = rotator.rotate(x)
    assert y.dtype == np.float32 and y.shape == (7, 128)
    np.testing.assert_allclose(
        np.linalg.norm(y, axis=1), np.linalg.norm(x, axis=1), rtol=1e-4
    )
    for loaded in (
        quant.Rotator.from_bytes(rotator.to_bytes()),
        quant.Rotator.from_bytes(bytearray(rotator.to_bytes())),
        quant.Rotator.from_bytes(memoryview(rotator.to_bytes())),
        quant.Rotator.from_bytes(np.frombuffer(rotator.to_bytes(), np.uint8)),
        pickle.loads(pickle.dumps(rotator)),
    ):
        assert repr(loaded) == repr(rotator)
        np.testing.assert_array_equal(loaded.rotate(x), y)
    assert copy.copy(rotator) is rotator and copy.deepcopy(rotator) is rotator


def test_seeds_threads_and_out():
    x = vectors(50, 64)
    first = quant.Rotator(64, seed=11)
    np.testing.assert_array_equal(quant.Rotator(64, seed=11).rotate(x), first.rotate(x))
    assert not np.array_equal(quant.Rotator(64, seed=12).rotate(x), first.rotate(x))
    np.testing.assert_array_equal(
        first.rotate(x, num_threads=1), first.rotate(x, num_threads=4)
    )
    drawn = quant.Rotator(64)
    assert isinstance(drawn.seed, int)
    np.testing.assert_array_equal(
        quant.Rotator(64, seed=drawn.seed).rotate(x), drawn.rotate(x)
    )
    assert quant.Rotator(64, method="fht").method == "fht_kac"
    out = np.empty((50, 64), np.float32)
    assert first.rotate(x, out=out) is out
    np.testing.assert_array_equal(out, first.rotate(x))
    assert first.rotate(np.empty((0, 64), np.float32)).shape == (0, 64)


def test_golden_blobs():
    for name, rotator in (
        ("rotator_fht_kac_dim64_seed0.bin", quant.Rotator(64, seed=0)),
        (
            "rotator_fht_kac_dim100_padded128_seed42.bin",
            quant.Rotator(100, padded_dim=128, seed=42),
        ),
    ):
        golden = (FIXTURES / name).read_bytes()
        assert rotator.to_bytes() == golden, name
        assert repr(quant.Rotator.from_bytes(golden)) == repr(rotator)


def test_blob_header_fields():
    blob = quant.Rotator(100, padded_dim=128, seed=42).to_bytes()
    assert blob[:8] == b"RABQBLOB"
    fields = np.frombuffer(blob[8:64], dtype="<u4")
    kind, version, header_bytes, flags = fields[:4]
    assert (kind, version, header_bytes, flags) == (1, 1, 64, 1)
    assert int.from_bytes(blob[24:32], "little") == len(blob) - 64 == 64
    method, scalar, dim, padded_dim, seed, reserved = fields[8:]
    assert (method, scalar, dim, padded_dim, seed, reserved) == (1, 1, 100, 128, 42, 0)
    unchecked = blob[:32] + b"\0\0\0\0" + blob[36:]
    assert int.from_bytes(blob[32:36], "little") == zlib.crc32(unchecked)


def patched(blob, offset, value):
    """blob with a u32 field replaced and the checksum recomputed."""
    data = bytearray(blob)
    data[offset : offset + 4] = int(value).to_bytes(4, "little")
    data[32:36] = b"\0\0\0\0"
    data[32:36] = zlib.crc32(bytes(data)).to_bytes(4, "little")
    return bytes(data)


def test_corrupt_and_unsupported_blobs():
    blob = quant.Rotator(64, seed=0).to_bytes()
    cases = [
        (blob[:39], "too short"),
        (b"X" + blob[1:], "bad magic"),
        (patched(blob, 8, 2), "not a rotator blob; got kind 2"),
        (patched(blob, 12, 2), "unsupported rotator format version 2"),
        (patched(blob, 16, 72), "invalid header size"),
        (blob[:-1], "truncated or padded"),
        (blob + b"\0", "truncated or padded"),
        (blob[:70] + bytes([blob[70] ^ 1]) + blob[71:], "checksum mismatch"),
        (patched(blob, 20, 3), "unknown flags"),
        (patched(blob, 36, 1), "non-zero reserved"),
        (patched(blob, 60, 1), "non-zero reserved"),
        (patched(blob, 40, 3), "unknown rotator method 3"),
        (patched(blob, 44, 2), "float64 rotator"),
        (patched(blob, 44, 7), "unknown scalar type 7"),
        (patched(blob, 48, 63), "invalid rotator dimensions"),
        (patched(blob, 52, 96), "body size does not match"),
        (patched(patched(blob, 20, 0), 56, 9), "seed without the seed_known flag"),
    ]
    for data, message in cases:
        with pytest.raises(quant.FormatError, match=message):
            quant.Rotator.from_bytes(data)
    assert issubclass(quant.FormatError, ValueError)
    assert issubclass(quant.FormatError, RuntimeError)
    unknown_seed = quant.Rotator.from_bytes(patched(patched(blob, 20, 0), 56, 0))
    assert unknown_seed.seed is None
    with pytest.raises(TypeError, match="contiguous bytes-like"):
        quant.Rotator.from_bytes(memoryview(blob)[::2])


def test_rejected_arguments():
    with pytest.raises(
        ValueError, match=r"dim must be at least 64 for method 'fht_kac'"
    ):
        quant.Rotator(40)
    with pytest.raises(ValueError, match=r"dim must be in \[1, 65536\]"):
        quant.Rotator(0, method="matrix")
    with pytest.raises(ValueError, match="padded_dim must be a multiple of 32"):
        quant.Rotator(64, padded_dim=80)
    with pytest.raises(ValueError, match="padded_dim must be a multiple of 32"):
        quant.Rotator(100, padded_dim=96)
    with pytest.raises(TypeError, match="padded_dim must be an integer"):
        quant.Rotator(64, padded_dim=True)
    with pytest.raises(ValueError, match="method must be 'fht_kac'"):
        quant.Rotator(64, method="hadamard")
    with pytest.raises(
        TypeError, match=r"seed must be an integer in \[0, 4294967295\]"
    ):
        quant.Rotator(64, seed=True)
    with pytest.raises(
        ValueError, match=r"seed must be an integer in \[0, 4294967295\]"
    ):
        quant.Rotator(64, seed=2**32)
    rotator = quant.Rotator(64, seed=0)
    with pytest.raises(ValueError, match="x dimension does not match dim"):
        rotator.rotate(vectors(2, 65))
    with pytest.raises(ValueError, match="x must contain only finite values"):
        rotator.rotate(np.full((1, 64), np.nan, np.float32))
    with pytest.raises(ValueError, match="out must be a writable NumPy array"):
        rotator.rotate(vectors(2, 64), out=np.empty((2, 64), np.float64))
    shared = vectors(3, 64)
    with pytest.raises(ValueError, match="out must not overlap x"):
        rotator.rotate(shared[:2], out=shared[1:])
    with pytest.raises(ValueError, match="out must not overlap x"):
        rotator.rotate(shared, out=shared)
