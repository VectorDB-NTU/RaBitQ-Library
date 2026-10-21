import struct

import numpy as np


def read_ivecs(filename):
    print(f"Reading File - {filename}")
    a = np.fromfile(filename, dtype="int32")
    d = a[0]
    print(f"\t{filename} readed")
    return a.reshape(-1, d + 1)[:, 1:]


def read_fvecs(filename):
    return read_ivecs(filename).view("float32")


_WRITE_BLOCK_BYTES = 4 * 1024 * 1024


def _blocks(m, order=None):
    rows = max(1, _WRITE_BLOCK_BYTES // ((m.shape[1] + 1) * 4))
    count = len(m) if order is None else len(order)
    for start in range(0, count, rows):
        yield (
            m[start : start + rows] if order is None else m[order[start : start + rows]]
        )


def _write_records(stream, blocks, dim):
    for block in blocks:
        # Preserve struct.pack's native byte order and signed-int32 validation.
        struct.pack("i", dim)
        if block.shape[1] != dim:
            raise struct.error(
                f"pack expected {dim} items for packing (got {block.shape[1]})"
            )
        if block.dtype.kind in "iu":
            if block.size and (block.min() < -(2**31) or block.max() > 2**31 - 1):
                raise struct.error(
                    "'i' format requires -2147483648 <= number <= 2147483647"
                )
        else:
            # Object arrays retain struct.pack's __index__ support; floats are rejected.
            packed = struct.pack(f"{block.size}i", *block.flat)
            block = np.frombuffer(packed, dtype="=i4").reshape(block.shape)
        records = np.empty((len(block), dim + 1), dtype="=i4")
        records[:, 0] = dim
        records[:, 1:] = block
        records.tofile(stream)


def write_ivecs(filename, m):
    print(f"Writing File - {filename}")
    _, dim = m.shape
    with open(filename, "wb") as stream:
        _write_records(stream, _blocks(m), dim)
    print(f"\t{filename} wrote")


def write_fvecs(filename, m):
    print(f"Writing File - {filename}")
    _, dim = m.shape
    blocks = (block.astype("float32").view("int32") for block in _blocks(m))
    with open(filename, "wb") as stream:
        _write_records(stream, blocks, dim)
    print(f"\t{filename} wrote")


def read_ibin(filename):
    n, d = np.fromfile(filename, count=2, dtype="int32")
    a = np.fromfile(filename, dtype="int32")
    print(f"\t{filename} readed")
    return a[2:].reshape(n, d)


def read_fbin(filename):
    return read_ibin(filename).view("float32")


def write_sampled_data(filename, m, num_sampled, seed=42):
    num_points = m.shape[0]
    d = m.shape[1]
    m = m.view("int32")

    np.random.seed(seed)
    sequence = np.random.permutation(num_points)

    count = len(range(num_sampled))
    if count > num_points:
        raise IndexError("sample count exceeds the number of points")
    with open(filename, "wb") as stream:
        _write_records(stream, _blocks(m, sequence[:count]), d)
