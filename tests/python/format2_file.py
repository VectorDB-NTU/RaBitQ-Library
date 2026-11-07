"""Read, patch and convert index format 2 files in tests (docs/formats.md)."""

from __future__ import annotations

import struct
import zlib

MAGIC = b"RABQINDX"
ROTATOR = 1
IVF_PARAMS = 256
IVF_CLUSTER_SIZES = 257
IVF_ROUTING = {1: 258, 2: 259, 3: 260}  # by stored initializer: flat, flat_rabitq, hnsw
IVF_HNSW_ROUTING = 260
IVF_CODES = 261
IVF_RERANK = 262
IVF_IDS = 263
HNSW_PARAMS = 512
QG_PARAMS = 768
QG_CENTROID = 769
QG_RECORDS = 770


def sections(data: bytes) -> dict[int, tuple[int, int]]:
    """Map each section kind of a format-2 file to its (offset, size)."""
    table, count = struct.unpack_from("<QI", data, 96)
    result = {}
    for i in range(count):
        kind, _flags, offset, size = struct.unpack_from("<IIQQ", data, table + 32 * i)
        result[kind] = (offset, size)
    return result


def section(data: bytes, kind: int) -> bytes:
    offset, size = sections(data)[kind]
    return bytes(data[offset : offset + size])


def padded_dim(data: bytes) -> int:
    return struct.unpack_from("<Q", data, 56)[0]


def reseal(data: bytearray) -> None:
    """Recompute section and header checksums after a same-size edit in place."""
    table, count = struct.unpack_from("<QI", data, 96)
    for i in range(count):
        entry = table + 32 * i
        _kind, flags, offset, size = struct.unpack_from("<IIQQ", data, entry)
        if flags & 2:  # checksummed
            crc = zlib.crc32(bytes(data[offset : offset + size]))
            struct.pack_into("<I", data, entry + 24, crc)
    header_bytes = struct.unpack_from("<I", data, 12)[0]
    header = bytearray(data[:header_bytes])
    header[108:112] = bytes(4)
    crc = zlib.crc32(bytes(data[table : table + 32 * count]), zlib.crc32(header))
    struct.pack_into("<I", data, 108, crc)


def to_rabqidx1(data: bytes) -> tuple[bytes, bytes | None]:
    """Convert an IVF-RaBitQ format-2 file to the layout releases up to 0.8 wrote.

    Returns the RABQIDX1 version 2 index and, for HNSW routing, its ``.hnsw``
    sidecar. Tests use it to build earlier layouts that no fixture covers.
    """
    assert data[:8] == MAGIC
    present = sections(data)
    metric = struct.unpack_from("<I", data, 44)[0]
    dim, padded = struct.unpack_from("<QQ", data, 48)
    code_bits, raw_type = struct.unpack_from("<II", data, 64)
    num_points = struct.unpack_from("<Q", data, 88)[0]
    num_clusters, initializer = struct.unpack_from("<QI", section(data, IVF_PARAMS))
    blob = section(data, ROTATOR)
    rotator_type = 1 if struct.unpack_from("<I", blob, 40)[0] == 1 else 0
    raw = raw_type == 1
    routing = section(data, IVF_ROUTING[initializer])
    main = b"RABQIDX1" + struct.pack("<IIQI", 2, int(raw), padded, initializer)
    main += struct.pack(
        "<QQQQBB",
        num_points,
        dim,
        num_clusters,
        0 if raw else code_bits - 1,
        rotator_type,  # RotatorType: MatrixRotator 0, FhtKacRotator 1
        metric - 1,  # MetricType: L2 0, inner product 1
    )
    main += section(data, IVF_CLUSTER_SIZES) + blob[64:]
    sidecar = routing if initializer == 3 else None
    if sidecar is None:
        main += routing
    main += section(data, IVF_CODES)
    if IVF_RERANK in present:
        main += section(data, IVF_RERANK)
    main += section(data, IVF_IDS)
    return main, sidecar
