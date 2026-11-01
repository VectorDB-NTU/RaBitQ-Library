# Byte formats

This is the specification of the byte formats `rabitqlib.quant` reads and writes. It is shared with the planned index format revision, which will add its own blob kinds. While `rabitqlib.quant` is provisional, a release may stop reading an old format version; reading it then raises `quant.FormatError` instead of returning wrong results. A promise that every released version stays readable comes with stabilisation.

Every format has a version. A change to any layout is a new version, never an edit of an existing one. The golden files in `tests/python/fixtures/quant/` freeze each version and are never regenerated.

All multi-byte fields are little-endian.

## Raw arrays

Raw arrays have no header: their layout is fixed by the version below, which the module exports as a constant. A caller who stores raw arrays records the constant next to them.

| Constant | Version | Format |
| --- | --- | --- |
| `PACKED_CODES_VERSION` | 1 | `pack_codes` output: one row of `d * nbits / 8` bytes per vector, for `nbits` 1–8 and `d` a positive multiple of 32 |
| `FACTORS_VERSION` | 1 | `FACTORS_DTYPE`, `ESTIMATE_DTYPE` and `QUERY_TERMS_DTYPE`: three `<f4` fields each, 12 bytes per record with no padding |

### Packed codes, version 1

The layout `packing_rabitqplus_code` writes, which the C++ uses for both the prefix (base) code and the extra bits. The coordinate-to-bit mapping of each width is in [Compact code storage](docs/compact_code.md): full 64-coordinate blocks, then a compact final 32-coordinate tail. It does not depend on the SIMD backend that wrote it.

At `nbits = 1` it is the sign bitset: bits are LSB-first within each byte, so a row equals `numpy.packbits(codes, axis=1, bitorder="little")`.

### Factor records, version 1

| Dtype | Fields, in order |
| --- | --- |
| `FACTORS_DTYPE` | `add`, `rescale`, `error` |
| `ESTIMATE_DTYPE` | `distance`, `error`, `refine_input` |
| `QUERY_TERMS_DTYPE` | `g_add`, `g_error`, `g_k1xsumq` |

Each field is a little-endian float32 at offsets 0, 4 and 8. Consumers that do not take structured dtypes can use `a.view(np.float32).reshape(-1, 3)`.

## Blobs

A blob is a self-describing byte string, such as `Rotator.to_bytes()`. It starts with a 40-byte common header, followed by the fields of its kind and then its body.

### Common header

| Offset | Size | Type | Field | Rule |
| --- | --- | --- | --- | --- |
| 0 | 8 | ASCII | `magic` = `"RABQBLOB"` | Anything else raises `FormatError` |
| 8 | 4 | u32 | `kind` | 1 = rotator; 2 = codec (reserved); 3 and up reserved for index formats. A kind the reader does not expect raises `FormatError` |
| 12 | 4 | u32 | `version` | Layout version of this kind; an unknown version raises `FormatError` |
| 16 | 4 | u32 | `header_bytes` | Common header plus kind fields, a multiple of 8; it must equal the size the kind and version define |
| 20 | 4 | u32 | `flags` | Kind-specific; unknown bits raise `FormatError` |
| 24 | 8 | u64 | `body_bytes` | The blob is exactly `header_bytes + body_bytes` long |
| 32 | 4 | u32 | `crc32` | CRC-32 as `zlib.crc32` computes it, over the whole blob with this field taken as zero; a mismatch raises `FormatError` |
| 36 | 4 | — | reserved | Writers write zero; readers reject anything else |

Readers check, in order: the length covers the common header, the magic, the kind, the version, `header_bytes`, the total length, the checksum, and then the flags, reserved fields and kind fields. Sizes are validated before anything is allocated. The body starts at a multiple of 8, so an aligned buffer can be read in place.

### Rotator, kind 1, version 1

`header_bytes` is 64.

| Offset | Size | Type | Field | Rule |
| --- | --- | --- | --- | --- |
| 40 | 4 | u32 | `method` | 1 = FHT/Kac, 2 = matrix |
| 44 | 4 | u32 | `scalar` | 1 = float32. 2 = float64 is reserved and rejected in version 1 |
| 48 | 4 | u32 | `dim` | 1–65536 for matrix; 64–65536 for FHT/Kac |
| 52 | 4 | u32 | `padded_dim` | A multiple of 32 in `[dim, 65536]` |
| 56 | 4 | u32 | `seed` | Informational. Valid only when flag bit 0 (`seed_known`) is set; zero otherwise |
| 60 | 4 | — | reserved | Writers write zero; readers reject anything else |

Flag bit 0, `seed_known`, is the only defined flag.

The body is what `Rotator<float>::save()` writes, and `body_bytes` must equal its size for the stored parameters:

| Method | Body | `body_bytes` |
| --- | --- | --- |
| FHT/Kac | The sign-flip bytes | `padded_dim / 2` |
| Matrix | The rotation matrix as row-major float32 | `4 * dim * padded_dim` |

The body defines the rotation; `seed` only records where it came from. A seeded FHT/Kac rotation is identical on every platform, while a matrix drawn from a seed is reproducible only with the same standard library and build, which is why the matrix itself is stored.
