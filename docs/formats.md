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
| `FASTSCAN_BLOCKS_VERSION` | 1 | `pack_fastscan_blocks` output: one row of `d * 4 + 384` bytes per block of 32 vectors |

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

### FastScan blocks, version 1

A block holds 32 vectors, the library's fixed FastScan batch (`fastscan::kBatchSize`), in the C++ `BatchDataMap<float>` layout. Vectors `32b` to `32b + 31` go to block `b`; in the last block, lanes past `n` have zero codes and zero factors.

| Offset | Size | Content |
| --- | --- | --- |
| 0 | `d * 4` | Sign codes, `d / 8` groups of 32 bytes |
| `d * 4` | 128 | `add` of lanes 0–31, float32 |
| `d * 4 + 128` | 128 | `rescale` of lanes 0–31, float32 |
| `d * 4 + 256` | 128 | `error` of lanes 0–31, float32 |

Group `i` covers coordinates `8i` to `8i + 7`. Let `a[v]` be byte `i` of vector `v`'s bitset with its bit order reversed, so coordinate `8i` is in bit 7. With `perm = [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15]`, for `j` in 0–15, `p = perm[j]`:

- byte `j` is `(a[p] >> 4) | (a[p + 16] & 0xF0)`: coordinates `8i` to `8i + 3` of vectors `p` (low nibble) and `p + 16` (high nibble);
- byte `j + 16` is `(a[p] & 0x0F) | (a[p + 16] << 4)`: coordinates `8i + 4` to `8i + 7` of the same vectors.

This is the order `fastscan::pack_codes` writes. The high-accuracy lookup table that scores the blocks is internal and not part of the format.

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
