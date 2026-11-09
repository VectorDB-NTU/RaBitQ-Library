# Byte formats

This is the specification of the byte formats `rabitqlib.quant` reads and writes. It is shared with the planned index format revision, which will add its own blob kinds. While `rabitqlib.quant` is provisional, a release may stop reading an old format version; reading it then raises `rabitqlib.FormatError` (also `quant.FormatError`) instead of returning wrong results. A promise that every released version stays readable comes with stabilisation.

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

<a id="index-format-2"></a>

## Index files, format 2 (draft)

> **Status: draft for 0.9 (#211, #212).** The container (`src/index/format2.*`) is implemented, and IVF-RaBitQ, HNSW-RaBitQ and SymphonyQG read and write format 2, including the cosine metric; IVF-SSD and companion files are not implemented. Field numbers, section kinds and the open questions at the end may still change until a release writes format 2.

Format 2 is one container for IVF-RaBitQ, HNSW-RaBitQ and SymphonyQG, and later for the SSD-resident IVF index from RaBitQ-SSD. It replaces the per-index layouts: IVF-RaBitQ `RABQIDX1` versions 1 and 2, the unversioned HNSW-RaBitQ layout, SymphonyQG's quantized version 1 and its raw layout, and the unversioned hnswlib sidecar `<file>.hnsw` of IVF-RaBitQ. Every released layout stays readable (see [Earlier layouts](#earlier-layouts)); `save` always writes format 2.

The number continues the name used in #212. An index file is told apart from earlier layouts by its magic, not by the version, so the unrelated IVF-RaBitQ `RABQIDX1` version 2 does not clash.

Unlike the provisional `rabitqlib.quant` formats above, index files carry the library's full compatibility promise: every format-2 file written by a release stays readable by every later release.

### Conventions

- Multi-byte fields are little-endian; types are fixed-width (`u32`, `u64`, `i32`, `f32`, `f64`). No field is `size_t`.
- Offsets are absolute byte offsets from the start of the file.
- The section table starts at a multiple of 8, and every section starts at a multiple of 64, so a reader can map the file and read sections in place. Writers fill gaps with zero bytes, and readers do not inspect them.
- Reserved fields and undefined flag bits are written as zero. A reader rejects a nonzero value unless a rule below says otherwise.
- Enumerations start at 1, so zero always means unset; their values are never reused or renumbered.
- Checksums are CRC-32 as `zlib.crc32` computes it, the same as in [blobs](#blobs).

### Layout

| Part | Position |
| --- | --- |
| Header | Offset 0, `header_bytes` long |
| Section table | `section_table_offset`, `section_count` entries of 32 bytes |
| Sections | Anywhere after the header, each at a multiple of 64 |

Every section starts at or after `header_bytes`. Non-empty sections do not overlap the table or each other; empty sections take no bytes and may share an offset. Table entries are in non-decreasing offset order. The file ends exactly where the table or the last section ends, whichever is later; an empty section still has an aligned offset, so writers pad up to it.

### Header

`header_bytes` is 128 in this version. It is a multiple of 8 and at least 128; a later version may append fields, which a reader that knows only these 128 bytes skips (see [Compatibility rules](#compatibility-rules)).

| Offset | Size | Type | Field | Rule |
| --- | --- | --- | --- | --- |
| 0 | 8 | ASCII | `magic` = `"RABQINDX"` | Anything else is not a format-2 file; the reader tries [earlier layouts](#earlier-layouts) |
| 8 | 4 | u32 | `version` = 2 | Any other value is rejected |
| 12 | 4 | u32 | `header_bytes` | See above |
| 16 | 8 | u64 | `incompat_features` | See [Feature flags](#feature-flags) |
| 24 | 8 | u64 | `ro_compat_features` | |
| 32 | 8 | u64 | `compat_features` | |
| 40 | 4 | u32 | `index_kind` | 1 = IVF-RaBitQ, 2 = HNSW-RaBitQ, 3 = SymphonyQG; 4 = IVF-SSD (reserved). Unknown values are rejected |
| 44 | 4 | u32 | `metric` | 1 = L2, 2 = inner product, 3 = cosine. See [Metrics](#metrics) |
| 48 | 8 | u64 | `dim` | Original dimension, at least 1 |
| 56 | 8 | u64 | `padded_dim` | Working dimension; a multiple of 32, at least `dim` and at most 65,536 (the rotator's limit). Readers use the stored value and never recompute it |
| 64 | 4 | u32 | `code_bits` | Bits per coordinate of the stored quantized code, 1–9; 0 when the index stores no quantized code of its points |
| 68 | 4 | u32 | `raw_type` | Element type of stored raw vectors: 0 = none, 1 = float32, 2 = uint8 |
| 72 | 4 | u32 | `id_bits` | Width of the point IDs stored in the file: 32 or 64 |
| 76 | 4 | u32 | `flags` | Bit 0, `seed_known`: `seed` is valid |
| 80 | 8 | u64 | `seed` | The index's construction seed. Zero when `seed_known` is clear, which a file migrated from an earlier layout has. IVF-RaBitQ: the rotation and training seed, at most 2^32 − 1. HNSW-RaBitQ: the graph-level and training seed, whose low 32 bits seeded the rotation. SymphonyQG: the rotation seed, at most 2^32 − 1, which Python also uses for graph construction. The rotator blob records the rotation seed again |
| 88 | 8 | u64 | `num_points` | Stored points, including removed ones |
| 96 | 8 | u64 | `section_table_offset` | A multiple of 8, at least `header_bytes` |
| 104 | 4 | u32 | `section_count` | At least 1 |
| 108 | 4 | u32 | `header_crc32` | Over the header with this field taken as zero, followed by the section table |
| 112 | 16 | ASCII | `writer_version` | Library version that wrote the file, printable ASCII, NUL-padded. For error messages only; readers never branch on it |

The Python `nbits` maps to these fields as follows. IVF-RaBitQ `nbits` 1–9 is `code_bits`. IVF-RaBitQ `nbits = 32` is `code_bits = 1` with `raw_type = float32`, because it keeps 1-bit FastScan codes and reranks with raw vectors. SymphonyQG `nbits = 32` is `code_bits = 0` with `raw_type = float32`.

### Section table

| Offset | Size | Type | Field | Rule |
| --- | --- | --- | --- | --- |
| 0 | 4 | u32 | `kind` | See [Section kinds](#section-kinds) |
| 4 | 4 | u32 | `flags` | Bit 0, `required`; bit 1, `checksummed` |
| 8 | 8 | u64 | `offset` | A multiple of 64 |
| 16 | 8 | u64 | `size` | Bytes; may be 0 |
| 24 | 4 | u32 | `crc32` | Over the section's bytes when `checksummed` is set; zero otherwise |
| 28 | 4 | — | reserved | |

Kind 0 is reserved, and a kind appears at most once. Section flags other than the two above are rejected; a later version cannot add a section flag without a new `version`. Writers checksum the header-like sections (parameters, cluster sizes, companions), whose values size later allocations, and leave bulk data unchecksummed, so that loading a large index does not read it twice. Per-point arrays such as `hnsw_levels` count as bulk data: the header fixes their size, and readers check every value against its limit.

A section the reader does not know is handled by its `required` bit: a required one rejects the file, and an optional one is skipped without reading it, so its checksum is not verified. A writer marks a section optional only if dropping it leaves a correct index, for example a cache that can be recomputed. `save` drops optional sections it does not know.

### Feature flags

The three masks follow the ext4 superblock: they say what an older reader may do with a file that uses something newer than itself.

| Mask | A reader that does not know a set bit |
| --- | --- |
| `incompat_features` | Rejects the file; reading it would give wrong results |
| `ro_compat_features` | May load and search; operations that change or save the index (`add`, `reserve`, `remove`, `save`) raise, because they would leave the unknown data stale. Rebuilding replaces that data and makes the index writable again: IVF-RaBitQ `construct`, or a new SymphonyQG build; HNSW-RaBitQ cannot be rebuilt in place |
| `compat_features` | Ignores the bit; `save` clears it |

The error names the unknown bits and `writer_version`, such as "index file uses incompatible features 0x4 that this release does not know; written by rabitqlib 0.10.0".

A writer sets only the bits it knows. Resaving a file therefore drops the unknown `compat` bits, together with the unknown optional sections and header fields past the writer's `header_bytes`. A `compat` bit may describe only data whose loss on such a resave is harmless.

No bit is defined in version 2. Bits are assigned when a feature needs one. Examples:

| Possible feature | Mask | Why |
| --- | --- | --- |
| Caller-provided IDs (`add(x, ids=...)`), an external-ID section | `incompat` | An older reader would return row numbers instead of the caller's IDs |
| A per-cluster statistics section that `add` updates | `ro_compat` | An older reader can search, but its `add` would not update the section |
| A new code layout of an existing section | `incompat`, or a new section kind | The bytes mean something else |

### Metrics

L2 and inner product are as in [Distance conventions](development.md#distance-conventions). Cosine stores the codes and factors of inner product over L2-normalized vectors. The index normalizes data on `train`, `add` and construction, and queries on `search`, and reports `1 - cos` as the distance. It is a stored metric, not a flag of the caller, so a reloaded index keeps normalizing.

Normalization computes `scale = 1 / sqrt(s)` in double precision, where `s` is the sum of the squared coordinates added in coordinate order in double precision; each coordinate is multiplied by `scale` in double precision and rounded once to float32. The product of two float32 values is exact in double precision, so the sum does not depend on whether a compiler fuses multiply and add. A zero vector stays zero, so its true distance to every vector is 1. Every vector section (codes, raw float32 vectors and the IVF rerank vectors) holds the normalized vectors, so a cosine file equals the inner-product file of the normalized vectors except for `metric`. Centroids are stored as the index was given or trained them.

### Point IDs

Python `search`, `add` and `remove` use `int64` IDs, and a missing search result is `-1` (#211). `id_bits` records what the file stores. The in-memory indexes keep 32-bit row numbers internally and write `id_bits = 32`; their upper limit stays below 2^31 points (the search-buffer marker bit). IVF-SSD writes `id_bits = 64`. A reader rejects an `id_bits` its index kind does not support.

### Section kinds

Kinds 1–255 are shared, 256–511 are IVF-RaBitQ, 512–767 HNSW-RaBitQ, 768–1023 SymphonyQG and 1024–1279 IVF-SSD. Every section listed here is `required`.

| Kind | Name | Used by | Content |
| --- | --- | --- | --- |
| 1 | `rotator` | All | A [rotator blob](#rotator-kind-1-version-1), byte for byte what `Rotator.to_bytes()` writes. Its `dim` and `padded_dim` must equal the header's. Not checksummed by the container: the blob carries its own CRC-32 |
| 2 | `companions` | IVF-SSD | The [companion table](#companion-files) |
| 256 | `ivf_params` | IVF | `num_clusters` u64; `initializer` u32 (1 = flat, 2 = FlatRaBitQ, 3 = HNSW); reserved u32 |
| 257 | `ivf_cluster_sizes` | IVF | `num_clusters` × u64; the sum equals `num_points` |
| 258 | `ivf_flat_routing` | IVF, flat | The rotated centroids, `num_clusters × padded_dim` f32 |
| 259 | `ivf_flat_rabitq_routing` | IVF, FlatRaBitQ | The rotated centroids as above, the global centroid (their mean, `padded_dim` f32), then `ceil(num_clusters / 32)` [FastScan blocks](#fastscan-blocks-version-1) of the centroids, quantized against the global centroid under the index metric |
| 260 | `ivf_hnsw_routing` | IVF, HNSW | The [HNSW routing graph](#hnsw-routing-graph), byte for byte the `<file>.hnsw` sidecar of earlier releases |
| 261 | `ivf_codes` | IVF | Per cluster in cluster order, `ceil(size / 32)` [FastScan blocks](#fastscan-blocks-version-1). Removed points have `add = +inf` |
| 262 | `ivf_rerank` | IVF, `code_bits > 1` or raw | Per point in cluster order: a [single-code record](#single-code-record) with `b = code_bits - 1`, or `dim` raw values of `raw_type` |
| 263 | `ivf_ids` | IVF | Per point in cluster order, `id_bits`-wide IDs |
| 512 | `hnsw_params` | HNSW | `max_elements`, `num_clusters`, `M`, `upper_max_degree`, `base_max_degree`, `ef_construction` as u64; `level_multiplier` f64; `max_level` i32; reserved u32; `entry_point` u64. Record sizes are derived, not stored (see [HNSW-RaBitQ records](#hnsw-rabitq-records)) |
| 513 | `hnsw_centroids` | HNSW | `num_clusters × padded_dim` f32 |
| 514 | `hnsw_records` | HNSW | `num_points` [base-layer records](#hnsw-rabitq-records) |
| 515 | `hnsw_levels` | HNSW | `num_points` × u32 levels |
| 516 | `hnsw_upper_links` | HNSW | The upper-layer link lists of points with level > 0 in point order; each is `level × upper_links_bytes` |
| 768 | `qg_params` | SymphonyQG | `max_degree` u64; `entry_point` u64; `ef_construction` u64 (0 = unknown); `init` u32 (0 = unknown, 1 = PiPNN, 2 = random); reserved u32 |
| 769 | `qg_centroid` | SymphonyQG, `code_bits > 0` | `padded_dim` f32 |
| 770 | `qg_records` | SymphonyQG | `num_points` [records](#symphonyqg-records) |

An IVF file has exactly one routing section, the one that `ivf_params` names; a rerank section exactly when `code_bits > 1` or raw vectors are stored.

The HNSW per-point length prefix of the current layout becomes `hnsw_levels`, so a reader knows every size before reading the link lists.

### Record layouts

These are the in-memory layouts the indexes use today, written as is, so that `load` copies sections without converting them. Record sizes follow from the header and parameter sections; readers derive them and compare them with the section sizes. Floats inside records are not aligned. Reserved bytes inside bulk data are written as zero and ignored by readers, which do not scan bulk data for them.

The tables use `id_bits = 32`, so an ID takes 4 bytes; this is the only value the in-memory indexes write.

#### Single-code record

One vector's `b`-bit code with its two factors. IVF-RaBitQ and HNSW-RaBitQ use it for the extra bits (`b = code_bits - 1`), SymphonyQG for its whole code (`b = code_bits`).

| Offset | Size | Content |
| --- | --- | --- |
| 0 | `padded_dim * b / 8` | The codes as [packed codes](#packed-codes-version-1) of width `b` |
| `padded_dim * b / 8` | 4 | `add`, f32 |
| `padded_dim * b / 8 + 4` | 4 | `rescale`, f32 |

The record has no `error` factor. It should also become a raw-array format of `rabitqlib.quant`, so that both share one definition.

#### HNSW sign code

HNSW-RaBitQ stores the 1-bit code in a different bit order from the `rabitqlib.quant` bitset. Split the `padded_dim` coordinates into groups of 64, where the last group has 32 coordinates when `padded_dim` is not a multiple of 64. A group of `G` bytes, read as a little-endian integer, stores coordinate `t` of the group in bit `8G - 1 - t`. This is what `pack_binary_to_bytes<uint64_t>` writes (checked against it for `padded_dim` 32, 64, 96, 160 and 1024); the implementation adds a golden test that freezes it.

#### HNSW-RaBitQ records

Let `L = 4 * (base_max_degree + 1)` and `U = 4 * (upper_max_degree + 1)`.

| Offset | Size | Content |
| --- | --- | --- |
| 0 | 2 | Base-layer neighbor count, u16 |
| 2 | 2 | Reserved |
| 4 | `4 * base_max_degree` | Base-layer neighbor IDs; entries past the count are unspecified |
| `L` | 4 | Cluster ID, u32; bit 31 is the removal mark (see [HNSW removal](development.md#hnsw-removal)) |
| `L + 4` | 4 | Row ID of the point, u32 |
| `L + 8` | `padded_dim / 8` | [Sign code](#hnsw-sign-code) |
| `L + 8 + padded_dim / 8` | 12 | `add`, `rescale` and `error`, f32 |
| `L + 20 + padded_dim / 8` | | When `code_bits > 1`, a [single-code record](#single-code-record) with `b = code_bits - 1` |

A point at level `l > 0` has `l` upper-layer link lists in `hnsw_upper_links`, for levels 1 to `l`, each `U` bytes: a u16 count, 2 reserved bytes, then `upper_max_degree` neighbor IDs.

#### HNSW routing graph

The centroid graph of IVF-RaBitQ's HNSW routing is the layout hnswlib's `saveIndex` writes. Unlike the rest of format 2 it is not defined field by field by this library, so it is fixed here as written on the supported 64-bit little-endian platforms (`size_t` is 8 bytes, `int` and `unsigned` 4):

| Offset | Size | Type | Field | Rule |
| --- | --- | --- | --- | --- |
| 0 | 8 | u64 | `offset_level0` | 0 |
| 8 | 8 | u64 | `max_elements` | At least `count` |
| 16 | 8 | u64 | `count` | `num_clusters` |
| 24 | 8 | u64 | `record_bytes` | `label_offset + 8` |
| 32 | 8 | u64 | `label_offset` | `data_offset + 4 * padded_dim` |
| 40 | 8 | u64 | `data_offset` | `4 + 4 * max_m0` |
| 48 | 4 | i32 | `max_level` | The entry point's level, at least 0 |
| 52 | 4 | u32 | `entry_point` | Below `count` |
| 56 | 8 | u64 | `max_m` | `m` |
| 64 | 8 | u64 | `max_m0` | `2 * m` |
| 72 | 8 | u64 | `m` | 2 to 10,000 |
| 80 | 8 | f64 | `level_multiplier` | Finite and positive |
| 88 | 8 | u64 | `ef_construction` | At least `m` |

Then `count` base records of `record_bytes`: a u16 neighbor count of at most `max_m0`, a flags byte whose bit 0 (hnswlib's deletion mark) must be clear, a reserved byte, `max_m0` u32 neighbor IDs, the rotated centroid (`padded_dim` f32) and its label (u64). Then, per point in order, a u32 byte length followed by that many bytes of upper-layer link lists: `level` lists of `4 + 4 * max_m` bytes each, laid out as in the base record. Every neighbor ID is below `count` and has at least the level of the list that holds it.

#### SymphonyQG records

Every vertex has `max_degree` neighbor slots, a multiple of 32. Let `V` be the size of the vector part: `4 * dim` for raw vectors, or the [single-code record](#single-code-record) size with `b = code_bits` otherwise.

| Offset | Size | Content |
| --- | --- | --- |
| 0 | `V` | The vector: `dim` float32 values (`raw_type = float32`, `code_bits = 0`), or a single-code record |
| `V` | `(max_degree / 32) * (padded_dim * 4 + 256)` | The neighbors' 1-bit codes, quantized against this vertex, as [FastScan blocks](#fastscan-blocks-version-1) without the `error` factor: sign codes, then 32 `add` and 32 `rescale` values |
| `V + (max_degree / 32) * (padded_dim * 4 + 256)` | `4 * max_degree` | Neighbor IDs |

Unused lanes of a partial neighbor block have zero factors. Their IDs are left over from construction, but like every ID slot they are below `num_points`, which readers check.

### IVF-RaBitQ files

Besides the rules above, an IVF-RaBitQ reader requires:

- `num_points` from 1 to 2^31, and every stored ID below it;
- `code_bits` from 1 to 9, with `raw_type = float32` only when `code_bits = 1`, and `raw_type` none or float32;
- `id_bits = 32`;
- in `ivf_params`, `num_clusters` from 1 to 2^31, `initializer` 1 to 3, and the reserved field zero;
- cluster sizes that add up to `num_points`;
- a seed, when known, of at most 2^32 − 1.

### HNSW-RaBitQ files

An HNSW-RaBitQ reader requires `num_points` from 1 to 2^31, `code_bits` from 1 to 9, `raw_type` none, `id_bits = 32`, `dim` at least 64, and an FHT/Kac rotator. In `hnsw_params`: `max_elements` from `num_points` to 2^31; `num_clusters` from 1 to 2^31; `M` from 2 to 10,000, with `upper_max_degree = M` and `base_max_degree = 2 * M`; `ef_construction` at least `M`; a finite positive `level_multiplier`; `max_level` at least 0; `entry_point` below `num_points`; and the reserved field zero. Every level in `hnsw_levels` is at most `max_level`, and the entry point's level equals it. Every record's cluster ID (without the removal mark) is below `num_clusters`, its labels are distinct, its base-layer neighbor count is at most `base_max_degree` and each upper-layer count at most `M`, and every neighbor ID is below `num_points` with at least the level of the list that holds it.

### SymphonyQG files

A SymphonyQG reader requires `id_bits = 32` and either raw vectors (`code_bits = 0`, `raw_type = float32`) or `code_bits` 4 or 8 with `raw_type` none; a seed, when known, of at most 2^32 − 1; `num_points` up to 2^31; in `qg_params`, a `max_degree` that is a positive multiple of 32 and below `num_points`, `entry_point` below `num_points`, `init` 0 to 2 and the reserved field zero; `qg_centroid` exactly when the codes are quantized; and every neighbor ID below `num_points`.

### Companion files

Only IVF-SSD uses them. `<base>.ssd` holds the page-packed inverted lists, read with `O_DIRECT` and too large to keep in the main file. The `companions` section lists each one:

| Offset | Size | Type | Field |
| --- | --- | --- | --- |
| 0 | 4 | u32 | `role`: 1 = SSD inverted lists |
| 4 | 4 | u32 | reserved |
| 8 | 32 | ASCII | `suffix` appended to the main file's path, such as `".ssd"`, NUL-padded; no path separators |
| 40 | 8 | u64 | `size` in bytes |
| 48 | 8 | u64 | `fingerprint` |

`fingerprint` is the 64-bit FNV-1a hash that RaBitQ-SSD already uses (offset basis 1469598103934665603, prime 1099511628211) over the rotated centroids as f32 bytes. It is deterministic, so saving the same index twice gives identical files.

A companion starts with a 4,096-byte header page: magic `"RABQSSD\0"`, version, `fingerprint`, `num_points` and `code_bits`, then zeros, with a CRC-32 over the page. Data pages follow at offset 4,096, keeping the 4,096-byte alignment that `O_DIRECT` needs. Loading rejects a missing companion, or one whose size, fingerprint or header does not match the main file.

RaBitQ-SSD's swappable coarse quantizer is not a companion: the base does not pin one. A coarse-quantizer file is its own file and records the base's fingerprint, so the pairing is checked from the coarse side. Its layout is specified when IVF-SSD moves into the library.

### Reading

A reader checks, in order, and validates every size before allocating. The loader of each index knows the index kind it expects, so a file of another kind is reported as such before any section is checked:

1. The file is at least 128 bytes long, and the magic, `version` and `header_bytes` are valid.
2. The section table is non-empty and lies within the file, and `header_crc32` matches.
3. Feature flags, as in the table above.
4. `index_kind` is the expected one; the fields that every kind shares are valid; then the expecting index's own header rules hold.
5. Each table entry, in order: its flags and reserved field, alignment, bounds, order, overlap and repetition, then its kind. An unknown required kind rejects the file here.
6. The file ends where the table or the last section ends.
7. Each known checksummed section matches its `crc32`.
8. The index's required sections are present, and each section's size equals what the header and parameter sections imply, before its content is read.

Steps 1–7 belong to the container; step 8 and the index's header rules in step 4 belong to each index. Readers read through the stream buffer, so the stream's exception mask does not matter, and need seekable input.

A file that fails any of these checks, or a damaged file of an [earlier layout](#earlier-layouts), raises `rabitqlib::FormatError` in C++, a subclass of `std::runtime_error`, and `rabitqlib.FormatError` in Python, a subclass of both `ValueError` and `RuntimeError`. `rabitqlib.quant` raises the same class for blobs. A file that cannot be opened raises a plain `std::runtime_error` (Python `RuntimeError`). A read that fails after opening cannot be told from a truncated file, so it raises `FormatError`.

Index files remain trusted input (see [Index file compatibility](docs/index/usage.md#index-file-compatibility)). These checks report truncated, corrupt or mismatched files; they do not make crafted files safe.

### Writing

A writer writes the file once, front to back, so it does not need seekable output (`save` writes devices and FIFOs in place): it buffers the checksummed sections, which are small, to know their checksums before the table. `save` writes to a temporary file and renames it into place, as today. With the routing graph inside the main file, IVF-RaBitQ no longer has a second rename that can leave a new sidecar beside an old index. Resaving a file loaded from an earlier layout writes format 2. IVF-SSD companion files are written before the main file, so a failed save never leaves a main file that points to a missing companion.

### Earlier layouts

Readers dispatch on the leading bytes: `"RABQINDX"`, then each index's legacy magic (IVF `RABQIDX1`, SymphonyQG quantized), then the unversioned fallbacks the readers accept today. Legacy files have no cosine metric, seed or 64-bit IDs; they load as `seed_known = false` with 32-bit IDs, as they do in 0.8. The frozen fixtures in `tests/fixtures/persistence/` keep covering every layout.

### Compatibility rules

- A new release reads every earlier released index file.
- An older release reads a newer format-2 file or rejects it with an error naming the problem; it never misreads it. Older releases reject `version` values other than 2, unknown `index_kind`, `metric` or `id_bits` values, unknown required sections and unknown `incompat` bits.
- An additive change appends header fields (raising `header_bytes`), adds sections, or adds feature bits, chosen by the table in [Feature flags](#feature-flags). A change that an older reader cannot detect from these is not allowed; it needs `version = 3`.
- Every released change adds a frozen fixture. Synthetic "future" fixtures check the rules above: an unknown optional section loads; an unknown required section, an `incompat` bit or a larger `version` is rejected with its message; a `ro_compat` bit loads read-only; a longer header loads.

### Open questions

- Whether the header should be a [blob](#blobs) kind instead of its own magic. Proposed: its own magic, as above. A blob checksums its whole body, which is too costly for large index files, so index files get their own container and nest blobs in sections.
