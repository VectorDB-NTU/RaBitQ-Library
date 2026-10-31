# Index usage contracts

Shared input, count, concurrency and persistence conventions for IVF-RaBitQ, HNSW-RaBitQ and
SymphonyQG. Start with the [Python quick start](../quick_start.md) for a complete example.

## Input requirements

`IvfIndex`, `HnswIndex`, and `SymqgIndex` accept data and query arrays with shape
`(count, dim)`, converted to contiguous float32. Their FHT/Kac rotation requires
`64 <= dim <= 65536`. Supply the original dimension; the indexes pad internally
to a multiple of 32. Queries and centroids must have the same original dimension
as the indexed data.

Arrays whose data pointers are not aligned for float32 are copied into aligned
storage before native processing; inputs remain unchanged.

Data, centroids and queries must be finite. Coordinates are also limited to the
magnitude the clustering APIs accept, `sqrt(sqrt(FLT_MAX) / (64 * dim))` (about
`5e7` for `dim = 128`), so that squared distances and their products stay inside
float32. `build`, `add` and `search` raise `ValueError` (C++:
`std::invalid_argument`) for NaN, infinite, or larger coordinates before changing
the index. The C++ `quantize_*` encoders in `core/encoder.hpp` assume their
callers have validated the input.

These limits apply to the Python indexes. The C++ [stable core API](../rabitq/core_api.md)
accepts working dimensions from 2, with multiples of 32 required for `SplitPacked`.
C++ [MatrixRotator](../rabitq/rotator.md) has separate dimension and storage requirements;
the Python indexes do not expose a rotator choice.

## Index counts

The unified count properties below are introduced in **0.7.0**. In particular,
HNSW-RaBitQ's `num_points` now excludes removed records; see the
[count migration guide](../modular_migration.md#unified-index-counts) when upgrading.

IVF-RaBitQ, HNSW-RaBitQ, and SymphonyQG expose these read-only Python properties
(C++ uses the same names with `()`):

| Property | Meaning |
| --- | --- |
| `max_elements` | Logical record capacity, excluding SIMD padding |
| `num_points` | Nonremoved points |
| `num_stored_points` | Stored records, including removed ones |

`0 <= num_points <= num_stored_points <= max_elements`. Before building, both
point counts are zero; configured capacity may be nonzero.

At build time, IVF-RaBitQ requires exactly `max_elements` input rows. HNSW-RaBitQ accepts
1 through `max_elements` rows, so it can reserve capacity for later additions.

Removal decreases only `num_points`; slots and IDs remain occupied. Append IDs
start at the previous `num_stored_points`. Use this stored count for ID bounds,
and `num_points` for a live count. Counts survive save/load.

IVF-RaBitQ grows automatically and has no spare logical capacity after build or append.
HNSW-RaBitQ's `resize` changes only capacity. SymphonyQG has no incremental updates;
all three counts are equal after build.

`repr(index)` shows dimension, metric, quantization bits, live point count and build
state. IVF/HNSW also show stored count, capacity and centroid count; IVF includes
the resolved initializer, and SymphonyQG includes its degree bound. The summary
reads only metadata and follows the same concurrency rules as property reads.

## Estimate storage

The following estimates describe the main float32 index storage, not total process
memory or peak construction memory. Let `d` be the original dimension, `D` the
stored padded dimension, and `B` the total quantized bits per dimension. New
indexes use `D = 32 * ceil(d / 32)`; loaded indexes retain their saved padding.

For IVF-RaBitQ/HNSW-RaBitQ, codes and factors occupy `V(B)` bytes per vector:

- One bit: `V(1) = D / 8 + 12` (three float32 factors).
- Two through nine bits: `V(B) = D * B / 8 + 20` (three binary and two extra-code factors).

| Index / mode | Main bytes per stored point or capacity slot |
| --- | --- |
| IVF-RaBitQ quantized | Approximately `V(B) + 4`, including the point ID |
| IVF-RaBitQ raw (`nbits=32`) | Approximately `D / 8 + 12 + 4*d + 4`, including filtering codes, raw vector and ID |
| HNSW-RaBitQ | `V(B) + 8*M + 12` per **capacity slot** for base-layer records: up to `2*M` neighbor IDs, a link count, cluster ID and label |
| SymphonyQG raw | `4*d + R*(D/8 + 12)`, where `R = max_degree` |
| SymphonyQG quantized | `D*B/8 + 8 + R*(D/8 + 12)`, with `B` equal to 4 or 8 |

IVF-RaBitQ allocates one-bit codes and their three factors in full batches of 32
**per cluster**. For cluster sizes `n_c`, that part costs
`sum(32 * ceil(n_c / 32)) * (D/8 + 12)` bytes. Extra codes/raw vectors and IDs
use the stored point count. Small clusters therefore increase padding overhead.
Centroids, routing structures and cluster metadata are additional.

HNSW-RaBitQ allocates base-layer records for `max_elements`, including unused capacity.
Upper-layer edges, label maps, locks, visited lists and other bookkeeping are
additional. Removing points does not reclaim their storage in either IVF-RaBitQ or HNSW-RaBitQ.

SymphonyQG stores a one-bit code and two factors **for each neighbor in each row**,
plus a four-byte neighbor ID. This accounts for `R*(D/8 + 12)` bytes per row,
even in quantized mode: reducing the vector's own code width does not shrink
these neighborhood copies.

For example, with `d=D=128`, four-bit storage, `M=16` and `R=32`, the terms above
are about 88 bytes per IVF-RaBitQ point before batch padding, 224 bytes per HNSW-RaBitQ capacity
slot, and 968 bytes per SymphonyQG row. These are layout calculations, not measured
RSS. All indexes also retain rotation state; construction, query scratch,
allocator overhead and update workspaces add to memory use. Measure process
memory on the intended workload when sizing a deployment.

These formulas follow `BatchDataMap` / `QGBatchDataMap` in
`fastscan/data_layout.hpp`, `BinDataMap` / `ExDataMap` in `core/data_layout.hpp`,
and the index record layouts described in the individual guides. Applications
allocating code buffers should call the corresponding `data_bytes()` helpers
rather than hardcode these estimates.

## Threading and file paths

The library releases the Python GIL during native index search,
construction, updates, and file I/O. The following contract applies to
`IvfIndex`, `HnswIndex`, and `SymqgIndex`:

| Operation on the same index | Concurrent access |
| --- | --- |
| `search`, `search_batch`, `save`, `repr`, and property reads | May run together |
| `build`, `add`, `remove`, `reserve`, and `resize`, where available | Require exclusive access |

A conflicting call immediately raises
`RuntimeError("Index is busy: conflicting operation in progress")`.
It does not wait or queue an update. Access protection includes Python input
conversion and is released when the operation returns or raises an exception.
Different index objects can operate independently. Concurrent saves must use
different output paths, including any sidecar files.

Search parameters such as `ef`, `nprobe`, precision, and `num_threads` belong to
each call. When using a Python thread pool, consider `num_threads=1` to avoid
starting multiple native worker pools. Keep input arrays and any shared backing
storage unchanged for the entire call; do not resize or modify them from another
thread. Returned result arrays own their storage.

These access checks belong to the Python wrappers. C++ callers must synchronize
index updates themselves and use separate search output buffers.

`num_threads=0` selects the detected available logical CPU count. Positive values
set an upper limit, capped at that count; small workloads may use fewer workers.
Python index methods default to one thread; clustering defaults to `0`.
On Linux, CPU detection accounts for affinity and OpenMP binding. Apply binding
before starting the program.

C++ paths use UTF-8 on Windows and native path bytes on POSIX.
Python `save`/`load` accept strings, `pathlib.Path` and `os.PathLike` objects;
existing `bytes` and `bytearray` paths remain supported. POSIX paths preserve native
bytes, but non-UTF-8 filenames require filesystem support. Windows byte paths use
UTF-8. Embedded null characters are rejected before file access. IVF HNSW sidecars
use the same path conversion.

`save` writes a temporary file beside the destination and renames it into place, so a
failed save leaves an existing index unchanged. A replaced file keeps its permission
bits, but not its owner or other hard links. Symbolic links and other non-regular
destinations are written in place. IVF renames its HNSW sidecar before the index file;
a failure between the two renames can leave a new sidecar beside the old index.

## Index file compatibility

The compatibility suite loads frozen files written by v0.5.2, compares search
IDs and distances, and verifies save/load migration and rejection of truncated
or unsupported versions. See the
[fixture provenance and coverage](https://github.com/VectorDB-NTU/RaBitQ-Library/tree/main/tests/fixtures/persistence).
Compatibility is format-specific; preserving old readers' ability to open new
files is not guaranteed. Consult each index's format documentation before
downgrading the library.
