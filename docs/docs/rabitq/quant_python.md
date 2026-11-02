# Python quantization building blocks

!!! warning "Provisional"
    `rabitqlib.quant` is provisional in the sense of [PEP 411](https://peps.python.org/pep-0411/): any minor release may change or remove part of its API, with a note in the release notes, until it is declared stable. Byte formats are versioned from the first release, and bytes of an unknown version raise `quant.FormatError` instead of giving wrong results.

`rabitqlib.quant` exposes the pieces the indexes are built from: rotation, quantization, correction factors, packed codes and distance estimation. Use it to store RaBitQ codes in your own database or file format and to score them with your own selection logic. For a ready-made index, use [IVF-RaBitQ](../index/ivf.md), [HNSW-RaBitQ](../index/hnsw.md) or [SymphonyQG](../index/qg.md) instead.

```python
from rabitqlib import quant
```

`import rabitqlib` does not import the module; import it explicitly.

## The working space

Every layer call works on rotated, padded vectors: float32 arrays of shape `(n, d)` with `d` a positive multiple of 32. `Rotator` maps original vectors into that space; callers may bring their own rotation instead. Rotate the data, the centroids and the queries with the same rotator.

```python
import numpy as np
from rabitqlib import quant

rng = np.random.default_rng(0)
data = rng.standard_normal((1000, 100)).astype(np.float32)
query = rng.standard_normal(100).astype(np.float32)

rot = quant.Rotator(100, seed=42)            # padded_dim = 128
x = rot.rotate(data)                          # float32[1000, 128]
q = rot.rotate(query[None])[0]
c = x.mean(axis=0)                            # a centroid in the working space
```

A seeded `"fht_kac"` rotation (the default, for 64 dimensions and up) is identical on every platform. A `"matrix"` rotation drawn from a seed is reproducible only with the same standard library and build, so persist rotators with `to_bytes()` and load them with `Rotator.from_bytes(data)`. Rotators also pickle.

## Encoding

```python
residuals = x - c
codes = quant.quantize(residuals, nbits=4)                      # uint8[1000, 128]
prefix, ex = quant.split(codes, nbits=4, prefix_bits=1)         # 1 sign bit + 3 extra bits
f_full = quant.code_factors(residuals, codes, nbits=4, metric="l2", centroid=c)
f_prefix = quant.code_factors(residuals, prefix, nbits=1, metric="l2", centroid=c)
bitsets = quant.pack_codes(prefix, nbits=1)                     # uint8[1000, 16]
packed_ex = quant.pack_codes(ex, nbits=3)                       # uint8[1000, 48]
```

- `nbits` is the total code width, 1–9, including the sign bit. Codes are unsigned integers in `[0, 2**nbits)`: `uint8` up to 8 bits, `uint16` at 9.
- `split` keeps the top `prefix_bits` bits as `prefix` and the low `ex_bits = nbits - prefix_bits` bits as `ex`. The prefix is itself a valid code of the same residual, so `code_factors` with `nbits=prefix_bits` gives its factors; `ex` has no factors of its own, because refinement pairs it with the full-code factors.
- `metric` is required on `code_factors`, `Query` and `query_terms`: factors for the wrong metric give wrong distances without an error. `centroid` is in the working space; `None` means zero.
- `quantize(..., fast_quantization=True)` uses the library's precomputed scale table instead of the exact search, as the indexes' `faster_config` does.
- `pack_codes` writes the layout of [compact code storage](../compact_code.md) for 1–8 bits; at `nbits=1` it equals `numpy.packbits(prefix, axis=1, bitorder="little")`. `unpack_codes` reverses it losslessly. Split 9-bit codes before packing.

Factors are structured arrays of `quant.FACTORS_DTYPE` (`add`, `rescale`, `error`), three float32 fields per record. Consumers that do not take structured dtypes can use `factors.view(np.float32).reshape(-1, 3)`.

## Estimation

```python
qry = quant.Query(q, metric="l2", centroid=c)
est = qry.estimate_bitset(bitsets, f_prefix)                    # ESTIMATE_DTYPE[1000]
lower = est["distance"] - est["error"]
upper = est["distance"] + est["error"]
candidates = np.flatnonzero(lower <= np.partition(upper, 9)[9])  # 10th-smallest upper bound
dist = qry.refine_packed(packed_ex, f_full, est[candidates], indices=candidates, ex_bits=3)
top = candidates[np.argsort(dist)[:10]]
```

`Query` copies the rotated query. `set_centroid(c)` switches to another list's factors without preparing the query again.

| Call | Reads | Returns |
| --- | --- | --- |
| `estimate_bitset(bitsets, factors)` | `pack_codes(prefix, nbits=1)` with the 1-bit factors, scored with the float query | `ESTIMATE_DTYPE` |
| `estimate_codes(codes, factors, nbits=…)` | Unpacked codes of any width with their factors | `ESTIMATE_DTYPE` |
| `refine(ex, f_full, first, ex_bits=…)` | Unpacked extra bits, the full-code factors and the first-stage estimates | `float32` |
| `refine_packed(packed, f_full, first, ex_bits=…)` | The same with `pack_codes(ex, nbits=ex_bits)` | `float32` |
| `estimate_fastscan(blocks, num_points=…)` | `pack_fastscan_blocks` output | `ESTIMATE_DTYPE` |

An estimate has three fields: `distance`, the `error` term, and `refine_input`, the prefix inner product that refinement reads. `distance ± m * error` is a probabilistic interval for the true distance. `error` already includes the encoder's confidence parameter ε₀ = 1.9, so `m = 1` is the interval the indexes use, and about 6% of true distances fall outside it; a larger `m` widens it, refining more candidates for fewer misses. To keep the top `k` within that confidence, refine every vector whose lower bound is at most the `k`-th smallest upper bound, as in the example above; comparing lower bounds with the `k`-th smallest estimate instead drops true neighbours. `refine` and `refine_packed` take `prefix_bits=` (default 1) for the width the first stage scored, and agree with each other within float32 rounding: the packed path uses the dispatched SIMD kernel. The scoring calls accept `indices=` to read just those rows in place; `first` and the output then line up with `indices`. Indices must be integers in `[0, n)`.

Distances follow the indexes. L2 is the estimated squared distance; IP is `1 - <q, x>`. Lower is better.

### FastScan blocks

```python
blocks = quant.pack_fastscan_blocks(bitsets, f_prefix)          # uint8[32, 128 * 4 + 384]
est_fs = qry.estimate_fastscan(blocks, num_points=len(bitsets))
```

A block always holds 32 vectors, the library's fixed FastScan batch; the last block is zero-padded. `estimate_fastscan` scores with a lookup table built from a quantized query on the first call, so its estimates agree with `estimate_bitset` only within that table's error. FastScan accumulation needs AVX2/FMA, AVX-512 or NEON, and raises `RuntimeError` on a CPU with none of them.

### The reference formula

Every estimate is an inner product followed by one formula. For an `nbits`-bit code with unsigned values `u`, factors `f` and the query terms of `query_terms`:

```text
ip       = <q, u>
distance = f.add + g_add + f.rescale * (ip + g_k1xsumq * (2^nbits - 1))
error    = f.error * g_error

g_k1xsumq = -0.5 * sum(q)
L2: g_add = |q - c|^2     IP: g_add = -<q, c>     both: g_error = |q - c|
```

For IP, the `1` of `1 - <q, x>` is folded into `f.add`, not into `g_add`. For a split code, `ip_full = 2^ex_bits * ip_prefix + ip_ex`, so refinement is the same formula with the full-code factors and `nbits = prefix_bits + ex_bits`. `quant.estimate_from_ip(ip, factors, terms, nbits=…)` applies the formula to inner products from your own kernel:

```python
terms = quant.query_terms(q, metric="l2", centroid=c)           # QUERY_TERMS_DTYPE record
ip = codes.astype(np.float32) @ q
est4 = quant.estimate_from_ip(ip, f_full, terms, nbits=4)
```

If your `ip` is itself approximate, `error` covers only the code's quantization error, not the error in `ip`.

## Threads, copies and pickling

- Build-side calls (`Rotator.rotate`, `quantize`, `code_factors`, `pack_codes`, `unpack_codes`, `pack_fastscan_blocks`) default to `num_threads=0`, every available CPU; their results do not depend on the thread count. Scoring calls (`estimate_*`, `refine*`, `estimate_from_ip`) default to 1.
- The GIL is released during computation. `Rotator` is immutable and safe to share. Scoring calls on one `Query` may run concurrently; a `set_centroid` during one raises `RuntimeError`. Use `qry.copy()` for one query per thread.
- `Query` supports `copy.copy`, `copy.deepcopy` and pickling of its query vector, metric and centroid; a copy never shares storage with the original. Queries have no `to_bytes`, because they are not a stored format.
- Every call that returns an array accepts `out=` with the exact dtype and shape. It must not share memory with any input, or the call raises `ValueError`. If a call raises, a caller's `out=` is left unchanged.
- Inputs are validated: wrong shapes, dtypes or values raise `ValueError`, and non-finite or oversized coordinates are rejected as the indexes reject them.

## Byte formats

The layouts are specified in [`docs/formats.md`](https://github.com/VectorDB-NTU/RaBitQ-Library/blob/main/docs/formats.md) in the repository. Raw arrays carry no header; record the version constant next to them:

| Constant | Covers |
| --- | --- |
| `quant.PACKED_CODES_VERSION` | `pack_codes` output, including the 1-bit bitset |
| `quant.FACTORS_VERSION` | `FACTORS_DTYPE`, `ESTIMATE_DTYPE` and `QUERY_TERMS_DTYPE` |
| `quant.FASTSCAN_BLOCKS_VERSION` | `pack_fastscan_blocks` output |

`Rotator.to_bytes()` writes a self-describing blob with a magic, a kind, a version and a CRC-32. A change to any layout is a new version, never an edit of an existing one, and golden files in the test suite freeze each version. While the module is provisional, a release may stop reading an old version; reading it then raises `FormatError`. A promise that every released version stays readable comes with stabilisation.

Byte layouts, rotator blobs and seeded FHT/Kac rotations are identical everywhere. Rotated values, codes, factors and distances are identical with the same wheel and CPU features, and otherwise agree within float32 rounding, because kernels are chosen at runtime by CPU.
