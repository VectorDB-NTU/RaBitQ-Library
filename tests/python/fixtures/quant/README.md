These files freeze the `rabitqlib.quant` byte formats described in [docs/formats.md](../../../../docs/formats.md). They were written once by the code that introduced each format and are never regenerated: a test failure against them means a format changed, which needs a new format version rather than new fixtures.

`rotator_fht_kac_dim64_seed0.bin` and `rotator_fht_kac_dim100_padded128_seed42.bin` are rotator blobs (kind 1, version 1) from `quant.Rotator(64, seed=0).to_bytes()` and `quant.Rotator(100, padded_dim=128, seed=42).to_bytes()`. The first body equals the frozen FHT/Kac sign flips for `(64, 64, seed 0)` in `tests/unit/rabitqlib/utils/rotator_test.cpp` (FNV-1a hash `0x7fcf0925975a6086`).

`packed_codes_nbits{1..8}_n3_d96.bin` are `quant.pack_codes(codes, nbits=nbits).tobytes()` (packed-codes version 1) for three rows of 96 coordinates, which covers one full 64-coordinate block and a 32-coordinate tail, with

```python
i, j = np.meshgrid(np.arange(3), np.arange(96), indexing="ij")
codes = ((i * 37 + j * 11 + (j // 32) * 5) % (1 << nbits)).astype(np.uint8)
```

`fastscan_blocks_n33_d64.bin` is `quant.pack_fastscan_blocks(bitsets, factors).tobytes()` (FastScan blocks version 1): two blocks, the second with one vector and 31 zero-padded lanes, with

```python
i, byte = np.arange(33)[:, None], np.arange(8)[None, :]
bitsets = ((i * 37 + byte * 11 + 3) % 256).astype(np.uint8)
factors = np.zeros(33, quant.FACTORS_DTYPE)
factors["add"] = np.arange(33) * 0.5
factors["rescale"] = -(np.arange(33) + 1) * 0.25
factors["error"] = np.arange(33) / 8
```

It was checked against an independent NumPy implementation of the layout in `docs/formats.md` before it was frozen.
