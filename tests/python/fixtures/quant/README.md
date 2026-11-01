These files freeze the `rabitqlib.quant` byte formats described in [docs/formats.md](../../../../docs/formats.md). They were written once by the code that introduced each format and are never regenerated: a test failure against them means a format changed, which needs a new format version rather than new fixtures.

`rotator_fht_kac_dim64_seed0.bin` and `rotator_fht_kac_dim100_padded128_seed42.bin` are rotator blobs (kind 1, version 1) from `quant.Rotator(64, seed=0).to_bytes()` and `quant.Rotator(100, padded_dim=128, seed=42).to_bytes()`. The first body equals the frozen FHT/Kac sign flips for `(64, 64, seed 0)` in `tests/unit/rabitqlib/utils/rotator_test.cpp` (FNV-1a hash `0x7fcf0925975a6086`).

`packed_codes_nbits{1..8}_n3_d96.bin` are `quant.pack_codes(codes, nbits=nbits).tobytes()` (packed-codes version 1) for three rows of 96 coordinates, which covers one full 64-coordinate block and a 32-coordinate tail, with

```python
i, j = np.meshgrid(np.arange(3), np.arange(96), indexing="ij")
codes = ((i * 37 + j * 11 + (j // 32) * 5) % (1 << nbits)).astype(np.uint8)
```
