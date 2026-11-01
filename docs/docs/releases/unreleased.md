# Unreleased

## Python changes

- New provisional module `rabitqlib.quant` with the quantization building blocks behind the indexes: `Rotator` (seeded FHT/Kac or matrix rotation, persisted with `to_bytes`/`from_bytes`), `quantize`, `split`, `code_factors`, `pack_codes`/`unpack_codes`, the `Query` estimators and refinement, `query_terms`/`estimate_from_ip`, and FastScan blocks with `pack_fastscan_blocks`/`Query.estimate_fastscan`. It is provisional in the sense of PEP 411: any minor release may change it, with a note here. Import it explicitly with `from rabitqlib import quant`. See [Python building blocks](../rabitq/quant_python.md).
- Its byte formats (packed codes, factor records, FastScan blocks and rotator blobs) are versioned from the start and frozen by golden files; bytes of an unknown version raise `quant.FormatError`, a subclass of `ValueError`.

## C++ changes

- New provisional header `rabitqlib/core/provisional/quantize.hpp` with `rabitqlib::quant::provisional::combined_code` and `code_factors`, which back `rabitqlib.quant`. It is not part of the stable v1 API and may change in any minor release.
