"""Core quantization building blocks: rotation, quantization, factors, packing and estimation.

This module is provisional in the sense of PEP 411: any minor release may change
or remove its API, with a changelog note, until it is declared stable. Byte
formats are versioned; bytes of an unknown version raise ``FormatError``.

The layer calls work on rotated, padded vectors: arrays of shape ``(n, d)``
where ``d`` is a positive multiple of 32. ``Rotator`` maps original vectors
into that space, or callers bring their own rotation.

Examples
--------
>>> import numpy as np
>>> from rabitqlib import quant
>>> rot = quant.Rotator(128, seed=0)
>>> residuals = rot.rotate(np.ones((4, 128), dtype=np.float32))
>>> codes = quant.quantize(residuals, nbits=4)
>>> factors = quant.code_factors(residuals, codes, nbits=4, metric="l2")
>>> quant.pack_codes(codes, nbits=4).shape
(4, 64)
>>> q = quant.Query(residuals[0], metric="l2")
>>> q.estimate_codes(codes, factors, nbits=4).dtype == quant.ESTIMATE_DTYPE
True
"""

from ._rabitqlib._quant import (
    ESTIMATE_DTYPE,
    FACTORS_DTYPE,
    QUERY_TERMS_DTYPE,
    FormatError,
    Query,
    Rotator,
    code_factors,
    estimate_from_ip,
    pack_codes,
    pack_fastscan_blocks,
    padded_dim,
    quantize,
    query_terms,
    split,
    unpack_codes,
)

PACKED_CODES_VERSION = 1
"""Version of the pack_codes layout, including the 1-bit sign bitset."""

FACTORS_VERSION = 1
"""Version of FACTORS_DTYPE, ESTIMATE_DTYPE and QUERY_TERMS_DTYPE."""

FASTSCAN_BLOCKS_VERSION = 1
"""Version of the pack_fastscan_blocks layout."""

# Classes report this module, so pickle and help() find them here. FormatError is
# rabitqlib.FormatError, shared with index loading.
for _public in (Query, Rotator):
    _public.__module__ = __name__
del _public

__all__ = [
    "ESTIMATE_DTYPE",
    "FACTORS_DTYPE",
    "FACTORS_VERSION",
    "FASTSCAN_BLOCKS_VERSION",
    "PACKED_CODES_VERSION",
    "QUERY_TERMS_DTYPE",
    "FormatError",
    "Query",
    "Rotator",
    "code_factors",
    "estimate_from_ip",
    "pack_codes",
    "pack_fastscan_blocks",
    "padded_dim",
    "quantize",
    "query_terms",
    "split",
    "unpack_codes",
]
