"""Core quantization building blocks: rotation, quantization, factors, packing and estimation.

This module is provisional in the sense of PEP 411: any minor release may change
or remove its API, with a changelog note, until it is declared stable.
"""

from typing import Any, Literal, overload

import numpy as np
import numpy.typing as npt

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

FACTORS_DTYPE: np.dtype[np.void]
ESTIMATE_DTYPE: np.dtype[np.void]
QUERY_TERMS_DTYPE: np.dtype[np.void]
PACKED_CODES_VERSION: int
FACTORS_VERSION: int
FASTSCAN_BLOCKS_VERSION: int

class FormatError(ValueError):
    """Corrupt or incompatible serialized bytes. A subclass of ValueError."""

class Rotator:
    """A seeded rotation from the original dimension to the padded working space."""

    def __init__(
        self,
        dim: int,
        *,
        method: Literal["fht_kac", "fht", "matrix"] = "fht_kac",
        padded_dim: int | None = None,
        seed: int | None = None,
    ) -> None: ...
    @property
    def dim(self) -> int:
        """Original vector dimension."""
    @property
    def padded_dim(self) -> int:
        """Working dimension of rotated vectors."""
    @property
    def method(self) -> Literal["fht_kac", "matrix"]:
        """Canonical method name, 'fht_kac' or 'matrix'."""
    @property
    def seed(self) -> int | None:
        """The seed the rotation was drawn from, or None if unknown."""
    def rotate(
        self,
        x: npt.ArrayLike,
        *,
        num_threads: int = 0,
        out: npt.NDArray[np.float32] | None = None,
    ) -> npt.NDArray[np.float32]:
        """Rotate vectors into the padded working space."""
    def to_bytes(self) -> bytes:
        """Serialize the rotation as a versioned rotator blob (docs/formats.md)."""
    @staticmethod
    def from_bytes(
        data: bytes | bytearray | memoryview | npt.NDArray[Any],
    ) -> Rotator:
        """Load a rotator blob written by to_bytes."""
    def __copy__(self) -> Rotator:
        """Return self; rotators are immutable."""
    def __deepcopy__(self, memo: Any) -> Rotator:
        """Return self; rotators are immutable."""

_Metric = Literal["l2", "ip", "innerproduct"]

class Query:
    """A rotated, padded query for scoring codes in the working space."""

    def __init__(
        self,
        rotated_query: npt.ArrayLike,
        *,
        metric: _Metric,
        centroid: npt.ArrayLike | None = None,
    ) -> None: ...
    @property
    def dim(self) -> int:
        """Working dimension d of the query."""
    @property
    def metric(self) -> Literal["l2", "ip"]:
        """The metric, 'l2' or 'ip'."""
    def set_centroid(self, centroid: npt.ArrayLike | None) -> None:
        """Switch to the factors of another centroid; None is zero."""
    def estimate_bitset(
        self,
        bitsets: npt.NDArray[np.uint8],
        factors: npt.NDArray[np.void],
        *,
        indices: npt.ArrayLike | None = None,
        num_threads: int = 1,
        out: npt.NDArray[np.void] | None = None,
    ) -> npt.NDArray[np.void]:
        """Estimate distances from packed 1-bit codes with the float query."""
    def estimate_codes(
        self,
        codes: npt.NDArray[np.uint8] | npt.NDArray[np.uint16],
        factors: npt.NDArray[np.void],
        *,
        nbits: int,
        indices: npt.ArrayLike | None = None,
        num_threads: int = 1,
        out: npt.NDArray[np.void] | None = None,
    ) -> npt.NDArray[np.void]:
        """Estimate distances from unpacked codes of any width."""
    def refine(
        self,
        ex: npt.NDArray[np.uint8],
        factors: npt.NDArray[np.void],
        first: npt.NDArray[np.void],
        *,
        ex_bits: int,
        prefix_bits: int = 1,
        indices: npt.ArrayLike | None = None,
        num_threads: int = 1,
        out: npt.NDArray[np.float32] | None = None,
    ) -> npt.NDArray[np.float32]:
        """Refine prefix estimates with the unpacked extra bits."""
    def refine_packed(
        self,
        packed: npt.NDArray[np.uint8],
        factors: npt.NDArray[np.void],
        first: npt.NDArray[np.void],
        *,
        ex_bits: int,
        prefix_bits: int = 1,
        indices: npt.ArrayLike | None = None,
        num_threads: int = 1,
        out: npt.NDArray[np.float32] | None = None,
    ) -> npt.NDArray[np.float32]:
        """Refine prefix estimates with packed extra bits."""
    def estimate_fastscan(
        self,
        blocks: npt.NDArray[np.uint8],
        *,
        num_points: int,
        num_threads: int = 1,
        out: npt.NDArray[np.void] | None = None,
    ) -> npt.NDArray[np.void]:
        """Estimate distances for FastScan blocks with a 4-bit query table."""
    def copy(self) -> Query:
        """Return an independent copy of the query."""
    def __copy__(self) -> Query:
        """Return an independent copy of the query."""
    def __deepcopy__(self, memo: Any) -> Query:
        """Return an independent copy of the query."""

def padded_dim(dim: int) -> int:
    """Return the working dimension for dim: dim rounded up to a multiple of 32."""

@overload
def quantize(
    residuals: npt.ArrayLike,
    *,
    nbits: Literal[1, 2, 3, 4, 5, 6, 7, 8],
    fast_quantization: bool = False,
    num_threads: int = 0,
    out: npt.NDArray[np.uint8] | None = None,
) -> npt.NDArray[np.uint8]:
    """Quantize rotated residuals into unsigned nbits-bit codes."""

@overload
def quantize(
    residuals: npt.ArrayLike,
    *,
    nbits: Literal[9],
    fast_quantization: bool = False,
    num_threads: int = 0,
    out: npt.NDArray[np.uint16] | None = None,
) -> npt.NDArray[np.uint16]:
    """Quantize rotated residuals into unsigned nbits-bit codes."""

@overload
def quantize(
    residuals: npt.ArrayLike,
    *,
    nbits: int,
    fast_quantization: bool = False,
    num_threads: int = 0,
    out: npt.NDArray[np.uint8] | npt.NDArray[np.uint16] | None = None,
) -> npt.NDArray[np.uint8] | npt.NDArray[np.uint16]:
    """Quantize rotated residuals into unsigned nbits-bit codes."""

def split(
    codes: npt.NDArray[np.uint8] | npt.NDArray[np.uint16],
    *,
    nbits: int,
    prefix_bits: int,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Split nbits-bit codes into a prefix and extra bits."""

def code_factors(
    residuals: npt.ArrayLike,
    codes: npt.NDArray[np.uint8] | npt.NDArray[np.uint16],
    *,
    nbits: int,
    metric: _Metric,
    centroid: npt.ArrayLike | None = None,
    num_threads: int = 0,
    out: npt.NDArray[np.void] | None = None,
) -> npt.NDArray[np.void]:
    """Return the factors of nbits-bit codes for their residuals."""

def pack_codes(
    codes: npt.NDArray[np.uint8],
    *,
    nbits: int,
    num_threads: int = 0,
    out: npt.NDArray[np.uint8] | None = None,
) -> npt.NDArray[np.uint8]:
    """Pack nbits-bit codes into the packed-codes format (version 1)."""

def unpack_codes(
    packed: npt.NDArray[np.uint8],
    *,
    nbits: int,
    num_threads: int = 0,
    out: npt.NDArray[np.uint8] | None = None,
) -> npt.NDArray[np.uint8]:
    """Unpack codes written by pack_codes; the inverse is lossless."""

def pack_fastscan_blocks(
    bitsets: npt.NDArray[np.uint8],
    factors: npt.NDArray[np.void],
    *,
    num_threads: int = 0,
    out: npt.NDArray[np.uint8] | None = None,
) -> npt.NDArray[np.uint8]:
    """Pack bitsets and their factors into FastScan blocks (version 1)."""

def query_terms(
    rotated_query: npt.ArrayLike,
    *,
    metric: _Metric,
    centroid: npt.ArrayLike | None = None,
) -> np.void:
    """Return the query-side terms of the reference estimator formula."""

def estimate_from_ip(
    ip: npt.ArrayLike,
    factors: npt.NDArray[np.void],
    terms: np.void | npt.NDArray[np.void],
    *,
    nbits: int,
    num_threads: int = 1,
    out: npt.NDArray[np.void] | None = None,
) -> npt.NDArray[np.void]:
    """Apply the reference estimator formula to the caller's inner products."""
