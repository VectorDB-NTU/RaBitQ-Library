"""Public Python interfaces for RaBitQ indexes and clustering.

Array shapes and numeric ranges are documented runtime constraints.
"""

from collections.abc import Sequence
from os import PathLike
from typing import Any, ClassVar, Literal, SupportsIndex, TypeAlias, overload

import numpy as np
import numpy.typing as npt
from typing_extensions import deprecated

__all__ = [
    "FinalAssignmentMode",
    "HnswIndex",
    "IvfIndex",
    "QGKMeans",
    "QGKMeansIterationStats",
    "QGKMeansParameters",
    "RaBitQKMeans",
    "RaBitQKMeansIterationStats",
    "RaBitQKMeansParameters",
    "SymqgIndex",
]

_Path: TypeAlias = str | bytes | bytearray | PathLike[str] | PathLike[bytes]

_Metric: TypeAlias = Literal["l2", "ip", "innerproduct"]
_Initializer: TypeAlias = Literal["auto", "flat", "flat_rabitq", "hnsw"]
# Rank, ID ranges and Python bool (an int subtype) are validated at runtime.
_IntegerIds: TypeAlias = npt.NDArray[np.integer[Any]] | Sequence[int | np.integer[Any]]

# pybind11 enums are not subclasses of enum.Enum.
class FinalAssignmentMode:
    """Final centroid assignment policy.

    Approximate uses the selected clustering algorithm. SymphonyQG is a
    compatibility alias for Approximate. Exact performs exhaustive assignment.
    """

    Approximate: ClassVar[FinalAssignmentMode]
    SymphonyQG: ClassVar[FinalAssignmentMode]
    Exact: ClassVar[FinalAssignmentMode]
    __members__: ClassVar[dict[str, FinalAssignmentMode]]

    def __init__(self, value: SupportsIndex) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...
    def __int__(self) -> int: ...
    def __index__(self) -> int: ...

class QGKMeansIterationStats:
    """Read-only statistics for one completed clustering iteration."""

    @property
    def iteration(self) -> int:
        """Iteration number, starting at 1."""
    @property
    def obj(self) -> float:
        """Objective measured before the centroid update."""
    @property
    def shift(self) -> float:
        """Total squared centroid displacement during the update."""
    @property
    def nsplit(self) -> int:
        """Number of empty centroids repaired by splitting."""

RaBitQKMeansIterationStats = QGKMeansIterationStats

class IvfIndex:
    """An approximate nearest-neighbor index using IVF.

    Parameters
    ----------
    dim : int
        Original vector dimension, from 64 through 65536.
    num_clusters : int
        Number of centroids, from train or set_centroids.
    nbits : int
        Keyword-only. Total quantization bits, 1 through 9, or 32 for owned
        float32 vectors with one-bit filtering and raw reranking.
    metric : {'l2', 'ip', 'innerproduct'}, optional
        Squared L2 or inner-product distance. Default is 'l2'. Vectors are not
        automatically normalized for cosine similarity.
    initializer : {'auto', 'flat', 'flat_rabitq', 'hnsw'}, optional
        Centroid routing strategy. Default is 'auto'.
    seed : int, optional
        Rotation seed in [0, 2**32 - 1], also the initial cp.seed. None
        (default) draws one at random; the seed property reports it.

    Notes
    -----
    Train the index or set its centroids, then add vectors before searching or
    saving; loaded indexes are already trained. Concurrent reads are supported;
    an overlapping mutation raises RuntimeError.
    Data, centroids and queries must be finite float32 values within the index
    coordinate range; other values raise ValueError.
    The 0.7 form IvfIndex(dim, max_elements, num_clusters, nbits, ...) and
    build() still work in 0.8, raise DeprecationWarning, and will be removed in
    0.9.

    Examples
    --------
    >>> import numpy as np
    >>> from rabitqlib import IvfIndex
    >>> x = np.random.default_rng(42).normal(size=(80, 64)).astype(np.float32)
    >>> index = IvfIndex(64, 2, nbits=4, seed=7)
    >>> index.train(x)
    >>> ids = index.add(x)
    >>> ids, distances = index.search(x[:2], k=3, nprobe=2)
    >>> ids.shape, distances.shape
    ((2, 3), (2, 3))
    """

    @overload
    def __init__(
        self,
        dim: SupportsIndex,
        num_clusters: SupportsIndex,
        *,
        nbits: SupportsIndex,
        metric: _Metric = "l2",
        initializer: _Initializer = "auto",
        seed: SupportsIndex | None = None,
    ) -> None:
        """An approximate nearest-neighbor index using IVF.

        Parameters
        ----------
        dim : int
            Original vector dimension, from 64 through 65536.
        num_clusters : int
            Number of centroids, from train or set_centroids.
        nbits : int
            Keyword-only. Total quantization bits, 1 through 9, or 32 for owned
            float32 vectors with one-bit filtering and raw reranking.
        metric : {'l2', 'ip', 'innerproduct'}, optional
            Squared L2 or inner-product distance. Default is 'l2'. Vectors are not
            automatically normalized for cosine similarity.
        initializer : {'auto', 'flat', 'flat_rabitq', 'hnsw'}, optional
            Centroid routing strategy. Default is 'auto'.
        seed : int, optional
            Rotation seed in [0, 2**32 - 1], also the initial cp.seed. None
            (default) draws one at random; the seed property reports it.

        Notes
        -----
        Train the index or set its centroids, then add vectors before searching or
        saving; loaded indexes are already trained. Concurrent reads are supported;
        an overlapping mutation raises RuntimeError.
        Data, centroids and queries must be finite float32 values within the index
        coordinate range; other values raise ValueError.
        The 0.7 form IvfIndex(dim, max_elements, num_clusters, nbits, ...) and
        build() still work in 0.8, raise DeprecationWarning, and will be removed in
        0.9.
        """
    @overload
    @deprecated(
        "IvfIndex(dim, max_elements, num_clusters, nbits, ...) is deprecated since "
        "0.8; use IvfIndex(dim, num_clusters, nbits=...)"
    )
    def __init__(
        self,
        dim: SupportsIndex,
        max_elements: SupportsIndex,
        num_clusters: SupportsIndex,
        nbits: SupportsIndex,
        metric: _Metric = "l2",
        initializer: _Initializer = "auto",
    ) -> None: ...
    def __repr__(self) -> str:
        """Return a one-line summary of configuration, counts and training state."""

    def __len__(self) -> int:
        """Return the number of live, searchable points."""

    @property
    def dim(self) -> int:
        """Original vector dimension."""

    @property
    def max_elements(self) -> int:
        """Logical capacity: stored points, or the build row count of a 0.7-form index before build."""

    @property
    def num_points(self) -> int:
        """Number of live, searchable points."""

    @property
    def num_stored_points(self) -> int:
        """Number of stored points, including removed points."""

    @property
    def is_trained(self) -> bool:
        """Whether centroids are set, by train, set_centroids, build or load."""

    @property
    @deprecated("IvfIndex.is_built is deprecated since 0.8; use is_trained or len()")
    def is_built(self) -> bool:
        """Deprecated since 0.8: whether the index stores points; use len()."""

    @property
    def seed(self) -> int | None:
        """Rotation seed, or None for an index loaded from disk or created in the 0.7 form."""

    @property
    def cp(self) -> RaBitQKMeansParameters:
        """Clustering parameters train uses, a RaBitQKMeansParameters. Defaults: niter 25, max_points_per_centroid 256, spherical for inner product, and seed equal to the index seed."""

    @cp.setter
    def cp(self, value: RaBitQKMeansParameters) -> None: ...
    @property
    def metric(self) -> Literal["l2", "ip"]:
        """Canonical metric name: 'l2' or 'ip'."""

    @property
    def nbits(self) -> int:
        """Total storage bits per coordinate."""

    @property
    def num_clusters(self) -> int:
        """Number of centroids used for quantization."""

    def train(
        self, x: npt.ArrayLike, *, num_threads: SupportsIndex | None = None
    ) -> None:
        """Train num_clusters centroids on vectors and set them.

        Parameters
        ----------
        x : array_like of float, shape (n, dim)
            Training vectors in the original, unpadded coordinates, with
            n >= num_clusters. Converted to contiguous float32; a copy may be made.
        num_threads : int, optional
            Keyword-only. Worker count for this call. None (default) uses
            cp.num_threads; 0 selects the available CPU count.

        Notes
        -----
        Clusters with the settings in cp: RaBitQKMeans up to 2000 clusters, and
        QGKMeans with its default graph settings above that (a provisional,
        unmeasured threshold). Only centroids are trained; add assigns vectors.
        Retraining replaces the centroids while the index stores no vectors, and
        raises RuntimeError afterwards. A failed call leaves the index unchanged.
        """

    def set_centroids(self, centroids: npt.ArrayLike) -> None:
        """Use the given centroids instead of training.

        Parameters
        ----------
        centroids : array_like of float, shape (num_clusters, dim)
            Centroids in the original, unpadded coordinates. Converted to
            contiguous float32; a copy may be made.

        Notes
        -----
        The index counts as trained afterwards. Replacing centroids is allowed
        while the index stores no vectors and raises RuntimeError afterwards. A
        failed call leaves the index unchanged.
        """

    @deprecated(
        "IvfIndex.build() is deprecated since 0.8; use set_centroids() or "
        "train(), then add()"
    )
    def build(
        self,
        data: npt.ArrayLike,
        centroids: npt.ArrayLike,
        cluster_ids: _IntegerIds,
        num_threads: SupportsIndex = 1,
        fast_quantization: bool | np.bool_ = False,
    ) -> None:
        """Build the index from vectors and supplied centroid assignments.

        Deprecated since 0.8 and removed in 0.9: use set_centroids() (or
        train()) and then add(). The defaults match 0.7, including a single
        thread.

        Parameters
        ----------
        data : array_like of float, shape (n, dim)
            Vectors in the original, unpadded coordinates. Converted to
            contiguous float32; a copy may be made.
        centroids : array_like of float, shape (num_clusters, dim)
            Vectors in the original, unpadded coordinates. Converted to
            contiguous float32; a copy may be made.
        cluster_ids : array_like of int, shape (n,)
            Centroid ID per vector, in [0, num_clusters). Floating-point and
            boolean IDs are rejected.
        num_threads : int, optional
            Worker count. Default is 1; 0 selects the available CPU count.
        fast_quantization : bool, optional
            Use faster quantizer configuration. Default is False.
        """

    @overload
    def add(
        self,
        data: npt.ArrayLike,
        *,
        cluster_ids: _IntegerIds | None = None,
        num_threads: SupportsIndex = 0,
        fast_quantization: bool | np.bool_ = False,
    ) -> npt.NDArray[np.uint32]:
        """Add vectors using the trained centroids.

        Parameters
        ----------
        data : array_like of float, shape (n, dim)
            Vectors in the original, unpadded coordinates. Converted to contiguous
            float32; a copy may be made.
        cluster_ids : array_like of int, shape (n,), optional
            Keyword-only. Existing centroid IDs. None routes each vector using the
            index's configured centroid initializer; routing may be approximate.
            Centroids do not move.
        num_threads : int, optional
            Keyword-only. Worker count. Default is 0, which selects the available
            CPU count.
        fast_quantization : bool, optional
            Keyword-only. Use faster quantizer configuration. Default is False.

        Returns
        -------
        ids : ndarray of uint32, shape (n,)
            Assigned point IDs.

        Notes
        -----
        Requires a trained or loaded index. Batch additions to reduce repeated
        cluster copying; reserve can preallocate room. Passing the options
        positionally is deprecated.
        """
    @overload
    @deprecated(
        "Passing options to IvfIndex.add() by position is deprecated since 0.8; "
        "pass them by keyword"
    )
    def add(
        self,
        data: npt.ArrayLike,
        cluster_ids: _IntegerIds | None,
        num_threads: SupportsIndex = ...,
        fast_quantization: bool | np.bool_ = ...,
    ) -> npt.NDArray[np.uint32]: ...
    def remove(self, ids: _IntegerIds) -> int:
        """Exclude point IDs from subsequent search results.

        Parameters
        ----------
        ids : array_like of int, shape (n,)
            IDs in [0, num_stored_points). Floating-point and boolean IDs are
            rejected.

        Returns
        -------
        count : int
            Number of newly removed points; repeats do not count twice.

        Notes
        -----
        Removal does not reclaim storage and cannot be undone.
        """

    def search(
        self,
        queries: npt.ArrayLike,
        k: SupportsIndex,
        nprobe: SupportsIndex,
        high_accuracy: bool | np.bool_ | None = None,
        num_threads: SupportsIndex = 1,
    ) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.float32]]:
        """Search a batch of query vectors.

        Parameters
        ----------
        queries : array_like of float, shape (n_queries, dim)
            Vectors in the original, unpadded coordinates. Converted to contiguous
            float32; a copy may be made.
        k : int
            Number of neighbors requested, between 1 and max_elements.
        nprobe : int
            Positive number of clusters to probe; capped at num_clusters.
        high_accuracy : bool, optional
            Enable HACC. None (default) selects it for 4-9 bits and standard
            FastScan otherwise. Raw storage uses standard FastScan.
        num_threads : int, optional
            Worker count. Default is 1; 0 selects the available CPU count.

        Returns
        -------
        ids : ndarray of uint32, shape (n_queries, k)
            Point IDs of the selected neighbors.
        distances : ndarray of float32, shape (n_queries, k)
            Lower values are better; squared L2 estimates for l2 and inner-product
            distance estimates for ip. These are distances, not similarity scores.

        Notes
        -----
        Requires a trained or loaded index. Returns (ids, distances), in that
        order. Missing neighbors have ID 4294967295 and distance inf. Array shapes
        and value ranges are checked at runtime, not by static type checkers.
        """

    def save(self, path: _Path) -> None:
        """Save an index with stored vectors to disk.

        Parameters
        ----------
        path : str, bytes, bytearray, or os.PathLike
            Index file path. PathLike objects are resolved via __fspath__.
            Byte paths use native bytes on POSIX and UTF-8 on Windows.

        Notes
        -----
        A trained index without vectors raises RuntimeError; add vectors first.
        IVF with HNSW centroid routing also writes its required sidecar file.
        """

    @staticmethod
    def load(path: _Path) -> IvfIndex:
        """Load an index and its saved rotation from disk.

        Parameters
        ----------
        path : str, bytes, bytearray, or os.PathLike
            Index file path. PathLike objects are resolved via __fspath__.
            Byte paths use native bytes on POSIX and UTF-8 on Windows.

        Returns
        -------
        index : IvfIndex
            A trained index ready for search.
        """

    @property
    def initializer(self) -> _Initializer:
        """Resolved centroid routing strategy."""

    def reserve(self, cluster_id: SupportsIndex, num_points: SupportsIndex) -> None:
        """Preallocate storage for a cluster.

        Parameters
        ----------
        cluster_id : int
            Cluster ID in [0, num_clusters).
        num_points : int
            Nonnegative capacity to reserve for this cluster.

        Notes
        -----
        Requires a trained or loaded index. Reserved room is not saved.
        """

class HnswIndex:
    """An approximate nearest-neighbor index using HNSW.

    Parameters
    ----------
    dim : int
        Original vector dimension, from 64 through 65536.
    nbits : int
        Keyword-only. Total quantization bits, 1 through 9.
    num_clusters : int, optional
        Keyword-only. Number of centroids train or set_centroids provides.
        Default is 16.
    M : int, optional
        Keyword-only. Graph connectivity, at least 2; the base layer keeps up to
        2 * M edges. Default is 16.
    ef_construction : int, optional
        Keyword-only. Construction search breadth. Default is 200.
    metric : {'l2', 'ip', 'innerproduct'}, optional
        Keyword-only. Squared L2 or inner-product distance. Default is 'l2'.
        Vectors are not automatically normalized for cosine similarity.
    seed : int, optional
        Keyword-only. Graph and rotation seed, also the initial cp.seed (its low
        32 bits). Default is 100.

    Notes
    -----
    Train the index or set its centroids, then add vectors before searching or
    saving; loaded indexes are already trained. Capacity grows automatically
    on add; resize can preallocate it. Concurrent reads are supported; an
    overlapping mutation raises RuntimeError.
    Data, centroids and queries must be finite float32 values within the index
    coordinate range; other values raise ValueError.
    With more than one thread, add links points in a nondeterministic order;
    pass num_threads=1 for a reproducible graph.
    The 0.7 forms HnswIndex(dim, max_elements, ...), random_seed= and build()
    still work in 0.8, raise DeprecationWarning, and will be removed in 0.9.

    Examples
    --------
    >>> import numpy as np
    >>> from rabitqlib import HnswIndex
    >>> x = np.random.default_rng(42).normal(size=(80, 64)).astype(np.float32)
    >>> index = HnswIndex(64, nbits=4, num_clusters=2)
    >>> index.train(x)
    >>> ids = index.add(x, num_threads=1)
    >>> ids, distances = index.search(x[:2], k=3)
    >>> ids.shape, distances.shape
    ((2, 3), (2, 3))
    """

    @overload
    def __init__(
        self,
        dim: SupportsIndex,
        *,
        nbits: SupportsIndex,
        num_clusters: SupportsIndex = 16,
        M: SupportsIndex = 16,
        ef_construction: SupportsIndex = 200,
        metric: _Metric = "l2",
        seed: SupportsIndex = 100,
    ) -> None:
        """An approximate nearest-neighbor index using HNSW.

        Parameters
        ----------
        dim : int
            Original vector dimension, from 64 through 65536.
        nbits : int
            Keyword-only. Total quantization bits, 1 through 9.
        num_clusters : int, optional
            Keyword-only. Number of centroids train or set_centroids provides.
            Default is 16.
        M : int, optional
            Keyword-only. Graph connectivity, at least 2; the base layer keeps up to
            2 * M edges. Default is 16.
        ef_construction : int, optional
            Keyword-only. Construction search breadth. Default is 200.
        metric : {'l2', 'ip', 'innerproduct'}, optional
            Keyword-only. Squared L2 or inner-product distance. Default is 'l2'.
            Vectors are not automatically normalized for cosine similarity.
        seed : int, optional
            Keyword-only. Graph and rotation seed, also the initial cp.seed (its low
            32 bits). Default is 100.

        Notes
        -----
        Train the index or set its centroids, then add vectors before searching or
        saving; loaded indexes are already trained. Capacity grows automatically
        on add; resize can preallocate it. Concurrent reads are supported; an
        overlapping mutation raises RuntimeError.
        Data, centroids and queries must be finite float32 values within the index
        coordinate range; other values raise ValueError.
        With more than one thread, add links points in a nondeterministic order;
        pass num_threads=1 for a reproducible graph.
        The 0.7 forms HnswIndex(dim, max_elements, ...), random_seed= and build()
        still work in 0.8, raise DeprecationWarning, and will be removed in 0.9.
        """
    @overload
    @deprecated(
        "HnswIndex(dim, max_elements, ...) is deprecated since 0.8; use "
        "HnswIndex(dim, nbits=...), whose capacity grows on add"
    )
    def __init__(
        self,
        dim: SupportsIndex,
        max_elements: SupportsIndex,
        M: SupportsIndex = 16,
        ef_construction: SupportsIndex = 200,
        nbits: SupportsIndex = 8,
        metric: _Metric = "l2",
        random_seed: SupportsIndex = 100,
    ) -> None: ...
    @overload
    @deprecated("HnswIndex(random_seed=...) is deprecated since 0.8; use seed=")
    def __init__(
        self,
        dim: SupportsIndex,
        *,
        nbits: SupportsIndex,
        num_clusters: SupportsIndex = 16,
        M: SupportsIndex = 16,
        ef_construction: SupportsIndex = 200,
        metric: _Metric = "l2",
        random_seed: SupportsIndex,
    ) -> None: ...
    def __repr__(self) -> str:
        """Return a one-line summary of configuration, counts and training state."""

    def __len__(self) -> int:
        """Return the number of live, searchable points."""

    @property
    def dim(self) -> int:
        """Original vector dimension."""

    @property
    def max_elements(self) -> int:
        """Current storage capacity."""

    @property
    def num_points(self) -> int:
        """Number of live, searchable points."""

    @property
    def num_stored_points(self) -> int:
        """Number of stored points, including removed points."""

    @property
    def is_trained(self) -> bool:
        """Whether centroids are set, by train, set_centroids, build or load."""

    @property
    @deprecated("HnswIndex.is_built is deprecated since 0.8; use is_trained or len()")
    def is_built(self) -> bool:
        """Deprecated since 0.8: whether the index stores points; use len()."""

    @property
    def seed(self) -> int | None:
        """Graph and rotation seed, or None for an index loaded from disk."""

    @property
    def cp(self) -> RaBitQKMeansParameters:
        """Clustering parameters train uses, a RaBitQKMeansParameters. Defaults: niter 25, max_points_per_centroid 256, spherical for inner product, and seed equal to the low 32 bits of the index seed."""

    @cp.setter
    def cp(self, value: RaBitQKMeansParameters) -> None: ...
    @property
    def metric(self) -> Literal["l2", "ip"]:
        """Canonical metric name: 'l2' or 'ip'."""

    @property
    def nbits(self) -> int:
        """Total storage bits per coordinate."""

    @property
    def num_clusters(self) -> int:
        """Number of centroids: the configured count until centroids are set."""

    def train(
        self, x: npt.ArrayLike, *, num_threads: SupportsIndex | None = None
    ) -> None:
        """Train num_clusters centroids on vectors with RaBitQKMeans and set them.

        Parameters
        ----------
        x : array_like of float, shape (n, dim)
            Training vectors in the original, unpadded coordinates, with
            n >= num_clusters. Converted to contiguous float32; a copy may be made.
        num_threads : int, optional
            Keyword-only. Worker count for this call. None (default) uses
            cp.num_threads; 0 selects the available CPU count.

        Notes
        -----
        Clusters with the settings in cp. Only centroids are trained; add assigns
        vectors. Retraining replaces the centroids while the index stores no
        vectors, and raises RuntimeError afterwards. A failed call leaves the index
        unchanged.
        """

    def set_centroids(self, centroids: npt.ArrayLike) -> None:
        """Use the given centroids instead of training.

        Parameters
        ----------
        centroids : array_like of float, shape (num_clusters, dim)
            Centroids in the original, unpadded coordinates. Converted to
            contiguous float32; a copy may be made.

        Notes
        -----
        The index counts as trained afterwards. Replacing centroids is allowed
        while the index stores no vectors and raises RuntimeError afterwards. A
        failed call leaves the index unchanged.
        """

    @deprecated(
        "HnswIndex.build() is deprecated since 0.8; use set_centroids() or "
        "train(), then add()"
    )
    def build(
        self,
        data: npt.ArrayLike,
        centroids: npt.ArrayLike,
        cluster_ids: _IntegerIds,
        num_threads: SupportsIndex = 1,
        fast_quantization: bool | np.bool_ = False,
    ) -> None:
        """Build the index from vectors and supplied centroid assignments.

        Deprecated since 0.8 and removed in 0.9: use set_centroids() (or
        train()) and then add(). The defaults match 0.7, including a single
        thread.

        Parameters
        ----------
        data : array_like of float, shape (n, dim)
            Vectors in the original, unpadded coordinates. Converted to
            contiguous float32; a copy may be made.
        centroids : array_like of float, shape (num_clusters, dim)
            Vectors in the original, unpadded coordinates. Converted to
            contiguous float32; a copy may be made.
        cluster_ids : array_like of int, shape (n,)
            Centroid ID per vector, in [0, num_clusters). Floating-point and
            boolean IDs are rejected.
        num_threads : int, optional
            Worker count. Default is 1; 0 selects the available CPU count.
        fast_quantization : bool, optional
            Use faster quantizer configuration. Default is False.
        """

    @overload
    def add(
        self,
        data: npt.ArrayLike,
        *,
        cluster_ids: _IntegerIds | None = None,
        num_threads: SupportsIndex = 0,
        fast_quantization: bool | np.bool_ = False,
    ) -> npt.NDArray[np.uint32]:
        """Add vectors using the trained centroids.

        Parameters
        ----------
        data : array_like of float, shape (n, dim)
            Vectors in the original, unpadded coordinates. Converted to contiguous
            float32; a copy may be made.
        cluster_ids : array_like of int, shape (n,), optional
            Keyword-only. Existing centroid IDs. None assigns each vector to its
            nearest centroid. Centroids do not move.
        num_threads : int, optional
            Keyword-only. Worker count. Default is 0, which selects the available
            CPU count; more than one thread links nondeterministically, so pass 1
            for a reproducible graph.
        fast_quantization : bool, optional
            Keyword-only. Use faster quantizer configuration. Default is False.

        Returns
        -------
        ids : ndarray of uint32, shape (n,)
            Assigned point IDs.

        Notes
        -----
        Requires a trained or loaded index. The first add into an empty index links
        vectors with their exact coordinates; later adds link through the stored
        codes. Capacity grows geometrically when needed. Removed IDs are not
        reused. Passing the options positionally is deprecated.
        """
    @overload
    @deprecated(
        "Passing options to HnswIndex.add() by position is deprecated since 0.8; "
        "pass them by keyword"
    )
    def add(
        self,
        data: npt.ArrayLike,
        cluster_ids: _IntegerIds | None,
        num_threads: SupportsIndex = ...,
        fast_quantization: bool | np.bool_ = ...,
    ) -> npt.NDArray[np.uint32]: ...
    def remove(self, ids: _IntegerIds) -> int:
        """Exclude point IDs from subsequent search results.

        Parameters
        ----------
        ids : array_like of int, shape (n,)
            IDs in [0, num_stored_points). Floating-point and boolean IDs are
            rejected.

        Returns
        -------
        count : int
            Number of newly removed points; repeats do not count twice.

        Notes
        -----
        Removal does not reclaim storage and cannot be undone.
        """

    def search(
        self,
        queries: npt.ArrayLike,
        k: SupportsIndex,
        ef: SupportsIndex = 0,
        num_threads: SupportsIndex = 1,
    ) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.float32]]:
        """Search a batch of query vectors.

        Parameters
        ----------
        queries : array_like of float, shape (n_queries, dim)
            Vectors in the original, unpadded coordinates. Converted to contiguous
            float32; a copy may be made.
        k : int
            Number of neighbors requested, between 1 and num_stored_points.
        ef : int, optional
            Search breadth. 0 (default) selects max(k, 10); values above the stored
            point count are clamped.
        num_threads : int, optional
            Worker count. Default is 1; 0 selects the available CPU count.

        Returns
        -------
        ids : ndarray of uint32, shape (n_queries, k)
            Point IDs of the selected neighbors.
        distances : ndarray of float32, shape (n_queries, k)
            Lower values are better; squared L2 estimates for l2 and inner-product
            distance estimates for ip. These are distances, not similarity scores.

        Notes
        -----
        Requires a trained or loaded index. Returns (ids, distances), in that
        order. Missing neighbors have ID 4294967295 and distance inf. Array shapes
        and value ranges are checked at runtime, not by static type checkers.
        """

    def save(self, path: _Path) -> None:
        """Save an index with stored vectors to disk.

        Parameters
        ----------
        path : str, bytes, bytearray, or os.PathLike
            Index file path. PathLike objects are resolved via __fspath__.
            Byte paths use native bytes on POSIX and UTF-8 on Windows.

        Notes
        -----
        A trained index without vectors raises RuntimeError; add vectors first.
        The saved state includes the rotation used to build the index.
        """

    @staticmethod
    def load(path: _Path) -> HnswIndex:
        """Load an index and its saved rotation from disk.

        Parameters
        ----------
        path : str, bytes, bytearray, or os.PathLike
            Index file path. PathLike objects are resolved via __fspath__.
            Byte paths use native bytes on POSIX and UTF-8 on Windows.

        Returns
        -------
        index : HnswIndex
            A trained index ready for search.
        """

    def resize(self, max_elements: SupportsIndex) -> None:
        """Change the storage capacity, for example to preallocate before add.

        Parameters
        ----------
        max_elements : int
            New capacity; must be at least num_stored_points.
        """

class SymqgIndex:
    """An approximate nearest-neighbor index using SymphonyQG.

    Parameters
    ----------
    dim : int
        Original vector dimension, from 64 through 65536.
    max_degree : int
        Graph degree bound; a positive multiple of 32.
    metric : {'l2', 'ip', 'innerproduct'}, optional
        Squared L2 or inner-product distance. Default is 'l2'. Vectors are not
        automatically normalized for cosine similarity.
    quantization_bits : {0, 4, 8}, optional
        Centroid-relative RaBitQ storage at 4 or 8 bits, or raw float32 at 0
        (default).

    Notes
    -----
    Build or load the index before searching or saving. Concurrent reads are
    supported; an overlapping mutation raises RuntimeError.
    Data, centroids and queries must be finite float32 values within the index
    coordinate range; other values raise ValueError.
    """

    def __init__(
        self,
        dim: SupportsIndex,
        max_degree: SupportsIndex,
        metric: _Metric = "l2",
        quantization_bits: SupportsIndex = 0,
    ) -> None:
        """An approximate nearest-neighbor index using SymphonyQG.

        Parameters
        ----------
        dim : int
            Original vector dimension, from 64 through 65536.
        max_degree : int
            Graph degree bound; a positive multiple of 32.
        metric : {'l2', 'ip', 'innerproduct'}, optional
            Squared L2 or inner-product distance. Default is 'l2'. Vectors are not
            automatically normalized for cosine similarity.
        quantization_bits : {0, 4, 8}, optional
            Centroid-relative RaBitQ storage at 4 or 8 bits, or raw float32 at 0
            (default).

        Notes
        -----
        Build or load the index before searching or saving. Concurrent reads are
        supported; an overlapping mutation raises RuntimeError.
        Data, centroids and queries must be finite float32 values within the index
        coordinate range; other values raise ValueError.
        """

    def __repr__(self) -> str:
        """Return a one-line summary of configuration, counts and build state."""

    @property
    def dim(self) -> int:
        """Original vector dimension."""

    @property
    def max_elements(self) -> int:
        """Current storage capacity."""

    @property
    def num_points(self) -> int:
        """Number of live, searchable points."""

    @property
    def num_stored_points(self) -> int:
        """Number of stored points, including removed points."""

    @property
    def is_built(self) -> bool:
        """Whether the index has been built or loaded."""

    @property
    def metric(self) -> Literal["l2", "ip"]:
        """Canonical metric name: 'l2' or 'ip'."""

    @property
    def max_degree(self) -> int:
        """Graph degree bound."""

    @property
    def quantization_bits(self) -> int:
        """Storage bits: 0 for raw float32, or 4 or 8 for RaBitQ."""

    def build(
        self,
        data: npt.ArrayLike,
        ef_construction: SupportsIndex,
        num_threads: SupportsIndex = 1,
        init: Literal["pipnn", "random"] = "pipnn",
    ) -> None:
        """Build the graph from input vectors.

        Parameters
        ----------
        data : array_like of float, shape (n, dim)
            Vectors in the original, unpadded coordinates. Converted to contiguous
            float32; a copy may be made.
        ef_construction : int
            Positive construction search breadth fitting uint32.
        num_threads : int, optional
            Worker count. Default is 1; 0 selects the available CPU count.
        init : {'pipnn', 'random'}, optional
            Graph initialization method. Default is 'pipnn'.

        Notes
        -----
        Requires n > max_degree. Input arrays need not be retained after this
        synchronous call.
        """

    def search(
        self,
        queries: npt.ArrayLike,
        k: SupportsIndex,
        ef: SupportsIndex,
        num_threads: SupportsIndex = 1,
    ) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.float32]]:
        """Search a batch of query vectors.

        Parameters
        ----------
        queries : array_like of float, shape (n_queries, dim)
            Vectors in the original, unpadded coordinates. Converted to contiguous
            float32; a copy may be made.
        k : int
            Number of neighbors requested, between 1 and num_points.
        ef : int
            Search breadth. Must be positive and at least k.
        num_threads : int, optional
            Worker count. Default is 1; 0 selects the available CPU count.

        Returns
        -------
        ids : ndarray of uint32, shape (n_queries, k)
            Point IDs of the selected neighbors.
        distances : ndarray of float32, shape (n_queries, k)
            Lower values are better; squared L2 estimates for l2 and inner-product
            distance estimates for ip. These are distances, not similarity scores.

        Notes
        -----
        Returns (ids, distances), in that order. Array shapes and value ranges are
        checked at runtime, not by static type checkers.
        """

    def search_batch(
        self,
        queries: npt.ArrayLike,
        k: SupportsIndex,
        ef: SupportsIndex,
        num_threads: SupportsIndex = 1,
    ) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.float32]]:
        """Search a batch of query vectors.

        Parameters
        ----------
        queries : array_like of float, shape (n_queries, dim)
            Vectors in the original, unpadded coordinates. Converted to contiguous
            float32; a copy may be made.
        k : int
            Number of neighbors requested, between 1 and num_points.
        ef : int
            Search breadth. Must be positive and at least k.
        num_threads : int, optional
            Worker count. Default is 1; 0 selects the available CPU count.

        Returns
        -------
        ids : ndarray of uint32, shape (n_queries, k)
            Point IDs of the selected neighbors.
        distances : ndarray of float32, shape (n_queries, k)
            Lower values are better; squared L2 estimates for l2 and inner-product
            distance estimates for ip. These are distances, not similarity scores.

        Notes
        -----
        Returns (ids, distances), in that order. Array shapes and value ranges are
        checked at runtime, not by static type checkers.
        """

    def save(self, path: _Path) -> None:
        """Save a built index to disk.

        Parameters
        ----------
        path : str, bytes, bytearray, or os.PathLike
            Index file path. PathLike objects are resolved via __fspath__.
            Byte paths use native bytes on POSIX and UTF-8 on Windows.

        Notes
        -----
        The saved state includes the rotation used to build the index.
        """

    @staticmethod
    def load(path: _Path) -> SymqgIndex:
        """Load an index and its saved rotation from disk.

        Parameters
        ----------
        path : str, bytes, bytearray, or os.PathLike
            Index file path. PathLike objects are resolved via __fspath__.
            Byte paths use native bytes on POSIX and UTF-8 on Windows.

        Returns
        -------
        index : SymqgIndex
            A built index ready for search.
        """

class QGKMeans:
    """Cluster vectors using SymphonyQG graph assignment.

    Parameters
    ----------
    d : int
        Original vector dimension, from 64 through 65536.
    k : int
        Number of centroids; greater than graph_degree and no more than the
        number of training vectors.
    niter : int, optional
        Maximum number of clustering iterations. Default is 25.
    verbose : bool, optional
        Print training progress when True. Default is False.
    spherical : bool, optional
        Normalize centroids after each update and use inner-product assignment.
        Normalize input vectors yourself for cosine clustering. Default is
        False.
    seed : int, optional
        Random seed. Default is 42.
    min_points_per_centroid : int, optional
        Warn below this many training vectors per centroid. Default is 39.
    max_points_per_centroid : int, optional
        When positive, train on a random sample of k * max_points_per_centroid
        vectors if there are more, then assign all vectors. Default is 0, which
        trains on all vectors; Faiss uses 256.
    early_stop_threshold : float, optional
        Stop when the absolute relative objective change is at most this
        threshold. Default is 0.0.
    num_threads : int, optional
        Worker count; 0 selects the available CPU count. Default is 0.
    final_assignment : FinalAssignmentMode, optional
        Approximate uses the training assignment method; Exact uses exhaustive
        assignment against final centroids. Default is
        FinalAssignmentMode.Approximate.
    graph_degree : int, optional
        Centroid graph degree; a positive multiple of 32. Default is 32.
    ef_build : int, optional
        Positive centroid-graph construction search breadth. Default is 240.
    ef_search : int, optional
        Positive centroid-graph assignment search breadth. Default is 16.
    graph_build_iterations : int, optional
        Positive number of graph-building passes after initialization. Default
        is 1.
    quantization_bits : int, optional
        Centroid storage: 0 for raw float32, or 4 or 8 for RaBitQ. Default is
        0.

    Notes
    -----
    Training results are None until the first successful train call. Set cp
    fields before training to customize subsequent runs. A failed train
    preserves previous results.

    When max_points_per_centroid is positive and n exceeds
    k * max_points_per_centroid, initialization and iterations use a random
    sample chosen by seed. obj and iteration_stats then describe the sample,
    while assignments, distances and final_obj cover all n vectors, so the two
    kinds of objective are not directly comparable.
    """

    d: int
    k: int
    cp: QGKMeansParameters

    def __init__(
        self,
        d: SupportsIndex,
        k: SupportsIndex,
        *,
        niter: SupportsIndex = 25,
        verbose: bool | np.bool_ = False,
        spherical: bool | np.bool_ = False,
        seed: SupportsIndex = 42,
        min_points_per_centroid: SupportsIndex = 39,
        max_points_per_centroid: SupportsIndex = 0,
        early_stop_threshold: float = 0.0,
        num_threads: SupportsIndex = 0,
        final_assignment: FinalAssignmentMode = FinalAssignmentMode.Approximate,
        graph_degree: SupportsIndex = 32,
        ef_build: SupportsIndex = 240,
        ef_search: SupportsIndex = 16,
        graph_build_iterations: SupportsIndex = 1,
        quantization_bits: SupportsIndex = 0,
    ) -> None:
        """Cluster vectors using SymphonyQG graph assignment.

        Parameters
        ----------
        d : int
            Original vector dimension, from 64 through 65536.
        k : int
            Number of centroids; greater than graph_degree and no more than the
            number of training vectors.
        niter : int, optional
            Maximum number of clustering iterations. Default is 25.
        verbose : bool, optional
            Print training progress when True. Default is False.
        spherical : bool, optional
            Normalize centroids after each update and use inner-product assignment.
            Normalize input vectors yourself for cosine clustering. Default is
            False.
        seed : int, optional
            Random seed. Default is 42.
        min_points_per_centroid : int, optional
            Warn below this many training vectors per centroid. Default is 39.
        max_points_per_centroid : int, optional
            When positive, train on a random sample of k * max_points_per_centroid
            vectors if there are more, then assign all vectors. Default is 0, which
            trains on all vectors; Faiss uses 256.
        early_stop_threshold : float, optional
            Stop when the absolute relative objective change is at most this
            threshold. Default is 0.0.
        num_threads : int, optional
            Worker count; 0 selects the available CPU count. Default is 0.
        final_assignment : FinalAssignmentMode, optional
            Approximate uses the training assignment method; Exact uses exhaustive
            assignment against final centroids. Default is
            FinalAssignmentMode.Approximate.
        graph_degree : int, optional
            Centroid graph degree; a positive multiple of 32. Default is 32.
        ef_build : int, optional
            Positive centroid-graph construction search breadth. Default is 240.
        ef_search : int, optional
            Positive centroid-graph assignment search breadth. Default is 16.
        graph_build_iterations : int, optional
            Positive number of graph-building passes after initialization. Default
            is 1.
        quantization_bits : int, optional
            Centroid storage: 0 for raw float32, or 4 or 8 for RaBitQ. Default is
            0.

        Notes
        -----
        Training results are None until the first successful train call. Set cp
        fields before training to customize subsequent runs. A failed train
        preserves previous results.

        When max_points_per_centroid is positive and n exceeds
        k * max_points_per_centroid, initialization and iterations use a random
        sample chosen by seed. obj and iteration_stats then describe the sample,
        while assignments, distances and final_obj cover all n vectors, so the two
        kinds of objective are not directly comparable.
        """

    def train(self, x: npt.ArrayLike) -> float:
        """Train centroids and retain the final assignments.

        Parameters
        ----------
        x : array_like of float, shape (n, d)
            Vectors in the original, unpadded coordinates. Converted to contiguous
            float32; a copy may be made.

        Returns
        -------
        objective : float
            Last pre-update objective, over the sample when training sampled the
            input. This can differ from final_obj, which uses all vectors, returned
            assignments and final centroids.

        Notes
        -----
        A failed fit preserves the last successful results. Returned arrays are
        copies.
        """

    def assign(
        self, x: npt.ArrayLike
    ) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.float32]]:
        """Assign vectors to the trained centroids.

        Parameters
        ----------
        x : array_like of float, shape (m, d)
            Vectors in the original, unpadded coordinates, with finite
            coordinates in the range accepted by train; m may be 0. Converted to
            contiguous float32; a copy may be made.

        Returns
        -------
        assignments : ndarray of uint32, shape (m,)
            Centroid IDs.
        distances : ndarray of float32, shape (m,)
            Distances to the assigned centroids, as in the distances property.

        Raises
        ------
        RuntimeError
            If no train call has succeeded.

        Notes
        -----
        Uses final_assignment and the other parameters of the last successful
        train call, but no labels or search hints from it, so each vector's
        result is independent of the other vectors in x. RaBitQKMeans encodes
        vectors around the training center. Approximate QGKMeans builds a
        centroid graph on first use and reuses it in later calls. For the
        training vectors, the result reproduces assignments and distances with
        Exact final assignment, with RaBitQKMeans, or when training sampled the
        input; unsampled approximate QGKMeans training uses search hints.
        """

    @property
    def centroids(self) -> npt.NDArray[np.float32] | None:
        """Final centroids, float32 array of shape (k, d), or None before training."""

    @property
    def assignments(self) -> npt.NDArray[np.uint32] | None:
        """Final centroid IDs, uint32 array of shape (n,), or None before training."""

    @property
    def distances(self) -> npt.NDArray[np.float32] | None:
        """Final assignment distances for all n vectors, float32 array of shape (n,), or None before training."""

    @property
    def obj(self) -> npt.NDArray[np.float64] | None:
        """Pre-update objectives over the vectors used for iterations, float64 array of shape (iterations,), or None before training."""

    @property
    def final_obj(self) -> float | None:
        """Objective for the final assignments of all n vectors, or None before training."""

    @property
    def iteration_stats(self) -> list[QGKMeansIterationStats] | None:
        """Per-iteration statistics over the vectors used for iterations, or None before training."""

class QGKMeansParameters:
    """Mutable training settings for QGKMeans.

    Notes
    -----
    Create with no arguments; set attributes directly or use keyword arguments
    on QGKMeans.
    """

    @property
    def niter(self) -> int:
        """Maximum number of clustering iterations. Default is 25."""
    @niter.setter
    def niter(self, value: SupportsIndex) -> None: ...
    @property
    def verbose(self) -> bool:
        """Print training progress when True. Default is False."""
    @verbose.setter
    def verbose(self, value: bool | np.bool_) -> None: ...
    @property
    def spherical(self) -> bool:
        """Normalize centroids after each update and use inner-product assignment. Normalize input vectors yourself for cosine clustering. Default is False."""
    @spherical.setter
    def spherical(self, value: bool | np.bool_) -> None: ...
    @property
    def seed(self) -> int:
        """Random seed. Default is 42."""
    @seed.setter
    def seed(self, value: SupportsIndex) -> None: ...
    @property
    def min_points_per_centroid(self) -> int:
        """Warn below this many training vectors per centroid. Default is 39."""
    @min_points_per_centroid.setter
    def min_points_per_centroid(self, value: SupportsIndex) -> None: ...
    @property
    def max_points_per_centroid(self) -> int:
        """When positive, train on a random sample of k * max_points_per_centroid vectors if there are more, then assign all vectors. Default is 0, which trains on all vectors; Faiss uses 256."""
    @max_points_per_centroid.setter
    def max_points_per_centroid(self, value: SupportsIndex) -> None: ...

    early_stop_threshold: float
    """Stop when the absolute relative objective change is at most this threshold. Default is 0.0."""

    @property
    def num_threads(self) -> int:
        """Worker count; 0 selects the available CPU count. Default is 0."""
    @num_threads.setter
    def num_threads(self, value: SupportsIndex) -> None: ...

    final_assignment: FinalAssignmentMode
    """Approximate uses the training assignment method; Exact uses exhaustive assignment against final centroids. Default is FinalAssignmentMode.Approximate."""

    @property
    def graph_degree(self) -> int:
        """Centroid graph degree; a positive multiple of 32. Default is 32."""
    @graph_degree.setter
    def graph_degree(self, value: SupportsIndex) -> None: ...
    @property
    def ef_build(self) -> int:
        """Positive centroid-graph construction search breadth. Default is 240."""
    @ef_build.setter
    def ef_build(self, value: SupportsIndex) -> None: ...
    @property
    def ef_search(self) -> int:
        """Positive centroid-graph assignment search breadth. Default is 16."""
    @ef_search.setter
    def ef_search(self, value: SupportsIndex) -> None: ...
    @property
    def graph_build_iterations(self) -> int:
        """Positive number of graph-building passes after initialization. Default is 1."""
    @graph_build_iterations.setter
    def graph_build_iterations(self, value: SupportsIndex) -> None: ...
    @property
    def quantization_bits(self) -> int:
        """Centroid storage: 0 for raw float32, or 4 or 8 for RaBitQ. Default is 0."""
    @quantization_bits.setter
    def quantization_bits(self, value: SupportsIndex) -> None: ...
    def __init__(self) -> None:
        """Create parameters with their documented defaults."""

class RaBitQKMeans:
    """Cluster vectors using flat RaBitQ assignment.

    Parameters
    ----------
    d : int
        Original vector dimension, from 64 through 65536.
    k : int
        Number of centroids, between 1 and the number of training vectors.
    niter : int, optional
        Maximum number of clustering iterations. Default is 25.
    verbose : bool, optional
        Print training progress when True. Default is False.
    spherical : bool, optional
        Normalize centroids after each update and use inner-product assignment.
        Normalize input vectors yourself for cosine clustering. Default is
        False.
    seed : int, optional
        Random seed. Default is 42.
    min_points_per_centroid : int, optional
        Warn below this many training vectors per centroid. Default is 39.
    max_points_per_centroid : int, optional
        When positive, train on a random sample of k * max_points_per_centroid
        vectors if there are more, then assign all vectors. Default is 0, which
        trains on all vectors; Faiss uses 256.
    early_stop_threshold : float, optional
        Stop when the absolute relative objective change is at most this
        threshold. Default is 0.0.
    num_threads : int, optional
        Worker count; 0 selects the available CPU count. Default is 0.
    final_assignment : FinalAssignmentMode, optional
        Approximate uses the training assignment method; Exact uses exhaustive
        assignment against final centroids. Default is
        FinalAssignmentMode.Approximate.

    Notes
    -----
    Training results are None until the first successful train call. Set cp
    fields before training to customize subsequent runs. A failed train
    preserves previous results.

    When max_points_per_centroid is positive and n exceeds
    k * max_points_per_centroid, initialization and iterations use a random
    sample chosen by seed. obj and iteration_stats then describe the sample,
    while assignments, distances and final_obj cover all n vectors, so the two
    kinds of objective are not directly comparable.

    Examples
    --------
    >>> import numpy as np
    >>> from rabitqlib import RaBitQKMeans
    >>> x = np.random.default_rng(42).normal(size=(64, 64)).astype(np.float32)
    >>> model = RaBitQKMeans(64, 2, niter=2, num_threads=1)
    >>> objective = model.train(x)
    >>> model.centroids.shape
    (2, 64)
    """

    d: int
    k: int
    cp: RaBitQKMeansParameters

    def __init__(
        self,
        d: SupportsIndex,
        k: SupportsIndex,
        *,
        niter: SupportsIndex = 25,
        verbose: bool | np.bool_ = False,
        spherical: bool | np.bool_ = False,
        seed: SupportsIndex = 42,
        min_points_per_centroid: SupportsIndex = 39,
        max_points_per_centroid: SupportsIndex = 0,
        early_stop_threshold: float = 0.0,
        num_threads: SupportsIndex = 0,
        final_assignment: FinalAssignmentMode = FinalAssignmentMode.Approximate,
    ) -> None:
        """Cluster vectors using flat RaBitQ assignment.

        Parameters
        ----------
        d : int
            Original vector dimension, from 64 through 65536.
        k : int
            Number of centroids, between 1 and the number of training vectors.
        niter : int, optional
            Maximum number of clustering iterations. Default is 25.
        verbose : bool, optional
            Print training progress when True. Default is False.
        spherical : bool, optional
            Normalize centroids after each update and use inner-product assignment.
            Normalize input vectors yourself for cosine clustering. Default is
            False.
        seed : int, optional
            Random seed. Default is 42.
        min_points_per_centroid : int, optional
            Warn below this many training vectors per centroid. Default is 39.
        max_points_per_centroid : int, optional
            When positive, train on a random sample of k * max_points_per_centroid
            vectors if there are more, then assign all vectors. Default is 0, which
            trains on all vectors; Faiss uses 256.
        early_stop_threshold : float, optional
            Stop when the absolute relative objective change is at most this
            threshold. Default is 0.0.
        num_threads : int, optional
            Worker count; 0 selects the available CPU count. Default is 0.
        final_assignment : FinalAssignmentMode, optional
            Approximate uses the training assignment method; Exact uses exhaustive
            assignment against final centroids. Default is
            FinalAssignmentMode.Approximate.

        Notes
        -----
        Training results are None until the first successful train call. Set cp
        fields before training to customize subsequent runs. A failed train
        preserves previous results.

        When max_points_per_centroid is positive and n exceeds
        k * max_points_per_centroid, initialization and iterations use a random
        sample chosen by seed. obj and iteration_stats then describe the sample,
        while assignments, distances and final_obj cover all n vectors, so the two
        kinds of objective are not directly comparable.

        Examples
        --------
        >>> import numpy as np
        >>> from rabitqlib import RaBitQKMeans
        >>> x = np.random.default_rng(42).normal(size=(64, 64)).astype(np.float32)
        >>> model = RaBitQKMeans(64, 2, niter=2, num_threads=1)
        >>> objective = model.train(x)
        >>> model.centroids.shape
        (2, 64)
        """

    def train(self, x: npt.ArrayLike) -> float:
        """Train centroids and retain the final assignments.

        Parameters
        ----------
        x : array_like of float, shape (n, d)
            Vectors in the original, unpadded coordinates. Converted to contiguous
            float32; a copy may be made.

        Returns
        -------
        objective : float
            Last pre-update objective, over the sample when training sampled the
            input. This can differ from final_obj, which uses all vectors, returned
            assignments and final centroids.

        Notes
        -----
        A failed fit preserves the last successful results. Returned arrays are
        copies.
        """

    def assign(
        self, x: npt.ArrayLike
    ) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.float32]]:
        """Assign vectors to the trained centroids.

        Parameters
        ----------
        x : array_like of float, shape (m, d)
            Vectors in the original, unpadded coordinates, with finite
            coordinates in the range accepted by train; m may be 0. Converted to
            contiguous float32; a copy may be made.

        Returns
        -------
        assignments : ndarray of uint32, shape (m,)
            Centroid IDs.
        distances : ndarray of float32, shape (m,)
            Distances to the assigned centroids, as in the distances property.

        Raises
        ------
        RuntimeError
            If no train call has succeeded.

        Notes
        -----
        Uses final_assignment and the other parameters of the last successful
        train call, but no labels or search hints from it, so each vector's
        result is independent of the other vectors in x. RaBitQKMeans encodes
        vectors around the training center. Approximate QGKMeans builds a
        centroid graph on first use and reuses it in later calls. For the
        training vectors, the result reproduces assignments and distances with
        Exact final assignment, with RaBitQKMeans, or when training sampled the
        input; unsampled approximate QGKMeans training uses search hints.
        """

    @property
    def centroids(self) -> npt.NDArray[np.float32] | None:
        """Final centroids, float32 array of shape (k, d), or None before training."""

    @property
    def assignments(self) -> npt.NDArray[np.uint32] | None:
        """Final centroid IDs, uint32 array of shape (n,), or None before training."""

    @property
    def distances(self) -> npt.NDArray[np.float32] | None:
        """Final assignment distances for all n vectors, float32 array of shape (n,), or None before training."""

    @property
    def obj(self) -> npt.NDArray[np.float64] | None:
        """Pre-update objectives over the vectors used for iterations, float64 array of shape (iterations,), or None before training."""

    @property
    def final_obj(self) -> float | None:
        """Objective for the final assignments of all n vectors, or None before training."""

    @property
    def iteration_stats(self) -> list[QGKMeansIterationStats] | None:
        """Per-iteration statistics over the vectors used for iterations, or None before training."""

class RaBitQKMeansParameters:
    """Mutable training settings for RaBitQKMeans.

    Notes
    -----
    Create with no arguments; set attributes directly or use keyword arguments
    on RaBitQKMeans.
    """

    @property
    def niter(self) -> int:
        """Maximum number of clustering iterations. Default is 25."""
    @niter.setter
    def niter(self, value: SupportsIndex) -> None: ...
    @property
    def verbose(self) -> bool:
        """Print training progress when True. Default is False."""
    @verbose.setter
    def verbose(self, value: bool | np.bool_) -> None: ...
    @property
    def spherical(self) -> bool:
        """Normalize centroids after each update and use inner-product assignment. Normalize input vectors yourself for cosine clustering. Default is False."""
    @spherical.setter
    def spherical(self, value: bool | np.bool_) -> None: ...
    @property
    def seed(self) -> int:
        """Random seed. Default is 42."""
    @seed.setter
    def seed(self, value: SupportsIndex) -> None: ...
    @property
    def min_points_per_centroid(self) -> int:
        """Warn below this many training vectors per centroid. Default is 39."""
    @min_points_per_centroid.setter
    def min_points_per_centroid(self, value: SupportsIndex) -> None: ...
    @property
    def max_points_per_centroid(self) -> int:
        """When positive, train on a random sample of k * max_points_per_centroid vectors if there are more, then assign all vectors. Default is 0, which trains on all vectors; Faiss uses 256."""
    @max_points_per_centroid.setter
    def max_points_per_centroid(self, value: SupportsIndex) -> None: ...

    early_stop_threshold: float
    """Stop when the absolute relative objective change is at most this threshold. Default is 0.0."""

    @property
    def num_threads(self) -> int:
        """Worker count; 0 selects the available CPU count. Default is 0."""
    @num_threads.setter
    def num_threads(self, value: SupportsIndex) -> None: ...

    final_assignment: FinalAssignmentMode
    """Approximate uses the training assignment method; Exact uses exhaustive assignment against final centroids. Default is FinalAssignmentMode.Approximate."""

    def __init__(self) -> None:
        """Create parameters with their documented defaults."""
