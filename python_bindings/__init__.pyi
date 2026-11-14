"""Public Python interfaces for RaBitQ indexes and clustering.

Array shapes and numeric ranges are documented runtime constraints.
"""

from collections.abc import Sequence
from os import PathLike
from typing import Any, Literal, SupportsIndex, TypeAlias

import numpy as np
import numpy.typing as npt

__version__: str

__all__ = [
    "FormatError",
    "HnswIndex",
    "IvfIndex",
    "KMeansIterationStats",
    "QGKMeans",
    "QGKMeansParameters",
    "RaBitQKMeans",
    "RaBitQKMeansParameters",
    "SymqgIndex",
]

_Path: TypeAlias = str | bytes | bytearray | PathLike[str] | PathLike[bytes]

_Metric: TypeAlias = Literal["l2", "ip", "innerproduct", "cosine"]
_Routing: TypeAlias = Literal["auto", "flat", "flat_rabitq", "hnsw"]
_QGInit: TypeAlias = Literal["pipnn", "random"]
_FinalAssignment: TypeAlias = Literal["approximate", "exact"]
# Rank, ID ranges and Python bool (an int subtype) are validated at runtime.
_IntegerIds: TypeAlias = npt.NDArray[np.integer[Any]] | Sequence[int | np.integer[Any]]

class FormatError(ValueError, RuntimeError):
    """A damaged or mismatched index file or serialized byte string, or one this release cannot read. A subclass of both ValueError and RuntimeError."""

class KMeansIterationStats:
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

class IvfIndex:
    """An approximate nearest-neighbor index using IVF.

    Parameters
    ----------
    dim : int
        Original vector dimension, from 64 through 65536.
    num_clusters : int
        Number of centroids, from train, set_centroids or the first add.
    nbits : int, optional
        Keyword-only. Total quantization bits, 1 through 9, or 32 for owned
        float32 vectors with one-bit filtering and raw reranking. Default is 7.
    metric : {'l2', 'ip', 'innerproduct', 'cosine'}, optional
        Squared L2, inner-product or cosine distance. Default is 'l2'. 'cosine'
        scales data and queries to unit length.
    routing : {'auto', 'flat', 'flat_rabitq', 'hnsw'}, optional
        Centroid routing strategy. Default is 'auto'.
    seed : int, optional
        Rotation seed in [0, 2**32 - 1], also the initial cp.seed. None
        (default) draws one at random; the seed property reports it.

    Notes
    -----
    The first add trains an untrained index on the added vectors. Call train
    on a sample, or set_centroids, first to choose the training data or the
    centroids. Loaded indexes are already trained. Concurrent reads are
    supported; an overlapping mutation raises RuntimeError.
    Data, centroids and queries must be finite float32 values within the index
    coordinate range; other values raise ValueError.

    Examples
    --------
    >>> import numpy as np
    >>> from rabitqlib import IvfIndex
    >>> x = np.random.default_rng(42).normal(size=(80, 64)).astype(np.float32)
    >>> index = IvfIndex(64, 2, seed=7)
    >>> ids = index.add(x)  # trains on x first
    >>> ids, distances = index.search(x[:2], k=3)
    >>> ids.shape, distances.shape
    ((2, 3), (2, 3))
    """

    def __init__(
        self,
        dim: SupportsIndex,
        num_clusters: SupportsIndex,
        *,
        nbits: SupportsIndex = 7,
        metric: _Metric = "l2",
        routing: _Routing = "auto",
        seed: SupportsIndex | None = None,
    ) -> None:
        """An approximate nearest-neighbor index using IVF.

        Parameters
        ----------
        dim : int
            Original vector dimension, from 64 through 65536.
        num_clusters : int
            Number of centroids, from train, set_centroids or the first add.
        nbits : int, optional
            Keyword-only. Total quantization bits, 1 through 9, or 32 for owned
            float32 vectors with one-bit filtering and raw reranking. Default is 7.
        metric : {'l2', 'ip', 'innerproduct', 'cosine'}, optional
            Squared L2, inner-product or cosine distance. Default is 'l2'. 'cosine'
            scales data and queries to unit length.
        routing : {'auto', 'flat', 'flat_rabitq', 'hnsw'}, optional
            Centroid routing strategy. Default is 'auto'.
        seed : int, optional
            Rotation seed in [0, 2**32 - 1], also the initial cp.seed. None
            (default) draws one at random; the seed property reports it.

        Notes
        -----
        The first add trains an untrained index on the added vectors. Call train
        on a sample, or set_centroids, first to choose the training data or the
        centroids. Loaded indexes are already trained. Concurrent reads are
        supported; an overlapping mutation raises RuntimeError.
        Data, centroids and queries must be finite float32 values within the index
        coordinate range; other values raise ValueError.
        """
    def __repr__(self) -> str:
        """Return a one-line summary of configuration, counts and training state."""

    def __len__(self) -> int:
        """Return the number of live, searchable points."""

    @property
    def dim(self) -> int:
        """Original vector dimension."""

    @property
    def num_points(self) -> int:
        """Number of live, searchable points."""

    @property
    def num_stored_points(self) -> int:
        """Number of stored points, including removed points."""

    @property
    def is_trained(self) -> bool:
        """Whether centroids are set, by train, set_centroids, the first add or load."""

    @property
    def seed(self) -> int | None:
        """Rotation seed, or None for an index loaded from a file from a release up to 0.8."""

    @property
    def cp(self) -> RaBitQKMeansParameters:
        """Clustering parameters train uses, a RaBitQKMeansParameters. Defaults: niter 10, max_points_per_centroid 256, spherical for inner product and cosine, and seed equal to the index seed."""

    @cp.setter
    def cp(self, value: RaBitQKMeansParameters) -> None: ...
    @property
    def metric(self) -> Literal["l2", "ip", "cosine"]:
        """Canonical metric name: 'l2', 'ip' or 'cosine'."""

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
            cp.num_threads; 0 selects the available CPU count, capped by OMP_NUM_THREADS.

        Notes
        -----
        Clusters with the settings in cp: RaBitQKMeans up to 2000 clusters, and
        QGKMeans with its default graph settings above that (a provisional,
        unmeasured threshold). Only centroids are trained; add assigns vectors.
        Retraining replaces the centroids while the index stores no vectors, and
        raises RuntimeError afterwards. A failed call leaves the index unchanged.
        """

    def set_centroids(
        self, centroids: npt.ArrayLike, *, num_threads: SupportsIndex | None = None
    ) -> None:
        """Use the given centroids instead of training.

        Parameters
        ----------
        centroids : array_like of float, shape (num_clusters, dim)
            Centroids in the original, unpadded coordinates. Converted to
            contiguous float32; a copy may be made.
        num_threads : int, optional
            Keyword-only. Worker count for rotating the centroids and building the
            centroid routing. None (default) and 0 select the available CPU count,
            capped by OMP_NUM_THREADS. Only the HNSW routing graph varies with it:
            build it with 1 thread and a fixed seed for a reproducible graph.

        Notes
        -----
        The index counts as trained afterwards. Replacing centroids is allowed
        while the index stores no vectors and raises RuntimeError afterwards. A
        failed call leaves the index unchanged.
        """

    def add(
        self,
        data: npt.ArrayLike,
        *,
        cluster_ids: _IntegerIds | None = None,
        num_threads: SupportsIndex | None = None,
        fast_quantization: bool | np.bool_ = False,
    ) -> npt.NDArray[np.int64]:
        """Add vectors, training the index first if needed.

        Parameters
        ----------
        data : array_like of float, shape (n, dim)
            Vectors in the original, unpadded coordinates. Converted to contiguous
            float32; a copy may be made.
        cluster_ids : array_like of int, shape (n,), optional
            Keyword-only. Existing centroid IDs. None routes each vector using the
            index's configured routing strategy, which may be approximate.
            Centroids do not move.
        num_threads : int, optional
            Keyword-only. Worker count. None (default) and 0 select the available
            CPU count, capped by OMP_NUM_THREADS.
        fast_quantization : bool, optional
            Keyword-only. Use faster quantizer configuration. Default is False.

        Returns
        -------
        ids : ndarray of int64, shape (n,)
            Assigned point IDs.

        Notes
        -----
        On an untrained index, add first trains num_clusters centroids on data,
        as train(data) does, so data needs at least num_clusters rows, and an
        empty first add raises RuntimeError; with
        num_threads None, training uses cp.num_threads. Passing cluster_ids to an
        untrained index raises RuntimeError. If the call fails, the index stays
        untrained and empty. Batch additions to reduce repeated cluster copying;
        reserve_cluster can preallocate room.
        """
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
        nprobe: SupportsIndex | None = None,
        *,
        high_accuracy: bool | np.bool_ | None = None,
        num_threads: SupportsIndex | None = None,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
        """Search a batch of query vectors.

        Parameters
        ----------
        queries : array_like of float, shape (n_queries, dim)
            Vectors in the original, unpadded coordinates. Converted to contiguous
            float32; a copy may be made.
        k : int
            Number of neighbors requested, at least 1. It may exceed the number of
            points; slots without a neighbor are padded as described below.
        nprobe : int, optional
            Number of clusters to probe, at least 1 and capped at num_clusters. None
            (default) selects num_clusters / 16, rounded to the nearest integer and
            at least 1.
        high_accuracy : bool, optional
            Keyword-only. Enable HACC. None (default) selects it for 4-9 bits and
            standard FastScan otherwise. Raw storage uses standard FastScan.
        num_threads : int, optional
            Keyword-only. Worker count. None (default) selects 1; 0 selects the
            available CPU count, capped by OMP_NUM_THREADS.

        Returns
        -------
        ids : ndarray of int64, shape (n_queries, k)
            Point IDs of the selected neighbors.
        distances : ndarray of float32, shape (n_queries, k)
            Lower values are better: squared L2 estimates for l2, 1 - dot for ip and
            1 - cos for cosine. These are distances, not similarity scores.

        Notes
        -----
        Requires a trained or loaded index. Returns (ids, distances), in that
        order. Missing neighbors have ID -1 and distance inf; test
        np.isfinite(distances) to find them. Array shapes
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
        The one file includes HNSW centroid routing; releases up to 0.8 cannot read it.
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

        Notes
        -----
        Files from releases up to 0.8 with HNSW centroid routing also need the
        accompanying ``<path>.hnsw`` file.
        """

    @property
    def routing(self) -> _Routing:
        """Resolved centroid routing strategy, never 'auto'."""

    def reserve_cluster(
        self, cluster_id: SupportsIndex, num_points: SupportsIndex
    ) -> None:
        """Preallocate storage for a cluster.

        Parameters
        ----------
        cluster_id : int
            Cluster ID in [0, num_clusters).
        num_points : int
            Nonnegative capacity to reserve for this cluster: the total number of
            points it can hold, including its current ones.

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
    nbits : int, optional
        Keyword-only. Total quantization bits, 1 through 9. Default is 7.
    num_clusters : int, optional
        Keyword-only. Number of centroids train, set_centroids or the first add
        provides; the first add caps it at its row count.
        Default is 16.
    M : int, optional
        Keyword-only. Graph connectivity, at least 2; the base layer keeps up to
        2 * M edges. Default is 16.
    ef_construction : int, optional
        Keyword-only. Construction search breadth. Default is 200.
    metric : {'l2', 'ip', 'innerproduct', 'cosine'}, optional
        Keyword-only. Squared L2, inner-product or cosine distance. Default is
        'l2'. 'cosine' scales data and queries to unit length.
    seed : int, optional
        Keyword-only. Graph and rotation seed in [0, 2**32 - 1], also the initial
        cp.seed. None (default) draws one at random; the seed property reports it.

    Notes
    -----
    The first add trains an untrained index on the added vectors. Call train
    on a sample, or set_centroids, first to choose the training data or the
    centroids. Loaded indexes are trained, except empty files saved untrained
    by 0.7. Capacity grows automatically
    on add; reserve can preallocate it. Concurrent reads are supported; an
    overlapping mutation raises RuntimeError.
    Data, centroids and queries must be finite float32 values within the index
    coordinate range; other values raise ValueError.
    With more than one thread, add links points in a nondeterministic order;
    pass num_threads=1 and a seed for a reproducible graph.

    Examples
    --------
    >>> import numpy as np
    >>> from rabitqlib import HnswIndex
    >>> x = np.random.default_rng(42).normal(size=(80, 64)).astype(np.float32)
    >>> index = HnswIndex(64, num_clusters=2, seed=7)
    >>> ids = index.add(x, num_threads=1)  # trains on x first
    >>> ids, distances = index.search(x[:2], k=3)
    >>> ids.shape, distances.shape
    ((2, 3), (2, 3))
    """

    def __init__(
        self,
        dim: SupportsIndex,
        *,
        nbits: SupportsIndex = 7,
        num_clusters: SupportsIndex = 16,
        M: SupportsIndex = 16,
        ef_construction: SupportsIndex = 200,
        metric: _Metric = "l2",
        seed: SupportsIndex | None = None,
    ) -> None:
        """An approximate nearest-neighbor index using HNSW.

        Parameters
        ----------
        dim : int
            Original vector dimension, from 64 through 65536.
        nbits : int, optional
            Keyword-only. Total quantization bits, 1 through 9. Default is 7.
        num_clusters : int, optional
            Keyword-only. Number of centroids train, set_centroids or the first add
            provides; the first add caps it at its row count.
            Default is 16.
        M : int, optional
            Keyword-only. Graph connectivity, at least 2; the base layer keeps up to
            2 * M edges. Default is 16.
        ef_construction : int, optional
            Keyword-only. Construction search breadth. Default is 200.
        metric : {'l2', 'ip', 'innerproduct', 'cosine'}, optional
            Keyword-only. Squared L2, inner-product or cosine distance. Default is
            'l2'. 'cosine' scales data and queries to unit length.
        seed : int, optional
            Keyword-only. Graph and rotation seed in [0, 2**32 - 1], also the initial
            cp.seed. None (default) draws one at random; the seed property reports it.

        Notes
        -----
        The first add trains an untrained index on the added vectors. Call train
        on a sample, or set_centroids, first to choose the training data or the
        centroids. Loaded indexes are trained, except empty files saved untrained
        by 0.7. Capacity grows automatically
        on add; reserve can preallocate it. Concurrent reads are supported; an
        overlapping mutation raises RuntimeError.
        Data, centroids and queries must be finite float32 values within the index
        coordinate range; other values raise ValueError.
        With more than one thread, add links points in a nondeterministic order;
        pass num_threads=1 and a seed for a reproducible graph.
        """
    def __repr__(self) -> str:
        """Return a one-line summary of configuration, counts and training state."""

    def __len__(self) -> int:
        """Return the number of live, searchable points."""

    @property
    def dim(self) -> int:
        """Original vector dimension."""

    @property
    def capacity(self) -> int:
        """Current storage capacity, in stored points."""

    @property
    def num_points(self) -> int:
        """Number of live, searchable points."""

    @property
    def num_stored_points(self) -> int:
        """Number of stored points, including removed points."""

    @property
    def is_trained(self) -> bool:
        """Whether centroids are set, by train, set_centroids, the first add or load."""

    @property
    def seed(self) -> int | None:
        """Graph and rotation seed, or None for an index loaded from a file from a release up to 0.8."""

    @property
    def cp(self) -> RaBitQKMeansParameters:
        """Clustering parameters train uses, a RaBitQKMeansParameters. Defaults: niter 10, max_points_per_centroid 256, spherical for inner product and cosine, and seed equal to the index seed."""

    @cp.setter
    def cp(self, value: RaBitQKMeansParameters) -> None: ...
    @property
    def metric(self) -> Literal["l2", "ip", "cosine"]:
        """Canonical metric name: 'l2', 'ip' or 'cosine'."""

    @property
    def nbits(self) -> int:
        """Total storage bits per coordinate."""

    @property
    def M(self) -> int:
        """Graph connectivity; the base layer keeps up to 2 * M edges."""

    @property
    def ef_construction(self) -> int:
        """Construction search breadth."""

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
            cp.num_threads; 0 selects the available CPU count, capped by OMP_NUM_THREADS.

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

    def add(
        self,
        data: npt.ArrayLike,
        *,
        cluster_ids: _IntegerIds | None = None,
        num_threads: SupportsIndex | None = None,
        fast_quantization: bool | np.bool_ = False,
    ) -> npt.NDArray[np.int64]:
        """Add vectors, training the index first if needed.

        Parameters
        ----------
        data : array_like of float, shape (n, dim)
            Vectors in the original, unpadded coordinates. Converted to contiguous
            float32; a copy may be made.
        cluster_ids : array_like of int, shape (n,), optional
            Keyword-only. Existing centroid IDs. None assigns each vector to its
            nearest centroid. Centroids do not move.
        num_threads : int, optional
            Keyword-only. Worker count. None (default) and 0 select the available
            CPU count, capped by OMP_NUM_THREADS; more than one thread links
            nondeterministically, so pass 1, and a seed to the constructor, for a
            reproducible graph.
        fast_quantization : bool, optional
            Keyword-only. Use faster quantizer configuration. Default is False.

        Returns
        -------
        ids : ndarray of int64, shape (n,)
            Assigned point IDs.

        Notes
        -----
        On an untrained index, add first trains num_clusters centroids on data, as
        train(data) does; with fewer rows it trains one centroid per row, and
        num_clusters then reports the row count. An empty first add raises
        RuntimeError. With num_threads None, training
        uses cp.num_threads. Passing cluster_ids to an untrained index raises
        RuntimeError, as does add on an untrained index loaded from a 0.7 file;
        call train first. If the call fails, the index stays untrained and empty. The first
        add into an empty index links vectors with their exact coordinates; later adds
        link through the stored codes, so add most vectors in the first call. Capacity
        grows geometrically when needed. Removed IDs are not reused.
        """
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
        ef_search: SupportsIndex | None = None,
        *,
        num_threads: SupportsIndex | None = None,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
        """Search a batch of query vectors.

        Parameters
        ----------
        queries : array_like of float, shape (n_queries, dim)
            Vectors in the original, unpadded coordinates. Converted to contiguous
            float32; a copy may be made.
        k : int
            Number of neighbors requested, at least 1. It may exceed the number of
            points; slots without a neighbor are padded as described below.
        ef_search : int, optional
            Search breadth, at least 1. None (default) selects max(k, 64); values
            above the stored point count are clamped.
        num_threads : int, optional
            Keyword-only. Worker count. None (default) selects 1; 0 selects the
            available CPU count, capped by OMP_NUM_THREADS.

        Returns
        -------
        ids : ndarray of int64, shape (n_queries, k)
            Point IDs of the selected neighbors.
        distances : ndarray of float32, shape (n_queries, k)
            Lower values are better: squared L2 estimates for l2, 1 - dot for ip and
            1 - cos for cosine. These are distances, not similarity scores.

        Notes
        -----
        Requires a trained or loaded index. Returns (ids, distances), in that
        order. Missing neighbors have ID -1 and distance inf; test
        np.isfinite(distances) to find them. Array shapes
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
        The saved state includes the rotation used to build the index and its seed;
        releases up to 0.8 cannot read the file.
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

    def reserve(self, num_points: SupportsIndex) -> None:
        """Preallocate room for at least num_points stored points.

        Parameters
        ----------
        num_points : int
            Nonnegative number of stored points to make room for, including the
            current ones.

        Notes
        -----
        Capacity only grows: a value at or below capacity changes nothing. add also
        grows the capacity when needed, so reserve is an optimization.
        """

class SymqgIndex:
    """An approximate nearest-neighbor index using SymphonyQG.

    Parameters
    ----------
    dim : int
        Original vector dimension, from 64 through 65536.
    nbits : {4, 8, 32}, optional
        Keyword-only. Storage bits: 32 (default) keeps raw float32 vectors, and 4
        or 8 store centroid-relative RaBitQ codes.
    max_degree : int, optional
        Keyword-only. Graph degree bound; a positive multiple of 32, smaller than
        the number of vectors added. Default is 32.
    ef_construction : int, optional
        Keyword-only. Positive construction search breadth fitting uint32.
        Default is 200.
    init : {'pipnn', 'random'}, optional
        Keyword-only. Graph initialization method. Default is 'pipnn'.
    metric : {'l2', 'ip', 'innerproduct', 'cosine'}, optional
        Keyword-only. Squared L2, inner-product or cosine distance. Default is
        'l2'. 'cosine' scales data and queries to unit length.
    seed : int, optional
        Keyword-only. Rotation and graph-construction seed in [0, 2**32 - 1].
        None (default) draws one at random; the seed property reports it.

    Notes
    -----
    There is nothing to train: train only checks its input, and is_trained is
    always True. add builds the graph from all vectors at once and may be called
    only once; to add more vectors, create a new index. Load an index or add
    vectors before searching or saving. Concurrent reads are supported; an
    overlapping mutation raises RuntimeError.
    Data and queries must be finite float32 values within the index coordinate
    range; other values raise ValueError.

    Examples
    --------
    >>> import numpy as np
    >>> from rabitqlib import SymqgIndex
    >>> x = np.random.default_rng(42).normal(size=(80, 64)).astype(np.float32)
    >>> index = SymqgIndex(64, ef_construction=40, seed=7)
    >>> ids = index.add(x, num_threads=1)
    >>> ids, distances = index.search(x[:2], k=3)
    >>> ids.shape, distances.shape
    ((2, 3), (2, 3))
    """

    def __init__(
        self,
        dim: SupportsIndex,
        *,
        nbits: SupportsIndex = 32,
        max_degree: SupportsIndex = 32,
        ef_construction: SupportsIndex = 200,
        init: _QGInit = "pipnn",
        metric: _Metric = "l2",
        seed: SupportsIndex | None = None,
    ) -> None:
        """An approximate nearest-neighbor index using SymphonyQG.

        Parameters
        ----------
        dim : int
            Original vector dimension, from 64 through 65536.
        nbits : {4, 8, 32}, optional
            Keyword-only. Storage bits: 32 (default) keeps raw float32 vectors, and 4
            or 8 store centroid-relative RaBitQ codes.
        max_degree : int, optional
            Keyword-only. Graph degree bound; a positive multiple of 32, smaller than
            the number of vectors added. Default is 32.
        ef_construction : int, optional
            Keyword-only. Positive construction search breadth fitting uint32.
            Default is 200.
        init : {'pipnn', 'random'}, optional
            Keyword-only. Graph initialization method. Default is 'pipnn'.
        metric : {'l2', 'ip', 'innerproduct', 'cosine'}, optional
            Keyword-only. Squared L2, inner-product or cosine distance. Default is
            'l2'. 'cosine' scales data and queries to unit length.
        seed : int, optional
            Keyword-only. Rotation and graph-construction seed in [0, 2**32 - 1].
            None (default) draws one at random; the seed property reports it.

        Notes
        -----
        There is nothing to train: train only checks its input, and is_trained is
        always True. add builds the graph from all vectors at once and may be called
        only once; to add more vectors, create a new index. Load an index or add
        vectors before searching or saving. Concurrent reads are supported; an
        overlapping mutation raises RuntimeError.
        Data and queries must be finite float32 values within the index coordinate
        range; other values raise ValueError.
        """
    def __repr__(self) -> str:
        """Return a one-line summary of configuration and counts."""

    def __len__(self) -> int:
        """Return the number of live, searchable points."""

    @property
    def dim(self) -> int:
        """Original vector dimension."""

    @property
    def num_points(self) -> int:
        """Number of live, searchable points."""

    @property
    def num_stored_points(self) -> int:
        """Number of stored points, including removed points."""

    @property
    def is_trained(self) -> bool:
        """Always True: SymphonyQG needs no training."""

    @property
    def metric(self) -> Literal["l2", "ip", "cosine"]:
        """Canonical metric name: 'l2', 'ip' or 'cosine'."""

    @property
    def max_degree(self) -> int:
        """Graph degree bound."""

    @property
    def ef_construction(self) -> int | None:
        """Construction search breadth of the current graph or the next add, or None for an index loaded from a file from a release up to 0.8."""

    @property
    def init(self) -> _QGInit | None:
        """Graph initialization method of the current graph or the next add, or None for an index loaded from a file from a release up to 0.8."""

    @property
    def nbits(self) -> int:
        """Storage bits: 32 for raw float32, or 4 or 8 for RaBitQ."""

    @property
    def seed(self) -> int | None:
        """Rotation and graph-construction seed, or None for an index loaded from a file from a release up to 0.8."""

    def train(
        self, x: npt.ArrayLike, *, num_threads: SupportsIndex | None = None
    ) -> None:
        """Check training vectors; SymphonyQG has nothing to train.

        Parameters
        ----------
        x : array_like of float, shape (n, dim)
            Vectors in the original, unpadded coordinates, checked like the data of
            add.
        num_threads : int, optional
            Keyword-only. Accepted for interface consistency with the other indexes.
            Default is None.

        Notes
        -----
        Provided so that SymphonyQG follows the same train/add/search workflow as
        the other indexes. It does not change the index, so it may run alongside
        searches.
        """

    def add(
        self, data: npt.ArrayLike, *, num_threads: SupportsIndex | None = None
    ) -> npt.NDArray[np.int64]:
        """Build the graph from all vectors.

        Parameters
        ----------
        data : array_like of float, shape (n, dim)
            Vectors in the original, unpadded coordinates, with n > max_degree.
            Converted to contiguous float32; a copy may be made.
        num_threads : int, optional
            Keyword-only. Worker count. None (default) and 0 select the available
            CPU count, capped by OMP_NUM_THREADS.

        Returns
        -------
        ids : ndarray of int64, shape (n,)
            Point IDs, the row numbers of data.

        Notes
        -----
        Uses the constructor's ef_construction, init and seed. The graph cannot be
        extended: a second call raises RuntimeError, so add every vector in one
        call, or create a new index to rebuild. Point IDs are the row numbers of
        data. Input arrays need not be retained after this synchronous call.
        """

    def search(
        self,
        queries: npt.ArrayLike,
        k: SupportsIndex,
        ef_search: SupportsIndex | None = None,
        *,
        num_threads: SupportsIndex | None = None,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
        """Search a batch of query vectors.

        Parameters
        ----------
        queries : array_like of float, shape (n_queries, dim)
            Vectors in the original, unpadded coordinates. Converted to contiguous
            float32; a copy may be made.
        k : int
            Number of neighbors requested, at least 1. It may exceed the number of
            points; slots without a neighbor are padded as described below.
        ef_search : int, optional
            Search breadth, at least 1. None (default) selects max(k, 64); smaller
            values are raised to k.
        num_threads : int, optional
            Keyword-only. Worker count. None (default) selects 1; 0 selects the
            available CPU count, capped by OMP_NUM_THREADS.

        Returns
        -------
        ids : ndarray of int64, shape (n_queries, k)
            Point IDs of the selected neighbors.
        distances : ndarray of float32, shape (n_queries, k)
            Lower values are better: squared L2 estimates for l2, 1 - dot for ip and
            1 - cos for cosine. These are distances, not similarity scores.

        Notes
        -----
        Returns (ids, distances), in that order. When the graph reaches fewer than k
        points, the remaining neighbors have ID -1 and distance inf; test
        np.isfinite(distances) to find them. Array shapes and value ranges are
        checked at runtime, not by static type checkers.
        """

    def save(self, path: _Path) -> None:
        """Save an index with vectors to disk.

        Parameters
        ----------
        path : str, bytes, bytearray, or os.PathLike
            Index file path. PathLike objects are resolved via __fspath__.
            Byte paths use native bytes on POSIX and UTF-8 on Windows.

        Notes
        -----
        The saved state includes the rotation used to build the index and its seed;
        releases up to 0.8 cannot read the file.
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
            An index ready for search; add raises because it already has a graph.
        """

class QGKMeans:
    """Cluster vectors using SymphonyQG graph assignment.

    Parameters
    ----------
    d : int
        Original vector dimension, from 64 through 65536.
    k : int
        Number of centroids; greater than max_degree and no more than the
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
        Worker count; 0 selects the available CPU count,
        capped by OMP_NUM_THREADS. Default is 0.
    final_assignment : {'approximate', 'exact'}, optional
        'approximate' uses the training assignment method; 'exact' uses exhaustive
        assignment against final centroids. Default is 'approximate'.
    max_degree : int, optional
        Centroid graph degree; a positive multiple of 32. Default is 32.
    ef_construction : int, optional
        Positive centroid-graph construction search breadth. Default is 240.
    ef_search : int, optional
        Positive centroid-graph assignment search breadth. Default is 16.
    graph_build_iterations : int, optional
        Positive number of graph-building passes after initialization. Default
        is 1.
    nbits : int, optional
        Centroid storage: 32 for raw float32, or 4 or 8 for RaBitQ. Default is
        32.

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
        final_assignment: _FinalAssignment = "approximate",
        max_degree: SupportsIndex = 32,
        ef_construction: SupportsIndex = 240,
        ef_search: SupportsIndex = 16,
        graph_build_iterations: SupportsIndex = 1,
        nbits: SupportsIndex = 32,
    ) -> None:
        """Cluster vectors using SymphonyQG graph assignment.

        Parameters
        ----------
        d : int
            Original vector dimension, from 64 through 65536.
        k : int
            Number of centroids; greater than max_degree and no more than the
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
            Worker count; 0 selects the available CPU count,
            capped by OMP_NUM_THREADS. Default is 0.
        final_assignment : {'approximate', 'exact'}, optional
            'approximate' uses the training assignment method; 'exact' uses exhaustive
            assignment against final centroids. Default is 'approximate'.
        max_degree : int, optional
            Centroid graph degree; a positive multiple of 32. Default is 32.
        ef_construction : int, optional
            Positive centroid-graph construction search breadth. Default is 240.
        ef_search : int, optional
            Positive centroid-graph assignment search breadth. Default is 16.
        graph_build_iterations : int, optional
            Positive number of graph-building passes after initialization. Default
            is 1.
        nbits : int, optional
            Centroid storage: 32 for raw float32, or 4 or 8 for RaBitQ. Default is
            32.

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
    def iteration_stats(self) -> list[KMeansIterationStats] | None:
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
        """Worker count; 0 selects the available CPU count, capped by OMP_NUM_THREADS. Default is 0."""
    @num_threads.setter
    def num_threads(self, value: SupportsIndex) -> None: ...
    @property
    def final_assignment(self) -> _FinalAssignment:
        """'approximate' uses the training assignment method; 'exact' uses exhaustive assignment against final centroids. Default is 'approximate'."""
    @final_assignment.setter
    def final_assignment(self, value: _FinalAssignment) -> None: ...
    @property
    def max_degree(self) -> int:
        """Centroid graph degree; a positive multiple of 32. Default is 32."""
    @max_degree.setter
    def max_degree(self, value: SupportsIndex) -> None: ...
    @property
    def ef_construction(self) -> int:
        """Positive centroid-graph construction search breadth. Default is 240."""
    @ef_construction.setter
    def ef_construction(self, value: SupportsIndex) -> None: ...
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
    def nbits(self) -> int:
        """Centroid storage: 32 for raw float32, or 4 or 8 for RaBitQ. Default is 32."""
    @nbits.setter
    def nbits(self, value: SupportsIndex) -> None: ...
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
        Worker count; 0 selects the available CPU count,
        capped by OMP_NUM_THREADS. Default is 0.
    final_assignment : {'approximate', 'exact'}, optional
        'approximate' uses the training assignment method; 'exact' uses exhaustive
        assignment against final centroids. Default is 'approximate'.

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
        final_assignment: _FinalAssignment = "approximate",
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
            Worker count; 0 selects the available CPU count,
            capped by OMP_NUM_THREADS. Default is 0.
        final_assignment : {'approximate', 'exact'}, optional
            'approximate' uses the training assignment method; 'exact' uses exhaustive
            assignment against final centroids. Default is 'approximate'.

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
    def iteration_stats(self) -> list[KMeansIterationStats] | None:
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
        """Worker count; 0 selects the available CPU count, capped by OMP_NUM_THREADS. Default is 0."""
    @num_threads.setter
    def num_threads(self, value: SupportsIndex) -> None: ...
    @property
    def final_assignment(self) -> _FinalAssignment:
        """'approximate' uses the training assignment method; 'exact' uses exhaustive assignment against final centroids. Default is 'approximate'."""
    @final_assignment.setter
    def final_assignment(self, value: _FinalAssignment) -> None: ...
    def __init__(self) -> None:
        """Create parameters with their documented defaults."""
