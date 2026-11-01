"""Python bindings for RaBitQLib indexes and clustering.

The installed ``rabitqlib`` package loads its compiled ``._rabitqlib`` extension
and adds the clustering wrappers. Import public classes from ``rabitqlib``.
"""

import inspect
import warnings

from ._rabitqlib import (
    FinalAssignmentMode,
    HnswIndex,
    IvfIndex,
    QGKMeansIterationStats,
    QGKMeansParameters,
    RaBitQKMeansIterationStats,
    RaBitQKMeansParameters,
    SymqgIndex,
    _QGKMeans,
    _RaBitQKMeans,
)

# --- 0.8 compatibility layer; remove in 0.9 --------------------------------
# The extension classes expose only the 0.8 API. As Faiss does for its SWIG
# classes, the 0.7 call forms are restored by replacing methods on those classes
# at import time, so the public types stay the native ones.

_MIGRATION = (
    "see https://vectordb-ntu.github.io/RaBitQ-Library/python_migration/; "
    "the old form will be removed in 0.9"
)
_MISSING = object()


def _deprecated(message):
    # Called from a replacement method, so the caller of that method is two up.
    warnings.warn(f"{message}; {_MIGRATION}", DeprecationWarning, stacklevel=3)


def _bind(signature, name, args, kwargs, hint=""):
    try:
        bound = signature.bind(*args, **kwargs)
    except TypeError as error:
        raise TypeError(f"{name}() {error}{hint}") from None
    bound.apply_defaults()
    arguments = dict(bound.arguments)
    del arguments["self"]
    return arguments


def _ivf_signature(
    self, dim, num_clusters, *, nbits, metric="l2", initializer="auto", seed=None
): ...


def _ivf_legacy_signature(
    self, dim, max_elements, num_clusters, nbits, metric="l2", initializer="auto"
): ...


def _hnsw_signature(
    self,
    dim,
    *,
    nbits,
    num_clusters=16,
    M=16,
    ef_construction=200,
    metric="l2",
    seed=100,
): ...


def _hnsw_legacy_signature(
    self,
    dim,
    max_elements,
    M=16,
    ef_construction=200,
    nbits=8,
    metric="l2",
    random_seed=100,
): ...


def _add_signature(
    self, data, *, cluster_ids=None, num_threads=0, fast_quantization=False
): ...


def _patch_ivf_init(cls):
    native_init = cls.__init__
    signature = inspect.signature(_ivf_signature)
    legacy_signature = inspect.signature(_ivf_legacy_signature)

    def __init__(self, *args, **kwargs):
        # A third positional argument, max_elements, or num_clusters by keyword
        # after two positional arguments can only be the 0.7 form.
        if (
            len(args) > 2
            or "max_elements" in kwargs
            or (len(args) == 2 and "num_clusters" in kwargs)
        ):
            arguments = _bind(
                legacy_signature,
                "IvfIndex",
                (self, *args),
                kwargs,
                " (read as the 0.7 form IvfIndex(dim, max_elements, num_clusters, "
                "nbits, ...); the 0.8 form is IvfIndex(dim, num_clusters, nbits=...))",
            )
            _deprecated(
                "IvfIndex(dim, max_elements, num_clusters, nbits, ...) is deprecated; "
                "use IvfIndex(dim, num_clusters, nbits=...) and train() or "
                "set_centroids() before add()"
            )
            native_init(
                self,
                arguments["dim"],
                arguments["num_clusters"],
                nbits=arguments["nbits"],
                metric=arguments["metric"],
                initializer=arguments["initializer"],
                _max_elements=arguments["max_elements"],
            )
            return
        native_init(self, **_bind(signature, "IvfIndex", (self, *args), kwargs))

    __init__.__doc__ = cls.__doc__
    __init__.__signature__ = signature
    cls.__init__ = __init__


def _patch_hnsw_init(cls):
    native_init = cls.__init__
    signature = inspect.signature(_hnsw_signature)
    legacy_signature = inspect.signature(_hnsw_legacy_signature)

    def __init__(self, *args, **kwargs):
        if len(args) > 1 or "max_elements" in kwargs:
            arguments = _bind(
                legacy_signature,
                "HnswIndex",
                (self, *args),
                kwargs,
                " (read as the 0.7 form HnswIndex(dim, max_elements, ...); "
                "the 0.8 form is HnswIndex(dim, nbits=...))",
            )
            _deprecated(
                "HnswIndex(dim, max_elements, ...) is deprecated; use "
                "HnswIndex(dim, nbits=...), whose capacity grows on add"
            )
            native_init(
                self,
                arguments["dim"],
                _max_elements=arguments["max_elements"],
                M=arguments["M"],
                ef_construction=arguments["ef_construction"],
                nbits=arguments["nbits"],
                metric=arguments["metric"],
                random_seed=arguments["random_seed"],
            )
            return
        random_seed = kwargs.pop("random_seed", _MISSING)
        if random_seed is not _MISSING:
            if "seed" in kwargs:
                raise TypeError(
                    "HnswIndex() got both seed and its old name random_seed"
                )
            _deprecated("HnswIndex(random_seed=...) is deprecated; use seed=")
            kwargs["seed"] = random_seed
        native_init(self, **_bind(signature, "HnswIndex", (self, *args), kwargs))

    __init__.__doc__ = cls.__doc__
    __init__.__signature__ = signature
    cls.__init__ = __init__


def _patch_add(cls):
    native_add = cls.add
    name = cls.__name__
    options = ("cluster_ids", "num_threads", "fast_quantization")

    def add(self, data, *args, **kwargs):
        if args:
            if len(args) > len(options):
                raise TypeError(
                    f"{name}.add() takes at most {len(options) + 1} positional "
                    f"arguments ({len(args) + 1} given)"
                )
            _deprecated(
                f"passing {', '.join(options[: len(args)])} to {name}.add() by "
                "position is deprecated; pass them by keyword"
            )
            for option, value in zip(options, args, strict=False):
                if option in kwargs:
                    raise TypeError(
                        f"{name}.add() got multiple values for argument {option!r}"
                    )
                kwargs[option] = value
        return native_add(self, data, **kwargs)

    add.__doc__ = native_add.__doc__
    add.__signature__ = inspect.signature(_add_signature)
    cls.add = add


def _patch_build(cls):
    name = cls.__name__

    def build(
        self, data, centroids, cluster_ids, num_threads=1, fast_quantization=False
    ):
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
        _deprecated(
            f"{name}.build() is deprecated; use set_centroids() or train(), then add()"
        )
        self._build(data, centroids, cluster_ids, num_threads, fast_quantization)

    def is_built(self):
        """Deprecated since 0.8: whether the index stores points; use len()."""
        _deprecated(f"{name}.is_built is deprecated; use is_trained or len()")
        return self.num_stored_points != 0

    cls.build = build
    cls.is_built = property(is_built, doc=is_built.__doc__)


_patch_ivf_init(IvfIndex)
_patch_hnsw_init(HnswIndex)
for _cls in (IvfIndex, HnswIndex):
    _patch_add(_cls)
    _patch_build(_cls)
del _cls
# --- end of the 0.8 compatibility layer -------------------------------------


class _KMeans:
    """Shared result handling for the two clustering methods."""

    def __init__(self, d, k, **kwargs):
        self.d = int(d)
        self.k = int(k)
        self.cp = self._parameters_type()
        for name, value in kwargs.items():
            try:
                getattr(self.cp, name)
            except AttributeError:
                raise AttributeError(
                    f"{type(self).__name__}() got an unexpected keyword argument {name!r}"
                ) from None
            try:
                setattr(self.cp, name, value)
            except TypeError as error:
                raise TypeError(
                    f"{type(self).__name__}() could not convert parameter {name!r} "
                    f"from {type(value).__name__}"
                ) from error
        self._impl = None

    def train(self, x):
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
        implementation = self._implementation_type(self.d, self.k, self.cp)
        objective = implementation.train(x)
        self._impl = implementation
        return objective

    def assign(self, x):
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
        if self._impl is None:
            raise RuntimeError(f"{type(self).__name__} has not been trained")
        return self._impl.assign(x)

    @property
    def centroids(self):
        """Final centroids, float32 array of shape (k, d), or None before training."""
        return None if self._impl is None else self._impl.centroids

    @property
    def assignments(self):
        """Final centroid IDs, uint32 array of shape (n,), or None before training."""
        return None if self._impl is None else self._impl.assignments

    @property
    def distances(self):
        """Final assignment distances for all n vectors, float32 array of shape (n,), or None before training."""
        return None if self._impl is None else self._impl.distances

    @property
    def obj(self):
        """Pre-update objectives over the vectors used for iterations, float64 array of shape (iterations,), or None before training."""
        return None if self._impl is None else self._impl.obj

    @property
    def final_obj(self):
        """Objective for the final assignments of all n vectors, or None before training."""
        return None if self._impl is None else self._impl.final_obj

    @property
    def iteration_stats(self):
        """Per-iteration statistics over the vectors used for iterations, or None before training."""
        return None if self._impl is None else self._impl.iteration_stats


class QGKMeans(_KMeans):
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

    _parameters_type = QGKMeansParameters
    _implementation_type = _QGKMeans


class RaBitQKMeans(_KMeans):
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

    _parameters_type = RaBitQKMeansParameters
    _implementation_type = _RaBitQKMeans


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
