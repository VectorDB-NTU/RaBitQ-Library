"""Python bindings for RaBitQLib indexes and clustering.

The installed ``rabitqlib`` package loads its compiled ``._rabitqlib`` extension
and adds the clustering wrappers. Import public classes from ``rabitqlib``.
"""

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
