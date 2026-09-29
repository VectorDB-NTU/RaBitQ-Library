"""Python bindings for RaBitQLib indexes and clustering.

This module re-exports symbols from the compiled `rabitqlib` extension
so code can import using `from python_bindings import ...` if needed.
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
            getattr(self.cp, name)
            setattr(self.cp, name, value)
        self._impl = None

    def train(self, x):
        """Train on ``(n, d)`` data and return the last pre-update objective.

        ``final_obj`` measures the returned assignments against the final centroids.
        A failed fit preserves the results of the previous successful fit.
        """
        implementation = self._implementation_type(self.d, self.k, self.cp)
        objective = implementation.train(x)
        self._impl = implementation
        return objective

    @property
    def centroids(self):
        return None if self._impl is None else self._impl.centroids

    @property
    def assignments(self):
        return None if self._impl is None else self._impl.assignments

    @property
    def distances(self):
        return None if self._impl is None else self._impl.distances

    @property
    def obj(self):
        return None if self._impl is None else self._impl.obj

    @property
    def final_obj(self):
        return None if self._impl is None else self._impl.final_obj

    @property
    def iteration_stats(self):
        return None if self._impl is None else self._impl.iteration_stats


class QGKMeans(_KMeans):
    """SymphonyQG graph assignment; recommended for many centroids."""

    _parameters_type = QGKMeansParameters
    _implementation_type = _QGKMeans


class RaBitQKMeans(_KMeans):
    """Flat RaBitQ assignment; recommended for few centroids."""

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
