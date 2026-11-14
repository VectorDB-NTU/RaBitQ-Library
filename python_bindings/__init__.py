"""Python bindings for RaBitQLib indexes and clustering.

The installed ``rabitqlib`` package loads its compiled ``._rabitqlib`` extension
and adds the clustering wrappers. Import public classes from ``rabitqlib``.
"""

import inspect
import operator
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
from ._rabitqlib import (
    __version__ as __version__,
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


def _is_zero(value):
    """Whether an optional integer argument is one the extension reads as 0."""
    if value is None or isinstance(value, (float, str, bytes)):
        return False
    try:
        return operator.index(value) == 0
    except TypeError:
        pass
    try:
        # pybind11 also converts objects that only define __int__, such as
        # numpy floating scalars.
        return int(value) == 0
    except (TypeError, ValueError, OverflowError):
        return False


def _exceeds_32_bits(value):
    try:
        return operator.index(value) > 0xFFFFFFFF
    except TypeError:
        return False


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
    self, dim, num_clusters, *, nbits=7, metric="l2", routing="auto", seed=None
): ...


def _ivf_legacy_signature(
    self, dim, max_elements, num_clusters, nbits, metric="l2", initializer="auto"
): ...


def _hnsw_signature(
    self,
    dim,
    *,
    nbits=7,
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
    self, data, *, cluster_ids=None, num_threads=None, fast_quantization=False
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
                "use IvfIndex(dim, num_clusters, nbits=...), whose first add() trains it"
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
        # 0.9 names the routing strategy routing; initializer is its 0.8 spelling.
        initializer = kwargs.pop("initializer", _MISSING)
        if initializer is not _MISSING:
            if "routing" in kwargs:
                raise TypeError(
                    "IvfIndex() got both routing and its old name initializer"
                )
            _deprecated("IvfIndex(initializer=...) is deprecated; use routing=")
            kwargs["routing"] = initializer
        arguments = _bind(signature, "IvfIndex", (self, *args), kwargs)
        arguments["initializer"] = arguments.pop("routing")
        try:
            native_init(self, **arguments)
        except ValueError as error:
            # The extension names the argument initializer; a caller who wrote
            # routing= gets that name back.
            message = str(error)
            if initializer is not _MISSING or not message.startswith(
                "initializer must be"
            ):
                raise
            raise ValueError("routing" + message[len("initializer") :]) from None

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
        arguments = _bind(signature, "HnswIndex", (self, *args), kwargs)
        if _exceeds_32_bits(arguments["seed"]):
            _deprecated(
                "HnswIndex seeds above 2**32 - 1 are deprecated; 0.9 accepts "
                "[0, 2**32 - 1], as the other indexes do"
            )
        native_init(self, **arguments)

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
            Worker count. Default is 1; 0 selects the available CPU count,
            capped by OMP_NUM_THREADS.
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


def _symqg_signature(
    self,
    dim,
    *,
    nbits=32,
    max_degree=32,
    ef_construction=200,
    init="pipnn",
    metric="l2",
    seed=None,
): ...


def _symqg_legacy_signature(
    self, dim, max_degree, metric="l2", quantization_bits=0
): ...


def _symqg_nbits(quantization_bits):
    # 0.7 stored raw vectors as 0 bits; 0.8 calls that 32.
    if isinstance(quantization_bits, bool) or quantization_bits not in (0, 4, 8):
        raise ValueError("quantization_bits must be 0, 4, or 8")
    return 32 if quantization_bits == 0 else quantization_bits


def _patch_symqg(cls):
    native_init = cls.__init__
    native_search = cls.search
    signature = inspect.signature(_symqg_signature)
    legacy_signature = inspect.signature(_symqg_legacy_signature)

    def __init__(self, *args, **kwargs):
        if len(args) > 1:
            arguments = _bind(
                legacy_signature,
                "SymqgIndex",
                (self, *args),
                kwargs,
                " (read as the 0.7 form SymqgIndex(dim, max_degree, metric, "
                "quantization_bits); the 0.8 form is SymqgIndex(dim, max_degree=...))",
            )
            _deprecated(
                "SymqgIndex(dim, max_degree, metric, quantization_bits) is deprecated; "
                "use SymqgIndex(dim, max_degree=..., metric=..., nbits=...)"
            )
            nbits = _symqg_nbits(arguments["quantization_bits"])
            native_init(
                self,
                arguments["dim"],
                nbits=nbits,
                max_degree=arguments["max_degree"],
                metric=arguments["metric"],
            )
            return
        quantization_bits = kwargs.pop("quantization_bits", _MISSING)
        if quantization_bits is not _MISSING:
            if "nbits" in kwargs:
                raise TypeError(
                    "SymqgIndex() got both nbits and its old form quantization_bits"
                )
            _deprecated(
                "SymqgIndex(quantization_bits=...) is deprecated; use nbits=, with 32 "
                "instead of 0 for raw vectors"
            )
            kwargs["nbits"] = _symqg_nbits(quantization_bits)
        native_init(self, **_bind(signature, "SymqgIndex", (self, *args), kwargs))

    def build(self, data, ef_construction, num_threads=1, init="pipnn"):
        """Build the graph from input vectors, replacing any existing graph.

        Deprecated since 0.8 and removed in 0.9: pass ef_construction and init to
        the constructor, then call add() once. The defaults match 0.7, including
        a single thread.

        Parameters
        ----------
        data : array_like of float, shape (n, dim)
            Vectors in the original, unpadded coordinates, with n > max_degree.
            Converted to contiguous float32; a copy may be made.
        ef_construction : int
            Positive construction search breadth fitting uint32.
        num_threads : int, optional
            Worker count. Default is 1; 0 selects the available CPU count,
            capped by OMP_NUM_THREADS.
        init : {'pipnn', 'random'}, optional
            Graph initialization method. Default is 'pipnn'.
        """
        _deprecated(
            "SymqgIndex.build() is deprecated; pass ef_construction and init to "
            "SymqgIndex() and call add()"
        )
        self._build(data, ef_construction, num_threads, init)

    def search_batch(self, queries, k, ef, num_threads=1):
        """Deprecated since 0.8: the same as search(); use search()."""
        _deprecated("SymqgIndex.search_batch() is deprecated; use search()")
        return native_search(self, queries, k, ef, num_threads)

    def quantization_bits(self):
        """Deprecated since 0.8: storage bits with 0 for raw vectors; use nbits."""
        _deprecated(
            "SymqgIndex.quantization_bits is deprecated; use nbits, which is 32 "
            "instead of 0 for raw vectors"
        )
        return self._quantization_bits

    def is_built(self):
        """Deprecated since 0.8: whether the index holds a graph; use len()."""
        _deprecated("SymqgIndex.is_built is deprecated; use len()")
        return self.num_stored_points != 0

    __init__.__doc__ = cls.__doc__
    __init__.__signature__ = signature
    cls.__init__ = __init__
    cls.build = build
    cls.search_batch = search_batch
    cls.quantization_bits = property(quantization_bits, doc=quantization_bits.__doc__)
    cls.is_built = property(is_built, doc=is_built.__doc__)


_patch_ivf_init(IvfIndex)
_patch_hnsw_init(HnswIndex)
for _cls in (IvfIndex, HnswIndex):
    _patch_add(_cls)
    _patch_build(_cls)
del _cls
_patch_symqg(SymqgIndex)
# --- end of the 0.8 compatibility layer -------------------------------------


# --- 0.9 names; the 0.8 spellings they replace are removed in 0.9 -----------
# 0.9 renames these members and arguments (issue #232). This release accepts both
# spellings and warns on the old one. The extension classes still carry the 0.8
# names, so the new ones are added here, again by replacing members on the classes.


def _member(cls, function, name=None):
    """function named as a member of cls, so errors and help() show the class."""
    if name is not None:
        function.__name__ = name
    function.__qualname__ = f"{cls.__name__}.{function.__name__}"
    function.__module__ = cls.__module__
    return function


def _renamed_property(cls, old, new, new_doc, replacement):
    """Add property new for the native property old, which then warns."""
    native = vars(cls)[old]
    name = cls.__name__

    def current(self):
        return native.__get__(self, cls)

    def former(self):
        _deprecated(f"{name}.{old} is deprecated; use {replacement}")
        return native.__get__(self, cls)

    _member(cls, former, old)
    if new is not None:
        _member(cls, current, new)
        _set_property(cls, new, current, new_doc)
    _set_property(cls, old, former, f"Deprecated since 0.8.5: use {replacement}.")


def _set_property(cls, name, getter, documentation):
    """Assign a read-only property, named as a class body would name it."""
    member = property(getter, doc=documentation)
    setattr(cls, name, member)
    # Assignment after class creation skips __set_name__, and before Python 3.13
    # a property without it reports errors without its name.
    member.__set_name__(cls, name)


def _options_from_positions(name, options, args, kwargs):
    if len(args) > len(options):
        raise TypeError(
            f"{name}.search() takes at most {len(options) + 2} positional "
            f"arguments ({len(args) + 2} given)"
        )
    for option, value in zip(options, args, strict=False):
        if option in kwargs:
            raise TypeError(
                f"{name}.search() got multiple values for argument {option!r}"
            )
        kwargs[option] = value


def _edited(documentation, replacements):
    """A docstring with replacements applied; fails if the native text moved."""
    for old, new in replacements:
        if old not in documentation:
            raise AssertionError(f"native documentation changed: {old!r}")
        documentation = documentation.replace(old, new)
    return documentation


def _redocumented(native, replacements):
    """The native docstring without pybind11's signature line, with edits applied."""
    return _edited(native.__doc__.split("\n\n", 1)[1], replacements)


_THREADS_DOC = (
    "    Worker count. None (default) selects 1; 0 selects the available CPU\n"
    "    count, capped by OMP_NUM_THREADS.",
    "    Keyword-only; passing it by position is deprecated since 0.8.5. Worker\n"
    "    count. None (default) selects 1; 0 selects the available CPU count,\n"
    "    capped by OMP_NUM_THREADS.",
)


def _ivf_search_signature(
    self, queries, k, nprobe=None, *, high_accuracy=None, num_threads=None
): ...


def _graph_search_signature(self, queries, k, ef_search=None, *, num_threads=None): ...


def _patch_ivf_names(cls):
    native_reserve = cls.reserve
    native_search = cls.search
    options = ("nprobe", "high_accuracy", "num_threads")

    def reserve_cluster(self, cluster_id, num_points):
        """Preallocate storage for a cluster.

        Parameters
        ----------
        cluster_id : int
            Cluster ID in [0, num_clusters).
        num_points : int
            Nonnegative capacity to reserve for this cluster: the total number
            of points it can hold, including its current ones.

        Notes
        -----
        Requires a trained or loaded index. Reserved room is not saved. Named
        reserve before 0.8.5.
        """
        try:
            native_reserve(self, cluster_id, num_points)
        except RuntimeError as error:
            # The extension still names the operation reserve.
            if not str(error).endswith("before reserve"):
                raise
            raise RuntimeError(f"{error}_cluster") from None

    def reserve(self, cluster_id, num_points):
        """Deprecated since 0.8.5: the same as reserve_cluster(); use reserve_cluster()."""
        _deprecated("IvfIndex.reserve() is deprecated; use reserve_cluster()")
        native_reserve(self, cluster_id, num_points)

    def search(self, queries, k, *args, **kwargs):
        _options_from_positions("IvfIndex", options, args, kwargs)
        if len(args) > 1:
            _deprecated(
                f"passing {', '.join(options[1 : len(args)])} to IvfIndex.search() "
                "by position is deprecated; use keywords"
            )
        if _is_zero(kwargs.get("nprobe")):
            _deprecated(
                "IvfIndex.search(nprobe=0) is deprecated; pass None, or omit nprobe, "
                "for the default"
            )
        return native_search(self, queries, k, **kwargs)

    search.__doc__ = _redocumented(
        native_search,
        [
            (
                "    Number of clusters to probe, capped at num_clusters. None (default) and\n"
                "    0 select num_clusters / 16, rounded to the nearest integer and at least 1.",
                "    Number of clusters to probe, capped at num_clusters. None (default)\n"
                "    selects num_clusters / 16, rounded to the nearest integer and at least\n"
                "    1; 0 does the same and is deprecated since 0.8.5.",
            ),
            (
                "    Enable HACC. None (default) selects it for 4-9 bits and standard\n"
                "    FastScan otherwise. Raw storage uses standard FastScan.",
                "    Keyword-only; passing it by position is deprecated since 0.8.5.\n"
                "    Enable HACC. None (default) selects it for 4-9 bits and standard\n"
                "    FastScan otherwise. Raw storage uses standard FastScan.",
            ),
            _THREADS_DOC,
        ],
    )
    search.__signature__ = inspect.signature(_ivf_search_signature)
    cls.__doc__ = _edited(
        cls.__doc__,
        [
            (
                "initializer : {'auto', 'flat', 'flat_rabitq', 'hnsw'}, optional\n"
                "    Centroid routing strategy. Default is 'auto'.",
                "routing : {'auto', 'flat', 'flat_rabitq', 'hnsw'}, optional\n"
                "    Centroid routing strategy. Default is 'auto'. Named initializer\n"
                "    before 0.8.5.",
            )
        ],
    )
    cls.__init__.__doc__ = cls.__doc__
    cls.add.__doc__ = _edited(
        cls.add.__doc__,
        [
            ("reserve can preallocate room.", "reserve_cluster can preallocate room."),
            (
                "index's configured centroid initializer; routing may be approximate.",
                "index's configured routing strategy, which may be approximate.",
            ),
        ],
    )
    cls.reserve_cluster = _member(cls, reserve_cluster)
    cls.reserve = _member(cls, reserve)
    cls.search = _member(cls, search)
    _renamed_property(
        cls,
        "initializer",
        "routing",
        "Resolved centroid routing strategy. Named initializer before 0.8.5.",
        "routing",
    )
    _renamed_property(cls, "max_elements", None, None, "num_stored_points")


def _patch_graph_search(cls, breadth_doc):
    native_search = cls.search
    name = cls.__name__
    options = ("ef_search", "num_threads")

    def search(self, queries, k, *args, **kwargs):
        _options_from_positions(name, options, args, kwargs)
        if "ef" in kwargs and "ef_search" in kwargs:
            raise TypeError(f"{name}.search() got both ef_search and its old name ef")
        if len(args) == len(options):
            _deprecated(
                f"passing num_threads to {name}.search() by position is deprecated; "
                "use the keyword"
            )
        if "ef" in kwargs:
            _deprecated(f"{name}.search(ef=...) is deprecated; use ef_search=")
            kwargs["ef_search"] = kwargs.pop("ef")
        ef_search = kwargs.pop("ef_search", None)
        if _is_zero(ef_search):
            _deprecated(
                f"{name}.search(ef_search=0) is deprecated; pass None, or omit "
                "ef_search, for the default"
            )
        return native_search(self, queries, k, ef_search, **kwargs)

    search.__doc__ = _redocumented(native_search, [breadth_doc, _THREADS_DOC])
    search.__signature__ = inspect.signature(_graph_search_signature)
    cls.search = _member(cls, search)


def _patch_hnsw_names(cls):
    native_resize = cls.resize
    native_max_elements = vars(cls)["max_elements"]

    def reserve(self, num_points):
        """Preallocate room for at least num_points stored points.

        Parameters
        ----------
        num_points : int
            Nonnegative number of stored points to make room for, including the
            current ones.

        Notes
        -----
        Capacity only grows: a value at or below capacity changes nothing. add
        also grows the capacity when needed, so reserve is an optimization.
        """
        needed = operator.index(num_points)
        if needed < 0:
            raise ValueError("num_points must be non-negative")
        if needed > native_max_elements.__get__(self, cls):
            native_resize(self, needed)

    def resize(self, max_elements):
        """Deprecated since 0.8.5: change the capacity; use reserve(), which only grows it."""
        _deprecated(
            "HnswIndex.resize() is deprecated; use reserve(), which only grows the "
            "capacity"
        )
        native_resize(self, max_elements)

    cls.__doc__ = _edited(
        cls.__doc__,
        [
            ("resize can preallocate it.", "reserve can preallocate it."),
            (
                "32 bits). Default is 100.",
                "32 bits), in [0, 2**32 - 1]; larger values are deprecated since\n"
                "    0.8.5. Default is 100.",
            ),
        ],
    )
    cls.__init__.__doc__ = cls.__doc__
    cls.reserve = _member(cls, reserve)
    cls.resize = _member(cls, resize)
    _renamed_property(
        cls,
        "max_elements",
        "capacity",
        "Current storage capacity, in stored points. Named max_elements before 0.8.5.",
        "capacity",
    )


_patch_ivf_names(IvfIndex)
_patch_hnsw_names(HnswIndex)
_renamed_property(SymqgIndex, "max_elements", None, None, "num_stored_points")
_patch_graph_search(
    HnswIndex,
    (
        "ef : int, optional\n    Search breadth. None or 0 (default) selects max(k, 64); values above the stored\n"
        "    point count are clamped.",
        "ef_search : int, optional\n    Search breadth. None (default) selects max(k, 64); values above the stored\n"
        "    point count are clamped. 0 selects the default and is deprecated since\n"
        "    0.8.5. Named ef before 0.8.5.",
    ),
)
_patch_graph_search(
    SymqgIndex,
    (
        "ef : int, optional\n    Search breadth. None or 0 (default) selects max(k, 64); smaller values\n"
        "    are raised to k.",
        "ef_search : int, optional\n    Search breadth. None (default) selects max(k, 64); smaller values\n"
        "    are raised to k. 0 selects the default and is deprecated since 0.8.5.\n"
        "    Named ef before 0.8.5.",
    ),
)
# --- end of the 0.9 names ----------------------------------------------------


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
        Worker count; 0 selects the available CPU count,
        capped by OMP_NUM_THREADS. Default is 0.
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
        Worker count; 0 selects the available CPU count,
        capped by OMP_NUM_THREADS. Default is 0.
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
