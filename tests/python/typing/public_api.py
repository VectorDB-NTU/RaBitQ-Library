"""Static-only fixture: assert precise types and reject invalid usage."""

from os import PathLike
from pathlib import Path
from typing import Literal, assert_type

import numpy as np
import numpy.typing as npt
import rabitqlib
from rabitqlib import (
    FinalAssignmentMode,
    HnswIndex,
    IvfIndex,
    QGKMeans,
    QGKMeansIterationStats,
    QGKMeansParameters,
    RaBitQKMeans,
    RaBitQKMeansIterationStats,
    RaBitQKMeansParameters,
    SymqgIndex,
)


def check(x: npt.NDArray[np.float32], labels: npt.NDArray[np.uint32]) -> None:
    assert_type(rabitqlib.__all__, list[str])
    ivf = IvfIndex(64, 2, nbits=4, seed=7)
    hnsw = HnswIndex(64, nbits=4, num_clusters=2)
    graph = SymqgIndex(64, 32)
    for index in (ivf, hnsw, graph):
        assert_type(index.__repr__(), str)
    for index in (ivf, hnsw):
        assert_type(index.cp, RaBitQKMeansParameters)
        index.cp.niter = 10
        index.train(x, num_threads=1)
        index.set_centroids(x[:2])
        assert_type(index.is_trained, bool)
        assert_type(index.seed, int | None)
        assert_type(index.add(x, cluster_ids=labels), npt.NDArray[np.uint32])
        assert_type(index.remove([0, 1]), int)
        assert_type(index.dim, int)
        assert_type(len(index), int)
        index.save("index.bin")
    ids, distances = ivf.search(x, k=5, nprobe=2)
    assert_type(ids, npt.NDArray[np.uint32])
    assert_type(distances, npt.NDArray[np.float32])
    assert_type(
        hnsw.search(x, 5), tuple[npt.NDArray[np.uint32], npt.NDArray[np.float32]]
    )
    graph.build(x, 64)
    assert_type(
        graph.search_batch(x, 5, 32),
        tuple[npt.NDArray[np.uint32], npt.NDArray[np.float32]],
    )
    assert_type(IvfIndex.load("index.bin"), IvfIndex)
    assert_type(HnswIndex.load("index.bin"), HnswIndex)
    assert_type(SymqgIndex.load("index.bin"), SymqgIndex)
    ivf.reserve(0, 128)
    hnsw.resize(256)
    qg = QGKMeans(64, 40, niter=2, quantization_bits=4)
    flat = RaBitQKMeans(64, 2, final_assignment=FinalAssignmentMode.Exact)
    assert_type(qg.cp, QGKMeansParameters)
    assert_type(flat.cp, RaBitQKMeansParameters)
    for model in (qg, flat):
        assert_type(model.train(x), float)
        assert_type(model.centroids, npt.NDArray[np.float32] | None)
        assert_type(model.assignments, npt.NDArray[np.uint32] | None)
        assert_type(model.distances, npt.NDArray[np.float32] | None)
        assert_type(model.obj, npt.NDArray[np.float64] | None)
        assert_type(model.final_obj, float | None)
        assert_type(
            model.assign(x),
            tuple[npt.NDArray[np.uint32], npt.NDArray[np.float32]],
        )
        if model.iteration_stats is not None:
            assert_type(model.iteration_stats[0], QGKMeansIterationStats)
    qg.cp.final_assignment = FinalAssignmentMode.Approximate
    ivf.search(x, "five", 2)  # type: ignore[arg-type]
    ivf.dim = 16  # type: ignore[misc]
    qg.cp.niter = "two"  # type: ignore[assignment]
    RaBitQKMeans(64, 2, graph_degree=32)  # type: ignore[call-arg]
    hnsw.search(x, 5, nprobe=2)  # type: ignore[call-arg]
    IvfIndex(64, 2, 4)  # type: ignore[call-overload]
    hnsw.train(x, 1)  # type: ignore[call-arg]


def stats_alias(stats: RaBitQKMeansIterationStats) -> QGKMeansIterationStats:
    return stats


def check_parameter_reads(
    parameters: QGKMeansParameters | RaBitQKMeansParameters,
) -> None:
    # Read in a separate scope: Pyright narrows union-typed property writes to
    # the assigned NumPy scalar, although the native setter normalizes it.
    assert_type(parameters.niter, int)
    assert_type(parameters.verbose, bool)
    assert_type(parameters.max_points_per_centroid, int)
    assert type(parameters.niter) is int
    assert type(parameters.max_points_per_centroid) is int
    assert type(parameters.verbose) is bool


def check_numpy_scalars() -> None:
    """The same positive cases run through mypy and the compiled bindings."""
    x = np.random.default_rng(42).normal(size=(80, 64)).astype(np.float32)
    labels = np.zeros(80, dtype=np.uint32)
    centroids = x.mean(axis=0, keepdims=True)
    ivf = IvfIndex(np.int64(64), np.int64(1), nbits=np.int64(4), seed=np.uint32(7))
    hnsw = HnswIndex(
        np.int64(64), nbits=np.int64(4), num_clusters=np.int64(1), M=np.int64(16)
    )
    graph = SymqgIndex(np.int64(64), np.int64(32), quantization_bits=np.int64(4))
    for index in (ivf, hnsw):
        index.set_centroids(centroids)
        index.add(
            x,
            cluster_ids=labels,
            num_threads=np.int64(1),
            fast_quantization=np.bool_(False),
        )
        assert_type(index.dim, int)
        assert type(index.dim) is int
    ivf.reserve(np.uint32(0), np.int64(100))
    ids, distances = ivf.search(
        x[:2], k=np.int64(3), nprobe=np.int64(1), high_accuracy=np.bool_(True)
    )
    assert_type(ids, npt.NDArray[np.uint32])
    assert_type(distances, npt.NDArray[np.float32])
    hnsw.resize(np.int64(100))
    hnsw.search(x[:2], np.int64(3), ef=np.int64(10))
    graph.build(x, np.int64(40), num_threads=np.int64(1))
    graph.search(x[:2], np.int64(3), np.int64(10))
    flat = RaBitQKMeans(
        np.int64(64),
        np.int64(2),
        niter=np.int64(1),
        spherical=np.bool_(False),
        num_threads=np.int64(1),
    )
    qg = QGKMeans(
        np.int64(64),
        np.int64(40),
        niter=np.int64(1),
        graph_degree=np.int64(32),
        num_threads=np.int64(1),
    )
    for model in (flat, qg):
        model.cp.niter = np.int64(1)
        model.cp.seed = np.uint32(42)
        model.cp.min_points_per_centroid = np.int64(1)
        model.cp.max_points_per_centroid = np.int64(1)
        model.cp.num_threads = np.int64(1)
        model.cp.verbose = np.bool_(False)
        model.cp.spherical = np.bool_(False)
        check_parameter_reads(model.cp)
        assert_type(model.train(x), float)
        assignments, distances = model.assign(x[:3])
        assert assignments.shape == distances.shape == (3,)
    qg.cp.graph_degree = np.int64(32)
    qg.cp.ef_build = np.int64(40)
    qg.cp.ef_search = np.int64(16)
    qg.cp.graph_build_iterations = np.int64(1)
    qg.cp.quantization_bits = np.int64(4)
    members = FinalAssignmentMode.__members__
    assert_type(members, dict[str, FinalAssignmentMode])
    assert_type(members["Exact"], FinalAssignmentMode)
    assert members["Exact"] == FinalAssignmentMode.Exact
    assert members["SymphonyQG"] == members["Approximate"]
    assert FinalAssignmentMode(np.int64(1)) == members["Exact"]


def check_input_contracts() -> None:
    """Valid literal options and integer ID containers also execute at runtime."""
    x = np.random.default_rng(42).normal(size=(80, 64)).astype(np.float32)
    centroids = x.mean(axis=0, keepdims=True)
    signed = np.zeros(80, dtype=np.int64)
    unsigned = np.zeros(80, dtype=np.uint32)
    metrics: tuple[Literal["l2", "ip", "innerproduct"], ...] = (
        "l2",
        "ip",
        "innerproduct",
    )
    for metric in metrics:
        ivf = IvfIndex(64, 1, nbits=4, metric=metric)
        HnswIndex(64, nbits=4, metric=metric)
        SymqgIndex(64, 32, metric=metric)
    initializers: tuple[Literal["auto", "flat", "flat_rabitq", "hnsw"], ...] = (
        "auto",
        "flat",
        "flat_rabitq",
        "hnsw",
    )
    for initializer in initializers:
        IvfIndex(64, 1, nbits=4, initializer=initializer)
    for ids in (signed, unsigned, [0] * 80, (0,) * 80):
        ivf = IvfIndex(64, 1, nbits=4)
        ivf.set_centroids(centroids)
        ivf.add(x, cluster_ids=ids)
        ivf.add(x[:1], cluster_ids=[np.int64(0)])
        ivf.add(x[:1], cluster_ids=(np.uint32(0),))
        ivf.remove(np.array([0], dtype=np.int32))
        ivf.remove(np.array([1], dtype=np.uint64))
        ivf.remove([2])
        ivf.remove((3,))
        ivf.remove([])
        ivf.remove(())
        hnsw = HnswIndex(64, nbits=4, num_clusters=1)
        hnsw.set_centroids(centroids)
        hnsw.add(x, cluster_ids=ids, num_threads=1)
        hnsw.add(x[:1], cluster_ids=[np.uint32(0)])
        hnsw.remove([0])
        hnsw.remove(np.array([1], dtype=np.int64))
    initializations: tuple[Literal["random", "pipnn"], ...] = ("random", "pipnn")
    for init in initializations:
        graph = SymqgIndex(64, 32)
        graph.build(x, 40, init=init)
    # Returned option values must be usable as inputs without casts.
    IvfIndex(64, 1, nbits=4, metric=ivf.metric, initializer=ivf.initializer)


def reject_invalid_inputs(x: npt.NDArray[np.float32]) -> None:
    float_ids: npt.NDArray[np.float64] = np.zeros(80, dtype=np.float64)
    bool_ids: npt.NDArray[np.bool_] = np.array([True], dtype=np.bool_)
    string_ids: npt.NDArray[np.str_] = np.array(["1"], dtype=np.str_)
    ivf = IvfIndex(64, 1, nbits=4)
    hnsw = HnswIndex(64, nbits=4)
    graph = SymqgIndex(64, 32)
    IvfIndex(64, 1, nbits=4, metric="cosine")  # type: ignore[call-overload]
    HnswIndex(64, nbits=4, metric="L2")  # type: ignore[call-overload]
    SymqgIndex(64, 32, metric="inner_product")  # type: ignore[arg-type]
    IvfIndex(64, 1, nbits=4, initializer="flat_rabit")  # type: ignore[call-overload]
    graph.build(x, 40, init="pippn")  # type: ignore[arg-type]
    ivf.add(x, cluster_ids=[0.0] * 80)  # type: ignore[list-item]
    hnsw.add(x, cluster_ids=float_ids)  # type: ignore[arg-type]
    ivf.train(x, num_threads="1")  # type: ignore[arg-type]
    hnsw.add(x, cluster_ids=(0.0,))  # type: ignore[arg-type]
    ivf.remove([1.5])  # type: ignore[list-item]
    hnsw.remove(bool_ids)  # type: ignore[arg-type]
    ivf.remove(string_ids)  # type: ignore[arg-type]


def check_paths(
    ivf: IvfIndex,
    hnsw: HnswIndex,
    graph: SymqgIndex,
    path: Path,
    text_path: PathLike[str],
    bytes_path: PathLike[bytes],
) -> None:
    for filename in (
        path,
        text_path,
        bytes_path,
        "index.bin",
        b"index.bin",
        bytearray(b"index.bin"),
    ):
        for index in (ivf, hnsw, graph):
            index.save(filename)
        assert_type(IvfIndex.load(filename), IvfIndex)
        assert_type(HnswIndex.load(filename), HnswIndex)
        assert_type(SymqgIndex.load(filename), SymqgIndex)
    ivf.save(42)  # type: ignore[arg-type]
    HnswIndex.load(object())  # type: ignore[arg-type]
