"""Static-only fixture: assert precise types and reject invalid usage."""

from os import PathLike
from pathlib import Path
from typing import Literal, assert_type

import numpy as np
import numpy.typing as npt
import rabitqlib
from rabitqlib import (
    HnswIndex,
    IvfIndex,
    KMeansIterationStats,
    QGKMeans,
    QGKMeansParameters,
    RaBitQKMeans,
    RaBitQKMeansParameters,
    SymqgIndex,
    quant,
)


def check(x: npt.NDArray[np.float32], labels: npt.NDArray[np.uint32]) -> None:
    assert_type(rabitqlib.__all__, list[str])
    ivf = IvfIndex(64, 2, nbits=4, seed=7)
    hnsw = HnswIndex(64, nbits=4, num_clusters=2)
    graph = SymqgIndex(64, ef_construction=64)
    for index in (ivf, hnsw, graph):
        assert_type(index.__repr__(), str)
    for index in (ivf, hnsw):
        assert_type(index.cp, RaBitQKMeansParameters)
        index.cp.niter = 10
        index.train(x, num_threads=1)
        index.set_centroids(x[:2])
        assert_type(index.is_trained, bool)
        assert_type(index.seed, int | None)
        assert_type(index.add(x, cluster_ids=labels), npt.NDArray[np.int64])
        assert_type(index.remove([0, 1]), int)
        assert_type(index.dim, int)
        assert_type(len(index), int)
        index.save("index.bin")
    ids, distances = ivf.search(x, k=5, nprobe=2)
    assert_type(ids, npt.NDArray[np.int64])
    assert_type(distances, npt.NDArray[np.float32])
    assert_type(
        hnsw.search(x, 5), tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]
    )
    graph.train(x)
    graph.add(x, num_threads=1)
    assert_type(graph.is_trained, bool)
    assert_type(graph.nbits, int)
    assert_type(len(graph), int)
    assert_type(
        graph.search(x, 5),
        tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]],
    )
    assert_type(IvfIndex.load("index.bin"), IvfIndex)
    assert_type(HnswIndex.load("index.bin"), HnswIndex)
    assert_type(SymqgIndex.load("index.bin"), SymqgIndex)
    ivf.reserve_cluster(0, 128)
    hnsw.reserve(256)
    assert_type(hnsw.capacity, int)
    assert_type(ivf.num_stored_points, int)
    ivf.search(x, 5, 1, high_accuracy=None, num_threads=1)
    hnsw.search(x, 5, 32, num_threads=1)
    graph.search(x, 5, ef_search=32, num_threads=1)
    qg = QGKMeans(64, 40, niter=2, nbits=4, max_degree=32, ef_construction=100)
    flat = RaBitQKMeans(64, 2, final_assignment="exact")
    assert_type(qg.cp.nbits, int)
    assert_type(flat.cp.final_assignment, Literal["approximate", "exact"])
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
            assert_type(model.iteration_stats[0], KMeansIterationStats)
    qg.cp.final_assignment = "approximate"
    ivf.search(x, "five", 2)  # type: ignore[arg-type]
    ivf.dim = 16  # type: ignore[misc]
    qg.cp.niter = "two"  # type: ignore[assignment]
    RaBitQKMeans(64, 2, graph_degree=32)  # type: ignore[call-arg]
    hnsw.search(x, 5, nprobe=2)  # type: ignore[call-arg]
    IvfIndex(64, 2, 4)  # type: ignore[call-arg]
    hnsw.train(x, 1)  # type: ignore[call-arg]
    qg.cp.final_assignment = "Exact"  # type: ignore[assignment]
    QGKMeans(64, 40, ef_build=1, bogus=1)  # type: ignore[call-arg]


def reject_removed_forms(
    ivf: IvfIndex,
    hnsw: HnswIndex,
    graph: SymqgIndex,
    qg: QGKMeans,
    x: npt.NDArray[np.float32],
    labels: npt.NDArray[np.uint32],
) -> None:
    # The 0.7 call forms and the 0.8 spellings that 0.9 removed.
    IvfIndex(64, 80, 1, 4)  # type: ignore[arg-type, call-arg]
    IvfIndex(dim=64, max_elements=80, num_clusters=1, nbits=4)  # type: ignore[call-arg]
    IvfIndex(64, 1, initializer="flat")  # type: ignore[call-arg]
    HnswIndex(64, 80)  # type: ignore[call-arg]
    HnswIndex(64, max_elements=80)  # type: ignore[call-arg]
    HnswIndex(64, random_seed=5)  # type: ignore[call-arg]
    SymqgIndex(64, 32, "l2", 4)  # type: ignore[arg-type, call-arg]
    SymqgIndex(64, quantization_bits=0)  # type: ignore[call-arg]
    ivf.build(x, x[:1], labels)  # type: ignore[attr-defined]
    hnsw.build(x, x[:1], labels)  # type: ignore[attr-defined]
    graph.build(x, 40)  # type: ignore[attr-defined]
    ivf.add(x, labels)  # type: ignore[call-arg]
    hnsw.add(x, labels)  # type: ignore[call-arg]
    ivf.is_built  # type: ignore[attr-defined]
    graph.is_built  # type: ignore[attr-defined]
    ivf.max_elements  # type: ignore[attr-defined]
    hnsw.max_elements  # type: ignore[attr-defined]
    graph.max_elements  # type: ignore[attr-defined]
    ivf.initializer  # type: ignore[attr-defined]
    ivf.reserve(0, 8)  # type: ignore[attr-defined]
    hnsw.resize(128)  # type: ignore[attr-defined]
    graph.quantization_bits  # type: ignore[attr-defined]
    graph.search_batch(x, 5, 32)  # type: ignore[attr-defined]
    ivf.search(x, 5, 1, None)  # type: ignore[call-arg]
    hnsw.search(x, 5, ef=32)  # type: ignore[call-arg]
    hnsw.search(x, 5, 32, 1)  # type: ignore[call-arg]
    graph.search(x, 5, ef=32)  # type: ignore[call-arg]
    graph.search(x, 5, 32, 1)  # type: ignore[call-arg]
    QGKMeans(64, 40, quantization_bits=4)  # type: ignore[call-arg]
    QGKMeans(64, 40, graph_degree=32)  # type: ignore[call-arg]
    QGKMeans(64, 40, ef_build=100)  # type: ignore[call-arg]
    qg.cp.quantization_bits = 4  # type: ignore[attr-defined]
    qg.cp.graph_degree = 32  # type: ignore[attr-defined]
    qg.cp.ef_build = 100  # type: ignore[attr-defined]
    rabitqlib.FinalAssignmentMode  # type: ignore[attr-defined]
    rabitqlib.QGKMeansIterationStats  # type: ignore[attr-defined]
    rabitqlib.RaBitQKMeansIterationStats  # type: ignore[attr-defined]


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
    graph = SymqgIndex(
        np.int64(64),
        nbits=np.int64(4),
        max_degree=np.int64(32),
        ef_construction=np.int64(40),
        seed=np.uint32(5),
    )
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
    ivf.reserve_cluster(np.uint32(0), np.int64(100))
    ids, distances = ivf.search(
        x[:2], k=np.int64(3), nprobe=np.int64(1), high_accuracy=np.bool_(True)
    )
    assert_type(ids, npt.NDArray[np.int64])
    assert_type(distances, npt.NDArray[np.float32])
    hnsw.reserve(np.int64(100))
    hnsw.search(x[:2], np.int64(3), ef_search=np.int64(10))
    graph.add(x, num_threads=np.int64(1))
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
        max_degree=np.int64(32),
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
    qg.cp.max_degree = np.int64(32)
    qg.cp.ef_construction = np.int64(40)
    qg.cp.ef_search = np.int64(16)
    qg.cp.graph_build_iterations = np.int64(1)
    qg.cp.nbits = np.int64(4)
    for mode in ("approximate", "exact"):
        qg.cp.final_assignment = mode
        assert qg.cp.final_assignment == mode


def check_input_contracts() -> None:
    """Valid literal options and integer ID containers also execute at runtime."""
    x = np.random.default_rng(42).normal(size=(80, 64)).astype(np.float32)
    centroids = x.mean(axis=0, keepdims=True)
    signed = np.zeros(80, dtype=np.int64)
    unsigned = np.zeros(80, dtype=np.uint32)
    metrics: tuple[Literal["l2", "ip", "innerproduct", "cosine"], ...] = (
        "l2",
        "ip",
        "innerproduct",
        "cosine",
    )
    for metric in metrics:
        ivf = IvfIndex(64, 1, nbits=4, metric=metric)
        HnswIndex(64, nbits=4, metric=metric)
        SymqgIndex(64, metric=metric)
    routings: tuple[Literal["auto", "flat", "flat_rabitq", "hnsw"], ...] = (
        "auto",
        "flat",
        "flat_rabitq",
        "hnsw",
    )
    for routing in routings:
        IvfIndex(64, 1, nbits=4, routing=routing)
    for ids in (signed, unsigned, [0] * 80, (0,) * 80):
        ivf = IvfIndex(64, 1, nbits=4)
        ivf.set_centroids(centroids, num_threads=1)
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
        graph = SymqgIndex(64, ef_construction=40, init=init)
        graph.add(x)
    # Returned option values must be usable as inputs without casts.
    IvfIndex(64, 1, nbits=4, metric=ivf.metric, routing=ivf.routing)
    assert_type(ivf.metric, Literal["l2", "ip", "cosine"])


def reject_invalid_inputs(x: npt.NDArray[np.float32]) -> None:
    float_ids: npt.NDArray[np.float64] = np.zeros(80, dtype=np.float64)
    bool_ids: npt.NDArray[np.bool_] = np.array([True], dtype=np.bool_)
    string_ids: npt.NDArray[np.str_] = np.array(["1"], dtype=np.str_)
    ivf = IvfIndex(64, 1, nbits=4)
    hnsw = HnswIndex(64, nbits=4)
    graph = SymqgIndex(64)
    IvfIndex(64, 1, nbits=4, metric="cos")  # type: ignore[arg-type]
    HnswIndex(64, nbits=4, metric="L2")  # type: ignore[arg-type]
    SymqgIndex(64, metric="inner_product")  # type: ignore[arg-type]
    IvfIndex(64, 1, nbits=4, routing="flat_rabit")  # type: ignore[arg-type]
    SymqgIndex(64, init="pippn")  # type: ignore[arg-type]
    graph.add(x, 1)  # type: ignore[call-arg]
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


def check_quant(x: npt.NDArray[np.float32]) -> None:
    rotator = quant.Rotator(64, method="fht_kac", seed=0)
    assert_type(rotator.padded_dim, int)
    assert_type(rotator.seed, int | None)
    assert_type(rotator.method, Literal["fht_kac", "matrix"])
    residuals = rotator.rotate(x, num_threads=1)
    assert_type(residuals, npt.NDArray[np.float32])
    assert_type(quant.Rotator.from_bytes(rotator.to_bytes()), quant.Rotator)
    codes = quant.quantize(residuals, nbits=4)
    assert_type(codes, npt.NDArray[np.uint8])
    assert_type(quant.quantize(residuals, nbits=9), npt.NDArray[np.uint16])
    prefix, ex = quant.split(codes, nbits=4, prefix_bits=1)
    assert_type(prefix, npt.NDArray[np.uint8])
    assert_type(
        quant.split(codes, nbits=4, prefix_bits=1, num_threads=2, out=(prefix, ex)),
        tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]],
    )
    factors = quant.code_factors(residuals, codes, nbits=4, metric="l2")
    assert_type(factors, npt.NDArray[np.void])
    assert_type(quant.pack_codes(ex, nbits=3), npt.NDArray[np.uint8])
    assert_type(
        quant.unpack_codes(quant.pack_codes(ex, nbits=3), nbits=3),
        npt.NDArray[np.uint8],
    )
    assert_type(quant.padded_dim(100), int)
    assert_type(quant.FACTORS_DTYPE, np.dtype[np.void])
    assert issubclass(quant.FormatError, ValueError)
    f_prefix = quant.code_factors(residuals, prefix, nbits=1, metric="l2")
    query = quant.Query(residuals[0], metric="l2", centroid=None)
    assert_type(query.metric, Literal["l2", "ip"])
    bitsets = quant.pack_codes(prefix, nbits=1)
    first = query.estimate_bitset(bitsets, f_prefix, indices=[0], num_threads=1)
    assert_type(first, npt.NDArray[np.void])
    assert_type(query.estimate_codes(codes, factors, nbits=4), npt.NDArray[np.void])
    assert_type(
        query.refine(ex, factors, first, ex_bits=3, indices=[0]),
        npt.NDArray[np.float32],
    )
    assert_type(
        query.refine_packed(quant.pack_codes(ex, nbits=3), factors, first, ex_bits=3),
        npt.NDArray[np.float32],
    )
    query.set_centroid(None)
    assert_type(query.copy(), quant.Query)
    terms = quant.query_terms(residuals[0], metric="ip")
    assert_type(terms, np.void)
    assert_type(
        quant.estimate_from_ip(first["refine_input"], f_prefix, terms, nbits=1),
        npt.NDArray[np.void],
    )
    blocks = quant.pack_fastscan_blocks(bitsets, f_prefix, num_threads=0)
    assert_type(blocks, npt.NDArray[np.uint8])
    assert_type(
        query.estimate_fastscan(blocks, num_points=len(bitsets)), npt.NDArray[np.void]
    )
    assert_type(quant.FASTSCAN_BLOCKS_VERSION, int)
    quant.Query(residuals[0], metric="cosine")  # type: ignore[arg-type]
    quant.code_factors(residuals, codes, nbits=4, metric="cosine")  # type: ignore[arg-type]
    quant.quantize(residuals, 4)  # type: ignore[call-overload]
    quant.Rotator(64, method="hadamard")  # type: ignore[arg-type]
