"""Repeated, matched-settings comparison of QGKMeans, RaBitQKMeans, and FAISS."""

import argparse
import multiprocessing
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=str,
        help="optional float32 .npy or row-major .bin matrix",
    )
    parser.add_argument("--n", type=int, default=1_000_000)
    parser.add_argument("--d", type=int, default=500)
    parser.add_argument("--k", type=int, default=4_000)
    parser.add_argument("--niter", type=int, default=25)
    parser.add_argument("--threads", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-seeds", type=int, default=1, help="number of consecutive training seeds"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="trials per seed, rotating train order",
    )
    parser.add_argument(
        "--spherical",
        action="store_true",
        help="normalize centroids and compare by inner product",
    )
    return parser.parse_args()


def initialize_worker(input_path):
    # Both workers map the same temporary input; only FAISS normalizes it, before
    # training starts. Spawn keeps their native OpenMP runtimes independent.
    global worker_data
    worker_data = np.load(input_path, mmap_mode="r+")


def normalize_input(threads):
    import faiss

    faiss.omp_set_num_threads(threads)
    faiss.normalize_L2(worker_data)
    worker_data.flush()


def evaluate_centroids(centroids, spherical, assignments, save_nearest, threads):
    import faiss

    faiss.omp_set_num_threads(threads)
    x = worker_data
    nearest_out = np.empty(len(x), dtype=np.int64) if save_nearest else None
    index = (
        faiss.IndexFlatIP(x.shape[1]) if spherical else faiss.IndexFlatL2(x.shape[1])
    )
    index.add(np.ascontiguousarray(centroids, dtype=np.float32))
    total = 0.0
    mismatches = 0
    for begin in range(0, x.shape[0], 100_000):
        end = begin + 100_000
        distances, nearest = index.search(x[begin:end], 1)
        if nearest_out is not None:
            nearest_out[begin:end] = nearest[:, 0]
        if spherical:
            total += distances.shape[0] - distances.sum(dtype=np.float64)
        else:
            total += distances.sum(dtype=np.float64)
        if assignments is not None:
            mismatches += int(np.count_nonzero(assignments[begin:end] != nearest[:, 0]))
    return total, mismatches, nearest_out


def exact_objective(
    worker, threads, centroids, spherical, assignments=None, nearest_out=None
):
    """Evaluate all methods in the FAISS worker, outside training time."""
    total, mismatches, nearest = worker.submit(
        evaluate_centroids,
        centroids,
        spherical,
        assignments,
        nearest_out is not None,
        threads,
    ).result()
    if nearest_out is not None:
        nearest_out[:] = nearest
    return total, mismatches


def train_method(method, args, seed):
    x = worker_data
    n, d = x.shape
    common = dict(
        niter=args.niter,
        seed=seed,
        spherical=args.spherical,
        min_points_per_centroid=1,
        early_stop_threshold=0.0,
    )
    if method != "FAISS":
        from rabitqlib import QGKMeans, RaBitQKMeans

        cls = QGKMeans if method == "QGKMeans" else RaBitQKMeans
        kmeans = cls(d, args.k, num_threads=args.threads, **common)
    else:
        import faiss

        faiss.omp_set_num_threads(args.threads)
        # Prevent subsampling so all methods train on all vectors.
        kmeans = faiss.Kmeans(
            d,
            args.k,
            nredo=1,
            verbose=False,
            gpu=False,
            max_points_per_centroid=n,
            **common,
        )
    start = time.perf_counter()
    kmeans.train(x)
    elapsed = time.perf_counter() - start
    result = dict(
        seconds=elapsed,
        iterations=len(kmeans.iteration_stats),
        centroids=kmeans.centroids,
    )
    if method != "FAISS":
        result.update(assignments=kmeans.assignments, final_obj=kmeans.final_obj)
    return result


def main():
    args = parse_args()
    if args.num_seeds < 1 or args.repeats < 1:
        raise ValueError("--num-seeds and --repeats must be positive")

    if args.input:
        if args.input.endswith(".npy"):
            x = np.load(args.input)
        else:
            x = np.fromfile(args.input, dtype=np.float32)
            if x.size % args.d != 0:
                raise ValueError("raw --input size must be divisible by --d")
            x = x.reshape(-1, args.d)
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("--input must contain a two-dimensional array")
    else:
        rng = np.random.default_rng(args.seed)
        x = rng.standard_normal((args.n, args.d), dtype=np.float32)

    n, d = x.shape
    if n < args.k:
        raise ValueError(f"number of vectors ({n}) must be at least k ({args.k})")

    # Stage once, share pages across workers, and delete after both exit. Native
    # imports must stay in workers: even importing FAISS can pin the parent CPU.
    with TemporaryDirectory(prefix="qgkmeans-comparison-") as directory:
        input_path = Path(directory) / "input.npy"
        np.save(input_path, x)
        del x
        context = multiprocessing.get_context("spawn")
        with (
            ProcessPoolExecutor(
                max_workers=1,
                mp_context=context,
                initializer=initialize_worker,
                initargs=(input_path,),
            ) as faiss_worker,
            ProcessPoolExecutor(
                max_workers=1,
                mp_context=context,
                initializer=initialize_worker,
                initargs=(input_path,),
            ) as rabitq_worker,
        ):
            if args.spherical:
                faiss_worker.submit(normalize_input, args.threads).result()
            compare_trials(args, n, d, rabitq_worker, faiss_worker)


def compare_trials(args, n, d, rabitq_worker, faiss_worker):
    metric = "cosine distance" if args.spherical else "squared L2"
    print(
        f"n={n}, d={d}, k={args.k}, requested niter={args.niter}, "
        f"threads={args.threads}, metric={metric}, "
        f"seeds={args.num_seeds}, repeats/seed={args.repeats}"
    )
    print("Input data stays fixed across trials; training seeds change.")
    print("The same seed does not give identical initial centroids across methods.")
    print(
        "Training includes the library methods' default approximate final assignment; "
        "exhaustive evaluation is outside training time."
    )
    methods = ["QGKMeans", "RaBitQKMeans", "FAISS"]
    results = {method: [] for method in methods}
    for seed_offset in range(args.num_seeds):
        seed = args.seed + seed_offset
        cache = {}
        nearest = {
            method: np.empty(n, dtype=np.int64) if args.repeats > 1 else None
            for method in methods
            if method != "FAISS"
        }
        for repeat in range(args.repeats):
            offset = (seed_offset * args.repeats + repeat) % len(methods)
            order = methods[offset:] + methods[:offset]
            trained = {}
            for method in order:
                worker = faiss_worker if method == "FAISS" else rabitq_worker
                trained[method] = worker.submit(
                    train_method, method, args, seed
                ).result()

            print(f"seed={seed}, repeat={repeat + 1}, train order={', '.join(order)}")
            print(
                f"{'method':<12} {'train (s)':>12} {'iterations':>11} "
                f"{'exhaustive objective':>21} {'per vector':>14}"
            )
            for method in methods:
                result = trained[method]
                centroids = result["centroids"]
                assignments = result.get("assignments")
                previous = cache.get(method)
                if previous is not None and np.array_equal(centroids, previous[0]):
                    objective = previous[1]
                    mismatches = (
                        int(np.count_nonzero(assignments != nearest[method]))
                        if assignments is not None
                        else 0
                    )
                else:
                    objective, mismatches = exact_objective(
                        faiss_worker,
                        args.threads,
                        centroids,
                        args.spherical,
                        assignments,
                        nearest.get(method),
                    )
                    cache[method] = (centroids.copy(), objective)
                results[method].append((result["seconds"], objective))
                print(
                    f"{method:<12} {result['seconds']:>12.6f} "
                    f"{result['iterations']:>11} {objective:>21.6g} "
                    f"{objective / n:>14.6g}"
                )
                if assignments is not None:
                    print(
                        f"{method} returned-label objective: {result['final_obj']:.6g}; "
                        f"label mismatches vs exhaustive: {mismatches}/{n} "
                        f"({mismatches / n:.4%})"
                    )

    print("Training time over all trials (median [min, max], seconds):")
    for method in methods:
        times = [seconds for seconds, _ in results[method]]
        print(
            f"  {method}: {statistics.median(times):.6f} "
            f"[{min(times):.6f}, {max(times):.6f}]"
        )
    for method in methods[:-1]:
        paired = list(zip(results[method], results["FAISS"]))
        ratios = [seconds / baseline for (seconds, _), (baseline, _) in paired]
        print(
            f"{method}/FAISS paired training-time ratio: "
            f"median {statistics.median(ratios):.3f}x "
            f"[{min(ratios):.3f}x, {max(ratios):.3f}x]"
        )
        differences = [objective - baseline for (_, objective), (_, baseline) in paired]
        print(
            f"{method} exhaustive objective difference vs FAISS (absolute): "
            f"median {statistics.median(differences):+.6g} "
            f"[{min(differences):+.6g}, {max(differences):+.6g}]"
        )
        if all(baseline != 0 for _, (_, baseline) in paired):
            percentages = [
                (objective / baseline - 1) * 100
                for (_, objective), (_, baseline) in paired
            ]
            print(
                f"{method} exhaustive objective difference vs FAISS (%): "
                f"median {statistics.median(percentages):+.4f}% "
                f"[{min(percentages):+.4f}%, {max(percentages):+.4f}%]"
            )
        else:
            print(
                f"{method} exhaustive objective difference vs FAISS (%): N/A "
                "(at least one FAISS objective is zero)"
            )


if __name__ == "__main__":
    main()
