"""Download and prepare GIST1M and OpenAI-1536."""

import argparse
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

import h5py
import numpy as np

if __package__:
    from .run import load_vectors
else:
    from run import load_vectors

ROOT = Path(__file__).resolve().parents[1]
GIST_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz"
OPENAI_ID = "1cwIY76n_HEbbZAANJVWiVkjBsIy3b9jB"
GIST_FILES = {
    "gist_base.fvecs": {
        "bytes": 3844000000,
    },
    "gist_query.fvecs": {
        "bytes": 3844000,
    },
    "gist_groundtruth.ivecs": {
        "bytes": 404000,
    },
}


def write_fvecs(stream, vectors):
    rows = np.empty((len(vectors), vectors.shape[1] + 1), dtype="<i4")
    rows[:, 0] = vectors.shape[1]
    rows[:, 1:] = vectors.view("<i4")
    rows.tofile(stream)


def write_ivecs(stream, ids):
    rows = np.empty((len(ids), ids.shape[1] + 1), dtype="<i4")
    rows[:, 0] = ids.shape[1]
    rows[:, 1:] = ids.astype("<i4", copy=False)
    rows.tofile(stream)


def exact_groundtruth(base_path, query_path, output_path, k, threads):
    import faiss

    base, _ = load_vectors(base_path)
    queries, _ = load_vectors(query_path)
    faiss.omp_set_num_threads(threads)
    index = faiss.IndexFlatIP(base.shape[1])
    index.add(base)
    with output_path.open("xb") as stream:
        for start in range(0, len(queries), 256):
            ids = index.search(queries[start : start + 256], k)[1]
            write_ivecs(stream, ids)


def check_file(path, expected):
    if not path.exists():
        return False
    if path.stat().st_size != expected["bytes"]:
        raise ValueError(f"unexpected file size; refusing to overwrite {path}")
    print(f"Checked {path}", flush=True)
    return True


def download(cache, name, size, *, url=None, drive_id=None):
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / name
    if path.exists():
        if path.stat().st_size != size:
            raise ValueError(f"unexpected download size: {path}")
        return path
    partial = path.with_name(path.name + ".part")
    print(f"Downloading {name}", flush=True)
    if drive_id:
        import gdown

        gdown.download(
            id=drive_id,
            output=str(partial),
            resume=True,
            use_cookies=False,
            timeout=30,
            retries=2,
        )
    else:
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--retry",
                "2",
                "--connect-timeout",
                "30",
                "--continue-at",
                "-",
                "--output",
                str(partial),
                url,
            ],
            check=True,
        )
    if not partial.is_file() or partial.stat().st_size != size:
        raise ValueError(f"incomplete download: {partial}")
    partial.replace(path)
    return path


def unpack_gist(archive, destination, expected):
    # Extract only the three known regular files; never trust archive paths.
    remaining = {f"gist/{name}": name for name in expected}
    with tarfile.open(archive, "r|gz") as source:
        for member in source:
            if member.name not in remaining:
                continue
            name = remaining.pop(member.name)
            metadata = expected[name]
            if not member.isfile() or member.size != metadata["bytes"]:
                raise ValueError(f"unexpected GIST archive member: {name}")
            with (
                source.extractfile(member) as src,
                (destination / name).open("xb") as dst,
            ):
                shutil.copyfileobj(src, dst, 1024 * 1024)
            check_file(destination / name, metadata)
            if not remaining:
                break
    if remaining:
        raise ValueError(f"missing GIST archive members: {sorted(remaining)}")


def convert_openai(source, destination, files, base_count=999000, dimension=1536):
    with h5py.File(source, "r") as data:
        for key, name, count in (
            ("train", "base.fvecs", base_count),
            ("test", "queries.fvecs", 1000),
        ):
            if name not in files:
                continue
            if data[key].shape != (count, dimension):
                raise ValueError(f"unexpected {key} shape in {source}")
            with (destination / name).open("xb") as stream:
                for start in range(0, count, 4096):
                    # Match SuperKMeans: cast to float32 without renormalizing or shuffling.
                    batch = np.asarray(data[key][start : start + 4096], dtype="<f4")
                    if not np.isfinite(batch).all():
                        raise ValueError(f"non-finite vectors in {source}")
                    write_fvecs(stream, batch)
            check_file(destination / name, files[name])


def prepare(dataset, data_dir, cache, threads):
    if dataset == "gist1m":
        destination = data_dir / "gist"
        files = GIST_FILES
    else:
        destination = data_dir / "dbpedia"
        manifest = json.loads((ROOT / "benchmarks/DBPEDIA_MANIFEST.json").read_text())
        files = dict(manifest["files"])
    destination.mkdir(parents=True, exist_ok=True)
    missing = {
        name: metadata
        for name, metadata in files.items()
        if not check_file(destination / name, metadata)
    }
    if not missing:
        return
    # Validate all new outputs before publishing them alongside existing files.
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=destination) as tmp:
        staging = Path(tmp)
        if dataset == "gist1m":
            archive = download(cache, "gist.tar.gz", 2740172684, url=GIST_URL)
            unpack_gist(archive, staging, missing)
        else:
            if any(name.endswith(".fvecs") for name in missing):
                source = download(
                    cache, "openai-1536-angular.hdf5", 12288006144, drive_id=OPENAI_ID
                )
                convert_openai(source, staging, missing)
            if "queries.ivecs" in missing:
                print(
                    "Computing exact OpenAI inner-product top-100 ground truth",
                    flush=True,
                )
                exact_groundtruth(
                    (staging if "base.fvecs" in missing else destination)
                    / "base.fvecs",
                    (staging if "queries.fvecs" in missing else destination)
                    / "queries.fvecs",
                    staging / "queries.ivecs",
                    100,
                    threads,
                )
        for name in missing:
            (staging / name).replace(destination / name)
    output = {
        "dataset": dataset,
        "source": GIST_URL
        if dataset == "gist1m"
        else f"https://drive.google.com/file/d/{OPENAI_ID}/view",
        "metric": "l2" if dataset == "gist1m" else "inner_product",
        "groundtruth_k": 100,
        "files": {
            name: {
                "bytes": (destination / name).stat().st_size,
            }
            for name in files
        },
    }
    (destination / "download-manifest.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("gist1m", "dbpedia"),
        default=["gist1m", "dbpedia"],
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--cache-dir", type=Path, default=ROOT / "data/benchmark-downloads"
    )
    parser.add_argument("--threads", type=int, default=48)
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("--threads must be positive")
    for dataset in args.datasets:
        prepare(dataset, args.data_dir, args.cache_dir, args.threads)
    print("Datasets ready. K-means computes its separate L2 ground truth when needed.")


if __name__ == "__main__":
    main()
