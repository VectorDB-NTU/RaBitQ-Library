# Benchmark overview

Compare RaBitQ-Library, Faiss, and SuperKMeans on one Intel Xeon Gold 6418H:
[IVF indexes](ivf.md), [graph indexes](graphs.md), and [k-means](kmeans.md).

## Key results

![GIST1M: queries per second and index size for IVF-RaBitQ, HNSW-RaBitQ and SymphonyQG](figures/homepage-gist1m.svg){ .chart }

On GIST1M, each bar is the fastest measured setting with Recall@10 >= 90%,
without interpolation. IVF-RaBitQ/HNSW-RaBitQ use 5 total bits (one sign bit plus
four extra bits) and SymphonyQG uses 4 bits. Actual recall and total storage
differ; this is neither an exact-recall nor an equal-memory comparison.

| Configuration | Measured Recall@10 | Queries / second | Index GiB | Selected search setting |
| --- | ---: | ---: | ---: | --- |
| IVF-RaBitQ · 5-bit | 92.82% | 1,038 | 0.60 | `nprobe=128` |
| HNSW-RaBitQ · 5-bit | 92.70% | 1,866 | 0.72 | `ef=512` |
| SymphonyQG · 4-bit | 90.36% | 3,086 | 4.39 | `ef=512` |

These use the 0.7.1 measurements with one search thread and one query per call.
Each setting has one timing pass. QPS includes Python call overhead after warm-up
and excludes loading. Index size is serialized storage including sidecars, not
resident or peak RAM. The figure and this table are a static export:
`reproduce.py` does not regenerate them, so update both by hand after new runs.

Full curves, baselines, construction costs and other precisions:
[IVF indexes](ivf.md) · [graph indexes](graphs.md) · [k-means](kmeans.md).

## Reading the figures

Recall@10 is the mean fraction of exact top-10 neighbors returned. Each figure
groups methods with similar storage per dimension into panels that share the QPS
axis. Faint markers show all measured settings within the axis range; lines connect
each method's Pareto frontier. QPS uses a logarithmic axis. The recall axis starts
at 0.5, or lower so every method's best recall sits at least 0.1 inside the axis;
settings below the start are clipped, and the CSV contains every measured point.

## Reading the indexing tables

| Column | Meaning |
| --- | --- |
| Train | Training or centroid clustering time |
| Construct | Encoding and index construction after training |
| Total | Construction including setup and training, excluding saving |
| Save | Serialization time |
| Index GiB | Serialized files, including sidecars and retained raw vectors |

Times are seconds. Data preparation, ground-truth generation, and correctness
checks are outside construction timing.

## Datasets and ground truth

| Dataset | Base vectors | Dimensions | ANN metric | Queries |
| --- | ---: | ---: | --- | ---: |
| GIST1M | 1,000,000 | 960 | L2 | 1,000 |
| DBpedia / OpenAI-1536 | 999,000 | 1,536 | Inner product | 1,000 |

GIST1M uses the standard TexMex split and supplied L2 ground truth. OpenAI-1536
uses [SuperKMeans' dataset](https://github.com/cwida/SuperKMeans/blob/main/BENCHMARKING.md),
preserving its split, row order, and normalized float32 values. ANN ground truth
for OpenAI is computed by exhaustive inner-product search.

All methods share one fixed query set per dataset. Queries are used only for
evaluation. [K-means](kmeans.md) uses L2 for both datasets and computes separate
L2 ground truth for OpenAI.

## Hardware and software

| Setting | Value |
| --- | --- |
| CPU | One Intel Xeon Gold 6418H, 24 cores, 48 hardware threads |
| Construction / clustering | 48 threads |
| Query throughput | 1 process, 1 search thread, 1 query per call |
| Platform | Ubuntu 22.04, x86-64; RaBitQ 0.7.1: Linux 6.8.0-138-generic; retained baselines: Linux 6.8.0-84-generic |
| Python / NumPy | 3.13.15 / 2.5.2 |
| Compiler / CMake | RaBitQ 0.7.1: GCC 14.3.0 / 3.28.3, Release, native optimization OFF; retained baselines: GCC 13.1.0 / 3.28.3, Release |
| RaBitQ-Library | 0.7.1, measured 2026-10-23 (Asia/Singapore) |
| Baselines | Faiss 1.15.0, SuperKMeans 0.2.0; Intel MKL 2020.4, GNU OpenMP |

## Construction and search settings

IVF uses 4,096 clusters. HNSW uses `M=16` and `efConstruction=200`;
SymphonyQG uses degree 32. Family pages list all method settings.

| Family | Search grid |
| --- | --- |
| IVF | `nprobe=1,2,4,8,16,32,64,128,256,512` |
| Faiss IVF with RFlat | Above × `k_factor=1,2,5,10,20` |
| Graph | `ef=16,32,64,128,256,512,1024` |

Each setting searches all 1,000 queries at `k=10`, after warm-up. QPS is queries
divided by elapsed search time, including Python call overhead and excluding
index loading. Each setting has one timing pass.

## Scope and limitations

Each method has one build or fit per dataset; results do not measure run-to-run
variation.
The committed CSVs support checking the displayed tables and plotted points,
but do not include raw run logs or measurement binaries. The 0.7.1 raw records,
logs, indexes, source/build manifest and validation logs are not published.
Historical baseline raw logs and
binaries are not included. No run-to-run or shared-host noise correction is
applied. The experiments in the cited papers are separate measurements.

## Package installation

The RaBitQ-Library rows and curves were refreshed on 2026-10-23 (Asia/Singapore)
using a locally built 0.7.1 wheel. The measurement checkout was
`1f2a271fea16afabee63a96cf15ecbb3ff1c3723`, with only the package versions in
`CMakeLists.txt` and `pyproject.toml` changed from 0.7.0 to 0.7.1 before building.
The SHA-256 of that `git diff --binary` snapshot was
`c678630e695c45434b1c5296c8b37404abc66f1739a97d9ffe7550f41de68432`.
The measured extension SHA-256 was
`c80b90dc666006c5bdcf9884e445da288b8f280f53232d0560c1ff55d4a06d4d`.

The latest available Faiss and SuperKMeans measurements are retained unchanged from
snapshot `a332c1ce97588aaed4f6a8a5476f84a81f947313`; they were not rerun.
The baseline configurations, search grids and timing definitions are unchanged.
These comparisons combine separate measurement sessions and different compiler
and kernel versions, as recorded above; they are not a controlled measurement
of the effect of upgrading RaBitQ alone.

To use the measured library source, start in a separate checkout and
activate your benchmark Python environment:

```bash
git clone --no-checkout https://github.com/VectorDB-NTU/RaBitQ-Library.git RaBitQ-benchmark-snapshot
cd RaBitQ-benchmark-snapshot
git checkout --detach 1f2a271fea16afabee63a96cf15ecbb3ff1c3723
```

Set the two package version declarations to 0.7.1 to reproduce the wheel metadata;
the library code is unchanged from that commit. The retained baseline snapshot
does not identify its original measurement binaries, whose exact reproduction
remains unverified. For any new measurement, record the source commit, local changes,
compiler and build options, and use a separate output directory.

Use a Python 3.13 environment on Ubuntu x86-64. Ubuntu 22.04 needs a package
source providing `g++-13` and the `multiverse` component for MKL.

```bash
sudo apt-get update
sudo apt-get install build-essential g++-13 git numactl curl \
  libmkl-dev libmkl-rt libmkl-gnu-thread

python benchmarks/install_baselines.py --jobs 12
source benchmarks/env.sh
```

The installer builds RaBitQ from the selected checkout, Faiss 1.15.0, and SuperKMeans 0.2.0
in the active environment. Both baselines link to MKL with GNU OpenMP. Build logs
and verification records are saved in `build/benchmark-baselines/`.
Use `--dry-run` to check prerequisites and print the installation commands
without running them. The installer defaults to the historical GCC 13 toolchain
and builds RaBitQ with `RABITQ_ENABLE_NATIVE_OPTIMIZATION=ON`; it does not
recreate the GCC 14.3.0, native-optimization-OFF wheel used for the 0.7.1
measurements. Match the recorded compiler and native-optimization setting when
comparing new library timings.

## Data and reproduction

Run from the selected checkout's repository root after installation. Use the
same checkout for installation and measurement. The runners target the recorded
hardware: the download, build and query stages run under
`numactl --cpunodebind=0 --membind=0`, ANN
construction uses 48 threads on socket 0, and the runners stop with an error when
the process spans more than one socket, the socket has fewer CPUs than threads,
or k-means does not see 24 physical cores and 48 logical CPUs. Other hosts
need script changes.

```bash
# Prepare data, run all benchmarks, and export results
python benchmarks/reproduce.py

# Prepare data only
python benchmarks/download_datasets.py

# Repeat only the library methods (leave historical baseline exports unchanged)
python benchmarks/reproduce.py --output-dir results/benchmarks-new-run \
  --methods rabitq-ivf-raw rabitq-ivf-1p4 rabitq-ivf-1p8 \
  rabitq-symqg-raw rabitq-symqg-4 rabitq-symqg-8 \
  rabitq-hnsw-1p4 rabitq-hnsw-1p8 qgkmeans

# Selected results
python benchmarks/reproduce.py --benchmarks ivf --datasets gist1m
python benchmarks/reproduce.py --datasets dbpedia --methods qgkmeans
```

Downloads preserve the published splits and check file sizes and vector shapes. GIST and
OpenAI need about 26 GB for downloads and prepared data; indexes and temporary
workspace require additional space.

| Content | Default location |
| --- | --- |
| Download cache | `data/benchmark-downloads/` |
| GIST vectors and ground truth | `data/gist/` |
| OpenAI vectors and ANN ground truth | `data/dbpedia/` |
| ANN build records / query records | `results/benchmarks/ann/<dataset>/` / `results/benchmarks/ann/sweeps/<dataset>/` |
| Saved indexes | `results/benchmarks/ann/indexes/` |
| K-means records, centroids, assignments | `results/benchmarks/kmeans/<dataset>/` |
| Logs | `results/benchmarks/ann/logs/`, `results/benchmarks/kmeans/logs/` |
| Tables / CSVs and SVGs / website | `docs/docs/benchmarks/` / `docs/docs/benchmarks/figures/` / `docs/site/` |

Existing results are skipped. Use a fresh `--output-dir` to repeat measurements;
this changes raw result and log paths, while documentation exports retain the
paths above. Partial exports preserve unselected results; figure subtitles name
the CPU recorded by the selected records, and retained rows are not rechecked.

Use `--stage build`, `--stage query`, or `--stage export` for one stage. Querying
reuses saved indexes; export runs no experiments. `--dry-run` previews commands;
`--help` lists methods. Both datasets are selected by default.

`data/` and `results/` are ignored by Git. The downloader accepts `--datasets`,
`--data-dir`, and `--cache-dir`; runners expect the default data paths.

[Query CSV](figures/qps-recall-points.csv) · [Indexing CSV](figures/indexing.csv) ·
[K-means CSV](figures/kmeans.csv)
