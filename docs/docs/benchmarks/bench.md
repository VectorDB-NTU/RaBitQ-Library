# Benchmark overview

Compare RaBitQ-Library, Faiss, and SuperKMeans on one Intel Xeon Gold 6418H:
[IVF indexes](ivf.md), [graph indexes](graphs.md), and [k-means](kmeans.md).

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
| Platform | Ubuntu 22.04.5 LTS, Linux 6.8.0-84-generic, x86-64 |
| Python / NumPy | 3.13.15 / 2.5.2 |
| Construction and clustering compiler / CMake | GCC 13.1.0 / 3.28.3, Release |
| RaBitQ-Library | 0.5.2 |
| Baselines | Faiss 1.15.0, SuperKMeans 0.2.0; Intel MKL 2020.4, GNU OpenMP |

## Package installation

The tables and figures are historical results, labeled RaBitQ-Library 0.5.2.
The published source, scripts, and results are available together at snapshot
`a332c1ce97588aaed4f6a8a5476f84a81f947313`. To run that snapshot's protocol,
start in a separate checkout and activate your benchmark Python environment:

```bash
git clone --no-checkout https://github.com/VectorDB-NTU/RaBitQ-Library.git RaBitQ-benchmark-snapshot
cd RaBitQ-benchmark-snapshot
git checkout --detach a332c1ce97588aaed4f6a8a5476f84a81f947313
```

This pins the published snapshot, not a verified original measurement binary:
the published CSVs do not record the exact source commit used for those binaries.
The version label alone does not establish that they match the `v0.5.2` release tag.
Exact reproduction of the original binaries therefore remains unverified.

To benchmark current code instead, use your current checkout and record its
commit and any local changes. Treat those measurements as a new run, separate
from the historical results below.

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
Use `--dry-run` to preview installation commands.

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

## Reading the figures

Recall@10 is the mean fraction of exact top-10 neighbors returned. Faint markers
show all measured settings; lines connect each method's Pareto frontier. QPS uses
a logarithmic axis. The CSV contains every measured point.

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

## Scope and limitations

Each method has one build or fit per dataset; results do not measure run-to-run
variation.

## Data and reproduction

Run from the selected checkout's repository root after installation. Use the
same checkout for installation and measurement:

```bash
# Prepare data, run all benchmarks, and export results
python benchmarks/reproduce.py

# Prepare data only
python benchmarks/download_datasets.py

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
paths above. Partial exports preserve unselected results.

Use `--stage build`, `--stage query`, or `--stage export` for one stage. Querying
reuses saved indexes; export runs no experiments. `--dry-run` previews commands;
`--help` lists methods. Both datasets are selected by default.

`data/` and `results/` are ignored by Git. The downloader accepts `--datasets`,
`--data-dir`, and `--cache-dir`; runners expect the default data paths.

[Query CSV](figures/qps-recall-points.csv) · [Indexing CSV](figures/indexing.csv) ·
[K-means CSV](figures/kmeans.csv)
