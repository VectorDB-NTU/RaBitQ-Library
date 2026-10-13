# Benchmarks

For embedding the quantizer in another system, use the independent
[core C++ benchmark](core/README.md). It needs only core and reports per-repeat timings,
distance-estimate error and quantized full-scan recall. The workflows below benchmark
complete ANN/clustering systems. For dispatch, encoding, FastScan, sidecar loading and
Python ID-conversion regression measurements, use the
[component hot-path benchmarks](hotpaths/README.md).

See the [benchmark overview](../docs/docs/benchmarks/bench.md) for datasets,
installation, measurement definitions, file locations, and reproduction options.
Results: [IVF](../docs/docs/benchmarks/ivf.md),
[graphs](../docs/docs/benchmarks/graphs.md), and
[k-means](../docs/docs/benchmarks/kmeans.md).

From the repository root, after installing the documented system prerequisites
and activating a Python 3.13 environment, these commands measure the active checkout.
Published library results use 0.7.0; baseline results retain historical measurements.
See the [recorded revisions and environments](../docs/docs/benchmarks/bench.md#package-installation).
Keep measurements of different revisions in separate output directories.

```bash
python benchmarks/install_baselines.py --jobs 12
source benchmarks/env.sh
python benchmarks/reproduce.py
```

For a subset or a command preview:

```bash
python benchmarks/reproduce.py --benchmarks ivf --datasets gist1m
python benchmarks/reproduce.py --datasets dbpedia --methods qgkmeans
python benchmarks/reproduce.py --dry-run
```
