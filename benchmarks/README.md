# Benchmarks

See the [benchmark overview](../docs/docs/benchmarks/bench.md) for datasets,
installation, measurement definitions, file locations, and reproduction options.
Results: [IVF](../docs/docs/benchmarks/ivf.md),
[graphs](../docs/docs/benchmarks/graphs.md), and
[k-means](../docs/docs/benchmarks/kmeans.md).

From the repository root, after installing the documented system prerequisites
and activating a Python 3.13 environment:

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
