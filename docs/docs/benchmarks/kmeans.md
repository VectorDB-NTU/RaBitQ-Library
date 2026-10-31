# K-means benchmark

RaBitQ-Library results were measured with **0.7.1 on 2026-10-23 (Asia/Singapore)**.
Each baseline retains its latest measurement and was not rerun.
See the [source revisions, environments and comparison limits](bench.md#package-installation).

Compare **QGKMeans**, **Faiss K-means**, and **SuperKMeans** using their Python APIs.
Metrics are total time, WCSS, and recall@10 with 1% of clusters probed.

## Results

### GIST1M

<!-- kmeans:gist1m:start -->
| Method | Total time (s) | WCSS | Recall@10 (40 probes) | Iterations |
| --- | ---: | ---: | ---: | ---: |
| QGKMeans | 14.88 | 1034709 | 0.815900 | 25 |
| Faiss K-means | 84.17 | 1034624.88 | 0.815700 | 25 |
| SuperKMeans | 28.14 | 1034648.39 | 0.820700 | 25 |
<!-- kmeans:gist1m:end -->

### DBpedia

<!-- kmeans:dbpedia:start -->
| Method | Total time (s) | WCSS | Recall@10 (40 probes) | Iterations |
| --- | ---: | ---: | ---: | ---: |
| QGKMeans | 22.35 | 620213.444 | 0.894400 | 25 |
| Faiss K-means | 171.03 | 620737.534 | 0.894700 | 25 |
| SuperKMeans | 35.23 | 620268.628 | 0.896800 | 24 |
<!-- kmeans:dbpedia:end -->

[Download measurements (CSV)](figures/kmeans.csv)

## Settings

| Setting | Value |
| --- | --- |
| Threads | 48; methods run sequentially in separate processes |
| Clustering | Flat L2, 4,096 clusters |
| Training | All float32 base vectors; no input subsampling or prequantization; nonhierarchical training |
| Iterations | At most 25; stopping thresholds below |
| Final assignment | Exact nearest centroid after the last update |
| Seed / fits | 42 / one fit per method and dataset |
| Queries | One fixed set of 1,000 per dataset; never used for training or stopping |

Both [datasets](bench.md#datasets-and-ground-truth) use the full base set and
1,000 queries. GIST uses supplied L2 ground truth; OpenAI uses exhaustive L2
search. Centroids are not normalized.

| Method | Additional settings |
| --- | --- |
| `rabitqlib.QGKMeans` | `quantization_bits=0`, degree 32, `ef_build=240`, `ef_search=16`, one graph refinement, `early_stop_threshold=0`, `FinalAssignmentMode.Exact` |
| `faiss.Kmeans` | CPU, `nredo=1`, `max_points_per_centroid=N`, `spherical=False`, `early_stop_threshold=0`; exact final `index.search(x, 1)` |
| `superkmeans.SuperKMeans` | v0.2.0, `quantizer="f32"`, `hierarchical=False`, `sampling_fraction=1`, `angular=False`, `use_blas_only=False`, `early_termination=True`, `tol=1e-4`; exact final `assign(x, centroids)` |

`N` is the full base count. These input settings do not disable internal
quantization, such as QGKMeans' one-bit neighbor codes. Methods retain their own training assignment algorithms
and initializers; the same seed does not guarantee identical starting centroids.

## Metrics

- **Time:** training plus exact final assignment, including initialization and
  internal transformations. QGKMeans includes assignment inside `train()`;
  Faiss and SuperKMeans report separate components in the CSV. Loading, evaluation,
  and saving are excluded.
- **WCSS:** sum of squared L2 distances to final assigned centroids, recomputed
  with float64 accumulation. Lower is better.
- **Recall@10:** fraction of exact top-10 neighbors in the nearest
  `floor(clusters × 1%)` clusters, averaged over queries: 40 probes for both
  datasets, following SuperKMeans.
  This is the recall available from exact scanning of those lists.

Exact assignments are checked on 128 base vectors using float64 distances,
outside timing.

## Reproduction

Follow the [installation instructions](bench.md#package-installation), then run:

```bash
python benchmarks/reproduce.py --benchmarks kmeans
```

Results, centroids, and assignments are saved in `results/benchmarks/kmeans/`.
Use `--datasets gist1m` or `--methods qgkmeans` for a subset, and a fresh
`--output-dir` to repeat measurements. See the [reproduction options](bench.md#data-and-reproduction).

Protocol references: SuperKMeans' [benchmark guide](https://github.com/cwida/SuperKMeans/blob/main/BENCHMARKING.md)
and [recall evaluator](https://github.com/cwida/SuperKMeans/blob/main/benchmarks/bench_utils.h).
