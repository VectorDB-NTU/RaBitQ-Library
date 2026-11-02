# Graph benchmark

RaBitQ-Library results were measured with **0.8.2 on 2026-11-02 (Asia/Singapore)**.
Each baseline retains its latest measurement and was not rerun.
See the [source revisions, environments and comparison limits](bench.md#package-installation).

Settings and results for eight configurations. See the [overview](bench.md) for
hardware, datasets, and measurement definitions.

## Reading the results

Each figure groups methods with similar storage per dimension into panels;
method names are defined under [index configurations](#index-configurations).
Curves use one search thread and one query per call over all 1,000 queries.
Indexes are built with 48 threads and reused for the query sweeps.
See [figure definitions](bench.md#reading-the-figures) and
[indexing metrics](bench.md#reading-the-indexing-tables).

## GIST1M

![GIST1M graphs QPS–recall trade-off](figures/graphs-gist1m-qps-recall.svg){ .chart }

### Indexing time and size

<!-- indexing:gist1m:start -->
| Method | Train s | Construct s | Total s | Save s | Index GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| SymphonyQG raw | 0.00 | 34.61 | 34.61 | 3.23 | 7.51 |
| SymphonyQG 4-bit | 0.00 | 31.64 | 31.64 | 1.75 | 4.39 |
| SymphonyQG 8-bit | 0.00 | 33.09 | 33.09 | 2.13 | 4.84 |
| HNSW-RaBitQ 1+4 | 1.25 | 70.55 | 71.80 | 0.33 | 0.72 |
| HNSW-RaBitQ 1+8 | 1.25 | 73.42 | 74.68 | 0.52 | 1.16 |
| Faiss HNSW Flat | 0.00 | 78.94 | 78.94 | 1.61 | 3.71 |
| Faiss HNSW SQ4 | 1.06 | 55.11 | 56.17 | 0.24 | 0.58 |
| Faiss HNSW SQ8 | 1.08 | 47.12 | 48.20 | 0.42 | 1.03 |
<!-- indexing:gist1m:end -->

## DBpedia

![DBpedia graphs QPS–recall trade-off](figures/graphs-dbpedia-qps-recall.svg){ .chart }

### Indexing time and size

<!-- indexing:dbpedia:start -->
| Method | Train s | Construct s | Total s | Save s | Index GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| SymphonyQG raw | 0.00 | 55.98 | 55.98 | 5.41 | 11.79 |
| SymphonyQG 4-bit | 0.00 | 52.85 | 52.85 | 2.90 | 6.80 |
| SymphonyQG 8-bit | 0.00 | 53.23 | 53.23 | 3.13 | 7.51 |
| HNSW-RaBitQ 1+4 | 1.87 | 111.76 | 113.63 | 0.54 | 1.05 |
| HNSW-RaBitQ 1+8 | 1.55 | 123.92 | 125.48 | 0.81 | 1.76 |
| Faiss HNSW Flat | 0.00 | 147.53 | 147.53 | 2.69 | 5.85 |
| Faiss HNSW SQ4 | 1.74 | 101.82 | 103.57 | 0.36 | 0.85 |
| Faiss HNSW SQ8 | 1.79 | 80.66 | 82.45 | 0.64 | 1.56 |
<!-- indexing:dbpedia:end -->

## Index configurations

SymphonyQG uses maximum degree 32, `ef_construction=200`, `init="pipnn"`, and one
refinement iteration. Both HNSW implementations use `M=16` and
`efConstruction=200`, giving a base-layer degree bound of 32.

| Method | Stored representation | Additional construction settings |
| --- | --- | --- |
| SymphonyQG raw | `nbits=32`, raw vectors | Degree 32, PIPNN initialization |
| SymphonyQG 4-bit | `nbits=4` | Degree 32, PIPNN initialization |
| SymphonyQG 8-bit | `nbits=8` | Degree 32, PIPNN initialization |
| HNSW-RaBitQ 1+4 | `nbits=5` | `M=16`, seed 42, 16 residual centroids |
| HNSW-RaBitQ 1+8 | `nbits=9` | `M=16`, seed 42, 16 residual centroids |
| Faiss HNSW Flat | `HNSW16` | Float32 vectors |
| Faiss HNSW SQ4 | `HNSW16_SQ4` | Four-bit scalar quantization |
| Faiss HNSW SQ8 | `HNSW16_SQ8` | Eight-bit scalar quantization |

HNSW-RaBitQ trains 16 centroids with RaBitQKMeans (up to 10 iterations, seed 42,
exact final assignment, spherical training for DBpedia's inner-product metric).
Training is reported separately and included in total construction time.
SymphonyQG 4/8-bit and HNSW-RaBitQ 1+4/1+8 use different encodings; serialized
sizes include their full storage costs.

All graph methods sweep `ef=16,32,64,128,256,512,1024` (Faiss calls this
`efSearch`).

## Download measurements

[QPS–recall CSV](figures/qps-recall-points.csv) · [Indexing CSV](figures/indexing.csv)
