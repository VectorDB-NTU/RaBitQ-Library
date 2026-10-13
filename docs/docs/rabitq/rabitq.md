# RaBitQ

RaBitQ is an alternative to binary and uniform scalar quantization. Its
[one-bit](https://arxiv.org/abs/2405.12497) and
[multi-bit](https://doi.org/10.1145/3725413) variants encode each padded coordinate
with a low-bit unsigned integer and store per-vector factors for distance estimation.

## Workflow

1. **Rotate.** Apply the same sampled transformation to data, centroids and queries.
   See [Rotation](rotator.md).
2. **Encode.** Quantize each rotated data vector relative to its centroid.
   See [Quantization](quantizer.md).
3. **Estimate.** Combine the codes, factors and rotated query to estimate squared
   L2 or inner-product distance (`1 - dot`). See [Estimation](estimator.md).

For cosine search, normalize data and queries before rotation and use inner product.
The probabilistic error intervals can guide [reranking](reranking.md), but do not
make approximate search exact. Recall depends on the data and search settings.

For integration into your own storage, start with the [stable core API](core_api.md).
For a complete index, start with the [Quick start](../quick_start.md).
