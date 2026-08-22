# RaBitQ

The RaBitQ algorithm is a drop-in replacement of binary quantization and (uniform) scalar quantization, with its [1-bit version](https://arxiv.org/abs/2405.12497) (released in May 2024) and [multi-bit version](https://arxiv.org/abs/2409.09913) (released in Sep 2024), respectively.

<!-- It provides significantly better accuracy under the same space budget and is theoretically proven to be asymptotically optimal.  -->

<!-- For a given input data vector $\mathbf{x}$ and a bit budget $B$, the RaBitQ algorithm outputs a code vector $\mathbf{x}_u$ and a rescaling factor $\Delta_x$ such that we can estimate similarity metrics based on the code vector $\mathbf{x}_u$ and the rescaling factor $\Delta_x$ as accurately as possible. -->

The key advantages of RaBitQ include

- **High Accuracy with Tiny Space** - RaBitQ provides the state-of-the-art estimates under diverse space budgets. Its smallest representation uses a one-bit code per padded dimension plus per-vector factors.
- **Fast Distance Estimation** - The implementation uses bitwise operations and, for batched scans in IVF and SymphonyQG, [FastScan](https://arxiv.org/abs/1704.07355).
- **Theoretical Error Bound** - RaBitQ provides an asymptotically optimal error bound for the estimation of distances and inner product. The error bound can be used for reliable ordering and [reranking](reranking.md).


## Workflow 

The RaBitQ algorithm includes two steps:

1. **Random Rotation** - Sample a random rotation and apply it to all vectors (including the raw data vectors, the center vector and the raw query vectors). See [Rotator](rotator.md) for more details.

2. **Quantization** - After the random rotation, the quantization algorithm quantizes a vector of floating-point numbers into a vector of low-bit unsigned integers. See [Quantizer](quantizer.md) for more details.


After quantization, the library supports Euclidean distance and inner product
using the code vector $\mathbf{x}_u$ and its factors. Cosine similarity is
obtained by normalizing data and query vectors before using inner-product
search. See [Estimator](estimator.md) for details.
