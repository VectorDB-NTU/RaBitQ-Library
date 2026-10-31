# Random rotation
Random rotation is the preprocessing step used by RaBitQ's distance estimators.
Apply the same sampled transformation to data, centroids and queries.
RaBitQ Library provides FHT/Kac rotation (the default) and a dense random
orthogonal transformation.

The default FHT/Kac transform is an efficient structured approximation to a
random orthogonal rotation. [SymphonyQG, Section 3.1.4](https://dl.acm.org/doi/abs/10.1145/3709730)
describes using a Fast Hadamard-based Fast Johnson–Lindenstrauss Transform to
reduce rotation work from $O(D^2)$ to $O(D\log D)$. This library combines
Hadamard transforms, random sign flips and the Givens steps described below.
The transform preserves norms; its sampling distribution approximates the
uniform random rotation used in the RaBitQ analysis.

The default FHT/Kac rotator supports `float` and accepts input dimensions from 64 through 65,536 inclusive. It pads to a multiple of 32 and uses AVX or NEON kernels
for power-of-two blocks through `2^16`. Allocate the
output using `rotator->size()`, which can exceed the input dimension.

RaBitQKMeans, QGKMeans, SymphonyQG, IVF-RaBitQ, and HNSW-RaBitQ also use multiples of 32
for all their supported bit widths.
Index loading preserves the padded domain used when the index was built.
Reducing the padded dimension changes the sampled rotation and quantization even
with the same seed. Newly built indexes can therefore have different recall and
latency; smaller padding does not guarantee faster search or unchanged accuracy.
When restoring a standalone rotator saved with earlier 64-dimension padding,
pass that original padded dimension to `choose_rotator` before loading its state.
The saved sign bytes do not contain dimension metadata.

The rotation interfaces are in `rabitqlib/core/rotator.hpp`.

### Example

```cpp
#include <memory>
#include <vector>
#include "rabitqlib/core/rotator.hpp"

int main() {
    const size_t dim = 769;  // deliberately not a multiple of 32
    std::unique_ptr<rabitqlib::Rotator<float>> rotator(
        rabitqlib::choose_rotator<float>(dim, rabitqlib::RotatorType::FhtKacRotator)
    );
    std::vector<float> x(dim, 1.0F);
    std::vector<float> x_prime(rotator->size());  // 800 elements
    rotator->rotate(x.data(), x_prime.data());
}
```

`choose_rotator` returns an owning pointer; wrap it in `std::unique_ptr`.
Use the same rotator for data, centroids, and queries. Pass `rotator->size()`
to quantization and estimation routines that operate on rotated vectors.

For a random orthogonal transformation, select `RotatorType::MatrixRotator`.
It supports `float` and `double` and keeps the input dimension by default. Pass a
padded dimension explicitly if subsequent packed-code operations require a multiple
of 32. Stored space and per-vector work are proportional to `dim * padded_dim`
(quadratic when both dimensions are equal).
Constructing the random matrix also performs QR factorization. The `seed` argument
of `choose_rotator` seeds both rotators. FHT/Kac signs come from a fixed mapping of
`std::mt19937` output, so a seed gives the same rotation on every platform and
standard library. A seeded MatrixRotator is reproducible only with the same standard
library and build, because `std::normal_distribution` and QR rounding differ between
them. Save and restore the actual rotator state rather than relying on a seed to
recreate a matrix rotation.

## FFHT + Kac’s walk
On x86-64, transforms use the shared AVX intrinsics in `src/simd/fht_kernels.hpp`
within the AVX2 or AVX-512 backend selected at runtime. GCC, Clang, and MSVC
use the same implementation without inline assembly or MASM. ARM64 uses
NEON kernels in `src/simd/rotator_neon.cpp`; a portable scalar implementation
is also available. The backends preserve the FFHT butterfly order,
normalization, four sign-flip passes, and saved rotation state.

This method is a combination of the well-known Fast Johnson-Lindenstrauss Transformation algorithms based on [Fast Hadamard Transform](https://www.cs.princeton.edu/~chazelle/pubs/FJLT-sicomp09.pdf) and ideas in [Kac’s Walk](https://projecteuclid.org/journals/annals-of-applied-probability/volume-27/issue-1/Kacs-walk-on-n-sphere-mixes-in-nlog-n-steps/10.1214/16-AAP1214.full).
It first samples 4 sequences of random signs (i.e., Rademacher random variables).
Let `trunc_dim` be the largest power of two not exceeding the original input
dimension (`dim`), and `padded_dim` the working dimension.

When `trunc_dim == padded_dim`, each of the four rounds only flips signs and
applies normalized FFHT to the whole vector; no Givens/Kac step is performed.
Otherwise, each round performs the following steps:

1. Flip its coordinates with the $i$-th sequence of sampled random signs.
2. Apply normalized FFHT on the first/last `trunc_dim` coordinates (alternately) of the padded vector.
3. Apply Givens rotation with a fixed angle $\theta = \frac{\pi}{4}$ to the 1st and the 2nd halves of coordinates.

| Space Consumption | Time Complexity |
| ----------------- | --------------- |
| $4D$ binary values ($4D$ bits) | $O(D\log D)$ |

This implementation is based on the [FFHT library](https://github.com/FALCONN-LIB/FFHT) developed by Alexandr Andoni, Piotr Indyk, Thijs Laarhoven, Ilya Razenshteyn and Ludwig Schmidt.

## Random orthogonal transformation
This method samples a Gaussian matrix and orthogonalizes it with QR decomposition.
It then multiplies each vector by the stored transform. The table describes stored
space and per-vector rotation work when input and output dimensions are both $D$;
QR construction is separate setup work.

| Space Consumption | Time Complexity |
| ----------------- | --------------- |
| $D^2$ floating-point numbers | $O(D^2)$ |
