# Random Rotation
Random rotation (i.e., Johnson Lindenstrauss Transformation) is a crucial step to ensure robust performance and theoretical error bounds of RaBitQ. It is applied to all vectors (including raw data vectors, center vectors and raw query vectors) as a preprocessing step. This section describes the usage of the random rotation.

RaBitQLib provides two types of random rotation. All implementations sample and store a random rotation at first. Then they apply the sampled random rotation to every input vector and return the rotated vector. 

By default, the library uses the `FFHT + Kac’s Walk` method. 

The default rotator accepts input dimensions from 64 through 65,536 inclusive,
including 16,384. It pads to a multiple of 64 and uses AVX or NEON kernels
for power-of-two blocks through `2^16`. Allocate the
output using `rotator->size()`, which can exceed the input dimension.

The implementation can be found in `rotator.hpp`.

```css
.
├── rabitqlib
│   ├── ...
│   └── utils
│       ├── ...
│       └── rotator.hpp
└── ...
```

### Example

```cpp
// Initialize a rotator
// Version 1 - the default rotator
// vectors are padded to the smallest multiple of 64
// storage - 4D bits, time - O(D * log D)
rabitqlib::Rotator<float>* rotator = rabitqlib::choose_rotator<float>(
    dim = dim, 
    RotatorType type = RotatorType::FhtKacRotator);

// Initialize a rotator
// Version 2 - the random orthogonal transformation
// vectors are padded to the smallest multiple of 64
// storage - D * D floats, time - O(D * D)
rabitqlib::Rotator<float>* rotator = rabitqlib::choose_rotator<float>(
    dim = dim, 
    RotatorType type = RotatorType::MatrixRotator);

// Apply a rotator to a vector
size_t dim = 768;
std::vector<float> x(dim);
std::vector<float> x_prime(dim);
... 
rotator -> rotate(x.data(), x_prime.data())



```

## FFHT + Kac’s Walk
### Description
On x86-64, transforms use the shared AVX intrinsics in `src/simd/fht_kernels.hpp`
within the AVX2 or AVX-512 backend selected at runtime. GCC, Clang, and MSVC
use the same implementation without inline assembly or MASM. ARM64 uses
NEON kernels in `src/simd/rotator_neon.cpp`; a portable scalar implementation
is also available. The backends preserve the FFHT butterfly order,
normalization, four sign-flip passes, and saved rotation state.

This method is a combination of the well-known Fast Johnson-Lindenstrauss Transformation algorithms based on [Fast Hadamard Transform](https://www.cs.princeton.edu/~chazelle/pubs/FJLT-sicomp09.pdf) and ideas in [Kac’s Walk](https://projecteuclid.org/journals/annals-of-applied-probability/volume-27/issue-1/Kacs-walk-on-n-sphere-mixes-in-nlog-n-steps/10.1214/16-AAP1214.full). 
It first samples 4 sequences of random signs (i.e., Rademacher random variables). Then for each vector, it repeats the following procedures 4 times.

1. Flip its coordinates with the $i$-th sequence of sampled random signs. 
2. Apply FFHT on the first/last $2^k$ coordinates (alternately), where $2^k$ is the maximum power of 2 that is less than or equal to the dimensionality of the vector.
3. Apply Givens rotation with a fixed angle $\theta = \frac{\pi}{4}$ to the 1st and the 2nd halves of coordinates.

The following table summarizes the space and time complexity of this method.

| Space Consumption | Time Complexity |
| ----------------- | --------------- |
| $4D$ binary values ($4D$ bits) | $O(D\log D)$ |

This implementation is based on the [FFHT library](https://github.com/FALCONN-LIB/FFHT) developed by Alexandr Andoni, Piotr Indyk, Thijs Laarhoven, Ilya Razenshteyn and Ludwig Schmidt. 


## Random Orthogonal Transformation
### Description
This method is the classical Johnson-Lindenstrauss Transformation. It first samples a random gaussian matrix and orthogonalizes it with QR decomposition. Then it multiplies the matrix to every vector.

The following table summarizes the space and time complexity of this method.

| Space Consumption | Time Complexity |
| ----------------- | --------------- |
| $D^2$ floating-point numbers | $O(D^2)$ |
