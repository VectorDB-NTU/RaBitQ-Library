# Vendored Eigen

- Upstream: https://gitlab.com/libeigen/eigen
- Release: [5.0.1](https://gitlab.com/libeigen/eigen/-/tags/5.0.1)
- Source archive: https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.zip
- Archive SHA-256: `0dbb1f9e3aaad66f352c03227d8c983f6f0b49e0b07e71a7300f4abcc01aee12`
- Imported content: the complete upstream `Eigen/` directory, plus the root
  `COPYING.*` license files. The existing MPL-2.0 `LICENSE` is retained.
- Local code patches: none.

The previous snapshot matched the upstream 3.4.0 `Eigen/` directory after
normalizing line endings, with only an additional `LICENSE` file.

Keep upstream headers unmodified. RaBitQ's per-ISA namespace isolation lives in
`src/simd/matrix_kernels.hpp`; MSVC's `EIGEN_DONT_VECTORIZE` setting lives in the
root `CMakeLists.txt`. Preserve these settings when updating Eigen.
