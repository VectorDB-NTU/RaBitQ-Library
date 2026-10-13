# Installation

Platform requirements and source installation for RaBitQ Library **0.7.0**.
For the first Python example, see [Quick Start](quick_start.md).

## Requirements

| Platform | CPU baseline | Python wheel targets |
| --- | --- | --- |
| Linux x86-64 | AVX2 and FMA | CPython 3.11–3.14 |
| Linux ARM64 (AArch64) | NEON | CPython 3.11–3.14 |
| Windows x86-64 | AVX2 and FMA | CPython 3.11–3.14 |
| macOS 14+ ARM64 (Apple Silicon) | NEON | CPython 3.11–3.14 |

Source builds require a C++17 compiler and CMake 3.20 or newer. Full ANN/Python builds
also require OpenMP; core/FastScan can disable it with `RABITQ_USE_OPENMP=OFF`.
Windows uses MSVC (Visual Studio 2026 with the Desktop development with C++
workload and CMake 4.2+ for its generator). Apple Silicon uses AppleClang and
an external OpenMP runtime such as Homebrew `libomp`.

Linux ARM64 source builds and repaired wheels run on native AArch64 CI.
Linux ARM64 wheels carry `manylinux_2_27_aarch64` and
`manylinux_2_28_aarch64` tags.

<details>
<summary>CPU dispatch details</summary>

On x86-64, optional AVX-512 kernels require AVX2, FMA, and AVX-512F/BW/DQ;
MSVC builds also require AVX-512VL/CD. Popcount-specific kernels additionally
require AVX-512 VPOPCNTDQ. Detection checks CPU features and OS support for
the required register state. Missing AVX-512 features select the AVX2 backend.

On ARM64, the build excludes x86 kernels and uses NEON and portable scalar
implementations. Standard FastScan requires AVX2/FMA, a supported AVX-512
backend, or NEON; scalar fallbacks do not remove that requirement. A pure-core
build can use scalar kernels without the FastScan CPU baseline.

</details>

## Python wheels

```bash
python -m pip install "rabitqlib>=0.7.0"
```

For changes from 0.6.0, see the [0.7.0 upgrade notes](releases/0.7.0.md).

Wheels target the platforms above and require no compiler or CMake.
Linux ARM64 and macOS ARM64 wheels bundle OpenMP, so wheel users do not
need a separate OpenMP installation.
Release wheels disable native CPU tuning and select supported kernels at runtime.

## Get the source

For Python source builds or C++ use, obtain the 0.7.0 checkout once. With Git
installed, run:

```bash
git clone --branch v0.7.0 --depth 1 https://github.com/VectorDB-NTU/RaBitQ-Library.git
cd RaBitQ-Library
```

Run the source-build commands below from this repository root. If you already
have the matching checkout, use it directly.

## Build the Python bindings from source

Source builds require Python 3.11 or newer, a C++17 compiler, CMake 3.20 or
newer, and OpenMP. First [get the source](#get-the-source), then on Ubuntu or
Debian run:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libomp-dev
python -m pip install .
```

On Apple Silicon, use a native ARM64 Python environment. From the repository root:

```bash
brew install cmake ninja libomp
python -m pip install . -Ccmake.define.OpenMP_ROOT="$(brew --prefix libomp)"
```

On Windows, install the build tools listed above, then run `python -m pip install .`
from the repository root. For a portable source build on any supported platform,
also pass `-Ccmake.define.RABITQ_ENABLE_NATIVE_OPTIMIZATION=OFF`.
