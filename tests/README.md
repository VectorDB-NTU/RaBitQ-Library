# RaBitQ Testing Framework

This directory contains the comprehensive testing framework for the RaBitQ library 

## Prerequisites

- CMake 3.10 or higher
- C++17 compatible compiler (GCC, Clang, or MSVC)
- Google Test (automatically downloaded via CMake FetchContent)

### Installing CMake 

For macos
```bash
brew install cmake
```

for (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install cmake
```

## Building and Running Tests

### Quick Start

From the project root directory:

```bash
# Create build directory
mkdir build bin
cd build

# Configure with tests enabled (tests are OFF by default)
cmake .. -DRABITQ_BUILD_TESTS=ON

# Build the tests
make -j$(nproc)

# Run all tests
./tests/rabitq_tests

# Or use CTest for detailed output
ctest --output-on-failure
```

### Building without Tests

By default, tests are **not built**. If you want to build only the library:

```bash
cmake ..
```


## Test Structure

```
tests/
├── CMakeLists.txt              # Automatic test discovery & suite configuration
├── main.cpp                    # Test runner entry point
├── common/                     # Test utilities and helpers
│   ├── test_data.hpp           # Test data generation utilities
│   ├── test_data.cpp
│   └── test_helpers.hpp       # Custom assertions and helpers
├── unit/                       # Unit tests (auto-discovered)
├── integration/                # Integration tests (auto-discovered)
└── benchmark/                  # Performance benchmarks (to be added)
```

## Test Coverage

