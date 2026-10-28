cmake_minimum_required(VERSION 3.20)

project(RaBitQLib VERSION 0.7.2 LANGUAGES CXX)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(RABITQ_BUILD_FASTSCAN "Build batch scoring and FastScan SIMD kernels" ON)
option(RABITQ_BUILD_INDEX "Build ANN indexes and clustering" ON)
option(RABITQ_BUILD_SAMPLES "Build C++ sample programs" ON)
option(RABITQ_BUILD_TESTS "Build C++ tests" OFF)
option(RABITQ_BUILD_BENCHMARKS "Build component quantization and hot-path benchmarks" OFF)
option(RABITQ_BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)
option(RABITQ_USE_OPENMP "Use OpenMP for parallel helpers (required by ANN indexes)" ON)
option(RABITQ_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
# Enable native tuning for local builds; release wheels explicitly set this OFF.
option(
    RABITQ_ENABLE_NATIVE_OPTIMIZATION
    "Optimize generic code for the build machine"
    ON
)

if(RABITQ_BUILD_INDEX AND NOT RABITQ_BUILD_FASTSCAN)
    message(FATAL_ERROR "RABITQ_BUILD_INDEX requires RABITQ_BUILD_FASTSCAN=ON")
endif()

if(RABITQ_BUILD_PYTHON_BINDINGS AND NOT RABITQ_BUILD_INDEX)
    message(FATAL_ERROR "RABITQ_BUILD_PYTHON_BINDINGS requires RABITQ_BUILD_INDEX=ON")
endif()

if(RABITQ_BUILD_INDEX AND NOT RABITQ_USE_OPENMP)
    message(FATAL_ERROR "ANN indexes require RABITQ_USE_OPENMP=ON; disable RABITQ_BUILD_INDEX for an OpenMP-free numerical library")
endif()

if(RABITQ_USE_OPENMP)
    include(cmake/rabitqOpenMP.cmake)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

add_library(rabitq_compile_options INTERFACE)
target_compile_features(rabitq_compile_options INTERFACE cxx_std_17)
target_compile_definitions(rabitq_compile_options INTERFACE
    RABITQ_USE_OPENMP=$<BOOL:${RABITQ_USE_OPENMP}>
    # Eigen-backed TLS/storage may be allocated by native code and freed by a
    # generic consumer. Keep its layout and aligned-allocation policy identical.
    EIGEN_MAX_ALIGN_BYTES=64 EIGEN_MAX_STATIC_ALIGN_BYTES=64)
if(RABITQ_USE_OPENMP)
    target_link_libraries(rabitq_compile_options INTERFACE OpenMP::OpenMP_CXX)
endif()
if(MSVC)
    # MSVC spells the GNU restrict extension differently. Public headers use
    # __restrict__, so consumers need the same definition as the library.
    target_compile_definitions(rabitq_compile_options INTERFACE __restrict__=__restrict)
    # Eigen's inline code must have one ISA-independent definition across the
    # generic, AVX2, and AVX-512 translation units on MSVC.
    target_compile_definitions(rabitq_compile_options INTERFACE EIGEN_DONT_VECTORIZE)
endif()
if(RABITQ_ENABLE_SANITIZERS)
    # Keep sanitizer instrumentation consistent across public inline code and
    # compiled components, including their runtime linkage.
    target_compile_options(rabitq_compile_options INTERFACE
        "$<$<COMPILE_LANG_AND_ID:CXX,Clang,GNU>:-fsanitize=address,undefined;-fno-omit-frame-pointer>"
    )
    target_link_options(rabitq_compile_options INTERFACE
        $<$<CXX_COMPILER_ID:Clang,GNU>:-fsanitize=address,undefined>
    )
endif()

function(rabitq_configure_target target)
    # Apply project build policy only to first-party targets, never consumers.
    target_compile_options(${target} PRIVATE
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANG_AND_ID:CXX,Clang,GNU>>:-O3>
        $<$<AND:$<BOOL:${RABITQ_ENABLE_NATIVE_OPTIMIZATION}>,$<COMPILE_LANG_AND_ID:CXX,Clang,GNU>>:-march=native>
        "$<$<COMPILE_LANG_AND_ID:CXX,Clang,GNU>:-Wall;-Wextra>"
    )
endfunction()

set(RABITQ_CORE_SOURCES
    src/core/api.cpp
    src/utils/cpu_features.cpp
    src/utils/thread_count.cpp
    src/simd/dispatch.cpp
    src/simd/space_generic.cpp
    src/simd/quantization_generic.cpp
    src/simd/matrix_generic.cpp
)

set(RABITQ_CORE_AVX2_SOURCES
    src/simd/quantization_avx2.cpp
    src/simd/pack_excode_avx2.cpp
    src/simd/space_excode_avx2.cpp
    src/simd/space_avx2.cpp
    src/simd/matrix_avx2.cpp
    src/simd/warmup_avx2.cpp
    src/simd/rotator_avx2.cpp
)

set(RABITQ_CORE_AVX512_SOURCES
    src/simd/quantization_avx512.cpp
    src/simd/pack_excode_avx512.cpp
    src/simd/space_excode_avx512.cpp
    src/simd/space_avx512.cpp
    src/simd/matrix_avx512.cpp
    src/simd/rotator_avx512.cpp
)

set(RABITQ_AVX512_POPCNT_SOURCES
    src/simd/warmup_avx512.cpp
)

set(RABITQ_FASTSCAN_AVX2_SOURCES
    src/fastscan/fastscan_avx2.cpp
    src/fastscan/estimator_avx2.cpp
)

set(RABITQ_FASTSCAN_AVX512_SOURCES
    src/fastscan/fastscan_avx512.cpp
    src/fastscan/estimator_avx512.cpp
)

set(RABITQ_HNSW_AVX2_SOURCES
    src/index/hnsw_search_avx2.cpp
)

set(RABITQ_HNSW_AVX512_CORE_SOURCES
    src/index/hnsw_search_avx512_core.cpp
)

set(RABITQ_HNSW_AVX512_POPCNT_SOURCES
    src/index/hnsw_search_avx512_popcnt.cpp
)

# Select source groups from the target architecture, never the build host's ISA.
set(RABITQ_TARGET_PROCESSOR "${CMAKE_SYSTEM_PROCESSOR}")
if(APPLE AND CMAKE_OSX_ARCHITECTURES)
    set(RABITQ_TARGET_PROCESSOR "${CMAKE_OSX_ARCHITECTURES}")
endif()
string(TOLOWER "${RABITQ_TARGET_PROCESSOR}" RABITQ_TARGET_PROCESSOR)
if(RABITQ_TARGET_PROCESSOR MATCHES "^(x86_64|amd64)$")
    set(RABITQ_ARCH_SOURCES
        ${RABITQ_CORE_AVX2_SOURCES}
        ${RABITQ_CORE_AVX512_SOURCES}
        ${RABITQ_AVX512_POPCNT_SOURCES}
    )
    set(RABITQ_FASTSCAN_ARCH_SOURCES
        ${RABITQ_FASTSCAN_AVX2_SOURCES}
        ${RABITQ_FASTSCAN_AVX512_SOURCES}
    )
    set(RABITQ_INDEX_ARCH_SOURCES
        ${RABITQ_HNSW_AVX2_SOURCES}
        ${RABITQ_HNSW_AVX512_CORE_SOURCES}
        ${RABITQ_HNSW_AVX512_POPCNT_SOURCES}
    )
elseif(RABITQ_TARGET_PROCESSOR MATCHES "^(arm64|aarch64)$")
    set(RABITQ_ARCH_SOURCES
        src/simd/space_neon.cpp
        src/simd/space_packed_neon.cpp
        src/simd/rotator_neon.cpp
        src/simd/query_neon.cpp
    )
    set(RABITQ_FASTSCAN_ARCH_SOURCES src/fastscan/fastscan_neon.cpp)
    set(RABITQ_INDEX_ARCH_SOURCES src/index/hnsw_search_neon.cpp)
else()
    message(FATAL_ERROR "Supported targets are x86-64 and ARM64 (single architecture)")
endif()

add_library(rabitq_core STATIC
    ${RABITQ_CORE_SOURCES}
    ${RABITQ_ARCH_SOURCES}
    src/simd/portable_generic.cpp
)
add_library(rabitqlib::core ALIAS rabitq_core)

target_include_directories(rabitq_core PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(rabitq_core PUBLIC rabitq_compile_options)
set_target_properties(rabitq_core PROPERTIES
    EXPORT_NAME core
    POSITION_INDEPENDENT_CODE ON
)
rabitq_configure_target(rabitq_core)

if(MSVC)
    set_source_files_properties(
        ${RABITQ_CORE_AVX2_SOURCES} ${RABITQ_FASTSCAN_AVX2_SOURCES}
        ${RABITQ_HNSW_AVX2_SOURCES}
        PROPERTIES COMPILE_OPTIONS "/arch:AVX2"
    )
    # MSVC enables the complete F/CD/VL/BW/DQ group. Keep the optional
    # backend's runtime guard in cpu_features.cpp aligned with this flag.
    set_source_files_properties(
        ${RABITQ_CORE_AVX512_SOURCES} ${RABITQ_FASTSCAN_AVX512_SOURCES}
        ${RABITQ_AVX512_POPCNT_SOURCES}
        ${RABITQ_HNSW_AVX512_CORE_SOURCES} ${RABITQ_HNSW_AVX512_POPCNT_SOURCES}
        PROPERTIES COMPILE_OPTIONS "/arch:AVX512"
    )
else()
    set_source_files_properties(
        ${RABITQ_CORE_AVX2_SOURCES} ${RABITQ_FASTSCAN_AVX2_SOURCES}
        PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma"
    )
    set_source_files_properties(
        ${RABITQ_CORE_AVX512_SOURCES} ${RABITQ_FASTSCAN_AVX512_SOURCES}
        PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512dq;-mfma"
    )
    set_source_files_properties(${RABITQ_AVX512_POPCNT_SOURCES} PROPERTIES
        COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512dq;-mavx512vpopcntdq;-mfma"
    )
    set_source_files_properties(${RABITQ_HNSW_AVX2_SOURCES} PROPERTIES
        COMPILE_OPTIONS "-mavx2;-mfma"
    )
    set_source_files_properties(${RABITQ_HNSW_AVX512_CORE_SOURCES} PROPERTIES
        COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512dq;-mavx2;-mfma"
    )
    set_source_files_properties(${RABITQ_HNSW_AVX512_POPCNT_SOURCES} PROPERTIES
        COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512dq;-mavx512vpopcntdq;-mfma"
    )
endif()

set(RABITQ_INSTALL_TARGETS rabitq_core rabitq_compile_options)
if(RABITQ_BUILD_FASTSCAN)
    add_library(rabitq_fastscan STATIC
        src/fastscan/fastscan_dispatch.cpp
        src/fastscan/fastscan_generic.cpp
        src/fastscan/estimator_generic.cpp
        ${RABITQ_FASTSCAN_ARCH_SOURCES}
    )
    add_library(rabitqlib::fastscan ALIAS rabitq_fastscan)
    target_link_libraries(rabitq_fastscan PUBLIC rabitq_core)
    set_target_properties(rabitq_fastscan PROPERTIES
        EXPORT_NAME fastscan
        POSITION_INDEPENDENT_CODE ON
    )
    rabitq_configure_target(rabitq_fastscan)
    list(APPEND RABITQ_INSTALL_TARGETS rabitq_fastscan)
endif()
if(RABITQ_BUILD_INDEX)
    add_library(rabitq_index STATIC
        src/index/hnsw.cpp
        src/index/ivf.cpp
        src/index/ivf_initializer.cpp
        src/index/ivf_search.cpp
        src/index/qg.cpp
        src/index/qg_builder.cpp
        src/index/pipnn.cpp
        src/index/hnsw_dispatch.cpp
        src/index/hnsw_search_generic.cpp
        ${RABITQ_INDEX_ARCH_SOURCES}
    )
    add_library(rabitqlib::index ALIAS rabitq_index)
    target_link_libraries(rabitq_index PUBLIC rabitq_fastscan)
    set_target_properties(rabitq_index PROPERTIES
        EXPORT_NAME index
        POSITION_INDEPENDENT_CODE ON
    )
    rabitq_configure_target(rabitq_index)

    list(APPEND RABITQ_INSTALL_TARGETS rabitq_index)
endif()
set_target_properties(rabitq_compile_options PROPERTIES EXPORT_NAME compile_options)

install(TARGETS ${RABITQ_INSTALL_TARGETS}
    EXPORT rabitqlibTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
# Install only the enabled components and their header dependencies. Detail headers
# remain necessary for template implementations, but are not supported public APIs.
set(rabitq_include_dir "${CMAKE_INSTALL_INCLUDEDIR}/rabitqlib")
install(DIRECTORY include/rabitqlib/core DESTINATION "${rabitq_include_dir}")
install(FILES
    include/rabitqlib/utils/bitops.hpp
    include/rabitqlib/utils/cpu_features.hpp
    include/rabitqlib/utils/memory.hpp
    include/rabitqlib/utils/openmp.hpp
    include/rabitqlib/utils/packed.hpp
    include/rabitqlib/utils/path.hpp
    include/rabitqlib/utils/space.hpp
    include/rabitqlib/utils/tools.hpp
    include/rabitqlib/utils/warmup_space.hpp
    DESTINATION "${rabitq_include_dir}/utils")
install(DIRECTORY include/rabitqlib/third/Eigen
    DESTINATION "${rabitq_include_dir}/third")
if(RABITQ_BUILD_FASTSCAN)
    install(DIRECTORY include/rabitqlib/fastscan DESTINATION "${rabitq_include_dir}")
endif()
if(RABITQ_BUILD_INDEX)
    install(DIRECTORY include/rabitqlib/index include/rabitqlib/clustering
        DESTINATION "${rabitq_include_dir}")
    install(FILES include/rabitqlib/utils/io.hpp
        include/rabitqlib/utils/stopw.hpp DESTINATION "${rabitq_include_dir}/utils")
    install(DIRECTORY include/rabitqlib/third/hnswlib
        DESTINATION "${rabitq_include_dir}/third")
endif()

set(rabitqlib_cmake_dir "${CMAKE_INSTALL_LIBDIR}/cmake/rabitqlib")
configure_package_config_file(
    cmake/rabitqlibConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/rabitqlibConfig.cmake"
    INSTALL_DESTINATION "${rabitqlib_cmake_dir}"
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/rabitqlibConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
)
install(EXPORT rabitqlibTargets
    FILE rabitqlibTargets.cmake
    NAMESPACE rabitqlib::
    DESTINATION "${rabitqlib_cmake_dir}"
)
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/rabitqOpenMP.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/rabitqlibConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/rabitqlibConfigVersion.cmake"
    DESTINATION "${rabitqlib_cmake_dir}"
)

if(RABITQ_BUILD_TESTS OR RABITQ_BUILD_BENCHMARKS)
    enable_testing()
endif()

if(RABITQ_BUILD_SAMPLES)
    add_subdirectory(sample/cpp)
endif()

if(RABITQ_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks/core)
    add_subdirectory(benchmarks/hotpaths)
endif()

if(RABITQ_BUILD_PYTHON_BINDINGS)
    add_subdirectory(python_bindings)
endif()

if(RABITQ_BUILD_TESTS)
    add_subdirectory(tests)
endif()
