cmake_minimum_required(VERSION 3.15)

project(RaBitQLib VERSION 0.3.1 LANGUAGES CXX)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(RABITQ_BUILD_SAMPLES "Build C++ sample programs" ON)
option(RABITQ_BUILD_TESTS "Build C++ tests" OFF)
option(RABITQ_BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)
option(RABITQ_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(
    RABITQ_ENABLE_NATIVE_OPTIMIZATION
    "Optimize generic code for the build machine"
    ON
)

find_package(OpenMP REQUIRED)

add_library(rabitq_compile_options INTERFACE)
target_compile_features(rabitq_compile_options INTERFACE cxx_std_17)
target_compile_options(rabitq_compile_options INTERFACE
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANG_AND_ID:CXX,Clang,GNU>>:-O3>
    $<$<AND:$<BOOL:${RABITQ_ENABLE_NATIVE_OPTIMIZATION}>,$<COMPILE_LANG_AND_ID:CXX,Clang,GNU>>:-march=native>
)
target_link_libraries(rabitq_compile_options INTERFACE OpenMP::OpenMP_CXX)
if(RABITQ_ENABLE_SANITIZERS)
    target_compile_options(rabitq_compile_options INTERFACE
        $<$<COMPILE_LANG_AND_ID:CXX,Clang,GNU>:-fsanitize=address,undefined;-fno-omit-frame-pointer>
    )
    target_link_options(rabitq_compile_options INTERFACE
        $<$<CXX_COMPILER_ID:Clang,GNU>:-fsanitize=address,undefined>
    )
endif()

function(rabitq_enable_warnings target)
    target_compile_options(${target} PRIVATE
        "$<$<COMPILE_LANG_AND_ID:CXX,Clang,GNU>:-Wall;-Wextra>"
    )
endfunction()

set(RABITQ_COMMON_SOURCES
    src/utils/cpu_features.cpp
    src/simd/dispatch.cpp
)

set(RABITQ_AVX2_SOURCES
    src/simd/pack_excode_avx2.cpp
    src/simd/space_excode_avx2.cpp
    src/simd/space_avx2.cpp
    src/simd/fastscan_avx2.cpp
    src/simd/warmup_avx2.cpp
    src/simd/rotator_avx2.cpp
)

set(RABITQ_AVX512_SOURCES
    src/simd/pack_excode_avx512.cpp
    src/simd/space_excode_avx512.cpp
    src/simd/space_avx512.cpp
    src/simd/fastscan_avx512.cpp
    src/simd/rotator_avx512.cpp
)

set(RABITQ_AVX512_POPCNT_SOURCES
    src/simd/warmup_avx512.cpp
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

add_library(rabitq_core STATIC
    ${RABITQ_COMMON_SOURCES}
    ${RABITQ_AVX2_SOURCES}
    ${RABITQ_AVX512_SOURCES}
    ${RABITQ_AVX512_POPCNT_SOURCES}
    ${RABITQ_HNSW_AVX2_SOURCES}
    ${RABITQ_HNSW_AVX512_CORE_SOURCES}
    ${RABITQ_HNSW_AVX512_POPCNT_SOURCES}
)

target_include_directories(rabitq_core PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(rabitq_core PUBLIC rabitq_compile_options)
set_target_properties(rabitq_core PROPERTIES
    EXPORT_NAME core
    POSITION_INDEPENDENT_CODE ON
)
rabitq_enable_warnings(rabitq_core)

set_source_files_properties(${RABITQ_AVX2_SOURCES} PROPERTIES
    COMPILE_OPTIONS "-mavx2;-mfma"
)
set_source_files_properties(${RABITQ_AVX512_SOURCES} PROPERTIES
    COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512dq;-mfma"
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

add_library(rabitq_headers INTERFACE)
add_library(rabitqlib::rabitqlib ALIAS rabitq_headers)
target_include_directories(rabitq_headers INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(rabitq_headers INTERFACE rabitq_core)
set_target_properties(rabitq_headers PROPERTIES EXPORT_NAME rabitqlib)
set_target_properties(rabitq_compile_options PROPERTIES EXPORT_NAME compile_options)

install(TARGETS rabitq_headers rabitq_core rabitq_compile_options
    EXPORT rabitqlibTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

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
    "${CMAKE_CURRENT_BINARY_DIR}/rabitqlibConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/rabitqlibConfigVersion.cmake"
    DESTINATION "${rabitqlib_cmake_dir}"
)

if(RABITQ_BUILD_SAMPLES)
    add_subdirectory(sample/cpp)
endif()

if(RABITQ_BUILD_PYTHON_BINDINGS)
    add_subdirectory(python_bindings)
endif()

if(RABITQ_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
