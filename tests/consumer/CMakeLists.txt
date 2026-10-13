cmake_minimum_required(VERSION 3.20)

project(rabitqlib_consumer LANGUAGES CXX)

set(RABITQ_SOURCE_DIR "" CACHE PATH "Consume the source tree instead of an installed package")
option(RABITQ_EXPECT_SANITIZERS "Require sanitizer usage requirements from the package" OFF)
find_package(OpenMP REQUIRED)
if(RABITQ_SOURCE_DIR)
    set(RABITQ_BUILD_SAMPLES OFF)
    set(RABITQ_BUILD_TESTS OFF)
    set(RABITQ_BUILD_PYTHON_BINDINGS OFF)
    add_subdirectory("${RABITQ_SOURCE_DIR}" rabitqlib)
    set(rabitq_options_target rabitq_compile_options)
else()
    find_package(rabitqlib CONFIG REQUIRED)
    set(rabitq_options_target rabitqlib::compile_options)
endif()

# Check the raw public interface as well as evaluated consumer options below.
# The optimization generators are inactive with AppleClang, so an evaluated-only
# check would miss an accidental declaration that affects GCC and upstream Clang.
get_target_property(rabitq_public_options ${rabitq_options_target} INTERFACE_COMPILE_OPTIONS)
foreach(forbidden IN ITEMS -O3 -march=native -Wall -Wextra)
    if("${rabitq_public_options}" MATCHES "${forbidden}")
        message(FATAL_ERROR "Unexpected public compile option: ${forbidden}")
    endif()
endforeach()

add_executable(rabitqlib_consumer main.cpp)
target_link_libraries(rabitqlib_consumer PRIVATE rabitqlib::index)
if(TARGET rabitqlib::rabitqlib OR TARGET rabitq_headers)
    message(FATAL_ERROR "Removed full-library compatibility target is still exported")
endif()

add_executable(index_header_boundary
    index_headers_main.cpp hnsw_headers.cpp ivf_headers.cpp symqg_headers.cpp)
target_link_libraries(index_header_boundary PRIVATE rabitqlib::index)
# Ordinary index declarations must not parse encoding, scoring or persistence implementations.
foreach(header IN ITEMS
    core/encoder.hpp core/estimator.hpp core/query.hpp core/rotator.hpp
    fastscan/encoder.hpp fastscan/estimator.hpp fastscan/query.hpp
    index/hnsw/hnsw_quant.hpp index/ivf/initializer.hpp
    index/detail/buffer.hpp index/detail/visited_set.hpp
    index/detail/visited_set_hash.hpp index/detail/visited_set_epoch.hpp
    index/symqg/detail/search_scratch.hpp
    index/detail/parallel_for.hpp third/Eigen/Core third/Eigen/Dense
    third/hnswlib/hnswlib.h)
    get_filename_component(directory "${header}" DIRECTORY)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/forbidden/rabitqlib/${directory}")
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/forbidden/rabitqlib/${header}"
        "#error Index public header imports an implementation dependency\n")
endforeach()
target_include_directories(index_header_boundary BEFORE PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/forbidden")
add_library(hnsw_quant_header_boundary OBJECT hnsw_quant_headers.cpp)
target_link_libraries(hnsw_quant_header_boundary PRIVATE rabitqlib::index)
add_library(symqg_scratch_header_boundary OBJECT symqg_scratch_headers.cpp)
target_link_libraries(symqg_scratch_header_boundary PRIVATE rabitqlib::index)

add_library(index_types_boundary OBJECT index_types.cpp)
target_link_libraries(index_types_boundary PRIVATE rabitqlib::index)
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/types-forbidden/rabitqlib/third/Eigen")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/types-forbidden/rabitqlib/third/Eigen/Dense"
    "#error Index identifiers and candidate values must not depend on Eigen\n")
target_include_directories(index_types_boundary BEFORE PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/types-forbidden")

enable_testing()
add_test(NAME rabitqlib_consumer COMMAND rabitqlib_consumer)
add_test(NAME index_header_boundary COMMAND index_header_boundary)

# Inspect evaluated, transitive usage requirements rather than just the direct
# properties of the public interface target. This works with multi-config builds.
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/consumer-options-$<CONFIG>-$<COMPILE_LANGUAGE>.cmake"
    CONTENT "
set(consumer_options [==[$<TARGET_PROPERTY:rabitqlib_consumer,COMPILE_OPTIONS>]==])
set(consumer_features [==[$<TARGET_PROPERTY:rabitqlib_consumer,COMPILE_FEATURES>]==])
set(consumer_definitions [==[$<TARGET_PROPERTY:rabitqlib_consumer,COMPILE_DEFINITIONS>]==])
set(consumer_link_options [==[$<TARGET_PROPERTY:rabitqlib_consumer,LINK_OPTIONS>]==])
set(openmp_options [==[$<TARGET_PROPERTY:OpenMP::OpenMP_CXX,INTERFACE_COMPILE_OPTIONS>]==])
set(expect_msvc ${MSVC})
set(expect_sanitizers ${RABITQ_EXPECT_SANITIZERS})
"
    TARGET rabitqlib_consumer
)
add_test(NAME rabitqlib_consumer_options
    COMMAND "${CMAKE_COMMAND}"
        "-DOPTIONS_FILE=${CMAKE_CURRENT_BINARY_DIR}/consumer-options-$<CONFIG>-CXX.cmake"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/check_options.cmake"
)
