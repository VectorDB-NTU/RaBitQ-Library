cmake_minimum_required(VERSION 3.20)
project(rabitqlib_core_consumer LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(RABITQ_COMPONENT "core" CACHE STRING "core or fastscan")
if(NOT RABITQ_COMPONENT MATCHES "^(core|fastscan)$")
    message(FATAL_ERROR "Unknown module: ${RABITQ_COMPONENT}")
endif()

set(INTEGRATION "PACKAGE" CACHE STRING "PACKAGE, ADD_SUBDIRECTORY, or FETCHCONTENT")
if(INTEGRATION STREQUAL "PACKAGE")
    find_package(rabitqlib CONFIG REQUIRED COMPONENTS ${RABITQ_COMPONENT})
    find_path(component_include rabitqlib/core/api.hpp
        PATHS "${CMAKE_PREFIX_PATH}/include" NO_DEFAULT_PATH REQUIRED)
    set(forbidden_paths defines.hpp core/pack_codes.hpp quantization simd index clustering third/hnswlib utils/rotator.hpp
        utils/buffer.hpp utils/hashset.hpp utils/visited_pool.hpp utils/visited_set.hpp
        utils/visited_set_epoch.hpp utils/visited_set_hash.hpp)
    if(RABITQ_COMPONENT STREQUAL "core")
        list(APPEND forbidden_paths fastscan)
    endif()
    foreach(path IN LISTS forbidden_paths)
        if(EXISTS "${component_include}/rabitqlib/${path}")
            message(FATAL_ERROR "Disabled or retired header installed: ${path}")
        endif()
    endforeach()
    # Compile every installed first-party header on its own, without source-tree
    # include paths. This verifies the component's complete template dependency closure.
    file(GLOB_RECURSE installed_headers RELATIVE "${component_include}"
        "${component_include}/rabitqlib/*.hpp")
    list(FILTER installed_headers EXCLUDE REGEX "/third/")
    set(header_sources)
    foreach(header IN LISTS installed_headers)
        string(MAKE_C_IDENTIFIER "${header}" identifier)
        set(source "${CMAKE_CURRENT_BINARY_DIR}/headers/${identifier}.cpp")
        file(WRITE "${source}" "#include <${header}>\n")
        list(APPEND header_sources "${source}")
    endforeach()
    add_library(component_header_check OBJECT ${header_sources})
    target_link_libraries(component_header_check PRIVATE rabitqlib::${RABITQ_COMPONENT})
else()
    set(RABITQ_BUILD_INDEX OFF CACHE BOOL "" FORCE)
    set(RABITQ_BUILD_FASTSCAN OFF CACHE BOOL "" FORCE)
    if(RABITQ_COMPONENT STREQUAL "fastscan")
        set(RABITQ_BUILD_FASTSCAN ON CACHE BOOL "" FORCE)
    endif()
    set(RABITQ_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
    set(RABITQ_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(RABITQ_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
    set(RABITQ_ENABLE_NATIVE_OPTIMIZATION OFF CACHE BOOL "" FORCE)
    if(INTEGRATION STREQUAL "ADD_SUBDIRECTORY")
        add_subdirectory("${RABITQ_SOURCE_DIR}" rabitqlib)
    elseif(INTEGRATION STREQUAL "FETCHCONTENT")
        include(FetchContent)
        # Test the current checkout without fetching an unrelated published revision.
        FetchContent_Declare(rabitqlib SOURCE_DIR "${RABITQ_SOURCE_DIR}")
        FetchContent_MakeAvailable(rabitqlib)
    else()
        message(FATAL_ERROR "Unknown integration: ${INTEGRATION}")
    endif()
    if(TARGET rabitqlib::index OR TARGET rabitqlib::rabitqlib)
        message(FATAL_ERROR "Core-only configuration exposes an index/full target")
    endif()
endif()

if(TARGET rabitqlib::rabitqlib OR TARGET rabitq_headers)
    message(FATAL_ERROR "Removed full-library compatibility target is still available")
endif()

get_target_property(core_links rabitqlib::core INTERFACE_LINK_LIBRARIES)
if(core_links MATCHES "fastscan|index|headers|::rabitqlib")
    message(FATAL_ERROR "Core has an ANN dependency: ${core_links}")
endif()

add_executable(core_consumer main.cpp)
target_link_libraries(core_consumer PRIVATE rabitqlib::${RABITQ_COMPONENT})
# Canonical module headers must not transitively include a higher module. Poison
# those include paths for this executable.
set(restricted_modules index clustering)
if(RABITQ_COMPONENT STREQUAL "core")
    list(APPEND restricted_modules fastscan)
endif()
foreach(module IN LISTS restricted_modules)
    file(GLOB_RECURSE restricted_headers RELATIVE "${RABITQ_SOURCE_DIR}/include"
        "${RABITQ_SOURCE_DIR}/include/rabitqlib/${module}/*.hpp")
    foreach(header IN LISTS restricted_headers)
        get_filename_component(directory "${header}" DIRECTORY)
        file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/forbidden/${directory}")
        file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/forbidden/${header}"
            "#error Unexpected higher-module header dependency\n")
    endforeach()
endforeach()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/forbidden/rabitqlib/defines.hpp"
    "#error Removed mixed-responsibility header must not be included\n")
target_include_directories(core_consumer BEFORE PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/forbidden")
add_executable(core_api_v1_consumer consumer.cpp)
target_link_libraries(core_api_v1_consumer PRIVATE rabitqlib::core)
if(DEFINED RABITQ_USE_OPENMP AND NOT RABITQ_USE_OPENMP)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/no-openmp")
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/no-openmp/omp.h"
        "#error OpenMP-free consumer must not require omp.h\n")
    foreach(consumer IN ITEMS core_consumer core_api_v1_consumer)
        target_include_directories(${consumer} BEFORE PRIVATE
            "${CMAKE_CURRENT_BINARY_DIR}/no-openmp")
    endforeach()
endif()
# The stable header must be self-contained without exposing Eigen or record maps.
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/api-forbidden/rabitqlib/third/Eigen")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/api-forbidden/rabitqlib/third/Eigen/Dense"
    "#error Stable API must not include Eigen\n")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/api-forbidden/rabitqlib/core")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/api-forbidden/rabitqlib/core/data_layout.hpp"
    "#error Stable API must not include record layouts\n")
target_include_directories(core_api_v1_consumer BEFORE PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/api-forbidden" "${CMAKE_CURRENT_BINARY_DIR}/forbidden")

enable_testing()
add_test(NAME core_api_v1_consumer COMMAND core_api_v1_consumer)
add_test(NAME core_consumer COMMAND core_consumer)

if(CMAKE_NM AND NOT MSVC)
    add_test(NAME core_symbols COMMAND "${CMAKE_COMMAND}"
        "-DNM=${CMAKE_NM}" "-DCORE=$<TARGET_FILE:rabitqlib::core>"
        "-DCONSUMER=$<TARGET_FILE:core_consumer>" "-DCOMPONENT=${RABITQ_COMPONENT}"
        "-DUSE_OPENMP=${RABITQ_USE_OPENMP}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/check_symbols.cmake")
endif()

add_executable(core_static_init core_static_init.cpp)
target_link_libraries(core_static_init PRIVATE rabitqlib::core)
add_test(NAME CoreStaticInitialization COMMAND core_static_init)
