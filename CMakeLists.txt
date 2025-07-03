cmake_minimum_required (VERSION 3.10)

project(RaBitQLib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(cmake/cpu_features.cmake)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Building in debug mode")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Building in release mode")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2")
endif()

include_directories(${PROJECT_SOURCE_DIR}/rabitqlib)
include_directories(${PROJECT_SOURCE_DIR}/rabitqlib/third/simde)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

SET(CMAKE_CXX_FLAGS  "-Wall -Ofast -Wextra -lrt -march=native -fpic -fopenmp -ftree-vectorize -fexceptions -w")
SET(CMAKE_C_FLAGS  "-Wall -Ofast -Wextra -lrt -march=native -fpic -fopenmp -ftree-vectorize -fexceptions -w")

add_subdirectory(sample)

option(BUILD_TESTS "Build tests" ON)

if (BUILD_TESTS)
    add_subdirectory(contrib)
    add_subdirectory(tests)
endif()