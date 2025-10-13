cmake_minimum_required (VERSION 3.10)

project(RaBitQLib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


include_directories(${PROJECT_SOURCE_DIR}/include)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

SET(CMAKE_CXX_FLAGS  "-Wall -Ofast -Wextra -lrt -march=native -fpic -fopenmp -ftree-vectorize -fexceptions")

add_subdirectory(sample)

add_library(rabitq_headers INTERFACE)

target_include_directories(rabitq_headers INTERFACE ${PROJECT_SOURCE_DIR}/include)