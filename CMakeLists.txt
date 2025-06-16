cmake_minimum_required (VERSION 3.10)

if(NOT CMAKE_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER /usr/bin/g++)
endif()
project(RaBitQLib LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


include_directories(${PROJECT_SOURCE_DIR}/rabitqlib)
add_library(rabitq INTERFACE)

target_include_directories(rabitq INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}/rabitqlib
)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

SET(CMAKE_CXX_FLAGS  "-Wall -Ofast -Wextra -mavx512f -lrt -march=native -fpic -fopenmp -ftree-vectorize -fexceptions")

# add_subdirectory(sample)