cmake_minimum_required(VERSION 3.10)
# Fetch Google Test
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# Automatically register all tests under unit and <other folders> folders
file(GLOB_RECURSE UNIT_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/unit/*_test.cpp)
file(GLOB_RECURSE INTEG_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/integration/*_test.cpp)
file(GLOB_RECURSE COMMON_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/common/*.cpp)

# add executables
add_executable(rabitq_tests
    ${COMMON_SRCS}
    ${UNIT_TESTS}
    ${INTEG_TESTS}
)

# Link google test, rabitqlib headers, and OpenMP if available
target_link_libraries(rabitq_tests
    gtest
    gtest_main
    rabitq_headers
    pthread
)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  message(STATUS "Tests: OpenMP found, linking OpenMP::OpenMP_CXX")
endif()

if(OpenMP_CXX_FOUND)
  target_link_libraries(rabitq_tests PRIVATE OpenMP::OpenMP_CXX)
endif()

# Discover tests for CTest
include(GoogleTest)
gtest_discover_tests(rabitq_tests)

# Message to verify files are found
message(STATUS "Discovered test files:")
foreach(TEST_SRC ${UNIT_TESTS} ${INTEG_TESTS})
  file(RELATIVE_PATH REL_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${TEST_SRC})
  message(STATUS "  - ${REL_PATH}")
endforeach()