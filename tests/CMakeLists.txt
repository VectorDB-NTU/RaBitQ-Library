# Fetch Google Test
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Automatically register all tests under unit and <other folders> folders
file(GLOB_RECURSE UNIT_TESTS CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/*_test.cpp
)
file(GLOB_RECURSE INTEG_TESTS CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/integration/*_test.cpp
)
file(GLOB_RECURSE COMMON_SRCS CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/common/*.cpp
)

# add executables
add_executable(rabitq_tests
    ${COMMON_SRCS}
    ${UNIT_TESTS}
    ${INTEG_TESTS}
)
target_include_directories(rabitq_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/common)
rabitq_configure_target(rabitq_tests)

# Link google test and rabitqlib headers
target_link_libraries(rabitq_tests
    PRIVATE
        GTest::gtest_main
        rabitq_headers
)

# Discover tests for CTest
include(GoogleTest)
gtest_discover_tests(rabitq_tests DISCOVERY_MODE PRE_TEST)

# Binding is initialized before main; test it in its own process.
add_test(NAME OpenMPBinding COMMAND rabitq_tests
    --gtest_filter=ThreadCount.BoundParallelForUsesRuntimePlacementAndPropagatesErrors)
set_tests_properties(OpenMPBinding PROPERTIES
    ENVIRONMENT "OMP_PROC_BIND=spread;OMP_PLACES=threads;OMP_NUM_THREADS=4")

# Substitute CPU predicates only in this executable to exercise real cached dispatch
# on a CPU without SIMD. The static library's detector object is not linked here.
add_executable(fastscan_no_simd_test dispatch/fastscan_no_simd_test.cpp)
target_link_libraries(fastscan_no_simd_test PRIVATE GTest::gtest_main rabitq_headers)
rabitq_configure_target(fastscan_no_simd_test)
gtest_discover_tests(fastscan_no_simd_test DISCOVERY_MODE PRE_TEST)

# Keep the global allocation-failure override isolated from the main suite.
add_executable(hnsw_add_allocation_test failures/hnsw_add_allocation_test.cpp)
target_link_libraries(hnsw_add_allocation_test PRIVATE GTest::gtest_main rabitq_headers)
rabitq_configure_target(hnsw_add_allocation_test)
gtest_discover_tests(hnsw_add_allocation_test DISCOVERY_MODE PRE_TEST)

# Message to verify files are found
message(STATUS "Discovered test files:")
foreach(TEST_SRC ${UNIT_TESTS} ${INTEG_TESTS})
    file(RELATIVE_PATH REL_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${TEST_SRC})
    message(STATUS "  - ${REL_PATH}")
endforeach()
