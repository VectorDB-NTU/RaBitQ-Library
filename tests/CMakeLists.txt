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

# Automatically register the unit and kernel-reference tests.
file(GLOB_RECURSE UNIT_TESTS CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/*_test.cpp
)
file(GLOB_RECURSE COMMON_SRCS CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/common/*.cpp
)

# Core-only builds must not compile index/clustering tests or their consumers.
set(RABITQ_TEST_TARGET rabitqlib::core)
if(RABITQ_BUILD_FASTSCAN)
    set(RABITQ_TEST_TARGET rabitqlib::fastscan)
else()
    list(FILTER UNIT_TESTS EXCLUDE REGEX "/fastscan/")
endif()
if(RABITQ_BUILD_INDEX)
    set(RABITQ_TEST_TARGET rabitqlib::index)
else()
    list(FILTER UNIT_TESTS EXCLUDE REGEX "/(index|clustering)/")
    list(FILTER UNIT_TESTS EXCLUDE REGEX "/(neon_hnsw|thread_count|buffer|visited_set)_test[.]cpp$")
endif()

# add executables
add_executable(rabitq_tests
    ${COMMON_SRCS}
    ${UNIT_TESTS}
)
target_include_directories(rabitq_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/common)
if(RABITQ_BUILD_INDEX)
    target_include_directories(rabitq_tests PRIVATE ${PROJECT_SOURCE_DIR}/src/index)
endif()
set_source_files_properties(unit/rabitqlib/index/historical_persistence_test.cpp
    PROPERTIES COMPILE_DEFINITIONS
    "RABITQ_TEST_FIXTURE_ROOT=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures/persistence\"")
rabitq_configure_target(rabitq_tests)

# Link google test and rabitqlib headers
target_link_libraries(rabitq_tests
    PRIVATE
        GTest::gtest_main
        ${RABITQ_TEST_TARGET}
)

# Discover tests for CTest
include(GoogleTest)
gtest_discover_tests(rabitq_tests DISCOVERY_MODE PRE_TEST)

add_test(NAME CoreModuleBoundary COMMAND "${CMAKE_COMMAND}"
    "-DRABITQ_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/core_consumer/check_core_boundary.cmake")

add_test(NAME CoreOpenMPFlagCheck COMMAND "${CMAKE_COMMAND}"
    "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/openmp-flag-check"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/core_consumer/check_no_openmp_test.cmake")

add_executable(core_api_v1_consumer fixtures/core_api_v1/consumer.cpp)
target_link_libraries(core_api_v1_consumer PRIVATE rabitqlib::core)
add_test(NAME CoreApiV1Consumer COMMAND core_api_v1_consumer)

foreach(mode IN ITEMS dispatch scalar)
    set(target core_api_no_alloc_${mode})
    add_executable(${target} allocation/core_api_no_alloc.cpp)
    target_link_libraries(${target} PRIVATE rabitqlib::core)
    rabitq_configure_target(${target})
    if(mode STREQUAL "scalar")
        target_compile_definitions(${target} PRIVATE RABITQ_FORCE_SCALAR)
    endif()
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_compile_definitions(${target} PRIVATE RABITQ_WRAP_MALLOC)
        target_link_options(${target} PRIVATE
            -Wl,--wrap=malloc -Wl,--wrap=calloc -Wl,--wrap=realloc
            -Wl,--wrap=aligned_alloc -Wl,--wrap=posix_memalign)
    endif()
    add_test(NAME ${target} COMMAND ${target})
endforeach()

if(RABITQ_BUILD_INDEX)
    # Binding is initialized before main; test it in its own process.
    add_test(NAME OpenMPBinding COMMAND rabitq_tests
        --gtest_filter=ThreadCount.BoundParallelForUsesRuntimePlacementAndPropagatesErrors)
    set_tests_properties(OpenMPBinding PROPERTIES
        ENVIRONMENT "OMP_PROC_BIND=spread;OMP_PLACES=threads;OMP_NUM_THREADS=4")

    # Substitute CPU predicates only in this executable to exercise real cached dispatch
    # on a CPU without SIMD. The static library's detector object is not linked here.
    add_executable(fastscan_no_simd_test dispatch/fastscan_no_simd_test.cpp
        dispatch/fastscan_backend_test.cpp)
    target_include_directories(fastscan_no_simd_test PRIVATE ${PROJECT_SOURCE_DIR}/src/fastscan)
    target_link_libraries(fastscan_no_simd_test PRIVATE GTest::gtest_main rabitqlib::index)
    rabitq_configure_target(fastscan_no_simd_test)
    gtest_discover_tests(fastscan_no_simd_test DISCOVERY_MODE PRE_TEST)

    add_executable(construction_failure_test failures/construction_failure_test.cpp)
    target_link_libraries(construction_failure_test PRIVATE GTest::gtest_main rabitqlib::index)
    rabitq_configure_target(construction_failure_test)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_compile_definitions(construction_failure_test PRIVATE RABITQ_WRAP_CONSTRUCTION_ALLOCATIONS)
        target_link_options(construction_failure_test PRIVATE
            -Wl,--wrap=malloc -Wl,--wrap=aligned_alloc -Wl,--wrap=free)
    endif()
    gtest_discover_tests(construction_failure_test DISCOVERY_MODE PRE_TEST)

    # Keep the global allocation-failure override isolated from the main suite.
    add_executable(hnsw_add_allocation_test failures/hnsw_add_allocation_test.cpp)
    target_link_libraries(hnsw_add_allocation_test PRIVATE GTest::gtest_main rabitqlib::index)
    rabitq_configure_target(hnsw_add_allocation_test)
    gtest_discover_tests(hnsw_add_allocation_test DISCOVERY_MODE PRE_TEST)

endif()

# Message to verify files are found
message(STATUS "Discovered test files:")
foreach(TEST_SRC ${UNIT_TESTS})
    file(RELATIVE_PATH REL_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${TEST_SRC})
    message(STATUS "  - ${REL_PATH}")
endforeach()

add_executable(core_static_init core_consumer/core_static_init.cpp)
target_link_libraries(core_static_init PRIVATE rabitqlib::core)
add_test(NAME CoreStaticInitialization COMMAND core_static_init)

if(RABITQ_BUILD_FASTSCAN)
    add_executable(fastscan_backend_test dispatch/fastscan_backend_test.cpp)
    target_include_directories(fastscan_backend_test PRIVATE ${PROJECT_SOURCE_DIR}/src/fastscan)
    target_link_libraries(fastscan_backend_test PRIVATE GTest::gtest_main rabitqlib::fastscan)
    rabitq_configure_target(fastscan_backend_test)
    gtest_discover_tests(fastscan_backend_test DISCOVERY_MODE PRE_TEST TEST_PREFIX "Native.")

    add_executable(modules_static_init dispatch/modules_static_init.cpp)
    target_link_libraries(modules_static_init PRIVATE ${RABITQ_TEST_TARGET})
    if(RABITQ_BUILD_INDEX)
        target_compile_definitions(modules_static_init PRIVATE RABITQ_TEST_INDEX)
    endif()
    add_test(NAME ModulesStaticInitialization COMMAND modules_static_init)
endif()

# Run allocation replacement in a separate process; this also exercises core-only builds.
add_executable(compact_codes_failure_test failures/compact_codes_failure_test.cpp)
target_link_libraries(compact_codes_failure_test PRIVATE GTest::gtest_main rabitqlib::core)
if(RABITQ_BUILD_FASTSCAN)
    target_link_libraries(compact_codes_failure_test PRIVATE rabitqlib::fastscan)
    target_compile_definitions(compact_codes_failure_test PRIVATE RABITQ_TEST_FASTSCAN)
endif()
rabitq_configure_target(compact_codes_failure_test)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_compile_definitions(compact_codes_failure_test PRIVATE RABITQ_WRAP_COMPACT_ALLOCATIONS)
    target_link_options(compact_codes_failure_test PRIVATE -Wl,--wrap=malloc -Wl,--wrap=realloc)
endif()
gtest_discover_tests(compact_codes_failure_test DISCOVERY_MODE PRE_TEST)
