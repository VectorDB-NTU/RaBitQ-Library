cmake_minimum_required(VERSION 3.20)

project(rabitqlib_consumer LANGUAGES CXX)

set(RABITQ_SOURCE_DIR "" CACHE PATH "Consume the source tree instead of an installed package")
option(RABITQ_EXPECT_SANITIZERS "Require sanitizer usage requirements from the package" OFF)
if(RABITQ_SOURCE_DIR)
    set(RABITQ_BUILD_SAMPLES OFF)
    set(RABITQ_BUILD_TESTS OFF)
    set(RABITQ_BUILD_PYTHON_BINDINGS OFF)
    add_subdirectory("${RABITQ_SOURCE_DIR}" rabitqlib)
    # Imported targets created in the child directory are not visible here.
    find_package(OpenMP REQUIRED)
else()
    find_package(rabitqlib CONFIG REQUIRED)
endif()

add_executable(rabitqlib_consumer main.cpp)
target_link_libraries(rabitqlib_consumer PRIVATE rabitqlib::rabitqlib)

enable_testing()
add_test(NAME rabitqlib_consumer COMMAND rabitqlib_consumer)

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
