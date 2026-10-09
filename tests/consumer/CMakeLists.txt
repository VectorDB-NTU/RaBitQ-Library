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
