# Run with -DRABITQ_SOURCE_DIR=<checkout> -DWORK_DIR=<external directory> -P run.cmake.
cmake_minimum_required(VERSION 3.20)
if(NOT RABITQ_SOURCE_DIR OR NOT WORK_DIR)
    message(FATAL_ERROR "RABITQ_SOURCE_DIR and WORK_DIR are required")
endif()

function(run)
    execute_process(COMMAND ${ARGV} RESULT_VARIABLE result)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "Command failed (${result}): ${ARGV}")
    endif()
endfunction()

set(configure_options -DCMAKE_BUILD_TYPE=Release -DRABITQ_ENABLE_NATIVE_OPTIMIZATION=OFF)
if(NOT DEFINED RABITQ_USE_OPENMP)
    set(RABITQ_USE_OPENMP ON)
endif()
list(APPEND configure_options "-DRABITQ_USE_OPENMP=${RABITQ_USE_OPENMP}")
if(NOT RABITQ_USE_OPENMP)
    list(APPEND configure_options -DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=TRUE)
endif()
if(DEFINED FETCHCONTENT_SOURCE_DIR_GOOGLETEST)
    list(APPEND configure_options
        "-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=${FETCHCONTENT_SOURCE_DIR_GOOGLETEST}")
endif()
if(DEFINED OpenMP_ROOT)
    list(APPEND configure_options "-DOpenMP_ROOT=${OpenMP_ROOT}")
endif()
foreach(component IN ITEMS core fastscan)
    set(scan OFF)
    if(component STREQUAL "fastscan")
        set(scan ON)
    endif()
    set(work "${WORK_DIR}/${component}")
    file(MAKE_DIRECTORY "${work}/consumer")
    file(COPY "${RABITQ_SOURCE_DIR}/tests/core_consumer/CMakeLists.txt"
        "${RABITQ_SOURCE_DIR}/tests/core_consumer/check_symbols.cmake"
        "${RABITQ_SOURCE_DIR}/tests/core_consumer/core_static_init.cpp"
        "${RABITQ_SOURCE_DIR}/tests/fixtures/core_api_v1/consumer.cpp"
        "${RABITQ_SOURCE_DIR}/sample/cpp/${component}/main.cpp"
        DESTINATION "${work}/consumer")
    run("${CMAKE_COMMAND}" -S "${RABITQ_SOURCE_DIR}" -B "${work}/producer"
        ${configure_options} -DRABITQ_BUILD_INDEX=OFF
        "-DRABITQ_BUILD_FASTSCAN=${scan}" -DRABITQ_BUILD_TESTS=ON)
    run("${CMAKE_COMMAND}" --build "${work}/producer" --config Release --parallel 2)
    run("${CMAKE_CTEST_COMMAND}" --test-dir "${work}/producer" -C Release
        --output-on-failure --timeout 120)
    # Relocate the initial install; subsequent runs update that same test prefix.
    if(EXISTS "${work}/relocated")
        run("${CMAKE_COMMAND}" --install "${work}/producer" --config Release
            --prefix "${work}/relocated")
    else()
        run("${CMAKE_COMMAND}" --install "${work}/producer" --config Release
            --prefix "${work}/install")
        file(RENAME "${work}/install" "${work}/relocated")
    endif()
    foreach(method IN ITEMS ADD_SUBDIRECTORY FETCHCONTENT PACKAGE)
        set(build "${work}/${method}")
        set(consumer_source "${RABITQ_SOURCE_DIR}")
        if(method STREQUAL "FETCHCONTENT")
            # Exercise native Windows path spelling on every host.
            string(REPLACE "/" "\\" consumer_source "${consumer_source}")
        endif()
        run("${CMAKE_COMMAND}" -S "${work}/consumer" -B "${build}"
            ${configure_options} "-DINTEGRATION=${method}"
            "-DRABITQ_COMPONENT=${component}"
            "-DRABITQ_SOURCE_DIR=${consumer_source}"
            "-DCMAKE_PREFIX_PATH=${work}/relocated")
        run("${CMAKE_COMMAND}" --build "${build}" --config Release --parallel 2)
        run("${CMAKE_CTEST_COMMAND}" --test-dir "${build}" -C Release --output-on-failure)
    endforeach()
    foreach(build IN ITEMS producer ADD_SUBDIRECTORY FETCHCONTENT)
        if(EXISTS "${work}/${build}/compile_commands.json")
            file(READ "${work}/${build}/compile_commands.json" commands)
            if(NOT RABITQ_USE_OPENMP)
                run("${CMAKE_COMMAND}"
                    "-DCOMPILE_COMMANDS=${work}/${build}/compile_commands.json"
                    -P "${CMAKE_CURRENT_LIST_DIR}/check_no_openmp.cmake")
            endif()
            if(commands MATCHES "src/index/|python_bindings/|clustering/")
                message(FATAL_ERROR "ANN/Python source compiled by ${component}/${build}")
            endif()
            if(component STREQUAL "core" AND commands MATCHES "src/fastscan/|src/simd/(fastscan|estimator)_")
                message(FATAL_ERROR "FastScan source compiled by core/${build}")
            endif()
        endif()
    endforeach()
endforeach()

# Configuration must fail early for combinations that cannot provide full functionality.
execute_process(COMMAND "${CMAKE_COMMAND}" -S "${RABITQ_SOURCE_DIR}"
    -B "${WORK_DIR}/invalid-python" -DRABITQ_BUILD_INDEX=OFF
    -DRABITQ_BUILD_PYTHON_BINDINGS=ON RESULT_VARIABLE result
    OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(result EQUAL 0 OR NOT "${output}${error}" MATCHES "requires RABITQ_BUILD_INDEX=ON")
    message(FATAL_ERROR "Core-only Python configuration was not rejected clearly")
endif()

file(MAKE_DIRECTORY "${WORK_DIR}/require-index")
file(WRITE "${WORK_DIR}/require-index/CMakeLists.txt"
    "cmake_minimum_required(VERSION 3.20)\nproject(require_index LANGUAGES CXX)\nfind_package(rabitqlib CONFIG REQUIRED COMPONENTS index)\n")
execute_process(COMMAND "${CMAKE_COMMAND}" -S "${WORK_DIR}/require-index"
    -B "${WORK_DIR}/missing-index" ${configure_options}
    "-DCMAKE_PREFIX_PATH=${WORK_DIR}/core/relocated" RESULT_VARIABLE result
    OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(result EQUAL 0 OR NOT "${output}${error}" MATCHES "rabitqlib")
    message(FATAL_ERROR "Core-only installation accepted the missing index component")
endif()

execute_process(COMMAND "${CMAKE_COMMAND}" -S "${RABITQ_SOURCE_DIR}"
    -B "${WORK_DIR}/invalid-index" -DRABITQ_BUILD_FASTSCAN=OFF
    -DRABITQ_BUILD_INDEX=ON RESULT_VARIABLE result
    OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(result EQUAL 0 OR NOT "${output}${error}" MATCHES "requires RABITQ_BUILD_FASTSCAN=ON")
    message(FATAL_ERROR "Index without FastScan was not rejected clearly")
endif()
if(NOT RABITQ_USE_OPENMP)
    execute_process(COMMAND "${CMAKE_COMMAND}" -S "${RABITQ_SOURCE_DIR}"
        -B "${WORK_DIR}/invalid-openmp" -DRABITQ_BUILD_INDEX=ON
        -DRABITQ_BUILD_FASTSCAN=ON -DRABITQ_USE_OPENMP=OFF RESULT_VARIABLE result
        OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if(result EQUAL 0 OR NOT "${output}${error}" MATCHES "ANN indexes require RABITQ_USE_OPENMP=ON")
        message(FATAL_ERROR "Unsupported OpenMP-free ANN build was not rejected clearly")
    endif()
endif()
file(WRITE "${WORK_DIR}/require-index/CMakeLists.txt"
    "cmake_minimum_required(VERSION 3.20)\nproject(require_fastscan LANGUAGES CXX)\nfind_package(rabitqlib CONFIG REQUIRED COMPONENTS fastscan)\n")
execute_process(COMMAND "${CMAKE_COMMAND}" -S "${WORK_DIR}/require-index"
    -B "${WORK_DIR}/missing-fastscan" ${configure_options}
    "-DCMAKE_PREFIX_PATH=${WORK_DIR}/core/relocated" RESULT_VARIABLE result
    OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(result EQUAL 0 OR NOT "${output}${error}" MATCHES "rabitqlib")
    message(FATAL_ERROR "Core-only installation accepted the missing FastScan component")
endif()
run("${CMAKE_COMMAND}" -S "${RABITQ_SOURCE_DIR}/benchmarks/core"
    -B "${WORK_DIR}/benchmark" ${configure_options}
    "-DCMAKE_PREFIX_PATH=${WORK_DIR}/core/relocated")
run("${CMAKE_COMMAND}" --build "${WORK_DIR}/benchmark" --config Release --parallel 2)
run("${CMAKE_CTEST_COMMAND}" --test-dir "${WORK_DIR}/benchmark" -C Release
    --output-on-failure)
message(STATUS "Core and FastScan external consumers and core benchmark passed")
