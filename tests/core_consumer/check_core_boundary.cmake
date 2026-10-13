cmake_minimum_required(VERSION 3.20)
if(NOT RABITQ_SOURCE_DIR)
    message(FATAL_ERROR "RABITQ_SOURCE_DIR is required")
endif()

# Header poisoning alone cannot detect a FastScan declaration copied into core.
# Guard the format-specific vocabulary as well as direct upward includes.
file(GLOB_RECURSE core_files
    "${RABITQ_SOURCE_DIR}/include/rabitqlib/core/*.hpp"
    "${RABITQ_SOURCE_DIR}/include/rabitqlib/utils/*.hpp"
    "${RABITQ_SOURCE_DIR}/src/core/*.cpp"
    "${RABITQ_SOURCE_DIR}/src/simd/*.cpp"
    "${RABITQ_SOURCE_DIR}/src/simd/*.hpp")
foreach(path IN LISTS core_files)
    file(READ "${path}" contents)
    if(contents MATCHES "fastscan::|BatchDataMap|kBatchSize|kPerm0|one_bit_batch_code|quantize_(one|split|qg)_batch")
        message(FATAL_ERROR "FastScan batch-format dependency in core: ${path}")
    endif()
    if(contents MATCHES "# *include[^\n]*rabitqlib/(fastscan|index|clustering)/")
        message(FATAL_ERROR "Upward module include in core: ${path}")
    endif()
endforeach()
