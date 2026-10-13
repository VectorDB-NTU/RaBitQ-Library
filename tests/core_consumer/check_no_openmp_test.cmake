cmake_minimum_required(VERSION 3.20)

file(MAKE_DIRECTORY "${WORK_DIR}")
set(database_path "${WORK_DIR}/compile_commands.json")
set(database_template [=[
[
  {"directory": "/openmp", "file": "/tmp/-fopenmp/input.cpp",
   "command": "c++ -I\"/tmp/openmp-off/include dir\" -c \"/tmp/-fopenmp/input.cpp\""},
  {"directory": "/tmp/openmp-off", "file": "input.cpp", "output": "/openmp",
   "command": "c++ -DNAME=/openmp -I/tmp/-fopenmp @option@ -c input.cpp"}
]
]=])

function(check_case option expect_failure)
    string(CONFIGURE "${database_template}" database @ONLY)
    file(WRITE "${database_path}" "${database}")
    execute_process(COMMAND "${CMAKE_COMMAND}"
        "-DCOMPILE_COMMANDS=${database_path}"
        -P "${CMAKE_CURRENT_LIST_DIR}/check_no_openmp.cmake"
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if(expect_failure)
        if(result EQUAL 0 OR NOT "${output}${error}" MATCHES "OpenMP compiler flag in disabled build")
            message(FATAL_ERROR "Failed to reject ${option}: ${output}${error}")
        endif()
    elseif(NOT result EQUAL 0)
        message(FATAL_ERROR "Rejected paths without OpenMP flags: ${output}${error}")
    endif()
endfunction()

check_case("" FALSE)
foreach(flag IN ITEMS -fopenmp -fopenmp=libomp -fopenmp-simd /openmp /openmp:llvm /openmp:experimental)
    check_case("${flag}" TRUE)
    # JSON escapes preserve quotes around a single command-line argument.
    check_case("\\\"${flag}\\\"" TRUE)
endforeach()
message(STATUS "OpenMP flag detection accepts paths and rejects actual compiler flags")
