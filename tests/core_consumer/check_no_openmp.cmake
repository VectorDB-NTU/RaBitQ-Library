cmake_minimum_required(VERSION 3.20)

# CMake's compilation database stores each invocation in the command field.
# Inspect parsed arguments, not paths or other JSON fields containing /openmp.
file(READ "${COMPILE_COMMANDS}" database)
string(JSON count LENGTH "${database}")
if(count GREATER 0)
    math(EXPR last "${count} - 1")
    foreach(index RANGE ${last})
        string(JSON command GET "${database}" ${index} command)
        separate_arguments(arguments NATIVE_COMMAND "${command}")
        foreach(argument IN LISTS arguments)
            if(argument MATCHES "^-fopenmp($|[=-])" OR argument MATCHES "^/openmp($|:)")
                message(FATAL_ERROR "OpenMP compiler flag in disabled build: ${argument}")
            endif()
        endforeach()
    endforeach()
endif()
