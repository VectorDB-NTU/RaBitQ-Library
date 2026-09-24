# AppleClang needs an external OpenMP runtime. The same hints work with
# Homebrew libomp and a project environment's llvm-openmp package.
if(APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    find_path(RABITQ_OPENMP_INCLUDE omp.h HINTS "${OpenMP_ROOT}" ENV CONDA_PREFIX
        PATH_SUFFIXES include)
    find_library(RABITQ_OPENMP_LIBRARY omp HINTS "${OpenMP_ROOT}" ENV CONDA_PREFIX
        PATH_SUFFIXES lib)
    if(RABITQ_OPENMP_INCLUDE AND RABITQ_OPENMP_LIBRARY)
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_CXX_INCLUDE_DIR "${RABITQ_OPENMP_INCLUDE}")
        set(OpenMP_CXX_LIB_NAMES omp)
        set(OpenMP_omp_LIBRARY "${RABITQ_OPENMP_LIBRARY}")
    endif()
endif()
