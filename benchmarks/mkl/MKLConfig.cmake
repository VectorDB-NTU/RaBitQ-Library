# MKL runtime package for local Linux/GCC Python baseline builds.
# Set MKL_THREADING_LAYER=GNU at runtime to share the caller's OpenMP runtime.
find_path(MKL_INCLUDE_DIR mkl.h PATH_SUFFIXES mkl REQUIRED)
find_library(MKL_RUNTIME_LIBRARY NAMES mkl_rt REQUIRED)
if(NOT TARGET MKL::MKL)
    add_library(MKL::MKL INTERFACE IMPORTED)
    set_target_properties(MKL::MKL PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${MKL_RUNTIME_LIBRARY}"
    )
endif()
set(MKL_FOUND TRUE)
