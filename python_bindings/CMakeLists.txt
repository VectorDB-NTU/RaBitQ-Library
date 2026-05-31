cmake_minimum_required(VERSION 3.15)

# 1. Locate pybind11 matching the pip environment
find_package(pybind11 CONFIG REQUIRED)

# 2. Grab the NumPy include path from the active Python environment
execute_process(
    COMMAND "${Python_EXECUTABLE}" "-c" "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# 3. Find OpenMP to ensure parallel batched searches compile correctly
find_package(OpenMP REQUIRED)

# 4. Define the shared pybind11 module 
pybind11_add_module(_rabitqlib
    rabitq_bindings.cpp
    hnsw_bindings.cpp
    ivf_bindings.cpp
    symqg_bindings.cpp
)

# 5. Mirror the exact include paths from your old setup.py
target_include_directories(_rabitqlib PRIVATE 
    ${NUMPY_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}          # For bindings_common.hpp
    ${PROJECT_SOURCE_DIR}/include        # For rabitqlib core headers
)

# 6. Link OpenMP flags natively
target_link_libraries(_rabitqlib PRIVATE 
    OpenMP::OpenMP_CXX
)

# 7. Map files to a 'rabitqlib' directory 
# inside the target wheel, completely bypassing your local layout.
install(TARGETS _rabitqlib DESTINATION rabitqlib)
install(FILES __init__.py DESTINATION rabitqlib)