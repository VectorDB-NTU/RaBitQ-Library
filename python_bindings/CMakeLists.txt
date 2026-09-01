# 1. Tell pybind11 to use the modern FindPython module.
set(PYBIND11_FINDPYTHON ON)
if(PYTHON_EXECUTABLE AND NOT Python_EXECUTABLE)
    set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
endif()

# 2. Locate Python, NumPy, and pybind11 from the active Python environment.
find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(pybind11 CONFIG REQUIRED)

# 3. Define the shared pybind11 module.
pybind11_add_module(_rabitqlib
    rabitq_bindings.cpp
    hnsw_bindings.cpp
    ivf_bindings.cpp
    symqg_bindings.cpp
)

target_include_directories(_rabitqlib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
rabitq_enable_warnings(_rabitqlib)

# 4. Imported targets provide the matching Python, NumPy, and OpenMP settings.
target_link_libraries(_rabitqlib PRIVATE
    rabitq_headers
    Python::NumPy
)

# 5. Map files to a 'rabitqlib' directory
# inside the target wheel, completely bypassing your local layout.
install(TARGETS _rabitqlib DESTINATION rabitqlib)
install(FILES __init__.py DESTINATION rabitqlib)
