# 1. Tell pybind11 to use the modern FindPython module.
set(PYBIND11_FINDPYTHON ON)
if(PYTHON_EXECUTABLE AND NOT Python_EXECUTABLE)
    set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
endif()

# 2. Locate Python, NumPy, and pybind11 from the active Python environment.
# CMake 3.18+ can request only the module-development files. This avoids
# requiring libpython for embedding, which manylinux images do not provide.
if(CMAKE_VERSION VERSION_LESS 3.18)
    find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
else()
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module NumPy)
endif()
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

# Keep the OpenMP runtime discoverable in source installs and during wheel
# repair. Delocate copies it into the wheel and rewrites the loader path.
if(APPLE)
    set_property(TARGET _rabitqlib PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)
    # CMake omits link directories inside the source/build tree from the
    # automatic install RPATH, including project-local Python environments.
    foreach(runtime IN LISTS OpenMP_CXX_LIBRARIES)
        get_filename_component(runtime_dir "${runtime}" DIRECTORY)
        set_property(TARGET _rabitqlib APPEND PROPERTY INSTALL_RPATH "${runtime_dir}")
    endforeach()
    install(FILES licenses/LLVM-OpenMP.txt DESTINATION rabitqlib/licenses)
endif()

# 5. Map files to a 'rabitqlib' directory
# inside the target wheel, completely bypassing your local layout.
install(TARGETS _rabitqlib DESTINATION rabitqlib)
install(FILES __init__.py DESTINATION rabitqlib)
