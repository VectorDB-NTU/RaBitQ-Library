# 1. Tell pybind11 to use the modern FindPython module.
set(PYBIND11_FINDPYTHON ON)
if(PYTHON_EXECUTABLE AND NOT Python_EXECUTABLE)
    set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
endif()

# 2. Locate Python, NumPy, and pybind11 from the active Python environment.
# Request module-development files without requiring libpython for embedding,
# which manylinux images do not provide.
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module NumPy)
find_package(pybind11 CONFIG REQUIRED)

# 3. Define the shared pybind11 module.
pybind11_add_module(_rabitqlib
    rabitq_bindings.cpp
    hnsw_bindings.cpp
    ivf_bindings.cpp
    symqg_bindings.cpp
    kmeans_bindings.cpp
)

add_library(rabitqlib::python ALIAS _rabitqlib)

target_include_directories(_rabitqlib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
rabitq_configure_target(_rabitqlib)

# 4. Imported targets provide the matching Python, NumPy, and OpenMP settings.
target_link_libraries(_rabitqlib PRIVATE
    rabitqlib::index
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

# Linux release wheels bundle GNU OpenMP during auditwheel repair.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    install(FILES
        licenses/GNU-OpenMP.txt
        licenses/GPL-3.0.txt
        licenses/GCC-Runtime-Library-Exception.txt
        DESTINATION rabitqlib/licenses
    )
endif()

# 5. Map files to a 'rabitqlib' directory
# inside the target wheel, completely bypassing your local layout.
install(TARGETS _rabitqlib DESTINATION rabitqlib)
install(FILES __init__.py __init__.pyi py.typed DESTINATION rabitqlib)

if(RABITQ_BUILD_BENCHMARKS)
    # Benchmark the production conversion helper without importing or replacing
    # the installed package. This module is never installed in the wheel.
    pybind11_add_module(rabitq_ids_benchmark NO_EXTRAS ../benchmarks/hotpaths/ids.cpp)
    target_include_directories(rabitq_ids_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(rabitq_ids_benchmark PRIVATE rabitqlib::index)
    rabitq_configure_target(rabitq_ids_benchmark)
    set_target_properties(rabitq_ids_benchmark PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)
    add_test(NAME PythonIdsHotpath COMMAND "${Python_EXECUTABLE}"
        "${PROJECT_SOURCE_DIR}/benchmarks/hotpaths/ids.py"
        --module-dir "$<TARGET_FILE_DIR:rabitq_ids_benchmark>" --rows 32 --iterations 2 --repetitions 1)
endif()
