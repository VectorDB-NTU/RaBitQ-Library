add_executable(stable_core_example stable_core/main.cpp)
target_link_libraries(stable_core_example PRIVATE rabitqlib::core)
rabitq_configure_target(stable_core_example)

add_executable(core_example core/main.cpp)
target_link_libraries(core_example PRIVATE rabitqlib::core)
rabitq_configure_target(core_example)

if(RABITQ_BUILD_FASTSCAN)
    add_executable(fastscan_example fastscan/main.cpp)
    target_link_libraries(fastscan_example PRIVATE rabitqlib::fastscan)
    rabitq_configure_target(fastscan_example)
endif()

if(NOT RABITQ_BUILD_INDEX)
    return()
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_executable(symqg_indexing symqg_indexing.cpp)
add_executable(symqg_querying symqg_querying.cpp)
add_executable(qgkmeans qgkmeans.cpp)
add_executable(rabitqkmeans rabitqkmeans.cpp)

add_executable(ivf_rabitq_indexing ivf_rabitq_indexing.cpp)
add_executable(ivf_rabitq_querying ivf_rabitq_querying.cpp)

add_executable(hnsw_rabitq_indexing hnsw_rabitq_indexing.cpp)
add_executable(hnsw_rabitq_querying hnsw_rabitq_querying.cpp)

foreach(RABITQ_SAMPLE_TARGET
    symqg_indexing
    symqg_querying
    qgkmeans
    rabitqkmeans
    ivf_rabitq_indexing
    ivf_rabitq_querying
    hnsw_rabitq_indexing
    hnsw_rabitq_querying
)
    target_link_libraries(${RABITQ_SAMPLE_TARGET} PRIVATE rabitqlib::index)
    rabitq_configure_target(${RABITQ_SAMPLE_TARGET})
endforeach()
