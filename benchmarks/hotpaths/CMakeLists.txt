# Keep each benchmark's dependency closure at the boundary it measures.
function(rabitq_hotpath_target name component)
    add_executable(rabitq_${name}_hotpaths ${name}.cpp)
    target_link_libraries(rabitq_${name}_hotpaths PRIVATE rabitqlib::${component})
    rabitq_configure_target(rabitq_${name}_hotpaths)
    set_target_properties(rabitq_${name}_hotpaths PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)
endfunction()
rabitq_hotpath_target(core core)
add_test(NAME CoreHotpathEncoding COMMAND rabitq_core_hotpaths 64 32 1 2 1 encode)
add_test(NAME CoreHotpathL2 COMMAND rabitq_core_hotpaths 128 32 1 2 1 l2)
if(RABITQ_BUILD_FASTSCAN)
    rabitq_hotpath_target(fastscan fastscan)
    add_test(NAME FastScanHotpathStandard COMMAND rabitq_fastscan_hotpaths 128 2 1 standard)
    add_test(NAME FastScanHotpathHacc COMMAND rabitq_fastscan_hotpaths 128 2 1 hacc)
endif()
if(RABITQ_BUILD_INDEX)
    rabitq_hotpath_target(load index)
    add_test(NAME SidecarHotpathPrepare COMMAND rabitq_load_hotpaths prepare
        "${CMAKE_CURRENT_BINARY_DIR}/sidecar-smoke" 128 64 1)
    add_test(NAME SidecarHotpathLoad COMMAND rabitq_load_hotpaths load
        "${CMAKE_CURRENT_BINARY_DIR}/sidecar-smoke" 128 64 2)
    set_tests_properties(SidecarHotpathPrepare PROPERTIES FIXTURES_SETUP sidecar_hotpath)
    set_tests_properties(SidecarHotpathLoad PROPERTIES FIXTURES_REQUIRED sidecar_hotpath)

    rabitq_hotpath_target(ivf_add index)
    add_test(NAME IvfAddHotpathSingle COMMAND rabitq_ivf_add_hotpaths
        "${CMAKE_CURRENT_BINARY_DIR}/ivf-add-smoke-single" 64 8 256 16 1 single)
    add_test(NAME IvfAddHotpathRouted COMMAND rabitq_ivf_add_hotpaths
        "${CMAKE_CURRENT_BINARY_DIR}/ivf-add-smoke-routed" 64 8 256 16 1 routed)
    add_test(NAME IvfAddHotpathSmall COMMAND rabitq_ivf_add_hotpaths
        "${CMAKE_CURRENT_BINARY_DIR}/ivf-add-smoke-small" 64 8 8 1 1 single)
endif()
