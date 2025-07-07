# check if the cpu supports avx512
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    execute_process(COMMAND lscpu OUTPUT_VARIABLE CPU_INFO)
    string(REGEX MATCH "avx512f" AVX512_SUPPORT "${CPU_INFO}")
    if(AVX512_SUPPORT)
        set(AVX512_SUPPORT ON)
    endif()

    string(REGEX MATCH "avx2" AVX2_SUPPORT "${CPU_INFO}")
    if(AVX2_SUPPORT)
        set(AVX2_SUPPORT ON)
    endif()

    string(REGEX MATCH "avx" AVX_SUPPORT "${CPU_INFO}")
    if(AVX_SUPPORT)
        set(AVX_SUPPORT ON)
    endif()

    string(REGEX MATCH "sse4_2" SSE4_2_SUPPORT "${CPU_INFO}")
    if(SSE4_2_SUPPORT)
        set(SSE4_2_SUPPORT ON)
    endif()

    string(REGEX MATCH "sse4_1" SSE4_1_SUPPORT "${CPU_INFO}")
    if(SSE4_1_SUPPORT)
        set(SSE4_1_SUPPORT ON)
    endif()
endif()


if(AVX512_SUPPORT)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx512f")
endif()

if(AVX2_SUPPORT)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx2")
endif()

if(AVX_SUPPORT)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx")
endif()

if(SSE4_2_SUPPORT)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msse4.2")
endif()

if(SSE4_1_SUPPORT)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msse4.1")
endif()