# Run with -DVERSION_FILE=<rabitqlibConfigVersion.cmake> -DPROJECT_VERSION=<X.Y.Z> -P.
# Before 1.0 a minor release may break the API, so find_package(rabitqlib X.Y)
# accepts only the same major and minor version. A policy change at 1.0 must
# update this check deliberately.
cmake_minimum_required(VERSION 3.20)
if(NOT VERSION_FILE OR NOT PROJECT_VERSION)
    message(FATAL_ERROR "VERSION_FILE and PROJECT_VERSION are required")
endif()

string(REPLACE "." ";" version_parts "${PROJECT_VERSION}")
list(GET version_parts 0 major)
list(GET version_parts 1 minor)

function(check_request requested expected)
    string(REPLACE "." ";" parts "${requested}")
    list(LENGTH parts PACKAGE_FIND_VERSION_COUNT)
    list(APPEND parts 0 0)
    list(GET parts 0 PACKAGE_FIND_VERSION_MAJOR)
    list(GET parts 1 PACKAGE_FIND_VERSION_MINOR)
    list(GET parts 2 PACKAGE_FIND_VERSION_PATCH)
    set(PACKAGE_FIND_VERSION "${requested}")
    unset(PACKAGE_VERSION_COMPATIBLE)
    include("${VERSION_FILE}")
    if(expected AND NOT PACKAGE_VERSION_COMPATIBLE)
        message(FATAL_ERROR "rabitqlib ${PROJECT_VERSION} rejected a request for ${requested}")
    elseif(NOT expected AND PACKAGE_VERSION_COMPATIBLE)
        message(FATAL_ERROR "rabitqlib ${PROJECT_VERSION} accepted a request for ${requested}")
    endif()
endfunction()

check_request("${major}.${minor}" TRUE)
check_request("${major}.${minor}.0" TRUE)
# Requests for an older release are what the compatibility mode decides; every
# mode already rejects newer ones.
if(minor GREATER 0)
    math(EXPR previous_minor "${minor} - 1")
    check_request("${major}.${previous_minor}" FALSE)
endif()
if(major GREATER 0)
    math(EXPR previous_major "${major} - 1")
    check_request("${previous_major}.${minor}" FALSE)
endif()
message(STATUS "Package version checks passed for ${PROJECT_VERSION}")
