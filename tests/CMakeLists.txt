cmake_policy(SET CMP0135 NEW)

file(GLOB TEST_FILES "*.cpp")

include(FetchContent)
FetchContent_Declare(googletest
  URL "https://github.com/google/googletest/archive/release-1.12.1.tar.gz")
set(BUILD_GMOCK CACHE BOOL OFF)
set(INSTALL_GTEST CACHE BOOL OFF)
FetchContent_MakeAvailable(googletest)


add_executable(${PROJECT_NAME}_test ${TEST_FILES})
target_link_libraries(${PROJECT_NAME}_test PRIVATE gtest gtest_main)
target_include_directories(${PROJECT_NAME}_test PRIVATE ${PROJECT_SOURCE_DIR})

include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME}_test)

