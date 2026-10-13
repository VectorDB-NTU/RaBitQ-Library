cmake_minimum_required(VERSION 3.20)
project(rabitqlib_core_example LANGUAGES CXX)

find_package(rabitqlib CONFIG REQUIRED COMPONENTS core)
add_executable(core_example main.cpp)
target_link_libraries(core_example PRIVATE rabitqlib::core)

enable_testing()
add_test(NAME core_example COMMAND core_example)
