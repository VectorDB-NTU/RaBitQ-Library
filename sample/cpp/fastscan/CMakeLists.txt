cmake_minimum_required(VERSION 3.20)
project(rabitqlib_fastscan_example LANGUAGES CXX)

find_package(rabitqlib CONFIG REQUIRED COMPONENTS fastscan)
add_executable(fastscan_example main.cpp)
target_link_libraries(fastscan_example PRIVATE rabitqlib::fastscan)

enable_testing()
add_test(NAME fastscan_example COMMAND fastscan_example)
