cmake_minimum_required(VERSION 3.15)

project(rabitqlib_consumer LANGUAGES CXX)

find_package(rabitqlib CONFIG REQUIRED)

add_executable(rabitqlib_consumer main.cpp)
target_link_libraries(rabitqlib_consumer PRIVATE rabitqlib::rabitqlib)

enable_testing()
add_test(NAME rabitqlib_consumer COMMAND rabitqlib_consumer)
