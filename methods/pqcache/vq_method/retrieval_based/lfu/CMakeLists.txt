cmake_minimum_required(VERSION 3.18)
project(lfucache LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
aux_source_directory(src CACHE_SRC_LIST)

find_package(pybind11 2.6.0 CONFIG REQUIRED)

pybind11_add_module(lfucache ${CACHE_SRC_LIST})
target_include_directories(lfucache PUBLIC include)
