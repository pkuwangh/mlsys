cmake_minimum_required(VERSION 3.8)

project(MLSYS LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_VERBOSE_MAKEFILE ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(common)
add_subdirectory(gpu)
