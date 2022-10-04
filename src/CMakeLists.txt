cmake_minimum_required(VERSION 3.8)

find_program(NVCC nvcc)
if(NVCC)
    project(MLSYS LANGUAGES CUDA CXX)
    set(CMAKE_CUDA_STANDARD 14)
else()
    message(WARNING "nvcc not found")
    project(MLSYS LANGUAGES CXX)
endif()

find_library(BOOST_PROGRAM_OPTIONS boost_program_options)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-Wall")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(common)

if(NVCC)
    add_subdirectory(gpu)
endif()

if(BOOST_PROGRAM_OPTIONS)
    add_subdirectory(cpu_micro)
endif()
