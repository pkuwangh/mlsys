cmake_minimum_required(VERSION 3.20)

project(cuda_demos
    LANGUAGES CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES 90-real;90-virtual;100-real;100-virtual)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

# set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_STATIC_LIBS OFF)
find_package(Boost COMPONENTS program_options REQUIRED)
execute_process(
    COMMAND git describe --always --tags
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    OUTPUT_VARIABLE GIT_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGIT_VERSION=\\\"\"${GIT_VERSION}\"\\\"")

if (MULTINODE)
    find_package(MPI REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMULTINODE")
    string(REPLACE "-O3" "-O0" CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE}")
endif()

if(WIN32)
    set(NVML_LIB_NAME "nvml")
else()
    set(NVML_LIB_NAME "nvidia-ml")
endif()

set(TESTS
    "uvm_pingpong"
    "uvm_vec_add"
    "mpi_egm"
)

foreach(TEST ${TESTS})
    add_executable(${TEST} ${TEST}.cu)

    if (MULTINODE)
        target_include_directories(${TEST} PRIVATE ${MPI_INCLUDE_PATH} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} .)
        target_link_libraries(${TEST} MPI::MPI_CXX Boost::program_options ${NVML_LIB_NAME} cuda)
    else()
        target_include_directories(${TEST} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} .)
        target_link_libraries(${TEST} Boost::program_options ${NVML_LIB_NAME} cuda)
    endif()

    install(TARGETS ${TEST} DESTINATION "bin")
endforeach()
