cmake_minimum_required(VERSION 3.20)

project(cuda_demos
    LANGUAGES CUDA CXX)

if(NOT DEFINED MLSYS_CUDA_ARCHITECTURES)
    set(MLSYS_CUDA_ARCHITECTURES "90a-real;90a-virtual")
endif()

set(CMAKE_CUDA_ARCHITECTURES "${MLSYS_CUDA_ARCHITECTURES}")

message(STATUS "CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")

# Check if we're building for SM_90+ (Hopper) - enables WGMMA/TMA kernels
foreach(ARCH ${CMAKE_CUDA_ARCHITECTURES})
    string(REGEX MATCH "^([0-9]+)" ARCH_NUM "${ARCH}")
    if(ARCH_NUM AND ARCH_NUM EQUAL 90)
        set(HAS_GEN4_TENSOR_CORE TRUE)
        break()
    endif()
endforeach()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

# set git version
execute_process(
    COMMAND git describe --always --tags
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    OUTPUT_VARIABLE GIT_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGIT_VERSION=\\\"\"${GIT_VERSION}\"\\\"")

if(WIN32)
    set(NVML_LIB_NAME "nvml")
else()
    set(NVML_LIB_NAME "nvidia-ml")
endif()

set(TESTS
    "matmul_main"
)

foreach(TEST ${TESTS})
    add_executable(${TEST} ${TEST}.cu)

    target_include_directories(${TEST} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} .)
    target_compile_options(${TEST} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
    target_link_libraries(${TEST} ${NVML_LIB_NAME} cuda)

    # Enable SM_90+ kernels (WGMMA/TMA) if building for Hopper+
    if(HAS_GEN4_TENSOR_CORE)
        target_compile_definitions(${TEST} PRIVATE HAS_GEN4_TENSOR_CORE)
    endif()

    install(TARGETS ${TEST} DESTINATION "bin")
endforeach()
