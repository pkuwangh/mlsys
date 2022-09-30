enable_language(CUDA)

add_executable(gpu_query_device gpu_query_device.cu)

target_link_libraries(gpu_query_device LINK_PUBLIC Timing)

install(TARGETS gpu_query_device DESTINATION "bin")
