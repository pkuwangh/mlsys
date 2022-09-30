#include <cstdio>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "common/timing.h"


void addForDisplayString(
        std::string& output,
        const std::string& input,
        int hanging=30)
{
    std::string hanging_str(hanging, ' ');
    size_t current_length = output.size();
    size_t current_start = output.find_last_of('\n');
    if (current_start != std::string::npos) {
        current_length = output.size() - (current_start + hanging) + 1;
    }
    if (output.size() > 0) {
        if (current_length + input.size() > (100 - hanging)) {
            output.append(",\n");
            output.append(hanging_str);
        } else {
            output.append(", ");
        }
    }
    output.append(input);
}


void printDevProp(const cudaDeviceProp& device_prop) {
    std::string supported_mem_op;
    std::string unsupported_mem_op;
    std::vector<std::pair<int, std::string>> features = {
        std::make_pair(
                device_prop.canMapHostMemory, "canMapHostMemory"),
        std::make_pair(
                device_prop.canUseHostPointerForRegisteredMem,
                "canUseHostPointerForRegisteredMem"),
        std::make_pair(
                device_prop.deviceOverlap, "deviceOverlap"),
        std::make_pair(
                device_prop.managedMemory, "managedMemory"),
        std::make_pair(
                device_prop.concurrentManagedAccess,
                "concurrentManagedAccess"),
        std::make_pair(
                device_prop.directManagedMemAccessFromHost,
                "directManagedMemAccessFromHost"),
        std::make_pair(
                device_prop.pageableMemoryAccess, "pageableMemoryAccess"),
        std::make_pair(
                device_prop.pageableMemoryAccessUsesHostPageTables,
                "pageableMemoryAccessUsesHostPageTables"),
        std::make_pair(
                device_prop.unifiedAddressing, "unifiedAddressing"),
    };
    for (const auto& item : features) {
        std::string& mem_op = item.first ? supported_mem_op : unsupported_mem_op;
        addForDisplayString(mem_op, item.second);
    }
    printf("Name:                         %s\n", device_prop.name);
    printf("# of SMs:                     %d\n", device_prop.multiProcessorCount);
    printf("Clock rate:                   %d\n", device_prop.clockRate);
    printf("# of Async Engines:           %d\n", device_prop.asyncEngineCount);
    printf("Major/Minor revisions:        %d / %d\n",
            device_prop.major, device_prop.minor);
    printf("Total global memory:          %lu\n", device_prop.totalGlobalMem);
    printf("Memory bus width:             %d\n", device_prop.memoryBusWidth);
    printf("Memory clock rate:            %d\n", device_prop.memoryClockRate);
    printf("Total constant memory:        %lu\n", device_prop.totalConstMem);
    printf("Max pitch allowed for memcpy: %lu\n", device_prop.memPitch);
    printf("L2 cache size:                %d\n", device_prop.l2CacheSize);
    printf("Supported mem operations:     %s\n", supported_mem_op.c_str());
    printf("Un-Supported mem operations:  %s\n", unsupported_mem_op.c_str());
    printf("Warp size:                    %d\n", device_prop.warpSize);
    printf("Max blocks per SM:            %d\n", device_prop.maxBlocksPerMultiProcessor);
    printf("32-bit regs per block/SM:     %d / %d\n",
            device_prop.regsPerBlock, device_prop.regsPerMultiprocessor);
    printf("Shared memory per block/SM:   %lu / %lu\n",
            device_prop.sharedMemPerBlock, device_prop.sharedMemPerMultiprocessor);
    printf("Max threads per block:        %d\n", device_prop.maxThreadsPerBlock);
    printf("Max block dimension:          ");
    fflush(stdout);
    for (int i = 0; i < 3; ++i) {
        printf("%d  ", device_prop.maxThreadsDim[i]);
    }
    printf("\n");
    printf("Max grid dimension:           ");
    for (int i = 0; i < 3; ++i) {
        printf("%d  ", device_prop.maxGridSize[i]);
    }
    printf("\n");
    fflush(stdout);
}


void queryAllDevice() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    int curr_device_count = 0;
    std::string curr_device_name = "??";
    cudaDeviceProp device_prop;
    for (int i = 0; i < device_count; ++i) {
        cudaGetDeviceProperties(&device_prop, i);
        if (curr_device_name.compare(device_prop.name) == 0) {
            curr_device_count += 1;
            continue;
        } else {
            if (curr_device_count) {
                printf("<<<<<<<< There are %d such Devices\n", curr_device_count);
            }
            curr_device_name = device_prop.name;
            curr_device_count = 1;
        }
        printf("\n-------- CUDA Device #%d --------\n", i);
        printDevProp(device_prop);
    }
    printf("\n>>>>>>>> Total Device Count: %d\n", device_count);
}


int main() {
    mm_utils::start_timer("qery device info");
    queryAllDevice();
    mm_utils::end_timer("qery device info", std::cout);
    return 0;
}
