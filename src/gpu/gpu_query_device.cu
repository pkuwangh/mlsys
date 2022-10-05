#include <cstdio>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "common/kmg_parser.h"
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


void printDevProp(const cudaDeviceProp& device_prop, bool id_only=false) {
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
    printf("%#x - %s\n", device_prop.pciBusID, device_prop.name);
    if (id_only) {
        return;
    }
    printf("# of SMs:                     %d\n", device_prop.multiProcessorCount);
    printf("Clock rate:                   %sHz\n",
            mm_utils::get_count_str(device_prop.clockRate, 1000, 1).c_str());
    printf("# of Async Engines:           %d\n", device_prop.asyncEngineCount);
    printf("Major/Minor revisions:        %d / %d\n",
            device_prop.major, device_prop.minor);
    printf("Total global memory:          %sB\n",
            mm_utils::get_byte_str(device_prop.totalGlobalMem).c_str());
    printf("Memory bus width:             %d\n", device_prop.memoryBusWidth);
    printf("Memory clock rate:            %sHz\n",
            mm_utils::get_count_str(device_prop.memoryClockRate, 1000, 1).c_str());
    printf("Total constant memory:        %s\n",
            mm_utils::get_byte_str(device_prop.totalConstMem).c_str());
    printf("Max pitch allowed for memcpy: %sB\n",
            mm_utils::get_byte_str(device_prop.memPitch).c_str());
    printf("L2 cache size:                %sB\n",
            mm_utils::get_byte_str(device_prop.l2CacheSize).c_str());
    printf("Supported mem operations:     %s\n", supported_mem_op.c_str());
    printf("Un-Supported mem operations:  %s\n", unsupported_mem_op.c_str());
    printf("Warp size:                    %d\n", device_prop.warpSize);
    printf("Max blocks per SM:            %d\n", device_prop.maxBlocksPerMultiProcessor);
    printf("32-bit regs per block/SM:     %s / %s\n",
            mm_utils::get_byte_str(device_prop.regsPerBlock).c_str(),
            mm_utils::get_byte_str(device_prop.regsPerMultiprocessor).c_str());
    printf("Shared memory per block/SM:   %sB / %sB\n",
            mm_utils::get_byte_str(device_prop.sharedMemPerBlock).c_str(),
            mm_utils::get_byte_str(device_prop.sharedMemPerMultiprocessor).c_str());
    printf("Max threads per block:        %s\n",
            mm_utils::get_count_str(device_prop.maxThreadsPerBlock, 1024).c_str());
    printf("Max block dimension:          %s\n",
            mm_utils::get_dims_str(device_prop.maxThreadsDim, 3, 1024).c_str());
    printf("Max grid dimension:           %s\n",
            mm_utils::get_dims_str(device_prop.maxGridSize, 3, 1024).c_str());
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
        } else {
            curr_device_name = device_prop.name;
            curr_device_count = 1;
        }
        printf("-------- CUDA Device #%-2d:  ", i);
        printDevProp(device_prop, (curr_device_count > 1));
        fflush(stdout);
    }
    printf("\n>>>>>>>> Total Device Count: %d <<<<<<<<\n", device_count);
}


int main() {
    mm_utils::start_timer("qery device info");
    queryAllDevice();
    mm_utils::end_timer("qery device info", std::cout);
    return 0;
}
