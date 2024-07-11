#include <boost/program_options.hpp>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <set>
#include <string>

// CUDA error handling
#define CU_ASSERT_RESULT(x)                                                    \
    do {                                                                       \
        CUresult cuResult = (x);                                               \
        if ((cuResult) != CUDA_SUCCESS) {                                      \
            const char *errDescStr, *errNameStr;                               \
            cuGetErrorString(cuResult, &errDescStr);                           \
            cuGetErrorName(cuResult, &errNameStr);                             \
            fprintf(stderr, "[%s] %s in expression %s in %s() : %s:%d\n",      \
                    errNameStr, errDescStr, #x, __PRETTY_FUNCTION__, __FILE__, \
                    __LINE__);                                                 \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

namespace po = boost::program_options;

// Kernel function to add the elements of two arrays
__global__ void add(size_t n, float *x, float *y) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = index; i < n; i += stride) {
        y[i] = x[i] + y[i];
    }
}

int benchmark(const std::string &allocator) {
    size_t N = (size_t)(2) << 30; // 2G x 4B x 2
    size_t allocSize = N * sizeof(float);
    float *x, *y;
    CUmemGenericAllocationHandle handleX, handleY;

    if (allocator == "cudaMallocManaged") {
        cudaMallocManaged(&x, N * sizeof(float));
        cudaMallocManaged(&y, N * sizeof(float));
    } else if (allocator == "cudaMallocHost") {
        cudaMallocHost(&x, N * sizeof(float));
        cudaMallocHost(&y, N * sizeof(float));
    } else if (allocator == "cuMemCreate-Device" ||
               allocator == "cuMemCreate-Host") {
        // get device handle
        int cudaDev;
        CUdevice currentDev;
        cudaGetDevice(&cudaDev);
        cuDeviceGet(&currentDev, cudaDev);
        std::cout << "get cuda device " << cudaDev << "/" << currentDev
                  << std::endl;
        // get cpu NUMA id and set location type
        int cpuNumaNodeId = -1;
        CUmemLocationType type = CU_MEM_LOCATION_TYPE_DEVICE;
        cuDeviceGetAttribute(&cpuNumaNodeId, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID,
                             currentDev);
        bool hostMem =
            (cpuNumaNodeId != -1) && (allocator == "cuMemCreate-Host");
        type = hostMem ? CU_MEM_LOCATION_TYPE_HOST_NUMA : type;
        std::cout << "hostMem-" << hostMem << ", host numa ID=" << cpuNumaNodeId
                  << std::endl;
        // memory allocation property
        CUmemAllocationProp memprop = {};
        memprop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        memprop.location.type = type;
        memprop.location.id = hostMem ? cpuNumaNodeId : currentDev;
        // size & granularity
        size_t allocSize = N * sizeof(float);
        size_t granu = 0;
        cuMemGetAllocationGranularity(&granu, &memprop,
                                      CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        size_t origSize = allocSize;
        if (allocSize % granu > 0) {
            allocSize = granu * (allocSize / granu + 1);
        }
        std::cout << "requested size=" << origSize
                  << ", padded alloc size=" << allocSize
                  << ", granularity=" << granu << std::endl;
        // physical memory allocation
        CU_ASSERT_RESULT(cuMemCreate(&handleX, allocSize, &memprop, 0));
        CU_ASSERT_RESULT(cuMemCreate(&handleY, allocSize, &memprop, 0));
        // reserve an address space and map it to a pointer
        CU_ASSERT_RESULT(
            cuMemAddressReserve((CUdeviceptr *)(&x), allocSize, 0, 0, 0));
        CU_ASSERT_RESULT(
            cuMemAddressReserve((CUdeviceptr *)(&y), allocSize, 0, 0, 0));
        CU_ASSERT_RESULT(cuMemMap((CUdeviceptr)x, allocSize, 0, handleX, 0));
        CU_ASSERT_RESULT(cuMemMap((CUdeviceptr)y, allocSize, 0, handleY, 0));
        // explicitly protect mapped VA ranges
        CUmemAccessDesc accessDesc[2] = {{}};
        accessDesc[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        accessDesc[0].location.id = currentDev;
        accessDesc[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        accessDesc[1].location.type = type;
        accessDesc[1].location.id = hostMem ? cpuNumaNodeId : currentDev;
        accessDesc[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CU_ASSERT_RESULT(cuMemSetAccess((CUdeviceptr)x, allocSize, accessDesc,
                                        hostMem ? 2 : 1));
        CU_ASSERT_RESULT(cuMemSetAccess((CUdeviceptr)y, allocSize, accessDesc,
                                        hostMem ? 2 : 1));
    } else if (allocator == "malloc") {
        x = (float *)(malloc(N * sizeof(float)));
        y = (float *)(malloc(N * sizeof(float)));
    } else {
        std::cout << "Unknown allocator " << allocator << std::endl;
        return -1;
    }

    float *x_cpu, *y_cpu;
    if (allocator == "cuMemCreate-Device") {
        x_cpu = (float *)(malloc(N * sizeof(float)));
        y_cpu = (float *)(malloc(N * sizeof(float)));
        for (size_t i = 0; i < N; ++i) {
            x_cpu[i] = 1.0f;
            y_cpu[i] = 2.0f;
        }
        cudaMemcpy(x, x_cpu, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(y, y_cpu, N * sizeof(float), cudaMemcpyHostToDevice);
    } else {
        for (size_t i = 0; i < N; ++i) {
            x[i] = 1.0f;
            y[i] = 2.0f;
        }
    }

    cudaEvent_t ckpt1, ckpt2, ckpt3;
    cudaEventCreate(&ckpt1);
    cudaEventCreate(&ckpt2);
    cudaEventCreate(&ckpt3);

    std::cout << "Initialized " << N << " elements using " << allocator
              << std::endl;

    size_t blockSize = 1024;
    size_t numBlocks = (N + blockSize - 1) / blockSize;

    cudaEventRecord(ckpt1);

    add<<<numBlocks, blockSize>>>(N, x, y);
    cudaEventRecord(ckpt2);

    for (int k = 0; k < 10; ++k) {
        add<<<numBlocks, blockSize>>>(N, x, y);
    }
    cudaEventRecord(ckpt3);

    cudaEventSynchronize(ckpt3);
    // cudaDeviceSynchronize();

    float elapsed = 0;
    cudaEventElapsedTime(&elapsed, ckpt1, ckpt2);
    std::cout << "Elapsed time " << int(elapsed) << " mili-seconds - initial"
              << std::endl;
    cudaEventElapsedTime(&elapsed, ckpt2, ckpt3);
    std::cout << "Elapsed time " << int(elapsed / 10)
              << " mili-seconds - warmedup" << std::endl;

    float maxError = 0.0f;
    if (allocator == "cuMemCreate-Device") {
        cudaMemcpy(y_cpu, y, N * sizeof(float), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < N; i++) {
            maxError = fmax(maxError, fabs(y_cpu[i] - 13.0f));
        }
    } else {
        for (size_t i = 0; i < N; i++) {
            maxError = fmax(maxError, fabs(y[i] - 13.0f));
        }
    }

    cudaEventDestroy(ckpt1);
    cudaEventDestroy(ckpt2);
    cudaEventDestroy(ckpt3);

    if (allocator == "cuMemCreate") {
        cuMemAddressFree((CUdeviceptr)x, allocSize);
        cuMemAddressFree((CUdeviceptr)y, allocSize);
        cuMemRelease(handleX);
        cuMemRelease(handleY);
    } else if (allocator == "malloc") {
        free(x);
        free(y);
    } else {
        cudaFree(x);
        cudaFree(y);
    }
    if (allocator == "cuMemCreate-Device") {
        free(x_cpu);
        free(y_cpu);
    }
    return int(maxError);
}

int main(int argc, char **argv) {
    std::set<std::string> avail_allocators(
        {"cudaMallocHost", "cudaMallocManaged", "cuMemCreate-Device",
         "cuMemCreate-Host", "malloc"});
    std::string allocator;
    po::options_description all_opts("uvm_vec_add CLI");
    all_opts.add_options()("help,h", "Help message");
    all_opts.add_options()("list,l", "list available allocator options");
    all_opts.add_options()(
        "allocator,a", po::value<std::string>(&allocator)->default_value("all"),
        "Which memory allocation API to use");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, all_opts), vm);
        po::notify(vm);
    } catch (...) {
        std::cout << "Error: Invalid Arguments " << std::endl;
        std::cout << all_opts << std::endl;
        return 1;
    }
    if (vm.count("help")) {
        std::cout << all_opts << std::endl;
        return 0;
    }
    if (vm.count("list")) {
        for (const std::string &x : avail_allocators) {
            std::cout << x << std::endl;
        }
        return 0;
    }

    int maxError = 0;
    if (allocator == "all") {
        for (const std::string &x : avail_allocators) {
            std::cout << "\nBenchmarking " << x << std::endl;
            maxError = benchmark(x);
            std::cout << "Max error is " << maxError << std::endl;
        }
    } else {
        maxError = benchmark(allocator);
        std::cout << "Max error is " << maxError << std::endl;
    }

    return 0;
}
