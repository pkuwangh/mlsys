#include <functional>
#include <string>
#include <vector>

#include "matmul_kernel_a01_basic.cuh"
#include "matmul_kernel_a02_shmem.cuh"
#include "matmul_kernel_a03_thread_tile_1d.cuh"
#include "matmul_kernel_a04_thread_tile_2d.cuh"
#include "matmul_kernel_a05_thread_tile_2d_cache.cuh"
#include "matmul_kernel_a06_thread_tile_float4.cuh"
#include "matmul_kernel_a07_double_buffering.cuh"
#include "matmul_kernel_a08_tuning.cuh"
#include "matmul_kernel_b01_warp_tile.cuh"
#include "matmul_kernel_c01_warp_tile_bf16.cuh"
#include "matmul_utils.cuh"

// dump cuda-related device information
int dumpAndGetDeviceInfo() {
    int device_count;
    checkCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    std::printf("Device count: %d\n", device_count);
    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    std::printf("Device name: %s\n", prop.name);
    std::printf("Device compute capability: %d.%d\n", prop.major, prop.minor);
    std::printf("Device SM count: %d\n", prop.multiProcessorCount);
    std::printf("Device total memory: %zu bytes\n", prop.totalGlobalMem);
    // per block limits
    std::printf("max threads per block: %d\n", prop.maxThreadsPerBlock);
    std::printf("shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
    // dump factors that limit the number of blocks
    std::printf("max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    std::printf("max blocks per SM: %d\n", prop.maxBlocksPerMultiProcessor);
    std::printf("# of registers per SM: %d\n", prop.regsPerMultiprocessor);
    std::printf("shared memory per SM: %zu bytes\n", prop.sharedMemPerMultiprocessor);
    return prop.major * 10 + prop.minor;
}

class MatmulRunner {
  public:
    std::string name;
    std::function<void(MatmulBuffers &)> func;
    bool use_bf16;
    MatmulRunner(std::string name, std::function<void(MatmulBuffers &)> func, bool use_bf16 = false)
        : name(name), func(func), use_bf16(use_bf16) {}

    void run(MatmulBuffers &buffers) { func(buffers); }
};

// sanity tests on basic kernel with small matrix sizes
void sanityTests(std::vector<MatmulRunner> runners) {
    std::printf("Sanity functional correctness check\n");
    MatmulBuffers buffers = MatmulBuffers(4, 8, 8);
    printMatrix(buffers.hA.data(), buffers.M, buffers.K, "A");
    printMatrix(buffers.hB.data(), buffers.K, buffers.N, "B");

    for (auto runner : runners) {
        std::printf("%s kernel\n", runner.name.c_str());
        buffers.reset();
        runner.run(buffers);
        buffers.printResult();
    }

    std::printf("--------------------------------\n");
}

// verify correctness against reference result
void verifyCorrectness(const std::vector<float> &ref, MatmulBuffers &buffers, std::string kernel_name, bool use_bf16) {
    std::vector<float> result = buffers.copyResultVector(use_bf16);
    int num_errors = 0;
    for (int i = 0; i < result.size(); i++) {
        float error = std::abs(result[i] - ref[i]);
        float relative_error = error / std::abs(ref[i]);
        if (relative_error > 0.01f) {
            std::printf("Error at index %d: %f != %f\n", i, result[i], ref[i]);
            ++num_errors;
            if (num_errors >= 4) {
                break;
            }
        }
    }
    std::printf("%s correctness check %s\n", kernel_name.c_str(), num_errors == 0 ? "passed" : "failed");
}

// functional tests against a01-basic kernel
void functionalTests(std::vector<MatmulRunner> runners) {
    std::printf("Functional correctness check against a01-basic kernel\n");
    MatmulBuffers buffers = MatmulBuffers(128, 256, 256);
    runMatmulA01Basic(buffers);
    std::vector<float> ref = buffers.copyResultVector();

    for (auto runner : runners) {
        buffers.reset();
        runner.run(buffers);
        verifyCorrectness(ref, buffers, runner.name, runner.use_bf16);
    }
    std::printf("--------------------------------\n");
}

// performance tests
void perfTest(MatmulBuffers &buffers, std::function<void(MatmulBuffers &)> run_kernel, std::string kernel_name) {
    int num_warmup_iters = 1;
    int num_total_iters = 10;
    DeviceTimer timer;
    for (int i = 0; i < num_warmup_iters; i++) {
        run_kernel(buffers);
    }
    buffers.reset();
    timer.start();
    for (int i = 0; i < num_total_iters; i++) {
        run_kernel(buffers);
    }
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    buffers.printTFLOPS(timer.stop(), kernel_name);
}

int main() {
    int device_capability = dumpAndGetDeviceInfo();

    // sanity functional correctness check on a01-basic kernel
    sanityTests({MatmulRunner("a01-basic", runMatmulA01Basic)});

    std::vector<MatmulRunner> all_runners = {
        MatmulRunner("a01-basic", runMatmulA01Basic),
        MatmulRunner("a02-shmem", runMatmulA02Shmem),
        MatmulRunner("a03-thread-tile-1d", runMatmulA03ThreadTile1D),
        MatmulRunner("a04-thread-tile-2d", runMatmulA04ThreadTile2D),
        MatmulRunner("a05-thread-tile-2d-cache", runMatmulA05ThreadTile2DCache),
        MatmulRunner("a06-thread-tile-float4", runMatmulA06ThreadTileFloat4),
        MatmulRunner("a07-double-buffering", runMatmulA07DoubleBuffering),
        MatmulRunner("a08-tuning", runMatmulA08Tuning),
        MatmulRunner("b01-warp-tile", runMatmulB01WarpTile),
        MatmulRunner("c01-warp-tile-bf16", runMatmulC01WarpTileBf16, true),
    };

    // verify correctness against a01-basic kernel
    functionalTests(all_runners);

    // perf test
    MatmulBuffers buffers = MatmulBuffers(4096, 8192, 8192);
    for (auto runner : all_runners) {
        perfTest(buffers, runner.func, runner.name);
    }

    return 0;
}
