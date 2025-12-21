#include <functional>
#include <string>
#include <vector>

#include "matmul_kernel_a01_basic.cuh"
#include "matmul_kernel_a02_shmem.cuh"
#include "matmul_kernel_a03_thread_tile_1d.cuh"
#include "matmul_utils.cuh"

// dump cuda-related device information
void dumpDeviceInfo() {
    int device_count;
    checkCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    std::printf("Device count: %d\n", device_count);
    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    std::printf("Device name: %s\n", prop.name);
    std::printf("Device compute capability: %d.%d\n", prop.major, prop.minor);
    std::printf("Device total memory: %zu bytes\n", prop.totalGlobalMem);
    // dump factors that limit the number of blocks
    std::printf("max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    std::printf("max blocks per SM: %d\n", prop.maxBlocksPerMultiProcessor);
    std::printf("# of registers per SM: %d\n", prop.regsPerMultiprocessor);
    std::printf("shared memory per SM: %zu bytes\n", prop.sharedMemPerMultiprocessor);
}

class MatmulRunner {
  public:
    std::string name;
    std::function<void(MatmulBuffers &)> func;
    MatmulRunner(std::string name, std::function<void(MatmulBuffers &)> func) : name(name), func(func) {}

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
void verifyCorrectness(const std::vector<float> &ref, MatmulBuffers &buffers, std::string kernel_name) {
    std::vector<float> result = buffers.copyResultVector();
    int num_errors = 0;
    for (int i = 0; i < result.size(); i++) {
        float error = std::abs(result[i] - ref[i]);
        if (error > 1.0) {
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
        verifyCorrectness(ref, buffers, runner.name);
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
    dumpDeviceInfo();

    // sanity functional correctness check on a01-basic kernel
    sanityTests({MatmulRunner("a01-basic", runMatmulA01Basic)});

    std::vector<MatmulRunner> all_runners = {
        MatmulRunner("a01-basic", runMatmulA01Basic),
        MatmulRunner("a02-shmem", runMatmulA02Shmem),
        MatmulRunner("a03-thread-tile-1d", runMatmulA03ThreadTile1D),
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
