#include <functional>
#include <string>
#include <vector>

#include "matmul_kernel_basic.cuh"
#include "matmul_kernel_tiled_shmem.cuh"
#include "matmul_utils.cuh"

void sanityTests() {
    std::printf("Sanity functional correctness check\n");
    MatmulBuffers buffers = MatmulBuffers(2, 4, 2);
    printMatrix(buffers.hA.data(), buffers.M, buffers.K, "A");
    printMatrix(buffers.hB.data(), buffers.K, buffers.N, "B");

    std::printf("basic kernel\n");
    buffers.reset();
    runMatmulBasic(buffers);
    buffers.printResult();

    // std::printf("tiled shmem kernel\n");
    // buffers.reset();
    // runMatmulTiledShmem(buffers);
    // buffers.printResult();

    std::printf("--------------------------------\n");
}

void verifyCorrectness(const std::vector<float> &ref, MatmulBuffers &buffers, std::string kernel_name) {
    std::vector<float> result = buffers.copyResultVector();
    int num_errors = 0;
    for (int i = 0; i < result.size(); i++) {
        float error = std::abs(result[i] - ref[i]);
        if (error > 1.0) {
            std::printf("Error at index %d: %f != %f\n", i, result[i], ref[i]);
            ++ num_errors;
            if (num_errors >= 4) {
                break;
            }
        }
    }
    std::printf("%s correctness check %s\n", kernel_name.c_str(), num_errors == 0 ? "passed" : "failed");
}

void functionalTests() {
    std::printf("Functional correctness check against basic kernel\n");
    MatmulBuffers buffers = MatmulBuffers(32, 64, 64);
    runMatmulBasic(buffers);
    std::vector<float> ref = buffers.copyResultVector();
    verifyCorrectness(ref, buffers, "basic");

    // verify tiled shmem kernel
    buffers.reset();
    runMatmulTiledShmem(buffers);
    verifyCorrectness(ref, buffers, "tiled shmem");
}

void perfTests(MatmulBuffers &buffers, std::function<void(MatmulBuffers &)> run_kernel, std::string kernel_name) {
    int num_warmup_iters = 1;
    int num_total_iters = 100;
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
    // sanity functional correctness check on basic kernel
    sanityTests();

    // verify correctness against basic kernel
    functionalTests();

    // perf test
    MatmulBuffers buffers = MatmulBuffers(4096, 8192, 8192);
    perfTests(buffers, runMatmulBasic, "matmul-basic");
    perfTests(buffers, runMatmulTiledShmem, "matmul-tiled-shmem");

    return 0;
}
