#include <string>
#include <vector>

#include "matmul_kernel_basic.cuh"
#include "matmul_utils.cuh"

void functionalTest() {
    MatmulBuffers buffers = MatmulBuffers(2, 4, 3);
    printMatrix(buffers.hA.data(), buffers.M, buffers.K, "A");
    printMatrix(buffers.hB.data(), buffers.K, buffers.N, "B");
    runMatmulBasic(buffers);
    buffers.printResult();
}

int main() {
    // sanity functional correctness check on basic kernel
    functionalTest();
    std::printf("--------------------------------\n");

    // perf test
    DeviceTimer timer;
    MatmulBuffers buffers = MatmulBuffers(4096, 8192, 8192);
    int num_total_iters = 10;

    // perf test on basic kernel
    buffers.reset();
    timer.start();
    for (int i = 0; i < num_total_iters; i++) {
        runMatmulBasic(buffers);
    }
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    buffers.printTFLOPS(timer.stop(), "matmul-basic");

    return 0;
}
