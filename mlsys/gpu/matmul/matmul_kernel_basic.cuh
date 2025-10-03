#pragma once

#include <cuda_runtime.h>

#include "matmul_utils.cuh"

__global__ void matmul_basic(const float *A, const float *B, float *C, int M, int K, int N) {
    // each thread computes one element of the result matrix
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) {
        return;
    }

    float acc = 0.0f;
    for (int e = 0; e < K; ++e) {
        acc += A[row * K + e] * B[e * N + col];
    }
    C[row * N + col] = acc;
}

void runMatmulBasic(MatmulBuffers &buffers) {
    dim3 blockDim = dim3(16, 16);
    dim3 gridDim = makeGrid2D(buffers.M, buffers.N, blockDim);

    matmul_basic<<<gridDim, blockDim>>>(buffers.dA, buffers.dB, buffers.dC, buffers.M, buffers.K, buffers.N);
    checkCuda(cudaGetLastError(), "launch matmul_basic");
    buffers.num_iters += 1;
}
