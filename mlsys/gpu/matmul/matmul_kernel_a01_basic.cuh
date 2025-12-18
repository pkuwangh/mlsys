#pragma once

#include "matmul_utils.cuh"
#include <cuda_runtime.h>

#define A01_BLOCK_DIM 32

// each thread computes one element of the result matrix

__global__ void matmul_a01_basic(float *A, float *B, float *C, int M, int K, int N) {
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    if (gy >= M || gx >= N) {
        return;
    }

    float acc = 0.0f;
    // for each element in MxN matrix, K iterations
    for (int e = 0; e < K; ++e) {
        // 2 global mem accesses and 1 fma per iteration
        acc += A[gy * K + e] * B[e * N + gx];
    }
    // total: M*N*K*2 global mem access + M*N*K*1 fma
    C[gy * N + gx] = acc;
}

void runMatmulA01Basic(MatmulBuffers &buffers) {
    dim3 blockDim = dim3(A01_BLOCK_DIM, A01_BLOCK_DIM);
    dim3 gridDim = makeGrid2D(buffers.M, buffers.N, blockDim);

    matmul_a01_basic<<<gridDim, blockDim>>>(buffers.dA, buffers.dB, buffers.dC, buffers.M, buffers.K, buffers.N);
    checkCuda(cudaGetLastError(), "launch matmul_basic");
    buffers.num_iters += 1;
}
