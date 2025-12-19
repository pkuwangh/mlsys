#pragma once

#include "matmul_utils.cuh"
#include <cuda_runtime.h>

#define A02_BLOCK_DIM 32

// each block computes one BM * BN tile of result C matrix
// in each iteration,
// - loads BM * BK tile from A and BK * BN tile from B into shared memory
// - do matmul of two tiles, giving a partial sum of result BM * BN tile
// then move to next tile along the BK dimension
// each thread corresponds to one element of the result BM * BN tile

template <const int TILE_SIZE>
__global__ void matmul_a02_shmem(float *A, float *B, float *C, int M, int K, int N) {
    const int BM = TILE_SIZE;
    const int BN = TILE_SIZE;
    const int BK = TILE_SIZE;
    // shared memory size: 2 * TILE_SIZE * TILE_SIZE * sizeof(float)
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x % TILE_SIZE;
    int ty = threadIdx.x / TILE_SIZE;

    // move to the first tile; and point to the first element of the tile
    A += by * BM * K;   // (by * BM) row, first column
    B += bx * BN;       // first row, (bx * BN) column
    C += by * BM * N + bx * BN;

    float acc = 0.0f;
    for (int b = 0; b < K; b += BK) {
        As[ty * BK + tx] = A[ty * K + tx];
        Bs[ty * BN + tx] = B[ty * N + tx];
        // sync to make sure all threads have loaded the tile
        __syncthreads();

        // move A & B
        A += BK;        // move right for BK columns
        B += BK * N;    // move down for BK rows

        // each thread computes one element of the result BM * BN tile
        for (int i = 0; i < BK; ++i) {
            acc += As[ty * BK + i] * Bs[i * BN + tx];
        }
        // sync before loading next tile
        __syncthreads();
    }
    // each block will read (K/BK)*(BM*BK+BK*BN) elements from global memory
    // that is K*(BM+BN) mem accesses per block
    // total mem accesses is (M/BM)*(N/BN)*K*(BM+BN) = M*N*K*(BM+BN)/BM/BN
    // compare with a01_basic, the ratio is (BM+BN)/(2*BM*BN)
    C[ty * N + tx] = acc;
}

void runMatmulA02Shmem(MatmulBuffers &buffers) {
    dim3 blockDim = dim3(A02_BLOCK_DIM * A02_BLOCK_DIM, 1);
    dim3 gridDim = dim3(buffers.N / A02_BLOCK_DIM, buffers.M / A02_BLOCK_DIM);

    matmul_a02_shmem<A02_BLOCK_DIM><<<gridDim, blockDim>>>(buffers.dA, buffers.dB, buffers.dC, buffers.M, buffers.K, buffers.N);
    // checkCuda(cudaGetLastError(), "launch matmul_a02_shmem");
    buffers.num_iters += 1;
}
