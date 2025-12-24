#pragma once

#include "matmul_utils.cuh"
#include <cuda_runtime.h>

#define A04_BLOCK_M_DIM 128
#define A04_BLOCK_N_DIM 128
#define A04_BLOCK_K_DIM 8
#define A04_THREAD_TILE_TM_DIM 8
#define A04_THREAD_TILE_TN_DIM 8

// a03_thread_tile_1d
// - uses 512 threads to cover a 64x64 sub-matrix.
// - BK is reduced to 8 since it does not affect the global memory access.
// - each thread handles a 8x1 tile to improve the compute / mem access ratio.
// next we can use 2D thread tile to
// - increase BM/BN to further reduce global memory access without increasing threads per block.
// - let each thread runs longer that should be more friendly to pipelining.

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void matmul_a04_thread_tile_2d(float *A, float *B, float *C, int M, int K, int N) {
    // shared memory size: BM * BK * sizeof(float) + BK * BN * sizeof(float)
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int block_size = blockDim.x;  // BM * BN / (TM * TN), as each thread handles TM * TN elements.

    // (tx, ty) points to this thread's tile's top-left element's position in result BM * BN tile.
    int tx = (threadIdx.x % (BN / TN)) * TN;
    int ty = (threadIdx.x / (BN / TN)) * TM;

    // move to the first tile; and point to the first element of the tile
    A += by * BM * K;
    B += bx * BN;
    C += by * BM * N + bx * BN;

    // all block_size threads need to load (BM * BK) tile from A matrix into shared memory.
    // TM/TN is kind of irrelevant here.
    int a_tile_y = threadIdx.x / BK;
    int a_tile_x = threadIdx.x % BK;
    int a_tile_y_stride = block_size / BK;
    // same for B, this thread loads (b_tile_y, b_tile_x) element from B matrix.
    int b_tile_y = threadIdx.x / BN;
    int b_tile_x = threadIdx.x % BN;
    int b_tile_y_stride = block_size / BN;

    float acc[TM][TN] = {0.0f};
    #pragma unroll
    for (int b = 0; b < K; b += BK) {
        #pragma unroll
        for (int i = 0; i < BM; i += a_tile_y_stride) {
            As[(a_tile_y + i) * BK + a_tile_x] = A[(a_tile_y + i) * K + a_tile_x];
        }
        #pragma unroll
        for (int i = 0; i < BK; i += b_tile_y_stride) {
            Bs[(b_tile_y + i) * BN + b_tile_x] = B[(b_tile_y + i) * N + b_tile_x];
        }
        __syncthreads();

        // move A & B
        A += BK;
        B += BK * N;

        // compute!
        #pragma unroll
        for (int i = 0; i < BK; ++i) {
            // calcualte (TM, TN) elements
            #pragma unroll
            for (int j = 0; j < TM; ++j) {
                #pragma unroll
                for (int l = 0; l < TN; ++l) {
                    acc[j][l] += As[(ty + j) * BK + i] * Bs[i * BN + (tx + l)];
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int j = 0; j < TM; ++j) {
        #pragma unroll
        for (int l = 0; l < TN; ++l) {
            C[(ty + j) * N + (tx + l)] += acc[j][l];
        }
    }
}

void runMatmulA04ThreadTile2D(MatmulBuffers &buffers) {
    dim3 blockDim = dim3(A04_BLOCK_M_DIM * A04_BLOCK_N_DIM / A04_THREAD_TILE_TM_DIM / A04_THREAD_TILE_TN_DIM, 1);
    dim3 gridDim = dim3(buffers.N / A04_BLOCK_N_DIM, buffers.M / A04_BLOCK_M_DIM);

    matmul_a04_thread_tile_2d<A04_BLOCK_M_DIM, A04_BLOCK_N_DIM, A04_BLOCK_K_DIM, A04_THREAD_TILE_TM_DIM, A04_THREAD_TILE_TN_DIM>
        <<<gridDim, blockDim>>>(buffers.dA, buffers.dB, buffers.dC, buffers.M, buffers.K, buffers.N);
    // checkCuda(cudaGetLastError(), "launch matmul_a04_thread_tile_2d");
    buffers.num_iters += 1;
}
