#pragma once

#include "matmul_utils.cuh"
#include <cuda_runtime.h>

#define A03_BLOCK_MN_DIM 64
#define A03_BLOCK_K_DIM 8
#define A03_THREAD_TILE_SIZE 8

// comparing a02_shmem vs. a01_basic, there are 2 observations:
// 1. global mem access can be reduced with larger BM & BN but invariant to BK,
//    i.e. only related to the size of the result matrix tile; so, we can use
//    a larger BM & BN but a smaller BK to further reduce global mem access while
//    maintaining the same shared memory usage.
// 2. the compute / mem access (from shared memory) ratio did not improve, i.e. still 1:2;
//    this is because still each thread reads 2K elements to compute 1 element.
//    so if we let each thread be responsible for 1 column of TM elements, for each of the
//    K elements reading from B matrix, it can be shared by TM elements from A matrix.

template <const int BM, const int BN, const int BK, const int TM>
__global__ void matmul_a03_thread_tile_1d(float* A, float* B, float* C, int M, int K, int N) {
    // shared memory size: BM * BK * sizeof(float) + BK * BN * sizeof(float)
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int block_size = blockDim.x;  // BM * BN / TM, as each thread handles TM elements.

    // (tx, ty) points to this thread's tile's top-left element's position in result BM * BN tile.
    int tx = threadIdx.x % BN;
    int ty = threadIdx.x / BN * TM;

    // move to the first tile; and point to the first element of the tile
    A += by * BM * K;
    B += bx * BN;
    C += by * BM * N + bx * BN;

    // all (BM * BN / TM) threads need to load (BM * BK) tile from A matrix into shared memory.
    // here, should just consider using block_size threads to load BM * BK elements.
    int a_tile_y = threadIdx.x / BK;
    int a_tile_x = threadIdx.x % BK;
    int a_tile_y_stride = block_size / BK;
    // same for B, this thread loads (b_tile_y, b_tile_x) element from B matrix.
    int b_tile_y = threadIdx.x / BN;
    int b_tile_x = threadIdx.x % BN;
    int b_tile_y_stride = block_size / BN;

    float acc[TM + 1] = {0.0f};  // allocate an extra element for caching.
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
            acc[TM] = Bs[i * BN + tx];  // cache Bs matrix (i, tx)
            // calcualte (TM, 1) elements
            #pragma unroll
            for (int j = 0; j < TM; ++j) {
                acc[j] += As[(ty + j) * BK + i] * acc[TM];
            }
        }
        __syncthreads();
    }
    // for every TM=8 fma, load 8 elements from As and 1 element from Bs;
    // so the compute / mem access ratio is 8:9; improved from 1:2.
    #pragma unroll
    for (int j = 0; j < TM; ++j) {
        C[(ty + j) * N + tx] += acc[j];
    }
}

void runMatmulA03ThreadTile1D(MatmulBuffers &buffers) {
    dim3 blockDim = dim3(A03_BLOCK_MN_DIM * A03_BLOCK_MN_DIM / A03_THREAD_TILE_SIZE, 1);
    dim3 gridDim = dim3(buffers.N / A03_BLOCK_MN_DIM, buffers.M / A03_BLOCK_MN_DIM);

    matmul_a03_thread_tile_1d<A03_BLOCK_MN_DIM, A03_BLOCK_MN_DIM, A03_BLOCK_K_DIM, A03_THREAD_TILE_SIZE>
        <<<gridDim, blockDim>>>(buffers.dA, buffers.dB, buffers.dC, buffers.M, buffers.K, buffers.N);
    // checkCuda(cudaGetLastError(), "launch matmul_a03_thread_tile_1d");
    buffers.num_iters += 1;
}
