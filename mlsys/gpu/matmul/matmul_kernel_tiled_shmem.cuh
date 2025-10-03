#pragma once

#include "matmul_utils.cuh"
#include <cuda_runtime.h>

#define TILE_SIZE_X1 16
#define DEBUG_X1 0

__device__ int get_tile_element_offset(int tile_row, int tile_col, int row, int col, int width) {
    return (tile_row * TILE_SIZE_X1 + row) * width + (tile_col * TILE_SIZE_X1 + col);
}

__global__ void matmul_tiled_shmem(const float *A, const float *B, float *C, int M, int K, int N) {
    __shared__ float shared_A[TILE_SIZE_X1][TILE_SIZE_X1];
    __shared__ float shared_B[TILE_SIZE_X1][TILE_SIZE_X1];
    // each block computes one tile of the final result matrix
    int tile_row = blockIdx.y;
    int tile_col = blockIdx.x;
    // each thread computes one element of the tile
    int row = threadIdx.y;
    int col = threadIdx.x;
    // iterate tile by tile in the K dimension
    float acc = 0.0f;
    for (int i = 0; i < K / TILE_SIZE_X1; ++i) {
        // load the tile of A and B into shared memory
        int idx_A = get_tile_element_offset(tile_row, i, row, col, K);
        int idx_B = get_tile_element_offset(i, tile_col, row, col, N);
        shared_A[row][col] = A[idx_A];
        shared_B[row][col] = B[idx_B];
        if (DEBUG_X1 && tile_row == 0 && tile_col == 0) {
            printf(
                "row=%d col=%d loading A[%d]=%f B[%d]=%f\n",
                row, col, idx_A, A[idx_A], idx_B, B[idx_B]
            );
        }
        // sync to make sure the tile is loaded
        __syncthreads();
        // multiple Asub & Bsub
        for (int j = 0; j < TILE_SIZE_X1; ++j) {
            acc += shared_A[row][j] * shared_B[j][col];
        }
        if (DEBUG_X1 && tile_row == 0 && tile_col == 0) {
            printf(">>>> row=%d col=%d acc=%f\n", row, col, acc);
        }
        // sync before loading next tile
        __syncthreads();
    }
    // each thread stores one element of the tile
    C[get_tile_element_offset(tile_row, tile_col, row, col, N)] = acc;
}

void runMatmulTiledShmem(MatmulBuffers &buffers) {
    dim3 blockDim = dim3(TILE_SIZE_X1, TILE_SIZE_X1);
    dim3 gridDim = makeGrid2D(buffers.M, buffers.N, blockDim);

    matmul_tiled_shmem<<<gridDim, blockDim>>>(buffers.dA, buffers.dB, buffers.dC, buffers.M, buffers.K, buffers.N);

    // checkCuda(cudaGetLastError(), "launch matmul_tiled_shmem");
    buffers.num_iters += 1;
}
