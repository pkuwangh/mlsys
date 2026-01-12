#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include "matmul_utils.cuh"

#define D01_WMMA_M 16
#define D01_WMMA_N 16
#define D01_WMMA_K 16

// minimal example of using tensor core for matmul
// each block has exactly one warp, which sends a 16x16 tile to tensor core

__global__ void matmul_d01_tc3_wmma_minimal(bf16 *A, bf16 *B, float *C, int M, int K, int N) {
    // tiles organized as 2d grid
    int tile_m = blockIdx.y;
    int tile_n = blockIdx.x;

    const int WMMA_M = D01_WMMA_M;
    const int WMMA_N = D01_WMMA_N;
    const int WMMA_K = D01_WMMA_K;

    // declare fragments
    // A fragment is a warp-level object
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, bf16, nvcuda::wmma::row_major> frag_a;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, bf16, nvcuda::wmma::row_major> frag_b;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c;
    // C is not loaded from memory, so no need to specify the layout

    // initialize output fragment to zero
    nvcuda::wmma::fill_fragment(frag_c, 0.0f);

    // loop through K dimension, in chunks of WMMA_K
    for (int blk_offset = 0; blk_offset < K; blk_offset += WMMA_K) {
        int a_row = tile_m * WMMA_M;
        int a_col = blk_offset;
        int b_row = blk_offset;
        int b_col = tile_n * WMMA_N;

        // load inputs
        // this is a warp collective, all 32 threads participate to load the tile
        nvcuda::wmma::load_matrix_sync(frag_a, A + a_row * K + a_col, K);
        nvcuda::wmma::load_matrix_sync(frag_b, B + b_row * N + b_col, N);

        // matmul with tensor core
        // again a warp collective to issue tensor-core operation
        nvcuda::wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
    }

    // store results
    int c_row = tile_m * WMMA_M;
    int c_col = tile_n * WMMA_N;
    // mem_row_major instead of row_major for storing to memory
    nvcuda::wmma::store_matrix_sync(C + c_row * N + c_col, frag_c, N, nvcuda::wmma::mem_row_major);
}

void runMatmulD01Tc3WmmaMinimal(MatmulBuffers &buffers) {
    dim3 blockDim = dim3(WARPSIZE, 1);
    dim3 gridDim = dim3(buffers.N / D01_WMMA_N, buffers.M / D01_WMMA_M);

    matmul_d01_tc3_wmma_minimal<<<gridDim, blockDim>>>(buffers.dA_bf16, buffers.dB_bf16, buffers.dC, buffers.M,
                                                          buffers.K, buffers.N);
    // checkCuda(cudaGetLastError(), "launch matmul_d01_tc3_wmma_minimal");
    buffers.num_iters += 1;
}
