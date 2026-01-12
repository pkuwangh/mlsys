#pragma once

#include "matmul_utils.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#define D02_OFFSET(row, col, num_cols) ((row) * (num_cols) + (col))
#define D02_FETCH_FLOAT2(element) (reinterpret_cast<float2 *>(&(element))[0])
#define D02_FETCH_FLOAT4(element) (reinterpret_cast<float4 *>(&(element))[0])

#define D02_WMMA_M 16
#define D02_WMMA_N 16
#define D02_WMMA_K 16
#define D02_BLOCK_M 128
#define D02_BLOCK_N 128
#define D02_BLOCK_K 16

#define D02_NUM_WARP_M 2
#define D02_NUM_WARP_N 2
#define D02_BLOCK_SIZE (D02_NUM_WARP_M * D02_NUM_WARP_N * WARPSIZE)

// each block tile is 128x128, i.e. 8x8 warp tiles per block
// each block has 2x2 warps
// each warp tile is 16x16, map to wmma operation

template <const int BM, const int BN, const int BK, const int WMMA_M, const int WMMA_N, const int WMMA_K,
          const int NUM_WARP_M, const int NUM_WARP_N>
__global__ void matmul_d02_tc3_wmma_shmem(bf16 *A, bf16 *B, float *C, int M, int K, int N) {
    int block_tile_m = blockIdx.y;
    int block_tile_n = blockIdx.x;

    // move A/B/C to the first element of the block tile
    A += block_tile_m * BM * K;
    B += block_tile_n * BN;
    C += block_tile_m * BM * N + block_tile_n * BN;

    // num_warp_m x num_warp_n warps per block tile
    const int block_size = NUM_WARP_M * NUM_WARP_N * WARPSIZE;
    // all warps in one iteration handles a warp_sub_m x warp_sub_n sub-matrix
    constexpr int warp_sub_m = WMMA_M * NUM_WARP_M;
    constexpr int warp_sub_n = WMMA_N * NUM_WARP_N;
    // hence each warp needs to run warp_iter_m x warp_iter_n iterations to finish BM x BN block tile
    constexpr int warp_iter_m = BM / warp_sub_m;
    constexpr int warp_iter_n = BN / warp_sub_n;

    // warp position in block tile
    int warp_idx = threadIdx.x / WARPSIZE;
    int warp_row = warp_idx / NUM_WARP_N;
    int warp_col = warp_idx % NUM_WARP_N;

    // declare fragments, warp-level objects
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, bf16, nvcuda::wmma::row_major> frag_a;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, bf16, nvcuda::wmma::row_major>
        frag_b[warp_iter_n];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c[warp_iter_m][warp_iter_n];

    // initialize output fragment to zero
    for (int warp_sub_row_idx = 0; warp_sub_row_idx < warp_iter_m; ++warp_sub_row_idx) {
        for (int warp_sub_col_idx = 0; warp_sub_col_idx < warp_iter_n; ++warp_sub_col_idx) {
            nvcuda::wmma::fill_fragment(frag_c[warp_sub_row_idx][warp_sub_col_idx], 0.0f);
        }
    }

    // shared memory for the block tile
    __shared__ bf16 As[BM * BK];
    __shared__ bf16 Bs[BK * BN];

    // for global mem -> shared mem loading, calculate the indices for this thread
    int tile_m_a = threadIdx.x / (BK / 4);
    int tile_n_a = threadIdx.x % (BK / 4);
    int tile_m_stride_a = block_size / (BK / 4);
    // same for B matrix
    int tile_m_b = threadIdx.x / (BN / 4);
    int tile_n_b = threadIdx.x % (BN / 4);
    int tile_m_stride_b = block_size / (BN / 4);

    // loop through K dimension, in chunks of BK
    for (int blk_offset = 0; blk_offset < K; blk_offset += BK) {
        // populate shared memory; this process has no concept of warp tiling
        for (int offset = 0; offset < BM; offset += tile_m_stride_a) {
            const float2 tmp = D02_FETCH_FLOAT2(A[D02_OFFSET(tile_m_a + offset, tile_n_a * 4, K)]);
            D02_FETCH_FLOAT2(As[D02_OFFSET(tile_m_a + offset, tile_n_a * 4, BK)]) = tmp;
        }
        for (int offset = 0; offset < BK; offset += tile_m_stride_b) {
            const float2 tmp = D02_FETCH_FLOAT2(B[D02_OFFSET(tile_m_b + offset, tile_n_b * 4, N)]);
            D02_FETCH_FLOAT2(Bs[D02_OFFSET(tile_m_b + offset, tile_n_b * 4, BN)]) = tmp;
        }
        __syncthreads();

        // move A & B
        A += BK;
        B += BK * N;

        for (int k = 0; k < BK; k += WMMA_K) {
            #pragma unroll
            for (int warp_sub_row_idx = 0; warp_sub_row_idx < warp_iter_m; ++warp_sub_row_idx) {
                const int row = warp_row * WMMA_M + warp_sub_row_idx * warp_sub_m;
                nvcuda::wmma::load_matrix_sync(frag_a, As + row * BK + k, BK);
                #pragma unroll
                for (int warp_sub_col_idx = 0; warp_sub_col_idx < warp_iter_n; ++warp_sub_col_idx) {
                    const int col = warp_col * WMMA_N + warp_sub_col_idx * warp_sub_n;
                    nvcuda::wmma::load_matrix_sync(frag_b[warp_sub_col_idx], Bs + k * BN + col, BN);
                    nvcuda::wmma::mma_sync(frag_c[warp_sub_row_idx][warp_sub_col_idx], frag_a, frag_b[warp_sub_col_idx],
                                           frag_c[warp_sub_row_idx][warp_sub_col_idx]);
                }
            }
        }

        __syncthreads();
    }

    // store results
    for (int warp_sub_row_idx = 0; warp_sub_row_idx < warp_iter_m; ++warp_sub_row_idx) {
        const int row = warp_row * WMMA_M + warp_sub_row_idx * warp_sub_m;
        for (int warp_sub_col_idx = 0; warp_sub_col_idx < warp_iter_n; ++warp_sub_col_idx) {
            const int col = warp_col * WMMA_N + warp_sub_col_idx * warp_sub_n;
            nvcuda::wmma::store_matrix_sync(C + row * N + col, frag_c[warp_sub_row_idx][warp_sub_col_idx], N,
                                            nvcuda::wmma::mem_row_major);
        }
    }
}

void runMatmulD02Tc3WmmaShmem(MatmulBuffers &buffers) {
    dim3 blockDim = dim3(D02_BLOCK_SIZE, 1);
    dim3 gridDim = dim3(buffers.N / D02_BLOCK_N, buffers.M / D02_BLOCK_M);

    matmul_d02_tc3_wmma_shmem<D02_BLOCK_M, D02_BLOCK_N, D02_BLOCK_K, D02_WMMA_M, D02_WMMA_N, D02_WMMA_K, D02_NUM_WARP_M,
                              D02_NUM_WARP_N>
        <<<gridDim, blockDim>>>(buffers.dA_bf16, buffers.dB_bf16, buffers.dC, buffers.M, buffers.K, buffers.N);
    // checkCuda(cudaGetLastError(), "launch matmul_d02_tc3_wmma_shmem");
    buffers.num_iters += 1;
}
