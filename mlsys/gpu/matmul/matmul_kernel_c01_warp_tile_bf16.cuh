#pragma once

#include "matmul_utils.cuh"
#include <cuda_runtime.h>

#define C01_OFFSET(row, col, num_cols) ((row) * (num_cols) + (col))
#define C01_FETCH_FLOAT2(element) (reinterpret_cast<float2 *>(&(element))[0])

// block tile 128 x 128 - each thread block handles a 128x128 Csub.
#define C01_BLOCK_TILE_M 128
#define C01_BLOCK_TILE_N 128
// each step, load 16 columns/rows along the K dimension into shared memory.
// this size will not change how we partition the work,
// because each thread computes outer product and accumulates along this direction.
#define C01_BLOCK_CHUNK_K 16
// warp tile 64 x 64 - each warp handles a 64x64 Csub.
// so warp count per block is (128 x 128) / (64 x 64) = 4
#define C01_WARP_TILE_M 64
#define C01_WARP_TILE_N 64
// thread tile 8 x 4 - each thread handles a 8x4 Csub.
#define C01_THREAD_TILE_M 8
#define C01_THREAD_TILE_N 4
// when computing, organize 32-thread warp as 8x4 so it computes a 64x16 Csub.
// to finish a 64x64 Csub, it needs 1 iteration on Y axis and 4 iterations on X axis.
#define C01_WARP_ITER_Y 1
#define C01_WARP_ITER_X 4

template <const int BM, const int BN, const int BK, const int WM, const int WN, const int WM_ITER, const int TM,
          const int TN>
__global__ void matmul_c01_warp_tile_bf16(bf16* A, bf16* B, bf16* C, int M, int K, int N) {
    // shared memory size: BM * BK * sizeof(bf16) + BK * BN * sizeof(bf16)
    __shared__ bf16 As[BM * BK];
    __shared__ bf16 Bs[BK * BN];

    int bx = blockIdx.x;
    int by = blockIdx.y;

    // move the pointers
    // A to the first element of by-th BM-sized row
    A += by * BM * K;
    // B to the first element of bx-th BN-sized column
    B += bx * BN;
    // C to the first element of the target block tile
    C += by * BM * N + bx * BN;

    // block size can be derived
    constexpr int BLOCK_SIZE = (BM * BN) / (WM * WN) * WARPSIZE;

    // when loading BK x BN chunk from A into As
    // organize 32-thread warp as 8x4, so each warp loads 8 rows with 16 elements each per iteration.
    // still the 4x with float2 vs. bf16
    const int tile_y_a = threadIdx.x / (BK / 4);
    const int tile_x_a = threadIdx.x % (BK / 4);
    const int tile_y_stride_a = BLOCK_SIZE / (BK / 4);

    // when loading BK x BN chunk from B into Bs
    // organize 32-thread warp as 1x32, so each warp loads 1 row of 128 elements per iteration.
    const int tile_y_b = threadIdx.x / (BN / 4);
    const int tile_x_b = threadIdx.x % (BN / 4);
    const int tile_y_stride_b = BLOCK_SIZE / (BN / 4);

    // position of this warp in block
    const int warp_idx = threadIdx.x / WARPSIZE;
    const int warp_col = warp_idx % (BN / WN);
    const int warp_row = warp_idx / (BN / WN);

    // it takes multiple iterations for a warp to finish its WM x WN warp tile.
    // calculate the
    constexpr int WN_ITER = (WM * WN) / (TM * TN) / WARPSIZE / WM_ITER; // 64 * 64 / (8 * 4) / 32 / 1 = 4
    constexpr int WSUBM = WM / WM_ITER;                                 // 64
    constexpr int WSUBN = WN / WN_ITER;                                 // 16

    // help to locate the thread tile
    const int thread_idx = threadIdx.x % WARPSIZE;
    const int thread_col = thread_idx % (WSUBN / TN);
    const int thread_row = thread_idx / (WSUBN / TN);

    // cache As & Bs
    bf16 reg_a[TM * WM_ITER] = {0.0f};
    bf16 reg_b[TN * WN_ITER] = {0.0f};
    // accumulate results, bf16 * bf16 -> float
    float acc[TM * WM_ITER][TN * WN_ITER] = {0.0f};

    // loop through K dimension, in chunks of BK
    for (int blk_offset = 0; blk_offset < K; blk_offset += BK) {
        // load BM x BK chunk from A into As
        for (int offset = 0; offset < BM; offset += tile_y_stride_a) {
            bf16 tmp[4];
            const float2 x = C01_FETCH_FLOAT2(A[C01_OFFSET(tile_y_a + offset, tile_x_a * 4, K)]);
            memcpy(&tmp[0], &x, sizeof(bf16) * 4);
            // transpose so As is BK x BM
            As[C01_OFFSET(tile_x_a * 4 + 0, tile_y_a + offset, BM)] = tmp[0];
            As[C01_OFFSET(tile_x_a * 4 + 1, tile_y_a + offset, BM)] = tmp[1];
            As[C01_OFFSET(tile_x_a * 4 + 2, tile_y_a + offset, BM)] = tmp[2];
            As[C01_OFFSET(tile_x_a * 4 + 3, tile_y_a + offset, BM)] = tmp[3];
        }
        // load BK x BN chunk from B into Bs
        for (int offset = 0; offset < BK; offset += tile_y_stride_b) {
            const float2 tmp = C01_FETCH_FLOAT2(B[C01_OFFSET(tile_y_b + offset, tile_x_b * 4, N)]);
            C01_FETCH_FLOAT2(Bs[C01_OFFSET(tile_y_b + offset, tile_x_b * 4, BN)]) = tmp;
        }
        // sync threads to make sure As & Bs are loaded
        __syncthreads();

        // move A & B
        A += BK;     // move right
        B += BK * N; // move down

        // compute
        for (int dot_idx = 0; dot_idx < BK; dot_idx += 1) {
            // load As into reg_a
            for (int warp_sub_row_idx = 0; warp_sub_row_idx < WM_ITER; warp_sub_row_idx += 1) {
                for (int i = 0; i < TM; i += 1) {
                    reg_a[warp_sub_row_idx * TM + i] =
                        As[C01_OFFSET(dot_idx, warp_row * WM + warp_sub_row_idx * WSUBM + thread_row * TM + i, BM)];
                }
            }
            // load Bs into reg_b
            for (int warp_sub_col_idx = 0; warp_sub_col_idx < WN_ITER; warp_sub_col_idx += 1) {
                for (int j = 0; j < TN; j += 1) {
                    reg_b[warp_sub_col_idx * TN + j] =
                        Bs[C01_OFFSET(dot_idx, warp_col * WN + warp_sub_col_idx * WSUBN + thread_col * TN + j, BN)];
                }
            }
            // compute
            for (int warp_sub_row_idx = 0; warp_sub_row_idx < WM_ITER; warp_sub_row_idx += 1) {
                for (int warp_sub_col_idx = 0; warp_sub_col_idx < WN_ITER; warp_sub_col_idx += 1) {
                    for (int i = 0; i < TM; i += 1) {
                        for (int j = 0; j < TN; j += 1) {
                            acc[warp_sub_row_idx * TM + i][warp_sub_col_idx * TN + j] +=
                                __bfloat162float(reg_a[warp_sub_row_idx * TM + i]) *
                                __bfloat162float(reg_b[warp_sub_col_idx * TN + j]);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
    // accumulate results
    for (int warp_sub_row_idx = 0; warp_sub_row_idx < WM_ITER; warp_sub_row_idx += 1) {
        for (int warp_sub_col_idx = 0; warp_sub_col_idx < WN_ITER; warp_sub_col_idx += 1) {
            // get the pointer to the sub warp tile
            bf16 *c_warp =
                C + C01_OFFSET(warp_row * WM + warp_sub_row_idx * WSUBM, warp_col * WN + warp_sub_col_idx * WSUBN, N);
            for (int i = 0; i < TM; i += 1) {
                for (int j = 0; j < TN; j += 4) {
                    bf16 tmp[4];
                    const int local_row_idx = warp_sub_row_idx * TM + i;
                    const int local_col_idx = warp_sub_col_idx * TN + j;
                    tmp[0] = acc[local_row_idx][local_col_idx + 0];
                    tmp[1] = acc[local_row_idx][local_col_idx + 1];
                    tmp[2] = acc[local_row_idx][local_col_idx + 2];
                    tmp[3] = acc[local_row_idx][local_col_idx + 3];
                    float2 x;
                    memcpy(&x, &tmp, sizeof(bf16) * 4);
                    C01_FETCH_FLOAT2(c_warp[C01_OFFSET(thread_row * TM + i, thread_col * TN + j, N)]) = x;
                }
            }
        }
    }
}

void runMatmulC01WarpTileBf16(MatmulBuffers &buffers) {
    dim3 blockDim = dim3((C01_BLOCK_TILE_M * C01_BLOCK_TILE_N) / (C01_WARP_TILE_M * C01_WARP_TILE_N) * WARPSIZE, 1);
    dim3 gridDim = dim3(buffers.N / C01_BLOCK_TILE_N, buffers.M / C01_BLOCK_TILE_M);

    matmul_c01_warp_tile_bf16<C01_BLOCK_TILE_M, C01_BLOCK_TILE_N, C01_BLOCK_CHUNK_K, C01_WARP_TILE_M, C01_WARP_TILE_N,
                              C01_WARP_ITER_Y, C01_THREAD_TILE_M, C01_THREAD_TILE_N>
        <<<gridDim, blockDim>>>(buffers.dA_bf16, buffers.dB_bf16, buffers.dC_bf16, buffers.M, buffers.K, buffers.N);
    // checkCuda(cudaGetLastError(), "launch matmul_c01_warp_tile_bf16");
    buffers.num_iters += 1;
}
