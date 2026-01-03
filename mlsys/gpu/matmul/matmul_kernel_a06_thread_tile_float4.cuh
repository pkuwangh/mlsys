#pragma once

#include "matmul_utils.cuh"
#include <cuda_runtime.h>

#define A06_OFFSET(row, col, num_cols) ((row)*(num_cols)+(col))
#define A06_FETCH_FLOAT4(element) (reinterpret_cast<float4 *>(&(element))[0])

#define A06_BLOCK_M_DIM 128
#define A06_BLOCK_N_DIM 128
#define A06_BLOCK_K_DIM 8
#define A06_THREAD_TILE_TM_DIM 8
#define A06_THREAD_TILE_TN_DIM 8

// use float4 to reduce load/store & compute instructions.
// the reason A is transposed instead of B is because we want to
// keep accumulate out product which is the partial sum of a tile.

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void matmul_a06_thread_tile_float4(float *A, float *B, float *C, int M, int K, int N) {
    // shared memory size: BM * BK * sizeof(float) + BK * BN * sizeof(float)
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    int bx = blockIdx.x;
    int by = blockIdx.y;

    // (tx, ty) points to this thread's tile's top-left element's position in result BM * BN tile.
    int tx = (threadIdx.x % (BN / TN)) * TN;
    int ty = (threadIdx.x / (BN / TN)) * TM;

    // move to the first tile; and point to the first element of the tile
    A += by * BM * K;
    B += bx * BN;
    C += by * BM * N + bx * BN;

    // all block_size threads need to load (BM * BK) tile from A matrix into shared memory.
    const int block_size = BM * BN / (TM * TN);
    // each thread loads 4 elements, need so many rounds to load all elements.
    const int ld_As_rounds = (BM * BK) / (block_size * 4);

    // this thread loads (a_tile_y, a_tile_x) element from A matrix.
    int a_tile_y = (threadIdx.x / (BK / 4));
    int a_tile_x = (threadIdx.x % (BK / 4)) * 4;
    int a_tile_y_stride = block_size / (BK / 4);
    // same for B, this thread loads (b_tile_y, b_tile_x) element from B matrix.
    int b_tile_y = (threadIdx.x / (BN / 4));
    int b_tile_x = (threadIdx.x % (BN / 4)) * 4;
    int b_tile_y_stride = block_size / (BN / 4);

    // load As and put them into register to do transpose
    // not really necessary here, but useful when doing double buffering.
    float ld_a_reg[4 * ld_As_rounds] = {0.0f};

    // cache As & Bs
    float a_cache[TM] = {0.0f};
    float b_cache[TN] = {0.0f};

    // accumulate results
    float acc[TM][TN] = {0.0f};

    #pragma unroll
    for (int b = 0; b < K; b += BK) {
        #pragma unroll
        for (int i = 0; i < BM; i += a_tile_y_stride) {
            const int ld_idx = i / a_tile_y_stride * 4;
            const int a_idx = A06_OFFSET(a_tile_y + i, a_tile_x, K);
            A06_FETCH_FLOAT4(ld_a_reg[ld_idx]) = A06_FETCH_FLOAT4(A[a_idx]);
            // transpose and write into As, so that we can use float4 when reading from As.
            As[A06_OFFSET(a_tile_x + 0, a_tile_y + i, BM)] = ld_a_reg[ld_idx + 0];
            As[A06_OFFSET(a_tile_x + 1, a_tile_y + i, BM)] = ld_a_reg[ld_idx + 1];
            As[A06_OFFSET(a_tile_x + 2, a_tile_y + i, BM)] = ld_a_reg[ld_idx + 2];
            As[A06_OFFSET(a_tile_x + 3, a_tile_y + i, BM)] = ld_a_reg[ld_idx + 3];
        }
        #pragma unroll
        for (int i = 0; i < BK; i += b_tile_y_stride) {
            A06_FETCH_FLOAT4(Bs[A06_OFFSET(b_tile_y + i, b_tile_x, BN)]) =
                A06_FETCH_FLOAT4(B[A06_OFFSET(b_tile_y + i, b_tile_x, N)]);
        }
        __syncthreads();

        // move A & B
        A += BK;
        B += BK * N;

        // compute!
        #pragma unroll
        for (int i = 0; i < BK; ++i) {
            #pragma unroll
            for (int j = 0; j < TM; j += 4) {
                // note As is transposed
                A06_FETCH_FLOAT4(a_cache[j]) = A06_FETCH_FLOAT4(As[A06_OFFSET(i, ty + j, BM)]);
            }
            #pragma unroll
            for (int l = 0; l < TN; l += 4) {
                A06_FETCH_FLOAT4(b_cache[l]) = A06_FETCH_FLOAT4(Bs[A06_OFFSET(i, tx + l, BN)]);
            }
            #pragma unroll
            for (int j = 0; j < TM; j += 1) {
                #pragma unroll
                for (int l = 0; l < TN; l += 1) {
                    acc[j][l] += a_cache[j] * b_cache[l];
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int j = 0; j < TM; j += 1) {
        #pragma unroll
        for (int l = 0; l < TN; l += 4) {
            float4 ctmp = A06_FETCH_FLOAT4(C[A06_OFFSET(ty + j, tx + l, N)]);
            ctmp.x += acc[j][l];
            ctmp.y += acc[j][l + 1];
            ctmp.z += acc[j][l + 2];
            ctmp.w += acc[j][l + 3];
            A06_FETCH_FLOAT4(C[A06_OFFSET(ty + j, tx + l, N)]) = ctmp;
        }
    }
}

void runMatmulA06ThreadTileFloat4(MatmulBuffers &buffers) {
    dim3 blockDim = dim3(A06_BLOCK_M_DIM * A06_BLOCK_N_DIM / A06_THREAD_TILE_TM_DIM / A06_THREAD_TILE_TN_DIM, 1);
    dim3 gridDim = dim3(buffers.N / A06_BLOCK_N_DIM, buffers.M / A06_BLOCK_M_DIM);

    matmul_a06_thread_tile_float4<A06_BLOCK_M_DIM, A06_BLOCK_N_DIM, A06_BLOCK_K_DIM, A06_THREAD_TILE_TM_DIM, A06_THREAD_TILE_TN_DIM>
        <<<gridDim, blockDim>>>(buffers.dA, buffers.dB, buffers.dC, buffers.M, buffers.K, buffers.N);
    // checkCuda(cudaGetLastError(), "launch matmul_a06_thread_tile_float4");
    buffers.num_iters += 1;
}
