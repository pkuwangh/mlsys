#pragma once

#include "matmul_utils.cuh"
#include <cuda_runtime.h>

#define A07_OFFSET(row, col, num_cols) ((row)*(num_cols)+(col))
#define A07_FETCH_FLOAT4(element) (reinterpret_cast<float4 *>(&(element))[0])

#define A07_BLOCK_M_DIM 128
#define A07_BLOCK_N_DIM 128
#define A07_BLOCK_K_DIM 8
#define A07_THREAD_TILE_TM_DIM 8
#define A07_THREAD_TILE_TN_DIM 8

// double buffering so when some warps are loading data, others can compute.

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void matmul_a07_double_buffering(float* A, float* B, float* C, int M, int K, int N) {
    // shared memory size: BM * BK * sizeof(float) + BK * BN * sizeof(float)
    // then double them
    __shared__ float As[2][BM * BK];
    __shared__ float Bs[2][BK * BN];

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
    const int ld_Bs_rounds = (BK * BN) / (block_size * 4);

    // this thread loads (a_tile_y, a_tile_x) element from A matrix.
    int a_tile_y = (threadIdx.x / (BK / 4));
    int a_tile_x = (threadIdx.x % (BK / 4)) * 4;
    int a_tile_y_stride = block_size / (BK / 4);
    // same for B, this thread loads (b_tile_y, b_tile_x) element from B matrix.
    int b_tile_y = (threadIdx.x / (BN / 4));
    int b_tile_x = (threadIdx.x % (BN / 4)) * 4;
    int b_tile_y_stride = block_size / (BN / 4);

    // load As and put them into register to do transpose
    float ld_a_reg[4 * ld_As_rounds] = {0.0f};
    float ld_b_reg[4 * ld_Bs_rounds] = {0.0f};

    // cache As & Bs
    float a_cache[TM] = {0.0f};
    float b_cache[TN] = {0.0f};

    // accumulate results
    float acc[TM][TN] = {0.0f};

    // first round of loading from global memory to shared memory.
    #pragma unroll
    for (int i = 0; i < BM; i += a_tile_y_stride) {
        const int ld_idx = i / a_tile_y_stride * 4;
        const int a_idx = A07_OFFSET(a_tile_y + i, a_tile_x, K);
        A07_FETCH_FLOAT4(ld_a_reg[ld_idx]) = A07_FETCH_FLOAT4(A[a_idx]);
        // transpose and write into As, so that we can use float4 when reading from As.
        // writing to As[0][]
        As[0][A07_OFFSET(a_tile_x + 0, a_tile_y + i, BM)] = ld_a_reg[ld_idx + 0];
        As[0][A07_OFFSET(a_tile_x + 1, a_tile_y + i, BM)] = ld_a_reg[ld_idx + 1];
        As[0][A07_OFFSET(a_tile_x + 2, a_tile_y + i, BM)] = ld_a_reg[ld_idx + 2];
        As[0][A07_OFFSET(a_tile_x + 3, a_tile_y + i, BM)] = ld_a_reg[ld_idx + 3];
    }
    #pragma unroll
    for (int i = 0; i < BK; i += b_tile_y_stride) {
        A07_FETCH_FLOAT4(Bs[0][A07_OFFSET(b_tile_y + i, b_tile_x, BN)]) =
            A07_FETCH_FLOAT4(B[A07_OFFSET(b_tile_y + i, b_tile_x, N)]);
    }
    // sync
    __syncthreads();

    // now the loop begins
    int write_idx = 1;
    int read_idx = write_idx ^ 1;
    int k = 0;
    do {
        // first, load next-iteration data from global memory to registers.
        k += BK;
        if (k < K) {
            // move A & B
            A += BK;
            B += BK * N;
            // load BMxBK tile from A matrix into registers.
            #pragma unroll
            for (int i = 0; i < BM; i += a_tile_y_stride) {
                const int ld_idx = i / a_tile_y_stride * 4;
                A07_FETCH_FLOAT4(ld_a_reg[ld_idx]) = A07_FETCH_FLOAT4(A[A07_OFFSET(a_tile_y + i, a_tile_x, K)]);
            }
            // same for B
            #pragma unroll
            for (int i = 0; i < BK; i += b_tile_y_stride) {
                const int ld_idx = i / b_tile_y_stride * 4;
                A07_FETCH_FLOAT4(ld_b_reg[ld_idx]) = A07_FETCH_FLOAT4(B[A07_OFFSET(b_tile_y + i, b_tile_x, N)]);
            }
        }

        // second, compute
        read_idx = write_idx ^ 1;
        #pragma unroll
        for (int i = 0; i < BK; ++i) {
            #pragma unroll
            for (int j = 0; j < TM; j += 4) {
                A07_FETCH_FLOAT4(a_cache[j]) = A07_FETCH_FLOAT4(As[read_idx][A07_OFFSET(i, ty + j, BM)]);
            }
            #pragma unroll
            for (int l = 0; l < TN; l += 4) {
                A07_FETCH_FLOAT4(b_cache[l]) = A07_FETCH_FLOAT4(Bs[read_idx][A07_OFFSET(i, tx + l, BN)]);
            }
            #pragma unroll
            for (int j = 0; j < TM; j += 1) {
                for (int l = 0; l < TN; l += 1) {
                    acc[j][l] += a_cache[j] * b_cache[l];
                }
            }
        }

        // third, write next-iteration data from registers to shared memory.
        if (k < K) {
            #pragma unroll
            for (int i = 0; i < BM; i += a_tile_y_stride) {
                const int ld_idx = i / a_tile_y_stride * 4;
                As[write_idx][A07_OFFSET(a_tile_x + 0, a_tile_y + i, BM)] = ld_a_reg[ld_idx + 0];
                As[write_idx][A07_OFFSET(a_tile_x + 1, a_tile_y + i, BM)] = ld_a_reg[ld_idx + 1];
                As[write_idx][A07_OFFSET(a_tile_x + 2, a_tile_y + i, BM)] = ld_a_reg[ld_idx + 2];
                As[write_idx][A07_OFFSET(a_tile_x + 3, a_tile_y + i, BM)] = ld_a_reg[ld_idx + 3];
            }
            #pragma unroll
            for (int i = 0; i < BK; i += b_tile_y_stride) {
                const int ld_idx = i / b_tile_y_stride * 4;
                A07_FETCH_FLOAT4(Bs[write_idx][A07_OFFSET(b_tile_y + i, b_tile_x, BN)]) = A07_FETCH_FLOAT4(ld_b_reg[ld_idx]);
            }
        }

        // only sync point
        __syncthreads();

        // move buffer pointer
        write_idx = write_idx ^ 1;

    } while (k < K);

    #pragma unroll
    for (int j = 0; j < TM; j += 1) {
        #pragma unroll
        for (int l = 0; l < TN; l += 4) {
            float4 ctmp = A07_FETCH_FLOAT4(C[A07_OFFSET(ty + j, tx + l, N)]);
            ctmp.x += acc[j][l];
            ctmp.y += acc[j][l + 1];
            ctmp.z += acc[j][l + 2];
            ctmp.w += acc[j][l + 3];
            A07_FETCH_FLOAT4(C[A07_OFFSET(ty + j, tx + l, N)]) = ctmp;
        }
    }
}

void runMatmulA07DoubleBuffering(MatmulBuffers &buffers) {
    dim3 blockDim = dim3(A07_BLOCK_M_DIM * A07_BLOCK_N_DIM / A07_THREAD_TILE_TM_DIM / A07_THREAD_TILE_TN_DIM, 1);
    dim3 gridDim = dim3(buffers.N / A07_BLOCK_N_DIM, buffers.M / A07_BLOCK_M_DIM);

    matmul_a07_double_buffering<A07_BLOCK_M_DIM, A07_BLOCK_N_DIM, A07_BLOCK_K_DIM, A07_THREAD_TILE_TM_DIM, A07_THREAD_TILE_TN_DIM>
        <<<gridDim, blockDim>>>(buffers.dA, buffers.dB, buffers.dC, buffers.M, buffers.K, buffers.N);
    // checkCuda(cudaGetLastError(), "launch matmul_a07_double_buffering");
    buffers.num_iters += 1;
}
