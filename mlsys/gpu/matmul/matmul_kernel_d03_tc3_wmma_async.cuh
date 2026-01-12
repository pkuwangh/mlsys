#pragma once

#include <cooperative_groups.h>
#include <cuda/pipeline>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include "matmul_utils.cuh"

#define D03_OFFSET(row, col, num_cols) ((row) * (num_cols) + (col))
#define D03_FETCH_FLOAT2(element) (reinterpret_cast<float2 *>(&(element))[0])

#define D03_WMMA_M 16
#define D03_WMMA_N 16
#define D03_WMMA_K 16
#define D03_BLOCK_M 128
#define D03_BLOCK_N 128
#define D03_BLOCK_K 16

#define D03_NUM_WARP_M 2
#define D03_NUM_WARP_N 2
#define D03_BLOCK_SIZE (D03_NUM_WARP_M * D03_NUM_WARP_N * WARPSIZE)

// copy in 16B chunks, i.e. 8x bf16
#define D03_CP_ASYNC_BYTES 16

template <const int BM, const int BN, const int BK, const int WMMA_M, const int WMMA_N, const int WMMA_K,
          const int NUM_WARP_M, const int NUM_WARP_N>
__global__ void matmul_d03_tc3_wmma_async(bf16 *A, bf16 *B, float *C, int M, int K, int N) {
    int block_tile_m = blockIdx.y;
    int block_tile_n = blockIdx.x;

    // move A/B/C to the first element of the block tile
    A += block_tile_m * BM * K;
    B += block_tile_n * BN;
    C += block_tile_m * BM * N + block_tile_n * BN;

    // all warps in one iteration handles a warp_sub_m x warp_sub_n sub-matrix
    constexpr int warp_sub_m = WMMA_M * NUM_WARP_M;
    constexpr int warp_sub_n = WMMA_N * NUM_WARP_N;
    // hence each warp needs to run warp_iter_m x warp_iter_n iterations to finish BM x BN block tile
    constexpr int warp_iter_m = BM / warp_sub_m;
    constexpr int warp_iter_n = BN / warp_sub_n;

    // warp position in block
    int warp_idx = threadIdx.x / WARPSIZE;
    int warp_row = warp_idx / NUM_WARP_N;
    int warp_col = warp_idx % NUM_WARP_N;

    // declare fragments, warp-level objects
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, bf16, nvcuda::wmma::row_major> frag_a;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, bf16, nvcuda::wmma::row_major>
        frag_b[warp_iter_n];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c[warp_iter_m][warp_iter_n];

// initialize output fragment to zero
#pragma unroll
    for (int warp_sub_row_idx = 0; warp_sub_row_idx < warp_iter_m; ++warp_sub_row_idx) {
#pragma unroll
        for (int warp_sub_col_idx = 0; warp_sub_col_idx < warp_iter_n; ++warp_sub_col_idx) {
            nvcuda::wmma::fill_fragment(frag_c[warp_sub_row_idx][warp_sub_col_idx], 0.0f);
        }
    }

    // 2-stage shared buffers
    alignas(16) __shared__ bf16 As[2][BM * BK];
    alignas(16) __shared__ bf16 Bs[2][BK * BN];

    namespace cg = cooperative_groups;
    auto cta = cg::this_thread_block();
    // define the shared state: silence the warning about static variable with dynamic init
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> pipe_state;
    // make the pipeline
    cuda::pipeline<cuda::thread_scope_block> pipe = cuda::make_pipeline(cta, &pipe_state);

    // helper: async copy one (BMxBK) + (BKxBN) tile into a buffer
    auto cp_async_tile = [&](int buf, int tile_idx) {
        // tile_idx is along K
        const int blk_offset = tile_idx * BK;

        // As: BM*BK bf16 = (BM*BK*2) bytes. With BM=128,BK=16 -> 4096B -> 256 chunks of 16B
        constexpr int As_bytes = BM * BK * sizeof(bf16);
        constexpr int Bs_bytes = BK * BN * sizeof(bf16);
        constexpr int As_chunks = As_bytes / D03_CP_ASYNC_BYTES;
        constexpr int Bs_chunks = Bs_bytes / D03_CP_ASYNC_BYTES;

        // each thread copies one chunk at a time
        for (int c = threadIdx.x; c < As_chunks; c += blockDim.x) {
            const int offset = c * (D03_CP_ASYNC_BYTES / sizeof(bf16));
            const int row = offset / BK;
            const int col = offset % BK;

            // blk_offset is on the K (col) dimension
            const bf16 *src = A + row * K + (blk_offset + col);
            bf16 *dst = &As[buf][row * BK + col];

            // TODO: this seems to be not really acting asynchronously
            cuda::memcpy_async(dst, src, cuda::aligned_size_t<D03_CP_ASYNC_BYTES>(D03_CP_ASYNC_BYTES), pipe);
        }

        for (int c = threadIdx.x; c < Bs_chunks; c += blockDim.x) {
            const int offset = c * (D03_CP_ASYNC_BYTES / sizeof(bf16));
            const int row = offset / BN;
            const int col = offset % BN;

            // for B matrix, the offset is on the K (row) dimension
            const bf16 *src = B + (blk_offset + row) * N + col;
            bf16 *dst = &Bs[buf][row * BN + col];

            cuda::memcpy_async(dst, src, cuda::aligned_size_t<D03_CP_ASYNC_BYTES>(D03_CP_ASYNC_BYTES), pipe);
        }
    };

    const int num_tiles = K / BK;

    // prologue, prefetch tile 0 into buf0
    pipe.producer_acquire(); // open a new copy stage; i.e. acquires an empty mailbox slot
    cp_async_tile(0, 0);     // threads issue async copies
    pipe.producer_commit();  // mark the stage as fully described, ready to wait-on

    // main loop
    for (int t = 0; t < num_tiles; ++t) {
        const int curr_buf = t & 1;

        // wait until the next committed stage is ready for consumption
        pipe.consumer_wait();
        // make sure all threads reach this point because next-stage will be warp-level
        __syncthreads();

        // start prefetching tile t+1, use the other buffer
        if (t + 1 < num_tiles) {
            pipe.producer_acquire();
            const int next_buf = 1 - curr_buf;
            cp_async_tile(next_buf, t + 1);
            pipe.producer_commit();
        }

        for (int k = 0; k < BK; k += WMMA_K) {
#pragma unroll
            for (int warp_sub_col_idx = 0; warp_sub_col_idx < warp_iter_n; ++warp_sub_col_idx) {
                const int col = warp_col * WMMA_N + warp_sub_col_idx * warp_sub_n;
                nvcuda::wmma::load_matrix_sync(frag_b[warp_sub_col_idx], &Bs[curr_buf][k * BN + col], BN);
            }

#pragma unroll
            for (int warp_sub_row_idx = 0; warp_sub_row_idx < warp_iter_m; ++warp_sub_row_idx) {
                const int row = warp_row * WMMA_M + warp_sub_row_idx * warp_sub_m;
                nvcuda::wmma::load_matrix_sync(frag_a, &As[curr_buf][row * BK + k], BK);

#pragma unroll
                for (int warp_sub_col_idx = 0; warp_sub_col_idx < warp_iter_n; ++warp_sub_col_idx) {
                    nvcuda::wmma::mma_sync(frag_c[warp_sub_row_idx][warp_sub_col_idx], frag_a, frag_b[warp_sub_col_idx],
                                           frag_c[warp_sub_row_idx][warp_sub_col_idx]);
                }
            }
        }

        __syncthreads();
        pipe.consumer_release();
    }

// Store results
#pragma unroll
    for (int warp_sub_row_idx = 0; warp_sub_row_idx < warp_iter_m; ++warp_sub_row_idx) {
        const int row = warp_row * WMMA_M + warp_sub_row_idx * warp_sub_m;
#pragma unroll
        for (int warp_sub_col_idx = 0; warp_sub_col_idx < warp_iter_n; ++warp_sub_col_idx) {
            const int col = warp_col * WMMA_N + warp_sub_col_idx * warp_sub_n;
            nvcuda::wmma::store_matrix_sync(C + row * N + col, frag_c[warp_sub_row_idx][warp_sub_col_idx], N,
                                            nvcuda::wmma::mem_row_major);
        }
    }
}

inline void runMatmulD03Tc3WmmaAsync(MatmulBuffers &buffers) {
    dim3 blockDim(D03_BLOCK_SIZE, 1);
    dim3 gridDim(buffers.N / D03_BLOCK_N, buffers.M / D03_BLOCK_M);

    matmul_d03_tc3_wmma_async<D03_BLOCK_M, D03_BLOCK_N, D03_BLOCK_K, D03_WMMA_M, D03_WMMA_N, D03_WMMA_K, D03_NUM_WARP_M,
                              D03_NUM_WARP_N>
        <<<gridDim, blockDim>>>(buffers.dA_bf16, buffers.dB_bf16, buffers.dC, buffers.M, buffers.K, buffers.N);
    // checkCuda(cudaGetLastError(), "launch matmul_d03_tc3_wmma_async");
    buffers.num_iters += 1;
}
