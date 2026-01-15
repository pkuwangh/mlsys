#pragma once

#include <cuda.h>
#include <cuda/barrier>

#include "matmul_utils.cuh"

#define E01_BM 64
#define E01_BN 64
#define E01_BK 64
#define E01_BLOCK_SIZE 128
#define E01_WGMMA_M 64
#define E01_WGMMA_N 64
#define E01_WGMMA_K 16

namespace e01 {

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

// basic warp-tiling with tensor core and TMA
// better warp-tiling would be having each warp handle multiple tiles over iterations.
// high-level flow:
// 1. launch TMA bulk async copy from GMEM to SMEM, from thread 0
// 2. barrier to block until all threads arrived & TMA finished
// 3. wgmma fence before the first wgmma.mma_async
// 4. submit wgmma.mma_async calls, back-to-back with same shape and accumulating into the same regs.
// 5. commit into a wgmma-group and wait for it.

__device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) { return (((x) & 0x3FFFF) >> 0x4); }

__device__ uint64_t make_smem_desc(bf16 *ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0x0000000000000000;
    desc |= matrix_descriptor_encode(addr);
    desc |= matrix_descriptor_encode((uint64_t)16) << 16;
    desc |= matrix_descriptor_encode((uint64_t)1024) << 32;
    desc |= 1llu << 62; // 128B swizzle
    return desc;
}

__device__ void warpgroup_arrive() {
    // `::: "memory"` is a compiler hint to disable memory reordering over this fence
    // fence to establish ordering b/w prior access to any warpgroup registers
    // and subsequent access to the same registers
    // `.sync` means the executing thread to wait until all threads in the warp excute this fence
    // `.aligned` means all threads in the warpgroup execute this in lockstep
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ void warpgroup_commit_batch() {
    // close off all the wgmma.mma_async ops in a group
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N> __device__ void warpgroup_wait() {
    static_assert(N >= 0 && N <= 7, "WGMMA wait: N must be in range [0, 7]");
    // wait for all launched groups are done
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

template <int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ void wgmma64(float d[4][8], bf16 *sA, bf16 *sB) {
    uint64_t desc_a = make_smem_desc(&sA[0]);
    uint64_t desc_b = make_smem_desc(&sB[0]);
    // similar to the fence, the mma_async is sync & aligned
    asm volatile("{\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
                 "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
                 " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
                 " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
                 " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31},"
                 " %32,"
                 " %33,"
                 " %34, %35, %36, %37, %38;\n"
                 "}\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]),
                   "+f"(d[0][6]), "+f"(d[0][7]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
                   "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]), "+f"(d[2][0]), "+f"(d[2][1]),
                   "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
                   "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]),
                   "+f"(d[3][6]), "+f"(d[3][7])
                 : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)),
                   "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

template <int BlockMajorSize, int BlockMinorSize>
void create_tensor_map(CUtensorMap *tma_map, bf16 *gmem_ptr, int blocks_height, int blocks_width) {
    void *gmem_address = (void *)gmem_ptr;
    uint64_t gmem_prob_shape[5] = {(uint64_t)BlockMinorSize * blocks_width, (uint64_t)BlockMajorSize * blocks_height, 1,
                                   1, 1};
    uint64_t gmem_prob_stride[5] = {sizeof(bf16), sizeof(bf16) * BlockMinorSize * blocks_width, 0, 0, 0};
    uint32_t smem_box_shape[5] = {uint32_t(BlockMinorSize), uint32_t(BlockMajorSize), 1, 1, 1};
    uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};

    // form a tensor map
    // encode all the metadata needed to transfer chunks of GMEM to SMEM
    // dtype: bf16
    // rank: 2 (matrix)
    // pointer: gemm_address/gmem_ptr
    // shape: fastest stride dimension first, (width, height); (K,M) for A
    // row stride: K*sizeof(bf16) for A
    // smem shape: (BK, BM) for A
    // swizzle: 128B pattern
    CUresult result = cuTensorMapEncodeTiled(tma_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, gmem_address,
                                             gmem_prob_shape, gmem_prob_stride + 1, smem_box_shape, smem_box_stride,
                                             CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                             CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    assert(result == CUDA_SUCCESS);
}

template <int BM, int BN, int BK, int WGMMA_M, int WGMMA_N, int WGMMA_K, int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
matmul_e01_tc4_basic(const CUtensorMap *tensorMapA, const CUtensorMap *tensorMapB, bf16 *C, int M, int N, int K) {
    // 128B-aligned smem
    __shared__ alignas(128) bf16 sA[BM * BK];
    __shared__ alignas(128) bf16 sB[BK * BN];

    // accumulator, each thread covers 4 x 8 thread tile
    // it looks C matrix is column-major
    // with (BN x BM) = 64 x 64, there are 16 x 8 such tiles
    // corresponding to 4x 32-thread warps
    float d[WGMMA_N / 16][8];
    static_assert(sizeof(d) * 128 == BM * BN * sizeof(float));
    memset(d, 0, sizeof(d));

    // the number of sliding blocks in the K dimension
    const int num_blocks_k = K / BK;
    // the position of this block in the result matrix
    const int num_block_n = blockIdx.x % (N / BN);
    const int num_block_m = blockIdx.x / (N / BN);

    // SMEM barriers for A & B
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier barA;
    __shared__ barrier barB;

    if (threadIdx.x == 0) {
        init(&barA, blockDim.x);
        init(&barB, blockDim.x);
        // make initialized barrier visible to async proxy (TMA)
        // Hopper's proxy memory model orders visibility b/w the async proxy (TMA) and
        // the generic proxy (normal thread ld/st) at CTA scope.
        cde::fence_proxy_async_shared_cta();
    }
    // make sure barriers are visible to all threads
    __syncthreads();

    barrier::arrival_token tokenA, tokenB;

    for (int block_k_iter = 0; block_k_iter < num_blocks_k; ++block_k_iter) {
        // only one thread launches TMA
        if (threadIdx.x == 0) {
            // dest ptr, tensor map, coord-0, coord-1, barrier
            // offset into GMEM for this CTA: (block_k_iter * BK, num_block_m * BM)
            cde::cp_async_bulk_tensor_2d_global_to_shared(&sA[0], tensorMapA, block_k_iter * BK, num_block_m * BM,
                                                          barA);
            // count thread arrivals, i.e. 1 from thread 0
            // update barrier with # bytes to wait for
            tokenA = cuda::device::barrier_arrive_tx(barA, 1, sizeof(sA));
            // note B is column-major, so the block window also slides right
            // to (block_k_iter * BK, num_block_n * BN)
            cde::cp_async_bulk_tensor_2d_global_to_shared(&sB[0], tensorMapB, block_k_iter * BK, num_block_n * BN,
                                                          barB);
            tokenB = cuda::device::barrier_arrive_tx(barB, 1, sizeof(sB));
        } else {
            // only contribute to thread arrival
            tokenA = barA.arrive();
            tokenB = barB.arrive();
        }
        // block until all threads arrived && TMA finished
        barA.wait(std::move(tokenA));
        barB.wait(std::move(tokenB));
        // TODO: why do we need this?
        __syncthreads();

        // tensor core matmul
        // wgmma.mma_async leverages 4 collaborating warps to compute the matmul
        // for bf16 operands, `wgmma` supports the shapes in the form of `m64nNk16`,
        // where `N` can be 8, 16, 24, ..., 256 and larger N value tends to be more efficient.

        // all 4 warps need to arrive at this fence
        warpgroup_arrive();
        // BK=64, WGMMA_K=16, so 4 wgmma calls with the same shape accumulating into the same regs.
        wgmma64<1, 1, 1, 0, 0>(d, &sA[0], &sB[0]);
        wgmma64<1, 1, 1, 0, 0>(d, &sA[WGMMA_K], &sB[WGMMA_K]);
        wgmma64<1, 1, 1, 0, 0>(d, &sA[2 * WGMMA_K], &sB[2 * WGMMA_K]);
        wgmma64<1, 1, 1, 0, 0>(d, &sA[3 * WGMMA_K], &sB[3 * WGMMA_K]);
        // commit all prior wgmma.mma_async operations into a wgmma-group.
        warpgroup_commit_batch();
        warpgroup_wait<0>();
    }

    // Store
    {
        int tid = threadIdx.x;
        int lane = tid % WARPSIZE;
        int warp = tid / WARPSIZE;
        // C matrix is column-major, but row still refers to M dimension
        // each wrap covers 16 rows in the 64x64 C submatrix
        // row index below covers 0-7 of my allocated rows, but it also writes row + 8
        uint32_t row = warp * 16 + lane / 4;
        bf16 *block_C = C + num_block_n * BN * M + num_block_m * BM;

        for (int m_it = 0; m_it < BM / WGMMA_M; ++m_it) {
            for (int n_it = 0; n_it < BN / WGMMA_N; ++n_it) {
                for (int w = 0; w < WGMMA_N / 16; ++w) {
                    // enumerate 4 chunks of 16 columns
                    // 2 * (tid % 4) yields 0,2,4,6 and it writes col + 1 also
                    int col = 16 * w + 2 * (tid % 4);
#define IDX(i, j) ((j + n_it * WGMMA_N) * M + ((i) + m_it * WGMMA_M))

                    // so each thread writes 2 consecutive elements in each 8 x 8 chunk of C
                    block_C[IDX(row, col)] = d[w][0];
                    block_C[IDX(row, col + 1)] = d[w][1];
                    block_C[IDX(row + 8, col)] = d[w][2];
                    block_C[IDX(row + 8, col + 1)] = d[w][3];

                    block_C[IDX(row, col + 8)] = d[w][4];
                    block_C[IDX(row, col + 9)] = d[w][5];
                    block_C[IDX(row + 8, col + 8)] = d[w][6];
                    block_C[IDX(row + 8, col + 9)] = d[w][7];

#undef IDX
                }
            }
        }
    }
}

template <int BlockMajorSize, int BlockMinorSize>
__host__ static inline CUtensorMap *allocate_and_create_tensor_map(bf16 *src, int blocks_height, int blocks_width) {
    CUtensorMap *tma_map_d;
    cudaMalloc(&tma_map_d, sizeof(CUtensorMap));
    CUtensorMap tma_map_host;
    create_tensor_map<BlockMajorSize, BlockMinorSize>(&tma_map_host, src, blocks_height, blocks_width);
    cudaMemcpy(tma_map_d, &tma_map_host, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    return tma_map_d;
}

} // namespace e01

CUtensorMap *e01_d_tma_map_A = 0;
CUtensorMap *e01_d_tma_map_B = 0;

void runMatmulE01Tc4Basic(MatmulBuffers &buffers) {
    // always recreate; TODO: is there memory leak here?
    e01_d_tma_map_A =
        e01::allocate_and_create_tensor_map<E01_BM, E01_BK>(buffers.dA_bf16, buffers.M / E01_BM, buffers.K / E01_BK);
    e01_d_tma_map_B =
        e01::allocate_and_create_tensor_map<E01_BN, E01_BK>(buffers.dB_bf16_t, buffers.N / E01_BN, buffers.K / E01_BK);
    dim3 blockDim = dim3(E01_BLOCK_SIZE, 1);
    dim3 gridDim = dim3((buffers.M / E01_BM) * (buffers.N / E01_BN), 1);

    e01::matmul_e01_tc4_basic<E01_BM, E01_BN, E01_BK, E01_WGMMA_M, E01_WGMMA_N, E01_WGMMA_K, E01_BLOCK_SIZE>
        <<<gridDim, blockDim>>>(e01_d_tma_map_A, e01_d_tma_map_B, buffers.dC_bf16, buffers.M, buffers.N, buffers.K);
    // checkCuda(cudaGetLastError(), "launch matmul_e01_tc4_basic");
    buffers.num_iters += 1;
}
