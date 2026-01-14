#pragma once

#include <iostream>
#include <cuda.h>
#include <cuda/barrier>

#include "matmul_utils.cuh"

#define E02_BM 128
#define E02_BN 128
#define E02_BK 64
#define E02_BLOCK_SIZE 128

namespace e02 {

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

// increase the block tile size to 128x128 from 64x64
// also increase the tc size to m64n128k16 from m64n64k16
// then run each warpgroup for multiple iterations

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

template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ void wgmma256(float d[16][8], bf16* sA, bf16* sB) {
    uint64_t desc_a = make_smem_desc(&sA[0]);
    uint64_t desc_b = make_smem_desc(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
        " %96,  %97,  %98,  %99,  %100, %101, %102, %103,  "
        " %104, %105, %106, %107, %108, %109, %110, %111,  "
        " %112, %113, %114, %115, %116, %117, %118, %119,  "
        " %120, %121, %122, %123, %124, %125, %126, %127},"
        " %128,"
        " %129,"
        " %130,    %131,  %132,  %133,  %134;\n"
        "}\n"
        :   "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
            "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
            "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
            "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7]),
            "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]),
            "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
            "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[6][4]), "+f"(d[6][5]), "+f"(d[6][6]), "+f"(d[6][7]),
            "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7]),
            "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]), "+f"(d[8][4]), "+f"(d[8][5]), "+f"(d[8][6]), "+f"(d[8][7]),
            "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]), "+f"(d[9][4]), "+f"(d[9][5]), "+f"(d[9][6]), "+f"(d[9][7]),
            "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]), "+f"(d[10][3]), "+f"(d[10][4]), "+f"(d[10][5]), "+f"(d[10][6]), "+f"(d[10][7]),
            "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]), "+f"(d[11][3]), "+f"(d[11][4]), "+f"(d[11][5]), "+f"(d[11][6]), "+f"(d[11][7]),
            "+f"(d[12][0]), "+f"(d[12][1]), "+f"(d[12][2]), "+f"(d[12][3]), "+f"(d[12][4]), "+f"(d[12][5]), "+f"(d[12][6]), "+f"(d[12][7]),
            "+f"(d[13][0]), "+f"(d[13][1]), "+f"(d[13][2]), "+f"(d[13][3]), "+f"(d[13][4]), "+f"(d[13][5]), "+f"(d[13][6]), "+f"(d[13][7]),
            "+f"(d[14][0]), "+f"(d[14][1]), "+f"(d[14][2]), "+f"(d[14][3]), "+f"(d[14][4]), "+f"(d[14][5]), "+f"(d[14][6]), "+f"(d[14][7]),
            "+f"(d[15][0]), "+f"(d[15][1]), "+f"(d[15][2]), "+f"(d[15][3]), "+f"(d[15][4]), "+f"(d[15][5]), "+f"(d[15][6]), "+f"(d[15][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ void wgmma128(float d[8][8], bf16* sA, bf16* sB) {
    uint64_t desc_a = make_smem_desc(&sA[0]);
    uint64_t desc_b = make_smem_desc(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
        " %64,"
        " %65,"
        " %66,    %67,  %68,  %69,  %70;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]),
            "+f"(d[0][6]), "+f"(d[0][7]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
            "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]), "+f"(d[2][0]), "+f"(d[2][1]),
            "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
            "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]),
            "+f"(d[3][6]), "+f"(d[3][7]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
            "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]), "+f"(d[5][0]), "+f"(d[5][1]),
            "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
            "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[6][4]), "+f"(d[6][5]),
            "+f"(d[6][6]), "+f"(d[6][7]), "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
            "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ __forceinline__ void wgmma192(float d[12][8], bf16* sA, bf16* sB) {
    uint64_t desc_a = make_smem_desc(&sA[0]);
    uint64_t desc_b = make_smem_desc(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n192k16.f32.bf16.bf16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95},  "
        " %96,"
        " %97,"
        " %98,    %99,  %100,  %101,  %102;\n"
        "}\n"
        :   "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
            "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
            "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
            "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7]),
            "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]),
            "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
            "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[6][4]), "+f"(d[6][5]), "+f"(d[6][6]), "+f"(d[6][7]),
            "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7]),
            "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]), "+f"(d[8][4]), "+f"(d[8][5]), "+f"(d[8][6]), "+f"(d[8][7]),
            "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]), "+f"(d[9][4]), "+f"(d[9][5]), "+f"(d[9][6]), "+f"(d[9][7]),
            "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]), "+f"(d[10][3]), "+f"(d[10][4]), "+f"(d[10][5]), "+f"(d[10][6]), "+f"(d[10][7]),
            "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]), "+f"(d[11][3]), "+f"(d[11][4]), "+f"(d[11][5]), "+f"(d[11][6]), "+f"(d[11][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
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

template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ void wgmma32(float d[2][8], bf16* sA, bf16* sB) {
    uint64_t desc_a = make_smem_desc(&sA[0]);
    uint64_t desc_b = make_smem_desc(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15},  "
        " %16,"
        " %17,"
        " %18, %19, %20, %21, %22;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]),
            "+f"(d[0][6]), "+f"(d[0][7]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
            "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ void wgmma16(float d[1][8], bf16* sA, bf16* sB) {
    uint64_t desc_a = make_smem_desc(&sA[0]);
    uint64_t desc_b = make_smem_desc(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7},   "
        " %8,"
        " %9,"
        " %10, %11, %12, %13, %14;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]),
            "+f"(d[0][6]), "+f"(d[0][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

template<int WGMMA_N, int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ inline void wgmma(float d[WGMMA_N/16][8], bf16* sA, bf16* sB) {
    static_assert(WGMMA_N == 32 || WGMMA_N == 64 || WGMMA_N == 128 || WGMMA_N == 192 || WGMMA_N == 256);
    if  constexpr (WGMMA_N == 256)
        wgmma256<1, 1, 1, 0, 0>(d, sA, sB);
    if  constexpr (WGMMA_N == 192)
        wgmma192<1, 1, 1, 0, 0>(d, sA, sB);
    if  constexpr (WGMMA_N == 128)
        wgmma128<1, 1, 1, 0, 0>(d, sA, sB);
    if constexpr (WGMMA_N == 64)
        wgmma64<1, 1, 1, 0, 0>(d, sA, sB);
    if constexpr (WGMMA_N == 32)
        wgmma32<1, 1, 1, 0, 0>(d, sA, sB);
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

template <int BM, int BN, int BK, int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
matmul_e02_tc4_wg_tiling(const CUtensorMap *tensorMapA, const CUtensorMap *tensorMapB, bf16 *C, int M, int N, int K) {
    constexpr int WGMMA_M = 64, WGMMA_K = 16, WGMMA_N = BN;
    constexpr int B_WG_M = BM / (NUM_THREADS / 128);
    // shared memory for A & B
    __shared__ alignas(128) bf16 sA[BM*BK];
    __shared__ alignas(128) bf16 sB[BN*BK];

    // SMEM barriers for A & B
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier barA, barB;

    // accumulator, note C is column-major
    // each 128-thread warpgroup covers B_WG_M rows so it runs B_WG_M / WGMMA_M iterations
    // then the same (WGMMA_N / 16) x 8 organization
    float d[B_WG_M / WGMMA_M][WGMMA_N / 16][8];
    static_assert(sizeof(d) * NUM_THREADS == BM * BN * sizeof(float));
    memset(d, 0, sizeof(d));

    // the number of sliding blocks in the K dimension
    const int num_blocks_k = K / BK;
    // the position of this block in the result matrix
    const int num_block_n = blockIdx.x % (N / BN);
    const int num_block_m = blockIdx.x / (N / BN);

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
    // warpgroup index in case NUM_THREADS > 128
    int wg_idx = threadIdx.x / 128;

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
            tokenA = cuda::device::barrier_arrive_tx(barA, 1, BM * BK * sizeof(bf16));
            // note B is column-major, so the block window also slides right
            // to (block_k_iter * BK, num_block_n * BN)
            cde::cp_async_bulk_tensor_2d_global_to_shared(&sB[0], tensorMapB, block_k_iter * BK, num_block_n * BN,
                                                          barB);
            tokenB = cuda::device::barrier_arrive_tx(barB, 1, BN * BK * sizeof(bf16));
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
        #pragma unroll
        for (int m_it = 0; m_it < B_WG_M / WGMMA_M; ++m_it) {
            // shift by my warpgroup assignment plus the iteration index
            bf16* wgmma_sA = sA + BK * (m_it + wg_idx * B_WG_M / WGMMA_M) * WGMMA_M;
            #pragma unroll
            for (int k_it = 0; k_it < BK / WGMMA_K; ++k_it) {
                // shift As & Bs on K dimension, still same set of d[] along K dimension
                wgmma<WGMMA_N, 1, 1, 1, 0, 0>(d[m_it], &wgmma_sA[k_it * WGMMA_K], &sB[k_it * WGMMA_K]);
            }
        }
        // commit all prior wgmma.mma_async operations into a wgmma-group.
        warpgroup_commit_batch();
        warpgroup_wait<0>();
    }

    // Store
    {
        int tid = threadIdx.x % 128;
        int lane = tid & (WARPSIZE - 1);
        int warp = tid / WARPSIZE;
        // C matrix is column-major, but row still refers to M dimension
        // each wrap covers 16 rows in the 64x64 C submatrix
        // row index below covers 0-7 of my allocated rows, but it also writes row + 8
        uint32_t row = warp * 16 + lane / 4;
        bf16 *block_C = C + num_block_n * BN * M + num_block_m * BM;

        #pragma unroll
        for (uint32_t m_it = 0; m_it < B_WG_M/WGMMA_M; ++m_it) {
            int yo = m_it*WGMMA_M + wg_idx*B_WG_M;
            #pragma unroll
            for (uint32_t w = 0; w < WGMMA_N/16; ++w) {
                int col = 16*w + 2*(tid % 4);
                #define IDX(i, j) ((j)*M + ((i) + yo))

                block_C[IDX(row, col)] = d[m_it][w][0];
                block_C[IDX(row, col+1)] = d[m_it][w][1];
                block_C[IDX(row+8, col)] = d[m_it][w][2];
                block_C[IDX(row+8, col+1)] = d[m_it][w][3];
                block_C[IDX(row, col+8)] = d[m_it][w][4];
                block_C[IDX(row, col+9)] = d[m_it][w][5];
                block_C[IDX(row+8, col+8)] = d[m_it][w][6];
                block_C[IDX(row+8, col+9)] = d[m_it][w][7];
                
                #undef IDX
            }
        }
    }
}

template <int st_rows, int st_cols>
__host__ static inline CUtensorMap *allocate_and_create_tensor_map(bf16 *src, int blocks_height, int blocks_width) {
    CUtensorMap *tma_map_d;
    cudaMalloc(&tma_map_d, sizeof(CUtensorMap));
    CUtensorMap tma_map_host;
    create_tensor_map<st_rows, st_cols>(&tma_map_host, src, blocks_height, blocks_width);
    cudaMemcpy(tma_map_d, &tma_map_host, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    return tma_map_d;
}

} // namespace e02

CUtensorMap *e02_d_tma_map_A = 0;
CUtensorMap *e02_d_tma_map_B = 0;

void runMatmulE02Tc4WgTiling(MatmulBuffers &buffers) {
    if (!e02_d_tma_map_A) {
        e02_d_tma_map_A = e02::allocate_and_create_tensor_map<E02_BM, E02_BK>(buffers.dA_bf16, buffers.M / E02_BM, buffers.K / E02_BK);
        e02_d_tma_map_B = e02::allocate_and_create_tensor_map<E02_BN, E02_BK>(buffers.dB_bf16_t, buffers.N / E02_BN, buffers.K / E02_BK);
    }
    dim3 blockDim = dim3(E02_BLOCK_SIZE, 1);
    dim3 gridDim = dim3((buffers.M / E02_BM) * (buffers.N / E02_BN), 1);

    e02::matmul_e02_tc4_wg_tiling<E02_BM, E02_BN, E02_BK, E02_BLOCK_SIZE>
        <<<gridDim, blockDim>>>(e02_d_tma_map_A, e02_d_tma_map_B, buffers.dC_bf16, buffers.M, buffers.N, buffers.K);
    checkCuda(cudaGetLastError(), "launch matmul_e02_tc4_wg_tiling");
    buffers.num_iters += 1;
}
