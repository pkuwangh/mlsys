#pragma once

#include <iostream>
#include <cuda.h>
#include <cuda/barrier>

#include "matmul_utils.cuh"

#define E04_BM 128
#define E04_BN 256
#define E04_BK 64
#define E04_BLOCK_SIZE 128*3
#define E04_QSIZE 3

namespace e04 {

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

// compared e03 version
// - increase block tile size to 128x256 from 128x128
// - increase tc size to m64n256k16 from m64n128k16
// - use 2 consumer warpgroups, each handles 64 rows
// - BK is still 64, so 4 wgmma calls in a single wgmma group each time

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
__device__ __forceinline__ void wgmma256(float d[16][8], bf16* sA, bf16* sB) {
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

template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ __forceinline__ void wgmma128(float d[8][8], bf16* sA, bf16* sB) {
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

template <int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ __forceinline__ void wgmma64(float d[4][8], bf16 *sA, bf16 *sB) {
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
__device__ __forceinline__ void wgmma32(float d[2][8], bf16* sA, bf16* sB) {
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
__device__ __forceinline__ void wgmma16(float d[1][8], bf16* sA, bf16* sB) {
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
__device__ __forceinline__ void wgmma(float d[WGMMA_N/16][8], bf16* sA, bf16* sB) {
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

template <int BM, int BN, int BK, int QSIZE>
struct SMem {
    alignas(128) bf16 A[BM*BK*QSIZE];
    alignas(128) bf16 B[BK*BN*QSIZE];
};

template <uint32_t RegCount>
__device__ void warpgroup_reg_alloc() {
    // warpgroup-scoped instruction to raise the max # registers current warpgroup can use
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

template <uint32_t RegCount>
__device__ void warpgroup_reg_dealloc() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

template <int BM, int BN, int BK, int NUM_THREADS, int QSIZE>
__global__ void __launch_bounds__(NUM_THREADS)
matmul_e04_tc4_multi_consumer(const CUtensorMap *tensorMapA, const CUtensorMap *tensorMapB, bf16 *C, int M, int N, int K) {
    constexpr int WGMMA_M = 64, WGMMA_K = 16, WGMMA_N = BN;
    constexpr int NUM_CONSUMERS = (NUM_THREADS / 128) - 1;
    // each consumer warpgroup covers B_WG_M rows total
    // that's 64 rows which wgmma can handle in one iteration
    constexpr int B_WG_M = BM / NUM_CONSUMERS;
    // shared memory for A & B
    extern __shared__ __align__(128) uint8_t smem[];
    SMem<BM, BN, BK, QSIZE> &s = *reinterpret_cast<SMem<BM, BN, BK, QSIZE>*>(smem);
    bf16 *sA = s.A;
    bf16 *sB = s.B;

    // full/empty barriers for the pipeline
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier full[QSIZE], empty[QSIZE];

    // the number of sliding blocks in the K dimension
    const int num_blocks_k = K / BK;
    // the position of this block in the result matrix
    const int num_block_n = blockIdx.x % (N / BN);
    const int num_block_m = blockIdx.x / (N / BN);

    // warpgroup index and thread index within my warpgroup
    const int cta_scope_wg_idx = threadIdx.x / 128;
    const int tid = threadIdx.x % 128;

    if (threadIdx.x == 0) {
        for (int i = 0 ; i < QSIZE; ++i) {
            // all threads from consumer warpgroup plus 1 as producer
            init(&full[i], NUM_CONSUMERS * 128 + 1);
            init(&empty[i], NUM_CONSUMERS * 128 + 1);
        }
        // make initialized barrier visible to async proxy (TMA)
        // Hopper's proxy memory model orders visibility b/w the async proxy (TMA) and
        // the generic proxy (normal thread ld/st) at CTA scope.
        cde::fence_proxy_async_shared_cta();
    }
    // make sure barriers are visible to all threads
    __syncthreads();

    if (cta_scope_wg_idx == 0) {
        constexpr int num_regs = (NUM_CONSUMERS <= 2 ? 24 : 32);
        // producer does NOT need so many registers
        warpgroup_reg_dealloc<num_regs>();
        // producer
        if (tid == 0) {
            // only have one thread issues TMA commands
            int q_idx = 0;
            for (int block_k_iter = 0; block_k_iter < num_blocks_k; ++block_k_iter, ++q_idx) {
                if (q_idx == QSIZE) q_idx = 0;
                // contributes to the arrival, empty[q_idx].arrive()
                // gets back a token and wait until the barrier reaches the phase associated the token
                empty[q_idx].wait(empty[q_idx].arrive());
                // launch TMA bulk async copy
                cde::cp_async_bulk_tensor_2d_global_to_shared(&sA[q_idx * BK * BM], tensorMapA, block_k_iter * BK,
                                                              num_block_m * BM, full[q_idx]);
                cde::cp_async_bulk_tensor_2d_global_to_shared(&sB[q_idx * BK * BN], tensorMapB, block_k_iter * BK,
                                                              num_block_n * BN, full[q_idx]);
                // update barrier with # bytes expected
                // contributes to my arrival for the full state of current slot
                barrier::arrival_token _ =
                    cuda::device::barrier_arrive_tx(full[q_idx], 1, (BK * BN + BK * BM) * sizeof(bf16));
            }
        }
    } else {
        // consumers
        constexpr int num_regs = (NUM_CONSUMERS == 1 ? 256 : (NUM_CONSUMERS == 2 ? 240 : 160));
        warpgroup_reg_alloc<num_regs>();
        // make the consumer wg_idx 0-based
        const int wg_idx = cta_scope_wg_idx - 1;
        // first, mark all slots empty
        for (int i = 0; i < QSIZE; ++i) {
            barrier::arrival_token _ = empty[i].arrive();
        }
        // each consumer warpgroup handles B_WG_M rows instead of BM rows
        float d[B_WG_M / WGMMA_M][WGMMA_N / 16][8];
        memset(d, 0, sizeof(d));
        int q_idx = 0;
        for (int block_k_iter = 0; block_k_iter < num_blocks_k; ++block_k_iter, ++q_idx) {
            if (q_idx == QSIZE) q_idx = 0;
            // arrive behave like a sync and block on the producer's arrival
            full[q_idx].wait(full[q_idx].arrive());
            // wgmma calls
            warpgroup_arrive();
            #pragma unroll
            for (int m_it = 0; m_it < B_WG_M / WGMMA_M; ++m_it) {
                // shift by
                // 1. q_idx in the smem block, which is now larger to include all QSIZE block tiles
                // 2. interaction index when each warpgroup handles multiple warpgroup tiles
                // 3. consumer warpgroup index that handles B_WG_M rows
                bf16* wgmma_sA = sA + BK * (q_idx * BM + m_it * WGMMA_M + wg_idx * B_WG_M);
                bf16* wgmma_sB = sB + BK * (q_idx * BN);
                #pragma unroll
                for (int k_it = 0; k_it < BK / WGMMA_K; ++k_it) {
                    wgmma<WGMMA_N, 1, 1, 1, 0, 0>(d[m_it], &wgmma_sA[k_it * WGMMA_K], &wgmma_sB[k_it * WGMMA_K]);
                }
            }
            warpgroup_commit_batch();
            warpgroup_wait<0>();
            // mark this slot empty by contributing to my arrival here
            barrier::arrival_token _ = empty[q_idx].arrive();
        }

        // store back to C
        int lane = tid % 32;
        int warp = tid / 32;
        int row = warp * 16 + lane / 4;
        bf16 *block_C = C + num_block_n * BN * M + num_block_m * BM;

        #pragma unroll
        for (int m_it = 0; m_it < B_WG_M/WGMMA_M; ++m_it) {
            int yo = m_it*WGMMA_M + wg_idx*B_WG_M;
            #pragma unroll
            for (int w = 0; w < WGMMA_N/16; ++w) {
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

} // namespace e04

CUtensorMap *e04_d_tma_map_A = 0;
CUtensorMap *e04_d_tma_map_B = 0;

void runMatmulE04Tc4MultiConsumer(MatmulBuffers &buffers) {
    if (!e04_d_tma_map_A) {
        e04_d_tma_map_A = e04::allocate_and_create_tensor_map<E04_BM, E04_BK>(buffers.dA_bf16, buffers.M / E04_BM,
                                                                              buffers.K / E04_BK);
        e04_d_tma_map_B = e04::allocate_and_create_tensor_map<E04_BN, E04_BK>(buffers.dB_bf16_t, buffers.N / E04_BN,
                                                                              buffers.K / E04_BK);
    }
    dim3 blockDim = dim3(E04_BLOCK_SIZE, 1);
    dim3 gridDim = dim3((buffers.M / E04_BM) * (buffers.N / E04_BN), 1);

    auto* kernel = e04::matmul_e04_tc4_multi_consumer<E04_BM, E04_BN, E04_BK, E04_BLOCK_SIZE, E04_QSIZE>;
    size_t shmem_size = sizeof(e04::SMem<E04_BM, E04_BN, E04_BK, E04_QSIZE>);
    checkCuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size),
              "cudaFuncSetAttribute");

    kernel<<<gridDim, blockDim, shmem_size>>>(e04_d_tma_map_A, e04_d_tma_map_B, buffers.dC_bf16, buffers.M, buffers.N,
                                              buffers.K);
    checkCuda(cudaGetLastError(), "launch matmul_e04_tc4_multi_consumer");
    buffers.num_iters += 1;
}
