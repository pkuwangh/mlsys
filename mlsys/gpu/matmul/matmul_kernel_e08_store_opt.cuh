#pragma once

#include <iostream>
#include <cuda.h>
#include <cuda/barrier>

#include "matmul_utils.cuh"

#define E08_BM 128
#define E08_BN 256
#define E08_BK 64
#define E08_BLOCK_SIZE 128*3
#define E08_QSIZE 3
#define E08_CLUSTER_M 2
#define E08_CLUSTER_N 1
#define E08_NUM_SM 128

namespace e08 {

// using barrier = cuda::barrier<cuda::thread_scope_block>;
// namespace cde = cuda::device::experimental;

// optimizing the store operation
// - bypass L1/L2 using __stwt
// - organize the order

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
        wgmma256<ScaleD, ScaleA, ScaleB, TransA, TransB>(d, sA, sB);
    if  constexpr (WGMMA_N == 192)
        wgmma192<ScaleD, ScaleA, ScaleB, TransA, TransB>(d, sA, sB);
    if  constexpr (WGMMA_N == 128)
        wgmma128<ScaleD, ScaleA, ScaleB, TransA, TransB>(d, sA, sB);
    if constexpr (WGMMA_N == 64)
        wgmma64<ScaleD, ScaleA, ScaleB, TransA, TransB>(d, sA, sB);
    if constexpr (WGMMA_N == 32)
        wgmma32<ScaleD, ScaleA, ScaleB, TransA, TransB>(d, sA, sB);
}

template <int BlockMajorSize, int BlockMinorSize>
__host__ static inline CUtensorMap create_tensor_map(bf16* gmem_ptr, int global_height, int global_width) {
    CUtensorMap tma_map;
    void* gmem_address = (void*)gmem_ptr;
    static_assert(BlockMinorSize >= 64);
    assert(global_width % 64 == 0);
    // previously encode the GMEM tensor as a straightforward 2D matrix, i.e. rank=2, width x heigh
    // here encode as 64 x height x (width/64)
    // so dim0 is 64 contiguous elements, dim2 means how many 64-stride chunks across width
    uint64_t gmem_prob_shape[5] = {64, (uint64_t)global_height, (uint64_t)global_width/64, 1, 1};
    uint64_t gmem_prob_stride[5] = {sizeof(bf16) * global_width, 64*sizeof(bf16), 0, 0, 0};
    uint32_t smem_box_shape[5] = {64, uint32_t(BlockMajorSize), uint32_t(BlockMinorSize/64), 1, 1};
    uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};

    // form a tensor map
    // encode all the metadata needed to transfer chunks of GMEM to SMEM
    // dtype: bf16
    // rank: 3 (strided)
    // pointer: gemm_address/gmem_ptr
    // shape: fastest stride dimension first, (64, height, width/64)
    // stride: (width * sizeof(bf16), 64 * sizeof(bf16), 0)
    // smem shape: (64, BlockMajorSize, BlockMinorSize/64)
    // swizzle: 128B pattern
    CUresult result = cuTensorMapEncodeTiled(&tma_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, gmem_address,
                                             gmem_prob_shape, gmem_prob_stride, smem_box_shape, smem_box_stride,
                                             CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                             CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    assert(result == CUDA_SUCCESS);
    return tma_map;
}

template <int BM, int BN, int BK, int QSIZE>
struct SMem {
    alignas(128) bf16 A[BM*BK*QSIZE];
    alignas(128) bf16 B[BK*BN*QSIZE];
};

template<int VERSION, int NUM_SM, int BM, int BN, int TM, int TN>
struct Schedule;

template<int NUM_SM, int BM, int BN, int TM, int TN>
struct Schedule<1, NUM_SM, BM, BN, TM, TN> {
    // version 1: clustered traversal to be L2 friendly
    int cta_idx;        // persistent worker id
    int it;             // iteration index for grid-stride
    int total_blocks_m;
    int total_blocks_n;

    __device__ __forceinline__ Schedule(int M, int N, int _cta_idx) {
        cta_idx = _cta_idx;
        it = 0;
        total_blocks_m = M / BM;
        total_blocks_n = N / BN;
        assert(total_blocks_m % TM == 0 && total_blocks_n % TN == 0);
    }

    __device__ __forceinline__ bool next(int &block_m, int& block_n) {
        // num is the i-th block in a 1-D list of all blocks
        // so a CTA will take cta_idx, cta_idx+NUM_SM, cta_idx+2*NUM_SM ... blocks
        int num = it * NUM_SM + cta_idx;
        if (num >= total_blocks_m * total_blocks_n) {
            return false;
        }

        // neighboring CTAs are expected to pick up blocks within a TM x TN micro-tile
        // calculate which micro-tile and which block within the micro-tile
        int cur_tile = num / (TM * TN);
        int cur_tile_pos = num % (TM * TN);
        // map micro-tile index back to (m, n) indices in 2-D view
        block_m = TM * (cur_tile / (total_blocks_n / TN));
        block_n = TN * (cur_tile % (total_blocks_n / TN));
        // add the intra-micro-tile offset
        block_m += cur_tile_pos / TN;
        block_n += cur_tile_pos % TN;
        ++it;
        return true;
    }
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

// ptx impl of init barrier, update expected bytes, async-load, wait, mark arrive
__device__ static __forceinline__ void init_barrier(uint64_t *bar, int thread_count, int transaction_count) {
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar_ptr), "r"(thread_count + transaction_count));
}

__device__ static __forceinline__ void expect_bytes(uint64_t *bar, uint32_t bytes) {
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    // removed .release.cta
    // no longer enforcing release-ordering semantics
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar_ptr), "r"(bytes));
}

__device__ static inline void load_async(bf16 *dst, void const *const src_tma_map, uint64_t *bar, int global_col_idx,
                                         int global_row_idx) {
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(src_tma_map);
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));

    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                 " [%0], [%1, {%3, %4, %5}], [%2];"
                 :
                 : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr), "n"(0), "r"(global_row_idx), "r"(global_col_idx / 64)
                 : "memory");
}

__device__ static __forceinline__ void wait(uint64_t *bar, int kPhaseBit) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("{\n"
                 ".reg .pred                P1;\n"
                 "LAB_WAIT:\n"
                 "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1;\n"
                 "@P1                       bra.uni DONE;\n"
                 "bra.uni                   LAB_WAIT;\n"
                 "DONE:\n"
                 "}\n" ::"r"(mbar_ptr),
                 "r"(kPhaseBit));
}

__device__ static __forceinline__ void arrive(uint64_t* bar, uint32_t count=1) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar)); 
    asm volatile (
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n"
        :
        : "r"(mbar_ptr), "r"(count)
        : "memory"
    );
}

// thread-block cluster orchestration
__device__ static __forceinline__ void wait_cluster(uint64_t *bar, int kPhaseBit) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("{\n"
                 ".reg .pred                P1;\n"
                 "LAB_WAIT:\n"
                 "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], %1;\n"
                 "@P1                       bra.uni DONE;\n"
                 "bra.uni                   LAB_WAIT;\n"
                 "DONE:\n"
                 "}\n" ::"r"(mbar_ptr),
                 "r"(kPhaseBit));
}

__device__ static inline void load_async_multicast(bf16 *dst, void const *const src_tma_map, uint64_t *bar,
                                                   int global_col_idx, int global_row_idx, uint16_t cluster_mask) {
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(src_tma_map);
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));

    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
                 " [%0], [%1, {%3, %4, %5}], [%2], %6;"
                 :
                 : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr), "n"(0), "r"(global_row_idx), "r"(global_col_idx / 64),
                   "h"(cluster_mask)
                 : "memory");
}

__device__ void arrive_cluster(uint64_t *bar, uint32_t cta_id, uint32_t count = 1) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("{\n\t"
                 ".reg .b32 remAddr32;\n\t"
                 "mapa.shared::cluster.u32  remAddr32, %0, %1;\n\t"
                 "mbarrier.arrive.shared::cluster.b64  _, [remAddr32], %2;\n\t"
                 "}"
                 :
                 : "r"(smem_addr), "r"(cta_id), "r"(count));
}

template <int BM, int BN, int BK, int NUM_THREADS, int QSIZE, int NUM_SM, int TM, int TN, int CLUSTER_M, int CLUSTER_N>
__global__ void __launch_bounds__(NUM_THREADS) __cluster_dims__(CLUSTER_M * CLUSTER_N, 1, 1)
matmul_e08_store_opt(const __grid_constant__ CUtensorMap tensorMapA, const __grid_constant__ CUtensorMap tensorMapB, bf16 *C, int M, int N, int K) {
    constexpr int WGMMA_M = 64, WGMMA_K = 16, WGMMA_N = BN;
    constexpr int NUM_CONSUMERS = (NUM_THREADS / 128) - 1;
    // each consumer warpgroup covers B_WG_M rows total
    // that's 64 rows which wgmma can handle in one iteration
    constexpr int B_WG_M = BM / NUM_CONSUMERS;
    // thread-block cluster
    constexpr int CTAS_PER_CLUSTER = CLUSTER_M * CLUSTER_N;
    assert((M / BM) % CLUSTER_M == 0 && (N / BN) % CLUSTER_N == 0);
    // shared memory for A & B
    extern __shared__ __align__(128) uint8_t smem[];
    SMem<BM, BN, BK, QSIZE> &s = *reinterpret_cast<SMem<BM, BN, BK, QSIZE>*>(smem);
    bf16 *sA = s.A;
    bf16 *sB = s.B;

    // full/empty barriers for the pipeline
    __shared__ __align__(8) uint64_t full[QSIZE], empty[QSIZE];
    // get this CTA's cluster ID
    uint32_t cluster_id;
    asm volatile("mov.u32 %0, %clusterid.x;\n" : "=r"(cluster_id) :);

    // the number of sliding blocks in the K dimension
    const int num_blocks_k = K / BK;

    // warpgroup index and thread index within my warpgroup
    const int cta_scope_wg_idx = threadIdx.x / 128;
    const int tid = threadIdx.x % 128;

    if (threadIdx.x == 0) {
        for (int i = 0 ; i < QSIZE; ++i) {
            // thread_count, transaction_count
            init_barrier(&full[i], 0, 1);
            // scope expand to the cluster
            init_barrier(&empty[i], 0, NUM_CONSUMERS * CTAS_PER_CLUSTER);
        }
    }
    // need the cluster-wide barrier to sync; __syncthreads() is not enough
    asm volatile("barrier.cluster.arrive;\n" : :);
    asm volatile("barrier.cluster.wait;\n" : :);

    // now cluster is the unit to pick up work
    Schedule<1, NUM_SM / CTAS_PER_CLUSTER, BM * CLUSTER_M, BN * CLUSTER_N, TM / CLUSTER_M, TN / CLUSTER_N> schedule(
        M, N, cluster_id);

    // get the CTA's rank within the cluster
    uint32_t rank;
    asm volatile("mov.u32 %0, %cluster_ctarank;\n" : "=r"(rank) :);
    uint32_t rank_m = rank / CLUSTER_N;
    uint32_t rank_n = rank % CLUSTER_N;

    if (cta_scope_wg_idx == 0) {
        constexpr int num_regs = (NUM_CONSUMERS <= 2 ? 24 : 32);
        // producer does NOT need so many registers
        warpgroup_reg_dealloc<num_regs>();
        // producer
        if (tid == 0) {
            // only have one thread issues TMA commands
            int phase = 0;
            int q_idx = 0;
            uint32_t col_mask = 0;
            for (int i = 0; i < CLUSTER_M; ++i) {
                col_mask |= (1 << (i * CLUSTER_N));
            }
            int num_block_m = 0, num_block_n = 0;
            while (schedule.next(num_block_m, num_block_n)) {
                // now a block is shared across the cluster, so get my subblock
                num_block_n = num_block_n * CLUSTER_N + rank_n;
                num_block_m = num_block_m * CLUSTER_M + rank_m;

                for (int block_k_iter = 0; block_k_iter < num_blocks_k; ++block_k_iter, ++q_idx) {
                    if (q_idx == QSIZE) {
                        q_idx = 0;
                        // mbarrier.try_wait.parity.acquire.cta.shared::cta.b64
                        // the barrier flip the phase bit when it completes a full cycle and is reused
                        phase ^= 1;
                    }
                    wait(&empty[q_idx], phase);
                    expect_bytes(&full[q_idx], (BK * BN + BK * BM) * sizeof(bf16));
                    // load A
                    if constexpr (CLUSTER_N > 1) {
                        uint32_t mask = ((1 << CLUSTER_N) - 1) << (rank_m * CLUSTER_N);
                        if (rank_n == 0) {
                            // only the first CTA in the cluster loads A and multicast
                            load_async_multicast(&sA[q_idx * BK * BM], &tensorMapA, &full[q_idx], block_k_iter * BK,
                                                 num_block_m * BM, mask);
                        }
                    } else {
                        load_async(&sA[q_idx * BK * BM], &tensorMapA, &full[q_idx], block_k_iter * BK, num_block_m * BM);
                    }
                    // load B
                    if constexpr (CLUSTER_M > 1) {
                        if (rank_m == 0) {
                            load_async_multicast(&sB[q_idx * BK * BN], &tensorMapB, &full[q_idx], block_k_iter * BK,
                                                 num_block_n * BN, col_mask << rank_n);
                        }
                    } else {
                        load_async(&sB[q_idx * BK * BN], &tensorMapB, &full[q_idx], block_k_iter * BK, num_block_n * BN);
                    }
                }
            }
        }
    } else {
        // consumers
        constexpr int num_regs = (NUM_CONSUMERS == 1 ? 256 : (NUM_CONSUMERS == 2 ? 240 : 160));
        warpgroup_reg_alloc<num_regs>();
        // each consumer warpgroup handles B_WG_M rows instead of BM rows
        float d[B_WG_M / WGMMA_M][WGMMA_N / 16][8];
        // make the consumer wg_idx 0-based
        const int wg_idx = cta_scope_wg_idx - 1;
        // first, mark all slots empty
        for (int i = 0; i < QSIZE; ++i) {
            if (tid < CTAS_PER_CLUSTER) {
                // pass tid as cta_id to arrive_cluster
                // the cluster-wide barrier is stored in each CTA's shared memory
                // so wach warpgroup will write each CTA's shared memory
                arrive_cluster(&empty[i], tid);
            }
        }
        int phase = 0;
        int q_idx = 0;
        int lane = tid % 32, warp = tid / 32, row = warp * 16 + lane / 4;
        int num_block_m = 0, num_block_n = 0;
        while (schedule.next(num_block_m, num_block_n)) {
            num_block_n = num_block_n * CLUSTER_N + rank_n;
            num_block_m = num_block_m * CLUSTER_M + rank_m;
            memset(d, 0, sizeof(d));
            for (int block_k_iter = 0; block_k_iter < num_blocks_k; ++block_k_iter, ++q_idx) {
                if (q_idx == QSIZE) {
                    q_idx = 0;
                    phase ^= 1;
                }
                wait(&full[q_idx], phase);
                warpgroup_arrive();
                #pragma unroll
                for (int m_it = 0; m_it < B_WG_M / WGMMA_M; ++m_it) {
                    // smem_box_shape[5] = {64, BlockMajorSize, BlockMinorSize / 64, 1, 1};
                    bf16* wgmma_sA = sA + BK * q_idx * BM + 64 * (m_it * WGMMA_M + wg_idx * B_WG_M);
                    bf16* wgmma_sB = sB + BK * q_idx * BN;
                    #pragma unroll
                    for (int bk = 0; bk < BK; bk += 64) {
                        #pragma unroll
                        for (int k_it = 0; k_it < 64 / WGMMA_K; ++k_it) {
                            wgmma<WGMMA_N, 1, 1, 1, 0, 0>(d[m_it], &wgmma_sA[k_it * WGMMA_K],
                                                          &wgmma_sB[k_it * WGMMA_K]);
                        }
                        wgmma_sA += 64 * BM;
                        wgmma_sB += 64 * BN;
                    }
                }
                warpgroup_commit_batch();
                warpgroup_wait<0>();
                if (tid < CTAS_PER_CLUSTER) {
                    arrive_cluster(&empty[q_idx], tid);
                }
            }

            bf16 *block_C = C + num_block_n * BN * M + num_block_m * BM;

            #pragma unroll
            for (int m_it = 0; m_it < B_WG_M / WGMMA_M; ++m_it) {
                int yo = m_it * WGMMA_M + wg_idx * B_WG_M;
                if (row + 8 + yo + num_block_m * BM >= M)
                    continue;
                #pragma unroll
                for (int w = 0; w < WGMMA_N; w += 16) {
                    if (w < N - num_block_n * BN) {
                        int col = w + 2*(tid % 4);
                        #define IDX(i, j) ((j)*M + ((i) + yo))
                        #define ST(i, j, v) __stwt(&block_C[IDX(i, j)], v);

                        ST(row+8, col, d[m_it][w/16][2]);
                        ST(row, col, d[m_it][w/16][0]);
                    
                        ST(row+8, col+1, d[m_it][w/16][3]);
                        ST(row, col+1, d[m_it][w/16][1]);
                    
                        ST(row+8, col+8, d[m_it][w/16][6]);
                        ST(row, col+8, d[m_it][w/16][4]);
                    
                        ST(row+8, col+9, d[m_it][w/16][7]);
                        ST(row, col+9, d[m_it][w/16][5]);

                        #undef IDX
                        #undef ST
                    }
                }
            }
        }
    }
}

template <typename Kernel> void _run_kernel(MatmulBuffers &buffers, Kernel kernel, bool check_error) {
    CUtensorMap tma_map_A = create_tensor_map<E08_BM, E08_BK>(buffers.dA_bf16, buffers.M, buffers.K);
    CUtensorMap tma_map_B = create_tensor_map<E08_BN, E08_BK>(buffers.dB_bf16_t, buffers.N, buffers.K);

    size_t shmem_size = sizeof(SMem<E08_BM, E08_BN, E08_BK, E08_QSIZE>);
    checkCuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size),
              "cudaFuncSetAttribute");

    // NUM_SM does not have to match exactly the physical SMs on GPU
    kernel<<<E08_NUM_SM, E08_BLOCK_SIZE, shmem_size>>>(tma_map_A, tma_map_B, buffers.dC_bf16, buffers.M, buffers.N,
                                                   buffers.K);
    if (check_error) {
        checkCuda(cudaGetLastError(), "launch matmul_e07_cta_cluster");
    }
    buffers.num_iters += 1;
}

} // namespace e08

void runMatmulE08StoreOpt(MatmulBuffers &buffers) {
    constexpr int TM = 16;
    constexpr int TN = 8;

    auto *kernel = e08::matmul_e08_store_opt<E08_BM, E08_BN, E08_BK, E08_BLOCK_SIZE, E08_QSIZE, E08_NUM_SM, TM, TN,
                                             E08_CLUSTER_M, E08_CLUSTER_N>;
    e08::_run_kernel(buffers, kernel, true);
}
