// Isolate the peer-CTA TMA barrier diagnostic from the Wan kernels.
// PTX permits a peer-CTA mbarrier with shared::cluster and cta_group::2.
// Both modes must copy all 32 values exactly. With --check-tensor-ops yes,
// Compute Sanitizer 2026.3 rejects the peer mode but accepts local.
// Build on B300:
//   nvcc -std=c++17 -gencode arch=compute_103a,code=sm_103a \
//     tests/repro_tma_peer_barrier.cu -lcuda -o /tmp/repro_tma_peer_barrier
// Run without arguments for peer completion, or with "local" for the control.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>

#define CUDA(call)                                                        \
    do {                                                                  \
        auto error = (call);                                               \
        if (error != cudaSuccess) {                                       \
            fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(error)); \
            return 1;                                                     \
        }                                                                 \
    } while (0)

template <bool Peer>
__global__ void load_pair(const __grid_constant__ CUtensorMap map, float* output) {
    __shared__ __align__(128) float tile[16];
    __shared__ __align__(8) unsigned long long barrier;
    auto cluster = cooperative_groups::this_cluster();
    unsigned rank = cluster.block_rank();
    unsigned bar = __cvta_generic_to_shared(&barrier);
    unsigned dst = __cvta_generic_to_shared(tile);
    unsigned target;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(target) : "r"(bar), "r"(Peer ? 0u : rank));
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar));
        asm volatile("fence.mbarrier_init.release.cluster;");
        asm volatile("fence.proxy.async.shared::cta;");
    }
    cluster.sync();
    if (threadIdx.x == 0) {
        if (!Peer || rank == 0) {
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(Peer ? 128 : 64) : "memory");
        }
        int coordinate = rank * 16;
        asm volatile("cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2 [%0], [%1, {%2}], [%3];"
                     :: "r"(dst), "l"(&map), "r"(coordinate), "r"(target) : "memory");
        if (!Peer || rank == 0) {
            unsigned ready;
            do {
                asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0; selp.b32 %0, 1, 0, p; }"
                             : "=r"(ready) : "r"(bar) : "memory");
            } while (!ready);
        }
    }
    cluster.sync();
    if (threadIdx.x < 16) output[rank * 16 + threadIdx.x] = tile[threadIdx.x];
    cluster.sync();
}
int main(int argc, char**) {
    float input[32], actual[32];
    for (int i = 0; i < 32; ++i) input[i] = i + 1;
    float *x, *y;
    CUDA(cudaMalloc(&x, sizeof(input)));
    CUDA(cudaMalloc(&y, sizeof(actual)));
    CUDA(cudaMemcpy(x, input, sizeof(input), cudaMemcpyHostToDevice));
    alignas(64) CUtensorMap map;
    cuuint64_t dims[] = {32}, global_strides[] = {128};
    cuuint32_t box[] = {16}, strides[] = {1};
    auto status = cuTensorMapEncodeTiled(
        &map, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 1, x, dims, global_strides, box, strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (status != CUDA_SUCCESS) {
        fprintf(stderr, "tensor map error %d\n", status);
        return 1;
    }
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(2);
    cfg.blockDim = dim3(32);
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {2, 1, 1};
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
    bool peer = argc == 1;
    if (peer) {
        CUDA(cudaLaunchKernelEx(&cfg, load_pair<true>, map, y));
    } else {
        CUDA(cudaLaunchKernelEx(&cfg, load_pair<false>, map, y));
    }
    CUDA(cudaDeviceSynchronize());
    CUDA(cudaMemcpy(actual, y, sizeof(actual), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 32; ++i) {
        if (actual[i] != input[i]) {
            fprintf(stderr, "mismatch %d: %g != %g\n", i, actual[i], input[i]);
            return 1;
        }
    }
    CUDA(cudaFree(x));
    CUDA(cudaFree(y));
    printf("%s barrier: PASS\n", peer ? "Peer CTA" : "Local CTA");
}
