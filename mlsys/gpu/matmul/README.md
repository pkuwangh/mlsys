# MatMul

## Torch

matmul kernels:
- `nvjet_hsh_256x128_64x4_1x2_h_bz_coopA_NNT`:
  - `nvjet`: cuBLASt "JET` codegen family
  - `256x128`: CTA/block tile MxN
  - `64x4`: per-warp / per-warp-group tile
  - `1x2`: warp-group clustering / pipe staging arrangement
  - `h`: half precision (FP16) on tensor core
  - `bz`: block-swizzle to reduce L2 set conflicts
  - `coopA`: cooperative staging for operand A
  - `NNT`: layout/operation flags
- `sm90_xmma_gemm_f32f32_f32_nn_n_tilesize128x128x32_warpgroupsize1x1x1_execute_segment_k_on_kernel__5x_cublas`
  - `xmma_gemm`: uses `XMMA` path, i.e. Tensor Core MMA family
  - `f32f32_f32`: A: FP32, B: FP32, accumulate/output: FP32
  - `nn_n`: op(A)=N, op(B)=N (`N`: no transpose, vs. `T`), last `n` is layout tag for C, i.e. row-major layout
  - `tilesize128x128x32`: CTA/block tile `M x N x K`
  - `warpgroupsize1x1x1`: no multi-warp-group cooperations per CTA tile
  - `execute_segment_k_on_kernel`: K-dimension is segmented inside the kernel
  - `5x`: unroll / pipeline depth

## CUDA

- Basic kernel
  - `--ptxas-options=-v` shows
    - 32 registers per thread, limit # blocks to 65536 / (1024 * 32) = 2
    - with max threads per SM of 2048, that limit is also 2 blocks.
- Shared memory
  - Use shared memory to reduce global memory access
- Thread tiling
  - Let each thread handle a small tile, so each thread block cover a larger sub-matrix to further reduce global memory access
  - cache elements in registers to reduce shared memory access and increase compute/mem-access ratio
- Float4 load/store
  - reduce load/store instructions
- Double buffering
  - overlap data loading and compute

## Profiling

```bash
# nsys profiling
nsys profile ./torch_matmul.py
sudo nsys profile --gpu-metrics-devices=all ./bin/matmul_main

# ncu profiling
sudo $CUDA_HOME/bin/ncu ./bin/matmul_main
```