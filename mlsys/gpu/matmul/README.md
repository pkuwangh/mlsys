# MatMul

## Torch

```bash
nsys profile ./torch_matmul.py
```

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
  - `xmma_gemm`: uses `XMMA` path, i.e. Tensor Core MMA familya
  - `f32f32_f32`: A: FP32, B: FP32, accumulate/output: FP32
  - `nn_n`: op(A)=N, op(B)=N (`N`: no transpose, vs. `T`), last `n` is layout tag for C, i.e. row-major layout
  - `tilesize128x128x32`: CTA/block tile `M x N x K`
  - `warpgroupsize1x1x1`: no multi-warp-group cooperations per CTA tile
  - `execute_segment_k_on_kernel`: K-dimension is segmented inside the kernel
  - `5x`: unroll / pipeline depth
