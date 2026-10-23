# Wan VAE kernels

This directory supports the Wan 2.1 VAE and the Wan 2.2 TI2V-5B VAE.
The shared benchmark scripts are `bench_wan_vae_encoder.py` and
`bench_wan_vae_decoder.py`.
Pass a Diffusers checkpoint with `--model`; omitting it still selects the
synthetic Wan 2.2 architecture.

See [BENCHMARKS.md](BENCHMARKS.md) for measured results and validation limitations.

## Runtime

`MegaWanVaeEncoder(vae)` and `MegaWanVaeDecoder(vae)` select their execution
path from the loaded `AutoencoderKLWan` configuration. Use BF16 or FP32 CUDA weights,
`.eval()`, and `torch.inference_mode()`.

BF16 remains the default. FP32 storage supports both framework convolution policies:

| Storage | `torch.backends.cudnn.allow_tf32` | Convolution and post-ops |
|---|---|---|
| BF16 | Either | BF16 Tensor Core kernels and fusion |
| FP32 | `False` | Strict cuDNN convolution followed by separate fused post-ops |
| FP32 | `True` | Ordinary single-product TF32 kernels, with supported integrated fusion |

Configure the framework policy **before constructing** a full model and keep it
stable throughout inference and CUDA graph replay. Both model constructors accept
an optional `allow_tf32` argument that checks the framework setting. Model entry
points reject subsequent policy mismatches because their remaining Torch layers
also use that setting. They never change process-wide backend flags.

Primitive `WanConv3d(weight, precision="ieee" | "tf32" | "bf16")` objects pin
their policy at construction and remain independent of later backend changes.
Low-level `ConvConfig` must explicitly specify the matching FP32 policy; its default
is BF16. Packed weights and their policy must travel together through custom ops.
Strict convolution passes its TF32 permission directly to the library operation.
TF32 and experimental TF32x3 periodically sum partial accumulators with FP32 SIMT
adds; ordinary TF32 still uses one product per operand pair. This is a generic
reduction strategy, independent of model stage or observed output values.

Experimental `precision="tf32x3"` is available only as an explicitly selected
primitive. Neither full-model FP32 mode selects it.

The full-model benchmarks accept `--dtype fp32 --allow-tf32` and
`--dtype fp32 --no-allow-tf32` (the default is no TF32). All timed implementations
receive the same convolution permission; FP32 matmul stays strict in both cases.
By default, validation compares Mega and Torch variants against Diffusers with
the same dtype and TF32 permission, and reports the corresponding PASS/FAIL.
Add `--accuracy-diagnostics` to a TF32 run to also compute an untimed strict FP32
Diffusers reference and print detailed elementwise `1e-4` comparisons. This option
adds one full forward; model criteria are also checked against that extra reference.
Elementwise diagnostics say `DIAG within/outside tolerance` and do not determine
acceptance. Strict FP32 uses the `atol=rtol=1e-4` gate.
Full cache validation remains separately opt-in.

TF32 full-model acceptance uses criteria from the checked-in vLLM-Omni submodule
at `2d9909525a8a805ed9c3a56a5287a48687654270`, rather than increasing the strict
FP32 tolerance: encoder moments, mean and log variance each require normalized
RMSE <= 0.01; decoder pixels require maximum absolute error <= 0.01. The reference
is same-permission Diffusers by default. With `--accuracy-diagnostics`, these
criteria also apply against the independent strict FP32 reference.
These thresholds come from
[`encoder_nrmse_limit`](../../../inference/vllm/vllm-omni/vllm_omni/diffusion/distributed/autoencoders/wan_vae_fastpath/_utils.py)
and
[`test_channels_last_decoder_keeps_layout_and_stays_close`](../../../inference/vllm/vllm-omni/tests/diffusion/distributed/test_wan_vae_fastpath_kernels.py).
vLLM-Omni's encoder benchmark also uses a 50 dB reconstruction PSNR floor when
checking reconstructions; posterior validation alone does not establish that criterion.

In SGLang `baa9befa381511cd4d56f83980951e5a45fffc6d`, the
[`Wan parallel-model test`](../../../inference/sglang/sglang/python/sglang/multimodal_gen/test/single_test_file/test_wan_vae_parallel_groups_4_gpu.py)
uses `atol=rtol=1e-4` with TF32 explicitly disabled. Its lossless normalization
tests require bitwise equality for those particular operations. Neither is an
established TF32 full-model tolerance for these kernels; the vLLM-Omni criteria
above are labeled accordingly, not claimed as SGLang integration acceptance.

Optional full-cache comparisons keep their existing elementwise tolerances,
including when TF32 is enabled. Such a comparison can fail despite passing the
TF32 model criteria; the benchmark preserves and reports that failure. Primitive
TF32 convolution/post-op/cache checks independently round operands and retain
`atol=rtol=1e-4` against a strict convolution of those rounded operands.

`mega_wan_vae/kernels/conv.py` compares raw convolution and the norm/SiLU,
residual, and spatial-padding variants against Torch/cuDNN. Its fusion table
also compares integrated and separate execution under the same dtype and TF32
permission. Raw, total, and separate timings include per-call activation
conversion; kernel-only timings explicitly exclude it. Strict FP32 uses cuDNN
plus separate post-ops and has no fused-kernel timing. Use
`--matched-shape N T H W` to hold geometry fixed across channel widths.

These standalone kernels are intended to support a subsequent extension of cuDNN's
`conv3d_postops` library API, with Wan 2.1/2.2 adapters for vLLM-Omni and SGLang.
This directory does not itself install or patch either serving framework or cuDNN.
Kernel arithmetic, validation, packing and compilation policies must remain reusable;
model adapters handle causal caches and architecture-specific wiring.

- Encoder: NCTHW pixels to posterior parameters (mean and log variance).
- Decoder: unnormalized NCTHW latents to pixels clamped to `[-1, 1]`.
- Latent sampling and latent mean/std scaling belong to the caller.
- Temporal caches are private to each forward call. Videos use `4n+1` frames.
- Spatial tiling and training are outside these inference implementations.

The custom convolutions target Blackwell B200 (SM100) and B300 (SM103).
The full BF16/FP32/TF32 validation matrix uses B300, PyTorch 2.13.0+cu132 and
the repository's CuTe DSL environment. B200 FP32 validation remains outstanding.

## Running

From the repository root, use the existing GPU-enabled PyTorch/CuTe DSL environment:

```bash
python mlsys/gpu/kernels/wan_vae/bench_wan_vae_encoder.py \
  --model ./models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --dtype bf16 --batch-size 1 --frames 17 --resolutions 480x832

python mlsys/gpu/kernels/wan_vae/bench_wan_vae_decoder.py \
  --model ./models/Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --dtype fp32 --allow-tf32 --batch-size 1 --frames 17 --resolutions 256x320

python mlsys/gpu/kernels/wan_vae/mega_wan_vae/kernels/conv.py \
  --dtype fp32 --allow-tf32 --matched-shape 1 4 64 148 --history 0 2
```

Run the complete two-model, two-operation, three-policy matrix with:

```bash
MODEL_ROOT="$PWD/models/Wan-AI" PYTHON=python \
  bash mlsys/gpu/kernels/wan_vae/run-all.sh
```

Each invocation creates a new timestamped directory under `logs/`, with one
stdout/stderr log per model/operation/policy and a `summary.tsv` containing exit
statuses. Set `LOG_DIR` to choose a new destination. The runner completes the
matrix even if a benchmark fails, then returns nonzero if any run failed.
Set `OPERATION=encoder` or `OPERATION=decoder` to run only that half of the matrix.
Set `POLICY=bf16`, `fp32` or `tf32` to restrict the arithmetic policy as well.

For remote execution, use `./run_cute_dsl.py --node NODE SCRIPT ARGS` with an
absolute checkpoint path. Add `--nsys` before the script and `--profile` after it
to run and capture an extra forward per variant; encoder `--profile-mega-only`
restricts capture to Mega. Multiple cases require Nsight's repeat capture mode.

The encoder can choose a batch from available memory and halve it after OOM,
rerunning all variants and validation. Explicit `--batch-size` is never reduced.
FP32 automatic batches divide the memory-scaled estimate by `FP32_BATCH_DIVISOR`
(2). With TF32 disabled, `STRICT_FP32_EXTRA_BATCH_DIVISOR` (4) applies as well,
for a total divisor of 8. Both parameters live in `bench_utils.py`.
FP32 cases below `MIN_AUTO_BATCH_SIZE` (0.8 in `bench_utils.py`) are omitted before
rounding; retained fractional batches below one run with batch one.
Use equal explicit batches for comparisons between GPUs. Compilation, warmup,
input generation and validation are excluded from timings. Peak additional
allocation excludes resident weights and retained compilation buffers; it is
not total process VRAM. Validation also warms up each implementation; the default
then runs only the requested timing samples, reported in the final summary.
The decoder defaults to 17 output frames. Both benchmarks time Diffusers, compiled
Torch and Mega, with three samples each by default. Eager Torch runs only for
validation and optional cache diagnostics. With the default variants and repeats,
all policies require 13 full forwards including validation. TF32 runs with
`--accuracy-diagnostics` add one strict FP32 reference forward.
Profiling and cache validation add untimed work only when requested.
The summary reports Mega speedups versus Diffusers and compiled Torch.

`--no-torch-compile` disables only the Compiled baseline. Wan 2.1 encode/decode
and Wan 2.2 decode still compile Mega's internal chunk implementations; Wan 2.2
Mega encoding calls its kernels directly. Initial warmup reports CuTe compile
calls/cache hits and Torch compile/cache setup time separately, with Inductor FX
and AOTAutograd graph-cache hits, misses and bypasses when those counters change.
Setup time includes tracing and cache loading; it does not imply a disk-cache miss.
CuTe compile time includes DSL disk-cache lookup/load. Nested timers can overlap
and should not be summed.

Each sweep loads the checkpoint and constructs its model wrappers once. Successful
cases retain compiler state while releasing their activations and temporal caches;
OOM recovery resets compiler state but reuses the weights and wrappers. Compilation
remains static: new batch/spatial shapes can require new graphs. The benchmark
scopes Torch's graph limit to preserve the normal per-case budget across the
requested sweep. Repeated chunk shapes reuse their existing graphs.

The standalone convolution script provides callsite and fusion benchmarks
with the same `--dtype` and TF32 options. Its compile-only and verify-only
suites remain BF16; `tests/check_arithmetic_modes.py` verifies all three policies:

```bash
python mlsys/gpu/kernels/wan_vae/mega_wan_vae/kernels/conv.py --model wan21 --verify-only
python mlsys/gpu/kernels/wan_vae/mega_wan_vae/kernels/conv.py \
  --matched-shape 1 4 240 416 --history 0 2
```

Convolution benchmark timings exclude external padding, weight packing and
compilation. Architecture-specific callsite suites use different geometries and
batches, so their rows do not isolate channel-family efficiency.

## Validation and measured limits

Model benchmarks compare complete outputs against Diffusers, with separate
temporal history for every implementation. Shape/dtype mismatches and nonfinite
outputs fail validation. Compilation, allocation and numerical failures remain
visible in the summary and cause a nonzero exit.

Optional `--validate-caches mega` or `--validate-caches torch-compiled` compares
chunk outputs and intermediate caches using the same eager input history.
Both BF16 and FP32 Mega paths support this diagnostic; decoder Mega checks
require `--fused`. Compiled cache checks require the Compiled baseline.
Cache tensors are compared in bounded CPU blocks, outside latency and memory
measurements. The compiled diagnostic alone allows adjacent finite BF16 cache
values; chunk outputs use the elementwise atol/rtol gate.

Measured performance, the optional BF16 cache discrepancy, recovered cuDNN
workspace allocation warnings, and the tensor-operation sanitizer limitation
are documented in [BENCHMARKS.md](BENCHMARKS.md#validation-limitations).
B200 FP32 capacity, reconstruction quality and serving-framework integration
remain unvalidated.

## Tests

These are standalone checks, run as `python mlsys/gpu/kernels/wan_vae/tests/SCRIPT`
from the repository root. They use the project's Python dependencies. CPU-only
runs in the frontend environment need `NVCTM_DSL_DISABLE_PTH=1`; GPU runs need
the normal CUDA/CuTe environment and a supported Blackwell GPU.

| Script / arguments | Environment | Coverage |
|---|---|---|
| `check_arithmetic_policy.py` | CPU | Policy selection, acceptance criteria and opt-in accuracy diagnostics |
| `check_benchmark_reporting.py` | CPU | Output lifetimes, failure reporting, timing summaries and model/graph reuse |
| `check_compile_stats.py` | CPU | CuTe/Torch compilation counters, timers and cache reporting |
| `check_cache_tolerance.py --device cpu` | CPU; also accepts `--device cuda` | Cache tolerances, bounded comparisons, ownership and OOM recovery |
| `check_conv_indexing.py` | CPU; optional `--device cuda` | Index-width boundaries and large-offset arithmetic |
| `check_torch_layouts.py` | CPU; optional CUDA `--model CHECKPOINT` | Shortcut grouping/layouts; Wan 2.2 compiled encoder regression with a checkpoint |
| `check_arithmetic_modes.py --wide-index` | GPU | BF16/strict FP32/TF32 arithmetic, independent TF32 rounding, histories and graph replay |
| `check_conv_contracts.py` | GPU | Inference-only bindings, dtype/device validation, output ownership and streams |
| `check_conv_outputs.py` | GPU | Bitwise BF16 prepared outputs and executable reuse; also run `--operation spatial` |
| `check_pointwise_dtypes.py` | GPU | FP32 norm/residual/stage math, BF16/FP32 packing and dtype separation |
| `check_dynamic_shapes.py` | GPU | BF16 pointwise/C12 geometry reuse, wide indexing and cache writes |
| `check_kernels.py` | GPU | Wan 2.1 BF16 normalization and convolution smoke checks |
| `check_tf32x3.py` | GPU, experimental | Compensated TF32x3 packing/convolution against strict FP32 |
| `check_checkpoint.py --model CHECKPOINT` | GPU + either Wan checkpoint | BF16 encoder/decoder output agreement, fresh-cache repeatability and reference immutability |
| `check_conv_memory.py --precision bf16` / `--precision tf32` | GPU + Compute Sanitizer | Small raw/fused/history/graph memory-check workload |
| `repro_tma_peer_barrier.cu` | B300 + nvcc + Compute Sanitizer | Standalone reproducer for the tensor-operation instrumentation false positive |

`check_pointwise_dtypes.py --large` adds full-buffer checks around and above
2^31 elements and needs about 40 GiB of VRAM. The optional checkpoint case in
`check_torch_layouts.py` uses N=16 at 480x832 to exercise the compiled layout
regression. These heavier checks are explicit opt-ins.

Run `check_conv_memory.py` under `compute-sanitizer --tool memcheck` for the
ordinary memory checks. The C++ reproducer includes its build command; run its
binary without arguments for peer completion or with `local` for the control.
Tensor-operation instrumentation is a separate diagnostic with a known tool
limitation, described in [BENCHMARKS.md](BENCHMARKS.md#sanitizer-checks).

## Organization and compilation

```text
bench_*.py                    Full-model encoder and decoder benchmarks
wan_vae_reference.py           Eager/compiled Torch benchmark baselines
bench_utils.py                Benchmark policy, reports and cache diagnostics
mega_wan_vae/
  encoder.py, decoder.py       Public model dispatchers and policy checks
  models/wan21/, models/wan22/ Architecture and temporal-stage adapters
  layers/torch.py              Shared Torch layers and temporal loops
  layers/fused.py              Custom-op and fusion adapters
  kernels/                    Reusable CuTe DSL primitives
    _conv_config.py           Geometry, specialization and launch policy
    _compile.py               Executable caches and compilation counters
    _geometry.py              Runtime coordinate decomposition
    _precision.py             Arithmetic policy and TF32 operand packing
tests/                        Primitive, policy and checkpoint checks
```

Runtime modules never import benchmark references. Model adapters own causal
cache ordering and stage wiring; kernel specializations depend on arithmetic,
channels, layout, fusion, tiling, pipeline and index width.

General convolution, C12 input gathering, history packing, standalone preparation
and stage shortcuts carry geometry at runtime. Changing supported batch,
frame/spatial sizes, history length or input strides reuses executables when
capabilities are unchanged. CPU-visible metadata builds tensor maps and divisors
without GPU synchronization. Addresses use 64 bits; coordinate ranges select
narrow or wide variants. Each CUDA graph still captures particular geometry and
buffers. Torch chunk compilation retains its separate static-shape policy.

Wan 2.1 uses N96 tiles for C96 and N192 tiles for C192/C384. Its model adapters
retain cuDNN at C384 because differing BF16 accumulation order can amplify in
posterior outputs; primitive C384 remains available for experiments. Supported
stages fuse conv1 preparation, conv2 residual/next-input preparation, and final
spatial padding or head preparation. Fused and separate entry points remain
available; measured benefit should determine upstream dispatch.

## Next investigation: strict FP32 encoder post-op performance

Wan 2.2 strict FP32 encoding shows parity between Mega and Compiled: at N=2,
T=17, 720x1280, Mega takes 7129.020 ms and Compiled takes 7123.947 ms.
The local run is recorded in
`logs/20261026-final/Wan2.2-TI2V-5B-Diffusers-encoder-fp32.log`.

Keep this investigation focused on Mega versus Compiled. Both paths use cuDNN
convolution; Mega uses our custom fused post-ops and Compiled uses Torch's fused
post-processing kernels. Earlier post-op studies showed a clear advantage for
our kernels. Why that advantage does not appear in the full encoder remains
unresolved; using the same convolution backend alone does not explain it.

Next steps:

1. Profile matched strict FP32 encoder runs and account separately for cuDNN
   convolution, fused post-processing, and remaining execution overhead.
2. Verify matching convolution calls, selected kernels and durations, then compare
   the actual post-op workloads and timings at corresponding encoder sites.
3. Locate where the previously measured post-op advantage is lost or offset,
   and reconcile that accounting with the end-to-end timing before drawing a conclusion.

## Upstream integration

The next work is to port reusable arithmetic/validation/compilation contracts
into cuDNN's existing `conv3d_postops` API and improve TF32 performance before
claiming a meaningful decoder speedup. The isolated sanitizer tool limitation
needs an upstream fix. Keep Wan-specific cache/model wiring in framework adapters. vLLM-Omni and SGLang integration then
needs their own tests, serving-shape benchmarks and reconstruction-quality checks.
