# Wan VAE kernels

This directory supports the Wan 2.1 VAE and the Wan 2.2 TI2V-5B VAE.
The top-level directory retains its original name; the shared benchmark
scripts are `bench_wan_vae_encoder.py` and `bench_wan_vae_decoder.py`.
Pass a Diffusers checkpoint with `--model`; omitting it still selects the
synthetic Wan 2.2 architecture.

## Runtime

`MegaWanVaeEncoder(vae)` and `MegaWanVaeDecoder(vae)` select their execution
path from the loaded `AutoencoderKLWan` configuration. Use BF16 CUDA weights,
`.eval()`, and `torch.inference_mode()`.

- Encoder: NCTHW pixels to posterior parameters (mean and log variance).
- Decoder: unnormalized NCTHW latents to pixels clamped to `[-1, 1]`.
- Latent sampling and latent mean/std scaling belong to the caller.
- Temporal caches are private to each forward call. Videos use `4n+1` frames.
- Spatial tiling and training are outside these inference implementations.

The custom convolutions use Blackwell tensor-core instructions. Validation
used an SM103 GPU, PyTorch 2.13.0+cu132 and the existing CuTe DSL environment.

## Organization

```text
mega_wan_vae/
  encoder.py, decoder.py       Public dispatchers
  models/
    wan21/
      encoder.py, decoder.py  Wan 2.1 stage and temporal execution
      _common.py              Wan 2.1 compilation and convolution policy
    wan22/
      encoder.py, decoder.py  Wan 2.2 stage and temporal execution
  layers/torch.py              Shared Torch layers and temporal loops
  layers/fused.py              Shared custom-op and fusion adapters
  kernels/                    Shape-specialized CuTe DSL primitives
wan_vae_reference.py           Eager/compiled benchmark baselines
bench_wan_vae_encoder.py       Shared Wan 2.1/2.2 encoder benchmark
bench_wan_vae_decoder.py       Shared Wan 2.1/2.2 decoder benchmark
tests/                        Kernel and checkpoint verification
```

Runtime modules never import the benchmark reference. Benchmarks compare all
paths against Diffusers as the independent correctness oracle.

Wan 2.1 uses N96 convolution tiles for C96 outputs and N192 tiles for
C192/C384 outputs. The wider tile halves repeated input loads compared with
using N96 for every stage; its seven-stage operand pipeline fits within the
shared-memory limit.
The 384-channel stages retain cuDNN: the custom convolution's accumulation
order can cross BF16 rounding ties and amplify in encoder posterior outputs.
The standalone convolution implementation also supports those wider shapes
for correctness/performance experiments.

Normalization supports 96/192/384 channels, including the scalar reduction
tree used below 128 channels. Fused preparation combines bias/residual,
RMSNorm, SiLU, padding and temporal-cache writes. Attention/head normalization
keeps a fixed reduction tree across compilation boundaries; compiled Wan 2.1
also requests eager division rounding. The existing numerical tolerances are
unchanged.

## Running

From the repository root, use a GPU-enabled Python environment:

```bash
python mlsys/gpu/kernels/wan22_vae/bench_wan_vae_encoder.py \
  --model ./models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --batch-size 1 --frames 5 --resolutions 480x832 --no-torch-compile

python mlsys/gpu/kernels/wan22_vae/bench_wan_vae_decoder.py \
  --model ./models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --frames 5 --resolutions 480x832

python mlsys/gpu/kernels/wan22_vae/tests/check_kernels.py --benchmark
python mlsys/gpu/kernels/wan22_vae/tests/check_checkpoint.py \
  --model ./models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers

python mlsys/gpu/kernels/wan22_vae/mega_wan_vae/kernels/conv.py --model wan21
python mlsys/gpu/kernels/wan22_vae/mega_wan_vae/kernels/conv.py --model wan21 --verify-only
```

The standalone `conv.py` defaults to both architectures; `--model wan21` or
`--model wan22` selects benchmarks, correctness checks, or `--only-compile`.
Wan 2.1 timings cover supported encoder/decoder convolution callsites for
480x832 video at batch 1, including first and subsequent temporal chunks.
They include C384 experiments even though the model keeps those stages on
cuDNN. Raw-convolution timings exclude padding and packing, and print whether
cuDNN autotuning is enabled.

For remote runs, use `./run_cute_dsl.py --node YOUR_GPU_NODE SCRIPT ...`
with an absolute checkpoint path. Add `--nsys` before the script to capture
the benchmarks' CUDA profiler ranges; encoder `--profile-mega-only` restricts
capture to the optimized path. `--no-torch-compile` excludes the optional
compiled Torch baseline, not the optimized model's internal compilation.

Timings exclude compilation and input generation. Peak additional allocation
is the CUDA allocator high-water mark above live allocations at entry to each
timed call; it excludes resident weights and retained compilation buffers and
is not total process VRAM.

The encoder's automatic reference batch scales with free GPU memory, capped at
N=16 for 17 frames at 720x1280, then scales by resolution and frame count. This
reduces the initial workload on smaller GPUs such as B200. If a case still
OOMs, automatic mode halves its batch and reruns all variants and validation;
the retry is printed and the final table shows the actual batch. An explicit
`--batch-size` is never reduced. Comparisons across GPUs should use the same
explicit batch when comparing latency.

Compiled chunk/cache validation keeps eager snapshots in CPU memory, releases
the previous compiled chunk before advancing, and compares caches in blocks
of at most one million elements. Host snapshots/transfers occur only during
validation and are excluded from GPU timings and peak-memory measurements.

The independent compiled Torch encoder can fail the existing tolerance on
the Wan 2.1 480x832, five-frame workload. The optimized Wan 2.1 path preserves
the relevant arithmetic and passes; benchmark failures are still reported
when the optional compiled baseline fails.

`Compiled cache FAIL` specifically means that a compiled chunk output or
temporal-cache tensor fails validation against eager Torch. Chunk outputs use
the final-output atol/rtol. Intermediate caches accept that same tolerance or
adjacent finite BF16 values (one representable step), with matching shapes and
dtypes required. This accounts for rare compiled rounding differences such as
66.0 versus 66.5 without increasing the absolute tolerance for all values.
This check can fail even when the final posterior passes. It does not mark
Mega as failing: each implementation's final-output comparison is printed
separately. Cache diagnostics report values outside atol/rtol, those accepted
as BF16 neighbors, and those still rejected. The full compiled encode also
validates its final posterior using its own accumulated temporal history.
Use `--no-torch-compile` to exclude this optional baseline while retaining
Mega and eager Torch validation against Diffusers.

## Measured Wan 2.1 results

After N192 tuning, standalone raw-convolution measurements on B300 (SM103),
batch 1, BF16, channels-last operands, CUDA-graph replay, with cuDNN autotuning
disabled:

| Channels | Output T x H x W | Custom ms | Torch/cuDNN ms | Speedup |
|---|---|---:|---:|---:|
| 96 -> 192 | 4 x 240 x 416 | 0.2915 | 0.3565 | 1.223x |
| 192 -> 192 | 4 x 240 x 416 | 0.4554 | 0.5502 | 1.208x |
| 384 -> 384 | 2 x 120 x 208 | 0.2368 | 0.2733 | 1.154x |
| 16 -> 384 | 1 x 60 x 104 | 0.0132 | 0.0149 | 1.130x |

All 21 encoder/decoder cases in `conv.py --model wan21` passed correctness
and measured 1.046x to 1.292x versus the reference in this run. The tuning
comparison matched the previous N96 kernel bitwise on its six test shapes.
These measurements exclude preparation and weight packing. C384 remains a
kernel experiment; its model path still uses cuDNN for numerical accuracy.
The Wan 2.1 convolutions still run separately from fused normalization/SiLU/
padding/cache preparation; convolution epilogue fusion is not implemented for
these channel counts.

The full-VAE measurements below preceded this N192 convolution tuning.
Batch 1, BF16, real checkpoint, channels-last weights, four full-call timing
samples, SM103:

| Operation | Frames | Resolution | Diffusers ms | Custom ms | Speedup |
|---|---:|---|---:|---:|---:|
| Encode | 5 | 480x832 | 71.031 | 37.308 | 1.90x |
| Decode | 5 | 480x832 | 132.429 | 65.747 | 2.01x |
| Encode | 9 | 720x1280 | 308.693 | 173.953 | 1.77x |
| Decode | 9 | 720x1280 | 540.964 | 276.525 | 1.96x |

Custom outputs matched Diffusers bitwise on these workloads. This is observed
agreement for the tested inputs, not a guarantee for every shape/input.
Checkpoint tests additionally cover batches 1/2, frames 1/5/9, spatial dimensions
that are multiples of eight but not sixteen, and cache isolation across calls.

Compiled Torch decoding was faster: 62.144 ms and 257.768 ms for the two decode
rows, versus 65.747 ms and 276.525 ms custom. Custom additional peak allocation
was 3.661/10.588 GiB versus 5.436/16.935 GiB compiled. The custom path preserves
the eager arithmetic; compiled Torch's outputs passed the existing decoder
tolerance but were not bitwise equal.

Nsight Systems identified normalization preparation as a substantial encoder
cost. The new vectorized C96 preparation measured 0.507 ms versus 3.381 ms for
eager Torch on `(N,T,H,W,C)=(1,4,480,832,96)` with two history frames. Nsight
Compute reported about 84% compute throughput, 24% DRAM throughput, 44 registers
per thread and 56% achieved occupancy for that kernel. The implementation
retains precise division and activation arithmetic to preserve numerical
agreement.

The Wan 2.2 reorganization was also checked against the original implementation
using its real checkpoint, batch 1, five frames, 480x832. Encoder and decoder
outputs were bitwise unchanged. Median times were 14.642/73.621 ms before and
14.637/73.003 ms after. The original encoder already exceeded its Diffusers
tolerance on 23 of 299,520 posterior values in that workload (maximum absolute
error 0.5); that existing discrepancy is unchanged by this work.
