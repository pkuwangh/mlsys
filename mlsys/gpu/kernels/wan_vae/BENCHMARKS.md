# Wan VAE benchmark results

Current runtime contracts and commands are in [README.md](README.md). This file
records the measured results and remaining validation limitations.

## Complete validation matrix (2026-10-26)

The recorded `run-all.sh` matrix covers all 12 model/operation/policy
combinations with exit status zero: **32 measured cases, all output gates passing**.
The matrix covers Wan 2.1 and Wan 2.2 encoders/decoders in BF16, strict FP32 and
FP32 with TF32 allowed. Local stdout/stderr logs are indexed in
`logs/20261026-final/summary.tsv`, with `environment.txt` and `source.sha256`
alongside them. These generated artifacts are ignored by Git and are not included
in a fresh checkout; use `run-all.sh` below to generate a new set.
Validation for these measurements includes the extra strict FP32 reference and
elementwise diagnostics. These checks correspond to `--accuracy-diagnostics`
in the model benchmark scripts and are excluded from timings. Default validation
uses only the same-policy reference.

Hardware: one NVIDIA B300 SXM6 AC, 1100 W limit; PyTorch 2.13.0+cu132,
CUDA 13.2, cuDNN 9.20. Local Diffusers checkpoints and deterministic synthetic
video/latents were used. These are numerical and performance checks, not a
natural-video quality evaluation. Each row is the median of three full forwards
after compilation and warmup, including activation packing and normal launches.
Baselines and Mega use the same storage dtype, convolution permission, strict
FP32 matmul, input and output geometry. Speedup is Compiled / Mega.

Representative **T=17, 720x1280** results follow. Encoder batch sizes intentionally
differ by arithmetic policy; compare implementations within a row. The full logs
also cover 480x832 and longer encoder inputs, subject to the automatic batch cutoff.

| Model/path | Policy | Batch | Mega ms | Diffusers ms | Compiled ms | Speedup |
|---|---|---:|---:|---:|---:|---:|
| Wan 2.2 encoder | BF16 | 16 | 880.370 | 2605.325 | 1191.513 | 1.353x |
| Wan 2.1 encoder | BF16 | 16 | 1921.418 | 5391.020 | 2290.677 | 1.192x |
| Wan 2.2 encoder | FP32 strict | 2 | 7129.020 | 7320.772 | 7123.947 | 0.999x |
| Wan 2.1 encoder | FP32 strict | 2 | 11795.744 | 12189.072 | 11828.317 | 1.003x |
| Wan 2.2 encoder | FP32 / TF32 | 8 | 879.723 | 1700.692 | 1021.000 | 1.161x |
| Wan 2.1 encoder | FP32 / TF32 | 8 | 1661.051 | 3235.572 | 1908.855 | 1.149x |
| Wan 2.2 decoder | BF16 | 1 | 227.492 | 527.211 | 249.328 | 1.096x |
| Wan 2.1 decoder | BF16 | 1 | 171.181 | 528.573 | 214.586 | 1.254x |
| Wan 2.2 decoder | FP32 strict | 1 | 11843.665 | 12170.491 | 11856.529 | 1.001x |
| Wan 2.1 decoder | FP32 strict | 1 | 10142.922 | 10494.647 | 10178.709 | 1.004x |
| Wan 2.2 decoder | FP32 / TF32 | 1 | 450.684 | 733.442 | 452.770 | 1.005x |
| Wan 2.1 decoder | FP32 / TF32 | 1 | 330.126 | 675.422 | 383.401 | 1.161x |

Strict FP32 passes the `atol=rtol=1e-4` gate. TF32 passes the
model criteria against both same-policy and independent strict FP32 Diffusers:
encoder moments, mean and logvar each have NRMSE <= 0.01; decoder maximum pixel
error is <= 0.01. Elementwise TF32 comparisons at 1e-4 are diagnostics and do not
determine acceptance. BF16 uses `atol=0.25, rtol=0.001` for encoding and
`atol=0.05, rtol=0.02` for decoding.

Additional coverage includes the large-batch compiled Wan 2.2 encoder and
independent full-buffer packing, history, residual and initial-bias checks above
2^31 elements in BF16/FP32.

## Validation limitations

Wan 2.2 strict FP32 decoding completes at N=1, T=17, 720x1280. Its oversized
cuDNN workspace allocation attempt can still print an OOM warning, but the plan
selection recovers; it is not a failed benchmark. Strict FP32 remains slow and
near Compiled latency. Wan 2.2 TF32 decoder performance is also near Compiled;
profiling and a five-sample A/B test did not justify changing integrated dispatch.
No universal speedup is claimed.

Optional intermediate-cache validation is not part of the default matrix.
Wan 2.2 BF16 intermediate caches can exceed `atol=0.05, rtol=0.02` despite passing
final-output checks. Independent FP32 probes indicate BF16 rounding and
cancellation differences; the cache diagnostic continues to report these failures.
Reconstruction PSNR, B200 FP32 capacity and serving-framework integration have
not been established.

## Primitive validation and matched convolution measurements

Primitive measurements use different geometries and timing methodology from
the full-model 720p table above.

Primitive verification covers every supported channel pair, narrow and forced
wide indexing, executable reuse across geometry, histories 0/1/2, residual and
spatial outputs, and graph replay with changed inputs/history. Ordinary TF32
uses an independent integer-bit rounding oracle before strict cuDNN convolution.
BF16 spatial validation checks raw convolution against cuDNN, then requires
bitwise agreement with separate BF16 additions of that raw result. A rounding
difference between convolution implementations can become large relative to the
post-residual value after cancellation; checking the epilogue against its own raw
result isolates post-op correctness from convolution rounding.
The primitive suite passes in all three policies, as does the experimental
TF32x3 suite. Sanitizer coverage and its instrumentation
limitation are described below.

Matched TF32 convolution timing uses output N/T/H/W=1/4/64/148 (balanced
waves on 148 SMs), two history frames, bias, residual bias, residual addition,
RMSNorm/SiLU and preparation. Each path is measured in rotating CUDA-graph
order, eight samples of five replays after 100 ms warmup. Activation packing
is included; weight packing and compilation are excluded. All paths use FP32
storage with TF32 allowed. The prepared/separate outputs pass `1e-4` checks.

| Channels | Raw ms | Raw + separate post-ops ms | Integrated ms | cuDNN + post-ops ms | Separate / integrated |
|---:|---:|---:|---:|---:|---:|
| 96 | 0.037683 | 0.057754 | 0.061440 | 0.055706 | 0.9400x |
| 160 | 0.079258 | 0.108141 | 0.118374 | 0.127181 | 0.9135x |
| 192 | 0.109158 | 0.142746 | 0.153190 | 0.154006 | 0.9318x |
| 256 | 0.180224 | 0.225485 | 0.242688 | 0.205210 | 0.9291x |
| 320 | 0.282214 | 0.343242 | 0.376115 | 0.421786 | 0.9126x |

Integrated TF32 preparation is 6–10% slower than separate post-ops here. It is
not a demonstrated fusion optimization at this geometry; both entry points
remain available for upstream integration and further profiling. With TF32
disabled, prepared API and explicit separate post-ops timings agree within
0.1%, as expected because both use strict library convolution plus post-ops.
BF16 integrated fusion at the same geometry is 1.03–1.07x faster than separate
post-ops across those channel widths.

Reproduce the full-model matrix and the primitive checks separately:

```bash
MODEL_ROOT="$PWD/models/Wan-AI" PYTHON=python bash mlsys/gpu/kernels/wan_vae/run-all.sh
python mlsys/gpu/kernels/wan_vae/tests/check_arithmetic_modes.py --wide-index
python mlsys/gpu/kernels/wan_vae/tests/check_conv_contracts.py
python mlsys/gpu/kernels/wan_vae/mega_wan_vae/kernels/conv.py \
  --dtype fp32 --allow-tf32 --matched-shape 1 4 64 148 --history 2
```

For individual benchmark scripts, use `--allow-tf32` or `--no-allow-tf32` to
select the FP32 convolution permission. Use `--validate-caches mega` or
`--validate-caches torch-compiled` for the separate, more expensive full-cache check.

The current convolution CLI runs the shared benchmark suite. Its matched-shape
rows include variants with and without a residual, with residual bias disabled;
they are different workloads from the residual-bias table above. Kernel-only
timings exclude activation conversion; raw, total and separate timings include it.

## Sanitizer checks

Ordinary Compute Sanitizer memcheck reports zero errors for C96/C320 BF16 and
TF32 persistent raw/fused/history/graph cases in
[tests/check_conv_memory.py](tests/check_conv_memory.py).
Enabling `--check-tensor-ops yes` reproduces a peer-CTA barrier report with
Compute Sanitizer 2026.2.1 and 2026.3.0.

The [PTX instruction contract](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor)
permits peer-CTA completion with `shared::cluster` and `cta_group::2`.
[tests/repro_tma_peer_barrier.cu](tests/repro_tma_peer_barrier.cu) isolates this
operation without Python, CuTe, convolution or MMA. Both modes copy all 32 values
correctly; 2026.3.0 tensor-operation instrumentation rejects the permitted peer
mode and accepts the local-barrier control. This identifies a tool false positive
for that report. No suppression or kernel workaround was added. An upstream tool
fix is still needed before this instrumentation mode can be a clean acceptance gate.
