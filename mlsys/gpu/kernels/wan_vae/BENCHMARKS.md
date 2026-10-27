# Wan VAE benchmark results

Current runtime contracts and commands are in [README.md](README.md). This file
records the measured results and remaining validation limitations.

## Complete validation matrix (2026-10-27)

The recorded `run-all.sh` matrix covers all 12 model/operation/policy
combinations with exit status zero: **32 measured cases, all output gates passing**.
The matrix covers Wan 2.1 and Wan 2.2 encoders/decoders in BF16, strict FP32 and
FP32 with TF32 allowed. Local stdout/stderr logs are indexed in
`logs/20261027-b300-cudnn-dispatch/summary.tsv`, with `environment.txt` and `source.sha256`
alongside them. These generated artifacts are ignored by Git and are not included
in a fresh checkout; use `run-all.sh` below to generate a new set.
Validation uses the same-policy Diffusers reference. TF32's optional extra strict
FP32 reference and elementwise diagnostics are disabled in this matrix; enable
`--accuracy-diagnostics` for that additional check. The earlier baseline remains
in `logs/20261026-b300-baseline/`.

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
| Wan 2.2 encoder | BF16 | 16 | 883.525 | 2600.201 | 1195.615 | 1.353x |
| Wan 2.1 encoder | BF16 | 16 | 1931.441 | 5389.648 | 2296.426 | 1.189x |
| Wan 2.2 encoder | FP32 strict | 2 | 7117.188 | 7319.659 | 7123.123 | 1.001x |
| Wan 2.1 encoder | FP32 strict | 2 | 11803.575 | 12195.440 | 11832.636 | 1.002x |
| Wan 2.2 encoder | FP32 / TF32 | 8 | 872.819 | 1703.740 | 1030.340 | 1.180x |
| Wan 2.1 encoder | FP32 / TF32 | 8 | 1678.046 | 3238.830 | 1913.984 | 1.141x |
| Wan 2.2 decoder | BF16 | 1 | 229.491 | 532.444 | 247.135 | 1.077x |
| Wan 2.1 decoder | BF16 | 1 | 170.983 | 529.421 | 215.387 | 1.260x |
| Wan 2.2 decoder | FP32 strict | 1 | 11848.611 | 12173.196 | 11864.285 | 1.001x |
| Wan 2.1 decoder | FP32 strict | 1 | 10148.617 | 10497.879 | 10181.071 | 1.003x |
| Wan 2.2 decoder | FP32 / TF32 | 1 | 437.510 | 732.638 | 453.066 | 1.036x |
| Wan 2.1 decoder | FP32 / TF32 | 1 | 331.465 | 677.372 | 382.490 | 1.154x |

Strict FP32 passes the `atol=rtol=1e-4` gate. TF32 passes the
model criteria against same-policy Diffusers: encoder moments, mean and logvar
each have NRMSE <= 0.01; decoder maximum pixel error is <= 0.01. Elementwise TF32
comparisons at 1e-4 are diagnostics and do not determine acceptance.
BF16 uses `atol=0.25, rtol=0.001` for encoding and
`atol=0.05, rtol=0.02` for decoding.

Additional coverage includes the large-batch compiled Wan 2.2 encoder and
independent full-buffer packing, history, residual and initial-bias checks above
2^31 elements in BF16/FP32.

## FP32/TF32 performance analysis

Matched profiles cover both models, both directions and both FP32 policies at
T=17, 720x1280: encoder N=2 for strict FP32 and N=8 for TF32, decoder N=1.
Each comparison uses the same checkpoint, input, dtype and backend permissions;
profiles use seed 0, while the complete benchmark matrix uses seed 42.
Local evidence is under `logs/20261027-b300-investigation/`: per-variant Chrome
traces, convolution shapes/kernel counts, `accounting.json`, A/B logs, and
Nsight Compute reports. The profiling/replay scripts are saved alongside them.

Timing samples run before profiling. Operator accounting counts device activities
once, excludes annotation ranges, and merges overlapping intervals within each
category. This matters for cuDNN's concurrent decoder kernels: simply summing
kernel durations overstates elapsed time. Attention and copies are recorded
separately in the JSON. Profiling can perturb absolute latency; use the unprofiled
matrix above for speedups and the matched traces for the execution breakdown.

### Strict FP32: post-ops help, but convolution dominates

GPU-active milliseconds from matched traces:

| Model/path | Mega cuDNN | Compiled cuDNN | Mega postprocessing | Compiled postprocessing |
|---|---:|---:|---:|---:|
| Wan 2.2 encoder | 7073.07 | 7069.35 | 29.42 | 48.09 |
| Wan 2.2 decoder | 11780.65 | 11775.70 | 51.22 | 66.19 |
| Wan 2.1 encoder | 11663.76 | 11654.68 | 76.00 | 115.06 |
| Wan 2.1 decoder | 10043.39 | 10047.69 | 61.47 | 95.43 |

Postprocessing includes the standalone CuTe kernels and Torch/Triton fused
normalization, activation, residual, padding, cache and layout work. Mega saves
tens of milliseconds here. cuDNN accounts for 98.5–99.5% of device-active time;
these savings are only a few tenths of a percent of a 7–12 second forward.
Matching convolution shapes use the same selected kernels and call counts.
Python/launch gaps are small in these traces and do not explain the parity.

Replaying the encoder's 36 distinct preparation signatures (105 calls per
encode) confirms the post-op advantage against **compiled** Torch, not just eager
Torch. Our kernels pass all signatures; 35 compiled reference signatures pass.
Across those 35 signatures/104 calls, call-weighted CUDA-graph medians total
24.40 ms custom versus 51.49 ms compiled Torch. Isolated compilation has different
fusion boundaries from the full model, so these totals do not replace the trace
accounting above. Representative steady-state cases, all with two history frames:

| N/T/H/W/C | Extra inputs/output | Custom ms | Compiled ms |
|---|---|---:|---:|
| 2/4/360/640/160 | Input bias | 0.674 | 1.798 |
| 2/4/360/640/160 | Input bias, residual and saved sum | 1.042 | 1.848 |
| 2/4/180/320/320 | Input bias | 0.416 | 0.907 |
| 2/2/90/160/640 | Input bias | 0.118 | 0.315 |

One isolated compiled reference produces incorrect padded output at
N/T/H/W/C=2/2/90/160/640 with one history frame, input bias, residual and residual
bias. Its cache and residual outputs are correct. The failure reproduces in
`logs/20261027-b300-investigation/repro_inductor_prep.py` using only PyTorch,
without Wan/CuTe imports or custom operators, including with the DSL startup
hook disabled (`NVCTM_DSL_DISABLE_PTH=1`). This reference is excluded from
performance comparisons; no tolerance or compiler workaround is applied.
The full-model compiled graphs pass their separate validation gates.

Wan 2.2's input convolution uses native C12 in strict FP32. The C16-padding
control uses the same cuDNN kernel family but performs unnecessary channel work.
A rotating nine-sample CUDA-graph comparison at N=2, input spatial size 360x640
isolates this cost:

| Output chunk frames | Calls per encode | Native C12 ms/call | Padded C16 ms/call |
|---:|---:|---:|---:|
| 1 | 1 | 2.503 | 3.369 |
| 4 | 4 | 9.904 | 13.344 |

That is 42.12 versus 56.74 ms per encode, a **14.63 ms** difference. It explains
why the padded input could offset most of the post-op advantage. TF32 and BF16
retain C16 packing where required by their custom input convolution.

### TF32: convolution, fusion and packing must be measured together

GPU-active milliseconds; Mega's convolution column includes post-ops integrated
inside convolution kernels, so it must not be interpreted as raw convolution alone:

| Model/path | Compiled conv | Compiled post-ops | Mega conv + integrated post-ops | Mega rounding/packing | Mega separate post-ops |
|---|---:|---:|---:|---:|---:|
| Wan 2.2 encoder | 777.86 | 202.26 | 718.51 | 69.65 | 41.91 |
| Wan 2.2 decoder | 355.62 | 70.30 | 359.87 | 0.01 | 58.16 |
| Wan 2.1 encoder | 1170.07 | 498.24 | 1158.27 | 139.89 | 130.66 |
| Wan 2.1 decoder | 230.32 | 99.74 | 228.58 | 27.24 | 25.70 |

The encoders and Wan 2.1 decoder benefit from the complete fused pipeline.
Wan 2.2's C256 decoder sites instead favor cuDNN plus our preparation kernels.
With integrated custom C256 dispatch, the trace attributed 153.14 ms to those
convolutions/fused post-ops plus packing, versus 114.49 ms for the corresponding
cuDNN convolutions. Mega saved 30.52 ms in the remaining postprocessing, largely
offsetting that extra cost. This explains near-parity with Compiled.

A full-model A/B changes only the C256 backend/fusion choice. All four paths
pass the same-policy Diffusers TF32 criterion. Medians of seven rotating samples,
N=1 and T=17:

| Resolution | Custom integrated ms | Custom + separate post-ops ms | cuDNN + our post-ops ms | Compiled ms |
|---|---:|---:|---:|---:|
| 256x320 | 48.447 | 46.807 | 43.641 | 45.416 |
| 480x832 | 204.538 | 205.025 | 194.922 | 202.750 |
| 720x1280 | 460.132 | 461.372 | 441.969 | 453.887 |

The adapter therefore selects cuDNN at these TF32 sites. The improvement is
modest: 4% versus integrated Mega and 2.7% versus Compiled at 720p in this A/B.
It does not establish the same benefit on B200 or every serving shape.

The isolated steady-state C256->C256 site uses output N/T/H/W=1/4/360/640,
two history frames and norm/SiLU/preparation. Integrated custom execution takes
4.866 ms versus 5.446 ms with separate custom post-ops: fusion itself helps.
cuDNN plus the same post-ops takes 4.566 ms. These CUDA-graph timings include
activation packing. Nsight Compute separately measures the convolution kernels:

| Kernel | Kernel ms | Tensor-pipe active cycles (% elapsed) | DRAM throughput (% peak) |
|---|---:|---:|---:|
| Custom raw | 3.501 | 87.1 | 15.4 |
| Custom integrated | 3.693 | 83.8 | 20.4 |
| cuDNN raw | 3.317 | 95.6 | 15.9 |

These counters point to convolution execution, rather than DRAM bandwidth, as
the main cost at this site. They do not isolate which scheduling or accumulation
change would close the gap. Hardware-counter collection uses replay and excludes
the separate packing kernel; its timings are not interchangeable with full-model
or CUDA-graph totals.

TF32 activation packing costs about 70 ms and 140 ms in the Wan 2.2/2.1 encoder
profiles. A large-buffer probe moves 14.28 GB of input/output in 2.147 ms
(6.65 TB/s effective read+write bandwidth). Increasing vector width from four to
eight elements improves that pass by only about 4%; sixteen is slower. The current
packer is retained. Eliminating the pass during preparation has more potential
than this small isolated retuning, but must preserve unrounded FP32 caches and
residuals and the independent TF32 rounding contract.

## Validation limitations

Wan 2.2 strict FP32 decoding completes at N=1, T=17, 720x1280. Its oversized
cuDNN workspace allocation attempt can still print an OOM warning, but the plan
selection recovers; it is not a failed benchmark. Strict FP32 remains slow and
near Compiled latency. Wan 2.2 TF32 decoding has a modest measured benefit
from cuDNN at C256 with separate custom post-ops, as detailed above. No universal
speedup is claimed.

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
