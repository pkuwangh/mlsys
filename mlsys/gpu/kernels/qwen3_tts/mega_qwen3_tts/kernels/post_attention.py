"""Fused output projection, residual, RMSNorm, SwiGLU, and down projection.

Hopper/Blackwell, BF16, hidden=1024, FFN=3072, attention width=2048, B<=32.
All batches reuse 96 CTAs: 64 produce output/down and all 96 produce gate/up.
Transaction-barrier mailboxes reduce split-K partials.
Packed sentinel polling publishes residuals and gate/up
activations without release flags; each CTA computes its own norm statistics.

Run this file to validate and benchmark the complete fused operation.
"""

import argparse
import sys
from functools import cache
from pathlib import Path

import cuda.bindings.driver as cuda
import cutlass
import torch
from cutlass import cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from torch import nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mega_qwen3_tts.kernels._utils import device_arch, measure, report_error

# Packed BF16 negative zeros, matching the device kernels' unpublished payload.
PAYLOAD_EMPTY = -2147450880


@cache
def compile_kernel(batch, bucket, eps, arch):
    if arch == "sm_90a":
        from mega_qwen3_tts.kernels.sm90.post_attention import post_attention_host
    else:
        from mega_qwen3_tts.kernels.sm100.post_attention import post_attention_host

    def fake(dtype, *shape):
        return make_fake_compact_tensor(dtype, shape, stride_order=tuple(reversed(range(len(shape)))))

    return cute.compile(
        post_attention_host,
        fake(cutlass.BFloat16, batch, 2048),
        fake(cutlass.BFloat16, batch, 1024),
        fake(cutlass.BFloat16, 16, 32, 64, 64),
        fake(cutlass.BFloat16, 96, 16, 64, 64),
        fake(cutlass.BFloat16, 16, 48, 64, 64),
        fake(cutlass.BFloat16, 1024),
        fake(cutlass.Int32, 2, bucket, 512),
        fake(cutlass.Int32, 2, 96, bucket, 16),
        fake(cutlass.Int32, 112),
        fake(cutlass.BFloat16, batch, 1024),
        make_fake_stream(),
        batch,
        bucket,
        eps,
        options=f"--enable-tvm-ffi --gpu-arch={arch}",
    )


class FusedPostAttention(nn.Module):
    """Inference-only replacement; each instance owns a serialized workspace."""

    def __init__(self, reference):
        super().__init__()
        if tuple(reference.o_proj.weight.shape) != (1024, 2048) or tuple(reference.gate_proj.weight.shape) != (
            3072,
            1024,
        ):
            raise ValueError("The fused block supports the 0.6B talker's 1024/2048/3072 dimensions")
        if reference.o_proj.weight.dtype != torch.bfloat16 or not reference.o_proj.weight.is_cuda:
            raise ValueError("The fused block requires CUDA BF16 weights")
        self.arch = device_arch(reference.o_proj.weight.device)
        self.reference = reference
        packed = torch.cat(
            (reference.gate_proj.weight.reshape(96, 32, 1024), reference.up_proj.weight.reshape(96, 32, 1024)),
            dim=1,
        ).detach()
        self.register_buffer("packed_gate_up", packed.reshape(96, 64, 16, 64).permute(0, 2, 1, 3).contiguous())
        self.register_buffer(
            "packed_o", reference.o_proj.weight.detach().reshape(16, 64, 32, 64).permute(0, 2, 1, 3).contiguous()
        )
        self.register_buffer(
            "packed_down", reference.down_proj.weight.detach().reshape(16, 64, 48, 64).permute(0, 2, 1, 3).contiguous()
        )
        self.workspaces = {}

    @torch.compiler.disable
    def forward(self, attention, residual):
        # Prefill stays in Torch; only one-token decoding uses this schedule.
        if attention.ndim == 3 and attention.shape[1] != 1:
            return self.reference(attention, residual)
        batch = attention.shape[0]
        if not 1 <= batch <= 32:
            raise ValueError("Fused decode supports 1 <= B <= 32")
        # N=16 minimum here, so the reduction epilogue has >=256 elements/rank.
        bucket = 16 if batch <= 16 else 32
        if batch not in self.workspaces:
            device = attention.device
            self.workspaces[batch] = (
                torch.full((2, bucket, 512), PAYLOAD_EMPTY, device=device, dtype=torch.int32),
                torch.full((2, 96, bucket, 16), PAYLOAD_EMPTY, device=device, dtype=torch.int32),
                # 96 CTA epochs + 16 prefetch hints.
                torch.zeros(112, device=device, dtype=torch.int32),
            )
        output = torch.empty_like(residual, memory_format=torch.contiguous_format)
        fn = compile_kernel(batch, bucket, self.reference.norm.eps, self.arch)
        fn(
            attention.reshape(batch, 2048),
            residual.reshape(batch, 1024),
            self.packed_o,
            self.packed_gate_up,
            self.packed_down,
            self.reference.norm.weight,
            *self.workspaces[batch],
            output.reshape(batch, 1024),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )
        return output


@torch.inference_mode()
def trace_once(batch):
    """Launch one fused block without reference GEMMs or benchmark replays."""
    from qwen3_tts_reference import TalkerConfig, TorchPostAttention

    torch.manual_seed(42)
    reference = TorchPostAttention(TalkerConfig()).to(device="cuda", dtype=torch.bfloat16).eval()
    fused = FusedPostAttention(reference)
    attention = torch.randn(batch, 2048, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(batch, 1024, device="cuda", dtype=torch.bfloat16)
    fused(attention, residual)
    torch.cuda.synchronize()
    print(f"B={batch}: one fused block completed")


@torch.inference_mode()
def benchmark(batch, layers, profile=False):
    from qwen3_tts_reference import TalkerConfig, TorchPostAttention, post_attention

    torch.manual_seed(42)
    references = [
        TorchPostAttention(TalkerConfig()).to(device="cuda", dtype=torch.bfloat16).eval() for _ in range(layers)
    ]
    for reference in references:
        reference.norm.weight.uniform_(0.5, 1.5)
    customs = [FusedPostAttention(reference) for reference in references]
    compiled = torch.compile(post_attention, fullgraph=True, options={"emulate_precision_casts": True})
    weights = [
        (r.o_proj.weight, r.norm.weight, r.gate_proj.weight, r.up_proj.weight, r.down_proj.weight, r.norm.eps)
        for r in references
    ]
    attention = torch.randn(batch, 2048, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(batch, 1024, device="cuda", dtype=torch.bfloat16)
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        attention.normal_()
        residual.normal_()
        expected = torch.stack([reference(attention, residual) for reference in references])
        report_error(
            f"B={batch} custom seed={seed}", torch.stack([custom(attention, residual) for custom in customs]), expected
        )
        report_error(
            f"B={batch} compiled seed={seed}",
            torch.stack([compiled(attention, residual, *w) for w in weights]),
            expected,
        )
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        result = customs[0](attention, residual)
    for zero in (True, False):
        if zero:
            attention.zero_()
            residual.zero_()
        else:
            attention.normal_()
            residual.normal_()
        graph.replay()
        report_error(f"B={batch} graph {'zero' if zero else 'new input'}", result, references[0](attention, residual))
    timings = measure(
        {
            "Torch": lambda: [r(attention, residual) for r in references],
            "Compiled": lambda: [compiled(attention, residual, *w) for w in weights],
            "Fused": lambda: [c(attention, residual) for c in customs],
        },
        profile=profile,
    )
    return {name: us / layers for name, us in timings.items()}


def print_summary(results, layers):
    """Print per-block latency and baseline/fused speedups for every batch."""
    headers = ["Batch", "Fused (us)", "Torch (us)", "Compiled (us)", "vs Torch", "vs Compiled"]
    rows = [
        [
            str(batch),
            *[f"{times[name]:.3f}" for name in ("Fused", "Torch", "Compiled")],
            *[f"{times[name] / times['Fused']:.3f}x" for name in ("Torch", "Compiled")],
        ]
        for batch, times in results
    ]
    widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]
    print(f"\nSummary: BF16, {layers} weight sets, median interleaved CUDA-graph timing per block.")
    print("Speedup=baseline/Fused. Compilation, weight packing and validation excluded.")
    for index, row in enumerate([headers, *rows]):
        print(" | ".join(value.rjust(width) for value, width in zip(row, widths)), flush=True)
        if index == 0:
            print("-+-".join("-" * width for width in widths), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch-size", type=int, nargs="+", default=[16, 32], help="Decode batches")
    parser.add_argument("--layers", type=int, default=28, help="Distinct weight sets to rotate through")
    parser.add_argument("--profile", action="store_true", help="Capture named CUDA-graph replays")
    parser.add_argument("--trace-only", action="store_true", help="Launch only one custom block per batch for IKET")
    args = parser.parse_args()
    if args.layers < 1 or any(not 1 <= b <= 32 for b in args.batch_size):
        parser.error("Expected layers >= 1 and 1 <= batch size <= 32")
    if args.trace_only:
        for batch in args.batch_size:
            trace_once(batch)
    else:
        results = [(batch, benchmark(batch, args.layers, args.profile)) for batch in args.batch_size]
        print_summary(results, args.layers)
