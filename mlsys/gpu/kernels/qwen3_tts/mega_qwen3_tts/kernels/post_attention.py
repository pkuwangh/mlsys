"""Fused output projection, residual, RMSNorm, SwiGLU, and down projection.

Hopper/Blackwell, BF16, hidden=1024/2048, FFN=3*hidden, attention width=2048, B<=32.
Both sizes reuse 96 CTAs: 64 produce output/down and all 96 produce gate/up.
The larger shape uses 128-channel tiles and streams weights and activations.
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
def compile_kernel(batch, bucket, eps, arch, hidden=1024):
    tile_m = 128 if hidden == 2048 else 64
    gate_channels = tile_m // 2
    if arch == "sm_90a":
        if hidden == 2048:
            from mega_qwen3_tts.kernels.sm90.post_attention_wide import post_attention_host
        else:
            from mega_qwen3_tts.kernels.sm90.post_attention import post_attention_host
    else:
        if hidden == 2048:
            from mega_qwen3_tts.kernels.sm100.post_attention_wide import post_attention_host
        else:
            from mega_qwen3_tts.kernels.sm100.post_attention import post_attention_host

    def fake(dtype, *shape):
        return make_fake_compact_tensor(dtype, shape, stride_order=tuple(reversed(range(len(shape)))))

    return cute.compile(
        post_attention_host,
        fake(cutlass.BFloat16, batch, 2048),
        fake(cutlass.BFloat16, batch, hidden),
        fake(cutlass.BFloat16, hidden // tile_m, 32, tile_m, 64),
        fake(cutlass.BFloat16, 3 * hidden // gate_channels, hidden // 64, tile_m, 64),
        fake(cutlass.BFloat16, hidden // tile_m, 3 * hidden // 64, tile_m, 64),
        fake(cutlass.BFloat16, hidden),
        fake(cutlass.Int32, 2, bucket, hidden // 2),
        fake(cutlass.Int32, 2, 3 * hidden // gate_channels, bucket, gate_channels // 2),
        fake(cutlass.Int32, 112 if hidden == 1024 else 96),
        fake(cutlass.BFloat16, batch, hidden),
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
        hidden = reference.o_proj.weight.shape[0]
        if (
            hidden not in (1024, 2048)
            or tuple(reference.o_proj.weight.shape) != (hidden, 2048)
            or tuple(reference.gate_proj.weight.shape) != (3 * hidden, hidden)
        ):
            raise ValueError("The fused block supports hidden=1024/2048, attention=2048, FFN=3*hidden")
        if reference.o_proj.weight.dtype != torch.bfloat16 or not reference.o_proj.weight.is_cuda:
            raise ValueError("The fused block requires CUDA BF16 weights")
        self.arch = device_arch(reference.o_proj.weight.device)
        self.reference = reference
        self.hidden = hidden
        tile_m = 128 if hidden == 2048 else 64
        self.gate_channels = tile_m // 2
        tiles = 3 * hidden // self.gate_channels
        packed = torch.cat(
            (
                reference.gate_proj.weight.reshape(tiles, self.gate_channels, hidden),
                reference.up_proj.weight.reshape(tiles, self.gate_channels, hidden),
            ),
            dim=1,
        ).detach()
        self.register_buffer(
            "packed_gate_up", packed.reshape(tiles, tile_m, hidden // 64, 64).permute(0, 2, 1, 3).contiguous()
        )
        self.register_buffer(
            "packed_o",
            reference.o_proj.weight.detach().reshape(hidden // tile_m, tile_m, 32, 64).permute(0, 2, 1, 3).contiguous(),
        )
        self.register_buffer(
            "packed_down",
            reference.down_proj.weight.detach()
            .reshape(hidden // tile_m, tile_m, 3 * hidden // 64, 64)
            .permute(0, 2, 1, 3)
            .contiguous(),
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
                torch.full((2, bucket, self.hidden // 2), PAYLOAD_EMPTY, device=device, dtype=torch.int32),
                torch.full(
                    (2, 3 * self.hidden // self.gate_channels, bucket, self.gate_channels // 2),
                    PAYLOAD_EMPTY,
                    device=device,
                    dtype=torch.int32,
                ),
                # The 1024-wide schedule also uses 16 prefetch hints.
                torch.zeros(112 if self.hidden == 1024 else 96, device=device, dtype=torch.int32),
            )
        output = torch.empty_like(residual, memory_format=torch.contiguous_format)
        fn = compile_kernel(batch, bucket, self.reference.norm.eps, self.arch, self.hidden)
        fn(
            attention.reshape(batch, 2048),
            residual.reshape(batch, self.hidden),
            self.packed_o,
            self.packed_gate_up,
            self.packed_down,
            self.reference.norm.weight,
            *self.workspaces[batch],
            output.reshape(batch, self.hidden),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )
        return output


@torch.inference_mode()
def trace_once(batch, hidden=1024):
    """Launch one fused block without reference GEMMs or benchmark replays."""
    from qwen3_tts_reference import TalkerConfig, TorchPostAttention

    torch.manual_seed(42)
    reference = (
        TorchPostAttention(TalkerConfig(hidden_size=hidden, intermediate_size=3 * hidden))
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )
    fused = FusedPostAttention(reference)
    attention = torch.randn(batch, 2048, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(batch, hidden, device="cuda", dtype=torch.bfloat16)
    fused(attention, residual)
    torch.cuda.synchronize()
    print(f"B={batch}: one fused block completed")


@torch.inference_mode()
def benchmark(batch, layers, profile=False, hidden=1024):
    from qwen3_tts_reference import TalkerConfig, TorchPostAttention, post_attention

    torch.manual_seed(42)
    references = [
        TorchPostAttention(TalkerConfig(hidden_size=hidden, intermediate_size=3 * hidden))
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
        for _ in range(layers)
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
    residual = torch.randn(batch, hidden, device="cuda", dtype=torch.bfloat16)
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


def print_summary(results, layers, hidden=1024):
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
    print(f"\nSummary: BF16, hidden={hidden}, {layers} weight sets, median interleaved CUDA-graph timing per block.")
    print("Speedup=baseline/Fused. Compilation, weight packing and validation excluded.")
    for index, row in enumerate([headers, *rows]):
        print(" | ".join(value.rjust(width) for value, width in zip(row, widths)), flush=True)
        if index == 0:
            print("-+-".join("-" * width for width in widths), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch-size", type=int, nargs="+", default=[8, 16, 32], help="Decode batches")
    parser.add_argument("--hidden-size", type=int, choices=(1024, 2048), default=1024)
    parser.add_argument("--layers", type=int, default=28, help="Distinct weight sets to rotate through")
    parser.add_argument("--profile", action="store_true", help="Capture named CUDA-graph replays")
    parser.add_argument("--trace-only", action="store_true", help="Launch only one custom block per batch for IKET")
    args = parser.parse_args()
    if args.layers < 1 or any(not 1 <= b <= 32 for b in args.batch_size):
        parser.error("Expected layers >= 1 and 1 <= batch size <= 32")
    if args.trace_only:
        for batch in args.batch_size:
            trace_once(batch, args.hidden_size)
    else:
        results = [(batch, benchmark(batch, args.layers, args.profile, args.hidden_size)) for batch in args.batch_size]
        print_summary(results, args.layers, args.hidden_size)
