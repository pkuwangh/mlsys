"""Fused input RMSNorm, QKV projection, Q/K RMSNorm, RoPE, and KV-cache update.

Hopper/Blackwell BF16 decode, B<=32, hidden=1024, 16 Q heads, 8 KV heads, head_dim=128.
Each four-CTA cluster projects one head: two channel halves, each with two
K partitions. Rank 0 gathers the partials and performs the head-local epilogue.
An optional additive-mask output eliminates SDPA's boolean-mask conversion.
Run this file to validate and benchmark against eager and compiled Torch.
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


@cache
def compile_kernel(batch, bucket, capacity, eps, write_mask, arch):
    """Specialize the decode bucket and contiguous KV-cache strides."""
    if arch == "sm_90a":
        from mega_qwen3_tts.kernels.sm90.pre_attention import pre_attention_host
    else:
        from mega_qwen3_tts.kernels.sm100.pre_attention import pre_attention_host

    def fake(dtype, *shape):
        return make_fake_compact_tensor(dtype, shape, stride_order=tuple(reversed(range(len(shape)))))

    return cute.compile(
        pre_attention_host,
        fake(cutlass.BFloat16, batch, 1024),
        fake(cutlass.BFloat16, 1024),
        fake(cutlass.BFloat16, 64, 16, 64, 64),
        fake(cutlass.BFloat16, 128),
        fake(cutlass.BFloat16, 128),
        fake(cutlass.BFloat16, 128),
        fake(cutlass.BFloat16, 128),
        fake(cutlass.Int64, 1),
        fake(cutlass.BFloat16, 2, batch, 8, capacity, 128),
        fake(cutlass.BFloat16, batch, 16, 1, 128),
        fake(cutlass.BFloat16, 1, capacity) if write_mask else None,
        make_fake_stream(),
        batch,
        bucket,
        capacity,
        eps,
        # Preserve the BF16 multiply rounding before the RoPE add.
        options=f"--enable-tvm-ffi --gpu-arch={arch} --ptxas-options=--fmad=false",
    )


class FusedPreAttention(nn.Module):
    """Decode-only QKV preparation with a contiguous, caller-owned KV cache."""

    def __init__(self, norm, attention):
        super().__init__()
        c = attention.config
        if (c.hidden_size, c.num_attention_heads, c.num_key_value_heads, c.head_dim) != (1024, 16, 8, 128):
            raise ValueError("Fused QKV supports the 0.6B talker's 1024/16/8/128 dimensions")
        if norm.weight.dtype != torch.bfloat16 or not norm.weight.is_cuda:
            raise ValueError("Fused QKV requires CUDA BF16 weights")
        self.arch = device_arch(norm.weight.device)
        self.norm, self.attention = norm, attention
        weights = torch.cat((attention.q_proj.weight, attention.k_proj.weight, attention.v_proj.weight))
        self.register_buffer(
            "packed_weights", weights.detach().reshape(64, 64, 16, 64).permute(0, 2, 1, 3).contiguous()
        )

    @torch.compiler.disable
    def forward(self, x, kv, positions, cos, sin, mask=None):
        """Write Q/K/V and, when supplied, a BF16 additive mask of shape [1, capacity]."""
        batch = x.shape[0]
        if x.shape != (batch, 1, 1024) or not 1 <= batch <= 32:
            raise ValueError("Fused QKV supports one-token decode with 1 <= B <= 32")
        if not kv.is_contiguous() or kv.shape[:3] != (2, batch, 8) or kv.shape[-1] != 128:
            raise ValueError("Expected contiguous KV cache [2,B,8,capacity,128]")
        capacity = kv.shape[3]
        if mask is not None and (
            mask.shape != (1, capacity) or mask.dtype != x.dtype or not mask.is_contiguous() or mask.device != x.device
        ):
            raise ValueError("Expected contiguous additive mask [1,capacity] matching the input dtype/device")
        q = torch.empty((batch, 16, 1, 128), device=x.device, dtype=x.dtype)
        compile_kernel(batch, 16 if batch <= 16 else 32, capacity, self.norm.eps, mask is not None, self.arch)(
            x.reshape(batch, 1024),
            self.norm.weight,
            self.packed_weights,
            self.attention.q_norm.weight,
            self.attention.k_norm.weight,
            cos.reshape(128),
            sin.reshape(128),
            positions,
            kv,
            q,
            mask,
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )
        return q


def reference_args(layer, cos, sin, kv, positions):
    """Collect explicit baseline operands without packing or copying weights."""
    a = layer.self_attn
    return (
        layer.input_layernorm.weight,
        a.q_proj.weight,
        a.k_proj.weight,
        a.v_proj.weight,
        a.q_norm.weight,
        a.k_norm.weight,
        cos,
        sin,
        kv,
        positions,
        layer.input_layernorm.eps,
    )


@torch.inference_mode()
def trace_once(batch, capacity, write_mask=False):
    """Launch one custom preparation block for IKET."""
    from qwen3_tts_reference import TalkerConfig, TalkerLayer

    torch.manual_seed(42)
    ref = TalkerLayer(TalkerConfig()).cuda().bfloat16().eval()
    fused = FusedPreAttention(ref.input_layernorm, ref.self_attn)
    x = torch.randn(batch, 1, 1024, device="cuda", dtype=torch.bfloat16)
    kv = torch.zeros(2, batch, 8, capacity, 128, device="cuda", dtype=torch.bfloat16)
    position = torch.tensor([capacity - 1], device="cuda")
    cos = torch.ones(128, device="cuda", dtype=torch.bfloat16)
    mask = torch.empty((1, capacity), device="cuda", dtype=torch.bfloat16) if write_mask else None
    fused(x, kv, position, cos, torch.zeros_like(cos), mask)
    torch.cuda.synchronize()
    print(f"B={batch}: one fused preparation block completed")


@torch.inference_mode()
def benchmark(batch, layers, capacity, profile=False, write_mask=False):
    """Compare complete preparation blocks with rotating weights and fixed cache position."""
    from qwen3_tts_reference import TalkerConfig, TalkerLayer, pre_attention, prepare_qkv

    torch.manual_seed(42)
    refs = [TalkerLayer(TalkerConfig()).cuda().bfloat16().eval() for _ in range(layers)]
    for ref in refs:
        for norm in (ref.input_layernorm, ref.self_attn.q_norm, ref.self_attn.k_norm):
            norm.weight.uniform_(0.5, 1.5)
    fused = [FusedPreAttention(ref.input_layernorm, ref.self_attn) for ref in refs]
    x = torch.randn(batch, 1, 1024, device="cuda", dtype=torch.bfloat16)
    caches = [torch.randn(2, batch, 8, capacity, 128, device="cuda", dtype=torch.bfloat16) for _ in refs]
    custom_caches = [kv.clone() for kv in caches]
    compiled_caches = [kv.clone() for kv in caches]
    slot_caches = [kv.clone() for kv in caches]
    compiled = torch.compile(pre_attention, fullgraph=True, options={"emulate_precision_casts": True})
    compiled_math = torch.compile(prepare_qkv, fullgraph=True, options={"emulate_precision_casts": True})
    positions = torch.tensor([capacity - 1], device="cuda")
    inv = 1.0 / 1_000_000.0 ** (torch.arange(0, 128, 2, device="cuda").float() / 128)
    angles = torch.cat((inv, inv)) * positions[0]
    cos, sin = angles.cos().bfloat16().view(1, 1, 1, 128), angles.sin().bfloat16().view(1, 1, 1, 128)
    additive_mask = torch.empty((1, capacity), device="cuda", dtype=x.dtype) if write_mask else None

    def check_mask():
        if additive_mask is not None:
            valid = torch.arange(capacity, device="cuda")[None, :] <= positions[:, None]
            torch.testing.assert_close(
                additive_mask, torch.where(valid, 0.0, -float("inf")).to(x.dtype), atol=0, rtol=0
            )

    def compiled_slots(ref, kv):
        # Keep cache mutation outside Inductor's full-cache functionalization.
        # CUDA-graph timing still excludes the Python dispatch overhead.
        args = reference_args(ref, cos, sin, kv, positions)
        q, k, v = compiled_math(x, *args[:8], args[-1])
        kv[0].index_copy_(2, positions, k)
        kv[1].index_copy_(2, positions, v)
        return q

    for seed in range(3):
        torch.manual_seed(seed)
        x.zero_() if seed == 0 else x.normal_()
        expected, actual = [], []
        for ref, custom, kv, out_kv in zip(refs, fused, caches, custom_caches):
            expected.append(pre_attention(x, *reference_args(ref, cos, sin, kv, positions)))
            actual.append(custom(x, out_kv, positions, cos, sin, additive_mask))
        check_mask()
        report_error(f"B={batch} seed={seed} Q", torch.stack(actual), torch.stack(expected))
        report_error(
            f"B={batch} seed={seed} KV",
            torch.stack([kv[:, :, :, -1] for kv in custom_caches]),
            torch.stack([kv[:, :, :, -1] for kv in caches]),
        )
    for kv, out_kv in zip(caches, custom_caches):
        torch.testing.assert_close(out_kv[:, :, :, :-1], kv[:, :, :, :-1], atol=0, rtol=0)
    compiled_q = compiled(x, *reference_args(refs[0], cos, sin, compiled_caches[0], positions))
    report_error(f"B={batch} compiled Q", compiled_q, expected[0])
    report_error(f"B={batch} compiled KV", compiled_caches[0][:, :, :, -1], caches[0][:, :, :, -1])
    report_error(f"B={batch} compiled slots Q", compiled_slots(refs[0], slot_caches[0]), expected[0])
    report_error(f"B={batch} compiled slots KV", slot_caches[0][:, :, :, -1], caches[0][:, :, :, -1])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay_q = fused[0](x, custom_caches[0], positions, cos, sin, additive_mask)
    for position in (0, capacity // 2, capacity - 1):
        positions.fill_(position)
        cos.copy_((torch.cat((inv, inv)) * position).cos().view_as(cos))
        sin.copy_((torch.cat((inv, inv)) * position).sin().view_as(sin))
        x.normal_()
        graph.replay()
        check_mask()
        expected = pre_attention(x, *reference_args(refs[0], cos, sin, caches[0], positions))
        report_error(f"graph position={position} Q", replay_q, expected)
        report_error(f"graph position={position} KV", custom_caches[0][:, :, :, position], caches[0][:, :, :, position])
    # Isolate RoPE from GEMM reduction differences: contraction must not remove a BF16 cast.
    identity_q = fused[0](x, custom_caches[0], positions, torch.ones_like(cos), torch.zeros_like(sin), additive_mask)
    identity_k = custom_caches[0][0, :, :, -1:].clone()
    rotated_q = fused[0](x, custom_caches[0], positions, cos, sin, additive_mask)
    for unrotated, rotated in ((identity_q, rotated_q), (identity_k, custom_caches[0][0, :, :, -1:])):
        partner = torch.cat((-unrotated[..., 64:], unrotated[..., :64]), dim=-1)
        torch.testing.assert_close(rotated, unrotated * cos + partner * sin, atol=0, rtol=0)
    calls = {
        "Torch": lambda: [pre_attention(x, *reference_args(r, cos, sin, kv, positions)) for r, kv in zip(refs, caches)],
        "Compiled": lambda: [
            compiled(x, *reference_args(r, cos, sin, kv, positions)) for r, kv in zip(refs, compiled_caches)
        ],
        "Compiled slots": lambda: [compiled_slots(r, kv) for r, kv in zip(refs, slot_caches)],
        "Fused": lambda: [f(x, kv, positions, cos, sin, additive_mask) for f, kv in zip(fused, custom_caches)],
    }
    return {name: us / layers for name, us in measure(calls, profile=profile).items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch-size", nargs="+", type=int, default=[16, 32])
    parser.add_argument("--layers", type=int, default=28, help="Distinct weight sets")
    parser.add_argument("--capacity", type=int, default=129, help="KV-cache capacity")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--write-mask", action="store_true", help="Also write an additive attention mask in the fused kernel"
    )
    parser.add_argument("--trace-only", action="store_true", help="Launch only one custom block per batch for IKET")
    args = parser.parse_args()
    if args.layers < 1 or args.capacity < 1 or any(not 1 <= b <= 32 for b in args.batch_size):
        parser.error("Expected positive layers/capacity and 1 <= batch size <= 32")
    if args.trace_only:
        for b in args.batch_size:
            trace_once(b, args.capacity, args.write_mask)
    else:
        results = [
            (b, benchmark(b, args.layers, args.capacity, args.profile, args.write_mask)) for b in args.batch_size
        ]
        print(f"\nBF16 | {args.layers} weight sets | median interleaved CUDA-graph timing per block")
        print("Compiled slots: compiled math + direct index_copy_ cache writes, all CUDA-graphed.")
        if args.write_mask:
            print("Fused additionally writes the SDPA additive mask; Torch baselines time QKV preparation only.")
        print("Batch | Torch (us) | Compiled (us) | Compiled slots (us) | Fused (us) | vs slots")
        print("------+------------+---------------+---------------------+------------+---------")
        for b, result in results:
            print(
                f"{b:5} | {result['Torch']:10.3f} | {result['Compiled']:13.3f} | "
                f"{result['Compiled slots']:19.3f} | {result['Fused']:10.3f} | "
                f"{result['Compiled slots'] / result['Fused']:7.3f}x"
            )
