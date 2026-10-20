"""Fused input RMSNorm, QKV projection, Q/K RMSNorm, RoPE, and KV-cache update.

Hopper BF16 decode, B<=32, hidden=1024, 16 Q heads, 8 KV heads, head_dim=128.
Each four-CTA cluster projects one head: two channel halves, each with two
K partitions. Rank 0 gathers the partials and performs the head-local epilogue.
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
from cutlass.cute.experimental import iket
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.experimental import cuda as tensor_map
from cutlass.experimental import primitives as p
from torch import nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mega_qwen3_tts.kernels._hopper import descriptor, fragment_column, fragment_row, mma, swizzle
from mega_qwen3_tts.kernels._utils import measure, report_error


@cute.kernel
def pre_attention_kernel(
    x: cute.Tensor,
    gamma: cute.Tensor,
    weight_map: cutlass.GridConstant[tensor_map.TensorMap],
    qgamma: cute.Tensor,
    kgamma: cute.Tensor,
    cos: cute.Tensor,
    sin: cute.Tensor,
    positions: cute.Tensor,
    kv: cute.Tensor,
    q: cute.Tensor,
    B: cutlass.Constexpr,
    N: cutlass.Constexpr,
    CAPACITY: cutlass.Constexpr,
    EPS: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    head, rank = bid // 4, bid % 4
    split = rank % 2
    weights = cutlass.Array(cutlass.BFloat16, 32768, space=cutlass.AddressSpace.smem, alignment=1024)
    normalized = cutlass.Array(cutlass.BFloat16, N * 512, space=cutlass.AddressSpace.smem, alignment=1024)
    partials = cutlass.Array(cutlass.Float32, 4 * N * 64, space=cutlass.AddressSpace.smem, alignment=128)
    local_partial = cutlass.Array(cutlass.Float32, N * 64, space=cutlass.AddressSpace.smem, alignment=128)
    bars = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
    if tid == 0:
        iket.range_push("qkv_weight_prefetch")
        p.prefetch_tensormap(weight_map.get_ptr())
        p.mbarrier_init(bars, 1)
        if rank == 0:
            p.mbarrier_init(bars.subview(1), 1)
            p.mbarrier_arrive_expect_tx(bars.subview(1), 3 * N * 64 * 4)
        p.fence_mbarrier_init()
        p.mbarrier_arrive_expect_tx(bars, 65536)
        p.cp_async_bulk_tensor_shared_cluster_global(
            weights, weight_map.get_ptr(), [0, 0, split * 8, head * 2 + rank // 2], bars, []
        )
        iket.range_pop()

    iket.range_push("input_norm")
    lane, warp = tid % 32, tid // 32
    for token in cutlass.range_constexpr(N // 16):
        row = warp + token * 16
        values = cute.make_rmem_tensor((8, 4), cutlass.BFloat16)
        scales = cute.make_rmem_tensor((8, 4), cutlass.BFloat16)
        for group in cutlass.range_constexpr(4):
            col = lane * 8 + group * 256
            val = cutlass.full((8,), 0, cutlass.BFloat16)
            if row < B:
                val = (x.iterator.raw_ptr() + row * 1024 + col).load(count=8, alignment=16)
            scale = (gamma.iterator.raw_ptr() + col).load(count=8, alignment=16)
            for i in cutlass.range_constexpr(8):
                values[i, group], scales[i, group] = val[i], scale[i]
        total = cutlass.Float32(0)
        for group in cutlass.range_constexpr(4):
            v = values[None, group].load().to_vector().to(cutlass.Float32)
            squares = v * v
            total += ((squares[0] + squares[1]) + (squares[2] + squares[3])) + (
                (squares[4] + squares[5]) + (squares[6] + squares[7])
            )
        for shift in cutlass.range_constexpr(5):
            total += cute.arch.shuffle_sync_bfly(total, 1 << shift)
        inverse = cute.math.rsqrt(total / 1024 + EPS)
        for group in cutlass.range_constexpr(4):
            if group // 2 == split:
                col = lane * 8 + (group % 2) * 256
                v = values[None, group].load().to_vector().to(cutlass.Float32)
                norm_scale = scales[None, group].load().to_vector().to(cutlass.Float32)
                norm_result = ((v * inverse).to(cutlass.BFloat16).to(cutlass.Float32) * norm_scale).to(cutlass.BFloat16)
                normalized.store(norm_result, idx=(col // 64) * N * 64 + swizzle(row, col % 64), alignment=16)
    p.fence_proxy("async_shared", space=p.SharedSpace.shared_cta)
    p.barrier_cta_sync()
    # Join the cluster before publishing DSMEM, but overlap its arrival with MMA.
    p.barrier_cluster_arrive_relaxed()
    iket.range_pop()

    acc = cute.make_rmem_tensor((N // 2,), cutlass.Float32)
    acc.fill(0)
    if tid >= 128 and tid < 256:
        iket.range_push("qkv_operand_wait")
        while not p.mbarrier_try_wait_parity(bars, 0):
            pass
        iket.range_pop()
        p.wgmma_fence_aligned()
        iket.range_push("qkv_mma")
        for slab in cutlass.range_constexpr(8):
            da = descriptor(weights, slab * 4096)
            db = descriptor(normalized, slab * N * 64)
            for k in cutlass.range_constexpr(4):
                mma(acc, da + 2 * k, db + 2 * k, N)
        p.wgmma_commit_group_sync_aligned()
        p.wgmma_wait_group_sync_aligned(0)
        iket.range_pop()
    iket.range_push("qkv_partial_publish")
    if tid >= 128 and tid < 256:
        for i in cutlass.range_constexpr(N // 2):
            channel = fragment_row(tid - 128, i)
            row = fragment_column(tid - 128, i)
            if rank == 0:
                partials.store(acc[i], idx=row * 64 + channel)
            else:
                local_partial.store(acc[i], idx=row * 64 + channel)
    p.fence_proxy("async_shared", space=p.SharedSpace.shared_cta)
    p.barrier_cta_sync()
    p.barrier_cluster_wait()
    if tid == 0 and rank != 0:
        p.cp_async_bulk_shared_cluster_shared_cta(
            p.mapa(partials.subview(rank * N * 64), 0), local_partial, p.mapa(bars.subview(1), 0), N * 64 * 4
        )
    iket.range_pop()
    if rank == 0:
        iket.range_push("qkv_partial_wait")
        while not p.mbarrier_try_wait_parity(bars.subview(1), 0):
            pass
        iket.range_pop()
        iket.range_push("head_norm_rope_cache")
        position = positions[0]
        for token in cutlass.range_constexpr(N // 16):
            row = warp + token * 16
            channel = lane * 4
            half, local_channel = channel // 64, channel % 64
            first = partials.load(idx=(half * 2 * N + row) * 64 + local_channel, vector_size=4, alignment=16)
            second = partials.load(idx=((half * 2 + 1) * N + row) * 64 + local_channel, vector_size=4, alignment=16)
            projected = (first + second).to(cutlass.BFloat16)
            if head < 24:
                vf = projected.to(cutlass.Float32)
                head_squares = vf * vf
                total = (head_squares[0] + head_squares[1]) + (head_squares[2] + head_squares[3])
                for shift in cutlass.range_constexpr(5):
                    total += cute.arch.shuffle_sync_bfly(total, 1 << shift)
                inverse = cute.math.rsqrt(total / 128 + EPS)
                head_scale = cutlass.full((4,), 0, cutlass.BFloat16)
                if head < 16:
                    head_scale = (qgamma.iterator.raw_ptr() + channel).load(count=4, alignment=8)
                else:
                    head_scale = (kgamma.iterator.raw_ptr() + channel).load(count=4, alignment=8)
                unit = (vf * inverse).to(cutlass.BFloat16).to(cutlass.Float32)
                scaled = (unit * head_scale.to(cutlass.Float32)).to(cutlass.BFloat16).to(cutlass.Float32)
                rotated = cute.make_rmem_tensor((4,), cutlass.Float32)
                for i in cutlass.range_constexpr(4):
                    opposite = cute.arch.shuffle_sync_bfly(scaled[i], 16)
                    rotated[i] = cutlass.select_(channel < 64, -opposite, opposite)
                cv = (cos.iterator.raw_ptr() + channel).load(count=4, alignment=8).to(cutlass.Float32)
                sv = (sin.iterator.raw_ptr() + channel).load(count=4, alignment=8).to(cutlass.Float32)
                left = (scaled * cv).to(cutlass.BFloat16).to(cutlass.Float32)
                right = (rotated.load().to_vector() * sv).to(cutlass.BFloat16).to(cutlass.Float32)
                result = (left + right).to(cutlass.BFloat16)
                if row < B:
                    if head < 16:
                        (q.iterator.raw_ptr() + (row * 16 + head) * 128 + channel).store(result, alignment=8)
                    else:
                        (kv.iterator.raw_ptr() + ((row * 8 + head - 16) * CAPACITY + position) * 128 + channel).store(
                            result, alignment=8
                        )
            else:
                if row < B:
                    offset = B * 8 * CAPACITY * 128 + ((row * 8 + head - 24) * CAPACITY + position) * 128 + channel
                    (kv.iterator.raw_ptr() + offset).store(projected, alignment=8)
        iket.range_pop()
    p.barrier_cluster_arrive_relaxed()
    p.barrier_cluster_wait()


@cute.jit
def pre_attention_host(
    x: cute.Tensor,
    gamma: cute.Tensor,
    weights: cute.Tensor,
    qgamma: cute.Tensor,
    kgamma: cute.Tensor,
    cos: cute.Tensor,
    sin: cute.Tensor,
    positions: cute.Tensor,
    kv: cute.Tensor,
    q: cute.Tensor,
    stream: cuda.CUstream,
    B: cutlass.Constexpr,
    N: cutlass.Constexpr,
    CAPACITY: cutlass.Constexpr,
    EPS: cutlass.Constexpr,
):
    weight_map = tensor_map.create_tensor_map_tiled_from_view(
        weights, (1, 8, 64, 64), swizzle=tensor_map.TensorMapSwizzle.s128b
    )
    pre_attention_kernel(x, gamma, weight_map, qgamma, kgamma, cos, sin, positions, kv, q, B, N, CAPACITY, EPS).launch(
        grid=(128, 1, 1), block=(512, 1, 1), cluster=(4, 1, 1), stream=stream
    )


@cache
def compile_kernel(batch, bucket, capacity, eps):
    """Specialize the decode bucket and contiguous KV-cache strides."""

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
        make_fake_stream(),
        batch,
        bucket,
        capacity,
        eps,
        # Preserve the BF16 multiply rounding before the RoPE add.
        options="--enable-tvm-ffi --gpu-arch=sm_90a --ptxas-options=--fmad=false",
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
        if torch.cuda.get_device_capability(norm.weight.device)[0] != 9:
            raise ValueError("This WGMMA implementation targets Hopper")
        self.norm, self.attention = norm, attention
        weights = torch.cat((attention.q_proj.weight, attention.k_proj.weight, attention.v_proj.weight))
        self.register_buffer(
            "packed_weights", weights.detach().reshape(64, 64, 16, 64).permute(0, 2, 1, 3).contiguous()
        )

    @torch.compiler.disable
    def forward(self, x, kv, positions, cos, sin):
        batch = x.shape[0]
        if x.shape != (batch, 1, 1024) or not 1 <= batch <= 32:
            raise ValueError("Fused QKV supports one-token decode with 1 <= B <= 32")
        if not kv.is_contiguous() or kv.shape[:3] != (2, batch, 8) or kv.shape[-1] != 128:
            raise ValueError("Expected contiguous KV cache [2,B,8,capacity,128]")
        capacity = kv.shape[3]
        q = torch.empty((batch, 16, 1, 128), device=x.device, dtype=x.dtype)
        compile_kernel(batch, 16 if batch <= 16 else 32, capacity, self.norm.eps)(
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
def trace_once(batch, capacity):
    """Launch one custom preparation block for IKET."""
    from qwen3_tts_reference import TalkerConfig, TalkerLayer

    torch.manual_seed(42)
    ref = TalkerLayer(TalkerConfig()).cuda().bfloat16().eval()
    fused = FusedPreAttention(ref.input_layernorm, ref.self_attn)
    x = torch.randn(batch, 1, 1024, device="cuda", dtype=torch.bfloat16)
    kv = torch.zeros(2, batch, 8, capacity, 128, device="cuda", dtype=torch.bfloat16)
    position = torch.tensor([capacity - 1], device="cuda")
    cos = torch.ones(128, device="cuda", dtype=torch.bfloat16)
    fused(x, kv, position, cos, torch.zeros_like(cos))
    torch.cuda.synchronize()
    print(f"B={batch}: one fused preparation block completed")


@torch.inference_mode()
def benchmark(batch, layers, capacity, profile=False):
    """Compare complete preparation blocks with rotating weights and fixed cache position."""
    from unittest.mock import patch

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

    def compiled_slots(ref, kv):
        # Keep cache mutation outside Inductor's full-cache functionalization.
        # CUDA-graph timing still excludes the Python dispatch overhead.
        args = reference_args(ref, cos, sin, kv, positions)
        q, k, v = compiled_math(x, *args[:8], args[-1])
        kv[0].index_copy_(2, positions, k)
        kv[1].index_copy_(2, positions, v)
        return q

    # Verify the extracted block against the original attention module.
    module_cache = caches[0].clone()
    mask = torch.arange(capacity, device="cuda")[None, :] <= positions[:, None]
    with patch("torch.nn.functional.scaled_dot_product_attention", side_effect=lambda q, *a, **kw: q):
        original = refs[0].self_attn(refs[0].input_layernorm(x), module_cache, positions, cos, sin, mask)
    q = pre_attention(x, *reference_args(refs[0], cos, sin, caches[0], positions))
    extracted = q.transpose(1, 2).reshape(batch, 1, 2048)
    torch.testing.assert_close(extracted, original, atol=0, rtol=0)
    torch.testing.assert_close(caches[0], module_cache, atol=0, rtol=0)
    for seed in range(3):
        torch.manual_seed(seed)
        x.zero_() if seed == 0 else x.normal_()
        expected, actual = [], []
        for ref, custom, kv, out_kv in zip(refs, fused, caches, custom_caches):
            expected.append(pre_attention(x, *reference_args(ref, cos, sin, kv, positions)))
            actual.append(custom(x, out_kv, positions, cos, sin))
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
        replay_q = fused[0](x, custom_caches[0], positions, cos, sin)
    for position in (0, capacity // 2, capacity - 1):
        positions.fill_(position)
        cos.copy_((torch.cat((inv, inv)) * position).cos().view_as(cos))
        sin.copy_((torch.cat((inv, inv)) * position).sin().view_as(sin))
        x.normal_()
        graph.replay()
        expected = pre_attention(x, *reference_args(refs[0], cos, sin, caches[0], positions))
        report_error(f"graph position={position} Q", replay_q, expected)
        report_error(f"graph position={position} KV", custom_caches[0][:, :, :, position], caches[0][:, :, :, position])
    # Isolate RoPE from GEMM reduction differences: contraction must not remove a BF16 cast.
    identity_q = fused[0](x, custom_caches[0], positions, torch.ones_like(cos), torch.zeros_like(sin))
    identity_k = custom_caches[0][0, :, :, -1:].clone()
    rotated_q = fused[0](x, custom_caches[0], positions, cos, sin)
    for unrotated, rotated in ((identity_q, rotated_q), (identity_k, custom_caches[0][0, :, :, -1:])):
        partner = torch.cat((-unrotated[..., 64:], unrotated[..., :64]), dim=-1)
        torch.testing.assert_close(rotated, unrotated * cos + partner * sin, atol=0, rtol=0)
    calls = {
        "Torch": lambda: [pre_attention(x, *reference_args(r, cos, sin, kv, positions)) for r, kv in zip(refs, caches)],
        "Compiled": lambda: [
            compiled(x, *reference_args(r, cos, sin, kv, positions)) for r, kv in zip(refs, compiled_caches)
        ],
        "Compiled slots": lambda: [compiled_slots(r, kv) for r, kv in zip(refs, slot_caches)],
        "Fused": lambda: [f(x, kv, positions, cos, sin) for f, kv in zip(fused, custom_caches)],
    }
    return {name: us / layers for name, us in measure(calls, profile=profile).items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch-size", nargs="+", type=int, default=[16, 32])
    parser.add_argument("--layers", type=int, default=28, help="Distinct weight sets")
    parser.add_argument("--capacity", type=int, default=129, help="KV-cache capacity")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--trace-only", action="store_true", help="Launch only one custom block per batch for IKET")
    args = parser.parse_args()
    if args.layers < 1 or args.capacity < 1 or any(not 1 <= b <= 32 for b in args.batch_size):
        parser.error("Expected positive layers/capacity and 1 <= batch size <= 32")
    if args.trace_only:
        for b in args.batch_size:
            trace_once(b, args.capacity)
    else:
        results = [(b, benchmark(b, args.layers, args.capacity, args.profile)) for b in args.batch_size]
        print(f"\nBF16 | {args.layers} weight sets | median interleaved CUDA-graph timing per block")
        print("Compiled slots: compiled math + direct index_copy_ cache writes, all CUDA-graphed.")
        print("Batch | Torch (us) | Compiled (us) | Compiled slots (us) | Fused (us) | vs slots")
        print("------+------------+---------------+---------------------+------------+---------")
        for b, result in results:
            print(
                f"{b:5} | {result['Torch']:10.3f} | {result['Compiled']:13.3f} | "
                f"{result['Compiled slots']:19.3f} | {result['Fused']:10.3f} | "
                f"{result['Compiled slots'] / result['Fused']:7.3f}x"
            )
