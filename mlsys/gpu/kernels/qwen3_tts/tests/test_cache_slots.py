"""Token-major cache addressing and CUDA graph replay checks."""

import pytest
import torch
from mega_qwen3_tts.kernels.post_attention import PackedPostAttention
from mega_qwen3_tts.kernels.post_attention import compile_kernel as compile_post
from mega_qwen3_tts.kernels.pre_attention import (
    FusedPreAttention,
    SlotPreAttention,
    compile_kernel,
    compile_slot_kernel,
)
from torch_qwen3_tts.talker import TalkerConfig, TalkerLayer, prepare_qkv

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")


@pytest.mark.parametrize("hidden,batch", [(1024, 1), (1024, 7), (1024, 32), (2048, 16)])
@torch.inference_mode()
def test_cache_slots_graph_replay(hidden, batch):
    """Replay must use new per-request positions and slots without touching other cache entries."""
    torch.manual_seed(42)
    layer = TalkerLayer(TalkerConfig(hidden_size=hidden, intermediate_size=3 * hidden)).cuda().bfloat16().eval()
    attention = layer.self_attn
    fused = SlotPreAttention(
        layer.input_layernorm.weight,
        torch.cat((attention.q_proj.weight, attention.k_proj.weight, attention.v_proj.weight)),
        attention.q_norm.weight,
        attention.k_norm.weight,
        layer.input_layernorm.eps,
    )
    for norm in (layer.input_layernorm, attention.q_norm, attention.k_norm):
        norm.weight.uniform_(0.5, 1.5)
    x = torch.randn(batch, hidden, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(batch, device="cuda", dtype=torch.int64)
    slots = positions * 2 + 1
    if batch == 7:
        slots = slots.int()
    cache_size = 4 * batch + 8
    key_cache = torch.randn(cache_size, 8, 128, device="cuda", dtype=torch.bfloat16)
    value_cache = torch.randn_like(key_cache)
    angles = (
        torch.arange(128, device="cuda")[:, None]
        * (1_000_000.0 ** (-torch.arange(0, 128, 2, device="cuda").float() / 128))[None, :]
    )
    rope = torch.cat((angles.cos(), angles.sin()), dim=-1)
    if batch == 7:
        rope = rope.bfloat16()
    fused(x, positions, rope, slots, key_cache, value_cache)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual_q = fused(x, positions, rope, slots, key_cache, value_cache)
    for step in range(3):
        x.normal_()
        positions.copy_((torch.arange(batch, device="cuda") * 3 + step) % 128)
        slots.copy_(torch.arange(batch, device="cuda") * 2 + step)
        before_key, before_value = key_cache.clone(), value_cache.clone()
        graph.replay()
        cos, sin = rope[positions].chunk(2, dim=-1)
        cos = torch.cat((cos, cos), dim=-1).bfloat16()[:, None, None, :]
        sin = torch.cat((sin, sin), dim=-1).bfloat16()[:, None, None, :]
        expected_q, expected_k, expected_v = prepare_qkv(
            x[:, None],
            layer.input_layernorm.weight,
            attention.q_proj.weight,
            attention.k_proj.weight,
            attention.v_proj.weight,
            attention.q_norm.weight,
            attention.k_norm.weight,
            cos,
            sin,
            layer.input_layernorm.eps,
        )
        torch.testing.assert_close(actual_q, expected_q.reshape(batch, 2048), atol=0.02, rtol=0.02)
        torch.testing.assert_close(key_cache[slots], expected_k.squeeze(2), atol=0.02, rtol=0.02)
        torch.testing.assert_close(value_cache[slots], expected_v.squeeze(2), atol=0.02, rtol=0.02)
        untouched = torch.ones(cache_size, device="cuda", dtype=torch.bool)
        untouched[slots] = False
        torch.testing.assert_close(key_cache[untouched], before_key[untouched], atol=0, rtol=0)
        torch.testing.assert_close(value_cache[untouched], before_value[untouched], atol=0, rtol=0)

    reference = layer.post_attention
    post = PackedPostAttention(
        reference.o_proj.weight,
        reference.gate_proj.weight,
        reference.up_proj.weight,
        reference.down_proj.weight,
        reference.norm.weight,
        reference.norm.eps,
    )
    attention_output = torch.randn(batch, 2048, device="cuda", dtype=torch.bfloat16)
    post(attention_output, x)
    torch.cuda.synchronize()
    post_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(post_graph):
        actual_output = post(attention_output, x)
    for _ in range(3):
        x.normal_()
        attention_output.normal_()
        post_graph.replay()
        torch.testing.assert_close(actual_output, reference(attention_output, x), atol=0.02, rtol=0.02)


@pytest.mark.parametrize("hidden", [1024, 2048])
@torch.inference_mode()
def test_dynamic_sizes_reuse_compilation(hidden):
    """Live rows and cache capacities must not create a new specialization within a tile bucket."""
    torch.manual_seed(42)
    layer = TalkerLayer(TalkerConfig(hidden_size=hidden, intermediate_size=3 * hidden)).cuda().bfloat16().eval()
    attention = layer.self_attn
    contiguous = FusedPreAttention(layer.input_layernorm, attention)
    slotted = SlotPreAttention(
        layer.input_layernorm.weight,
        torch.cat((attention.q_proj.weight, attention.k_proj.weight, attention.v_proj.weight)),
        attention.q_norm.weight,
        attention.k_norm.weight,
        layer.input_layernorm.eps,
    )
    reference = layer.post_attention
    post = PackedPostAttention(
        reference.o_proj.weight,
        reference.gate_proj.weight,
        reference.up_proj.weight,
        reference.down_proj.weight,
        reference.norm.weight,
        reference.norm.eps,
    )
    compilers = (compile_kernel, compile_slot_kernel, compile_post)
    for compile in compilers:
        compile.cache_clear()
    for batch, capacity in ((1, 33), (7, 65), (16, 129), (17, 65), (32, 129), (7, 257)):
        x = torch.randn(batch, 1, hidden, device="cuda", dtype=torch.bfloat16)
        kv = torch.zeros(2, batch, 8, capacity, 128, device="cuda", dtype=x.dtype)
        position = torch.tensor([capacity - 1], device="cuda")
        cos, sin = torch.ones(128, device="cuda", dtype=x.dtype), torch.zeros(128, device="cuda", dtype=x.dtype)
        mask = torch.empty(1, capacity, device="cuda", dtype=x.dtype)
        q = contiguous(x, kv, position, cos, sin, mask)
        expected = prepare_qkv(
            x,
            layer.input_layernorm.weight,
            attention.q_proj.weight,
            attention.k_proj.weight,
            attention.v_proj.weight,
            attention.q_norm.weight,
            attention.k_norm.weight,
            cos,
            sin,
            layer.input_layernorm.eps,
        )
        for actual, target in zip((q, kv[0, :, :, -1:], kv[1, :, :, -1:]), expected):
            torch.testing.assert_close(actual, target, atol=0.02, rtol=0.02)
        assert torch.count_nonzero(kv[:, :, :, :-1]) == 0
        assert torch.count_nonzero(mask) == 0
        keys, values = (torch.zeros(capacity, 8, 128, device="cuda", dtype=x.dtype) for _ in range(2))
        rope = torch.cat((cos[:64], sin[:64])).expand(capacity, 128).contiguous()
        slots = torch.arange(batch, device="cuda")
        actual = slotted(x[:, 0], position.expand(batch).contiguous(), rope, slots, keys, values)
        torch.testing.assert_close(actual, expected[0].reshape(batch, 2048), atol=0.02, rtol=0.02)
        torch.testing.assert_close(keys[slots], expected[1].squeeze(2), atol=0.02, rtol=0.02)
        torch.testing.assert_close(values[slots], expected[2].squeeze(2), atol=0.02, rtol=0.02)
        attn = torch.randn(batch, 2048, device="cuda", dtype=x.dtype)
        torch.testing.assert_close(post(attn, x[:, 0]), reference(attn, x[:, 0]), atol=0.02, rtol=0.02)
    for compile in compilers:
        assert compile.cache_info().misses == 2
