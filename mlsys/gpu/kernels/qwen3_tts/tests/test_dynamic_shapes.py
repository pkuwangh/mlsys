"""Shape changes reuse compiled auxiliary kernels without changing their results."""

import pytest
import torch
from mega_qwen3_tts.kernels import argmax_embedding, embedding_update, sampling, snake_beta, state_copy

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")


@pytest.mark.parametrize("scatter", [False, True])
@torch.inference_mode()
def test_state_copy_graph_replay(scatter):
    """Reuse the binary across batches; duplicated padding slots retain the first row."""
    torch.manual_seed(42)
    state_copy.compile_kernel.cache_clear()
    storage = tuple(torch.randn(64, width, device="cuda", dtype=torch.bfloat16) for width in (8, 1040))
    for batch in (1, 7, 32, 64):
        indices = torch.arange(batch, device="cuda", dtype=torch.int64)
        rows = tuple(torch.randn(batch, buffer.shape[1], device="cuda", dtype=buffer.dtype) for buffer in storage)
        state_copy.copy_state_rows(storage, rows, indices, scatter=scatter)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            state_copy.copy_state_rows(storage, rows, indices, scatter=scatter)
        for step in range(3):
            slots = [(row // 2 + step) % 64 for row in range(batch)]
            indices.copy_(torch.tensor(slots, device="cuda"))
            for buffer in (*storage, *rows):
                buffer.normal_()
            if scatter:
                expected = tuple(buffer.clone() for buffer in storage)
                for target, source in zip(expected, rows):
                    for row in reversed(range(batch)):
                        target[slots[row]].copy_(source[row])
            else:
                expected = tuple(buffer[indices] for buffer in storage)
            graph.replay()
            for actual, target in zip(storage if scatter else rows, expected):
                torch.testing.assert_close(actual, target, atol=0, rtol=0)
    assert state_copy.compile_kernel.cache_info().misses == 1


@torch.inference_mode()
def test_auxiliary_batch_reuse():
    """Batch changes must reuse one binary per operation, including strided code outputs."""
    torch.manual_seed(42)
    compilers = (
        argmax_embedding.compile_kernel,
        embedding_update.compile_kernel,
        sampling.compile_kernel,
        sampling.compile_sorted,
    )
    for compile in compilers:
        compile.cache_clear()
    table = torch.randn(2048, 1024, device="cuda", dtype=torch.bfloat16)
    projection = torch.randn(2048, 2048, device="cuda", dtype=table.dtype)
    for batch in (1, 7, 32, 1):
        args = sampling.inputs(batch, 50)
        codes = torch.full((batch, 16), -1, device="cuda", dtype=torch.int64)
        for actual, expected in zip(
            argmax_embedding.argmax_embedding(args[0], table, projection, out_codes=codes[:, 3]),
            argmax_embedding.reference(args[0], table, projection),
        ):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        tokens = args[0].argmax(-1)
        accumulated = torch.randn(batch, 1024, device="cuda", dtype=table.dtype)
        expected_sum = accumulated.clone()
        for actual, expected in zip(
            embedding_update.embedding_update(tokens, table, accumulated, projection),
            embedding_update.reference(tokens, table, expected_sum, projection),
        ):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(accumulated, expected_sum, atol=0, rtol=0)
        for actual, expected in zip(
            sampling.sample_logits(*args, table=table, projected_table=projection, out_codes=codes[:, 3]),
            sampling.reference(*args, table=table, projected_table=projection),
        ):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        logprobs, indices = args[0].float().log_softmax(-1).topk(50, dim=-1)
        actual = sampling.sample_sorted(logprobs, indices, args[4], args[5])
        expected = sampling.sorted_reference(logprobs, indices, args[4], args[5])
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert torch.all(codes[:, :3] == -1)
        assert torch.all(codes[:, 4:] == -1)
    for compile in compilers:
        assert compile.cache_info().misses == 1


@pytest.mark.parametrize("channels_last", [False, True])
@torch.inference_mode()
def test_snake_length_reuse(channels_last):
    """Changing audio length must reuse the activation binary for the selected vector width."""
    torch.manual_seed(42)
    alpha, beta = (torch.randn(96, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    activation = snake_beta.SnakeBeta(alpha, beta, vector=4, lookup=False)
    snake_beta.compile_kernel.cache_clear()
    for batch, length in ((1, 32), (3, 132), (1, 68)):
        x = torch.randn(batch, 96, length, device="cuda", dtype=alpha.dtype)
        if channels_last:
            x = x.transpose(1, 2).contiguous().transpose(1, 2)
        torch.testing.assert_close(activation(x), snake_beta.reference(x, alpha, beta), atol=0, rtol=0)
    assert snake_beta.compile_kernel.cache_info().misses == 1


@torch.inference_mode()
def test_snake_large_offsets():
    """Cross the signed 32-bit element boundary without allocating a full reference."""
    if torch.cuda.mem_get_info()[0] < 12 * 1024**3:
        pytest.skip("Large-offset regression needs 12 GiB free device memory")
    length = 699056
    x = torch.ones(32, 96, length, device="cuda", dtype=torch.bfloat16)
    alpha = torch.linspace(-0.5, 0.5, 96, device=x.device, dtype=x.dtype)
    beta = -alpha
    actual = snake_beta.SnakeBeta(alpha, beta)(x)
    # Check every channel and batch at both ends, including rows beyond 2**31.
    expected = snake_beta.reference(x[:, :, :8], alpha, beta)
    torch.testing.assert_close(actual[:, :, :8], expected, atol=0, rtol=0)
    torch.testing.assert_close(actual[:, :, -8:], expected, atol=0, rtol=0)
