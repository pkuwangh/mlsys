"""Check reported BF16 cache drift and reject larger or invalid differences."""

import argparse
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bench_wan_vae_encoder as benchmark
import torch
from bench_wan_vae_encoder import cache_tolerance_counts, tensor_blocks, tensor_error


def check_bounded_comparison(device):
    storage = torch.full((2, 6, 4, 5, 6), 66.0, dtype=torch.bfloat16, device=device)
    actual = storage[:, ::2]
    expected = torch.full(actual.shape, 66.0, dtype=torch.bfloat16)
    actual[0, 0, 0, 0, 0] = 66.5
    actual[0, 1, 2, 3, 4] = float("nan")
    actual[-1, -1, -1, -1, -1] = 67.0
    elements = 0
    for a, b in tensor_blocks(actual, expected, max_elements=17):
        assert a.numel() <= 17 and a.shape == b.shape
        assert a.untyped_storage().data_ptr() == storage.untyped_storage().data_ptr()
        elements += a.numel()
    assert elements == actual.numel()
    assert cache_tolerance_counts(actual, expected, atol=0.25, rtol=0.001, max_elements=17) == (2, 1)
    actual[0, 1, 2, 3, 4] = 66.0
    diff = (actual.cpu().float() - expected.float()).abs()
    stats = tensor_error(actual, expected)
    torch.testing.assert_close(torch.tensor(stats), torch.stack((diff.max(), diff.mean(), diff.square().mean().sqrt())))


def check_chunk_lifetimes(device):
    references = []

    def encode(chunk, cache):
        assert all(ref() is None for ref in references), "Previous compiled chunk survived into the next encode"
        value = chunk[:, :, :1].clone()
        if cache[0] is not None:
            assert cache[0].device == chunk.device
            value += cache[0]
        return value, (value.clone(), "sentinel")

    def compiled_encode(chunk, cache):
        output, state = encode(chunk, cache)
        references[:] = [weakref.ref(output), weakref.ref(state[0])]
        return output, state

    eager = SimpleNamespace(patch_size=None, use_tiling=False, new_cache=lambda: (None, None), encode_chunk=encode)
    compiled = SimpleNamespace(chunk_encoder=compiled_encode)
    video = torch.ones((2, 3, 9, 4, 6), dtype=torch.bfloat16, device=device)
    assert benchmark.debug_compiled_chunk_mismatch(eager, compiled, video, atol=0.25, rtol=0.001)
    assert all(ref() is None for ref in references)


def check_oom_recovery():
    case = benchmark.BenchmarkCase(720, 1280, 17, 16)
    vae = SimpleNamespace(clear_cache=lambda: None)
    attempts = []

    def run_case(args, case, vae, device):
        attempts.append(case.batch_size)
        if case.batch_size > 4:
            raise torch.cuda.OutOfMemoryError("Simulated capacity limit")
        # Validation failures must survive a successful allocation retry.
        return benchmark.BenchmarkResult(case, {"Mega": 1.0}, "Compiled cache FAIL")

    with (
        patch.object(benchmark, "run_case", side_effect=run_case),
        patch.object(torch.cuda, "empty_cache"),
        patch.object(torch._dynamo, "reset"),
    ):
        result = benchmark.run_case_with_retry(SimpleNamespace(batch_size=None), case, vae, torch.device("cpu"))
        assert attempts == [16, 8, 4] and result.case.batch_size == 4
        assert result.status == "Compiled cache FAIL"
        attempts.clear()
        result = benchmark.run_case_with_retry(SimpleNamespace(batch_size=16), case, vae, torch.device("cpu"))
        assert attempts == [16] and result.case.batch_size == 16 and result.status == "OOM"

    assert benchmark.auto_reference_batch(160 * 2**30) < benchmark.auto_reference_batch(256 * 2**30)
    print("Automatic batch recovery and explicit batch preservation: PASS", flush=True)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), action="append")
    args = parser.parse_args()
    for device in args.device or ("cpu", "cuda"):
        # Include both sides of a binade boundary, where step sizes differ.
        for actual, expected, counts in (
            (66.5, 66.0, (0, 1)),
            (-64.0, -64.5, (0, 1)),
            (64.5, 64.0, (0, 1)),
            (63.75, 64.0, (0, 0)),
            (67.0, 66.0, (1, 0)),
            (63.5, 64.0, (1, 0)),
            (0.125, 0.0, (0, 0)),
            (float("nan"), 0.0, (1, 0)),
            (float("inf"), float("inf"), (1, 0)),
            (float("inf"), torch.finfo(torch.bfloat16).max, (1, 0)),
        ):
            a = torch.tensor([actual], dtype=torch.bfloat16, device=device)
            b = torch.tensor([expected], dtype=torch.bfloat16, device=device)
            result = cache_tolerance_counts(a, b, atol=0.25, rtol=0.001)
            assert result == counts, (device, actual, expected, result, counts)

        a = torch.tensor([66.5], dtype=torch.bfloat16, device=device)
        b = torch.tensor([66.0], dtype=torch.bfloat16, device=device)
        # The allowance applies only to caches; the output criterion still fails.
        assert not torch.allclose(a, b, atol=0.25, rtol=0.001)
        for invalid in (b.expand(2), b.float()):
            try:
                cache_tolerance_counts(a, invalid, atol=0.25, rtol=0.001)
            except ValueError:
                pass
            else:
                raise AssertionError("Cache shape/dtype mismatch was accepted")

        b = torch.tensor([1.0], device=device)
        a = torch.nextafter(b, torch.full_like(b, float("inf")))
        assert cache_tolerance_counts(a, b, atol=0, rtol=0) == (1, 0)
        check_bounded_comparison(device)
        check_chunk_lifetimes(device)
        print(f"Cache tolerance regression checks ({device}): PASS", flush=True)
    check_oom_recovery()


if __name__ == "__main__":
    main()
