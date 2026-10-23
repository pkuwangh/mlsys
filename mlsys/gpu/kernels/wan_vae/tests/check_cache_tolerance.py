"""Check optional cache diagnostics, buffer ownership, and comparison policy."""

import argparse
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bench_utils as validation
import bench_wan_vae_decoder as decoder_benchmark
import bench_wan_vae_encoder as benchmark
import torch
from bench_utils import cache_tolerance_counts, tensor_blocks, tensor_error


def check_cpu_storage_reuse(device):
    buffer = validation._CpuTensorBuffer()
    source = torch.arange(2 * 3 * 4 * 5 * 6, device=device, dtype=torch.float32).reshape(2, 3, 4, 5, 6)
    first = buffer.copy_from(source)
    pointer = first.untyped_storage().data_ptr()
    assert first.device.type == "cpu" and first.is_contiguous(memory_format=torch.channels_last_3d)
    torch.testing.assert_close(first, source.cpu())
    source.add_(1)
    # A snapshot must not alias the source, even in the CPU-only check.
    torch.testing.assert_close(first, source.cpu() - 1)
    for value in (source[:, :, :1], source, source[:, ::2, :, ::2]):
        copied = buffer.copy_from(value)
        assert copied.untyped_storage().data_ptr() == pointer
        torch.testing.assert_close(copied, value.cpu())
    larger = torch.ones((3, 3, 4, 5, 6), device=device)
    grown = buffer.copy_from(larger)
    assert buffer.storage.numel() == larger.numel()
    torch.testing.assert_close(grown, larger.cpu())
    grown_pointer = grown.untyped_storage().data_ptr()
    assert buffer.copy_from(source).untyped_storage().data_ptr() == grown_pointer
    bf16 = buffer.copy_from(source.bfloat16())
    assert bf16.dtype == torch.bfloat16
    torch.testing.assert_close(bf16, source.cpu().bfloat16())


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
    assert cache_tolerance_counts(
        actual, expected, atol=0.25, rtol=0.001, max_elements=17, allow_bf16_neighbors=True
    ) == (2, 1)
    actual[0, 1, 2, 3, 4] = 66.0
    diff = (actual.cpu().float() - expected.float()).abs()
    stats = tensor_error(actual, expected)
    torch.testing.assert_close(torch.tensor(stats), torch.stack((diff.max(), diff.mean(), diff.square().mean().sqrt())))


def check_chunk_lifetimes(device, cache_nthwc=False):
    references = []

    def encode(chunk, cache):
        assert all(ref() is None for ref in references), "Previous candidate chunk survived into the next encode"
        value = chunk[:, :, :1].clone()
        if cache[0] is not None:
            assert cache[0].device == chunk.device
            value += cache[0]
        return value, (value.clone(), "sentinel")

    def candidate_encode(chunk, cache):
        if cache_nthwc:
            cache = tuple(entry.permute(0, 4, 1, 2, 3) if isinstance(entry, torch.Tensor) else entry for entry in cache)
        output, state = encode(chunk, cache)
        references[:] = [weakref.ref(output), weakref.ref(state[0])]
        if cache_nthwc:
            state = tuple(entry.permute(0, 2, 3, 4, 1) if isinstance(entry, torch.Tensor) else entry for entry in state)
        return output, state

    eager = SimpleNamespace(patch_size=None, use_tiling=False, new_cache=lambda: (None, None), encode_chunk=encode)
    video = torch.ones((2, 3, 9, 4, 6), dtype=torch.bfloat16, device=device)

    def compare_cpu(actual, expected, **kwargs):
        assert actual.device.type == expected.device.type == "cpu"
        return cache_tolerance_counts(actual, expected, **kwargs)

    with patch.object(validation, "cache_tolerance_counts", side_effect=compare_cpu) as comparison:
        assert benchmark.validate_chunk_caches(
            eager, candidate_encode, video, atol=0.25, rtol=0.001, label="Mega", cache_nthwc=cache_nthwc
        )
        assert comparison.call_count == 6  # One output and one cache tensor per chunk.
    assert all(ref() is None for ref in references)


def check_chunk_output_contract(device):
    chunk = torch.ones((1, 1, 2, 1, 1), device=device)

    def eager(value, cache, index):
        return value.clone(), ()

    # Broadcast-compatible shapes must still fail, as must a changed dtype.
    for wrong in (lambda value: value[:, :, :1], lambda value: value.double()):
        assert not validation.validate_cache_chunks(
            [chunk],
            (),
            eager,
            lambda value, cache, index, wrong=wrong: (wrong(value), ()),
            iterations=1,
            atol=0,
            rtol=0,
            label="Mega",
        )
    assert validation.validate_cache_chunks(
        [chunk],
        (),
        eager,
        eager,
        iterations=1,
        atol=0,
        rtol=0,
        label="Mega",
    )

    # Equal infinities must not pass a chunk's output gate, even without caches.
    for value in (float("inf"), -float("inf"), float("nan")):
        assert not validation.validate_cache_chunks(
            [torch.full_like(chunk, value)],
            (),
            eager,
            eager,
            iterations=1,
            atol=0,
            rtol=0,
            label="Mega",
        )


def check_cache_options():
    for module in (benchmark, decoder_benchmark):
        for arguments, expected in (
            ([], None),
            (["--validate-caches", "mega"], "mega"),
            (["--validate-caches", "mega", "--no-torch-compile"], "mega"),
            (["--validate-caches", "mega", "--dtype", "fp32"], "mega"),
            (["--validate-caches", "mega", "--dtype", "fp32", "--allow-tf32"], "mega"),
            (["--validate-caches", "torch-compiled"], "torch-compiled"),
        ):
            with patch.object(sys, "argv", ["benchmark"] + arguments):
                assert module.parse_args().validate_caches == expected
        invalid = [["--validate-caches", "torch-compiled", "--no-torch-compile"]]
        if module is decoder_benchmark:
            invalid += [
                ["--validate-caches", "mega", "--no-fused"],
                ["--validate-caches", "mega", "--dtype", "fp16"],
            ]
        for arguments in invalid:
            with patch.object(sys, "argv", ["benchmark"] + arguments):
                try:
                    module.parse_args()
                except SystemExit as error:
                    assert error.code == 2
                else:
                    raise AssertionError("Cache validation accepted a disabled candidate")


def check_decoder_chunks(device):
    eager_calls, candidate_calls = [], []
    references = []

    def decode(chunk, cache, first_chunk, calls):
        assert all(ref() is None for ref in references), "Previous candidate chunk remained alive"
        index = len(calls)
        assert first_chunk == (index == 0)
        assert chunk.shape[2] == 1
        torch.testing.assert_close(chunk, torch.full_like(chunk, 2.0))
        assert cache[1] == (None if first_chunk else "sentinel")
        if not first_chunk:
            torch.testing.assert_close(cache[0], torch.full_like(chunk, 2.0 * index))
        calls.append(first_chunk)
        value = chunk.clone() if first_chunk else chunk + cache[0]
        output = value.repeat(1, 1, 1 if first_chunk else 4, 1, 1)
        return output, (value.clone(), "sentinel")

    def candidate(chunk, cache, first_chunk):
        output, state = decode(chunk, cache, first_chunk, candidate_calls)
        references[:] = [weakref.ref(output), weakref.ref(state[0])]
        return output, state

    eager = SimpleNamespace(
        use_tiling=False,
        post_quant_conv=lambda latent: latent + 1,
        new_cache=lambda: (None, None),
        decode_chunk=lambda chunk, cache, first: decode(chunk, cache, first, eager_calls),
    )
    latent = torch.ones((2, 3, 3, 4, 6), dtype=torch.bfloat16, device=device)
    assert decoder_benchmark.validate_decoder_caches(eager, candidate, latent, atol=0.05, rtol=0.02, label="Mega")
    assert eager_calls == candidate_calls == [True, False, False]
    assert all(ref() is None for ref in references)


def check_oom_recovery():
    case = benchmark.BenchmarkCase(720, 1280, 17, 16)
    vae = SimpleNamespace(clear_cache=lambda: None)
    models = {"Mega": object()}
    attempts = []

    def run_case(args, case, vae, device, actual_models):
        assert actual_models is models, "OOM retries rebuilt the model wrappers"
        attempts.append(case.batch_size)
        if case.batch_size > 4:
            raise torch.cuda.OutOfMemoryError("Simulated capacity limit")
        # Validation failures must survive a successful allocation retry.
        return benchmark.BenchmarkResult(case, {"Mega": 1.0}, "Mega cache FAIL")

    with (
        patch.object(benchmark, "run_case", side_effect=run_case),
        patch.object(torch.cuda, "empty_cache"),
        patch.object(torch.compiler, "reset") as reset,
    ):
        result = benchmark.run_case_with_retry(SimpleNamespace(batch_size=None), case, vae, torch.device("cpu"), models)
        assert attempts == [16, 8, 4] and result.case.batch_size == 4
        assert result.status == "Mega cache FAIL"
        assert reset.call_count == 2, "Successful cases discarded compiler state"
        attempts.clear()
        result = benchmark.run_case_with_retry(SimpleNamespace(batch_size=16), case, vae, torch.device("cpu"), models)
        assert attempts == [16] and result.case.batch_size == 16 and result.status == "OOM"
        assert reset.call_count == 3

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
            result = cache_tolerance_counts(a, b, atol=0.25, rtol=0.001, allow_bf16_neighbors=True)
            assert result == counts, (device, actual, expected, result, counts)

        a = torch.tensor([66.5], dtype=torch.bfloat16, device=device)
        b = torch.tensor([66.0], dtype=torch.bfloat16, device=device)
        # Mega uses strict atol/rtol for caches too; only compiled caches opt into neighbors.
        assert cache_tolerance_counts(a, b, atol=0.25, rtol=0.001) == (1, 0)
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
        check_cpu_storage_reuse(device)
        check_chunk_lifetimes(device)
        check_chunk_lifetimes(device, cache_nthwc=True)
        check_decoder_chunks(device)
        check_chunk_output_contract(device)
        print(f"Cache tolerance regression checks ({device}): PASS", flush=True)
    check_oom_recovery()
    check_cache_options()


if __name__ == "__main__":
    main()
