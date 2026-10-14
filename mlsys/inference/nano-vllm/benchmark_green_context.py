"""Measure nano-vLLM prefill/decode on a full GPU or a CUDA green-context SM partition.

    ./run_nano.py mlsys/inference/nano-vllm/benchmark_green_context.py --sm-fraction 1
    ./run_nano.py mlsys/inference/nano-vllm/benchmark_green_context.py --sm-fraction 0.5

Add --nsys before the script and --profile after it to capture a separate warmed run.
"""

import argparse
import atexit
import json
import math
from pathlib import Path
from random import Random
from statistics import median
from time import perf_counter


def run_batch(llm, args, seed):
    """Time completed scheduler steps, keeping prefill and decode accounting separate."""
    import torch
    from nanovllm import SamplingParams

    rng = Random(seed)
    vocab_size = llm.model_runner.config.hf_config.vocab_size
    params = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=args.output_tokens)
    # Fresh, distinct prompts avoid prefix-cache reuse across requests and repeats.
    for _ in range(args.batch_size):
        llm.add_request([rng.randrange(vocab_size) for _ in range(args.input_tokens)], params)
    torch.manual_seed(seed)
    torch.cuda.synchronize()
    prefill_seconds = decode_seconds = 0.0
    prefill_tokens = decode_tokens = generated_tokens = 0
    decode_steps = []
    start = perf_counter()
    while not llm.is_finished():
        step_start = perf_counter()
        outputs, tokens = llm.step()
        # ModelRunner.run() copies sampled tokens to the CPU before returning.
        elapsed = perf_counter() - step_start
        if tokens > 0:
            prefill_seconds += elapsed
            prefill_tokens += tokens
        else:
            decode_seconds += elapsed
            decode_tokens -= tokens
            decode_steps.append(elapsed * 1000)
        generated_tokens += sum(len(ids) for _, ids in outputs)
    torch.cuda.synchronize()
    elapsed = perf_counter() - start
    if prefill_tokens != args.batch_size * args.input_tokens:
        raise RuntimeError("Unexpected cached or recomputed prompt tokens; comparison would be misleading")
    if generated_tokens != args.batch_size * args.output_tokens:
        raise RuntimeError("Generation did not produce the requested output length")
    return {
        "elapsed_ms": elapsed * 1000,
        "prefill_ms": prefill_seconds * 1000,
        "decode_ms": decode_seconds * 1000,
        "decode_step_ms": median(decode_steps),
        "prefill_tokens": prefill_tokens,
        "decode_tokens": decode_tokens,
        "generated_tokens": generated_tokens,
        "prefill_tok_s": prefill_tokens / prefill_seconds,
        "decode_tok_s": decode_tokens / decode_seconds,
        "output_tok_s": generated_tokens / elapsed,
    }


def parse_args():
    """Parse workload and partition settings before importing GPU dependencies."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=Path, default=Path("~/huggingface/Qwen3-14B"), help="Local Qwen3 checkpoint")
    parser.add_argument(
        "--sm-fraction", type=float, default=1.0, help="Requested SM fraction; 1 uses the normal context"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Concurrent requests (TP1)")
    parser.add_argument("--input-tokens", type=int, default=1024, help="Prompt tokens per request")
    parser.add_argument("--output-tokens", type=int, default=128, help="Output tokens per request, ignoring EOS")
    parser.add_argument("--repeats", type=int, default=3, help="Unprofiled timing samples after warmup")
    parser.add_argument("--profile", action="store_true", help="Capture one additional batch with CUDA profiler APIs")
    args = parser.parse_args()
    if not 0 < args.sm_fraction <= 1:
        parser.error("--sm-fraction must be in (0, 1]")
    if not 1 <= args.batch_size <= 512:
        parser.error("--batch-size must be in [1, 512]")
    if args.input_tokens < 1 or args.output_tokens < 2 or args.repeats < 1:
        parser.error("input tokens and repeats must be positive; output tokens must be at least 2")
    return args


def main():
    args = parse_args()

    import torch
    from nanovllm import LLM

    torch.cuda.set_device(0)
    device = torch.cuda.get_device_properties(0)
    requested_sms = math.ceil(device.multi_processor_count * args.sm_fraction)
    llm = LLM(
        str(args.model.expanduser().resolve()),
        num_sms=requested_sms if args.sm_fraction < 1 else None,
        max_num_seqs=args.batch_size,
        max_model_len=args.input_tokens + args.output_tokens,
        max_num_batched_tokens=16384,
        enforce_eager=False,
    )
    try:
        actual_sms = device.multi_processor_count
        runner = llm.model_runner
        if runner.execution_stream is not None:
            from cuda.bindings import driver

            status, resource = driver.cuStreamGetDevResource(
                driver.CUstream(runner.execution_stream.cuda_stream),
                driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM,
            )
            if status != driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"Cannot query green-context SM allocation: {status}")
            actual_sms = resource.sm.smCount
        config = runner.config
        if config.max_model_len < args.input_tokens + args.output_tokens:
            raise ValueError("Requested sequence length exceeds the model's context length")
        required_blocks = args.batch_size * math.ceil(config.max_model_len / config.kvcache_block_size)
        if config.num_kvcache_blocks < required_blocks:
            raise RuntimeError("Not enough KV cache for this batch without preemption; reduce --batch-size")
        print(
            f"{device.name} | requested SMs={requested_sms} actual SMs={actual_sms}/{device.multi_processor_count} "
            f"| batch={args.batch_size} input={args.input_tokens} output={args.output_tokens}",
            flush=True,
        )
        print("Warming up the full workload...", flush=True)
        run_batch(llm, args, seed=0)
        samples = []
        for i in range(args.repeats):
            result = run_batch(llm, args, seed=i + 1)
            samples.append(result)
            print(
                f"sample={i + 1} prefill={result['prefill_ms']:.2f} ms decode={result['decode_ms']:.2f} ms "
                f"decode_step={result['decode_step_ms']:.3f} ms elapsed={result['elapsed_ms']:.2f} ms",
                flush=True,
            )
        summary = {key: median(sample[key] for sample in samples) for key in samples[0]}
        print(
            f"Median: prefill={summary['prefill_tok_s']:.1f} tok/s decode={summary['decode_tok_s']:.1f} tok/s "
            f"overall={summary['output_tok_s']:.1f} output tok/s",
            flush=True,
        )
        output_dir = Path(__file__).resolve().parent / "output/green-context"
        output_dir.mkdir(parents=True, exist_ok=True)
        mode = f"sm{actual_sms}" if runner.green_context is not None else "full"
        output = (
            output_dir / f"{args.model.name}-{mode}-b{args.batch_size}-i{args.input_tokens}-o{args.output_tokens}.json"
        )
        output.write_text(
            json.dumps(
                {
                    "model": config.model,
                    "gpu": device.name,
                    "torch": torch.__version__,
                    "dtype": str(config.hf_config.dtype),
                    "sm_fraction": args.sm_fraction,
                    "requested_sms": requested_sms,
                    "actual_sms": actual_sms,
                    "total_sms": device.multi_processor_count,
                    "batch_size": args.batch_size,
                    "input_tokens": args.input_tokens,
                    "output_tokens": args.output_tokens,
                    "max_num_batched_tokens": config.max_num_batched_tokens,
                    "kv_cache_blocks": config.num_kvcache_blocks,
                    "samples": samples,
                    "median": summary,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"Saved {output}", flush=True)
        if args.profile:
            print("Capturing a separate warmed batch...", flush=True)
            torch.cuda.profiler.start()
            try:
                with torch.cuda.nvtx.range(f"nano_{mode}_b{args.batch_size}"):
                    run_batch(llm, args, seed=args.repeats + 1)
            finally:
                torch.cuda.profiler.stop()
    finally:
        atexit.unregister(llm.exit)
        llm.exit()


if __name__ == "__main__":
    main()
