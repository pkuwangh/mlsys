"""Benchmark independent prefill/decode queues sharing one Qwen3 model and KV cache.

    ./run_nano.py mlsys/inference/nano-vllm/benchmark_concurrent.py
    ./run_nano.py mlsys/inference/nano-vllm/benchmark_concurrent.py --mode full

Add --nsys before the script and --profile after it for a separate warmed capture.
"""

import argparse
import atexit
import json
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from random import Random
from statistics import median


def steady_state(result):
    """Measure between the 25% and 75% request-completion boundaries."""
    completed = result["completed_ms"]
    start = completed[len(completed) // 4 - 1]
    end = completed[3 * len(completed) // 4 - 1]
    steps = [s for s in result["steps"] if start < s["collected_ms"] <= end]
    seconds = (end - start) / 1000
    summary = {
        "steady_start_ms": start,
        "steady_end_ms": end,
        "steady_seconds": seconds,
        "steady_output_tok_s": sum(s["output_tokens"] for s in steps) / seconds,
        "steady_requests_s": sum(s["completed_requests"] for s in steps) / seconds,
        "steady_prefill_tok_s": sum(s["tokens"] for s in steps if s["phase"] == "prefill") / seconds,
    }
    for phase in ("prefill", "decode"):
        phase_steps = [s for s in steps if s["phase"] == phase]
        summary[f"steady_{phase}_batch_mean"] = (
            sum(s["batch_size"] for s in phase_steps) / len(phase_steps) if phase_steps else 0
        )
    return summary


def summarize(result):
    """Summarize completed stream intervals, including host submission gaps."""
    intervals = {}
    summary = {"elapsed_ms": result["elapsed_ms"], "output_tok_s": result["output_tok_s"]}
    for phase in ("prefill", "decode"):
        steps = [s for s in result["steps"] if s["phase"] == phase]
        intervals[phase] = sorted((s["start_ms"], s["end_ms"]) for s in steps)
        summary[f"{phase}_stream_ms"] = sum(end - start for start, end in intervals[phase])
        summary[f"{phase}_steps"] = len(steps)
        summary[f"{phase}_tokens"] = sum(s["tokens"] for s in steps)
        summary[f"{phase}_submit_median_ms"] = median(s["submit_ms"] for s in steps) if steps else 0
        summary[f"{phase}_submit_max_ms"] = max((s["submit_ms"] for s in steps), default=0)
    p = d = 0
    overlap = 0.0
    while p < len(intervals["prefill"]) and d < len(intervals["decode"]):
        ps, pe = intervals["prefill"][p]
        ds, de = intervals["decode"][d]
        overlap += max(0, min(pe, de) - max(ps, ds))
        if pe < de:
            p += 1
        else:
            d += 1
    summary["stream_overlap_ms"] = overlap
    summary["ttft_median_ms"] = median(result["ttft_ms"])
    summary.update(steady_state(result))
    return summary


def validate_handoff(executor, rng, max_output_tokens):
    """Check chunk continuation, cache recycling, and first-token-only completion."""
    config = executor.config
    old_budget = config.max_num_batched_tokens
    config.max_num_batched_tokens = min(old_budget, 128)
    try:
        output_tokens = min(4, max_output_tokens)
        length = min(257, config.max_model_len - output_tokens)
        prompts = [
            [rng.randrange(config.hf_config.vocab_size) for _ in range(length - i)] for i in range(min(3, length))
        ]
        for outputs in sorted({1, output_tokens}):
            reference, _ = executor.generate(prompts, outputs, concurrent=False, active_requests=2)
            actual, _ = executor.generate(prompts, outputs, concurrent=True, active_requests=2)
            if actual != reference:
                raise AssertionError("Concurrent greedy outputs differ from serialized execution")
    finally:
        config.max_num_batched_tokens = old_budget


def parse_args():
    """Parse workload and partition settings before importing GPU dependencies."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=Path, default=Path("~/huggingface/Qwen3-14B"), help="Local checkpoint")
    parser.add_argument(
        "--mode",
        choices=("compare", "full", "serial", "concurrent"),
        default="compare",
        help="Compare alternates serial/concurrent on the same partitions; full uses all SMs",
    )
    parser.add_argument(
        "--prefill-sm-fraction", type=float, default=0.5, help="Prefill share; decode gets the remainder"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Maximum sequences per path submission")
    parser.add_argument("--requests", type=int, default=512, help="Total distinct requests, including startup/drain")
    parser.add_argument(
        "--active-requests", type=int, default=64, help="Outstanding requests; refill as requests finish"
    )
    parser.add_argument("--input-tokens", type=int, default=1024, help="Input tokens per request")
    parser.add_argument("--output-tokens", type=int, default=128, help="Greedy output tokens per request, ignoring EOS")
    parser.add_argument("--prefill-token-budget", type=int, default=16384, help="Maximum tokens per prefill submission")
    parser.add_argument("--repeats", type=int, default=3, help="Unprofiled samples per mode")
    parser.add_argument("--profile", action="store_true", help="Capture one additional warmed batch per mode")
    args = parser.parse_args()
    if not 0 < args.prefill_sm_fraction < 1:
        parser.error("--prefill-sm-fraction must be between 0 and 1")
    if not 1 <= args.batch_size <= 512:
        parser.error("--batch-size must be between 1 and 512")
    if (
        min(
            args.requests,
            args.active_requests,
            args.input_tokens,
            args.output_tokens,
            args.prefill_token_budget,
            args.repeats,
        )
        < 1
    ):
        parser.error("Counts and token budgets must be positive")
    if args.requests < 4 * args.active_requests:
        parser.error("Use at least 4 * --active-requests to leave room for startup/drain")
    return args


def main():
    args = parse_args()

    import torch
    from nanovllm import LLM
    from nanovllm.engine.concurrent import ConcurrentExecutor

    # Decode graph buckets, validation, and partial prefills specialize the same functions.
    torch._dynamo.config.recompile_limit = 64
    llm = LLM(
        str(args.model.expanduser().resolve()),
        max_num_seqs=args.batch_size,
        max_model_len=args.input_tokens + args.output_tokens,
        max_num_batched_tokens=args.prefill_token_budget,
        prefill_sm_fraction=None if args.mode == "full" else args.prefill_sm_fraction,
    )
    try:
        runner = llm.model_runner
        executor = ConcurrentExecutor(runner)
        generate = partial(executor.generate, max_tokens=args.output_tokens, active_requests=args.active_requests)
        device = torch.cuda.get_device_properties(0)
        counts = runner.partitions.sm_counts if runner.partitions else (device.multi_processor_count,) * 2
        print(
            f"{device.name} | prefill/decode SMs={counts} | requests={args.requests} "
            f"active={args.active_requests} batch={args.batch_size} input={args.input_tokens} output={args.output_tokens}",
            flush=True,
        )
        modes = ["serial", "concurrent"] if args.mode == "compare" else [args.mode]
        rng = Random(0)
        vocab = runner.config.hf_config.vocab_size
        prompts = [[rng.randrange(vocab) for _ in range(args.input_tokens)] for _ in range(args.requests)]

        validation = {}
        if runner.partitions:
            validate_handoff(executor, rng, args.output_tokens)
            validation["chunked_handoff_and_greedy_parity"] = "PASS"
            print("Validation: chunked prefill, handoff, block growth, greedy parity PASS", flush=True)

        print("Warming up with two active-request windows...", flush=True)
        for mode in modes:
            generate(prompts[: 2 * args.active_requests], concurrent=mode == "concurrent")
        samples = {mode: [] for mode in modes}
        last_outputs = {}
        for repeat in range(args.repeats):
            order = modes if repeat % 2 == 0 else list(reversed(modes))
            for mode in order:
                outputs, result = generate(prompts, concurrent=mode == "concurrent")
                summary = summarize(result)
                result["summary"] = summary
                samples[mode].append(result)
                last_outputs[mode] = outputs
                print(
                    f"{mode} sample={repeat + 1}: {summary['elapsed_ms']:.2f} ms | "
                    f"{summary['output_tok_s']:.1f} output tok/s | "
                    f"stream overlap={summary['stream_overlap_ms']:.2f} ms | "
                    f"prefill host submit={summary['prefill_submit_median_ms']:.2f} ms",
                    flush=True,
                )
                print(
                    f"  steady {summary['steady_seconds']:.2f}s: {summary['steady_output_tok_s']:.1f} output tok/s | "
                    f"{summary['steady_requests_s']:.2f} req/s | {summary['steady_prefill_tok_s']:.1f} input tok/s | "
                    f"mean decode batch={summary['steady_decode_batch_mean']:.1f}",
                    flush=True,
                )
        if len(modes) == 2:
            mismatches = sum(a != b for a, b in zip(last_outputs["serial"], last_outputs["concurrent"]))
            validation["workload_request_mismatches"] = mismatches
            print(f"Workload greedy parity: {args.requests - mismatches}/{args.requests} requests match", flush=True)
        medians = {}
        for mode, results in samples.items():
            summaries = [result["summary"] for result in results]
            medians[mode] = {key: median(s[key] for s in summaries) for key in summaries[0]}
            print(
                f"{mode} median: {medians[mode]['steady_output_tok_s']:.1f} steady output tok/s | "
                f"{medians[mode]['output_tok_s']:.1f} overall output tok/s",
                flush=True,
            )
        output_dir = Path(__file__).resolve().parent / "output/concurrent"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        path = output_dir / (
            f"{args.model.name}-{args.mode}-p{counts[0]}-d{counts[1]}-b{args.batch_size}-q{args.requests}-{timestamp}.json"
        )
        path.write_text(
            json.dumps(
                {
                    "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    "gpu": device.name,
                    "torch": torch.__version__,
                    "dtype": str(runner.config.hf_config.dtype),
                    "sm_counts": counts,
                    "kv_cache_blocks": runner.config.num_kvcache_blocks,
                    "validation": validation,
                    "median": medians,
                    "samples": samples,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"Saved {path}", flush=True)
        if args.profile:
            torch.cuda.profiler.start()
            try:
                for mode in modes:
                    with torch.cuda.nvtx.range(f"{mode}_p{counts[0]}_d{counts[1]}"):
                        generate(prompts, concurrent=mode == "concurrent")
            finally:
                torch.cuda.profiler.stop()
    finally:
        atexit.unregister(llm.exit)
        llm.exit()


if __name__ == "__main__":
    main()
