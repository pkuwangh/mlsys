"""Measure batch-1 Qwen3.8-27B latency with 1,024 input and output tokens.

    python benchmark/benchmark_qwen3p8_27b_latency.py --config tp1

    VLLM_WORKER_MULTIPROC_METHOD=spawn nsys profile \
        --gpu-metrics-devices=0 \
        --trace=cuda,nvtx --trace-fork-before-exec=true \
        --cuda-graph-trace=node --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        python benchmark/benchmark_qwen3p8_27b_latency.py --profile
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time

from utils import get_model_path

REPEATS = 3


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", choices=["tp1", "tp2"], default="tp1", help="Tensor parallel configuration")
    parser.add_argument("--model", default=get_model_path("Qwen/Qwen3.8-27B"), help="Target model path")
    parser.add_argument("--input-tokens", type=int, default=1024, help="Input token count")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Output token count (EOS ignored)")
    parser.add_argument("--profile", action="store_true", help="Capture one additional request after timing")
    parser.add_argument(
        "--linear-backend", choices=["auto", "flashinfer_cutedsl"], default="auto", help="BF16 GEMM backend"
    )
    args = parser.parse_args()
    if args.input_tokens <= 0 or args.max_tokens <= 0:
        parser.error("--input-tokens and --max-tokens must be positive")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=1 if args.config == "tp1" else 2,
        max_model_len=args.input_tokens + args.max_tokens,
        max_num_seqs=1,
        max_num_batched_tokens=args.input_tokens,
        gpu_memory_utilization=0.85,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        disable_log_stats=False,
        speculative_config=None,
        linear_backend=args.linear_backend,
        profiler_config={"profiler": "cuda"} if args.profile else None,
    )
    seed_tokens = llm.get_tokenizer().encode(" the", add_special_tokens=False)
    prompt_tokens = (seed_tokens * (args.input_tokens // len(seed_tokens) + 1))[: args.input_tokens]
    requests = [{"prompt_token_ids": prompt_tokens}]
    params = SamplingParams(
        temperature=0.0,
        min_tokens=args.max_tokens,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )

    print(
        f"model={args.model} config={args.config} linear_backend={args.linear_backend} batch=1 "
        f"input_tokens={args.input_tokens} output_tokens={args.max_tokens}",
        flush=True,
    )
    print("Warming up with one request...", flush=True)
    llm.generate(requests, params, use_tqdm=False)
    print("Warmup complete. Starting benchmark...", flush=True)

    latencies_ms = []
    for iteration in range(REPEATS):
        started_at = time.perf_counter()
        output = llm.generate(requests, params, use_tqdm=False)[0]
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        generated_tokens = len(output.outputs[0].token_ids)
        if generated_tokens != args.max_tokens:
            raise RuntimeError(f"Expected {args.max_tokens} output tokens, got {generated_tokens}")
        latencies_ms.append(elapsed_ms)
        print(f"run={iteration + 1} generated_tokens={generated_tokens} latency_ms={elapsed_ms:.3f}")
        token_hash = hashlib.sha256(json.dumps(output.outputs[0].token_ids).encode()).hexdigest()
        print(f"  output_token_sha256={token_hash}")
        metrics = output.metrics
        if metrics is not None:
            # Engine timestamps include scheduling effects, not just GPU kernel time.
            decode_ms = (metrics.last_token_ts - metrics.first_token_ts) * 1000
            print(f"  ttft_ms={metrics.first_token_latency * 1000:.3f} decode_ms={decode_ms:.3f}")
            if generated_tokens > 1:
                print(f"  mean_decode_token_ms={decode_ms / (generated_tokens - 1):.3f}")

    median_ms = statistics.median(latencies_ms)
    print(
        f"batch=1 median_latency_ms={median_ms:.3f} "
        f"min_ms={min(latencies_ms):.3f} max_ms={max(latencies_ms):.3f} "
        f"output_tok/s={args.max_tokens / (median_ms / 1000):.1f}",
        flush=True,
    )

    if args.profile:
        print("Profiling one additional request (excluded from timing)...", flush=True)
        llm.start_profile()
        try:
            llm.generate(requests, params, use_tqdm=False)
        finally:
            llm.stop_profile()


if __name__ == "__main__":
    main()
