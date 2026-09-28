"""Profile a synthetic Qwen3.5 text-only decode workload.

    VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_NVTX_SCOPES_FOR_PROFILING=1 \
    nsys profile \
        --gpu-metrics-devices=0 \
        --trace=cuda,nvtx,osrt,cublas,cudnn \
        --trace-fork-before-exec=true \
        --cuda-graph-trace=node \
        --capture-range=cudaProfilerApi \
        --capture-range-end=repeat \
    python benchmark/benchmark_qwen3p8_27b_synthetic.py \
      --config tp1 \
      --spec-decode none
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from utils import get_model_path

_PROFILE_DELAY_FRACTION = 0.0
_PROFILE_MAX_INTERATIONS = 21

# _TARGET_MODEL = "video_captioner/qwen35-27b-exp3-gpt55-seed21-general500k-yatian500k-fps2-framewise-step16193-20260812074816-vllm-v1-patch-mtp"
_TARGET_MODEL = "Qwen/Qwen3.8-27B"

_MTP_SPEC_TOKENS = 1

# _DFLASH_MODEL= "z-lab/Qwen3.5-27B-DFlash"
# _DFLASH_SPEC_TOKENS = 8
_DFLASH_MODEL= "z-lab/Qwen3.8-27B-DFlash2"
_DFLASH_SPEC_TOKENS = 7

_DSPARK_MODEL = "RadixArk/Qwen3.8-27B-DSpark"
_DSPARK_SPEC_TOKENS = 7

# Production Config.
CONFIGS = {
    "tp1": {
        "TENSOR_PARALLEL_SIZE": 1,
        "MAX_NUM_SEQS": 64,
        "INPUT_TOKENS": 28_672,
        "OUTPUT_TOKENS": 3_000,
        "MAX_MODEL_LEN": 65_536,
        "MAX_NUM_BATCHED_TOKENS": 131_072,
        "GPU_MEMORY_UTILIZATION": 0.85,

    },
    "tp2": {
        "TENSOR_PARALLEL_SIZE": 2,
        "MAX_NUM_SEQS": 64,
        "INPUT_TOKENS": 28_672,
        "OUTPUT_TOKENS": 3_000,
        "MAX_MODEL_LEN": 65_536,
        "MAX_NUM_BATCHED_TOKENS": 131_072,
        "GPU_MEMORY_UTILIZATION": 0.85,
    },
}


def read_spec_decode_metrics(llm, num_speculative_tokens: int):
    from vllm.v1.metrics.reader import Counter, Vector

    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0
    accepted_tokens_per_pos = [0] * num_speculative_tokens

    for metric in llm.get_metrics():
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_draft_tokens":
            assert isinstance(metric, Counter)
            num_draft_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            assert isinstance(metric, Counter)
            num_accepted_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for pos, count in enumerate(metric.values):
                accepted_tokens_per_pos[pos] += count

    return (
        num_drafts,
        num_draft_tokens,
        num_accepted_tokens,
        accepted_tokens_per_pos,
    )


def print_spec_decode_metrics(before, after) -> None:
    num_drafts = after[0] - before[0]
    num_draft_tokens = after[1] - before[1]
    num_accepted_tokens = after[2] - before[2]
    accepted_tokens_per_pos = [
        end - start for start, end in zip(before[3], after[3])
    ]

    mean_acceptance_length = (
        1 + num_accepted_tokens / num_drafts if num_drafts else 1.0
    )
    draft_acceptance_rate = (
        num_accepted_tokens / num_draft_tokens if num_draft_tokens else 0.0
    )
    per_position_rates = [
        count / num_drafts if num_drafts else 0.0
        for count in accepted_tokens_per_pos
    ]

    print(
        "spec_decode: "
        f"drafts={num_drafts} drafted_tokens={num_draft_tokens} "
        f"accepted_tokens={num_accepted_tokens} "
        f"mean_acceptance_length={mean_acceptance_length:.2f} "
        f"draft_acceptance_rate={draft_acceptance_rate:.1%}"
    )
    print(
        "spec_decode_per_position_acceptance="
        + ", ".join(f"{rate:.1%}" for rate in per_position_rates)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        choices=["tp1", "tp2"],
        default="tp1",
        help="Benchmark configuration",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        help="Override the selected configuration's maximum number of sequences",
    )
    parser.add_argument(
        "--spec-decode",
        choices=["none", "mtp", "dflash", "dspark"],
        default="none",
        help="Speculative decoding method",
    )
    parser.add_argument(
        "--dflash-model",
        default=get_model_path(_DFLASH_MODEL),
        help="Path to the DFlash/DFlash2 draft model",
    )
    parser.add_argument(
        "--dspark-model",
        default=get_model_path(_DSPARK_MODEL),
        help="Path to the DSpark draft model",
    )
    parser.add_argument(
        "--profile", action="store_true", help="Enable profiling with nsys"
    )
    args = parser.parse_args()
    if args.max_num_seqs is not None and args.max_num_seqs <= 0:
        parser.error("--max-num-seqs must be positive")

    config = CONFIGS[args.config]
    max_num_seqs = (
        args.max_num_seqs
        if args.max_num_seqs is not None
        else config["MAX_NUM_SEQS"]
    )
    max_num_batched_tokens = min(
        config["MAX_NUM_BATCHED_TOKENS"],
        max_num_seqs * config["MAX_MODEL_LEN"],
    )
    model_path = get_model_path(_TARGET_MODEL)
    speculative_config: dict[str, object] | None = None
    attention_config: dict[str, object] | None = None
    num_speculative_tokens = 0
    if args.spec_decode == "mtp":
        num_speculative_tokens = _MTP_SPEC_TOKENS
        speculative_config = {
            "method": "mtp",
            "num_speculative_tokens": num_speculative_tokens,
        }
    elif args.spec_decode == "dflash":
        if not Path(args.dflash_model).exists():
            parser.error(f"DFlash/DFlash2 model not found at {args.dflash_model!r}")
        num_speculative_tokens = _DFLASH_SPEC_TOKENS
        speculative_config = {
            "method": "dflash",
            "model": args.dflash_model,
            "num_speculative_tokens": num_speculative_tokens,
            "max_model_len": config["MAX_MODEL_LEN"],
            "attention_backend": "FLASHINFER",
        }
        attention_config = {"backend": "FLASHINFER"}
    elif args.spec_decode == "dspark":
        if not Path(args.dspark_model).exists():
            parser.error(f"DSpark model not found at {args.dspark_model!r}")
        num_speculative_tokens = _DSPARK_SPEC_TOKENS
        speculative_config = {
            "method": "dspark",
            "model": args.dspark_model,
            "num_speculative_tokens": num_speculative_tokens,
            "max_model_len": config["MAX_MODEL_LEN"],
            "attention_backend": "FLASHINFER",
        }
        attention_config = {"backend": "FLASHINFER"}

    input_tokens = config["INPUT_TOKENS"]
    output_tokens = config["OUTPUT_TOKENS"]
    # A speculative step can emit the target token plus all accepted drafts.
    delay_iterations = int(output_tokens * _PROFILE_DELAY_FRACTION / (num_speculative_tokens + 1))

    import torch
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=config["TENSOR_PARALLEL_SIZE"],
        max_model_len=config["MAX_MODEL_LEN"],
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=config["GPU_MEMORY_UTILIZATION"],
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        disable_log_stats=False,
        speculative_config=speculative_config,
        attention_config=attention_config,
        language_model_only=args.spec_decode in ("dflash", "dspark"),
        profiler_config={
            "profiler": "cuda",
            "delay_iterations": delay_iterations,
            "max_iterations": _PROFILE_MAX_INTERATIONS,
        },
    )

    tokenizer = llm.get_tokenizer()
    seed_tokens = tokenizer.encode(" the", add_special_tokens=False)
    prompt_tokens = (seed_tokens * (input_tokens // len(seed_tokens) + 1))[:input_tokens]
    requests = [{"prompt_token_ids": prompt_tokens} for _ in range(max_num_seqs)]

    params = SamplingParams(
        temperature=0.0,
        min_tokens=output_tokens,
        max_tokens=output_tokens,
        ignore_eos=True,
    )
    # warmup
    _ = llm.generate(requests, params, use_tqdm=False)

    metrics_before = (
        read_spec_decode_metrics(llm, num_speculative_tokens)
        if speculative_config is not None
        else None
    )
    if args.profile:
        llm.start_profile()
        torch.cuda.nvtx.range_push(
            f"synthetic_{input_tokens}_in_{output_tokens}_out_{config['MAX_NUM_SEQS']}_seqs"
        )
    started_at = time.perf_counter()
    try:
        outputs = llm.generate(requests, params, use_tqdm=False)
    finally:
        elapsed_seconds = time.perf_counter() - started_at
        if args.profile:
            torch.cuda.nvtx.range_pop()
            llm.stop_profile()

    generated_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    print(
        f"generated_tokens={generated_tokens} elapsed={elapsed_seconds:.3f}s tok/s={generated_tokens / elapsed_seconds:.1f}"
    )
    if metrics_before is not None:
        metrics_after = read_spec_decode_metrics(llm, num_speculative_tokens)
        print_spec_decode_metrics(metrics_before, metrics_after)


if __name__ == "__main__":
    main()
