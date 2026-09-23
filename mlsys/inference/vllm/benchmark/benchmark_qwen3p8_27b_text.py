"""Profile a synthetic Qwen3.5 text-only decode workload.

    VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_NVTX_SCOPES_FOR_PROFILING=1 \
    nsys profile \
        --trace=cuda,nvtx,osrt,cublas,cudnn \
        --trace-fork-before-exec=true \
        --cuda-graph-trace=node \
        --capture-range=cudaProfilerApi \
        --capture-range-end=repeat \
    python benchmark_qwen3p8_27b_text.py \
      --config tp1
"""

from __future__ import annotations

import argparse
import time

from utils import get_model_path

# Production Config.
CONFIGS = {
    "tp1": {
        "TENSOR_PARALLEL_SIZE": 1,
        "MAX_NUM_SEQS": 48,
        "INPUT_TOKENS": 27_300,
        "OUTPUT_TOKENS": 3_000,
        "MAX_MODEL_LEN": 49_152,
        "MAX_NUM_BATCHED_TOKENS": 65_536,
        "GPU_MEMORY_UTILIZATION": 0.8,

    },
    "tp2": {
        "TENSOR_PARALLEL_SIZE": 2,
        "MAX_NUM_SEQS": 64,
        "INPUT_TOKENS": 28_672,
        "OUTPUT_TOKENS": 4_096,
        "MAX_MODEL_LEN": 32_768,
        "MAX_NUM_BATCHED_TOKENS": 131_072,
        "GPU_MEMORY_UTILIZATION": 0.9,
    },
}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, choices=["tp1", "tp2"], default="tp1")
    args = parser.parse_args()

    import torch
    from vllm import LLM, SamplingParams

    model_path = get_model_path("Qwen/Qwen3.8-27B")
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=CONFIGS[args.config]["TENSOR_PARALLEL_SIZE"],
        max_model_len=CONFIGS[args.config]["MAX_MODEL_LEN"],
        max_num_batched_tokens=CONFIGS[args.config]["MAX_NUM_BATCHED_TOKENS"],
        max_num_seqs=CONFIGS[args.config]["MAX_NUM_SEQS"],
        gpu_memory_utilization=CONFIGS[args.config]["GPU_MEMORY_UTILIZATION"],
        enable_chunked_prefill=True,
        profiler_config={
            "profiler": "cuda",
            "delay_iterations": int(CONFIGS[args.config]["OUTPUT_TOKENS"] * 0.5),
            "max_iterations": 10,
        },
    )

    input_tokens = CONFIGS[args.config]["INPUT_TOKENS"]
    output_tokens = CONFIGS[args.config]["OUTPUT_TOKENS"]

    tokenizer = llm.get_tokenizer()
    seed_tokens = tokenizer.encode(" the", add_special_tokens=False)
    prompt_tokens = (seed_tokens * (input_tokens // len(seed_tokens) + 1))[:input_tokens]
    requests = [{"prompt_token_ids": prompt_tokens} for _ in range(CONFIGS[args.config]["MAX_NUM_SEQS"])]

    llm.generate(requests, SamplingParams(max_tokens=8, ignore_eos=True), use_tqdm=False)

    params = SamplingParams(
        temperature=0.0,
        min_tokens=output_tokens,
        max_tokens=output_tokens,
        ignore_eos=True,
    )
    # warmup
    _ = llm.generate(requests[0:2], params, use_tqdm=True)

    llm.start_profile()
    torch.cuda.nvtx.range_push("text_decode")
    started_at = time.perf_counter()
    try:
        outputs = llm.generate(requests, params, use_tqdm=True)
    finally:
        elapsed_seconds = time.perf_counter() - started_at
        torch.cuda.nvtx.range_pop()
        llm.stop_profile()

    generated_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    print(
        f"generated_tokens={generated_tokens} elapsed={elapsed_seconds:.3f}s tok/s={generated_tokens / elapsed_seconds:.1f}"
    )


if __name__ == "__main__":
    main()
