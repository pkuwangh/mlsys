"""Benchmark captioning with a local video and an already-running vLLM server.

From mlsys/inference/vllm, run these in separate terminals:
    bash benchmark/launch-vllm-serve.sh
    python benchmark/benchmark_qwen3p8_27b_client.py --video data-samples/sample1.mp4

The client sends a file:// URL; the server must see the same absolute path under
its allowed media directory. All video processing happens on the server.
Timing excludes warmup but includes HTTP, server queueing, and video processing.
Repeating one video may benefit from server prefix/multimodal caches.
"""

import argparse
import asyncio
import math
import statistics
import time
from pathlib import Path

from openai import AsyncOpenAI

BASE_URL = "http://127.0.0.1:8089/v1"
CONCURRENCY = 64
NUM_REQUESTS = 192
WARMUP_REQUESTS = 64
TIMEOUT = 1800
CAPTION_PROMPT_PATH = Path(
    "/home/haowan/scratch/imaginaire4-1/pipelines/video/caption/caption_v2/"
    "prompts/gpt55_cap_balance_qa_carla_affirmative_dedup_v2.txt"
)
SAMPLING_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.8,
    "max_tokens": 8192,
    "presence_penalty": 1.5,
    "extra_body": {
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "chat_template_kwargs": {"enable_thinking": False},
    },
}


async def run_batch(
    client, model: str, messages: list, count: int, sampling_params: dict
):
    """Send concurrent requests, measuring latency from each HTTP call."""
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def request():
        async with semaphore:
            started = time.perf_counter()
            response = await client.chat.completions.create(
                model=model, messages=messages, **sampling_params
            )
            if response.usage is None:
                raise ValueError("Server response is missing token usage")
            return response, time.perf_counter() - started

    started = time.perf_counter()
    results = await asyncio.gather(*(request() for _ in range(count)))
    return results, time.perf_counter() - started


async def run(video: Path, max_tokens: int) -> None:
    """Warm up, benchmark repeated caption requests, and print one caption."""
    sampling_params = {**SAMPLING_PARAMS, "max_tokens": max_tokens}
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": video.as_uri()}},
                {"type": "text", "text": CAPTION_PROMPT_PATH.read_text().strip()},
            ],
        }
    ]
    async with AsyncOpenAI(
        base_url=BASE_URL, api_key="EMPTY", timeout=TIMEOUT, max_retries=0
    ) as client:
        models = (await client.models.list()).data
        if len(models) != 1:
            raise ValueError("Expected exactly one served model")
        model = models[0].id
        print(
            f"model={model} video={video} requests={NUM_REQUESTS} concurrency={CONCURRENCY} max_tokens={max_tokens}",
            flush=True,
        )
        print(f"Warming up with {WARMUP_REQUESTS} requests...", flush=True)
        await run_batch(client, model, messages, WARMUP_REQUESTS, sampling_params)
        print("Starting measured requests...", flush=True)
        results, elapsed = await run_batch(
            client, model, messages, NUM_REQUESTS, sampling_params
        )

    generated_tokens = sum(response.usage.completion_tokens for response, _ in results)
    prompt_tokens = sum(response.usage.prompt_tokens for response, _ in results)
    latencies = sorted(latency for _, latency in results)
    print(
        f"requests={len(results)} prompt_tokens={prompt_tokens} "
        f"generated_tokens={generated_tokens} elapsed={elapsed:.3f}s "
        f"tok/s={generated_tokens / elapsed:.1f}"
    )
    print(
        f"avg_input_tokens={prompt_tokens / len(results):.1f} "
        f"avg_output_tokens={generated_tokens / len(results):.1f} "
        f"latency_p50={statistics.median(latencies):.3f}s "
        f"latency_p95={latencies[math.ceil(0.95 * len(latencies)) - 1]:.3f}s"
    )
    print("--- first output ---")
    print(results[0][0].choices[0].message.content or "")
    print("--- end first output ---")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--video", type=Path, required=True, help="Local video file")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=SAMPLING_PARAMS["max_tokens"],
        help="Maximum output tokens per request",
    )
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    video = args.video.expanduser().resolve()
    if not video.is_file():
        parser.error(f"Video not found: {video}")
    asyncio.run(run(video, args.max_tokens))


if __name__ == "__main__":
    main()
