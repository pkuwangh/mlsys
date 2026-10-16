"""Profile warmed Qwen3-TTS SeedTTS requests; run with run_sglang.py --nsys."""

import argparse
import asyncio
import os
import sys
from dataclasses import replace
from pathlib import Path

OMNI_DIR = Path(__file__).resolve().parents[1] / "sglang-omni"
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(OMNI_DIR))


async def profile_requests(config, concurrencies):
    import torch
    from benchmarks.dataset.seedtts import load_seedtts_samples
    from benchmarks.eval.benchmark_tts_seedtts import run_tts_seedtts_benchmark
    from benchmarks.metrics.performance import print_speed_summary

    samples = load_seedtts_samples(config.meta, 2 * max(concurrencies), split="en")
    for concurrency in concurrencies:
        point = replace(config, concurrency=concurrency, output_dir=str(Path(config.output_dir) / f"c{concurrency}"))
        print(f"Warming up {concurrency} requests...", flush=True)
        warmup = await run_tts_seedtts_benchmark(
            replace(point, output_dir=str(Path(point.output_dir) / "warmup")),
            samples=samples[:concurrency],
            save_audio=False,
        )
        if warmup["summary"]["failed_requests"]:
            raise RuntimeError("Warmup failed; see speed_results.json")
        print(f"Starting benchmark NVTX range: concurrency={concurrency}", flush=True)
        with torch.cuda.nvtx.range("benchmark"):
            result = await run_tts_seedtts_benchmark(point, samples=samples[: 2 * concurrency])
        print_speed_summary(result["summary"], config.model, concurrency=concurrency)
        if result["summary"]["failed_requests"]:
            raise RuntimeError("Some profiled requests failed; see speed_results.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 8], help="Concurrency levels to profile")
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", help="CustomVoice checkpoint")
    args = parser.parse_args()
    if min(args.concurrencies) < 1:
        parser.error("concurrency must be positive")

    from benchmarks.benchmarker.utils import managed_omni_server
    from benchmarks.eval.benchmark_tts_seedtts import TtsSeedttsBenchmarkConfig

    output = REPO_ROOT / "tmp" / "qwen3_tts_profile"
    config = TtsSeedttsBenchmarkConfig(
        model=args.model,
        meta="zhaochenyang20/seed-tts-eval-arrow",
        port=18123,
        voice="Ryan",
        task_type="CustomVoice",
        voice_clone=False,
        stream=True,
        response_format="pcm",
        seed=42,
        warmup=0,
        output_dir=str(output),
    )
    with managed_omni_server(
        model_path=config.model,
        port=config.port,
        host=config.host,
        server_config=str(OMNI_DIR / "examples/configs/qwen3_tts_1_7b_customvoice.yaml"),
        max_running_requests=config.max_running_requests,
        cuda_graph_max_bs=config.cuda_graph_max_bs,
        log_file=output / "tts_server.log",
        timeout=float(os.environ.get("SGLANG_OMNI_STARTUP_TIMEOUT", "1200")),
        wait_for_gpu_release=False,
    ):
        asyncio.run(profile_requests(config, args.concurrencies))


if __name__ == "__main__":
    main()
