"""Profile Qwen3.5 captioning with frames sampled from sample1.mp4.

    nsys profile \
        --gpu-metrics-devices=0 \
        --trace=cuda,nvtx,osrt,cublas,cudnn \
        --trace-fork-before-exec=true \
        --cuda-graph-trace=node \
        --capture-range=cudaProfilerApi \
        --capture-range-end=repeat \
    python benchmark/benchmark_qwen3p8_27b_video.py \
      --config tp1 \
      --spec-decode none
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import av
import numpy as np
from utils import get_model_path

_PROFILE_DELAY_FRACTION = 0.5
_PROFILE_MAX_INTERATIONS = 10

_ENABLE_PREFIX_CACHING = False

# _TARGET_MODEL = "video_captioner/qwen35-27b-exp3-gpt55-seed21-general500k-yatian500k-fps2-framewise-step16193-20260812074816-vllm-v1-patch-mtp"
# _TARGET_MODEL = "Qwen/Qwen3.5-27B"
_TARGET_MODEL = "Qwen/Qwen3.8-27B"

_MTP_SPEC_TOKENS = 1

# _DFLASH_MODEL = "z-lab/Qwen3.5-27B-DFlash"
# _DFLASH_SPEC_TOKENS = 2
_DFLASH_MODEL= "z-lab/Qwen3.8-27B-DFlash2"
_DFLASH_SPEC_TOKENS = 7

_DSPARK_MODEL = "RadixArk/Qwen3.8-27B-DSpark"
_DSPARK_SPEC_TOKENS = 7

_CAPTION_PROMPT_PATH = Path(
    "/home/haowan/scratch/imaginaire4-1/pipelines/video/caption/caption_v2/"
    "prompts/gpt55_cap_balance_qa_carla_affirmative_dedup_v2.txt"
)

# Production configurations.
CONFIGS = {
    "tp1": {
        "TENSOR_PARALLEL_SIZE": 1,
        "MAX_NUM_SEQS": 64,
        "OUTPUT_TOKENS": 8_192,
        "MAX_MODEL_LEN": 65_536,
        "MAX_NUM_BATCHED_TOKENS": 131_072,
        "GPU_MEMORY_UTILIZATION": 0.85,
    },
    "tp2": {
        "TENSOR_PARALLEL_SIZE": 2,
        "MAX_NUM_SEQS": 64,
        "OUTPUT_TOKENS": 8_192,
        "MAX_MODEL_LEN": 65_536,
        "MAX_NUM_BATCHED_TOKENS": 131_072,
        "GPU_MEMORY_UTILIZATION": 0.85,
    },
}

VIDEO_FPS = 2
MIN_VIDEO_FRAMES = 4
MAX_VIDEO_FRAMES = 256
MIN_VIDEO_PIXELS = 4_096
MAX_VIDEO_PIXELS = 49_152_000


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
    accepted_tokens_per_pos = [end - start for start, end in zip(before[3], after[3])]

    mean_acceptance_length = 1 + num_accepted_tokens / num_drafts if num_drafts else 1.0
    draft_acceptance_rate = (
        num_accepted_tokens / num_draft_tokens if num_draft_tokens else 0.0
    )
    per_position_rates = [
        count / num_drafts if num_drafts else 0.0 for count in accepted_tokens_per_pos
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


def sample_frames(video_path: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Select the nearest source frame at each 0.5-second PTS target."""
    frames: list[np.ndarray] = []
    frame_indices: list[int] = []
    timestamps: list[float] = []
    previous: tuple[Any, int, float] | None = None
    first_pts: float | None = None
    next_target = 0.0
    total_num_frames = 0
    original_fps = 0.0

    with av.open(video_path) as container:
        stream = container.streams.video[0]
        if stream.average_rate is not None:
            original_fps = float(stream.average_rate)
        for frame_index, frame in enumerate(container.decode(stream)):
            total_num_frames = frame_index + 1
            if frame.pts is None or frame.time_base is None:
                continue
            absolute_pts = float(frame.pts * frame.time_base)
            if first_pts is None:
                first_pts = absolute_pts
            timestamp = absolute_pts - first_pts
            current = (frame, frame_index, timestamp)
            if previous is None:
                previous = current

            while timestamp >= next_target:
                selected = (
                    previous
                    if abs(previous[2] - next_target) <= abs(timestamp - next_target)
                    else current
                )
                if not frame_indices or selected[1] > frame_indices[-1]:
                    frames.append(selected[0].to_ndarray(format="rgb24"))
                    frame_indices.append(selected[1])
                    timestamps.append(selected[2])
                next_target += 1 / VIDEO_FPS
            previous = current

    if not frames:
        raise ValueError(f"No video frames decoded from {video_path!r}")

    if len(frames) > MAX_VIDEO_FRAMES:
        keep = np.linspace(0, len(frames) - 1, MAX_VIDEO_FRAMES).round().astype(int)
        frames = [frames[index] for index in keep]
        frame_indices = [frame_indices[index] for index in keep]
        timestamps = [timestamps[index] for index in keep]

    while len(frames) < MIN_VIDEO_FRAMES:
        frames.append(frames[-1])
        frame_indices.append(frame_indices[-1])
        timestamps.append(timestamps[-1])

    if original_fps <= 0:
        original_fps = VIDEO_FPS
    duration = timestamps[-1] + 1 / original_fps
    metadata = {
        "fps": original_fps,
        "duration": duration,
        "total_num_frames": total_num_frames,
        "frames_indices": frame_indices,
        "video_backend": "pyav",
        "do_sample_frames": False,
    }
    return np.stack(frames), metadata


def build_prompt(caption_prompt: str) -> str:
    prefix = "<|im_start|>user\n"
    video = "<|vision_start|><|video_pad|><|vision_end|>"
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return prefix + video + caption_prompt + suffix


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        choices=CONFIGS,
        default="tp1",
        help="Benchmark configuration",
    )
    parser.add_argument(
        "--video", default="data-samples/sample1.mp4", help="Path to the video file"
    )
    parser.add_argument(
        "--model",
        default=get_model_path(_TARGET_MODEL),
        help="Path to the target model",
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=_CAPTION_PROMPT_PATH,
        help="Path to the captioning prompt",
    )
    parser.add_argument(
        "--spec-decode",
        choices=["none", "mtp", "dflash", "dspark"],
        default="none",
        help="Speculative decoding method",
    )
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        help=(
            "Number of speculative tokens; defaults by method to "
            f"mtp={_MTP_SPEC_TOKENS}, dflash={_DFLASH_SPEC_TOKENS}, "
            f"dspark={_DSPARK_SPEC_TOKENS}"
        ),
    )
    parser.add_argument(
        "--dflash-model",
        default=get_model_path(_DFLASH_MODEL),
        help="Path to the DFlash draft model",
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
    if args.num_speculative_tokens is not None and args.num_speculative_tokens <= 0:
        parser.error("--num-speculative-tokens must be positive")

    config = CONFIGS[args.config]
    model_path = str(Path(args.model).expanduser().resolve())
    if not Path(model_path).exists():
        parser.error(f"Target model not found at {model_path!r}")
    speculative_config: dict[str, object] | None = None
    attention_config: dict[str, object] | None = None
    num_speculative_tokens = 0
    if args.spec_decode == "mtp":
        num_speculative_tokens = (
            args.num_speculative_tokens
            if args.num_speculative_tokens is not None
            else _MTP_SPEC_TOKENS
        )
        speculative_config = {
            "method": "mtp",
            "num_speculative_tokens": num_speculative_tokens,
        }
    elif args.spec_decode == "dflash":
        if not Path(args.dflash_model).exists():
            parser.error(f"DFlash model not found at {args.dflash_model!r}")
        num_speculative_tokens = (
            args.num_speculative_tokens
            if args.num_speculative_tokens is not None
            else _DFLASH_SPEC_TOKENS
        )
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
        num_speculative_tokens = (
            args.num_speculative_tokens
            if args.num_speculative_tokens is not None
            else _DSPARK_SPEC_TOKENS
        )
        speculative_config = {
            "method": "dspark",
            "model": args.dspark_model,
            "num_speculative_tokens": num_speculative_tokens,
            "max_model_len": config["MAX_MODEL_LEN"],
            "attention_backend": "FLASHINFER",
        }
        attention_config = {"backend": "FLASHINFER"}

    # A speculative step can emit the target token plus all accepted drafts.
    output_tokens = config["OUTPUT_TOKENS"]
    delay_iterations = int(
        output_tokens * _PROFILE_DELAY_FRACTION / (num_speculative_tokens + 1)
    )

    import torch
    from vllm import LLM, SamplingParams

    video, video_metadata = sample_frames(args.video)

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=config["TENSOR_PARALLEL_SIZE"],
        max_model_len=config["MAX_MODEL_LEN"],
        max_num_batched_tokens=config["MAX_NUM_BATCHED_TOKENS"],
        max_num_seqs=config["MAX_NUM_SEQS"],
        gpu_memory_utilization=config["GPU_MEMORY_UTILIZATION"],
        enable_chunked_prefill=True,
        enable_prefix_caching=_ENABLE_PREFIX_CACHING,
        disable_log_stats=False,
        speculative_config=speculative_config,
        attention_config=attention_config,
        limit_mm_per_prompt={"image": 0, "video": 1},
        mm_processor_kwargs={
            "fps": VIDEO_FPS,
            "min_frames": MIN_VIDEO_FRAMES,
            "max_frames": MAX_VIDEO_FRAMES,
            "min_pixels": MIN_VIDEO_PIXELS,
            "max_pixels": MAX_VIDEO_PIXELS,
        },
        mm_processor_cache_gb=0,
        mm_encoder_tp_mode="data",
        profiler_config={
            "profiler": "cuda",
            "delay_iterations": delay_iterations,
            "max_iterations": _PROFILE_MAX_INTERATIONS,
        },
    )

    caption_prompt = args.prompt.read_text().strip()
    prompt = build_prompt(caption_prompt)
    requests = [
        {
            "prompt": prompt,
            "multi_modal_data": {"video": (video, video_metadata)},
        }
        for _ in range(config["MAX_NUM_SEQS"] * 3)
    ]

    params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        presence_penalty=1.5,
        repetition_penalty=1.0,
        max_tokens=output_tokens,
    )
    # warmup
    _ = llm.generate(requests[:config["MAX_NUM_SEQS"]], params, use_tqdm=False)

    metrics_before = (
        read_spec_decode_metrics(llm, num_speculative_tokens)
        if speculative_config is not None
        else None
    )
    if args.profile:
        llm.start_profile()
        torch.cuda.nvtx.range_push(
            f"sample1_2fps_24k_vision_{output_tokens}_out_{config['MAX_NUM_SEQS']}_seqs"
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
        f"frames={len(video)} generated_tokens={generated_tokens} elapsed={elapsed_seconds:.3f}s tok/s={generated_tokens / elapsed_seconds:.1f}"
    )
    if metrics_before is not None:
        metrics_after = read_spec_decode_metrics(llm, num_speculative_tokens)
        print_spec_decode_metrics(metrics_before, metrics_after)

    print("--- first output ---")
    print(outputs[0].outputs[0].text)
    print("--- end first output ---")


if __name__ == "__main__":
    main()
