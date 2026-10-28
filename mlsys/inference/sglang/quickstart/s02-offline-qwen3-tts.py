#!/usr/bin/env python3

"""Generate one WAV with Qwen3-TTS; Base checkpoints use the first English Seed-TTS reference voice."""

import argparse
import asyncio
import sys
from pathlib import Path

from utils import get_model_path


async def run(args):
    import numpy as np
    import soundfile as sf
    from sglang_omni.models.qwen3_tts.config import Qwen3TTSPipelineConfig
    from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
    from sglang_omni.proto import OmniRequest

    inputs = {"text": args.text}
    params = {"language": "English", "max_new_tokens": 1024}
    if Path(args.model).name.endswith("-Base"):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sglang-omni"))
        from benchmarks.dataset.seedtts import load_seedtts_samples

        reference = load_seedtts_samples("zhaochenyang20/seed-tts-eval-arrow", 1, split="en")[0]
        inputs["references"] = [{"audio_path": reference.ref_audio, "text": reference.ref_text}]
        params["task_type"] = "Base"
        print(f"Reference voice: Seed-TTS {reference.sample_id}", flush=True)
    else:
        params.update(task_type="CustomVoice", voice="Ryan")

    runner = MultiProcessPipelineRunner(Qwen3TTSPipelineConfig(model_path=args.model))
    print(f"Loading {args.model}", flush=True)
    try:
        await runner.start(timeout=600)
        data = await runner.coordinator.submit(
            "tts-quickstart",
            OmniRequest(inputs=inputs, params=params),
        )
        waveform = np.frombuffer(data["audio_waveform"], dtype=data["audio_waveform_dtype"])
        sample_rate = data["sample_rate"]
        sf.write(args.output, waveform, sample_rate)
        print(f"Saved {args.output}: {waveform.size / sample_rate:.2f}s at {sample_rate} Hz")
    finally:
        await runner.stop()


def main():
    model = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    local_model = get_model_path(model)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--model", default=local_model if Path(local_model).is_dir() else model, help="Model path or HF ID"
    )
    parser.add_argument(
        "--text", default="Hello! This is a speech generation test using SGLang Omni.", help="Text to speak"
    )
    parser.add_argument("--output", default="qwen3-tts.wav", help="Output WAV path")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
