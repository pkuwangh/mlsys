#!/usr/bin/env python3

"""Generate one WAV file with SGLang-Omni's Qwen3-TTS CustomVoice pipeline."""

import argparse
import asyncio
from pathlib import Path

from utils import get_model_path


async def run(args):
    import numpy as np
    import soundfile as sf
    from sglang_omni.models.qwen3_tts.config import Qwen3TTSPipelineConfig
    from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
    from sglang_omni.proto import OmniRequest

    runner = MultiProcessPipelineRunner(Qwen3TTSPipelineConfig(model_path=args.model))
    print(f"Loading {args.model}", flush=True)
    try:
        await runner.start(timeout=600)
        data = await runner.coordinator.submit(
            "tts-quickstart",
            OmniRequest(
                inputs={"text": args.text},
                params={
                    "task_type": "CustomVoice",
                    "voice": "Ryan",
                    "language": "English",
                    "max_new_tokens": 1024,
                },
            ),
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
