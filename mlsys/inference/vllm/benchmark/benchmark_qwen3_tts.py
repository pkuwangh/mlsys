"""Profile one warmed-up Qwen3-TTS batch through vLLM-Omni.

From the repository root:
    ./run_vllm.py --nsys mlsys/inference/vllm/benchmark/profile_qwen3_tts.py --batch-size 1
    ./run_vllm.py --nsys mlsys/inference/vllm/benchmark/profile_qwen3_tts.py --batch-size 8

Uses the Omni CustomVoice example's utterance, speaker, and full-utterance decoding.
Initialization, warmup, a separate timing run, and WAV saving are outside capture.
"""

import argparse
import copy
import json
import runpy
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch-size", type=int, choices=[1, 8], default=1, help="Concurrent requests")
    args = parser.parse_args()

    import soundfile as sf
    import torch
    from vllm_omni import Omni

    vllm_dir = Path(__file__).resolve().parents[1]
    example = runpy.run_path(
        str(vllm_dir / "vllm-omni/examples/offline_inference/text_to_speech/qwen3_tts/end2end.py")
    )
    query = example["get_custom_voice_query"]()
    output_dir = vllm_dir / "output/qwen3-tts-profile" / f"b{args.batch_size}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"model={query.model_name} batch_size={args.batch_size} gpu={torch.cuda.get_device_name()}", flush=True)
    print(f"prompt={query.inputs['additional_information']}", flush=True)

    omni = Omni(model=query.model_name, profiler_config={"profiler": "cuda"})

    def generate():
        return list(omni.generate([copy.deepcopy(query.inputs) for _ in range(args.batch_size)], use_tqdm=False))

    try:
        print("Warmup...", flush=True)
        generate()
        print("Unprofiled timing...", flush=True)
        started = time.perf_counter()
        timed_outputs = generate()
        elapsed = time.perf_counter() - started
        print(f"batch_size={args.batch_size} unprofiled_elapsed_s={elapsed:.4f}", flush=True)

        print("Capturing one batch (both stages)...", flush=True)
        omni.start_profile()
        try:
            profiled_outputs = generate()
        finally:
            omni.stop_profile()
        print("Capture complete.", flush=True)

        durations = {}
        for label, outputs in [("timed", timed_outputs), ("profiled", profiled_outputs)]:
            if len(outputs) != args.batch_size:
                raise RuntimeError(f"{label}: expected {args.batch_size} outputs, got {len(outputs)}")
            durations[label] = []
            for index, output in enumerate(outputs):
                mm = output.outputs[0].multimodal_output
                audio = mm["audio"]
                audio = torch.cat(audio, dim=-1) if isinstance(audio, list) else audio
                sr = mm["sr"]
                sr = int(sr[-1] if isinstance(sr, list) else sr)
                audio = audio.float().cpu().numpy().flatten()
                sf.write(output_dir / f"{label}_{index}.wav", audio, sr)
                durations[label].append(len(audio) / sr)
        summary = {
            "model": query.model_name,
            "batch_size": args.batch_size,
            "unprofiled_elapsed_s": elapsed,
            "audio_duration_s": durations,
            "audio_seconds_per_wall_second": sum(durations["timed"]) / elapsed,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        omni.close()


if __name__ == "__main__":
    main()
