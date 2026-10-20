"""Compare the eager, compiled, and custom 0.6B talker on prepared embeddings.

Prefill builds identical KV caches. Timings replay one decode step at a fixed
context length, excluding cache restoration, model loading, and compilation.
This measures the talker, not code prediction or waveform generation.
"""

import argparse
from dataclasses import replace
from functools import partial

import torch
from mega_qwen3_tts import MegaQwen3TTSTalker
from mega_qwen3_tts.kernels._utils import measure, report_error
from qwen3_tts_reference import TalkerConfig, TorchQwen3TTSTalker


@torch.inference_mode()
def run(args):
    torch.manual_seed(42)
    config = TalkerConfig.from_checkpoint(args.model) if args.model else TalkerConfig()
    if args.layers is not None:
        config = replace(config, num_hidden_layers=args.layers)
    reference = TorchQwen3TTSTalker(config).to(device="cuda", dtype=torch.bfloat16).eval()
    if args.model:
        reference.load_checkpoint(args.model)
    custom = MegaQwen3TTSTalker(reference, fused=args.fused, fused_qkv=args.fused_qkv).eval()
    if args.compile:
        custom = torch.compile(custom, options={"emulate_precision_casts": True})
    compiled = (
        torch.compile(reference, fullgraph=True, options={"emulate_precision_casts": True}) if args.compile else None
    )
    replay = torch.load(args.inputs, map_location="cpu", weights_only=True) if args.inputs else None
    context = replay["prefix"].shape[1] if replay else args.context
    steps = replay["decode"].shape[1] if replay else 3
    print(f"{torch.cuda.get_device_name()} | BF16 | layers={config.num_hidden_layers} | context={context}")
    custom_name = "QKV + post-attention" if args.fused_qkv else "post-attention only"
    print(
        f"Weights: {args.model or 'deterministic random'} | Custom: {custom_name if args.fused else 'Torch scaffold'}"
    )
    print(f"Inputs: {args.inputs or 'deterministic random embeddings'}")
    passed = True
    for batch in args.batch_size:
        print(f"\nB={batch}: shared eager prefill and {steps}-step validation", flush=True)
        if replay:
            if batch > min(replay["prefix"].shape[0], replay["decode"].shape[0]):
                raise ValueError("Requested batch exceeds the captured inputs' batch size")
            prefix = replay["prefix"][:batch].to(device="cuda", dtype=torch.bfloat16).contiguous()
            tokens = replay["decode"][:batch].to(device="cuda", dtype=torch.bfloat16).contiguous()
        else:
            prefix = torch.randn(batch, context, config.hidden_size, device="cuda", dtype=torch.bfloat16) * 0.1
            tokens = torch.randn(batch, steps, config.hidden_size, device="cuda", dtype=torch.bfloat16) * 0.1
        positions = torch.arange(context, device="cuda")
        cache = reference.make_cache(batch, context + steps)
        reference(prefix, cache, positions)
        # Compare decode paths from exactly the same prefix, including T=1.
        custom_cache = cache.clone()
        compiled_cache = cache.clone() if compiled is not None else None
        for step in range(steps):
            token = tokens[:, step : step + 1].contiguous()
            position = torch.tensor([context + step], device="cuda")
            expected = reference(token, cache, position)
            actual = custom(token, custom_cache, position)
            passed &= report_error(f"Custom step {step}", actual, expected, strict=False)
            print(f"Custom top-1 agreement: {(actual.argmax(-1) == expected.argmax(-1)).float().mean().item():.1%}")
            passed &= report_error(f"Custom cache {step}", custom_cache, cache, strict=False)
            if compiled is not None:
                passed &= report_error(
                    f"Compiled step {step}", compiled(token, compiled_cache, position), expected, strict=False
                )
                passed &= report_error(f"Compiled cache {step}", compiled_cache, cache, strict=False)

        # Each replay overwrites the same last slot; no growing context or reset in timing.
        functions = {
            "Torch": partial(reference, token, cache, position),
            "Custom": partial(custom, token, custom_cache, position),
        }
        if compiled is not None:
            functions["Compiled"] = partial(compiled, token, compiled_cache, position)
        with torch.cuda.nvtx.range(f"talker.B{batch}.context{context + steps - 1}"):
            timings = measure(functions, profile=args.profile)
        print(" | ".join(f"{name}={us:.3f} us" for name, us in timings.items()), flush=True)
        baseline = "Compiled" if compiled is not None else "Torch"
        print(f"Custom speedup vs {baseline}: {timings[baseline] / timings['Custom']:.3f}x", flush=True)
    if not passed:
        raise SystemExit(
            "Numerical validation FAILED; timings above are diagnostic, not a validated end-to-end speedup."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", help="Local Qwen3-TTS checkpoint; omitted for random weights")
    parser.add_argument("--inputs", help="Replay .pt with prefix/decode [B,T,hidden] embeddings; overrides --context")
    parser.add_argument("--batch-size", nargs="+", type=int, default=[16, 32], help="Decode batches")
    parser.add_argument("--context", type=int, default=128, help="Prefill token count")
    parser.add_argument("--layers", type=int, help="Override layer count for a short development run")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True, help="Compile Torch baseline")
    parser.add_argument("--fused", action=argparse.BooleanOptionalAction, default=True, help="Use custom decode blocks")
    parser.add_argument(
        "--fused-qkv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fuse QKV preparation as well; --no-fused-qkv isolates post-attention",
    )
    parser.add_argument("--profile", action="store_true", help="Emit a CUDA-profiler capture range after timing")
    args = parser.parse_args()
    if args.inputs and not args.model:
        parser.error("--inputs requires the matching --model checkpoint")
    if args.context < 1 or any(batch < 1 or batch > 32 for batch in args.batch_size):
        parser.error("Expected context >= 1 and 1 <= batch size <= 32")
    if args.layers is not None and args.layers < 1:
        parser.error("Expected layers >= 1")
    run(args)


if __name__ == "__main__":
    main()
