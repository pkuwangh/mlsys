"""Compare the eager, compiled, and custom 0.6B/1.7B talker on prepared embeddings.

Prefill builds identical KV caches. Timings replay one decode step at a fixed
context length, excluding cache restoration, model loading, and compilation.
This measures the talker, not code prediction or waveform generation.
"""

import argparse
from dataclasses import replace
from functools import partial

import torch
from mega_qwen3_tts import MegaQwen3TTSTalker
from mega_qwen3_tts.kernels._utils import measure, print_summary, report_difference
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
    custom = MegaQwen3TTSTalker(reference, fused_qkv=args.fused_qkv, fused_mask=args.fused_mask).eval()
    if args.compile:
        custom = torch.compile(custom, options={"emulate_precision_casts": True})
    compiled = (
        torch.compile(reference, fullgraph=True, options={"emulate_precision_casts": True}) if args.compile else None
    )
    replay = torch.load(args.inputs, map_location="cpu", weights_only=True) if args.inputs else None
    context = replay["prefix"].shape[1] if replay else args.context
    steps = replay["decode"].shape[1] if replay else 3
    print(
        f"{torch.cuda.get_device_name()} | BF16 | hidden={config.hidden_size} | layers={config.num_hidden_layers} | context={context}"
    )
    custom_name = "QKV + post-attention" if args.fused_qkv else "post-attention only"
    print(f"Weights: {args.model or 'deterministic random'} | Fused: {custom_name}")
    print(f"Inputs: {args.inputs or 'deterministic random embeddings'}")
    if args.fused_qkv:
        print(f"Fused additive mask: {args.fused_mask}")
    print("Numerical differences are relative to eager Torch, not a pass/fail criterion.")
    rows = []
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
        custom_cache = custom.make_cache(batch, context + steps)
        custom_cache.copy_(cache)
        compiled_cache = cache.clone() if compiled is not None else None
        comparisons = {"Fused": (custom, custom_cache)}
        if compiled is not None:
            comparisons = {"Compiled": (compiled, compiled_cache), **comparisons}
        for step in range(steps):
            token = tokens[:, step : step + 1].contiguous()
            position = torch.tensor([context + step], device="cuda")
            expected = reference(token, cache, position)
            for name, (model, model_cache) in comparisons.items():
                actual = model(token, model_cache, position)
                report_difference(f"{name} step {step} logits", actual, expected)
                report_difference(f"{name} step {step} KV", model_cache, cache)
                agreement = (actual.argmax(-1) == expected.argmax(-1)).float().mean().item()
                print(f"{name} top-1 agreement: {agreement:.1%}", flush=True)

        # Each replay overwrites the same last slot; no growing context or reset in timing.
        functions = {"Torch": partial(reference, token, cache, position)}
        functions.update(
            {name: partial(model, token, model_cache, position) for name, (model, model_cache) in comparisons.items()}
        )
        with torch.cuda.nvtx.range(f"talker.B{batch}.context{context + steps - 1}"):
            timings = measure(functions, profile=args.profile)
        print(" | ".join(f"{name}={us:.3f} us" for name, us in timings.items()), flush=True)
        rows.append((batch, timings))
    print_summary(rows, workload="decode step")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", help="Local Qwen3-TTS checkpoint; omitted for random weights")
    parser.add_argument("--inputs", help="Replay .pt with prefix/decode [B,T,hidden] embeddings; overrides --context")
    parser.add_argument("--batch-size", nargs="+", type=int, default=[1, 8, 16, 32], help="Decode batches")
    parser.add_argument("--context", type=int, default=128, help="Prefill token count")
    parser.add_argument("--layers", type=int, help="Override layer count for a short development run")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True, help="Compile Torch baseline")
    parser.add_argument(
        "--fused-qkv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fuse QKV preparation as well; --no-fused-qkv isolates post-attention",
    )
    parser.add_argument(
        "--fused-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write the SDPA additive mask inside fused QKV preparation",
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
