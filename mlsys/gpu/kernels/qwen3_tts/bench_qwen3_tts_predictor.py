"""Compare Torch, compiled, and fused prediction of all 15 remaining codes per frame.

Includes the two-token prefix, 14 decode steps, greedy selection, embeddings,
and feedback sum. Uses synthetic hidden states or captured talker outputs;
this is not an end-to-end speech-quality benchmark.
"""

import argparse
from copy import deepcopy
from functools import partial

import torch
from mega_qwen3_tts import MegaQwen3TTSPredictor
from mega_qwen3_tts.kernels._utils import measure, print_summary, report_difference
from qwen3_tts_predictor_reference import PredictorConfig, TorchQwen3TTSPredictor


@torch.inference_mode()
def run(args):
    torch.manual_seed(42)
    config = PredictorConfig.from_checkpoint(args.model) if args.model else PredictorConfig()
    reference = TorchQwen3TTSPredictor(config).to(device="cuda", dtype=torch.bfloat16).eval()
    if args.model:
        reference.load_checkpoint(args.model)
    compiled = deepcopy(reference)
    custom = MegaQwen3TTSPredictor(reference, fused_selection=args.fused_selection).eval()
    options = {"emulate_precision_casts": True}
    # Compile one transformer pass, not 15 unrolled copies. Head weights are tensor
    # arguments so changing code groups does not specialize/recompile the graph.
    compiled.forward = torch.compile(compiled.forward, fullgraph=True, options=options)
    custom.forward = torch.compile(custom.forward, options=options)
    models = {"Torch": reference, "Compiled": compiled, "Fused": custom}
    replay = torch.load(args.inputs, map_location="cpu", weights_only=True) if args.inputs else None
    print(f"{torch.cuda.get_device_name()} | BF16 | {config.num_hidden_layers} layers | {config.num_code_groups} codes")
    print(f"Weights: {args.model or 'deterministic random'} | greedy decoding")
    print(f"Inputs: {args.inputs or 'deterministic random hidden states'}", flush=True)
    print("Numerical differences are relative to eager Torch, not a pass/fail criterion.")
    rows = []
    for batch in args.batch_size:
        print(f"\nB={batch}: full-frame numerical comparison", flush=True)
        if replay:
            if batch > min(replay["hidden"].shape[0], replay["first_code"].shape[0]):
                raise ValueError("Requested batch exceeds the captured inputs' batch size")
            hidden = replay["hidden"][:batch].to(device="cuda", dtype=torch.bfloat16).contiguous()
            first_code = replay["first_code"][:batch].to(device="cuda", dtype=torch.int64).contiguous()
        else:
            hidden = torch.randn(batch, 1, config.talker_hidden_size, device="cuda", dtype=torch.bfloat16) * 0.1
            first_code = torch.randint(config.talker_vocab_size, (batch, 1), device="cuda")
        caches = {name: model.make_cache(batch) for name, model in models.items()}
        expected_codes, expected_feedback, expected_logits = reference.generate_frame(
            hidden, first_code, caches["Torch"], return_logits=True
        )
        for name in ("Compiled", "Fused"):
            codes, feedback, logits = models[name].generate_frame(
                hidden, first_code, caches[name], forced_codes=expected_codes[:, 1:], return_logits=True
            )
            torch.testing.assert_close(codes, expected_codes, atol=0, rtol=0)
            torch.testing.assert_close(feedback, expected_feedback, atol=0, rtol=0)
            report_difference(f"{name} teacher-forced logits", logits, expected_logits)
            report_difference(f"{name} teacher-forced KV", caches[name], caches["Torch"])
            agreement = (logits.argmax(-1) == expected_logits.argmax(-1)).float().mean().item()
            print(f"{name} teacher-forced top-1 agreement: {agreement:.1%}", flush=True)
            generated, _, _ = models[name].generate_frame(hidden, first_code, caches[name])
            agreement = (generated[:, 1:] == expected_codes[:, 1:]).float().mean().item()
            print(f"{name} free-running greedy code agreement: {agreement:.1%}", flush=True)
        functions = {
            name: partial(model.generate_frame, hidden, first_code, caches[name]) for name, model in models.items()
        }
        with torch.cuda.nvtx.range(f"predictor.B{batch}.frame"):
            timings = measure(functions, profile=args.profile)
        rows.append((batch, timings))
        print(" | ".join(f"{name}={us / 1000:.3f} ms/frame" for name, us in timings.items()), flush=True)

    print_summary(rows, workload="frame")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", help="Local Qwen3-TTS checkpoint; omitted for random weights")
    parser.add_argument("--inputs", help="Replay .pt with hidden [B,1,D] and first_code [B,1] from the matching talker")
    parser.add_argument("--batch-size", nargs="+", type=int, default=[1, 8, 16, 32], help="Frame batches")
    parser.add_argument("--profile", action="store_true", help="Capture one full-frame graph replay per variant")
    parser.add_argument(
        "--fused-selection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fuse greedy selection and embedding lookup in the custom path",
    )
    args = parser.parse_args()
    if args.inputs and not args.model:
        parser.error("--inputs requires the matching --model checkpoint")
    if any(not 1 <= batch <= 32 for batch in args.batch_size):
        parser.error("Expected 1 <= batch size <= 32")
    run(args)


if __name__ == "__main__":
    main()
