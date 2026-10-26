"""Benchmark the waveform-convolution tail of Qwen3-TTS's codec decoder.

The shape is [batch,1024,4*codec_frames], after the transformer and ConvNeXt
upsamplers. Timings include all tail operations and boundary layout copies,
but exclude the quantizer, transformer, and ConvNeXt prefix. With --full,
all variants share the upstream eager prefix and replace only the tail.
Inputs are synthetic features or random codec IDs, not generated speech.
"""

import argparse

import torch
from mega_qwen3_tts.kernels._utils import measure, print_table, report_difference
from mega_qwen3_tts.vocoder import MegaVocoderTail
from torch_qwen3_tts.vocoder import CodecDecoder, TorchVocoderTail, load_upstream_decoder


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", help="Local Qwen3-TTS or speech_tokenizer checkpoint; otherwise random weights")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--frames", type=int, default=64, help="Codec frames before the two 2x ConvNeXt upsamplers")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--full", action="store_true", help="Include the upstream codec prefix; requires --model and qwen-tts"
    )
    args = parser.parse_args()
    if args.full and not args.model:
        parser.error("--full requires --model")
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    print(torch.cuda.get_device_properties(0), flush=True)
    if args.model:
        reference = TorchVocoderTail.from_checkpoint(args.model)
    else:
        reference = TorchVocoderTail().cuda().bfloat16().eval()
    channels_last = MegaVocoderTail(reference, custom=False).eval()
    custom = MegaVocoderTail(reference).eval()
    compiled = torch.compile(reference, fullgraph=True, options={"emulate_precision_casts": True})
    compiled_btc = torch.compile(channels_last, fullgraph=True, options={"emulate_precision_casts": True})
    if args.full:
        upstream = load_upstream_decoder(args.model)
        codes = torch.randint(2048, (args.batch_size, 16, args.frames), device="cuda")
        captured = []
        handle = upstream.decoder[0].register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0]))
        expected = upstream(codes)
        handle.remove()
        torch.testing.assert_close(reference(captured[0]), expected, atol=0, rtol=0)
        print("PASS copied Torch tail vs upstream: exact waveform match", flush=True)
        # Prefix modules and immutable weights are shared. Only the tail differs.
        with_btc = CodecDecoder(upstream, compiled_btc).eval()
        with_compiled = CodecDecoder(upstream, compiled).eval()
        with_custom = CodecDecoder(upstream, custom).eval()
        functions = {
            "Upstream": lambda: upstream(codes),
            "Compiled tail": lambda: with_compiled(codes),
            "Compiled BTC tail": lambda: with_btc(codes),
            "Fused": lambda: with_custom(codes),
        }
        for name, call in functions.items():
            print(f"Warmup: {name}", flush=True)
            report_difference(name, call(), expected)
        times = measure(functions, profile=args.profile)
        print("\nFull codec decoder: quantizer + transformer + ConvNeXt + waveform tail; milliseconds.")
        print_table(
            ["B", "Frames", *times, "Compiled BTC tail/Fused"],
            [
                [
                    args.batch_size,
                    args.frames,
                    *[f"{value / 1000:.3f}" for value in times.values()],
                    f"{times['Compiled BTC tail'] / times['Fused']:.3f}x",
                ]
            ],
        )
        print("Waveform differences are reported, not a perceptual-quality validation.")
        return
    x = torch.randn(args.batch_size, 1024, args.frames * 4, device="cuda", dtype=torch.bfloat16)
    print(f"Waveform tail input={tuple(x.shape)}; weights={args.model or 'random'}", flush=True)
    functions = {
        "Torch": lambda: reference(x),
        "Compiled": lambda: compiled(x),
        "Compiled BTC": lambda: compiled_btc(x),
        "Fused": lambda: custom(x),
    }
    expected = reference(x)
    for name, call in functions.items():
        print(f"Warmup: {name}", flush=True)
        report_difference(name, call(), expected)
    times = measure(functions, profile=args.profile)
    print("\nConvolutional waveform tail only; median interleaved CUDA graphs, milliseconds.")
    print_table(
        ["B", "Frames", *times, "Compiled BTC/Fused"],
        [
            [
                args.batch_size,
                args.frames,
                *[f"{value / 1000:.3f}" for value in times.values()],
                f"{times['Compiled BTC'] / times['Fused']:.3f}x",
            ]
        ],
    )
    print("Waveform differences are reported, not a perceptual-quality validation.")


if __name__ == "__main__":
    main()
