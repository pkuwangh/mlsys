"""Check shortcut grouping and the compiled encoder's large-batch layout regression."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from bench_utils import compare_model_outputs
from bench_wan_vae_encoder import make_synthetic_video
from diffusers import AutoencoderKLWan
from diffusers.models.autoencoders.autoencoder_kl_wan import AvgDown3D
from mega_wan_vae.layers.torch import raw_avg_down3d
from wan_vae_reference import TorchCompileChunkWanVaeEncoder


@torch.inference_mode()
def check_grouping(device):
    for dtype in (torch.bfloat16, torch.float32):
        for ci, co, ft, fs in (
            (12, 24, 2, 2),
            (160, 320, 1, 2),
            (320, 640, 2, 2),
            (640, 640, 1, 1),
            (12, 12, 2, 2),
            (12, 6, 2, 2),
            (12, 6, 1, 1),
            (6, 8, 2, 2),  # Group size 6 crosses factor axes: general fallback.
        ):
            avg = AvgDown3D(ci, co, ft, fs)
            for frames in (1, 3, 4):
                x = torch.randn(2, ci, frames, 6, 8, device=device, dtype=dtype)
                for value in (x, x.contiguous(memory_format=torch.channels_last_3d)):
                    for view in (value, value[:, :, :1]):
                        actual = raw_avg_down3d(avg, view)
                        torch.testing.assert_close(actual, avg(view), atol=2e-6, rtol=2e-6)
                        if avg.group_size == 1 and ft == fs == 1:
                            assert actual is view
                        elif view.stride(1) == 1:
                            assert actual.is_contiguous(memory_format=torch.channels_last_3d)
        print(f"{dtype}: shortcut grouping/layouts PASS", flush=True)


@torch.inference_mode()
def check_compiled_checkpoint(model, allow_tf32, dtype):
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    vae = AutoencoderKLWan.from_pretrained(model, subfolder="vae", torch_dtype=dtype).cuda().eval()
    if not vae.config.is_residual:
        raise ValueError("The compiled layout regression requires a Wan 2.2 checkpoint")
    torch.nn.utils.convert_conv3d_weight_memory_format(vae, torch.channels_last_3d)
    torch.nn.utils.convert_conv2d_weight_memory_format(vae, torch.channels_last)
    compiled = TorchCompileChunkWanVaeEncoder(vae).eval()
    # Small operator probes passed with the faulty layout. This full cold chunk
    # reproduces the fusion that used inconsistent strides for the shortcut.
    atol, rtol = (0.25, 1e-3) if dtype == torch.bfloat16 else (1e-4, 1e-4)
    for n, frames, height, width in ((16, 1, 480, 832), (2, 17, 128, 192)):
        video = make_synthetic_video(
            batch_size=n,
            channels=3,
            frames=frames,
            height=height,
            width=width,
            seed=42,
            device=torch.device("cuda"),
            dtype=dtype,
        )
        expected = vae.encode(video, return_dict=False)[0].parameters
        actual = compiled(video)
        assert compare_model_outputs(
            f"Compiled T={frames} vs Diffusers",
            actual,
            expected,
            atol=atol,
            rtol=rtol,
            allow_tf32=allow_tf32,
            operation="encode",
        )
        # A batch-index bug must fail the same gate used by the benchmark.
        assert not compare_model_outputs(
            "Repeated batch 0",
            expected[:1].expand_as(expected),
            expected,
            atol=atol,
            rtol=rtol,
            allow_tf32=allow_tf32,
            operation="encode",
            emit=lambda line: None,
        ), "Validation accepted duplicated batch output"
        del actual, expected, video


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--model", help="Also run the CUDA compiled regression with a Wan 2.2 checkpoint.")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="fp32")
    args = parser.parse_args()
    torch.manual_seed(42)
    check_grouping(args.device)
    if args.model:
        check_compiled_checkpoint(
            args.model, args.allow_tf32, torch.bfloat16 if args.dtype == "bf16" else torch.float32
        )
