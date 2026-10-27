"""Hopper/Blackwell causal Conv1D for contiguous BTC inputs and OWC packed weights.

The kernel includes left zero padding through TMA bounds checks, optional bias,
residual addition, and SnakeBeta. Every epilogue operation rounds to BF16,
including the raw convolution before bias, matching Torch's cuDNN convolution
followed by bias addition. Both channel counts must be multiples of 32.
Only stride-one, groups-one convolution is supported. Inference only;
autograd is not implemented.
"""

import argparse
import math
import sys
from functools import lru_cache
from pathlib import Path

import cutlass
import torch
from cuda.bindings import driver
from cutlass import cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from torch.nn import functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mega_qwen3_tts.kernels._utils import device_arch, measure, print_kernel_summary, report_difference, report_error


def _transfer_tiles(ci, kernel, tile_n):
    """Avoid C96 reduction padding and batch epilogue transfers where worthwhile."""
    tile_k = 32 if ci == 96 else 64
    store_n = 96 if tile_n == 96 else 64 if kernel == 1 else 32
    return tile_k, store_n


@lru_cache(maxsize=64)
def _compile(weight_shape, dilation, tile_n, stages, has_bias, has_residual, has_activation, arch, full_rows):
    if arch == "sm_90a":
        from mega_qwen3_tts.kernels.sm90.conv1d import Conv1dLaunch
    else:
        from mega_qwen3_tts.kernels.sm100.conv1d import Conv1dLaunch
    tile_k, store_n = _transfer_tiles(weight_shape[2], weight_shape[1], tile_n)
    batch, length = cutlass.sym_int64(), cutlass.sym_int64()
    shape = (batch, length, weight_shape[2])
    output_shape = (batch, length, weight_shape[0])
    fake = lambda shape: make_fake_compact_tensor(
        cutlass.BFloat16, shape, stride_order=tuple(reversed(range(len(shape)))), assumed_align=16
    )
    return cute.compile(
        Conv1dLaunch(weight_shape, dilation, tile_n, stages, tile_k, store_n, full_rows),
        fake(shape),
        fake(weight_shape),
        fake(output_shape),
        fake((weight_shape[0],)) if has_bias else None,
        fake(output_shape) if has_residual else None,
        fake((weight_shape[0],)) if has_activation else None,
        fake((weight_shape[0],)) if has_activation else None,
        fake((65536,)) if has_activation else None,
        make_fake_stream(use_tvm_ffi_env_stream=False),
        cutlass.Int32(1),
        options=f"--enable-tvm-ffi --gpu-arch={arch} --ptxas-options=--fmad=false",
    )


def conv1d(
    x,
    weight,
    bias=None,
    *,
    dilation=1,
    residual=None,
    activation=None,
    tile_n=None,
    stages=None,
    ctas_per_sm=None,
):
    """Apply causal convolution to BTC input; weight packing is outside the launch."""
    if x.ndim != 3 or weight.ndim != 3 or x.shape[2] != weight.shape[2]:
        raise ValueError("Expected matching BTC input and OWC weight")
    if x.shape[2] % 32 or weight.shape[0] % 32:
        raise ValueError("Conv1D requires Ci and Co divisible by 32")
    if dilation < 1 or x.shape[0] < 1 or x.shape[1] < 1 or weight.shape[1] < 1:
        raise ValueError("Convolution dimensions and dilation must be positive")
    alpha = activation.alpha if activation is not None else None
    inverse = activation.inverse_beta if activation is not None else None
    sine = activation.sine_table if activation is not None else None
    for value in (x, weight, bias, residual, alpha, inverse, sine):
        if value is not None and (
            value.dtype != torch.bfloat16 or not value.is_cuda or value.device != x.device or not value.is_contiguous()
        ):
            raise ValueError("Operands must be contiguous CUDA BF16 tensors")
    if bias is not None and bias.shape != (weight.shape[0],):
        raise ValueError("Bias must have one value per output channel")
    if residual is not None and residual.shape != (*x.shape[:2], weight.shape[0]):
        raise ValueError("Residual shape must match the BTC output")
    if activation is not None and (alpha.shape != (weight.shape[0],) or sine is None):
        raise ValueError("Activation must be a prepared SnakeBeta with matching channels and a sine table")
    arch = device_arch(x.device)
    tile_m = 64 if arch == "sm_90a" else 128
    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
    ci, co, kernel = x.shape[2], weight.shape[0], weight.shape[1]
    if arch == "sm_90a":
        if co == 96:
            default_n, default_stages, default_ctas = 96, 4, 2
        elif ci == co and co in (192, 384) and kernel == 1:
            default_n, default_stages, default_ctas = 64, 4, 2
        else:
            default_n, default_stages, default_ctas = 192, 4 if kernel == 1 else 6, 1
            if x.shape[0] * math.ceil(x.shape[1] / 64) * math.ceil(co / default_n) < sm_count // 2:
                default_n = 64
    elif kernel == 1 and ci == co and x.shape[0] * math.ceil(x.shape[1] / 128) * math.ceil(co / 64) <= sm_count:
        # Short streaming chunks fit one wave with narrower channel tiles.
        default_n, default_stages, default_ctas = 64, 4, 1
    elif co == 96:
        default_n, default_stages, default_ctas = 96, 2 if kernel == 1 else 4, 2
    elif (ci, co, kernel) == (192, 192, 1):
        default_n, default_stages, default_ctas = 64, 2, 2
    elif (ci, co, kernel) == (384, 384, 1):
        default_n, default_stages, default_ctas = 128, 2, 2
    else:
        default_n, default_stages, default_ctas = (192 if ci == co else 128), 4, 1
        # Narrow channel tiles give small early layers more independent CTAs.
        if x.shape[0] * math.ceil(x.shape[1] / 128) * math.ceil(co / default_n) < sm_count // 2:
            default_n = 128 if ci == co else 64
    if tile_n is None:
        tile_n = default_n
    if stages is None:
        stages = default_stages if tile_n == default_n else (3 if tile_n == 96 else 4)
    if ctas_per_sm is None:
        ctas_per_sm = default_ctas if (tile_n, stages) == (default_n, default_stages) else 1
    if ctas_per_sm < 1:
        raise ValueError("ctas_per_sm must be positive")
    tile_k, store_n = _transfer_tiles(ci, kernel, tile_n)
    shared_bytes = (tile_m + tile_n) * tile_k * 2 * stages + 512 * store_n
    if tile_n not in (64, 96, 128, 192) or not 2 <= stages <= 8 or shared_bytes > 227328:
        raise ValueError("Unsupported tile or shared-memory stage count")
    tiles = x.shape[0] * math.ceil(x.shape[1] / tile_m) * math.ceil(weight.shape[0] / tile_n)
    ctas = min(tiles, ctas_per_sm * sm_count)
    output = x.new_empty((*x.shape[:2], weight.shape[0]))
    function = _compile(
        tuple(weight.shape),
        dilation,
        tile_n,
        stages,
        bias is not None,
        residual is not None,
        activation is not None,
        arch,
        x.shape[1] % tile_m == 0,
    )
    function(
        x,
        weight,
        output,
        bias,
        residual,
        alpha,
        inverse,
        sine,
        driver.CUstream(torch.cuda.current_stream(x.device).cuda_stream),
        ctas,
    )
    return output


@torch.inference_mode()
def main():
    from torch_qwen3_tts.vocoder import ChannelsLastConv1d
    from torch_qwen3_tts.vocoder import SnakeBeta as TorchSnakeBeta

    from mega_qwen3_tts.kernels.snake_beta import SnakeBeta

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--shape", default="4,122880,96,96,7", help="B,L,Ci,Co,K")
    parser.add_argument("--dilation", type=int, default=1)
    parser.add_argument("--tile-n", type=int, default=None, help="Output-channel tile; automatic when omitted")
    parser.add_argument("--stages", type=int, default=None, help="Operand stages; automatic when omitted")
    parser.add_argument("--variant", choices=("conv", "snake", "residual", "all"), default="all")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    b, length, ci, co, kernel = map(int, args.shape.split(","))
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    x = torch.randn(b, length, ci, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(co, kernel, ci, device="cuda", dtype=x.dtype) / math.sqrt(kernel * ci)
    bias = torch.randn(co, device="cuda", dtype=x.dtype)
    x_bct = x.transpose(1, 2).contiguous()
    w_oct = w.permute(0, 2, 1).contiguous()
    module = torch.nn.Conv1d(ci, co, kernel, dilation=args.dilation).cuda().bfloat16()
    module.weight.copy_(w_oct)
    module.bias.copy_(bias)
    btc_reference = ChannelsLastConv1d(module)
    torch_snake = TorchSnakeBeta(co).cuda().bfloat16()
    torch_snake.alpha.normal_(std=0.25)
    torch_snake.beta.normal_(std=0.25)
    snake = SnakeBeta(torch_snake.alpha, torch_snake.beta)
    skip = torch.randn(b, length, co, device=x.device, dtype=x.dtype)
    results = []
    for variant in ("conv", "snake", "residual") if args.variant == "all" else (args.variant,):

        def postprocess(y, variant=variant):
            if variant == "snake":
                y = torch_snake(y.transpose(1, 2)).transpose(1, 2)
            elif variant == "residual":
                y = y + skip
            return y

        def reference():
            y = F.conv1d(F.pad(x_bct, ((kernel - 1) * args.dilation, 0)), w_oct, bias, dilation=args.dilation)
            return postprocess(y.transpose(1, 2))

        def btc_call():
            return postprocess(btc_reference(x))

        def custom(variant=variant):
            return conv1d(
                x,
                w,
                bias,
                dilation=args.dilation,
                tile_n=args.tile_n,
                stages=args.stages,
                activation=snake if variant == "snake" else None,
                residual=skip if variant == "residual" else None,
            )

        compiled = torch.compile(btc_call, fullgraph=True, options={"emulate_precision_casts": True})
        for _ in range(3):
            x.normal_()
            x_bct.copy_(x.transpose(1, 2))
            actual = custom()
            if variant == "conv":
                report_error(variant, actual, reference(), atol=0.03125, rtol=0.02)
            else:
                raw = conv1d(x, w, bias, dilation=args.dilation, tile_n=args.tile_n, stages=args.stages)
                report_error(f"{variant} fusion", actual, postprocess(raw), atol=0, rtol=0)
                report_difference(f"{variant} vs cuDNN", actual, reference())
        times = measure({"Torch": btc_call, "Compiled": compiled, "Fused": custom}, profile=args.profile)
        results.append(((variant,), times))
    print(f"Shape B,L,Ci,Co,K={args.shape}, dilation={args.dilation}. Baselines use channels-last cuDNN.")
    print_kernel_summary(results, columns=["Variant"], workload="causal Conv1D+bias and optional post-op")


if __name__ == "__main__":
    main()
