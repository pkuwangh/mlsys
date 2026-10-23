"""Wan 2.1 primitive correctness and optional production-shape timings.

Run this file with run_cute_dsl.py. No checkpoint is required.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from mega_wan_vae.kernels.conv import benchmark_production
from mega_wan_vae.kernels.conv import verify as verify_conv
from mega_wan_vae.kernels.norm import rmsnorm, verify


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(42)
    verify()
    for channels in (96, 192, 384):
        x = torch.randn(2, 3, 5, 7, channels, device="cuda", dtype=torch.bfloat16)
        gamma = torch.randn(channels, device="cuda", dtype=x.dtype)
        bias = torch.randn_like(gamma)
        expected = F.normalize(x.float(), dim=-1).to(x.dtype) * channels**0.5 * gamma + bias
        torch.testing.assert_close(rmsnorm(x, gamma, bias), expected, atol=1e-3, rtol=1e-2)
    verify_conv("wan21")
    if args.benchmark:
        benchmark_production("wan21")


if __name__ == "__main__":
    main()
