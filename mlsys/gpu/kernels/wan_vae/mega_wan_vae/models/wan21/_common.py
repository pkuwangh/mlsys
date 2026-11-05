"""Shared Wan 2.1 compilation settings and convolution selection."""

import torch


# C384 custom accumulation can cross BF16 rounding ties and amplify through
# posterior parameters. Keep cuDNN there; N96/N192 kernels serve C96/C192 stages.
def _custom_channels(dtype):
    return (96, 192, 384) if dtype == torch.float32 and torch.backends.cudnn.allow_tf32 else (96, 192)
