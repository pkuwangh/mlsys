"""Shared Wan 2.1 compilation settings and convolution selection."""

import torch

from ...layers.fused import _FusedRMSNorm
from ...layers.torch import RawWanRMSNorm

# C384 custom accumulation can cross BF16 rounding ties and amplify through
# posterior parameters. Keep cuDNN there; N96/N192 kernels serve C96/C192 stages.
_CUSTOM_CONV_CHANNELS = (96, 192)


def _compile(module):
    # Inductor's reduction tree and approximate division can cross BF16 ties.
    # Keep the remaining attention/head norms opaque, like residual preparation.
    def replace_norms(parent):
        for name, child in parent.named_children():
            if isinstance(child, RawWanRMSNorm):
                setattr(parent, name, _FusedRMSNorm(child.norm))
            else:
                replace_norms(child)

    replace_norms(module)
    return torch.compile(
        module,
        fullgraph=True,
        dynamic=False,
        options={"emulate_precision_casts": True, "eager_numerics.division_rounding": True},
    )
