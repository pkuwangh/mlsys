# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Storage dtype policy shared by Wan pointwise kernel compilation."""

import cutlass
import torch


def storage_type(dtype: torch.dtype) -> type[cutlass.Numeric]:
    """Select compile-time storage without inspecting device tensor values."""
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    if dtype == torch.float32:
        return cutlass.Float32
    raise TypeError("Wan kernels support torch.bfloat16 and torch.float32 storage")
