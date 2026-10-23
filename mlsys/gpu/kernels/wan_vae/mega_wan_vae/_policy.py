"""Framework policy for full models that include ordinary Torch convolutions.

Primitive convolutions have explicit per-operation precision. Full models also
contain Torch convolutions, whose policy is process-wide. Configure that policy
before construction and keep it stable during inference, including graph replay.
No inference call mutates backend flags.
"""

import torch


def model_tf32_policy(dtype: torch.dtype, allow_tf32: bool | None) -> bool | None:
    if allow_tf32 is not None and type(allow_tf32) is not bool:
        raise TypeError("allow_tf32 must be bool or None")
    if dtype != torch.float32:
        return None
    selected = torch.backends.cudnn.allow_tf32 if allow_tf32 is None else allow_tf32
    check_model_policy(selected)
    return selected


def check_model_policy(allow_tf32: bool | None) -> None:
    if allow_tf32 is not None and torch.backends.cudnn.allow_tf32 != allow_tf32:
        raise ValueError(
            f"This model requires torch.backends.cudnn.allow_tf32={allow_tf32}; "
            "configure the framework before construction and keep its policy stable during inference"
        )
