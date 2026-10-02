"""Small host-side helpers for Wan weights, patchification, and cache ownership."""

import copy
from typing import TypeVar

import torch

ModuleT = TypeVar("ModuleT", bound=torch.nn.Module)
# Cache tensors belong to one temporal stream; the reference also uses a sentinel.
CacheEntry = torch.Tensor | str | None
FeatCache = tuple[CacheEntry, ...]


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Pack each spatial patch into channels, preserving Wan channel order."""
    if patch_size == 1:
        return x
    if x.dim() != 5:
        raise ValueError(f"Invalid input shape: {x.shape}")

    batch_size, channels, frames, height, width = x.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"Height ({height}) and width ({width}) must be divisible by patch_size ({patch_size})."
        )

    x = x.view(
        batch_size,
        channels,
        frames,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
    )
    x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    return x.view(
        batch_size,
        channels * patch_size * patch_size,
        frames,
        height // patch_size,
        width // patch_size,
    )


def validate_frame_count(frames: int) -> None:
    """Require one initial frame followed by complete four-frame chunks."""
    if frames != 1 and (frames - 1) % 4 != 0:
        raise ValueError(
            f"Input video frames must be 1 or 4n+1 for Wan VAE temporal chunking, got {frames}."
        )


def update_cache(feat_cache: FeatCache, idx: int, value: CacheEntry) -> FeatCache:
    """Replace one stream-owned history entry without mutating the input tuple."""
    next_cache = list(feat_cache)
    next_cache[idx] = value
    return tuple(next_cache)


def canonicalize_cache(
    feat_cache: FeatCache,
    memory_format: torch.memory_format = torch.channels_last_3d,
) -> FeatCache:
    """Densify cache views in the owning encoder's storage layout."""
    return tuple(
        entry.contiguous(memory_format=memory_format)
        if isinstance(entry, torch.Tensor)
        else entry
        for entry in feat_cache
    )


def with_channels_last_weights(module: ModuleT) -> ModuleT:
    """Reuse a prepared module, or copy and prepack its convolution weights.

    Preparation happens only during setup and never modifies the input module.
    Conv2d/Conv3d weights preserve channels-last outputs even for singleton-frame
    activations with ambiguous memory-format strides. Prepared modules can be
    shared by the inference wrappers without another copy or conversion.
    """
    if all(
        layer.weight.is_contiguous(
            memory_format=(
                torch.channels_last_3d
                if isinstance(layer, torch.nn.Conv3d)
                else torch.channels_last
            )
        )
        for layer in module.modules()
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Conv3d))
    ):
        return module
    packed = copy.deepcopy(module)
    torch.nn.utils.convert_conv3d_weight_memory_format(packed, torch.channels_last_3d)
    torch.nn.utils.convert_conv2d_weight_memory_format(packed, torch.channels_last)
    return packed
