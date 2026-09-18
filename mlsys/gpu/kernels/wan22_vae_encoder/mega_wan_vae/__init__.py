"""Wan VAE encoding with public CUTLASS DSL kernels.

Public API: from mega_wan_vae import MegaWanVaeEncoder.
Import lazily so standalone kernel modules do not initialize the whole model.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .encoder import MegaWanVaeEncoder

__all__ = ["MegaWanVaeEncoder"]


def __getattr__(name: str) -> type["MegaWanVaeEncoder"]:
    if name == "MegaWanVaeEncoder":
        from .encoder import MegaWanVaeEncoder

        globals()[name] = MegaWanVaeEncoder
        return MegaWanVaeEncoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
