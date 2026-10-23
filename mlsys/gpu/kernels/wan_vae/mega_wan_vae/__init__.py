"""Wan VAE encoding and decoding with public CUTLASS DSL kernels.

Public API: MegaWanVaeEncoder and MegaWanVaeDecoder.
Import lazily so standalone kernel modules do not initialize the whole model.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .decoder import MegaWanVaeDecoder
    from .encoder import MegaWanVaeEncoder

__all__ = ["MegaWanVaeDecoder", "MegaWanVaeEncoder"]


def __getattr__(name: str) -> type["MegaWanVaeEncoder"] | type["MegaWanVaeDecoder"]:
    if name == "MegaWanVaeEncoder":
        from .encoder import MegaWanVaeEncoder

        globals()[name] = MegaWanVaeEncoder
        return MegaWanVaeEncoder
    if name == "MegaWanVaeDecoder":
        from .decoder import MegaWanVaeDecoder

        globals()[name] = MegaWanVaeDecoder
        return MegaWanVaeDecoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
