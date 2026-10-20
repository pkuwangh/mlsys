"""Custom Qwen3-TTS talker and code-predictor modules."""

from .predictor import MegaQwen3TTSPredictor
from .talker import MegaQwen3TTSTalker

__all__ = ["MegaQwen3TTSPredictor", "MegaQwen3TTSTalker"]
