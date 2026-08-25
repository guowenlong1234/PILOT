"""Predicted-q1 active-lookahead building blocks.

This package deliberately has no Habitat import at module import time.  Geometry,
model and ablation contracts can therefore be tested on a CPU-only machine.
"""

from .types import FutureHeadOutput

__all__ = [
    "FutureHeadOutput",
]
