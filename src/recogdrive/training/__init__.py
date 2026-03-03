"""
ReCogDrive Training Package
Unified training entry points for all stages
"""

from .stage1_vlm import main as stage1_main
from .stage2_dit import main as stage2_main
from .stage3_rl import main as stage3_main

__all__ = [
    "stage1_main",
    "stage2_main",
    "stage3_main",
]
