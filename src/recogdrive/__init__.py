"""
ReCogDrive
Unified training framework for Vision-Language Model based autonomous driving
"""

__version__ = "0.1.0"

from .vlm import VLMBase, VLMRegistry, create_vlm
from .config import TrainingConfig, Stage, ConfigLoader

__all__ = [
    "__version__",
    "VLMBase",
    "VLMRegistry",
    "create_vlm",
    "TrainingConfig",
    "Stage",
    "ConfigLoader",
]
