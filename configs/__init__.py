"""
ReCogDrive Configuration Package
YAML-based configuration files for training
"""

from pathlib import Path

# Config directory
CONFIG_DIR = Path(__file__).parent

# Available configs
DEFAULT_CONFIG = CONFIG_DIR / "default.yaml"
STAGE_CONFIGS = {
    "stage1": CONFIG_DIR / "stage1.yaml",
    "stage2": CONFIG_DIR / "stage2.yaml",
    "stage3": CONFIG_DIR / "stage3.yaml",
}
VLM_CONFIGS = {
    "internvl3_8b": CONFIG_DIR / "vlm" / "internvl3_8b.yaml",
    "qwen2_vl_7b": CONFIG_DIR / "vlm" / "qwen2_vl_7b.yaml",
}

__all__ = [
    "CONFIG_DIR",
    "DEFAULT_CONFIG",
    "STAGE_CONFIGS",
    "VLM_CONFIGS",
]
