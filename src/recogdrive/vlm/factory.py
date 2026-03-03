"""
VLM Factory
Factory functions for creating VLM instances
"""

from typing import Any, Dict, Optional, Union
import logging
import torch

from .base import VLMBase
from .registry import VLMRegistry

logger = logging.getLogger(__name__)


class VLMFactory:
    """
    Factory for creating VLM instances with configuration.
    """

    @staticmethod
    def create(
        vlm_type: str,
        model_path: str,
        device: str = "cuda",
        **kwargs,
    ) -> VLMBase:
        """
        Create a VLM instance.

        Args:
            vlm_type: Type of VLM ('internvl', 'qwen', etc.)
            model_path: Path to model checkpoint
            device: Device to load model on
            **kwargs: Additional arguments

        Returns:
            VLMBase instance
        """
        logger.info(f"Creating VLM: type={vlm_type}, path={model_path}")

        # Use registry to create VLM
        vlm = VLMRegistry.create(
            name=vlm_type,
            model_path=model_path,
            device=device,
            **kwargs,
        )

        logger.info(f"VLM created successfully: {vlm.name}, hidden_dim={vlm.get_hidden_dim()}")
        return vlm

    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> VLMBase:
        """
        Create a VLM instance from configuration dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            VLMBase instance
        """
        vlm_type = config.get("vlm_type", "internvl")
        model_path = config.get("vlm_model_path", "")
        device = config.get("device", "cuda")

        # Extract additional kwargs
        kwargs = {}
        for key in ["model_size", "use_fast_tokenizer", "torch_dtype"]:
            if key in config:
                kwargs[key] = config[key]

        return VLMFactory.create(vlm_type, model_path, device, **kwargs)

    @staticmethod
    def list_available() -> list:
        """List all available VLM types"""
        return VLMRegistry.list_vlms()


# Convenience function
def create_vlm(
    vlm_type: str,
    model_path: str,
    device: str = "cuda",
    **kwargs,
) -> VLMBase:
    """
    Convenience function to create a VLM instance.

    Args:
        vlm_type: Type of VLM
        model_path: Path to model checkpoint
        device: Device to load on
        **kwargs: Additional arguments

    Returns:
        VLMBase instance
    """
    return VLMFactory.create(vlm_type, model_path, device, **kwargs)


def create_vlm_from_config(config: Dict[str, Any]) -> VLMBase:
    """
    Convenience function to create a VLM from config.

    Args:
        config: Configuration dictionary

    Returns:
        VLMBase instance
    """
    return VLMFactory.create_from_config(config)
