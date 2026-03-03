"""
VLM Registry
Provides a registry pattern for VLM implementations
"""

from typing import Any, Dict, Optional, Type, Union
import logging

from .base import VLMBase

logger = logging.getLogger(__name__)


class VLMRegistry:
    """
    Registry for VLM implementations.

    Usage:
        @VLMRegistry.register("internvl")
        class InternVL(VLMBase):
            ...

        # Create VLM instance
        vlm = VLMRegistry.create("internvl", model_path="/path/to/model")
    """

    _registry: Dict[str, Type[VLMBase]] = {}
    _config_schemas: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        config_schema: Optional[Dict[str, Any]] = None,
    ):
        """
        Register a VLM implementation

        Args:
            name: Name identifier for the VLM
            config_schema: Optional configuration schema

        Returns:
            Decorator function
        """
        def decorator(vlm_class: Type[VLMBase]):
            if name in cls._registry:
                logger.warning(
                    f"VLM '{name}' is already registered. "
                    f"Overwriting with {vlm_class.__name__}"
                )
            cls._registry[name] = vlm_class

            if config_schema is not None:
                cls._config_schemas[name] = config_schema

            logger.info(f"Registered VLM: {name} -> {vlm_class.__name__}")
            return vlm_class

        return decorator

    @classmethod
    def create(
        cls,
        name: str,
        model_path: str,
        device: str = "cuda",
        **kwargs,
    ) -> VLMBase:
        """
        Create a VLM instance

        Args:
            name: Name of the VLM to create
            model_path: Path to model weights or HuggingFace model name
            device: Device to load model on
            **kwargs: Additional arguments for the VLM

        Returns:
            VLM instance

        Raises:
            ValueError: If VLM name is not registered
        """
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unknown VLM: '{name}'. Available VLMs: {available}"
            )

        vlm_class = cls._registry[name]
        return vlm_class(model_path=model_path, device=device, **kwargs)

    @classmethod
    def get_config_schema(cls, name: str) -> Optional[Dict[str, Any]]:
        """Get configuration schema for a VLM"""
        return cls._config_schemas.get(name)

    @classmethod
    def list_vlms(cls) -> list:
        """List all registered VLM names"""
        return list(cls._registry.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a VLM is registered"""
        return name in cls._registry

    @classmethod
    def unregister(cls, name: str):
        """Unregister a VLM"""
        if name in cls._registry:
            del cls._registry[name]
            if name in cls._config_schemas:
                del cls._config_schemas[name]
            logger.info(f"Unregistered VLM: {name}")


def create_vlm(
    name: str,
    model_path: str,
    device: str = "cuda",
    **kwargs,
) -> VLMBase:
    """
    Convenience function to create a VLM instance

    Args:
        name: Name of the VLM
        model_path: Path to model weights
        device: Device to load on
        **kwargs: Additional arguments

    Returns:
        VLM instance
    """
    return VLMRegistry.create(name, model_path, device, **kwargs)


# Import and register built-in VLMs
# These will be auto-registered when this module is imported
def _register_builtin_vlms():
    """Register built-in VLM implementations"""
    try:
        from .internvl import InternVL
        VLMRegistry.register("internvl")(
            InternVL
        )
    except ImportError:
        logger.warning("InternVL not available")

    try:
        from .qwen import QwenVL
        VLMRegistry.register("qwen")(
            QwenVL
        )
    except ImportError:
        logger.warning("QwenVL not available")


# Auto-register built-in VLMs
_register_builtin_vlms()
