"""
VLM Base Classes
Defines the unified interface for Vision-Language Models
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn


@dataclass
class VLMOutput:
    """Output from VLM encoding"""
    hidden_states: torch.Tensor
    """Last hidden states, shape: [batch_size, seq_len, hidden_dim]"""

    logits: Optional[torch.Tensor] = None
    """Logits from language head, shape: [batch_size, seq_len, vocab_size]"""

    attention_mask: Optional[torch.Tensor] = None
    """Attention mask for the sequence"""

    pixel_values: Optional[torch.Tensor] = None
    """Processed pixel values"""

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-like access"""
        return getattr(self, key)

    def to(self, device: torch.device) -> "VLMOutput":
        """Move output to device"""
        return VLMOutput(
            hidden_states=self.hidden_states.to(device),
            logits=self.logits.to(device) if self.logits is not None else None,
            attention_mask=self.attention_mask.to(device) if self.attention_mask is not None else None,
            pixel_values=self.pixel_values.to(device) if self.pixel_values is not None else None,
        )


class VLMBase(ABC, nn.Module):
    """
    Base class for Vision-Language Models.

    All VLM implementations should inherit from this class and implement
    the required abstract methods.
    """

    def __init__(self, model_path: str, device: str = "cuda", **kwargs):
        """
        Initialize VLM

        Args:
            model_path: Path to model weights or HuggingFace model name
            device: Device to load model on ("cuda" or "cpu")
            **kwargs: Additional model-specific arguments
        """
        super().__init__()
        self.model_path = model_path
        self.device = torch.device(device)
        self._hidden_dim: Optional[int] = None

    @abstractmethod
    def encode_image(
        self,
        pixel_values: torch.Tensor,
        prompts: Optional[List[str]] = None,
    ) -> VLMOutput:
        """
        Encode images using the VLM

        Args:
            pixel_values: Image tensor, shape: [batch_size, channels, height, width]
            prompts: Optional text prompts for the images

        Returns:
            VLMOutput containing hidden states
        """
        pass

    @abstractmethod
    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> VLMOutput:
        """
        Encode text using the VLM

        Args:
            input_ids: Tokenized text, shape: [batch_size, seq_len]
            attention_mask: Attention mask

        Returns:
            VLMOutput containing hidden states
        """
        pass

    @abstractmethod
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        prompts: Optional[List[str]] = None,
    ) -> VLMOutput:
        """
        Full forward pass

        Args:
            pixel_values: Image tensor
            input_ids: Text token IDs
            attention_mask: Attention mask
            prompts: Text prompts (for image-to-text)

        Returns:
            VLMOutput containing hidden states
        """
        pass

    @abstractmethod
    def get_hidden_dim(self) -> int:
        """
        Get the hidden dimension of the model

        Returns:
            Hidden dimension size (e.g., 3584 for InternVL3-8B, 1536 for Qwen2-VL-7B)
        """
        pass

    @abstractmethod
    def get_tokenizer(self):
        """
        Get the tokenizer for this model

        Returns:
            Tokenizer object
        """
        pass

    @abstractmethod
    def preprocess_image(
        self,
        images: Any,
        size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Preprocess images for the model

        Args:
            images: Input images (PIL Image, numpy array, or tensor)
            size: Target size (height, width)

        Returns:
            Preprocessed pixel values tensor
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the VLM (e.g., 'internvl', 'qwen')"""
        pass

    @property
    def config(self) -> Dict[str, Any]:
        """Return model configuration"""
        return {
            "model_path": self.model_path,
            "device": str(self.device),
            "hidden_dim": self.get_hidden_dim(),
            "name": self.name,
        }

    def generate(
        self,
        pixel_values: torch.Tensor,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.9,
        **kwargs,
    ) -> List[str]:
        """
        Generate text from images

        Args:
            pixel_values: Image tensor
            prompts: Text prompts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Generated text strings
        """
        raise NotImplementedError("Generation not implemented for this VLM")

    def extract_features(
        self,
        pixel_values: torch.Tensor,
        layer: int = -1,
    ) -> torch.Tensor:
        """
        Extract features from a specific layer

        Args:
            pixel_values: Image tensor
            layer: Layer index (-1 for last layer)

        Returns:
            Feature tensor
        """
        output = self.encode_image(pixel_values)
        return output.hidden_states


class VLMCacheMixin:
    """Mixin class for VLM caching support"""

    def __init__(self, *args, cache_dir: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_dir = cache_dir
        self._cache: Dict[str, torch.Tensor] = {}

    def load_from_cache(self, key: str) -> Optional[torch.Tensor]:
        """Load features from cache"""
        if self.cache_dir is None:
            return None
        cache_path = f"{self.cache_dir}/{key}.pt"
        if torch.exists(cache_path):
            return torch.load(cache_path)
        return None

    def save_to_cache(self, key: str, features: torch.Tensor):
        """Save features to cache"""
        if self.cache_dir is None:
            return
        cache_path = f"{self.cache_dir}/{key}.pt"
        torch.save(features, cache_path)
