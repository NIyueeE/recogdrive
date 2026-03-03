"""
InternVL Implementation
Support for InternVL3-8B and other InternVL models
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

from .base import VLMBase, VLMOutput
from .registry import VLMRegistry

logger = logging.getLogger(__name__)


@VLMRegistry.register("internvl")
class InternVL(VLMBase):
    """
    InternVL Vision-Language Model

    Supports:
    - InternVL3-8B (hidden_dim=3584)
    - InternVL2-8B (hidden_dim=3584)
    - InternVL2-5B (hidden_dim=2560)
    """

    # Model configurations
    MODEL_CONFIGS = {
        "internvl3-8b": {"hidden_dim": 3584, "num_layers": 32},
        "internvl2-8b": {"hidden_dim": 3584, "num_layers": 32},
        "internvl2-5b": {"hidden_dim": 2560, "num_layers": 24},
    }

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        model_size: str = "8b",
        use_fast_tokenizer: bool = False,
        torch_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ):
        """
        Initialize InternVL model

        Args:
            model_path: Path to model or HuggingFace model name
            device: Device to load model on
            model_size: Model size variant ("8b", "5b")
            use_fast_tokenizer: Use fast tokenizer
            torch_dtype: Model dtype
        """
        super().__init__(model_path, device, **kwargs)

        self.model_size = model_size
        self.use_fast_tokenizer = use_fast_tokenizer
        self.torch_dtype = torch_dtype

        # Load model and tokenizer
        self._load_model()
        self._load_tokenizer()

        # Determine hidden dim
        model_key = f"internvl2-{model_size}" if model_size != "8b" else "internvl3-8b"
        self._hidden_dim = self.MODEL_CONFIGS.get(
            model_key, {}
        ).get("hidden_dim", 3584)

    def _load_model(self):
        """Load the InternVL model"""
        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError(
                "transformers package is required. "
                "Install with: pip install transformers"
            )

        logger.info(f"Loading InternVL model from {self.model_path}")

        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"InternVL model loaded successfully")

    def _load_tokenizer(self):
        """Load the tokenizer"""
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=self.use_fast_tokenizer,
        )

        # Add special tokens
        self._add_special_tokens()

    def _add_special_tokens(self):
        """Add special tokens for image handling"""
        # Check if image token exists
        special_tokens = ["<image>", "</image>", "<IMG_CONTEXT>"]
        needs_addition = False

        for token in special_tokens:
            if token not in self.tokenizer.get_vocab():
                needs_addition = True
                break

        if needs_addition:
            num_new_tokens = self.tokenizer.add_special_tokens({
                "additional_special_tokens": special_tokens
            })
            if num_new_tokens > 0:
                self.model.resize_token_embeddings(len(self.tokenizer))

    @property
    def name(self) -> str:
        return "internvl"

    def get_hidden_dim(self) -> int:
        return self._hidden_dim

    def get_tokenizer(self):
        return self.tokenizer

    def preprocess_image(
        self,
        images: Union[Image.Image, List[Image.Image], torch.Tensor],
        size: Optional[Tuple[int, int]] = (448, 448),
    ) -> torch.Tensor:
        """
        Preprocess images for InternVL

        Args:
            images: Input images
            size: Target size (height, width)

        Returns:
            Preprocessed pixel values
        """
        from transformers import AutoProcessor
        from torchvision import transforms

        if isinstance(images, torch.Tensor):
            # Already a tensor
            if images.dim() == 4:
                # [B, C, H, W] - normalize
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                images = images / 255.0
                images = (images - mean) / std
            return images.to(self.device)

        # Convert to list of PIL images
        if isinstance(images, Image.Image):
            images = [images]

        # Use AutoProcessor if available
        try:
            processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )
            # Process images
            pixel_values = processor(images=images, return_tensors="pt")
            return pixel_values["pixel_values"].to(self.device)
        except Exception:
            # Fallback to manual processing
            transform = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

            processed = [transform(img) for img in images]
            pixel_values = torch.stack(processed)
            return pixel_values.to(self.device)

    def encode_image(
        self,
        pixel_values: torch.Tensor,
        prompts: Optional[List[str]] = None,
    ) -> VLMOutput:
        """
        Encode images using InternVL

        Args:
            pixel_values: Preprocessed image tensor
            prompts: Optional text prompts

        Returns:
            VLMOutput with hidden states
        """
        with torch.no_grad():
            # If prompts provided, use them
            if prompts is not None:
                # Tokenize prompts
                inputs = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048,
                )
                input_ids = inputs["input_ids"].to(self.device)
                attention_mask = inputs["attention_mask"].to(self.device)

                # Forward pass
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

                hidden_states = outputs.hidden_states[-1]
            else:
                # Image-only encoding (CLIP-like)
                outputs = self.model(
                    pixel_values=pixel_values,
                    output_hidden_states=True,
                )
                hidden_states = outputs.hidden_states[-1]

        return VLMOutput(
            hidden_states=hidden_states,
            logits=outputs.logits if hasattr(outputs, "logits") else None,
            pixel_values=pixel_values,
        )

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> VLMOutput:
        """Encode text"""
        with torch.no_grad():
            outputs = self.model.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states[-1]

        return VLMOutput(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        prompts: Optional[List[str]] = None,
    ) -> VLMOutput:
        """Full forward pass"""
        if pixel_values is not None:
            return self.encode_image(pixel_values, prompts)
        elif input_ids is not None:
            return self.encode_text(input_ids, attention_mask)
        else:
            raise ValueError("Either pixel_values or input_ids must be provided")

    def generate(
        self,
        pixel_values: torch.Tensor,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.9,
        **kwargs,
    ) -> List[str]:
        """Generate text from images"""
        # Use chat template
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        # Prepare inputs
        conversations = []
        for prompt in prompts:
            conversations.append([
                {"role": "user", "content": f"<image>\n{prompt}"}
            ])

        # Apply chat template
        texts = processor.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Process images and text
        inputs = processor(
            text=texts,
            images=pixel_values,
            return_tensors="pt",
            padding=True,
        )

        # Move to device
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        pixel_values = inputs["pixel_values"].to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                **kwargs,
            )

        # Decode
        generated_texts = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            generated_texts.append(text)

        return generated_texts
