"""
QwenVL Implementation
Support for Qwen2-VL models
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


@VLMRegistry.register("qwen")
class QwenVL(VLMBase):
    """
    Qwen Vision-Language Model

    Supports:
    - Qwen2-VL-7B (hidden_dim=3584)
    - Qwen2-VL-72B (hidden_dim=3584)
    - QwenVL-7B (hidden_dim=4096)
    """

    # Model configurations
    MODEL_CONFIGS = {
        "qwen2-vl-7b": {"hidden_dim": 3584, "num_layers": 28},
        "qwen2-vl-72b": {"hidden_dim": 3584, "num_layers": 60},
        "qwenvl-7b": {"hidden_dim": 4096, "num_layers": 28},
    }

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        model_size: str = "7b",
        torch_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ):
        """
        Initialize QwenVL model

        Args:
            model_path: Path to model or HuggingFace model name
            device: Device to load model on
            model_size: Model size variant ("7b", "72b")
            torch_dtype: Model dtype
        """
        super().__init__(model_path, device, **kwargs)

        self.model_size = model_size
        self.torch_dtype = torch_dtype

        # Load model and tokenizer
        self._load_model()
        self._load_tokenizer()

        # Determine hidden dim
        model_key = f"qwen2-vl-{model_size}"
        self._hidden_dim = self.MODEL_CONFIGS.get(
            model_key, {}
        ).get("hidden_dim", 3584)

    def _load_model(self):
        """Load the QwenVL model"""
        try:
            from transformers import Qwen2VLForConditionalGeneration
        except ImportError:
            raise ImportError(
                "transformers>=4.37.0 is required for Qwen2-VL. "
                "Install with: pip install transformers>=4.37.0"
            )

        logger.info(f"Loading QwenVL model from {self.model_path}")

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"QwenVL model loaded successfully")

    def _load_tokenizer(self):
        """Load the tokenizer"""
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

    @property
    def name(self) -> str:
        return "qwen"

    def get_hidden_dim(self) -> int:
        return self._hidden_dim

    def get_tokenizer(self):
        return self.tokenizer

    def preprocess_image(
        self,
        images: Union[Image.Image, List[Image.Image], torch.Tensor],
        size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Preprocess images for QwenVL

        Args:
            images: Input images
            size: Target size (height, width)

        Returns:
            Preprocessed pixel values
        """
        from transformers import AutoProcessor

        if isinstance(images, torch.Tensor):
            return images.to(self.device)

        # Convert to list of PIL images
        if isinstance(images, Image.Image):
            images = [images]

        # Use AutoProcessor
        try:
            processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )
            pixel_values = processor(images=images, return_tensors="pt")
            return pixel_values["pixel_values"].to(self.device)
        except Exception as e:
            logger.warning(f"AutoProcessor failed: {e}, using manual processing")
            # Fallback to manual processing
            from torchvision import transforms

            if size is None:
                size = (448, 448)

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
        Encode images using QwenVL

        Args:
            pixel_values: Preprocessed image tensor
            prompts: Optional text prompts

        Returns:
            VLMOutput with hidden states
        """
        with torch.no_grad():
            # If prompts provided, use them
            if prompts is not None:
                # Tokenize prompts with image placeholder
                formatted_prompts = []
                for prompt in prompts:
                    formatted_prompt = f"<|im_start|>user\n<|image_1|>\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                    formatted_prompts.append(formatted_prompt)

                inputs = self.tokenizer(
                    formatted_prompts,
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
                # Image-only encoding
                outputs = self.model.vision_model(
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
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        # Prepare messages
        messages = []
        for prompt in prompts:
            messages.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ]
                }
            ])

        # Apply chat template
        texts = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Process inputs
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
