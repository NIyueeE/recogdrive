# --------------------------------------------------------
# QwenVL
# Copyright (c) 2024 Alibaba
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import logging
import math
import os
import random
import sys
import traceback
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

try:
    import orjson as json
except:
    import json

import torch
import torch.distributed as dist
import transformers
from PIL import Image, ImageFile, PngImagePlugin, UnidentifiedImageError
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.logging import (
    enable_default_handler,
    enable_explicit_format,
    set_verbosity,
)

# Try to import petrel_client for image loading, fallback to PIL if unavailable
try:
    from petrel_client.client import Client
    from petrel_client.common.config import Config
    has_tcs_loader = True
except ImportError as E:
    print('petrel_client is not installed. Using PIL to load images.')
    has_tcs_loader = False

# Set constants for image processing and logging
IGNORE_INDEX = -100
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


@dataclass
class ModelArguments:
    """
    Arguments for specifying model, tokenizer, and configurations.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    freeze_llm: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the LLM. Default is False.'},
    )
    freeze_vision: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the vision encoder. Default is False.'},
    )
    freeze_mlp: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the MLP projector. Default is False.'},
    )
    vision_select_layer: int = field(
        default=-1,
        metadata={'help': 'Specify the layer of vision feature map to use. Default is -1 for the last layer.'},
    )
    use_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the model. Default is 0.'}
    )
    grad_checkpoint: bool = field(
        default=True,
        metadata={'help': 'Set to True to use gradient checkpointing. Default is True.'},
    )
    drop_path_rate: float = field(
        default=0.0,
        metadata={'help': 'Set the drop path rate for the vision encoder. Default is 0.'},
    )
    conv_style: str = field(
        default='qwen',
        metadata={'help': 'Conversation style template. Default is qwen.'},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={'help': 'Whether to use fast tokenizer. Default is False.'},
    )
    force_image_size: Optional[int] = field(
        default=None,
        metadata={'help': 'Force image size for vision encoder. Default is None.'},
    )
    max_dynamic_patch: int = field(
        default=16,
        metadata={'help': 'Maximum number of dynamic patches. Default is 16.'},
    )
    down_sample_ratio: float = field(
        default=0.5,
        metadata={'help': 'Downsample ratio for visual tokens. Default is 0.5.'},
    )
    ps_version: Literal['v1', 'v2'] = field(
        default='v2',
        metadata={'help': 'Version of pixel shuffle. Default is v2.'},
    )


@dataclass
class DataArguments:
    """
    Arguments for specifying data, dataset, and preprocessing.
    """
    meta_path: str = field(
        default=None,
        metadata={'help': 'Path to the metadata file (JSON) containing dataset configuration.'}
    )
    data_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to the dataset root directory.'}
    )
    num_workers: int = field(
        default=32,
        metadata={'help': 'Number of workers for data loading. Default is 32.'}
    )
    max_seq_length: int = field(
        default=12288,
        metadata={'help': 'Maximum sequence length. Default is 12288.'}
    )
    dynamic_image_size: bool = field(
        default=True,
        metadata={'help': 'Whether to use dynamic image size. Default is True.'}
    )
    use_thumbnail: bool = field(
        default=True,
        metadata={'help': 'Whether to generate thumbnail for multi-tile images. Default is True.'}
    )
    pad2square: bool = field(
        default=False,
        metadata={'help': 'Whether to pad images to square. Default is False.'}
    )
    min_dynamic_patch: int = field(
        default=1,
        metadata={'help': 'Minimum number of dynamic patches. Default is 1.'}
    )
    conv_style: str = field(
        default='qwen',
        metadata={'help': 'Conversation style template. Default is qwen.'}
    )
    use_packed_ds: bool = field(
        default=False,
        metadata={'help': 'Whether to use packed dataset. Default is False.'}
    )
    group_by_length: bool = field(
        default=True,
        metadata={'help': 'Whether to group by length. Default is True.'}
    )


def preprocess_qwen(
    template_name: str,
    sources: List[Dict],
    processor: transformers.ProcessorMixin,
    num_image_token_list: List[int],
    text_only: bool = False,
    group_by_length: bool = False,
    use_packed_ds: bool = False,
    ds_name: str = None,
    num_image: int = 1,
) -> Dict:
    """
    Preprocess data for Qwen2.5-VL model.
    Adapted from InternVL's preprocess_internvl2_5 function.
    """
    assert len(sources) == 1, 'process only the first conversations'
    conversations = sources[0]

    if conversations[0]['from'] == 'system':
        system_prompt = conversations[0]['value']
        conversations = conversations[1:]  # remove system prompt
    else:
        # Use default system prompt for Qwen
        system_prompt = "You are a helpful assistant."

    if not text_only:
        new_conversations = []
        current_image_idx = 0
        for conversation in conversations:
            if conversation['from'] == 'human':
                image_cnt = conversation['value'].count('<image>')
                for i in range(image_cnt):
                    if current_image_idx == num_image:
                        break
                    # Qwen uses <image> placeholder, processor will handle it
                    # No need to replace with special tokens
                    current_image_idx += 1
            new_conversations.append(conversation)
        conversations = new_conversations
        assert current_image_idx == num_image, f'{current_image_idx} != {num_image}'

    # Build messages in ChatML format
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    for conversation in conversations:
        if conversation['from'] == 'human':
            messages.append({"role": "user", "content": conversation['value']})
        elif conversation['from'] == 'gpt':
            messages.append({"role": "assistant", "content": conversation['value']})
        else:
            raise NotImplementedError(f"Unknown role: {conversation['from']}")

    # Apply chat template
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize
    inputs = processor(
        text=[text],
        padding='max_length' if not group_by_length and not use_packed_ds else False,
        max_length=processor.tokenizer.model_max_length,
        truncation=True,
        return_tensors='pt'
    )

    # Prepare labels (ignore loss on human and system tokens)
    input_ids = inputs['input_ids'][0]
    attention_mask = inputs['attention_mask'][0]

    # Create labels (ignore index for non-assistant tokens)
    labels = input_ids.clone()
    # Find assistant token positions
    assistant_token_id = processor.tokenizer.convert_tokens_to_ids('<|im_start|>assistant')
    # Simple heuristic: mask everything before assistant start
    # This is a simplified approach; for production, need more sophisticated handling
    # For now, use the same approach as InternVL
    labels.fill_(IGNORE_INDEX)

    # Find the position of assistant start
    for i in range(len(input_ids) - 1):
        if input_ids[i] == assistant_token_id:
            # Start from after the assistant token
            start_idx = i + 1
            labels[start_idx:] = input_ids[start_idx:]
            break

    return {
        'input_ids': input_ids.unsqueeze(0),
        'attention_mask': attention_mask.unsqueeze(0),
        'labels': labels.unsqueeze(0),
    }


class QwenVLDataset(Dataset):
    """Dataset for Qwen2.5-VL fine-tuning."""

    def __init__(
        self,
        meta_path: str,
        processor: transformers.ProcessorMixin,
        data_path: Optional[str] = None,
        conv_style: str = 'qwen',
        max_seq_length: int = 12288,
        dynamic_image_size: bool = True,
        use_thumbnail: bool = True,
        min_dynamic_patch: int = 1,
        max_dynamic_patch: int = 16,
    ):
        super().__init__()
        self.processor = processor
        self.conv_style = conv_style
        self.max_seq_length = max_seq_length
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch

        # Load metadata
        with open(meta_path, 'r') as f:
            self.metadata = json.load(f)

        self.data_path = data_path
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Dict]:
        """Load and parse samples from metadata."""
        samples = []
        for dataset_config in self.metadata:
            dataset_name = dataset_config['dataset_name']
            json_path = dataset_config['json_path']
            repeat_time = dataset_config.get('repeat_time', 1)
            data_augment = dataset_config.get('data_augment', False)

            # Load dataset JSON
            dataset_json_path = json_path if self.data_path is None else os.path.join(self.data_path, json_path)
            with open(dataset_json_path, 'r') as f:
                dataset_data = json.load(f)

            for item in dataset_data:
                # Repeat based on repeat_time
                for _ in range(int(repeat_time)):
                    samples.append({
                        'dataset_name': dataset_name,
                        'item': item,
                        'data_augment': data_augment,
                    })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        sample = self.samples[index]
        item = sample['item']

        # Load image
        image_path = item.get('image')
        if image_path and self.data_path:
            image_path = os.path.join(self.data_path, image_path)

        # For simplicity, we'll handle image loading in collate_fn
        # Return the image path and conversation data
        conversations = item['conversations']

        return {
            'image_path': image_path,
            'conversations': conversations,
            'dataset_name': sample['dataset_name'],
        }


def qwenvl_collate_fn(batch: List[Dict], processor: transformers.ProcessorMixin) -> Dict:
    """Collate function for QwenVL dataset."""
    image_paths = []
    conversations_list = []

    for item in batch:
        image_paths.append(item['image_path'])
        conversations_list.append(item['conversations'])

    # Load images
    images = []
    for image_path in image_paths:
        if image_path and os.path.exists(image_path):
            image = Image.open(image_path).convert('RGB')
            images.append(image)
        else:
            # Use dummy image if path is invalid
            images.append(Image.new('RGB', (224, 224), color='gray'))

    # Prepare messages for each sample
    texts = []
    for conversations in conversations_list:
        messages = []
        # Check if first message is system
        if conversations[0]['from'] == 'system':
            messages.append({"role": "system", "content": conversations[0]['value']})
            conversations = conversations[1:]

        for conv in conversations:
            if conv['from'] == 'human':
                messages.append({"role": "user", "content": conv['value']})
            elif conv['from'] == 'gpt':
                messages.append({"role": "assistant", "content": conv['value']})

        # Apply chat template
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        texts.append(text)

    # Process with processor (handles both images and text)
    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        truncation=True,
        max_length=processor.tokenizer.model_max_length,
        return_tensors='pt'
    )

    # Prepare labels
    input_ids = inputs['input_ids']
    labels = input_ids.clone()

    # Mask non-assistant tokens
    assistant_token_id = processor.tokenizer.convert_tokens_to_ids('<|im_start|>assistant')
    for i in range(input_ids.shape[0]):
        seq = input_ids[i]
        # Find assistant start
        for j in range(len(seq) - 1):
            if seq[j] == assistant_token_id:
                start_idx = j + 1
                labels[i, :start_idx] = IGNORE_INDEX
                break
        else:
            # No assistant token found, mask everything
            labels[i, :] = IGNORE_INDEX

    return {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'pixel_values': inputs['pixel_values'],
        'labels': labels,
    }


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, distributed training: {training_args.parallel_mode.value == 'distributed'}, "
        f"16-bits training: {training_args.fp16 or training_args.bf16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed
    set_seed(training_args.seed)

    # Load processor and model
    logger.info(f"Loading Qwen2.5-VL model from {model_args.model_name_or_path}")
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if training_args.bf16 else "eager",
    )

    # Configure model based on arguments
    if model_args.grad_checkpoint:
        model.gradient_checkpointing_enable()

    # Freeze parameters if requested
    if model_args.freeze_vision:
        for param in model.vision_model.parameters():
            param.requires_grad = False
        logger.info("Vision encoder frozen")

    if model_args.freeze_llm:
        for param in model.language_model.parameters():
            param.requires_grad = False
        logger.info("Language model frozen")

    if model_args.freeze_mlp:
        # Qwen2.5-VL uses visual_abstractor as projector
        for param in model.visual_abstractor.parameters():
            param.requires_grad = False
        logger.info("Visual abstractor (MLP) frozen")

    # Prepare dataset
    logger.info(f"Loading dataset from {data_args.meta_path}")
    dataset = QwenVLDataset(
        meta_path=data_args.meta_path,
        processor=processor,
        data_path=data_args.data_path,
        conv_style=data_args.conv_style,
        max_seq_length=data_args.max_seq_length,
        dynamic_image_size=data_args.dynamic_image_size,
        use_thumbnail=data_args.use_thumbnail,
        min_dynamic_patch=data_args.min_dynamic_patch,
        max_dynamic_patch=data_args.max_dynamic_patch,
    )

    # Create collate function with processor
    collate_fn = partial(qwenvl_collate_fn, processor=processor)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
        tokenizer=processor.tokenizer,
    )

    # Training
    logger.info("*** Starting training ***")
    train_result = trainer.train()

    # Save model
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()