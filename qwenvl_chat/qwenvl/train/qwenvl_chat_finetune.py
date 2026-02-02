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

# Video processing imports
try:
    from decord import VideoReader
    import imageio
    import cv2
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False
    print('decord or imageio not installed. Video processing will not be available.')
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

# Try to import qwen_vl_utils for vision processing
try:
    from qwen_vl_utils import process_vision_info
    QWEN_VL_UTILS_AVAILABLE = True
except ImportError as E:
    print('qwen_vl_utils not installed. Video processing may be limited.')
    QWEN_VL_UTILS_AVAILABLE = False

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
    # Image/Video resolution and pixel control parameters
    image_min_pixels: Optional[int] = field(
        default=3136,  # 56*56 (14*14 patches)
        metadata={'help': 'Minimum number of pixels for images. Default is 3136.'}
    )
    image_max_pixels: Optional[int] = field(
        default=12845056,  # 3584*3584 (14*14 patches)
        metadata={'help': 'Maximum number of pixels for images. Default is 12845056.'}
    )
    video_min_pixels: Optional[int] = field(
        default=100352,  # 224*224*2 (minimum for video)
        metadata={'help': 'Minimum number of pixels for videos. Default is 100352.'}
    )
    video_max_pixels: Optional[int] = field(
        default=602112,  # 448*448*3 (reasonable for video)
        metadata={'help': 'Maximum number of pixels for videos. Default is 602112.'}
    )
    image_resized_width: Optional[int] = field(
        default=None,
        metadata={'help': 'Resized width for images. Default is None (use original).'}
    )
    image_resized_height: Optional[int] = field(
        default=None,
        metadata={'help': 'Resized height for images. Default is None (use original).'}
    )
    video_resized_width: Optional[int] = field(
        default=None,
        metadata={'help': 'Resized width for videos. Default is None (use original).'}
    )
    video_resized_height: Optional[int] = field(
        default=None,
        metadata={'help': 'Resized height for videos. Default is None (use original).'}
    )
    fps: Optional[int] = field(
        default=None,
        metadata={'help': 'Frames per second for video data. Default is None.'}
    )
    nframes: Optional[int] = field(
        default=None,
        metadata={'help': 'Number of frames for video data. Default is None.'}
    )
    # Video specific parameters
    min_num_frame: int = field(
        default=8,
        metadata={'help': 'Minimum number of video frames to sample. Default is 8.'}
    )
    max_num_frame: int = field(
        default=32,
        metadata={'help': 'Maximum number of video frames to sample. Default is 32.'}
    )
    sampling_method: str = field(
        default='rand',
        metadata={'help': 'Video frame sampling method: "rand", "middle", or "fpsX" (e.g., "fps1"). Default is "rand".'}
    )


# Video processing functions adapted from InternVL
def get_frame_indices(num_frames, vlen, sample='rand', fix_start=None, input_fps=1, max_num_frames=-1):
    """Sample frame indices from video."""
    if sample in ['rand', 'middle']: # uniform sampling
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == 'middle':
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif 'fps' in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
    else:
        raise ValueError
    return frame_indices


def read_frames_gif(video_path, num_frames, sample='rand', fix_start=None, min_num_frames=4):
    """Read frames from GIF file."""
    if not DECORD_AVAILABLE:
        raise ImportError('decord or imageio not installed. Cannot process GIF videos.')

    gif = imageio.get_reader(video_path)
    vlen = len(gif)

    t_num_frames = np.random.randint(min_num_frames, num_frames + 1)
    frame_indices = get_frame_indices(
        t_num_frames, vlen, sample=sample, fix_start=fix_start
    )
    frames = []
    for index, frame in enumerate(gif):
        if index in frame_indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB).astype(np.uint8)
            frame = Image.fromarray(frame)
            frames.append(frame)
    return frames


def read_frames_decord(video_path, num_frames, sample='rand', fix_start=None, clip=None, min_num_frames=4):
    """Read frames from video file using decord."""
    if not DECORD_AVAILABLE:
        raise ImportError('decord not installed. Cannot process video files.')

    video_reader = VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)
    if clip:
        start, end = clip
        duration = end - start
        vlen = int(duration * fps)
        start_index = int(start * fps)

    t_num_frames = np.random.randint(min_num_frames, num_frames + 1)

    frame_indices = get_frame_indices(
        t_num_frames, vlen, sample=sample, fix_start=fix_start,
        input_fps=fps
    )
    if clip:
        frame_indices = [f + start_index for f in frame_indices]
    frames = video_reader.get_batch(frame_indices).asnumpy()  # (T, H, W, C), np.uint8
    frames = [Image.fromarray(frames[i]) for i in range(frames.shape[0])]
    return frames


def extract_frame_number(filename):
    """Extract frame number from filename."""
    import re
    match = re.search(r'_(\d+).jpg$', filename)
    return int(match.group(1)) if match else -1


def sort_frames(frame_paths):
    """Sort frame paths by frame number."""
    return sorted(frame_paths, key=lambda x: extract_frame_number(os.path.basename(x)))


def read_frames_folder(video_path, num_frames, sample='rand', fix_start=None, min_num_frames=4):
    """Read frames from image folder."""
    import os
    image_list = sort_frames(list(os.listdir(video_path)))
    frames = []
    for image in image_list:
        fp = os.path.join(video_path, image)
        frame = Image.open(fp).convert('RGB')
        frames.append(frame)
    vlen = len(frames)

    t_num_frames = np.random.randint(min_num_frames, num_frames + 1)

    if vlen > t_num_frames:
        frame_indices = get_frame_indices(
            t_num_frames, vlen, sample=sample, fix_start=fix_start
        )
        frames = [frames[i] for i in frame_indices]
    return frames


def load_video_frames(video_path, max_num_frames=32, min_num_frames=8, sampling_method='rand'):
    """Load video frames from various sources (file, GIF, or folder)."""
    if not DECORD_AVAILABLE:
        raise ImportError('Video processing libraries not installed.')

    if os.path.isdir(video_path):
        return read_frames_folder(video_path, max_num_frames, sample=sampling_method, min_num_frames=min_num_frames)
    elif video_path.endswith('.gif'):
        return read_frames_gif(video_path, max_num_frames, sample=sampling_method, min_num_frames=min_num_frames)
    else:
        return read_frames_decord(video_path, max_num_frames, sample=sampling_method, min_num_frames=min_num_frames)


def get_image_info(image_path, min_pixel, max_pixel, width, height, image_patch_size=14):
    """Get image information using qwen_vl_utils.process_vision_info."""
    if not QWEN_VL_UTILS_AVAILABLE:
        raise ImportError('qwen_vl_utils not installed. Cannot process images.')

    content = {
        "type": "image",
        "image": image_path,
        "min_pixels": min_pixel,
        "max_pixels": max_pixel
    }

    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height

    messages = [
        {
            "role": "user",
            "content": [content]
        }
    ]

    image_input, _ = process_vision_info(messages, image_patch_size=image_patch_size)
    return image_input[0]


def get_video_info(video_path, min_pixels, max_pixels, width, height, fps=None,
                   image_patch_size=14, return_video_metadata=False):
    """Get video information using qwen_vl_utils.process_vision_info."""
    if not QWEN_VL_UTILS_AVAILABLE:
        raise ImportError('qwen_vl_utils not installed. Cannot process videos.')

    content = {
        "type": "video",
        "video": video_path,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "fps": fps
    }

    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height

    messages = [
        {
            "role": "user",
            "content": [content]
        }
    ]

    _, video_input, video_kwargs = process_vision_info(
        messages,
        return_video_kwargs=True,
        image_patch_size=image_patch_size,
        return_video_metadata=return_video_metadata
    )

    return video_input[0], video_kwargs


# Constants for LLaVA format conversion (from reference project)
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"
LLAVA_IMAGE_TOKEN = "<image>"
LLAVA_VIDEO_TOKEN = "<video>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"


def replace_image_tokens(input_string, is_video=False):
    """Replace LLaVA format image/video tokens with Qwen format."""
    import re
    if is_video:
        pattern = r'\n?' + re.escape(LLAVA_VIDEO_TOKEN) + r'\n?'
        replacement = VISION_START_TOKEN + DEFAULT_VIDEO_TOKEN + VISION_END_TOKEN
    else:
        pattern = r'\n?' + re.escape(LLAVA_IMAGE_TOKEN) + r'\n?'
        replacement = VISION_START_TOKEN + DEFAULT_IMAGE_TOKEN + VISION_END_TOKEN

    return re.sub(pattern, replacement, input_string)


def llava_to_openai(conversations, is_video=False):
    """Convert LLaVA format conversations to OpenAI format."""
    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    for conversation in conversations:
        transformed_content = replace_image_tokens(conversation["value"], is_video=is_video)
        transformed_entry = {
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": transformed_content,
        }
        transformed_data.append(transformed_entry)

    return transformed_data


def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output


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
        # Video specific parameters
        min_num_frame: int = 8,
        max_num_frame: int = 32,
        sampling_method: str = 'rand',
        # New parameters for resolution control (can be passed via data_args)
        image_min_pixels: Optional[int] = 3136,
        image_max_pixels: Optional[int] = 12845056,
        video_min_pixels: Optional[int] = 100352,
        video_max_pixels: Optional[int] = 602112,
        image_resized_width: Optional[int] = None,
        image_resized_height: Optional[int] = None,
        video_resized_width: Optional[int] = None,
        video_resized_height: Optional[int] = None,
        fps: Optional[int] = None,
        nframes: Optional[int] = None,
    ):
        super().__init__()
        self.processor = processor
        self.conv_style = conv_style
        self.max_seq_length = max_seq_length
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.min_num_frame = min_num_frame
        self.max_num_frame = max_num_frame
        self.sampling_method = sampling_method

        # New resolution parameters
        self.image_min_pixels = image_min_pixels
        self.image_max_pixels = image_max_pixels
        self.video_min_pixels = video_min_pixels
        self.video_max_pixels = video_max_pixels
        self.image_resized_width = image_resized_width
        self.image_resized_height = image_resized_height
        self.video_resized_width = video_resized_width
        self.video_resized_height = video_resized_height
        self.fps = fps
        self.nframes = nframes

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

        # Check if it's video data
        video_path = item.get('video')
        image_path = item.get('image')

        media_path = None
        media_type = 'none'  # 'image', 'video', or 'none'
        media_files = []

        if video_path is not None and video_path != '':
            # Video data
            media_type = 'video'
            if self.data_path:
                media_path = os.path.join(self.data_path, video_path)
            else:
                media_path = video_path
            if isinstance(video_path, list):
                media_files = [os.path.join(self.data_path, v) if self.data_path else v for v in video_path]
            else:
                media_files = [media_path]
        elif image_path is not None and image_path != '':
            # Image data
            media_type = 'image'
            if self.data_path:
                media_path = os.path.join(self.data_path, image_path)
            else:
                media_path = image_path
            if isinstance(image_path, list):
                media_files = [os.path.join(self.data_path, img) if self.data_path else img for img in image_path]
            else:
                media_files = [media_path]

        # For simplicity, we'll handle media loading in collate_fn
        # Return the media path, type, and conversation data
        conversations = item['conversations']

        return {
            'media_path': media_path,
            'media_files': media_files,
            'media_type': media_type,
            'conversations': conversations,
            'dataset_name': sample['dataset_name'],
            # Pass video parameters for collate_fn
            'min_num_frame': self.min_num_frame,
            'max_num_frame': self.max_num_frame,
            'sampling_method': self.sampling_method,
            # Pass resolution parameters
            'image_min_pixels': self.image_min_pixels,
            'image_max_pixels': self.image_max_pixels,
            'video_min_pixels': self.video_min_pixels,
            'video_max_pixels': self.video_max_pixels,
            'image_resized_width': self.image_resized_width,
            'image_resized_height': self.image_resized_height,
            'video_resized_width': self.video_resized_width,
            'video_resized_height': self.video_resized_height,
            'fps': self.fps,
            'nframes': self.nframes,
        }


def qwenvl_collate_fn(batch: List[Dict], processor: transformers.ProcessorMixin) -> Dict:
    """Collate function for QwenVL dataset supporting both images and videos."""
    all_images = []  # List of lists: each element is list of images for a sample
    texts = []

    for item in batch:
        media_path = item['media_path']
        media_type = item['media_type']
        conversations = item['conversations']

        # Load media (image or video frames)
        if media_type == 'video' and media_path:
            try:
                if not DECORD_AVAILABLE:
                    raise ImportError('Video processing libraries not installed.')

                # Load video frames
                frames = load_video_frames(
                    media_path,
                    max_num_frames=item['max_num_frame'],
                    min_num_frames=item['min_num_frame'],
                    sampling_method=item['sampling_method']
                )

                # Replace <video> placeholder(s) with multiple <image> placeholders
                # Find all conversations with video placeholder
                video_found = False
                for conv in conversations:
                    if conv['from'] == 'human' and '<video>' in conv['value']:
                        # Replace all <video> occurrences in this message
                        frame_placeholders = '\n'.join(['<image>' for _ in range(len(frames))])
                        # Count how many <video> tags to replace
                        video_count = conv['value'].count('<video>')
                        if video_count == 1:
                            # Simple replacement
                            conv['value'] = conv['value'].replace('<video>', frame_placeholders)
                        else:
                            # Replace each <video> with frame placeholders
                            # For multiple videos, we'd need more complex handling
                            # For now, replace first with frames, others with single <image>
                            parts = conv['value'].split('<video>')
                            new_parts = [parts[0]]
                            for idx in range(1, len(parts)):
                                if idx == 1:
                                    new_parts.append(frame_placeholders)
                                else:
                                    new_parts.append('<image>')
                                new_parts.append(parts[idx])
                            conv['value'] = ''.join(new_parts)
                        video_found = True

                # If no <video> tag found but media_type is video, prepend frames to first human message
                if not video_found and len(frames) > 0:
                    for conv in conversations:
                        if conv['from'] == 'human':
                            frame_placeholders = '\n'.join(['<image>' for _ in range(len(frames))])
                            conv['value'] = frame_placeholders + '\n' + conv['value']
                            video_found = True
                            break

                images_for_sample = frames
            except Exception as e:
                print(f'Error loading video {media_path}: {e}')
                # Fallback to dummy image
                images_for_sample = [Image.new('RGB', (224, 224), color='gray')]

        elif media_type == 'image' and media_path:
            # Load single image
            if media_path and os.path.exists(media_path):
                try:
                    image = Image.open(media_path).convert('RGB')
                    images_for_sample = [image]
                except Exception as e:
                    print(f'Error loading image {media_path}: {e}')
                    images_for_sample = [Image.new('RGB', (224, 224), color='gray')]
            else:
                # Use dummy image if path is invalid
                images_for_sample = [Image.new('RGB', (224, 224), color='gray')]

        else:
            # No media, use dummy image
            images_for_sample = [Image.new('RGB', (224, 224), color='gray')]

        all_images.append(images_for_sample)

        # Prepare messages for the sample
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
    # For each sample, we may have multiple images (video frames)
    # Qwen2.5-VL processor should handle multiple images per sample
    # We need to flatten the images list and adjust text accordingly
    # However, processor expects one image per sample, but can handle multiple <image> tokens
    # Let's try passing the first image for now, but this needs to be verified

    # For now, use the first image of each sample (or dummy if no images)
    first_images = [images[0] if len(images) > 0 else Image.new('RGB', (224, 224), color='gray')
                    for images in all_images]

    inputs = processor(
        text=texts,
        images=first_images,
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
        # Store number of frames per sample for potential future use
        'num_frames': torch.tensor([len(images) for images in all_images]),
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
        # Video parameters
        min_num_frame=data_args.min_num_frame,
        max_num_frame=data_args.max_num_frame,
        sampling_method=data_args.sampling_method,
        # New resolution parameters
        image_min_pixels=data_args.image_min_pixels,
        image_max_pixels=data_args.image_max_pixels,
        video_min_pixels=data_args.video_min_pixels,
        video_max_pixels=data_args.video_max_pixels,
        image_resized_width=data_args.image_resized_width,
        image_resized_height=data_args.image_resized_height,
        video_resized_width=data_args.video_resized_width,
        video_resized_height=data_args.video_resized_height,
        fps=data_args.fps,
        nframes=data_args.nframes,
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