"""
Configuration Base Classes
Defines the unified configuration structure
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class Stage(str, Enum):
    """Training stages"""
    STAGE1 = "stage1"  # VLM supervised fine-tuning
    STAGE2 = "stage2"  # DiT imitation learning
    STAGE3 = "stage3"  # DiffGRPO reinforcement learning


class VLMSize(str, Enum):
    """VLM size variants"""
    SMALL = "small"
    LARGE = "large"


class DiTType(str, Enum):
    """DiT model types"""
    SMALL = "small"
    LARGE = "large"


class SamplingMethod(str, Enum):
    """Diffusion sampling methods"""
    DDPM = "ddpm"
    DDIM = "ddim"
    FLOW = "flow"


@dataclass
class VLMConfig:
    """VLM configuration"""
    # Model configuration
    vlm_type: str = "internvl"
    """VLM type (internvl, qwen, etc.)"""

    vlm_model_path: str = ""
    """Path to VLM model weights"""

    vlm_size: str = "large"
    """VLM size (small, large)"""

    # Model-specific settings
    use_fast_tokenizer: bool = False
    """Use fast tokenizer"""

    torch_dtype: str = "bfloat16"
    """Model dtype (float16, bfloat16, float32)"""

    # Training settings
    freeze_llm: bool = False
    """Freeze language model"""

    freeze_mlp: bool = False
    """Freeze MLP projector"""

    freeze_backbone: bool = False
    """Freeze vision backbone"""

    # Vision settings
    force_image_size: int = 448
    """Image size for vision encoder"""

    max_dynamic_patch: int = 16
    """Maximum dynamic patches"""

    down_sample_ratio: float = 0.5
    """Vision token downsample ratio"""

    def get_hidden_dim(self) -> int:
        """Get VLM hidden dimension"""
        if self.vlm_size == "large":
            return 3584
        else:
            return 1536


@dataclass
class DiTConfig:
    """DiT (Diffusion Transformer) configuration"""
    # Model architecture
    dit_type: str = "small"
    """DiT model type (small, large)"""

    input_embedding_dim: int = 384
    """Input embedding dimension"""

    action_dim: int = 3
    """Action dimension (x, y, yaw)"""

    action_horizon: int = 8
    """Number of future action steps"""

    # Diffusion settings
    sampling_method: str = "ddim"
    """Sampling method (ddpm, ddim, flow)"""

    num_inference_steps: int = 5
    """Number of inference steps"""

    # Training settings
    dropout: float = 0.0
    """Dropout rate"""

    def get_input_dim(self) -> int:
        """Get input embedding dimension based on dit_type"""
        if self.dit_type == "large":
            return 1536
        else:
            return 384


@dataclass
class RLConfig:
    """Reinforcement Learning configuration"""
    # GRPO settings
    grpo: bool = False
    """Enable GRPO training"""

    metric_cache_path: str = ""
    """Path to PDM metric cache"""

    reference_policy_checkpoint: str = ""
    """Path to reference policy checkpoint"""

    # Training parameters
    num_samples: int = 8
    """Number of trajectory samples for GRPO"""

    kl_coef: float = 0.1
    """KL divergence coefficient"""

    gamma_denoising: float = 0.6
    """Denoising gamma for advantage estimation"""

    # Clipping
    denoised_clip_value: float = 1.0
    """Clip value for denoised trajectories"""

    randn_clip_value: float = 5.0
    """Clip value for sampling noise"""

    clip_advantage_lower_quantile: float = 0.0
    """Lower quantile for advantage clipping"""

    clip_advantage_upper_quantile: float = 1.0
    """Upper quantile for advantage clipping"""


@dataclass
class DataConfig:
    """Data configuration"""
    # Paths
    data_path: str = ""
    """Path to training data"""

    meta_path: str = ""
    """Path to dataset metadata JSON"""

    cache_dir: str = "./cache"
    """Cache directory for features"""

    output_dir: str = "./outputs"
    """Output directory"""

    # Dataset settings
    max_seq_length: int = 12288
    """Maximum sequence length"""

    group_by_length: bool = True
    """Group samples by length"""

    dynamic_image_size: bool = True
    """Use dynamic image size"""

    use_thumbnail: bool = True
    """Generate thumbnail for multi-tile images"""

    dataloader_num_workers: int = 32
    """Number of dataloader workers"""


@dataclass
class TrainingConfig:
    """
    Unified training configuration

    This configuration combines all aspects of training:
    - VLM settings
    - DiT settings
    - RL settings
    - Data settings
    - Training hyperparameters
    """
    # Stage selection
    stage: str = Stage.STAGE1.value
    """Training stage (stage1, stage2, stage3)"""

    # Hardware configuration
    num_gpus: int = 8
    """Number of GPUs"""

    num_nodes: int = 1
    """Number of nodes for distributed training"""

    node_rank: int = 0
    """Current node rank"""

    master_addr: str = "localhost"
    """Master node address"""

    master_port: int = 29500
    """Master node port"""

    # Training hyperparameters
    learning_rate: float = 1e-4
    """Learning rate"""

    weight_decay: float = 0.05
    """Weight decay"""

    warmup_ratio: float = 0.1
    """Learning rate warmup ratio"""

    num_train_epochs: int = 3
    """Number of training epochs"""

    per_device_batch_size: int = 1
    """Batch size per device"""

    gradient_accumulation_steps: int = 16
    """Gradient accumulation steps"""

    max_steps: int = -1
    """Maximum training steps (-1 for epoch-based)"""

    # Mixed precision
    bf16: bool = True
    """Use bfloat16"""

    fp16: bool = False
    """Use float16"""

    # Optimization
    grad_checkpoint: bool = True
    """Enable gradient checkpointing"""

    max_grad_norm: float = 1.0
    """Maximum gradient norm for clipping"""

    # Logging
    logging_steps: int = 1
    """Log every N steps"""

    save_steps: int = 500
    """Save checkpoint every N steps"""

    eval_steps: int = 500
    """Evaluate every N steps"""

    report_to: str = "tensorboard"
    """Report to (tensorboard, wandb, etc.)"""

    # DeepSpeed
    deepspeed_config: Optional[str] = None
    """Path to DeepSpeed config"""

    # Sub-configs
    vlm: VLMConfig = field(default_factory=VLMConfig)
    """VLM configuration"""

    dit: DiTConfig = field(default_factory=DiTConfig)
    """DiT configuration"""

    rl: RLConfig = field(default_factory=RLConfig)
    """RL configuration"""

    data: DataConfig = field(default_factory=DataConfig)
    """Data configuration"""

    # Extra parameters
    extra: Dict[str, Any] = field(default_factory=dict)
    """Extra parameters"""

    def __post_init__(self):
        """Validate and set defaults"""
        # Set input_embedding_dim based on dit_type
        if self.dit.input_embedding_dim == 384 or self.dit.input_embedding_dim == 1536:
            pass  # Already set
        else:
            self.dit.input_embedding_dim = self.dit.get_input_dim()

        # Validate stage
        valid_stages = [s.value for s in Stage]
        if self.stage not in valid_stages:
            raise ValueError(
                f"Invalid stage: {self.stage}. Must be one of {valid_stages}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, (VLMConfig, DiTConfig, RLConfig, DataConfig)):
                result[key] = value.__dict__
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        """Create from dictionary"""
        # Extract sub-configs
        vlm_data = data.pop("vlm", {})
        dit_data = data.pop("dit", {})
        rl_data = data.pop("rl", {})
        data.pop("data", {})

        # Create config
        config = cls(**data)
        config.vlm = VLMConfig(**vlm_data)
        config.dit = DiTConfig(**dit_data)
        config.rl = RLConfig(**rl_data)

        return config


@dataclass
class EvalConfig:
    """Evaluation configuration"""
    # Model paths
    vlm_model_path: str = ""
    """Path to VLM model"""

    dit_checkpoint_path: str = ""
    """Path to DiT checkpoint"""

    # Evaluation settings
    batch_size: int = 1
    """Batch size for evaluation"""

    num_samples: int = 1000
    """Number of samples to evaluate"""

    # Output
    output_dir: str = "./eval_results"
    """Output directory"""

    # Metrics
    save_trajectories: bool = True
    """Save predicted trajectories"""

    compute_pdm_score: bool = True
    """Compute PDM score"""

    # VLM/DiT configs
    vlm: VLMConfig = field(default_factory=VLMConfig)
    """VLM configuration"""

    dit: DiTConfig = field(default_factory=DiTConfig)
    """DiT configuration"""
