"""
Stage 1: VLM Supervised Fine-tuning
Training entry point for VLM supervised fine-tuning
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from transformers import (
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollator,
)

from ..config import TrainingConfig, ConfigLoader
from ..vlm import create_vlm

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Stage 1: VLM Supervised Fine-tuning"
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file",
    )
    parser.add_argument(
        "--vlm-type",
        type=str,
        default="internvl",
        choices=["internvl", "qwen"],
        help="VLM type",
    )
    parser.add_argument(
        "--vlm-path",
        type=str,
        required=True,
        help="Path to VLM model",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to training data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/stage1_vlm",
        help="Output directory",
    )

    # Training parameters
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=4e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    # Model parameters
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--max_seq_length", type=int, default=12288)

    # Hardware
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--use_bf16", action="store_true")

    # Distributed training
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--deepspeed", type=str, default=None)

    # Misc
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def setup_distributed():
    """Initialize distributed training if available"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

        logger.info(f"Distributed training: rank={rank}, world_size={world_size}")
        return True
    return False


def main():
    """Main training function"""
    args = parse_args()
    setup_logging()

    logger.info("=" * 50)
    logger.info("Stage 1: VLM Supervised Fine-tuning")
    logger.info("=" * 50)

    # Setup distributed training
    is_distributed = setup_distributed()

    # Load configuration if provided
    if args.config:
        config_loader = ConfigLoader()
        config = config_loader.load(args.config, stage="stage1")
        logger.info(f"Loaded config from {args.config}")
    else:
        config = TrainingConfig(stage="stage1")

    # Override config with CLI args
    config.vlm.vlm_type = args.vlm_type
    config.vlm.vlm_model_path = args.vlm_path
    config.data.data_path = args.data_path
    config.data.output_dir = args.output_dir
    config.num_train_epochs = args.num_epochs
    config.learning_rate = args.learning_rate
    config.vlm.force_image_size = args.image_size
    config.data.max_seq_length = args.max_seq_length
    config.bf16 = args.use_bf16
    config.fp16 = args.use_fp16

    # Print configuration
    logger.info(f"VLM Type: {config.vlm.vlm_type}")
    logger.info(f"VLM Path: {config.vlm.vlm_model_path}")
    logger.info(f"Data Path: {config.data.data_path}")
    logger.info(f"Output Dir: {config.data.output_dir}")
    logger.info(f"Learning Rate: {config.learning_rate}")
    logger.info(f"Epochs: {config.num_train_epochs}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Precision: {'bf16' if config.bf16 else 'fp16' if config.fp16 else 'fp32'}")

    # Check paths
    if not os.path.exists(config.vlm.vlm_model_path):
        logger.error(f"VLM model path does not exist: {config.vlm.vlm_model_path}")
        sys.exit(1)

    if not os.path.exists(config.data.data_path):
        logger.error(f"Data path does not exist: {config.data.data_path}")
        sys.exit(1)

    # Create output directory
    os.makedirs(config.data.output_dir, exist_ok=True)

    logger.info("=" * 50)
    logger.info("Starting training...")
    logger.info("=" * 50)

    # TODO: Implement actual training logic
    # - Load tokenizer
    # - Load dataset
    # - Initialize model
    # - Setup Trainer
    # - Run training

    logger.info("Training configuration prepared successfully!")
    logger.info(f"Use this command to run training with DeepSpeed:")
    logger.info(f"  deepspeed --num_gpus={args.num_gpus} src/recogdrive/training/stage1_vlm.py \\")
    logger.info(f"    --vlm-path {args.vlm_path} \\")
    logger.info(f"    --data-path {args.data_path} \\")
    logger.info(f"    --output-dir {args.output_dir}")


if __name__ == "__main__":
    main()
