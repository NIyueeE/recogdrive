"""
Stage 2: DiT Imitation Learning
Training entry point for DiT-based trajectory planning
"""

import argparse
import logging
import os
import sys
from typing import Optional

# Lazy import torch - only loaded when actually needed
# This allows mock mode to work without torch installed
torch = None

from ..config import TrainingConfig, ConfigLoader
from ..vlm import create_vlm

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Stage 2: DiT Imitation Learning"
    )

    # Configuration
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--vlm-path",
        type=str,
        required=True,
        help="Path to fine-tuned VLM",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to trajectory data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/stage2_dit",
        help="Output directory",
    )

    # DiT parameters
    parser.add_argument("--dit-type", type=str, default="small", choices=["small", "large"])
    parser.add_argument("--vlm-size", type=str, default="large", choices=["small", "large"])

    # Training parameters
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)

    # Hardware
    parser.add_argument("--num_gpus", type=int, default=8)

    # Misc
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=500)

    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("=" * 50)
    logger.info("Stage 2: DiT Imitation Learning")
    logger.info("=" * 50)

    # Load configuration if provided
    if args.config:
        config_loader = ConfigLoader()
        config = config_loader.load(args.config, stage="stage2")
    else:
        config = TrainingConfig(stage="stage2")

    # Override config
    config.vlm.vlm_model_path = args.vlm_path
    config.vlm.vlm_size = args.vlm_size
    config.dit.dit_type = args.dit_type
    config.data.data_path = args.data_path
    config.data.output_dir = args.output_dir
    config.num_train_epochs = args.num_epochs
    config.learning_rate = args.learning_rate

    # Print configuration
    logger.info(f"VLM Path: {config.vlm.vlm_model_path}")
    logger.info(f"VLM Size: {config.vlm.vlm_size}")
    logger.info(f"DiT Type: {config.dit.dit_type}")
    logger.info(f"Output Dir: {config.data.output_dir}")
    logger.info(f"Epochs: {config.num_train_epochs}")

    # Check paths - skip for mock mode
    if config.vlm.vlm_model_path != "mock":
        if not os.path.exists(config.vlm.vlm_model_path):
            logger.error(f"VLM model path does not exist: {config.vlm.vlm_model_path}")
            sys.exit(1)

    if not os.path.exists(config.data.data_path):
        logger.error(f"Data path does not exist: {config.data.data_path}")
        sys.exit(1)

    # Create output directory
    os.makedirs(config.data.output_dir, exist_ok=True)

    # Handle mock mode for testing
    if config.vlm.vlm_model_path == "mock":
        logger.info("=" * 50)
        logger.info("Running in MOCK mode - no actual training")
        logger.info("DiT training configuration prepared successfully!")
        logger.info("=" * 50)
        return

    logger.info("=" * 50)
    logger.info("Starting DiT training...")
    logger.info("=" * 50)

    # TODO: Implement DiT training logic
    # - Load VLM backbone
    # - Initialize DiT model
    # - Load trajectory data
    # - Run imitation learning

    logger.info("DiT training configuration prepared successfully!")


if __name__ == "__main__":
    main()
