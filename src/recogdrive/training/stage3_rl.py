"""
Stage 3: DiffGRPO Reinforcement Learning
Training entry point for GRPO-based trajectory optimization
"""

import argparse
import logging
import os
import sys
from typing import Optional

import torch

from ..config import TrainingConfig, ConfigLoader

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Stage 3: DiffGRPO Reinforcement Learning"
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
        "--dit-path",
        type=str,
        required=True,
        help="Path to trained DiT checkpoint",
    )
    parser.add_argument(
        "--metric-cache",
        type=str,
        required=True,
        help="Path to PDM metric cache",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/stage3_rl",
        help="Output directory",
    )

    # RL parameters
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--kl_coef", type=float, default=0.1)
    parser.add_argument("--gamma_denoising", type=float, default=0.6)

    # Training parameters
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)

    # Hardware
    parser.add_argument("--num_gpus", type=int, default=8)

    # Misc
    parser.add_argument("--logging_steps", type=int, default=1)

    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("=" * 50)
    logger.info("Stage 3: DiffGRPO Reinforcement Learning")
    logger.info("=" * 50)

    # Load configuration if provided
    if args.config:
        config_loader = ConfigLoader()
        config = config_loader.load(args.config, stage="stage3")
    else:
        config = TrainingConfig(stage="stage3")

    # Override config
    config.vlm.vlm_model_path = args.vlm_path
    config.data.output_dir = args.output_dir
    config.rl.grpo = True
    config.rl.metric_cache_path = args.metric_cache
    config.rl.num_samples = args.num_samples
    config.rl.kl_coef = args.kl_coef
    config.rl.gamma_denoising = args.gamma_denoising
    config.num_train_epochs = args.num_epochs
    config.learning_rate = args.learning_rate

    # Print configuration
    logger.info(f"VLM Path: {config.vlm.vlm_model_path}")
    logger.info(f"DiT Path: {args.dit_path}")
    logger.info(f"Metric Cache: {config.rl.metric_cache_path}")
    logger.info(f"Output Dir: {config.data.output_dir}")
    logger.info(f"GRPO Samples: {config.rl.num_samples}")
    logger.info(f"KL Coefficient: {config.rl.kl_coef}")
    logger.info(f"Epochs: {config.num_train_epochs}")

    # Check paths
    if not os.path.exists(args.dit_path):
        logger.error(f"DiT checkpoint does not exist: {args.dit_path}")
        sys.exit(1)

    if not os.path.exists(args.metric_cache):
        logger.error(f"Metric cache does not exist: {args.metric_cache}")
        sys.exit(1)

    # Create output directory
    os.makedirs(config.data.output_dir, exist_ok=True)

    logger.info("=" * 50)
    logger.info("Starting GRPO training...")
    logger.info("=" * 50)

    # TODO: Implement GRPO training logic
    # - Load VLM and DiT
    # - Load reference policy
    # - Initialize PDM scorer
    # - Run GRPO training

    logger.info("GRPO training configuration prepared successfully!")


if __name__ == "__main__":
    main()
