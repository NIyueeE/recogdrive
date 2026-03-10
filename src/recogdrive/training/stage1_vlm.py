"""
Stage 1: VLM Supervised Fine-tuning
Training entry point for VLM supervised fine-tuning
"""

import argparse
import logging
import os
import sys
import random
import time
import json
from pathlib import Path
from typing import Optional

# Conditional imports - only load heavy dependencies when needed
# For mock mode, we use Python's standard library only
torch = None
dist = None

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
    global torch, dist
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Lazy import torch and dist when actually needed for distributed training
        import torch
        import torch.distributed as dist

        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

        logger.info(f"Distributed training: rank={rank}, world_size={world_size}")
        return True
    return False


def run_mock_training(args, config):
    """Run mock training with dummy model and data for pipeline validation.

    This function simulates a training loop without requiring PyTorch or other
    heavy dependencies. It uses Python's standard library for reproducibility.
    """
    logger.info("Initializing mock training...")

    # Create a simple dummy model (just parameters, no actual weights)
    hidden_dim = config.vlm.get_hidden_dim()
    logger.info(f"Creating mock model with hidden_dim={hidden_dim}")

    # Simulate model parameters using Python's random (no torch needed)
    random.seed(42)
    dummy_model_params = {
        "vision_encoder": [[random.random() for _ in range(hidden_dim)] for _ in range(100)],
        "mlp_projector": [[random.random() for _ in range(hidden_dim)] for _ in range(hidden_dim)],
        "language_model": [[random.random() for _ in range(hidden_dim)] for _ in range(1000)],
    }

    # Create dummy dataset (5 samples)
    num_samples = 5
    logger.info(f"Creating mock dataset with {num_samples} samples")

    # Simulate training loop
    max_steps = 10
    batch_size = args.batch_size if hasattr(args, 'batch_size') else 1

    logger.info(f"Starting mock training for {max_steps} steps, batch_size={batch_size}")

    training_history = []
    for step in range(max_steps):
        step_start = time.time()

        # Simulate forward pass - compute random loss
        # Use deterministic seed for reproducibility
        random.seed(42 + step)
        loss = random.random() * 0.5 + 0.5  # Loss between 0.5 and 1.0

        # Simulate optimizer step
        learning_rate = config.learning_rate

        step_time = time.time() - step_start

        # Log progress
        logger.info(
            f"Step {step + 1}/{max_steps} | "
            f"Loss: {loss:.4f} | "
            f"LR: {learning_rate:.2e} | "
            f"Time: {step_time:.3f}s"
        )

        training_history.append({
            "step": step + 1,
            "loss": loss,
            "learning_rate": learning_rate,
            "step_time": step_time
        })

        # Simulate saving checkpoint every 5 steps
        if (step + 1) % 5 == 0:
            checkpoint_path = Path(config.data.output_dir) / f"checkpoint-step-{step + 1}"
            checkpoint_path.mkdir(parents=True, exist_ok=True)

            # Save dummy checkpoint
            checkpoint_file = checkpoint_path / "trainer_state.json"
            with open(checkpoint_file, "w") as f:
                json.dump({
                    "global_step": step + 1,
                    "epoch": (step + 1) / max_steps,
                    "loss": loss,
                }, f, indent=2)

            logger.info(f"Saved checkpoint to {checkpoint_path}")

    # Save final training summary
    summary_path = Path(config.data.output_dir) / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "total_steps": max_steps,
            "final_loss": training_history[-1]["loss"] if training_history else None,
            "avg_loss": sum(h["loss"] for h in training_history) / len(training_history) if training_history else None,
            "total_time": sum(h["step_time"] for h in training_history),
            "config": {
                "batch_size": batch_size,
                "learning_rate": config.learning_rate,
                "hidden_dim": hidden_dim,
            }
        }, f, indent=2)

    logger.info("=" * 50)
    logger.info("Mock training completed successfully!")
    logger.info(f"Final loss: {training_history[-1]['loss']:.4f}")
    logger.info(f"Average loss: {sum(h['loss'] for h in training_history) / len(training_history):.4f}")
    logger.info(f"Total time: {sum(h['step_time'] for h in training_history):.2f}s")
    logger.info(f"Summary saved to: {summary_path}")
    logger.info("=" * 50)


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

    logger.info("=" * 50)
    logger.info("Starting training...")
    logger.info("=" * 50)

    # Check if running in mock mode
    if config.vlm.vlm_model_path == "mock":
        logger.info("Running in MOCK mode - using dummy model and data")
        run_mock_training(args, config)
    else:
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
