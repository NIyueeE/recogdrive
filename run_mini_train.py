#!/usr/bin/env python3
"""
Standalone mini training script for pipeline validation.
This runs mock training without requiring PyTorch or other heavy dependencies.
"""
import os
import sys
import time
import json
import random
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_mock_training(config):
    """Run mock training with dummy model and data for pipeline validation.

    This function simulates a training loop without requiring PyTorch or other
    heavy dependencies. It uses Python's standard library for reproducibility.
    """
    logger.info("Initializing mock training...")

    # Create a simple dummy model (just parameters, no actual weights)
    hidden_dim = config.get("hidden_dim", 3584)
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
    max_steps = config.get("max_steps", 10)
    batch_size = config.get("batch_size", 1)
    learning_rate = config.get("learning_rate", 1e-4)

    logger.info(f"Starting mock training for {max_steps} steps, batch_size={batch_size}, lr={learning_rate}")

    training_history = []
    for step in range(max_steps):
        step_start = time.time()

        # Simulate forward pass - compute random loss
        # Use deterministic seed for reproducibility
        random.seed(42 + step)
        loss = random.random() * 0.5 + 0.5  # Loss between 0.5 and 1.0

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
            checkpoint_path = Path(config["output_dir"]) / f"checkpoint-step-{step + 1}"
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
    summary_path = Path(config["output_dir"]) / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "total_steps": max_steps,
            "final_loss": training_history[-1]["loss"] if training_history else None,
            "avg_loss": sum(h["loss"] for h in training_history) / len(training_history) if training_history else None,
            "total_time": sum(h["step_time"] for h in training_history),
            "config": config
        }, f, indent=2)

    logger.info("=" * 50)
    logger.info("Mock training completed successfully!")
    logger.info(f"Final loss: {training_history[-1]['loss']:.4f}")
    logger.info(f"Average loss: {sum(h['loss'] for h in training_history) / len(training_history):.4f}")
    logger.info(f"Total time: {sum(h['step_time'] for h in training_history):.2f}s")
    logger.info(f"Summary saved to: {summary_path}")
    logger.info("=" * 50)

    return training_history


def main():
    """Main function"""
    logger.info("=" * 50)
    logger.info("ReCogDrive Mini Training - Pipeline Validation")
    logger.info("=" * 50)

    # Configuration for mini training
    config = {
        "hidden_dim": 3584,  # InternVL3 hidden dimension
        "batch_size": 1,
        "learning_rate": 1e-4,
        "max_steps": 10,
        "output_dir": "./outputs/mini",
        "data_path": "./data/mini",
    }

    # Create necessary directories
    os.makedirs(config["data_path"], exist_ok=True)
    os.makedirs(config["output_dir"], exist_ok=True)

    # Print configuration
    logger.info(f"Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    # Run training
    logger.info("=" * 50)
    logger.info("Starting training...")
    logger.info("=" * 50)

    history = run_mock_training(config)

    logger.info("Training completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
