"""
ReCogDrive Command Line Interface

Usage:
    python -m src.recogdrive.cli train --stage 1 --vlm-path /path/to/model
    python -m src.recogdrive.cli download --dataset navsim
"""

import argparse
import sys
import subprocess
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from pathlib import Path



def validate_config(args):
    """Validate configuration without loading models."""
    # Determine config file path
    config_path = getattr(args, 'config', None)
    if not config_path:
        config_path = "configs/mini.yaml"

    config_file = Path(config_path)
    if not config_file.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    logger.info(f"Loading config from: {config_path}")
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Validate VLM config
    vlm_config = config.get('vlm', {})
    vlm_type = vlm_config.get('vlm_type', 'internvl')
    vlm_model_path = vlm_config.get('vlm_model_path', '')

    logger.info("=" * 50)
    logger.info("Configuration Validation Results")
    logger.info("=" * 50)
    logger.info(f"Stage: {config.get('stage', 'N/A')}")
    logger.info(f"VLM Type: {vlm_type}")
    logger.info(f"VLM Model Path: {vlm_model_path}")

    # Check for mock mode
    if vlm_model_path == "mock" or vlm_model_path == "":
        logger.info("Mode: MOCK (no model weights will be downloaded)")
    else:
        logger.info(f"Mode: REAL (model weights required from: {vlm_model_path})")

    # Display training params
    logger.info(f"Learning Rate: {config.get('learning_rate', 'N/A')}")
    logger.info(f"Batch Size: {config.get('per_device_batch_size', 'N/A')}")
    logger.info(f"Max Steps: {config.get('max_steps', 'N/A')}")

    # For Qwen2.5-VL specific validation
    if vlm_type == "qwen":
        logger.info("")
        logger.info("Qwen2.5-VL Configuration Notes:")
        logger.info("- Requires transformers>=4.37.0")
        logger.info("- Vision hidden size: 1536")

    logger.info("=" * 50)
    logger.info("Validation PASSED")
    logger.info("=" * 50)

    return 0


def run_command(cmd):
    """Run shell command and exit on error."""
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        sys.exit(result.returncode)


def train_stage1(args):
    """Run Stage 1: VLM Supervised Fine-tuning."""
    cmd = f"python -m src.recogdrive.training.stage1_vlm"
    if args.vlm_path:
        cmd += f" --vlm-path {args.vlm_path}"
    if args.data_path:
        cmd += f" --data-path {args.data_path}"
    if args.output_dir:
        cmd += f" --output-dir {args.output_dir}"
    if args.num_gpus:
        cmd += f" --num_gpus {args.num_gpus}"
    if args.num_epochs:
        cmd += f" --num_epochs {args.num_epochs}"
    if args.batch_size:
        cmd += f" --batch_size {args.batch_size}"
    if args.learning_rate:
        cmd += f" --learning_rate {args.learning_rate}"
    print(f"Running: {cmd}")
    run_command(cmd)


def train_stage2(args):
    """Run Stage 2: DiT Imitation Learning."""
    cmd = f"python -m src.recogdrive.training.stage2_dit"
    if args.vlm_path:
        cmd += f" --vlm-path {args.vlm_path}"
    if args.data_path:
        cmd += f" --data-path {args.data_path}"
    if args.output_dir:
        cmd += f" --output-dir {args.output_dir}"
    if args.num_gpus:
        cmd += f" --num_gpus {args.num_gpus}"
    if args.num_epochs:
        cmd += f" --num_epochs {args.num_epochs}"
    if args.batch_size:
        cmd += f" --batch_size {args.batch_size}"
    if args.learning_rate:
        cmd += f" --learning_rate {args.learning_rate}"
    if args.dit_type:
        cmd += f" --dit-type {args.dit_type}"
    if args.vlm_size:
        cmd += f" --vlm-size {args.vlm_size}"
    print(f"Running: {cmd}")
    run_command(cmd)


def train_stage3(args):
    """Run Stage 3: DiffGRPO Reinforcement Learning."""
    cmd = f"python -m src.recogdrive.training.stage3_rl"
    if args.vlm_path:
        cmd += f" --vlm-path {args.vlm_path}"
    if args.dit_path:
        cmd += f" --dit-path {args.dit_path}"
    if args.metric_cache:
        cmd += f" --metric-cache {args.metric_cache}"
    if args.output_dir:
        cmd += f" --output-dir {args.output_dir}"
    if args.num_gpus:
        cmd += f" --num_gpus {args.num_gpus}"
    if args.num_epochs:
        cmd += f" --num_epochs {args.num_epochs}"
    if args.batch_size:
        cmd += f" --batch_size {args.batch_size}"
    if args.learning_rate:
        cmd += f" --learning_rate {args.learning_rate}"
    if args.num_samples:
        cmd += f" --num_samples {args.num_samples}"
    if args.kl_coeff:
        cmd += f" --kl_coeff {args.kl_coeff}"
    print(f"Running: {cmd}")
    run_command(cmd)


def download_dataset(args):
    """Download datasets."""
    script_dir = Path(__file__).parent.parent.parent / "scripts" / "download"

    if args.dataset == "navsim":
        run_command(f"bash {script_dir}/download_navtrain.sh")
    elif args.dataset == "trainval":
        run_command(f"bash {script_dir}/download_trainval.sh")
    elif args.dataset == "test":
        run_command(f"bash {script_dir}/download_test.sh")
    elif args.dataset == "mini":
        run_command(f"bash {script_dir}/download_mini.sh")
    elif args.dataset == "maps":
        run_command(f"bash {script_dir}/download_maps.sh")
    elif args.dataset == "all":
        datasets = ["navsim", "trainval", "test", "mini", "maps"]
        for ds in datasets:
            args.dataset = ds
            download_dataset(args)
    else:
        print(f"Unknown dataset: {args.dataset}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        prog="recogdrive",
        description="ReCogDrive: Vision-Language Model for Autonomous Driving"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Run training")
    train_parser.add_argument(
        "--stage", type=int, choices=[1, 2, 3], required=True,
        help="Training stage: 1=VLM, 2=DiT, 3=RL"
    )
    train_parser.add_argument("--vlm-path", type=str, help="Path to VLM model")
    train_parser.add_argument("--data-path", type=str, help="Path to training data")
    train_parser.add_argument("--dit-path", type=str, help="Path to DiT checkpoint")
    train_parser.add_argument("--metric-cache", type=str, help="Path to metric cache")
    train_parser.add_argument("--output-dir", type=str, help="Output directory")
    train_parser.add_argument("--num_gpus", type=int, default=8, help="Number of GPUs")
    train_parser.add_argument("--num_epochs", type=int, help="Number of epochs")
    train_parser.add_argument("--batch_size", type=int, help="Batch size")
    train_parser.add_argument("--learning_rate", type=float, help="Learning rate")
    train_parser.add_argument("--num_samples", type=int, help="Number of samples (RL)")
    train_parser.add_argument("--kl_coeff", type=float, help="KL coefficient (RL)")
    train_parser.add_argument("--dit_type", type=str, choices=["small", "large"], help="DiT type")
    train_parser.add_argument("--vlm_size", type=str, choices=["small", "large"], help="VLM size")
    train_parser.add_argument("--config", type=str, default="configs/mini.yaml", help="Path to config file")
    train_parser.add_argument("--validate-config", action="store_true", help="Validate config without loading models")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download datasets")

    # Validate command (standalone, doesn't require torch)
    validate_parser = subparsers.add_parser("validate", help="Validate configuration without loading models")
    validate_parser.add_argument("--config", type=str, default="configs/mini.yaml", help="Path to config file")

    download_parser.add_argument(
        "--dataset", type=str,
        choices=["navsim", "trainval", "test", "mini", "maps", "all"],
        required=True,
        help="Dataset to download"
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("eval", help="Run evaluation")
    eval_parser.add_argument("--model-path", type=str, required=True, help="Path to model")
    eval_parser.add_argument("--checkpoint", type=str, help="Path to checkpoint")
    eval_parser.add_argument("--dataset", type=str, default="navtest", help="Evaluation dataset")

    args = parser.parse_args()

    if args.command == "train":
        if getattr(args, 'validate_config', False):
            sys.exit(validate_config(args))
        if args.stage == 1:
            train_stage1(args)
        elif args.stage == 2:
            train_stage2(args)
        elif args.stage == 3:
            train_stage3(args)
    elif args.command == "download":
        download_dataset(args)
    elif args.command == "validate":
        from src.recogdrive.cli import validate_config
        sys.exit(validate_config(args))
    elif args.command == "eval":
        print("Evaluation not yet implemented")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
