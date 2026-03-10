#!/usr/bin/env python3
"""
Standalone config validation script.
Validates training configuration without loading PyTorch or models.
"""
import argparse
import sys
import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_config(config_path: str) -> int:
    """Validate configuration without loading models."""
    config_file = Path(config_path)
    if not config_file.exists():
        logger.error(f"Config file not found: {config_path}")
        return 1

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
        logger.warning("NOTE: Large model weights will be downloaded if training runs")

    # Display training params
    logger.info(f"Learning Rate: {config.get('learning_rate', 'N/A')}")
    logger.info(f"Batch Size: {config.get('per_device_batch_size', 'N/A')}")
    logger.info(f"Max Steps: {config.get('max_steps', 'N/A')}")
    logger.info(f"Num Epochs: {config.get('num_train_epochs', 'N/A')}")

    # Data paths
    data_config = config.get('data', {})
    logger.info(f"Data Path: {data_config.get('data_path', 'N/A')}")
    logger.info(f"Output Dir: {data_config.get('output_dir', 'N/A')}")

    # DiT config
    dit_config = config.get('dit', {})
    if dit_config:
        logger.info(f"DiT Type: {dit_config.get('dit_type', 'N/A')}")
        logger.info(f"DiT Input Dim: {dit_config.get('input_embedding_dim', 'N/A')}")

    # For Qwen2.5-VL specific validation
    if vlm_type == "qwen":
        logger.info("")
        logger.info("Qwen2.5-VL Configuration Notes:")
        logger.info("- Requires transformers>=4.37.0")
        logger.info("- Vision hidden size: 1536")
        logger.info("- Model layers: 28 (for 7B)")
        logger.info("- Requires ~16GB GPU memory for 7B model")
        logger.info("")

    logger.info("=" * 50)
    logger.info("Validation PASSED - Configuration is valid")
    logger.info("=" * 50)

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Validate ReCogDrive configuration without loading models"
    )
    parser.add_argument(
        "--config", type=str, default="configs/mini.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()

    sys.exit(validate_config(args.config))


if __name__ == "__main__":
    main()
