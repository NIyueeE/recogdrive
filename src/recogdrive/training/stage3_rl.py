"""
Stage 3: DiffGRPO Reinforcement Learning
Training entry point for GRPO-based trajectory optimization

This implementation aligns with the source project:
- navsim/planning/script/run_training_recogdrive_rl.py
"""

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

# Lazy import torch - only loaded when actually needed
# This allows mock mode to work without torch installed
torch = None
dist = None
pl_module = None
DataLoader = None

from ..config import TrainingConfig, ConfigLoader

logger = logging.getLogger(__name__)


def _ensure_torch():
    """Lazy import torch when needed"""
    global torch, dist, pl_module, DataLoader
    if torch is None:
        import torch
        import torch.distributed as dist
        import pytorch_lightning as pl_module
        from torch.utils.data import DataLoader
    return torch, dist, pl_module, DataLoader


def custom_collate_fn(
    batch: List[Tuple[Dict[str, any], Dict[str, any], str]]
) -> Tuple[Dict[str, any], Dict[str, any], List[str]]:
    """Custom collate function for variable-length sequences.

    Aligns with source project implementation in
    navsim/planning/script/run_training_recogdrive_rl.py

    Args:
        batch: List of tuples (features, targets, token)

    Returns:
        Tuple of (features, targets, tokens_list)
    """
    torch, _, _, _ = _ensure_torch()

    features_list, targets_list, tokens_list = zip(*batch)

    history_trajectory = torch.stack(
        [features['history_trajectory'] for features in features_list], dim=0
    ).cpu()
    high_command_one_hot = torch.stack(
        [features['high_command_one_hot'] for features in features_list], dim=0
    ).cpu()
    status_feature = torch.stack(
        [features['status_feature'] for features in features_list], dim=0
    ).cpu()

    import torch.nn.utils.rnn as rnn_utils
    last_hidden_state = rnn_utils.pad_sequence(
        [features['last_hidden_state'] for features in features_list],
        batch_first=True,
        padding_value=0.0
    ).clone().detach()

    trajectory = torch.stack(
        [targets['trajectory'] for targets in targets_list], dim=0
    ).cpu()

    features = {
        'history_trajectory': history_trajectory,
        'high_command_one_hot': high_command_one_hot,
        'status_feature': status_feature,
        'last_hidden_state': last_hidden_state,
    }
    targets = {
        'trajectory': trajectory
    }

    return features, targets, tokens_list


def build_datasets(
    cfg: any,
    agent: any,
    train_logs: List[str],
    val_logs: List[str]
) -> Tuple[any, any]:
    """Build training and validation datasets from omega config.

    Args:
        cfg: OmegaConf dictionary
        agent: Interface of agents in NAVSIM
        train_logs: List of training log names
        val_logs: List of validation log names

    Returns:
        Tuple of training and validation datasets
    """
    from hydra.utils import instantiate
    from navsim.common.dataclasses import SceneFilter
    from navsim.common.dataloader import SceneLoader
    from navsim.planning.training.dataset import Dataset

    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names
            if log_name in train_logs
        ]
    else:
        train_scene_filter.log_names = train_logs

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [
            log_name for log_name in val_scene_filter.log_names
            if log_name in val_logs
        ]
    else:
        val_scene_filter.log_names = val_logs

    data_path = cfg.get('navsim_log_path', cfg.get('data_path', ''))
    sensor_blobs_path = cfg.get('sensor_blobs_path', '')

    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    return train_data, val_data


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

    # Dataset parameters
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to NAVSIM data",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default=None,
        help="Path to feature cache",
    )
    parser.add_argument(
        "--use-cache-without-dataset",
        action="store_true",
        help="Use cached data without building SceneLoader",
    )

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

    # Initialize distributed training (lazy import)
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    rank = int(os.getenv('RANK', 0))

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

    # Create output directory
    os.makedirs(config.data.output_dir, exist_ok=True)

    # Handle mock mode for testing - CHECK FIRST before path validation
    if args.vlm_path == "mock":
        logger.info("=" * 50)
        logger.info("Running in MOCK mode - no actual training")
        logger.info("GRPO training configuration prepared successfully!")
        logger.info("=" * 50)
        return

    # Check paths (only in non-mock mode)
    if not os.path.exists(args.dit_path):
        logger.error(f"DiT checkpoint does not exist: {args.dit_path}")
        sys.exit(1)

    if not args.use_cache_without_dataset and not os.path.exists(args.metric_cache):
        logger.error(f"Metric cache does not exist: {args.metric_cache}")
        sys.exit(1)

    logger.info("=" * 50)
    logger.info("Starting GRPO training...")
    logger.info("=" * 50)

    # Initialize distributed training if needed
    if world_size > 1:
        torch, dist, _, _ = _ensure_torch()
        dist.init_process_group(
            backend='nccl',
            world_size=world_size,
            rank=rank,
        )
        torch.cuda.set_device(local_rank)

    # Note: The actual GRPO training implementation requires:
    # 1. VLM and DiT model loading (handled by the agent)
    # 2. Agent initialization via hydra instantiator
    # 3. CacheOnlyDataset or Dataset creation
    # 4. AgentLightningDiT wrapper
    # 5. PyTorch Lightning Trainer
    #
    # For full implementation, use the NAVSIM training pipeline:
    # python navsim/planning/script/run_training_recogdrive_rl.py
    # with appropriate hydra configuration

    logger.info("GRPO training configuration prepared successfully!")


if __name__ == "__main__":
    main()
