"""
Configuration Loader
Load and merge configurations from YAML, JSON, and CLI
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

import yaml

from .base import TrainingConfig, VLMConfig, DiTConfig, RLConfig, DataConfig, Stage

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Configuration loader with support for:
    - YAML files
    - JSON files
    - CLI arguments
    - Environment variables
    - Configuration merging
    """

    DEFAULT_CONFIG = "default.yaml"

    def __init__(
        self,
        config_dir: Optional[Union[str, Path]] = None,
        default_config: Optional[str] = None,
    ):
        """
        Initialize config loader

        Args:
            config_dir: Directory containing config files
            default_config: Default config file name
        """
        self.config_dir = Path(config_dir) if config_dir else None
        self.default_config = default_config or self.DEFAULT_CONFIG

    @staticmethod
    def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
        """Load YAML file"""
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def load_json(path: Union[str, Path]) -> Dict[str, Any]:
        """Load JSON file"""
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def save_yaml(config: Dict[str, Any], path: Union[str, Path]):
        """Save config to YAML"""
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    @staticmethod
    def save_json(config: Dict[str, Any], path: Union[str, Path], indent: int = 2):
        """Save config to JSON"""
        with open(path, "w") as f:
            json.dump(config, f, indent=indent)

    def load(
        self,
        config_file: Optional[str] = None,
        stage: Optional[str] = None,
        **overrides,
    ) -> TrainingConfig:
        """
        Load configuration from file with overrides

        Args:
            config_file: Config file name (relative to config_dir)
            stage: Training stage (overrides config_file)
            **overrides: CLI overrides

        Returns:
            TrainingConfig
        """
        config_dict = {}

        # Load default config
        if self.config_dir and Path(self.config_dir / self.default_config).exists():
            default_path = self.config_dir / self.default_config
            logger.info(f"Loading default config: {default_path}")
            config_dict = self.load_yaml(default_path)

        # Load stage-specific config
        if stage:
            config_dict["stage"] = stage
            if self.config_dir:
                stage_config_path = self.config_dir / f"stage_{stage}.yaml"
                if stage_config_path.exists():
                    logger.info(f"Loading stage config: {stage_config_path}")
                    stage_config = self.load_yaml(stage_config_path)
                    config_dict = self._merge_configs(config_dict, stage_config)

        # Load user config file
        if config_file:
            config_path = Path(config_file)
            if not config_path.is_absolute() and self.config_dir:
                config_path = self.config_dir / config_path

            if config_path.exists():
                logger.info(f"Loading config: {config_path}")
                if config_path.suffix in [".yaml", ".yml"]:
                    user_config = self.load_yaml(config_path)
                elif config_path.suffix == ".json":
                    user_config = self.load_json(config_path)
                else:
                    raise ValueError(f"Unsupported config format: {config_path.suffix}")

                config_dict = self._merge_configs(config_dict, user_config)

        # Apply CLI overrides
        if overrides:
            logger.info(f"Applying CLI overrides: {overrides}")
            config_dict = self._merge_configs(config_dict, overrides)

        # Convert to TrainingConfig
        return self._dict_to_config(config_dict)

    def _dict_to_config(self, config_dict: Dict[str, Any]) -> TrainingConfig:
        """Convert dictionary to TrainingConfig"""
        # Extract sub-configs
        vlm_dict = config_dict.pop("vlm", {})
        dit_dict = config_dict.pop("dit", {})
        rl_dict = config_dict.pop("rl", {})
        data_dict = config_dict.pop("data", {})

        # Create TrainingConfig
        config = TrainingConfig(**config_dict)

        # Create sub-configs
        config.vlm = VLMConfig(**vlm_dict)
        config.dit = DiTConfig(**dit_dict)
        config.rl = RLConfig(**rl_dict)
        config.data = DataConfig(**data_dict)

        return config

    def _merge_configs(
        self,
        base: Dict[str, Any],
        override: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Deep merge two configs

        Args:
            base: Base config
            override: Override config

        Returns:
            Merged config
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    @staticmethod
    def from_cli(args: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Parse CLI arguments

        Args:
            args: Command line arguments (defaults to sys.argv)

        Returns:
            Dictionary of arguments
        """
        if args is None:
            args = sys.argv[1:]

        result = {}
        i = 0
        while i < len(args):
            arg = args[i]

            if arg.startswith("--"):
                key = arg[2:].replace("-", "_")

                # Check for key=value format
                if "=" in arg:
                    value = arg.split("=", 1)[1]
                    result[key] = ConfigLoader._parse_value(value)
                elif i + 1 < len(args) and not args[i + 1].startswith("-"):
                    value = args[i + 1]
                    result[key] = ConfigLoader._parse_value(value)
                    i += 1
                else:
                    result[key] = True

            i += 1

        return result

    @staticmethod
    def _parse_value(value: str) -> Any:
        """Parse string value to appropriate type"""
        # Try boolean
        if value.lower() in ["true", "yes", "on"]:
            return True
        if value.lower() in ["false", "no", "off"]:
            return False

        # Try integer
        try:
            return int(value)
        except ValueError:
            pass

        # Try float
        try:
            return float(value)
        except ValueError:
            pass

        # Return as string
        return value

    def save(
        self,
        config: TrainingConfig,
        path: Union[str, Path],
        format: str = "yaml",
    ):
        """
        Save configuration to file

        Args:
            config: TrainingConfig to save
            path: Output path
            format: Output format (yaml, json)
        """
        config_dict = config.to_dict()

        if format == "yaml":
            self.save_yaml(config_dict, path)
        elif format == "json":
            self.save_json(config_dict, path)
        else:
            raise ValueError(f"Unsupported format: {format}")


def create_config(
    config_file: Optional[str] = None,
    config_dir: Optional[str] = None,
    stage: Optional[str] = None,
    **overrides,
) -> TrainingConfig:
    """
    Convenience function to create a TrainingConfig

    Args:
        config_file: Config file path
        config_dir: Config directory
        stage: Training stage
        **overrides: CLI overrides

    Returns:
        TrainingConfig
    """
    loader = ConfigLoader(config_dir)
    return loader.load(config_file, stage, **overrides)


# CLI entry point
def main():
    """CLI for loading configs"""
    import argparse

    parser = argparse.ArgumentParser(description="ReCogDrive Config Loader")
    parser.add_argument("--config", type=str, help="Config file path")
    parser.add_argument("--config-dir", type=str, help="Config directory")
    parser.add_argument("--stage", type=str, help="Training stage")
    parser.add_argument("--output", type=str, help="Output path")
    parser.add_argument("--format", type=str, default="yaml", choices=["yaml", "json"])
    parser.add_argument("overrides", nargs="*", help="CLI overrides (key=value)")

    args = parser.parse_args()

    # Parse overrides
    overrides = {}
    for override in args.overrides:
        if "=" in override:
            key, value = override.split("=", 1)
            overrides[key] = ConfigLoader._parse_value(value)

    # Load config
    loader = ConfigLoader(args.config_dir)
    config = loader.load(args.config, args.stage, **overrides)

    # Save if output specified
    if args.output:
        loader.save(config, args.output, args.format)
        print(f"Config saved to {args.output}")
    else:
        # Print config
        print(yaml.dump(config.to_dict(), default_flow_style=False))


if __name__ == "__main__":
    main()
