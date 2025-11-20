"""Configuration management module for loading and accessing configuration files."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Config:
    """Configuration loader and manager."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to the main configuration file.
                        If None, uses default path.
        """
        if config_path is None:
            # Get project root directory
            self.project_root = Path(__file__).parent.parent.parent
            config_path = self.project_root / "configs" / "config.yaml"
        else:
            config_path = Path(config_path)
            self.project_root = config_path.parent.parent

        self.config_path = config_path
        self._config = self._load_config(config_path)
        self._model_config = None

    def _load_config(self, path: Path) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Dictionary containing configuration.

        Raises:
            FileNotFoundError: If configuration file not found.
            yaml.YAMLError: If YAML parsing fails.
        """
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r") as f:
            config = yaml.safe_load(f)

        return config

    @property
    def model_config(self) -> Dict[str, Any]:
        """
        Load model configuration lazily.

        Returns:
            Dictionary containing model configuration.
        """
        if self._model_config is None:
            model_config_path = self.project_root / "configs" / "model_config.yaml"
            self._model_config = self._load_config(model_config_path)
        return self._model_config

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key with dot notation.

        Args:
            key: Configuration key (supports dot notation, e.g., 'paths.data.raw')
            default: Default value if key not found

        Returns:
            Configuration value or default

        Examples:
            >>> config = Config()
            >>> config.get('project.name')
            'heart-disease-prediction'
            >>> config.get('paths.data.raw')
            'data/raw'
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def get_path(self, key: str) -> Path:
        """
        Get path from configuration and convert to absolute Path object.

        Args:
            key: Configuration key for path

        Returns:
            Absolute Path object

        Examples:
            >>> config = Config()
            >>> config.get_path('paths.data.raw')
            PosixPath('/home/user/project/data/raw')
        """
        path_str = self.get(key)
        if path_str is None:
            raise ValueError(f"Path not found in configuration: {key}")

        path = Path(path_str)
        if not path.is_absolute():
            path = self.project_root / path

        return path

    def get_model_params(self, model_name: str) -> Dict[str, Any]:
        """
        Get model parameters from model configuration.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary containing model parameters

        Raises:
            ValueError: If model not found in configuration
        """
        models = self.model_config.get("models", {})
        if model_name not in models:
            raise ValueError(f"Model '{model_name}' not found in configuration")

        return models[model_name].get("params", {})

    def get_hyperparameter_search_space(self, model_name: str) -> Dict[str, Any]:
        """
        Get hyperparameter search space for a model.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary containing hyperparameter search space

        Raises:
            ValueError: If model not found in configuration
        """
        models = self.model_config.get("models", {})
        if model_name not in models:
            raise ValueError(f"Model '{model_name}' not found in configuration")

        return models[model_name].get("hyperparameter_search", {})

    def is_model_enabled(self, model_name: str) -> bool:
        """
        Check if a model is enabled in configuration.

        Args:
            model_name: Name of the model

        Returns:
            True if model is enabled, False otherwise
        """
        models = self.model_config.get("models", {})
        if model_name not in models:
            return False

        return models[model_name].get("enabled", False)

    def get_enabled_models(self) -> list:
        """
        Get list of enabled models.

        Returns:
            List of enabled model names
        """
        models = self.model_config.get("models", {})
        return [
            name
            for name, config in models.items()
            if config.get("enabled", False)
        ]

    def update(self, key: str, value: Any) -> None:
        """
        Update configuration value.

        Args:
            key: Configuration key (supports dot notation)
            value: New value

        Examples:
            >>> config = Config()
            >>> config.update('training.cv_folds', 10)
        """
        keys = key.split(".")
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def save(self, path: Optional[Path] = None) -> None:
        """
        Save configuration to file.

        Args:
            path: Path to save configuration. If None, uses original path.
        """
        if path is None:
            path = self.config_path

        with open(path, "w") as f:
            yaml.dump(self._config, f, default_flow_style=False)

    def __repr__(self) -> str:
        """String representation of Config object."""
        return f"Config(config_path='{self.config_path}')"

    def __str__(self) -> str:
        """Human-readable string representation."""
        return yaml.dump(self._config, default_flow_style=False)


# Global configuration instance
_config_instance: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get or create global configuration instance (singleton pattern).

    Args:
        config_path: Path to configuration file (only used on first call)

    Returns:
        Global Config instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance
