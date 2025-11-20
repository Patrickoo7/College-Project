"""Configuration loader with support for environment-specific overrides."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from .schemas import AppConfig, Environment
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Loads and merges configuration files with environment-specific overrides."""

    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize the config loader.

        Args:
            config_dir: Directory containing configuration files.
                       Defaults to project_root/configs
        """
        if config_dir is None:
            # Get project root (3 levels up from this file)
            project_root = Path(__file__).parent.parent.parent
            config_dir = project_root / "configs"

        self.config_dir = Path(config_dir)

        if not self.config_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {self.config_dir}")

    def load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load a YAML configuration file.

        Args:
            filename: Name of the YAML file to load

        Returns:
            Dictionary containing the configuration

        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        filepath = self.config_dir / filename

        if not filepath.exists():
            logger.warning(f"Config file not found: {filepath}")
            return {}

        try:
            with open(filepath, 'r') as f:
                config = yaml.safe_load(f) or {}
            logger.info(f"Loaded config from {filepath}")
            return config
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {filepath}: {e}")
            raise

    def merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two configuration dictionaries.

        Args:
            base: Base configuration dictionary
            override: Override configuration dictionary

        Returns:
            Merged configuration dictionary
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def load(
        self,
        environment: Optional[str] = None,
        config_file: str = "app_config.yaml"
    ) -> AppConfig:
        """Load configuration with environment-specific overrides.

        The loading order is:
        1. Base configuration (app_config.yaml)
        2. Environment-specific overrides (dev_config.yaml, staging_config.yaml, prod_config.yaml)
        3. Environment variables (override any config value)

        Args:
            environment: Environment name (development, staging, production).
                        If None, reads from ENV environment variable, defaults to development.
            config_file: Name of the base configuration file

        Returns:
            Validated AppConfig instance

        Examples:
            >>> loader = ConfigLoader()
            >>> config = loader.load(environment="production")
            >>> print(config.api.port)
            8000
        """
        # Determine environment
        if environment is None:
            environment = os.getenv("ENV", "development")

        try:
            env = Environment(environment.lower())
        except ValueError:
            logger.warning(f"Invalid environment '{environment}', using development")
            env = Environment.DEVELOPMENT

        logger.info(f"Loading configuration for environment: {env.value}")

        # Load base configuration
        config_dict = self.load_yaml(config_file)

        # Load environment-specific overrides
        env_config_file = f"{env.value}_config.yaml"
        env_config = self.load_yaml(env_config_file)

        if env_config:
            logger.info(f"Merging environment config from {env_config_file}")
            config_dict = self.merge_configs(config_dict, env_config)

        # Apply environment variable overrides
        config_dict = self._apply_env_overrides(config_dict)

        # Set environment in config
        config_dict["environment"] = env.value

        # Validate and create Pydantic model
        try:
            app_config = AppConfig(**config_dict)
            logger.info("Configuration loaded and validated successfully")
            return app_config
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise

    def _apply_env_overrides(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration.

        Environment variables should be prefixed with APP_ and use double underscores
        to denote nested keys. For example:
        - APP_API__PORT=9000 -> config.api.port = 9000
        - APP_DEBUG=true -> config.debug = true

        Args:
            config_dict: Base configuration dictionary

        Returns:
            Configuration dictionary with environment overrides applied
        """
        prefix = "APP_"

        for env_key, env_value in os.environ.items():
            if not env_key.startswith(prefix):
                continue

            # Remove prefix and split by double underscore
            key_path = env_key[len(prefix):].lower().split("__")

            # Navigate to the nested key and set value
            current = config_dict
            for i, key in enumerate(key_path[:-1]):
                if key not in current:
                    current[key] = {}
                current = current[key]

            # Convert string values to appropriate types
            final_key = key_path[-1]
            converted_value = self._convert_env_value(env_value)

            current[final_key] = converted_value
            logger.debug(f"Applied env override: {env_key} = {converted_value}")

        return config_dict

    @staticmethod
    def _convert_env_value(value: str) -> Any:
        """Convert environment variable string to appropriate Python type.

        Args:
            value: String value from environment variable

        Returns:
            Converted value (bool, int, float, or str)
        """
        # Boolean conversion
        if value.lower() in ("true", "yes", "1", "on"):
            return True
        if value.lower() in ("false", "no", "0", "off"):
            return False

        # Integer conversion
        try:
            return int(value)
        except ValueError:
            pass

        # Float conversion
        try:
            return float(value)
        except ValueError:
            pass

        # Return as string
        return value

    def save(self, config: AppConfig, filename: str = "generated_config.yaml") -> Path:
        """Save configuration to a YAML file.

        Args:
            config: AppConfig instance to save
            filename: Name of the file to save to

        Returns:
            Path to the saved file
        """
        filepath = self.config_dir / filename

        # Convert Pydantic model to dict
        config_dict = config.dict()

        # Save to YAML
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Configuration saved to {filepath}")
        return filepath


# Convenience function
def load_config(environment: Optional[str] = None, config_dir: Optional[Path] = None) -> AppConfig:
    """Convenience function to load configuration.

    Args:
        environment: Environment name (development, staging, production)
        config_dir: Directory containing configuration files

    Returns:
        Validated AppConfig instance

    Examples:
        >>> config = load_config()
        >>> config = load_config(environment="production")
    """
    loader = ConfigLoader(config_dir=config_dir)
    return loader.load(environment=environment)
