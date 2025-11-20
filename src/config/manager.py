"""Configuration manager providing singleton access to application configuration."""

import threading
from typing import Optional
from pathlib import Path
from .loader import ConfigLoader
from .schemas import AppConfig
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """Singleton configuration manager for global config access.

    This class ensures that configuration is loaded once and shared across
    the entire application. It supports hot-reloading for development.

    Examples:
        >>> # Get the singleton instance
        >>> config_mgr = ConfigManager.get_instance()
        >>>
        >>> # Access configuration
        >>> config = config_mgr.config
        >>> print(config.api.port)
        8000
        >>>
        >>> # Reload configuration
        >>> config_mgr.reload()
    """

    _instance: Optional["ConfigManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, config_dir: Optional[Path] = None, environment: Optional[str] = None):
        """Initialize the configuration manager.

        Note: Use get_instance() instead of direct instantiation.

        Args:
            config_dir: Directory containing configuration files
            environment: Environment name (development, staging, production)
        """
        if ConfigManager._instance is not None:
            raise RuntimeError("ConfigManager is a singleton. Use get_instance() instead.")

        self._config_dir = config_dir
        self._environment = environment
        self._loader = ConfigLoader(config_dir=config_dir)
        self._config: Optional[AppConfig] = None
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from files."""
        try:
            self._config = self._loader.load(environment=self._environment)
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    @classmethod
    def get_instance(
        cls,
        config_dir: Optional[Path] = None,
        environment: Optional[str] = None,
        force_reload: bool = False
    ) -> "ConfigManager":
        """Get the singleton instance of ConfigManager.

        Args:
            config_dir: Directory containing configuration files
            environment: Environment name
            force_reload: Force reload even if instance exists

        Returns:
            ConfigManager singleton instance
        """
        if cls._instance is None or force_reload:
            with cls._lock:
                if cls._instance is None or force_reload:
                    cls._instance = cls(config_dir=config_dir, environment=environment)

        return cls._instance

    @property
    def config(self) -> AppConfig:
        """Get the current configuration.

        Returns:
            AppConfig instance

        Raises:
            RuntimeError: If configuration is not loaded
        """
        if self._config is None:
            raise RuntimeError("Configuration not loaded")
        return self._config

    def reload(self) -> AppConfig:
        """Reload configuration from files.

        This is useful for development when configuration changes need to be
        applied without restarting the application.

        Returns:
            Reloaded AppConfig instance
        """
        logger.info("Reloading configuration...")
        self._load_config()
        return self._config

    def get(self, key_path: str, default: any = None) -> any:
        """Get a configuration value by dot-separated path.

        Args:
            key_path: Dot-separated path to the configuration value
                     (e.g., 'api.port', 'mlops.monitoring.accuracy_drop_threshold')
            default: Default value if key not found

        Returns:
            Configuration value or default

        Examples:
            >>> config_mgr = ConfigManager.get_instance()
            >>> port = config_mgr.get('api.port', 8000)
            >>> threshold = config_mgr.get('mlops.monitoring.accuracy_drop_threshold')
        """
        try:
            keys = key_path.split('.')
            value = self.config

            for key in keys:
                value = getattr(value, key)

            return value
        except AttributeError:
            logger.warning(f"Configuration key not found: {key_path}")
            return default

    def update(self, key_path: str, value: any) -> None:
        """Update a configuration value by dot-separated path.

        Note: This only updates the in-memory configuration, not the file.
        Use with caution as changes will be lost on reload.

        Args:
            key_path: Dot-separated path to the configuration value
            value: New value to set

        Examples:
            >>> config_mgr = ConfigManager.get_instance()
            >>> config_mgr.update('api.port', 9000)
        """
        try:
            keys = key_path.split('.')
            obj = self.config

            # Navigate to the parent object
            for key in keys[:-1]:
                obj = getattr(obj, key)

            # Set the final value
            setattr(obj, keys[-1], value)
            logger.info(f"Updated configuration: {key_path} = {value}")

        except AttributeError as e:
            logger.error(f"Failed to update configuration key {key_path}: {e}")
            raise

    def save_current(self, filename: str = "runtime_config.yaml") -> Path:
        """Save the current in-memory configuration to a file.

        Args:
            filename: Name of the file to save to

        Returns:
            Path to the saved file
        """
        return self._loader.save(self.config, filename=filename)

    def __repr__(self) -> str:
        """String representation of the config manager."""
        env = self._config.environment if self._config else "not loaded"
        return f"<ConfigManager environment={env}>"


# Convenience function for global config access
_global_config_manager: Optional[ConfigManager] = None


def get_config(force_reload: bool = False) -> AppConfig:
    """Get the global application configuration.

    This is a convenience function for accessing the configuration without
    directly using the ConfigManager singleton.

    Args:
        force_reload: Force reload configuration from files

    Returns:
        AppConfig instance

    Examples:
        >>> from src.config import get_config
        >>> config = get_config()
        >>> print(config.api.port)
    """
    global _global_config_manager

    if _global_config_manager is None or force_reload:
        _global_config_manager = ConfigManager.get_instance(force_reload=force_reload)

    return _global_config_manager.config


def reload_config() -> AppConfig:
    """Reload the global application configuration.

    Returns:
        Reloaded AppConfig instance
    """
    global _global_config_manager

    if _global_config_manager is None:
        _global_config_manager = ConfigManager.get_instance()

    return _global_config_manager.reload()
