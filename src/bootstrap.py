"""Application bootstrap module for dependency injection setup.

This module initializes the dependency injection container and registers
all application services with their appropriate lifecycles.
"""

from pathlib import Path
from typing import Optional

from .config import get_config, ConfigManager
from .core import Container, get_container
from .core.interfaces import (
    IDataLoader,
    IDataValidator,
    IDataPreprocessor,
)
from .data.data_loader import DataLoader
from .data.data_validator import DataValidator
from .data.data_preprocessor import DataPreprocessor
from .utils.logger import get_logger

logger = get_logger(__name__)


def setup_container(environment: Optional[str] = None) -> Container:
    """Set up and configure the dependency injection container.

    This function registers all application services with the DI container,
    configuring their lifecycles (singleton vs transient) appropriately.

    Args:
        environment: Environment to load config for (development, staging, production).
                    If None, uses ENV environment variable.

    Returns:
        Configured Container instance

    Example:
        >>> container = setup_container(environment="development")
        >>> data_loader = container.resolve(IDataLoader)
        >>> data = data_loader.load_all_datasets()
    """
    logger.info(f"Setting up dependency injection container for environment={environment or 'default'}")

    # Load configuration
    if environment:
        ConfigManager.get_instance(environment=environment, force_reload=True)

    config = get_config()
    logger.info(f"Loaded configuration for environment: {config.environment}")

    # Get container instance
    container = get_container()

    # Register data layer services as singletons (shared across app)
    logger.debug("Registering data layer services...")

    container.register_singleton(
        IDataLoader,
        factory=lambda: DataLoader(config.data)
    )

    container.register_singleton(
        IDataValidator,
        factory=lambda: DataValidator(config.validation)
    )

    # Register preprocessor as transient (new instance per use)
    # This allows multiple preprocessing pipelines with different states
    container.register_transient(IDataPreprocessor, DataPreprocessor)

    # TODO: Register feature engineering services
    # container.register_transient(IFeatureEngineer, FeatureEngineer)

    # TODO: Register model services
    # container.register_singleton(
    #     IModelTrainer,
    #     factory=lambda: ModelTrainer(config.model, config.hyperparameters)
    # )
    # container.register_singleton(
    #     IPredictor,
    #     factory=lambda: Predictor(config.model)
    # )

    # TODO: Register MLOps services
    # container.register_singleton(
    #     IDriftDetector,
    #     factory=lambda: DriftDetector(config.mlops.drift_detection)
    # )
    # container.register_singleton(
    #     IMonitor,
    #     factory=lambda: Monitor(config.mlops.monitoring)
    # )
    # container.register_singleton(
    #     IRetrainer,
    #     factory=lambda: Retrainer(config.mlops.retraining)
    # )

    logger.info("Dependency injection container setup complete")

    return container


def initialize_app(environment: Optional[str] = None) -> dict:
    """Initialize the entire application.

    This is the main entry point for application initialization.
    It sets up configuration, dependency injection, and returns
    key application components.

    Args:
        environment: Environment to initialize for

    Returns:
        Dictionary containing initialized components:
        - config: Application configuration
        - container: DI container
        - data_loader: Data loader instance
        - data_validator: Data validator instance

    Example:
        >>> app = initialize_app(environment="production")
        >>> config = app['config']
        >>> data_loader = app['data_loader']
    """
    logger.info(f"Initializing application for environment={environment or 'default'}")

    # Setup container
    container = setup_container(environment=environment)

    # Get config
    config = get_config()

    # Resolve key services
    data_loader = container.resolve(IDataLoader)
    data_validator = container.resolve(IDataValidator)

    # Create directories if they don't exist
    _create_directories(config)

    logger.info("Application initialization complete")

    return {
        "config": config,
        "container": container,
        "data_loader": data_loader,
        "data_validator": data_validator,
    }


def _create_directories(config) -> None:
    """Create necessary directories if they don't exist.

    Args:
        config: Application configuration
    """
    directories = [
        Path(config.data.raw_data_dir),
        Path(config.data.processed_data_dir),
        Path(config.data.interim_data_dir),
        Path(config.model.model_dir),
        Path(config.model.backup_dir),
        Path(config.logging.log_dir),
        Path(config.visualization.plots_dir),
    ]

    for directory in directories:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {directory}")


def teardown_app() -> None:
    """Clean up application resources.

    This function should be called when shutting down the application
    to properly clean up resources.
    """
    logger.info("Tearing down application...")

    # Reset container
    from .core.container import reset_container
    reset_container()

    logger.info("Application teardown complete")


# Convenience functions for common initialization patterns

def initialize_for_training(environment: Optional[str] = None) -> dict:
    """Initialize application for model training.

    Args:
        environment: Environment to initialize for

    Returns:
        Dictionary with training-specific components
    """
    app = initialize_app(environment=environment)

    # Add training-specific components
    container = app['container']

    # Resolve training components
    preprocessor = container.resolve(IDataPreprocessor)

    app.update({
        "preprocessor": preprocessor,
        # TODO: Add when available
        # "feature_engineer": container.resolve(IFeatureEngineer),
        # "model_trainer": container.resolve(IModelTrainer),
    })

    logger.info("Initialized application for training")
    return app


def initialize_for_inference(environment: Optional[str] = None) -> dict:
    """Initialize application for model inference/prediction.

    Args:
        environment: Environment to initialize for

    Returns:
        Dictionary with inference-specific components
    """
    app = initialize_app(environment=environment)

    # Add inference-specific components
    container = app['container']

    # Resolve inference components
    preprocessor = container.resolve(IDataPreprocessor)

    app.update({
        "preprocessor": preprocessor,
        # TODO: Add when available
        # "predictor": container.resolve(IPredictor),
        # "monitor": container.resolve(IMonitor),
    })

    logger.info("Initialized application for inference")
    return app


def initialize_for_api(environment: Optional[str] = None) -> dict:
    """Initialize application for API serving.

    Args:
        environment: Environment to initialize for

    Returns:
        Dictionary with API-specific components
    """
    app = initialize_app(environment=environment)

    logger.info("Initialized application for API serving")
    return app


# Example usage
if __name__ == "__main__":
    # Example: Initialize for different environments
    import sys

    if len(sys.argv) > 1:
        env = sys.argv[1]
    else:
        env = "development"

    print(f"Initializing application for {env} environment...")

    app = initialize_app(environment=env)

    print(f"\nConfiguration loaded:")
    print(f"  Environment: {app['config'].environment}")
    print(f"  API Port: {app['config'].api.port}")
    print(f"  Debug Mode: {app['config'].debug}")

    print(f"\nTesting data loader...")
    data_loader = app['data_loader']
    info = data_loader.get_all_datasets_info()
    print(f"  Found {len(info)} datasets")

    print(f"\nTesting data validator...")
    validator = app['data_validator']
    print(f"  Validator configured with {len(validator.config.ranges)} feature ranges")

    print("\n✅ Application initialized successfully!")

    teardown_app()
