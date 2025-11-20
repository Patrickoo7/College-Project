"""Core application infrastructure.

This module provides core infrastructure components including dependency injection,
interfaces, and base classes.
"""

from .container import Container, get_container
from .interfaces import (
    IDataLoader,
    IDataPreprocessor,
    IDataValidator,
    IFeatureEngineer,
    IModelTrainer,
    IPredictor,
    IDriftDetector,
    IMonitor,
    IRetrainer,
)

__all__ = [
    "Container",
    "get_container",
    "IDataLoader",
    "IDataPreprocessor",
    "IDataValidator",
    "IFeatureEngineer",
    "IModelTrainer",
    "IPredictor",
    "IDriftDetector",
    "IMonitor",
    "IRetrainer",
]
