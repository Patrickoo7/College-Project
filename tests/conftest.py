"""Pytest configuration and shared fixtures.

This module provides common fixtures and configuration for all tests.
"""

import pytest
import os
from pathlib import Path
from typing import Generator
from unittest.mock import patch

# Set test environment before importing application code
os.environ["ENV"] = "testing"
os.environ["APP_DEBUG"] = "true"
os.environ["APP_API__ENABLE_AUTH"] = "false"  # Disable auth by default for testing


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Get the test data directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sample_csv_data() -> str:
    """Sample CSV data for testing."""
    return """age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target
63,1,3,145,233,1,0,150,0,2.3,0,0,1,1
37,1,2,130,250,0,1,187,0,3.5,0,0,2,1
41,0,1,130,204,0,0,172,0,1.4,2,0,2,1
56,1,1,120,236,0,1,178,0,0.8,2,0,2,1
57,0,0,120,354,0,1,163,1,0.6,2,0,2,1"""


@pytest.fixture(scope="function")
def temp_config_file(tmp_path) -> Generator[Path, None, None]:
    """Create a temporary config file for testing."""
    config_content = """
environment: testing
debug: true

data:
  raw_data_dir: tests/fixtures/raw
  processed_data_dir: tests/fixtures/processed
  interim_data_dir: tests/fixtures/interim
  datasets:
    - cleveland
    - hungarian
  target_column: target

api:
  host: 127.0.0.1
  port: 8000
  enable_auth: false
  max_file_size_mb: 10
  max_batch_rows: 10000
  default_model: random_forest

model:
  model_dir: tests/fixtures/models
  use_gpu: false
  n_jobs: 1

logging:
  level: DEBUG
  log_dir: tests/fixtures/logs
"""

    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)

    yield config_file

    # Cleanup (pytest handles tmp_path automatically)


@pytest.fixture(scope="function")
def mock_config():
    """Mock configuration for testing."""
    from src.config.schemas import (
        AppConfig,
        DataConfig,
        ValidationConfig,
        PreprocessingConfig,
        FeatureEngineeringConfig,
        ModelConfig,
        APIConfig
    )

    return AppConfig(
        environment="testing",
        debug=True,
        data=DataConfig(
            raw_data_dir="tests/fixtures/raw",
            processed_data_dir="tests/fixtures/processed",
            interim_data_dir="tests/fixtures/interim",
            datasets=["cleveland", "hungarian"],
            target_column="target"
        ),
        validation=ValidationConfig(),
        preprocessing=PreprocessingConfig(),
        feature_engineering=FeatureEngineeringConfig(),
        model=ModelConfig(),
        api=APIConfig(
            enable_auth=False,
            max_file_size_mb=10,
            max_batch_rows=10000,
            default_model="random_forest"
        )
    )


@pytest.fixture(autouse=True)
def reset_config():
    """Reset configuration singleton before each test."""
    from src.config.manager import reset_config_manager

    reset_config_manager()
    yield
    reset_config_manager()


@pytest.fixture(autouse=True)
def reset_container():
    """Reset DI container before each test."""
    from src.core.container import reset_container

    reset_container()
    yield
    reset_container()


@pytest.fixture
def mock_data_loader():
    """Mock DataLoader for testing."""
    from unittest.mock import Mock
    import pandas as pd

    loader = Mock()
    loader.load_dataset.return_value = pd.DataFrame({
        'age': [50, 60, 55, 65, 45],
        'chol': [200, 250, 220, 240, 210],
        'trestbps': [120, 130, 125, 140, 115],
        'target': [0, 1, 0, 1, 0]
    })

    return loader


@pytest.fixture
def mock_validator():
    """Mock DataValidator for testing."""
    from unittest.mock import Mock

    validator = Mock()
    validator.validate_all.return_value = {
        'summary': {
            'is_valid': True,
            'total_issues': 0,
            'failed_critical': []
        }
    }

    return validator


@pytest.fixture
def mock_preprocessor():
    """Mock DataPreprocessor for testing."""
    from unittest.mock import Mock
    import pandas as pd
    import numpy as np

    preprocessor = Mock()
    preprocessor.fit.return_value = preprocessor
    preprocessor.transform.return_value = pd.DataFrame({
        'age': [0.5, 0.6, 0.55, 0.65, 0.45],
        'chol': [0.5, 0.75, 0.6, 0.7, 0.55],
        'trestbps': [0.4, 0.5, 0.45, 0.6, 0.35]
    })

    return preprocessor


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "requires_model: marks tests that require trained model"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers."""
    # Skip GPU tests if no GPU available
    skip_gpu = pytest.mark.skip(reason="GPU not available")

    try:
        from src.utils.gpu_utils import get_gpu_manager
        gpu_manager = get_gpu_manager()
        gpu_available, _ = gpu_manager.detect_gpu()
    except Exception:
        gpu_available = False

    for item in items:
        if "requires_gpu" in item.keywords and not gpu_available:
            item.add_marker(skip_gpu)
