"""Unit tests for DataLoader module.

This module demonstrates testing with dependency injection and configuration mocking.
"""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.data.data_loader import DataLoader
from src.config.schemas import DataConfig
from src.utils.exceptions import DataLoadError


@pytest.fixture
def mock_config():
    """Create a mock DataConfig for testing."""
    return DataConfig(
        raw_data_dir="tests/fixtures/raw",
        processed_data_dir="tests/fixtures/processed",
        interim_data_dir="tests/fixtures/interim",
        datasets=["cleveland", "hungarian"],
        target_column="target"
    )


@pytest.fixture
def data_loader(mock_config):
    """Create a DataLoader instance with mock config."""
    return DataLoader(config=mock_config)


class TestDataLoaderInitialization:
    """Test DataLoader initialization and configuration."""

    def test_init_with_config(self, mock_config):
        """Test initialization with provided config."""
        loader = DataLoader(config=mock_config)

        assert loader.config == mock_config
        assert loader.raw_data_path == Path(mock_config.raw_data_dir)
        assert loader.processed_data_path == Path(mock_config.processed_data_dir)
        assert loader.target_column == mock_config.target_column

    def test_init_without_config(self):
        """Test initialization without config (uses global config)."""
        # This will use the global config from get_config()
        loader = DataLoader()

        assert loader.config is not None
        assert isinstance(loader.raw_data_path, Path)
        assert isinstance(loader.target_column, str)


class TestDatasetInfo:
    """Test dataset information methods."""

    def test_get_available_datasets(self, data_loader, mock_config):
        """Test getting list of available datasets."""
        datasets = data_loader.get_available_datasets()

        assert isinstance(datasets, list)
        assert datasets == mock_config.datasets

    def test_get_dataset_info_structure(self, data_loader):
        """Test dataset info returns correct structure."""
        with patch.object(data_loader, '_dataset_exists', return_value=True):
            with patch.object(data_loader, 'get_dataset_path') as mock_path:
                mock_path.return_value = Path("test.csv")

                with patch('pandas.read_csv') as mock_read:
                    # Create mock DataFrame
                    mock_df = pd.DataFrame({
                        'age': [50, 60, 55],
                        'target': [0, 1, 0]
                    })
                    mock_read.return_value = mock_df

                    info = data_loader.get_dataset_info("cleveland")

                    assert 'name' in info
                    assert 'exists' in info
                    assert 'n_rows' in info
                    assert 'n_features' in info
                    assert 'feature_names' in info
                    assert 'target_distribution' in info

    def test_get_all_datasets_info(self, data_loader):
        """Test getting info for all datasets."""
        with patch.object(data_loader, 'get_dataset_info') as mock_info:
            mock_info.return_value = {
                'name': 'cleveland',
                'exists': True,
                'n_rows': 303
            }

            all_info = data_loader.get_all_datasets_info()

            assert isinstance(all_info, list)
            assert len(all_info) == len(data_loader.config.datasets)


class TestDataLoading:
    """Test data loading functionality."""

    def test_load_dataset_success(self, data_loader):
        """Test successful dataset loading."""
        mock_df = pd.DataFrame({
            'age': [50, 60, 55],
            'chol': [200, 250, 220],
            'target': [0, 1, 0]
        })

        with patch('pandas.read_csv', return_value=mock_df):
            with patch.object(data_loader, '_dataset_exists', return_value=True):
                df = data_loader.load_dataset("cleveland")

                assert isinstance(df, pd.DataFrame)
                assert len(df) == 3
                assert 'age' in df.columns
                assert 'target' in df.columns

    def test_load_dataset_not_found(self, data_loader):
        """Test loading non-existent dataset raises error."""
        with patch.object(data_loader, '_dataset_exists', return_value=False):
            with pytest.raises(DataLoadError, match="Dataset .* does not exist"):
                data_loader.load_dataset("nonexistent")

    def test_load_all_datasets(self, data_loader):
        """Test loading all datasets."""
        mock_df = pd.DataFrame({'age': [50], 'target': [0]})

        with patch.object(data_loader, 'load_dataset', return_value=mock_df):
            all_data = data_loader.load_all_datasets()

            assert isinstance(all_data, dict)
            assert len(all_data) == len(data_loader.config.datasets)

            for name in data_loader.config.datasets:
                assert name in all_data
                assert isinstance(all_data[name], pd.DataFrame)

    def test_load_combined_datasets(self, data_loader):
        """Test loading and combining multiple datasets."""
        mock_df1 = pd.DataFrame({'age': [50, 60], 'target': [0, 1]})
        mock_df2 = pd.DataFrame({'age': [55, 65], 'target': [0, 1]})

        def mock_load(name):
            if name == "cleveland":
                return mock_df1
            elif name == "hungarian":
                return mock_df2
            return pd.DataFrame()

        with patch.object(data_loader, 'load_dataset', side_effect=mock_load):
            combined = data_loader.load_combined_datasets(
                datasets=["cleveland", "hungarian"]
            )

            assert isinstance(combined, pd.DataFrame)
            assert len(combined) == 4  # 2 + 2 rows
            assert 'age' in combined.columns
            assert 'target' in combined.columns


class TestDataSaving:
    """Test data saving functionality."""

    def test_save_processed_data(self, data_loader, tmp_path):
        """Test saving processed data."""
        mock_df = pd.DataFrame({'age': [50, 60], 'target': [0, 1]})

        # Update config to use temp directory
        data_loader.processed_data_path = tmp_path

        with patch('pandas.DataFrame.to_csv') as mock_to_csv:
            data_loader.save_processed_data(mock_df, "test_processed")

            mock_to_csv.assert_called_once()
            args, kwargs = mock_to_csv.call_args
            assert kwargs.get('index') == False


class TestDataSplitting:
    """Test data splitting functionality."""

    def test_split_data(self, data_loader):
        """Test train/test split."""
        df = pd.DataFrame({
            'age': range(100),
            'chol': range(100, 200),
            'target': [0, 1] * 50
        })

        X_train, X_test, y_train, y_test = data_loader.split_data(
            df,
            test_size=0.2,
            random_state=42
        )

        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)

        # Check sizes
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20

        # Check that target is not in X
        assert 'target' not in X_train.columns
        assert 'target' not in X_test.columns

    def test_split_data_stratified(self, data_loader):
        """Test stratified split maintains class distribution."""
        df = pd.DataFrame({
            'age': range(100),
            'target': [0] * 70 + [1] * 30  # 70% class 0, 30% class 1
        })

        X_train, X_test, y_train, y_test = data_loader.split_data(
            df,
            test_size=0.2,
            stratify=True,
            random_state=42
        )

        # Check class distribution in train set
        train_dist = y_train.value_counts(normalize=True)
        test_dist = y_test.value_counts(normalize=True)

        # Distributions should be similar (within tolerance)
        assert abs(train_dist[0] - 0.7) < 0.1
        assert abs(test_dist[0] - 0.7) < 0.1


# Run tests with: pytest tests/unit/test_data_loader.py -v
