"""Unit tests for DataValidator module.

This module demonstrates testing validation logic with various data scenarios.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from src.data.data_validator import DataValidator
from src.config.schemas import ValidationConfig, FeatureRangeConfig
from src.utils.exceptions import DataValidationError


@pytest.fixture
def mock_config():
    """Create a mock ValidationConfig for testing."""
    return ValidationConfig(
        required_features=["age", "chol", "trestbps"],
        target_column="target",
        ranges={
            "age": FeatureRangeConfig(min=0, max=120),
            "chol": FeatureRangeConfig(min=100, max=600),
            "trestbps": FeatureRangeConfig(min=80, max=200)
        },
        missing_threshold=0.2,
        min_class_percentage=0.05
    )


@pytest.fixture
def validator(mock_config):
    """Create a DataValidator instance with mock config."""
    return DataValidator(config=mock_config)


@pytest.fixture
def valid_df():
    """Create a valid DataFrame for testing."""
    return pd.DataFrame({
        'age': [50, 60, 55, 65, 45],
        'chol': [200, 250, 220, 240, 210],
        'trestbps': [120, 130, 125, 140, 115],
        'target': [0, 1, 0, 1, 0]
    })


class TestValidatorInitialization:
    """Test DataValidator initialization."""

    def test_init_with_config(self, mock_config):
        """Test initialization with provided config."""
        validator = DataValidator(config=mock_config)

        assert validator.config == mock_config
        assert validator.expected_features == mock_config.required_features
        assert validator.target_column == mock_config.target_column

    def test_init_without_config(self):
        """Test initialization without config (uses global config)."""
        validator = DataValidator()

        assert validator.config is not None
        assert isinstance(validator.expected_features, list)


class TestSchemaValidation:
    """Test schema validation."""

    def test_validate_schema_success(self, validator, valid_df):
        """Test schema validation with valid DataFrame."""
        is_valid, issues = validator.validate_schema(valid_df)

        assert is_valid is True
        assert len(issues) == 0

    def test_validate_schema_missing_columns(self, validator):
        """Test schema validation with missing columns."""
        df = pd.DataFrame({
            'age': [50, 60],
            'chol': [200, 250]
            # Missing 'trestbps'
        })

        is_valid, issues = validator.validate_schema(df)

        assert is_valid is False
        assert len(issues) > 0
        assert any('trestbps' in issue for issue in issues)

    def test_validate_schema_strict_mode(self, validator, valid_df):
        """Test strict schema validation with extra columns."""
        df = valid_df.copy()
        df['extra_column'] = [1, 2, 3, 4, 5]

        is_valid, issues = validator.validate_schema(df, strict=True)

        assert is_valid is False
        assert any('extra_column' in issue for issue in issues)

    def test_validate_schema_non_strict_mode(self, validator, valid_df):
        """Test non-strict schema validation allows extra columns."""
        df = valid_df.copy()
        df['extra_column'] = [1, 2, 3, 4, 5]

        is_valid, issues = validator.validate_schema(df, strict=False)

        assert is_valid is True


class TestRangeValidation:
    """Test value range validation."""

    def test_validate_ranges_success(self, validator, valid_df):
        """Test range validation with valid values."""
        is_valid, issues = validator.validate_ranges(valid_df)

        assert is_valid is True
        assert len(issues) == 0

    def test_validate_ranges_below_minimum(self, validator):
        """Test range validation with values below minimum."""
        df = pd.DataFrame({
            'age': [-5, 60, 55],  # -5 is below min (0)
            'chol': [200, 250, 220],
            'trestbps': [120, 130, 125]
        })

        is_valid, issues = validator.validate_ranges(df)

        assert is_valid is False
        assert any('age' in issue and 'below minimum' in issue for issue in issues)

    def test_validate_ranges_above_maximum(self, validator):
        """Test range validation with values above maximum."""
        df = pd.DataFrame({
            'age': [50, 60, 55],
            'chol': [650, 250, 220],  # 650 is above max (600)
            'trestbps': [120, 130, 125]
        })

        is_valid, issues = validator.validate_ranges(df)

        assert is_valid is False
        assert any('chol' in issue and 'above maximum' in issue for issue in issues)


class TestDataTypeValidation:
    """Test data type validation."""

    def test_validate_data_types_success(self, validator, valid_df):
        """Test data type validation with numeric columns."""
        is_valid, issues = validator.validate_data_types(valid_df)

        assert is_valid is True
        assert len(issues) == 0

    def test_validate_data_types_non_numeric(self, validator):
        """Test data type validation with non-numeric columns."""
        df = pd.DataFrame({
            'age': ['fifty', 'sixty', 'fifty-five'],  # Should be numeric
            'chol': [200, 250, 220],
            'trestbps': [120, 130, 125]
        })

        is_valid, issues = validator.validate_data_types(df)

        assert is_valid is False
        assert any('age' in issue and 'numeric' in issue for issue in issues)


class TestMissingValueValidation:
    """Test missing value validation."""

    def test_validate_missing_values_success(self, validator, valid_df):
        """Test missing value validation with no missing values."""
        is_valid, issues = validator.validate_missing_values(valid_df)

        assert is_valid is True
        assert len(issues) == 0

    def test_validate_missing_values_within_threshold(self, validator):
        """Test missing values within acceptable threshold."""
        df = pd.DataFrame({
            'age': [50, 60, np.nan, 65, 45],  # 20% missing
            'chol': [200, 250, 220, 240, 210],
            'trestbps': [120, 130, 125, 140, 115]
        })

        # Threshold is 0.2 (20%), so this should pass
        is_valid, issues = validator.validate_missing_values(df, threshold=0.2)

        assert is_valid is True

    def test_validate_missing_values_above_threshold(self, validator):
        """Test missing values above acceptable threshold."""
        df = pd.DataFrame({
            'age': [50, np.nan, np.nan, 65, 45],  # 40% missing
            'chol': [200, 250, 220, 240, 210],
            'trestbps': [120, 130, 125, 140, 115]
        })

        # Threshold is 0.2 (20%), so this should fail
        is_valid, issues = validator.validate_missing_values(df, threshold=0.2)

        assert is_valid is False
        assert any('age' in issue for issue in issues)


class TestDuplicateValidation:
    """Test duplicate row validation."""

    def test_validate_duplicates_none(self, validator, valid_df):
        """Test validation with no duplicates."""
        is_valid, issues = validator.validate_duplicates(valid_df)

        assert is_valid is True
        # Issues list may contain info message, but is_valid should be True

    def test_validate_duplicates_present(self, validator):
        """Test validation with duplicate rows."""
        df = pd.DataFrame({
            'age': [50, 60, 50, 65],  # First and third rows are duplicates
            'chol': [200, 250, 200, 240],
            'trestbps': [120, 130, 120, 140]
        })

        is_valid, issues = validator.validate_duplicates(df)

        # Duplicates are reported but don't fail validation
        assert is_valid is True
        assert len(issues) > 0


class TestTargetDistributionValidation:
    """Test target variable distribution validation."""

    def test_validate_target_distribution_balanced(self, validator, valid_df):
        """Test validation with balanced target distribution."""
        is_valid, issues = validator.validate_target_distribution(valid_df)

        assert is_valid is True
        assert len(issues) == 0

    def test_validate_target_distribution_imbalanced(self, validator):
        """Test validation with severely imbalanced target."""
        df = pd.DataFrame({
            'age': range(100),
            'chol': range(200, 300),
            'trestbps': range(120, 220),
            'target': [0] * 98 + [1] * 2  # 98% vs 2%
        })

        # min_class_percentage is 0.05 (5%), so 2% should fail
        is_valid, issues = validator.validate_target_distribution(df)

        assert is_valid is False
        assert any('imbalance' in issue.lower() for issue in issues)

    def test_validate_target_distribution_missing_column(self, validator):
        """Test validation with missing target column."""
        df = pd.DataFrame({
            'age': [50, 60, 55],
            'chol': [200, 250, 220]
            # No target column
        })

        is_valid, issues = validator.validate_target_distribution(df)

        assert is_valid is False
        assert any('not found' in issue for issue in issues)


class TestComprehensiveValidation:
    """Test comprehensive validation (validate_all)."""

    def test_validate_all_success(self, validator, valid_df):
        """Test comprehensive validation with valid data."""
        results = validator.validate_all(valid_df, raise_on_error=False)

        assert 'summary' in results
        assert results['summary']['is_valid'] is True
        assert len(results['summary']['failed_critical']) == 0

    def test_validate_all_critical_failure(self, validator):
        """Test comprehensive validation with critical failure."""
        df = pd.DataFrame({
            'age': ['not_numeric', 60, 55],  # Invalid data type
            'chol': [200, 250, 220],
            'trestbps': [120, 130, 125],
            'target': [0, 1, 0]
        })

        with pytest.raises(DataValidationError):
            validator.validate_all(df, raise_on_error=True)

    def test_validate_all_no_raise(self, validator):
        """Test comprehensive validation without raising errors."""
        df = pd.DataFrame({
            'age': ['not_numeric', 60, 55],  # Invalid data type
            'chol': [200, 250, 220],
            'trestbps': [120, 130, 125]
        })

        results = validator.validate_all(df, raise_on_error=False)

        assert 'summary' in results
        assert results['summary']['is_valid'] is False
        assert len(results['summary']['failed_critical']) > 0


class TestDataQualityReport:
    """Test data quality reporting."""

    def test_get_data_quality_report(self, validator, valid_df):
        """Test generating data quality report."""
        report = validator.get_data_quality_report(valid_df)

        assert 'shape' in report
        assert 'n_rows' in report
        assert 'n_columns' in report
        assert 'columns' in report
        assert 'dtypes' in report
        assert 'missing_values' in report
        assert 'duplicates' in report

        assert report['n_rows'] == len(valid_df)
        assert report['n_columns'] == len(valid_df.columns)

    def test_quality_report_with_target(self, validator, valid_df):
        """Test quality report includes target distribution."""
        report = validator.get_data_quality_report(valid_df)

        assert 'target_distribution' in report
        assert 'target_balance' in report


# Run tests with: pytest tests/unit/test_data_validator.py -v
