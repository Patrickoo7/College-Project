"""Data validation module for ensuring data quality."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils.config import get_config
from ..utils.exceptions import DataValidationError
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataValidator:
    """Validate data quality and schema for heart disease datasets."""

    def __init__(self):
        """Initialize DataValidator."""
        self.config = get_config()
        self.expected_features = self._get_expected_features()
        self.target_column = self.config.get("features.target", "target")

    def _get_expected_features(self) -> List[str]:
        """
        Get expected features from configuration.

        Returns:
            List of expected feature names
        """
        categorical = self.config.get("features.categorical", [])
        continuous = self.config.get("features.continuous", [])
        target = self.config.get("features.target", "target")

        return categorical + continuous + [target]

    def validate_schema(
        self,
        df: pd.DataFrame,
        strict: bool = False
    ) -> Tuple[bool, List[str]]:
        """
        Validate that DataFrame has expected columns.

        Args:
            df: DataFrame to validate
            strict: If True, extra columns cause validation to fail

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        # Check for missing columns
        missing_cols = set(self.expected_features) - set(df.columns)
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")

        # Check for extra columns in strict mode
        if strict:
            extra_cols = set(df.columns) - set(self.expected_features)
            if extra_cols:
                issues.append(f"Extra columns: {extra_cols}")

        is_valid = len(issues) == 0

        if is_valid:
            logger.info("Schema validation passed")
        else:
            logger.warning(f"Schema validation issues: {issues}")

        return is_valid, issues

    def validate_data_types(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate data types of columns.

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        categorical_features = self.config.get("features.categorical", [])
        continuous_features = self.config.get("features.continuous", [])

        # Check continuous features are numeric
        for col in continuous_features:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    issues.append(f"Column '{col}' should be numeric but is {df[col].dtype}")

        # Check categorical features (can be numeric or object)
        for col in categorical_features:
            if col in df.columns:
                if not (pd.api.types.is_numeric_dtype(df[col]) or
                       pd.api.types.is_object_dtype(df[col])):
                    issues.append(f"Column '{col}' has unexpected dtype: {df[col].dtype}")

        is_valid = len(issues) == 0

        if is_valid:
            logger.info("Data type validation passed")
        else:
            logger.warning(f"Data type validation issues: {issues}")

        return is_valid, issues

    def validate_value_ranges(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that values are within expected ranges.

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        # Define expected ranges
        ranges = {
            "age": (0, 120),
            "sex": (0, 1),
            "cp": (0, 4),
            "trestbps": (50, 250),
            "chol": (100, 600),
            "fbs": (0, 1),
            "restecg": (0, 2),
            "thalach": (50, 250),
            "exang": (0, 1),
            "oldpeak": (0, 10),
            "slope": (0, 3),
            "ca": (0, 4),
            "thal": (0, 7),
            "target": (0, 4),
        }

        for col, (min_val, max_val) in ranges.items():
            if col in df.columns:
                col_min = df[col].min()
                col_max = df[col].max()

                if col_min < min_val:
                    issues.append(
                        f"Column '{col}' has values below minimum: "
                        f"{col_min} < {min_val}"
                    )

                if col_max > max_val:
                    issues.append(
                        f"Column '{col}' has values above maximum: "
                        f"{col_max} > {max_val}"
                    )

        is_valid = len(issues) == 0

        if is_valid:
            logger.info("Value range validation passed")
        else:
            logger.warning(f"Value range validation issues: {issues}")

        return is_valid, issues

    def validate_missing_values(
        self,
        df: pd.DataFrame,
        threshold: float = 0.5
    ) -> Tuple[bool, List[str]]:
        """
        Validate missing values in DataFrame.

        Args:
            df: DataFrame to validate
            threshold: Maximum allowed missing percentage (0-1)

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        missing_pct = df.isnull().sum() / len(df)

        for col, pct in missing_pct.items():
            if pct > threshold:
                issues.append(
                    f"Column '{col}' has {pct*100:.2f}% missing values "
                    f"(threshold: {threshold*100:.2f}%)"
                )

        is_valid = len(issues) == 0

        if is_valid:
            logger.info("Missing value validation passed")
        else:
            logger.warning(f"Missing value validation issues: {issues}")

        return is_valid, issues

    def validate_duplicates(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Check for duplicate rows.

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        n_duplicates = df.duplicated().sum()

        if n_duplicates > 0:
            duplicate_pct = (n_duplicates / len(df)) * 100
            issues.append(
                f"Found {n_duplicates} duplicate rows ({duplicate_pct:.2f}%)"
            )
            logger.warning(f"Found {n_duplicates} duplicate rows")
        else:
            logger.info("No duplicate rows found")

        # Return True even if duplicates found (not critical)
        return True, issues

    def validate_target_distribution(
        self,
        df: pd.DataFrame,
        min_class_percentage: float = 0.1
    ) -> Tuple[bool, List[str]]:
        """
        Validate target variable distribution.

        Args:
            df: DataFrame to validate
            min_class_percentage: Minimum percentage for any class

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        if self.target_column not in df.columns:
            issues.append(f"Target column '{self.target_column}' not found")
            return False, issues

        target_counts = df[self.target_column].value_counts()
        target_pct = target_counts / len(df)

        # Check for class imbalance
        min_pct = target_pct.min()
        if min_pct < min_class_percentage:
            issues.append(
                f"Severe class imbalance: smallest class has {min_pct*100:.2f}% "
                f"(threshold: {min_class_percentage*100:.2f}%)"
            )

        # Log target distribution
        logger.info(f"Target distribution:\n{target_counts}")
        logger.info(f"Target percentages:\n{target_pct*100}")

        is_valid = len(issues) == 0
        return is_valid, issues

    def validate_all(
        self,
        df: pd.DataFrame,
        strict_schema: bool = False,
        missing_threshold: float = 0.5
    ) -> Dict[str, any]:
        """
        Run all validation checks.

        Args:
            df: DataFrame to validate
            strict_schema: Use strict schema validation
            missing_threshold: Threshold for missing values

        Returns:
            Dictionary containing validation results

        Raises:
            DataValidationError: If critical validation fails
        """
        logger.info("Starting comprehensive data validation")

        results = {
            "schema": self.validate_schema(df, strict=strict_schema),
            "data_types": self.validate_data_types(df),
            "value_ranges": self.validate_value_ranges(df),
            "missing_values": self.validate_missing_values(df, missing_threshold),
            "duplicates": self.validate_duplicates(df),
            "target_distribution": self.validate_target_distribution(df),
        }

        # Check if any critical validation failed
        critical_checks = ["schema", "data_types"]
        failed_critical = [
            check for check in critical_checks
            if not results[check][0]
        ]

        if failed_critical:
            error_msg = f"Critical validation failed: {failed_critical}"
            logger.error(error_msg)
            raise DataValidationError(error_msg)

        # Count total issues
        total_issues = sum(len(issues) for _, issues in results.values())

        logger.info(f"Validation complete. Total issues: {total_issues}")

        return results

    def get_data_quality_report(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Generate comprehensive data quality report.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary containing quality metrics
        """
        logger.info("Generating data quality report")

        report = {
            "shape": df.shape,
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
            "duplicates": df.duplicated().sum(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
        }

        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            report["numeric_summary"] = df[numeric_cols].describe().to_dict()

        # Categorical columns summary
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        if len(categorical_cols) > 0:
            report["categorical_summary"] = {
                col: df[col].value_counts().to_dict()
                for col in categorical_cols
            }

        # Target distribution if present
        if self.target_column in df.columns:
            report["target_distribution"] = df[self.target_column].value_counts().to_dict()

        return report
