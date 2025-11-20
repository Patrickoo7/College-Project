"""Data validation module for ensuring data quality."""

from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from ..config import get_config
from ..config.schemas import ValidationConfig
from ..core.interfaces import IDataValidator
from ..utils.exceptions import DataValidationError
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataValidator(IDataValidator):
    """Validate data quality and schema for heart disease datasets.

    This implementation uses configuration for validation rules, including
    feature ranges, missing value thresholds, and required features.
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        """Initialize DataValidator.

        Args:
            config: Validation configuration. If None, loads from global config.
        """
        if config is None:
            app_config = get_config()
            config = app_config.validation

        self.config = config
        self.expected_features = config.required_features
        self.target_column = config.target_column

        logger.debug(f"DataValidator initialized with {len(self.expected_features)} expected features")

    def validate_schema(
        self,
        df: pd.DataFrame,
        strict: bool = False
    ) -> Tuple[bool, List[str]]:
        """Validate that DataFrame has expected columns.

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
            issues.append(f"Missing required columns: {sorted(missing_cols)}")

        # Check for extra columns in strict mode
        if strict:
            extra_cols = set(df.columns) - set(self.expected_features + [self.target_column])
            if extra_cols:
                issues.append(f"Extra columns not in schema: {sorted(extra_cols)}")

        is_valid = len(issues) == 0

        if is_valid:
            logger.info("Schema validation passed")
        else:
            logger.warning(f"Schema validation issues: {issues}")

        return is_valid, issues

    def validate_ranges(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate that values are within expected ranges.

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        # Use configured ranges
        for feature, range_config in self.config.ranges.items():
            if feature not in df.columns:
                continue

            min_val = range_config.min
            max_val = range_config.max

            col_min = df[feature].min()
            col_max = df[feature].max()

            if col_min < min_val:
                issues.append(
                    f"Column '{feature}' has values below minimum: "
                    f"{col_min} < {min_val}"
                )

            if col_max > max_val:
                issues.append(
                    f"Column '{feature}' has values above maximum: "
                    f"{col_max} > {max_val}"
                )

        is_valid = len(issues) == 0

        if is_valid:
            logger.info("Value range validation passed")
        else:
            logger.warning(f"Value range validation issues: {issues}")

        return is_valid, issues

    def validate_data_types(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate data types of columns.

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        # Check that all required features are present and numeric
        for feature in self.expected_features:
            if feature in df.columns:
                if not pd.api.types.is_numeric_dtype(df[feature]):
                    issues.append(
                        f"Column '{feature}' should be numeric but is {df[feature].dtype}"
                    )

        # Check target column if present
        if self.target_column in df.columns:
            if not pd.api.types.is_numeric_dtype(df[self.target_column]):
                issues.append(
                    f"Target column '{self.target_column}' should be numeric "
                    f"but is {df[self.target_column].dtype}"
                )

        is_valid = len(issues) == 0

        if is_valid:
            logger.info("Data type validation passed")
        else:
            logger.warning(f"Data type validation issues: {issues}")

        return is_valid, issues

    def validate_missing_values(
        self,
        df: pd.DataFrame,
        threshold: Optional[float] = None
    ) -> Tuple[bool, List[str]]:
        """Validate missing values in DataFrame.

        Args:
            df: DataFrame to validate
            threshold: Maximum allowed missing percentage (0-1).
                      If None, uses config value.

        Returns:
            Tuple of (is_valid, list of issues)
        """
        if threshold is None:
            threshold = self.config.missing_threshold

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
            logger.info(f"Missing value validation passed (threshold={threshold*100:.1f}%)")
        else:
            logger.warning(f"Missing value validation issues: {issues}")

        return is_valid, issues

    def validate_duplicates(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Check for duplicate rows.

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
            logger.warning(f"Found {n_duplicates} duplicate rows ({duplicate_pct:.2f}%)")
        else:
            logger.info("No duplicate rows found")

        # Return True even if duplicates found (not critical)
        return True, issues

    def validate_target_distribution(
        self,
        df: pd.DataFrame,
        min_class_percentage: Optional[float] = None
    ) -> Tuple[bool, List[str]]:
        """Validate target variable distribution.

        Args:
            df: DataFrame to validate
            min_class_percentage: Minimum percentage for any class.
                                 If None, uses config value.

        Returns:
            Tuple of (is_valid, list of issues)
        """
        if min_class_percentage is None:
            min_class_percentage = self.config.min_class_percentage

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
        logger.info(f"Target percentages:\n{(target_pct*100).round(2)}")

        is_valid = len(issues) == 0
        return is_valid, issues

    def validate_all(
        self,
        df: pd.DataFrame,
        strict_schema: bool = False,
        raise_on_error: bool = True
    ) -> Dict[str, Any]:
        """Run all validation checks.

        Args:
            df: DataFrame to validate
            strict_schema: Use strict schema validation
            raise_on_error: Raise exception if critical validation fails

        Returns:
            Dictionary containing validation results

        Raises:
            DataValidationError: If critical validation fails and raise_on_error=True
        """
        logger.info(f"Starting comprehensive data validation on {len(df)} rows")

        results = {
            "schema": self.validate_schema(df, strict=strict_schema),
            "data_types": self.validate_data_types(df),
            "value_ranges": self.validate_ranges(df),
            "missing_values": self.validate_missing_values(df),
            "duplicates": self.validate_duplicates(df),
            "target_distribution": self.validate_target_distribution(df),
        }

        # Check if any critical validation failed
        critical_checks = ["schema", "data_types"]
        failed_critical = [
            check for check in critical_checks
            if not results[check][0]
        ]

        # Count total issues
        total_issues = sum(len(issues) for _, issues in results.values())

        # Create summary
        results["summary"] = {
            "total_checks": len(results) - 1,  # Exclude summary itself
            "passed_checks": sum(1 for k, v in results.items() if k != "summary" and v[0]),
            "failed_critical": failed_critical,
            "total_issues": total_issues,
            "is_valid": len(failed_critical) == 0,
        }

        if failed_critical:
            error_msg = f"Critical validation failed: {failed_critical}"
            logger.error(error_msg)
            if raise_on_error:
                raise DataValidationError(error_msg)
        else:
            logger.info(f"Validation complete. {total_issues} total issues found.")

        return results

    def get_data_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data quality report.

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
            "dtypes": {k: str(v) for k, v in df.dtypes.to_dict().items()},
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
            "duplicates": int(df.duplicated().sum()),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
        }

        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 0:
            report["numeric_summary"] = df[numeric_cols].describe().to_dict()
            report["numeric_columns"] = numeric_cols

        # Categorical columns summary
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if len(categorical_cols) > 0:
            report["categorical_summary"] = {
                col: df[col].value_counts().to_dict()
                for col in categorical_cols
            }
            report["categorical_columns"] = categorical_cols

        # Target distribution if present
        if self.target_column in df.columns:
            report["target_distribution"] = df[self.target_column].value_counts().to_dict()
            report["target_balance"] = (
                df[self.target_column].value_counts(normalize=True) * 100
            ).round(2).to_dict()

        return report

    def validate_feature_completeness(
        self,
        df: pd.DataFrame,
        min_completeness: float = 0.9
    ) -> Tuple[bool, List[str]]:
        """Validate that features have sufficient non-missing data.

        Args:
            df: DataFrame to validate
            min_completeness: Minimum completeness ratio (0-1)

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        completeness = (1 - df.isnull().sum() / len(df))

        for feature in self.expected_features:
            if feature in df.columns:
                if completeness[feature] < min_completeness:
                    issues.append(
                        f"Feature '{feature}' has insufficient completeness: "
                        f"{completeness[feature]*100:.1f}% (min: {min_completeness*100:.1f}%)"
                    )

        is_valid = len(issues) == 0

        if is_valid:
            logger.info(f"Feature completeness validation passed (min={min_completeness*100:.1f}%)")
        else:
            logger.warning(f"Feature completeness issues: {issues}")

        return is_valid, issues

    def validate_unique_ids(
        self,
        df: pd.DataFrame,
        id_column: Optional[str] = None
    ) -> Tuple[bool, List[str]]:
        """Validate that ID column has unique values.

        Args:
            df: DataFrame to validate
            id_column: Name of ID column. If None, skips validation.

        Returns:
            Tuple of (is_valid, list of issues)
        """
        if id_column is None or id_column not in df.columns:
            return True, []

        issues = []

        n_unique = df[id_column].nunique()
        n_rows = len(df)

        if n_unique != n_rows:
            issues.append(
                f"ID column '{id_column}' has duplicate values: "
                f"{n_unique} unique IDs for {n_rows} rows"
            )

        is_valid = len(issues) == 0

        if is_valid:
            logger.info(f"ID uniqueness validation passed for '{id_column}'")
        else:
            logger.warning(f"ID uniqueness issues: {issues}")

        return is_valid, issues
