"""Custom sklearn transformers for the heart disease prediction pipeline."""

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DataFrameSelector(BaseEstimator, TransformerMixin):
    """Select specific columns from a DataFrame."""

    def __init__(self, columns: List[str]):
        """
        Initialize transformer.

        Args:
            columns: List of column names to select
        """
        self.columns = columns

    def fit(self, X, y=None):
        """Fit method (does nothing)."""
        return self

    def transform(self, X):
        """
        Transform by selecting columns.

        Args:
            X: Input DataFrame

        Returns:
            Selected columns as DataFrame
        """
        return X[self.columns]


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical variables."""

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        method: str = "onehot",
        drop_first: bool = True
    ):
        """
        Initialize encoder.

        Args:
            columns: Columns to encode. If None, auto-detect
            method: Encoding method ('onehot' or 'label')
            drop_first: Drop first category in one-hot encoding
        """
        self.columns = columns
        self.method = method
        self.drop_first = drop_first
        self.encoding_map_ = {}

    def fit(self, X, y=None):
        """
        Fit encoder by learning categories.

        Args:
            X: Input DataFrame
            y: Target (not used)

        Returns:
            self
        """
        if self.columns is None:
            # Auto-detect categorical columns
            self.columns = X.select_dtypes(include=["object", "category"]).columns.tolist()

        if self.method == "label":
            for col in self.columns:
                unique_values = X[col].unique()
                self.encoding_map_[col] = {val: idx for idx, val in enumerate(unique_values)}

        return self

    def transform(self, X):
        """
        Transform by encoding categorical variables.

        Args:
            X: Input DataFrame

        Returns:
            Transformed DataFrame
        """
        X_transformed = X.copy()

        if self.method == "onehot":
            X_transformed = pd.get_dummies(
                X_transformed,
                columns=self.columns,
                drop_first=self.drop_first,
                dtype=int
            )
        elif self.method == "label":
            for col in self.columns:
                X_transformed[col] = X[col].map(self.encoding_map_[col])

        return X_transformed


class OutlierRemover(BaseEstimator, TransformerMixin):
    """Remove outliers using IQR or Z-score method."""

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        method: str = "iqr",
        threshold: float = 1.5
    ):
        """
        Initialize outlier remover.

        Args:
            columns: Columns to check for outliers. If None, use all numeric
            method: 'iqr' or 'zscore'
            threshold: Threshold for outlier detection
        """
        self.columns = columns
        self.method = method
        self.threshold = threshold
        self.bounds_ = {}

    def fit(self, X, y=None):
        """
        Fit by calculating outlier bounds.

        Args:
            X: Input DataFrame
            y: Target (not used)

        Returns:
            self
        """
        if self.columns is None:
            self.columns = X.select_dtypes(include=[np.number]).columns.tolist()

        for col in self.columns:
            if self.method == "iqr":
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - self.threshold * IQR
                upper = Q3 + self.threshold * IQR
                self.bounds_[col] = (lower, upper)

            elif self.method == "zscore":
                mean = X[col].mean()
                std = X[col].std()
                lower = mean - self.threshold * std
                upper = mean + self.threshold * std
                self.bounds_[col] = (lower, upper)

        return self

    def transform(self, X):
        """
        Transform by removing outliers.

        Args:
            X: Input DataFrame

        Returns:
            DataFrame with outliers removed
        """
        X_transformed = X.copy()

        for col, (lower, upper) in self.bounds_.items():
            mask = (X_transformed[col] >= lower) & (X_transformed[col] <= upper)
            X_transformed = X_transformed[mask]

        return X_transformed


class FeatureInteractionCreator(BaseEstimator, TransformerMixin):
    """Create interaction features between specified pairs."""

    def __init__(self, feature_pairs: List[tuple]):
        """
        Initialize interaction creator.

        Args:
            feature_pairs: List of (feature1, feature2) tuples
        """
        self.feature_pairs = feature_pairs

    def fit(self, X, y=None):
        """Fit method (does nothing)."""
        return self

    def transform(self, X):
        """
        Transform by creating interaction features.

        Args:
            X: Input DataFrame

        Returns:
            DataFrame with added interaction features
        """
        X_transformed = X.copy()

        for feat1, feat2 in self.feature_pairs:
            if feat1 in X.columns and feat2 in X.columns:
                interaction_name = f"{feat1}_x_{feat2}"
                X_transformed[interaction_name] = X[feat1] * X[feat2]

        return X_transformed


class PolynomialFeatureCreator(BaseEstimator, TransformerMixin):
    """Create polynomial features for specified columns."""

    def __init__(self, columns: List[str], degree: int = 2):
        """
        Initialize polynomial feature creator.

        Args:
            columns: Columns to create polynomial features for
            degree: Maximum polynomial degree
        """
        self.columns = columns
        self.degree = degree

    def fit(self, X, y=None):
        """Fit method (does nothing)."""
        return self

    def transform(self, X):
        """
        Transform by creating polynomial features.

        Args:
            X: Input DataFrame

        Returns:
            DataFrame with added polynomial features
        """
        X_transformed = X.copy()

        for col in self.columns:
            if col in X.columns:
                for d in range(2, self.degree + 1):
                    poly_name = f"{col}_pow{d}"
                    X_transformed[poly_name] = X[col] ** d

        return X_transformed


class DomainFeatureCreator(BaseEstimator, TransformerMixin):
    """Create domain-specific features for heart disease prediction."""

    def fit(self, X, y=None):
        """Fit method (does nothing)."""
        return self

    def transform(self, X):
        """
        Transform by creating domain-specific features.

        Args:
            X: Input DataFrame

        Returns:
            DataFrame with added domain features
        """
        X_transformed = X.copy()

        # Age risk
        if "age" in X.columns:
            X_transformed["age_risk"] = (X["age"] > 55).astype(int)

        # High cholesterol
        if "chol" in X.columns:
            X_transformed["high_chol"] = (X["chol"] > 240).astype(int)

        # High blood pressure
        if "trestbps" in X.columns:
            X_transformed["high_bp"] = (X["trestbps"] > 140).astype(int)

        # Heart rate percentage
        if "thalach" in X.columns and "age" in X.columns:
            max_hr_predicted = 220 - X["age"]
            X_transformed["hr_percentage"] = (X["thalach"] / max_hr_predicted) * 100

        # Cholesterol to age ratio
        if "chol" in X.columns and "age" in X.columns:
            X_transformed["chol_age_ratio"] = X["chol"] / X["age"]

        return X_transformed


class MissingValueImputer(BaseEstimator, TransformerMixin):
    """Impute missing values with specified strategy."""

    def __init__(self, strategy: str = "mean"):
        """
        Initialize imputer.

        Args:
            strategy: Imputation strategy ('mean', 'median', 'mode', 'constant')
        """
        self.strategy = strategy
        self.fill_values_ = {}

    def fit(self, X, y=None):
        """
        Fit by calculating fill values.

        Args:
            X: Input DataFrame
            y: Target (not used)

        Returns:
            self
        """
        for col in X.columns:
            if X[col].isnull().any():
                if self.strategy == "mean":
                    self.fill_values_[col] = X[col].mean()
                elif self.strategy == "median":
                    self.fill_values_[col] = X[col].median()
                elif self.strategy == "mode":
                    self.fill_values_[col] = X[col].mode()[0]
                elif self.strategy == "constant":
                    self.fill_values_[col] = 0
        return self

    def transform(self, X):
        """
        Transform by imputing missing values.

        Args:
            X: Input DataFrame

        Returns:
            DataFrame with imputed values
        """
        X_transformed = X.copy()

        for col, fill_value in self.fill_values_.items():
            X_transformed[col] = X_transformed[col].fillna(fill_value)

        return X_transformed


class TypeConverter(BaseEstimator, TransformerMixin):
    """Convert column data types."""

    def __init__(self, type_map: dict):
        """
        Initialize type converter.

        Args:
            type_map: Dictionary mapping column names to target types
        """
        self.type_map = type_map

    def fit(self, X, y=None):
        """Fit method (does nothing)."""
        return self

    def transform(self, X):
        """
        Transform by converting types.

        Args:
            X: Input DataFrame

        Returns:
            DataFrame with converted types
        """
        X_transformed = X.copy()

        for col, dtype in self.type_map.items():
            if col in X_transformed.columns:
                X_transformed[col] = X_transformed[col].astype(dtype)

        return X_transformed
