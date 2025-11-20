"""Data preprocessing module for cleaning and transforming data."""

from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from ..config import get_config
from ..config.schemas import PreprocessingConfig
from ..core.interfaces import IDataPreprocessor
from ..utils.exceptions import PreprocessingError
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataPreprocessor(IDataPreprocessor):
    """Preprocess and clean heart disease data.

    This implementation uses configuration for all preprocessing parameters,
    including scaling methods, imputation strategies, and outlier detection.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """Initialize DataPreprocessor.

        Args:
            config: Preprocessing configuration. If None, loads from global config.
        """
        if config is None:
            app_config = get_config()
            config = app_config.preprocessing

        self.config = config
        self.scaler = None
        self.imputer = None
        self._is_fitted = False

        logger.debug(f"DataPreprocessor initialized with scaling_method={config.scaling_method}")

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DataPreprocessor":
        """Fit the preprocessor on training data.

        Args:
            X: Features DataFrame
            y: Optional target Series (not used, kept for interface compatibility)

        Returns:
            self

        Raises:
            PreprocessingError: If fitting fails
        """
        logger.info(f"Fitting preprocessor on {len(X)} samples")

        try:
            # Fit imputer if needed
            if self.config.imputation_strategy != "none":
                self._fit_imputer(X)

            # Fit scaler
            if self.config.scaling_method != "none":
                self._fit_scaler(X)

            self._is_fitted = True
            logger.info("Preprocessor fitted successfully")

            return self

        except Exception as e:
            raise PreprocessingError(f"Failed to fit preprocessor: {str(e)}")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted preprocessor.

        Args:
            X: Features DataFrame to transform

        Returns:
            Transformed DataFrame

        Raises:
            PreprocessingError: If transformation fails or preprocessor not fitted
        """
        if not self._is_fitted:
            raise PreprocessingError("Preprocessor must be fitted before transform")

        logger.info(f"Transforming {len(X)} samples")

        try:
            X_transformed = X.copy()

            # Apply imputation
            if self.imputer is not None:
                X_transformed = self._apply_imputation(X_transformed)

            # Apply scaling
            if self.scaler is not None:
                X_transformed = self._apply_scaling(X_transformed)

            logger.info("Transformation completed")
            return X_transformed

        except Exception as e:
            raise PreprocessingError(f"Failed to transform data: {str(e)}")

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step.

        Args:
            X: Features DataFrame
            y: Optional target Series

        Returns:
            Transformed DataFrame
        """
        return self.fit(X, y).transform(X)

    def _fit_imputer(self, X: pd.DataFrame) -> None:
        """Fit imputer on data.

        Args:
            X: Features DataFrame
        """
        strategy = self.config.imputation_strategy

        if strategy == "mean":
            self.imputer = SimpleImputer(strategy="mean")
        elif strategy == "median":
            self.imputer = SimpleImputer(strategy="median")
        elif strategy == "mode":
            self.imputer = SimpleImputer(strategy="most_frequent")
        elif strategy == "knn":
            self.imputer = KNNImputer(n_neighbors=self.config.knn_neighbors)
        else:
            raise PreprocessingError(f"Unknown imputation strategy: {strategy}")

        # Fit on numeric columns only
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            self.imputer.fit(X[numeric_cols])
            logger.debug(f"Imputer fitted with strategy={strategy}")

    def _apply_imputation(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted imputer to data.

        Args:
            X: Features DataFrame

        Returns:
            Imputed DataFrame
        """
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return X

        X_imputed = X.copy()
        X_imputed[numeric_cols] = self.imputer.transform(X[numeric_cols])

        return X_imputed

    def _fit_scaler(self, X: pd.DataFrame) -> None:
        """Fit scaler on data.

        Args:
            X: Features DataFrame
        """
        method = self.config.scaling_method

        if method == "standard":
            self.scaler = StandardScaler()
        elif method == "minmax":
            self.scaler = MinMaxScaler()
        elif method == "robust":
            self.scaler = RobustScaler()
        else:
            raise PreprocessingError(f"Unknown scaling method: {method}")

        # Fit on numeric columns only
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            self.scaler.fit(X[numeric_cols])
            logger.debug(f"Scaler fitted with method={method}")

    def _apply_scaling(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted scaler to data.

        Args:
            X: Features DataFrame

        Returns:
            Scaled DataFrame
        """
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return X

        X_scaled = X.copy()
        X_scaled[numeric_cols] = self.scaler.transform(X[numeric_cols])

        return X_scaled

    # Additional helper methods for complete preprocessing pipeline

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: Optional[str] = None
    ) -> pd.DataFrame:
        """Handle missing values in the dataset.

        Args:
            df: Input DataFrame
            strategy: Imputation strategy. If None, uses config default

        Returns:
            DataFrame with missing values handled

        Raises:
            PreprocessingError: If handling fails
        """
        if strategy is None:
            strategy = self.config.imputation_strategy

        logger.info(f"Handling missing values with strategy: {strategy}")

        # Log missing values before handling
        missing_before = df.isnull().sum()
        if missing_before.sum() > 0:
            logger.info(f"Missing values before: {missing_before[missing_before > 0].to_dict()}")
        else:
            logger.info("No missing values found")
            return df.copy()

        try:
            if strategy == "mean":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                imputer = SimpleImputer(strategy="mean")
                df_clean = df.copy()
                df_clean[numeric_cols] = imputer.fit_transform(df[numeric_cols])

            elif strategy == "median":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                imputer = SimpleImputer(strategy="median")
                df_clean = df.copy()
                df_clean[numeric_cols] = imputer.fit_transform(df[numeric_cols])

            elif strategy == "mode":
                imputer = SimpleImputer(strategy="most_frequent")
                df_clean = pd.DataFrame(
                    imputer.fit_transform(df),
                    columns=df.columns,
                    index=df.index
                )

            elif strategy == "knn":
                imputer = KNNImputer(n_neighbors=self.config.knn_neighbors)
                df_clean = pd.DataFrame(
                    imputer.fit_transform(df),
                    columns=df.columns,
                    index=df.index
                )

            else:
                raise PreprocessingError(f"Unknown imputation strategy: {strategy}")

            logger.info("Missing values handled successfully")
            return df_clean

        except Exception as e:
            raise PreprocessingError(f"Failed to handle missing values: {str(e)}")

    def remove_outliers(
        self,
        df: pd.DataFrame,
        method: Optional[str] = None,
        threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """Remove outliers from continuous features.

        Args:
            df: Input DataFrame
            method: Outlier detection method. If None, uses config default
            threshold: Threshold for outlier detection. If None, uses config default

        Returns:
            DataFrame with outliers removed
        """
        if method is None:
            method = self.config.outlier_detection_method

        if threshold is None:
            threshold = self.config.zscore_threshold if method == "zscore" else self.config.iqr_multiplier

        logger.info(f"Removing outliers using {method} method (threshold={threshold})")

        df_clean = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        try:
            if method == "iqr":
                for col in numeric_cols:
                    Q1 = df_clean[col].quantile(0.25)
                    Q3 = df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR

                    outliers_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
                    n_outliers = outliers_mask.sum()

                    if n_outliers > 0:
                        logger.debug(f"Found {n_outliers} outliers in '{col}'")

                    df_clean = df_clean[~outliers_mask]

            elif method == "zscore":
                for col in numeric_cols:
                    z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                    outliers_mask = z_scores > threshold
                    n_outliers = outliers_mask.sum()

                    if n_outliers > 0:
                        logger.debug(f"Found {n_outliers} outliers in '{col}'")

                    df_clean = df_clean[~outliers_mask]

            else:
                raise PreprocessingError(f"Unknown outlier method: {method}")

            logger.info(f"Removed {len(df) - len(df_clean)} rows with outliers")
            return df_clean

        except Exception as e:
            raise PreprocessingError(f"Failed to remove outliers: {str(e)}")

    def scale_features(
        self,
        X: np.ndarray,
        method: Optional[str] = None,
        fit: bool = True
    ) -> np.ndarray:
        """Scale numerical features.

        Args:
            X: Feature matrix
            method: Scaling method. If None, uses config default
            fit: If True, fit scaler. If False, use existing scaler

        Returns:
            Scaled feature matrix

        Raises:
            PreprocessingError: If scaling fails
        """
        if method is None:
            method = self.config.scaling_method

        logger.info(f"Scaling features using {method} method")

        try:
            if fit or self.scaler is None:
                if method == "standard":
                    self.scaler = StandardScaler()
                elif method == "minmax":
                    self.scaler = MinMaxScaler()
                elif method == "robust":
                    self.scaler = RobustScaler()
                elif method == "none":
                    return X  # No scaling
                else:
                    raise PreprocessingError(f"Unknown scaling method: {method}")

                X_scaled = self.scaler.fit_transform(X)
                logger.debug("Fitted and transformed features")
            else:
                X_scaled = self.scaler.transform(X)
                logger.debug("Transformed features using existing scaler")

            return X_scaled

        except Exception as e:
            raise PreprocessingError(f"Failed to scale features: {str(e)}")

    def get_scaler(self):
        """Get the fitted scaler.

        Returns:
            Fitted scaler object

        Raises:
            PreprocessingError: If scaler not fitted
        """
        if self.scaler is None:
            raise PreprocessingError("Scaler not fitted. Call fit() first.")
        return self.scaler

    def get_imputer(self):
        """Get the fitted imputer.

        Returns:
            Fitted imputer object

        Raises:
            PreprocessingError: If imputer not fitted
        """
        if self.imputer is None:
            raise PreprocessingError("Imputer not fitted. Call fit() first.")
        return self.imputer

    def is_fitted(self) -> bool:
        """Check if preprocessor is fitted.

        Returns:
            True if fitted, False otherwise
        """
        return self._is_fitted

    def reset(self) -> None:
        """Reset the preprocessor to unfitted state."""
        self.scaler = None
        self.imputer = None
        self._is_fitted = False
        logger.info("Preprocessor reset to unfitted state")
