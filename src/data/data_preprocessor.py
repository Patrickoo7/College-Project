"""Data preprocessing module for cleaning and transforming data."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from ..utils.config import get_config
from ..utils.exceptions import PreprocessingError
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """Preprocess and clean heart disease data."""

    def __init__(self):
        """Initialize DataPreprocessor."""
        self.config = get_config()
        self.target_column = self.config.get("features.target", "target")
        self.scaler = None
        self.imputer = None

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Args:
            df: Input DataFrame
            strategy: Imputation strategy ('drop', 'mean', 'median', 'mode', 'knn')
                     If None, uses config default

        Returns:
            DataFrame with missing values handled

        Raises:
            PreprocessingError: If handling fails
        """
        if strategy is None:
            strategy = self.config.get("preprocessing.handle_missing", "drop")

        logger.info(f"Handling missing values with strategy: {strategy}")

        # Log missing values before handling
        missing_before = df.isnull().sum()
        if missing_before.sum() > 0:
            logger.info(f"Missing values before handling:\n{missing_before[missing_before > 0]}")
        else:
            logger.info("No missing values found")
            return df.copy()

        try:
            if strategy == "drop":
                df_clean = df.dropna()
                logger.info(f"Dropped {len(df) - len(df_clean)} rows with missing values")

            elif strategy == "mean":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                self.imputer = SimpleImputer(strategy="mean")
                df_clean = df.copy()
                df_clean[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
                logger.info("Imputed missing values with mean")

            elif strategy == "median":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                self.imputer = SimpleImputer(strategy="median")
                df_clean = df.copy()
                df_clean[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
                logger.info("Imputed missing values with median")

            elif strategy == "mode":
                self.imputer = SimpleImputer(strategy="most_frequent")
                df_clean = pd.DataFrame(
                    self.imputer.fit_transform(df),
                    columns=df.columns,
                    index=df.index
                )
                logger.info("Imputed missing values with mode")

            elif strategy == "knn":
                self.imputer = KNNImputer(n_neighbors=5)
                df_clean = pd.DataFrame(
                    self.imputer.fit_transform(df),
                    columns=df.columns,
                    index=df.index
                )
                logger.info("Imputed missing values with KNN")

            else:
                raise PreprocessingError(f"Unknown imputation strategy: {strategy}")

            # Log missing values after handling
            missing_after = df_clean.isnull().sum()
            if missing_after.sum() > 0:
                logger.warning(f"Missing values remaining:\n{missing_after[missing_after > 0]}")

            return df_clean

        except Exception as e:
            raise PreprocessingError(f"Failed to handle missing values: {str(e)}")

    def remove_outliers(
        self,
        df: pd.DataFrame,
        method: Optional[str] = None,
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Remove outliers from continuous features.

        Args:
            df: Input DataFrame
            method: Outlier detection method ('iqr', 'zscore')
                   If None, uses config default
            threshold: Threshold for outlier detection

        Returns:
            DataFrame with outliers removed
        """
        if method is None:
            method = self.config.get("preprocessing.outlier_method", "iqr")

        continuous_features = self.config.get("features.continuous", [])
        continuous_features = [f for f in continuous_features if f in df.columns]

        if not continuous_features:
            logger.info("No continuous features to check for outliers")
            return df.copy()

        logger.info(f"Removing outliers using {method} method")

        df_clean = df.copy()

        try:
            if method == "iqr":
                for col in continuous_features:
                    Q1 = df_clean[col].quantile(0.25)
                    Q3 = df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    outliers_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
                    n_outliers = outliers_mask.sum()

                    if n_outliers > 0:
                        logger.info(f"Found {n_outliers} outliers in '{col}'")

                    df_clean = df_clean[~outliers_mask]

            elif method == "zscore":
                for col in continuous_features:
                    z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                    outliers_mask = z_scores > threshold
                    n_outliers = outliers_mask.sum()

                    if n_outliers > 0:
                        logger.info(f"Found {n_outliers} outliers in '{col}'")

                    df_clean = df_clean[~outliers_mask]

            else:
                raise PreprocessingError(f"Unknown outlier method: {method}")

            logger.info(f"Removed {len(df) - len(df_clean)} rows with outliers")
            return df_clean

        except Exception as e:
            raise PreprocessingError(f"Failed to remove outliers: {str(e)}")

    def drop_features(
        self,
        df: pd.DataFrame,
        features_to_drop: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Drop specified features from DataFrame.

        Args:
            df: Input DataFrame
            features_to_drop: List of features to drop
                             If None, uses config default

        Returns:
            DataFrame with features dropped
        """
        if features_to_drop is None:
            features_to_drop = self.config.get("features.drop_features", [])

        if not features_to_drop:
            return df.copy()

        existing_features = [f for f in features_to_drop if f in df.columns]

        if existing_features:
            logger.info(f"Dropping features: {existing_features}")
            df_clean = df.drop(columns=existing_features)
        else:
            logger.info("No features to drop")
            df_clean = df.copy()

        return df_clean

    def encode_categorical_features(
        self,
        df: pd.DataFrame,
        method: str = "onehot"
    ) -> pd.DataFrame:
        """
        Encode categorical features.

        Args:
            df: Input DataFrame
            method: Encoding method ('onehot', 'label')

        Returns:
            DataFrame with encoded categorical features
        """
        categorical_features = self.config.get("features.categorical", [])
        categorical_features = [f for f in categorical_features if f in df.columns]

        # Remove target from categorical features
        if self.target_column in categorical_features:
            categorical_features.remove(self.target_column)

        if not categorical_features:
            logger.info("No categorical features to encode")
            return df.copy()

        logger.info(f"Encoding categorical features: {categorical_features}")

        try:
            if method == "onehot":
                df_encoded = pd.get_dummies(
                    df,
                    columns=categorical_features,
                    drop_first=True,
                    dtype=int
                )
                logger.info(f"One-hot encoded {len(categorical_features)} features")

            elif method == "label":
                df_encoded = df.copy()
                for col in categorical_features:
                    df_encoded[col] = pd.factorize(df[col])[0]
                logger.info(f"Label encoded {len(categorical_features)} features")

            else:
                raise PreprocessingError(f"Unknown encoding method: {method}")

            return df_encoded

        except Exception as e:
            raise PreprocessingError(f"Failed to encode categorical features: {str(e)}")

    def scale_features(
        self,
        X: np.ndarray,
        method: Optional[str] = None,
        fit: bool = True
    ) -> np.ndarray:
        """
        Scale numerical features.

        Args:
            X: Feature matrix
            method: Scaling method ('standard', 'minmax', 'robust')
                   If None, uses config default
            fit: If True, fit scaler. If False, use existing scaler

        Returns:
            Scaled feature matrix

        Raises:
            PreprocessingError: If scaling fails
        """
        if method is None:
            method = self.config.get("preprocessing.scaling_method", "standard")

        logger.info(f"Scaling features using {method} method")

        try:
            if fit or self.scaler is None:
                if method == "standard":
                    self.scaler = StandardScaler()
                elif method == "minmax":
                    self.scaler = MinMaxScaler()
                elif method == "robust":
                    self.scaler = RobustScaler()
                else:
                    raise PreprocessingError(f"Unknown scaling method: {method}")

                X_scaled = self.scaler.fit_transform(X)
                logger.info("Fitted and transformed features")
            else:
                X_scaled = self.scaler.transform(X)
                logger.info("Transformed features using existing scaler")

            return X_scaled

        except Exception as e:
            raise PreprocessingError(f"Failed to scale features: {str(e)}")

    def split_data(
        self,
        df: pd.DataFrame,
        test_size: Optional[float] = None,
        random_state: Optional[int] = None,
        stratify: Optional[bool] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets.

        Args:
            df: Input DataFrame
            test_size: Test set size (0-1). If None, uses config default
            random_state: Random seed. If None, uses config default
            stratify: Whether to stratify split. If None, uses config default

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)

        Raises:
            PreprocessingError: If splitting fails
        """
        if test_size is None:
            test_size = self.config.get("preprocessing.test_size", 0.25)

        if random_state is None:
            random_state = self.config.get("preprocessing.random_state", 42)

        if stratify is None:
            stratify = self.config.get("preprocessing.stratify", True)

        logger.info(f"Splitting data: test_size={test_size}, random_state={random_state}")

        try:
            if self.target_column not in df.columns:
                raise PreprocessingError(f"Target column '{self.target_column}' not found")

            X = df.drop(columns=[self.target_column])
            y = df[self.target_column]

            stratify_arg = y if stratify else None

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_arg
            )

            logger.info(f"Train set: {len(X_train)} samples")
            logger.info(f"Test set: {len(X_test)} samples")

            return X_train, X_test, y_train, y_test

        except Exception as e:
            raise PreprocessingError(f"Failed to split data: {str(e)}")

    def preprocess_pipeline(
        self,
        df: pd.DataFrame,
        handle_missing: bool = True,
        remove_outliers: bool = False,
        drop_features: bool = True,
        encode_categorical: bool = True
    ) -> pd.DataFrame:
        """
        Run complete preprocessing pipeline.

        Args:
            df: Input DataFrame
            handle_missing: Whether to handle missing values
            remove_outliers: Whether to remove outliers
            drop_features: Whether to drop configured features
            encode_categorical: Whether to encode categorical features

        Returns:
            Preprocessed DataFrame

        Raises:
            PreprocessingError: If preprocessing fails
        """
        logger.info("Starting preprocessing pipeline")

        df_processed = df.copy()

        try:
            # 1. Drop unwanted features
            if drop_features:
                df_processed = self.drop_features(df_processed)

            # 2. Handle missing values
            if handle_missing:
                df_processed = self.handle_missing_values(df_processed)

            # 3. Remove outliers
            if remove_outliers:
                outliers_enabled = self.config.get("preprocessing.handle_outliers", False)
                if outliers_enabled:
                    df_processed = self.remove_outliers(df_processed)

            # 4. Encode categorical features
            if encode_categorical:
                df_processed = self.encode_categorical_features(df_processed)

            logger.info("Preprocessing pipeline completed")
            logger.info(f"Final shape: {df_processed.shape}")

            return df_processed

        except Exception as e:
            raise PreprocessingError(f"Preprocessing pipeline failed: {str(e)}")
