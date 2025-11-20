"""Feature engineering module for creating and selecting features."""

from typing import List, Optional, Tuple, Any, Dict

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    mutual_info_classif,
    chi2,
    RFE
)
from sklearn.ensemble import RandomForestClassifier

from ..config import get_config
from ..config.schemas import FeatureEngineeringConfig
from ..core.interfaces import IFeatureEngineer
from ..utils.exceptions import FeatureEngineeringError
from ..utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer(IFeatureEngineer):
    """Create and engineer features for heart disease prediction.

    This implementation uses configuration for all feature engineering parameters,
    including interaction pairs, binning thresholds, and domain-specific rules.
    """

    def __init__(self, config: Optional[FeatureEngineeringConfig] = None):
        """Initialize FeatureEngineer.

        Args:
            config: Feature engineering configuration. If None, loads from global config.
        """
        if config is None:
            app_config = get_config()
            config = app_config.feature_engineering

        self.config = config
        self.feature_selector = None
        self.selected_features = None

        logger.debug(f"FeatureEngineer initialized with k_best={config.k_best_features}")

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all engineered features.

        This is the main entry point that creates all configured feature types.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with all engineered features

        Raises:
            FeatureEngineeringError: If feature creation fails
        """
        logger.info(f"Creating engineered features for {len(df)} samples")

        try:
            df_engineered = df.copy()

            # 1. Create interaction features
            df_engineered = self.create_interaction_features(df_engineered)

            # 2. Create polynomial features (if configured)
            if self.config.include_polynomial:
                df_engineered = self.create_polynomial_features(df_engineered)

            # 3. Create binned features
            df_engineered = self.create_binned_features(df_engineered)

            # 4. Create domain-specific features
            df_engineered = self.create_domain_features(df_engineered)

            logger.info(
                f"Feature engineering complete: {len(df.columns)} -> {len(df_engineered.columns)} features"
            )

            return df_engineered

        except Exception as e:
            raise FeatureEngineeringError(f"Failed to create features: {str(e)}")

    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Select best features using configured method.

        Args:
            X: Features DataFrame
            y: Target Series

        Returns:
            Tuple of (selected_features_df, selected_feature_names)

        Raises:
            FeatureEngineeringError: If feature selection fails
        """
        method = self.config.feature_selection_method
        k = self.config.k_best_features

        logger.info(f"Selecting features using method={method}, k={k}")

        if method in ["mutual_info", "f_classif", "chi2"]:
            return self.select_features_univariate(X, y, k=k, score_func=method)
        elif method == "rfe":
            return self.select_features_rfe(X, y, n_features=k)
        else:
            raise FeatureEngineeringError(f"Unknown feature selection method: {method}")

    def create_interaction_features(
        self,
        df: pd.DataFrame,
        feature_pairs: Optional[List[Tuple[str, str]]] = None
    ) -> pd.DataFrame:
        """Create interaction features between specified pairs.

        Args:
            df: Input DataFrame
            feature_pairs: List of (feature1, feature2) tuples.
                          If None, uses configured interaction pairs

        Returns:
            DataFrame with added interaction features
        """
        df_new = df.copy()

        if feature_pairs is None:
            feature_pairs = self.config.interaction_pairs

        logger.info(f"Creating {len(feature_pairs)} interaction features")

        try:
            for feat1, feat2 in feature_pairs:
                if feat1 in df.columns and feat2 in df.columns:
                    # Multiplicative interaction
                    interaction_name = f"{feat1}_x_{feat2}"
                    df_new[interaction_name] = df[feat1] * df[feat2]

                    logger.debug(f"Created interaction: {interaction_name}")

            logger.info(f"Total features after interactions: {len(df_new.columns)}")
            return df_new

        except Exception as e:
            raise FeatureEngineeringError(f"Failed to create interaction features: {str(e)}")

    def create_polynomial_features(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        degree: Optional[int] = None
    ) -> pd.DataFrame:
        """Create polynomial features for specified columns.

        Args:
            df: Input DataFrame
            features: List of features to create polynomials for.
                     If None, uses numeric columns
            degree: Polynomial degree. If None, uses config default

        Returns:
            DataFrame with added polynomial features
        """
        if degree is None:
            degree = self.config.polynomial_degree

        df_new = df.copy()

        if features is None:
            # Use all numeric columns
            features = df.select_dtypes(include=[np.number]).columns.tolist()

        logger.info(f"Creating polynomial features (degree={degree}) for {len(features)} features")

        try:
            for feat in features:
                if feat in df.columns:
                    for d in range(2, degree + 1):
                        poly_name = f"{feat}_pow{d}"
                        df_new[poly_name] = df[feat] ** d
                        logger.debug(f"Created polynomial: {poly_name}")

            logger.info(f"Total features after polynomials: {len(df_new.columns)}")
            return df_new

        except Exception as e:
            raise FeatureEngineeringError(f"Failed to create polynomial features: {str(e)}")

    def create_binned_features(
        self,
        df: pd.DataFrame,
        bins_config: Optional[Dict[str, Dict]] = None
    ) -> pd.DataFrame:
        """Create binned (discretized) features from continuous variables.

        Args:
            df: Input DataFrame
            bins_config: Dictionary mapping feature names to bin configurations.
                        If None, uses configured bins

        Returns:
            DataFrame with added binned features
        """
        df_new = df.copy()

        if bins_config is None:
            # Use configured bins
            bins_config = {
                "age": {
                    "bins": self.config.age_bins,
                    "labels": [f"age_{i}" for i in range(len(self.config.age_bins) - 1)]
                },
                "chol": {
                    "bins": self.config.chol_bins,
                    "labels": [f"chol_{i}" for i in range(len(self.config.chol_bins) - 1)]
                },
                "trestbps": {
                    "bins": self.config.bp_bins,
                    "labels": [f"bp_{i}" for i in range(len(self.config.bp_bins) - 1)]
                }
            }

        logger.info(f"Creating binned features for {len(bins_config)} features")

        try:
            for feat, config in bins_config.items():
                if feat in df.columns:
                    binned_name = f"{feat}_binned"
                    df_new[binned_name] = pd.cut(
                        df[feat],
                        bins=config["bins"],
                        labels=config["labels"],
                        include_lowest=True
                    )

                    # Convert to one-hot encoding
                    dummies = pd.get_dummies(df_new[binned_name], prefix=binned_name, drop_first=True)
                    df_new = pd.concat([df_new, dummies], axis=1)
                    df_new = df_new.drop(columns=[binned_name])

                    logger.debug(f"Created binned feature: {binned_name}")

            logger.info(f"Total features after binning: {len(df_new.columns)}")
            return df_new

        except Exception as e:
            raise FeatureEngineeringError(f"Failed to create binned features: {str(e)}")

    def create_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create domain-specific features based on medical knowledge.

        Uses configured thresholds from domain_rules.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with added domain features
        """
        df_new = df.copy()
        rules = self.config.domain_rules

        logger.info("Creating domain-specific features")

        try:
            # Age-related risk (configurable threshold)
            if "age" in df.columns:
                df_new["age_risk"] = (df["age"] > rules.age_risk_threshold).astype(int)

            # High cholesterol indicator (configurable threshold)
            if "chol" in df.columns:
                df_new["high_chol"] = (df["chol"] > rules.high_chol_threshold).astype(int)

            # High blood pressure indicator (configurable threshold)
            if "trestbps" in df.columns:
                df_new["high_bp"] = (df["trestbps"] > rules.high_bp_threshold).astype(int)

            # Low max heart rate (using configurable formula and threshold)
            if "thalach" in df.columns and "age" in df.columns:
                # Parse formula (e.g., "220-age")
                if rules.max_hr_formula == "220-age":
                    max_hr_predicted = 220 - df["age"]
                else:
                    # Fallback to standard formula
                    max_hr_predicted = 220 - df["age"]

                df_new["hr_percentage"] = (df["thalach"] / max_hr_predicted) * 100
                df_new["low_hr"] = (
                    df_new["hr_percentage"] < (rules.low_hr_percentage_threshold * 100)
                ).astype(int)

            # Combined risk factors
            risk_columns = [col for col in ["age_risk", "high_chol", "high_bp"] if col in df_new.columns]
            if risk_columns:
                df_new["total_risk_factors"] = df_new[risk_columns].sum(axis=1)

            # Cholesterol to age ratio
            if "chol" in df.columns and "age" in df.columns:
                df_new["chol_age_ratio"] = df["chol"] / df["age"]

            # Blood pressure-age interaction
            if "trestbps" in df.columns and "age" in df.columns:
                df_new["bp_age_interaction"] = df["trestbps"] * (df["age"] / 100)

            # Exercise-induced angina with oldpeak (ST depression)
            if "exang" in df.columns and "oldpeak" in df.columns:
                df_new["angina_severity"] = df["exang"] * df["oldpeak"]

            logger.info(f"Created {len(df_new.columns) - len(df.columns)} domain features")
            logger.info(f"Total features: {len(df_new.columns)}")

            return df_new

        except Exception as e:
            raise FeatureEngineeringError(f"Failed to create domain features: {str(e)}")

    def select_features_univariate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        k: Optional[int] = None,
        score_func: str = "mutual_info"
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Select top k features using univariate statistical tests.

        Args:
            X: Feature DataFrame
            y: Target Series
            k: Number of features to select. If None, uses config default
            score_func: Scoring function ('f_classif', 'mutual_info', 'chi2')

        Returns:
            Tuple of (selected features DataFrame, list of feature names)
        """
        if k is None:
            k = self.config.k_best_features

        # Ensure k doesn't exceed number of features
        k = min(k, X.shape[1])

        logger.info(f"Selecting top {k} features using {score_func}")

        try:
            if score_func == "f_classif":
                selector = SelectKBest(score_func=f_classif, k=k)
            elif score_func == "mutual_info":
                selector = SelectKBest(score_func=mutual_info_classif, k=k)
            elif score_func == "chi2":
                selector = SelectKBest(score_func=chi2, k=k)
            else:
                raise FeatureEngineeringError(f"Unknown score function: {score_func}")

            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()

            self.feature_selector = selector
            self.selected_features = selected_features

            logger.info(f"Selected features: {selected_features}")

            return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features

        except Exception as e:
            raise FeatureEngineeringError(f"Failed to select features: {str(e)}")

    def select_features_rfe(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: Optional[int] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Select features using Recursive Feature Elimination.

        Args:
            X: Feature DataFrame
            y: Target Series
            n_features: Number of features to select. If None, uses config default

        Returns:
            Tuple of (selected features DataFrame, list of feature names)
        """
        if n_features is None:
            n_features = self.config.k_best_features

        # Ensure n_features doesn't exceed number of features
        n_features = min(n_features, X.shape[1])

        logger.info(f"Selecting {n_features} features using RFE")

        try:
            # Get random_state from preprocessing config
            from ..config import get_config
            random_state = get_config().preprocessing.random_state

            estimator = RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
                n_jobs=-1
            )
            selector = RFE(estimator, n_features_to_select=n_features)

            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()

            self.feature_selector = selector
            self.selected_features = selected_features

            logger.info(f"Selected features: {selected_features}")

            return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features

        except Exception as e:
            raise FeatureEngineeringError(f"Failed to select features with RFE: {str(e)}")

    def get_feature_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> pd.DataFrame:
        """Get feature importance scores using Random Forest.

        Args:
            X: Feature DataFrame
            y: Target Series

        Returns:
            DataFrame with feature names and importance scores, sorted by importance
        """
        logger.info("Calculating feature importance")

        try:
            from ..config import get_config
            random_state = get_config().preprocessing.random_state

            rf = RandomForestClassifier(
                n_estimators=200,
                random_state=random_state,
                n_jobs=-1
            )
            rf.fit(X, y)

            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)

            logger.info(f"Top 5 features: {importance_df.head()['feature'].tolist()}")

            return importance_df

        except Exception as e:
            raise FeatureEngineeringError(f"Failed to calculate feature importance: {str(e)}")

    def transform_with_selected_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using previously selected features.

        Args:
            X: Feature DataFrame

        Returns:
            DataFrame with only selected features

        Raises:
            FeatureEngineeringError: If feature selection hasn't been performed
        """
        if self.selected_features is None:
            raise FeatureEngineeringError(
                "No features have been selected. Call select_features() first."
            )

        missing_features = set(self.selected_features) - set(X.columns)
        if missing_features:
            raise FeatureEngineeringError(
                f"Selected features not found in DataFrame: {missing_features}"
            )

        return X[self.selected_features]
