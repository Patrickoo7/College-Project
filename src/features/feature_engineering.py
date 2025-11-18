"""Feature engineering module for creating and selecting features."""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    mutual_info_classif,
    RFE
)
from sklearn.ensemble import RandomForestClassifier

from ..utils.config import get_config
from ..utils.exceptions import FeatureEngineeringError
from ..utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """Create and engineer features for heart disease prediction."""

    def __init__(self):
        """Initialize FeatureEngineer."""
        self.config = get_config()
        self.target_column = self.config.get("features.target", "target")
        self.feature_selector = None
        self.selected_features = None

    def create_interaction_features(
        self,
        df: pd.DataFrame,
        feature_pairs: Optional[List[Tuple[str, str]]] = None
    ) -> pd.DataFrame:
        """
        Create interaction features between specified pairs.

        Args:
            df: Input DataFrame
            feature_pairs: List of (feature1, feature2) tuples.
                          If None, creates common interactions

        Returns:
            DataFrame with added interaction features
        """
        df_new = df.copy()

        if feature_pairs is None:
            # Define common meaningful interactions
            feature_pairs = [
                ("age", "chol"),
                ("age", "thalach"),
                ("age", "trestbps"),
                ("chol", "trestbps"),
            ]

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
        degree: int = 2
    ) -> pd.DataFrame:
        """
        Create polynomial features for specified columns.

        Args:
            df: Input DataFrame
            features: List of features to create polynomials for
                     If None, uses continuous features
            degree: Polynomial degree (typically 2 or 3)

        Returns:
            DataFrame with added polynomial features
        """
        df_new = df.copy()

        if features is None:
            features = self.config.get("features.continuous", [])
            features = [f for f in features if f in df.columns]

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
        bins_config: Optional[dict] = None
    ) -> pd.DataFrame:
        """
        Create binned (discretized) features from continuous variables.

        Args:
            df: Input DataFrame
            bins_config: Dictionary mapping feature names to bin edges
                        If None, uses default binning

        Returns:
            DataFrame with added binned features
        """
        df_new = df.copy()

        if bins_config is None:
            # Default binning for age groups and other features
            bins_config = {
                "age": {
                    "bins": [0, 40, 50, 60, 70, 120],
                    "labels": ["<40", "40-50", "50-60", "60-70", "70+"]
                },
                "chol": {
                    "bins": [0, 200, 240, 280, 600],
                    "labels": ["normal", "borderline", "high", "very_high"]
                },
                "trestbps": {
                    "bins": [0, 120, 140, 180, 300],
                    "labels": ["normal", "elevated", "high", "very_high"]
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
        """
        Create domain-specific features based on medical knowledge.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with added domain features
        """
        df_new = df.copy()

        logger.info("Creating domain-specific features")

        try:
            # Age-related risk (older = higher risk)
            if "age" in df.columns:
                df_new["age_risk"] = (df["age"] > 55).astype(int)

            # High cholesterol indicator
            if "chol" in df.columns:
                df_new["high_chol"] = (df["chol"] > 240).astype(int)

            # High blood pressure indicator
            if "trestbps" in df.columns:
                df_new["high_bp"] = (df["trestbps"] > 140).astype(int)

            # Low max heart rate (concern for older patients)
            if "thalach" in df.columns and "age" in df.columns:
                max_hr_predicted = 220 - df["age"]
                df_new["hr_percentage"] = (df["thalach"] / max_hr_predicted) * 100
                df_new["low_hr"] = (df_new["hr_percentage"] < 80).astype(int)

            # Combined risk factors
            risk_columns = [col for col in ["age_risk", "high_chol", "high_bp"] if col in df_new.columns]
            if risk_columns:
                df_new["total_risk_factors"] = df_new[risk_columns].sum(axis=1)

            # BMI proxy (if height/weight available - typically not in these datasets)
            # Cholesterol to age ratio
            if "chol" in df.columns and "age" in df.columns:
                df_new["chol_age_ratio"] = df["chol"] / df["age"]

            # Blood pressure pulse pressure proxy
            # (systolic - diastolic, but we only have systolic)
            # Using age as proxy for diastolic risk
            if "trestbps" in df.columns and "age" in df.columns:
                df_new["bp_age_interaction"] = df["trestbps"] * (df["age"] / 100)

            logger.info(f"Created {len(df_new.columns) - len(df.columns)} domain features")
            logger.info(f"Total features: {len(df_new.columns)}")

            return df_new

        except Exception as e:
            raise FeatureEngineeringError(f"Failed to create domain features: {str(e)}")

    def select_features_univariate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        k: int = 10,
        score_func: str = "f_classif"
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select top k features using univariate statistical tests.

        Args:
            X: Feature DataFrame
            y: Target Series
            k: Number of features to select
            score_func: Scoring function ('f_classif', 'mutual_info')

        Returns:
            Tuple of (selected features DataFrame, list of feature names)
        """
        logger.info(f"Selecting top {k} features using {score_func}")

        try:
            if score_func == "f_classif":
                selector = SelectKBest(score_func=f_classif, k=k)
            elif score_func == "mutual_info":
                selector = SelectKBest(score_func=mutual_info_classif, k=k)
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
        n_features: int = 10
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select features using Recursive Feature Elimination.

        Args:
            X: Feature DataFrame
            y: Target Series
            n_features: Number of features to select

        Returns:
            Tuple of (selected features DataFrame, list of feature names)
        """
        logger.info(f"Selecting {n_features} features using RFE")

        try:
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            selector = RFE(estimator, n_features_to_select=n_features)

            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()

            self.feature_selector = selector
            self.selected_features = selected_features

            logger.info(f"Selected features: {selected_features}")

            return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features

        except Exception as e:
            raise FeatureEngineeringError(f"Failed to perform RFE: {str(e)}")

    def get_feature_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = "random_forest"
    ) -> pd.DataFrame:
        """
        Calculate feature importance scores.

        Args:
            X: Feature DataFrame
            y: Target Series
            method: Method to calculate importance ('random_forest')

        Returns:
            DataFrame with features and their importance scores
        """
        logger.info(f"Calculating feature importance using {method}")

        try:
            if method == "random_forest":
                model = RandomForestClassifier(n_estimators=200, random_state=42)
                model.fit(X, y)
                importances = model.feature_importances_

                importance_df = pd.DataFrame({
                    "feature": X.columns,
                    "importance": importances
                }).sort_values("importance", ascending=False)

                logger.info("Top 10 important features:")
                logger.info(f"\n{importance_df.head(10)}")

                return importance_df

            else:
                raise FeatureEngineeringError(f"Unknown importance method: {method}")

        except Exception as e:
            raise FeatureEngineeringError(f"Failed to calculate feature importance: {str(e)}")

    def engineer_features_pipeline(
        self,
        df: pd.DataFrame,
        create_interactions: bool = True,
        create_polynomials: bool = False,
        create_bins: bool = False,
        create_domain: bool = True
    ) -> pd.DataFrame:
        """
        Run complete feature engineering pipeline.

        Args:
            df: Input DataFrame
            create_interactions: Whether to create interaction features
            create_polynomials: Whether to create polynomial features
            create_bins: Whether to create binned features
            create_domain: Whether to create domain-specific features

        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering pipeline")

        df_engineered = df.copy()

        try:
            # Create domain features first (most important)
            if create_domain:
                df_engineered = self.create_domain_features(df_engineered)

            # Create interaction features
            if create_interactions:
                df_engineered = self.create_interaction_features(df_engineered)

            # Create polynomial features (be careful, can explode feature count)
            if create_polynomials:
                df_engineered = self.create_polynomial_features(df_engineered, degree=2)

            # Create binned features
            if create_bins:
                df_engineered = self.create_binned_features(df_engineered)

            logger.info("Feature engineering pipeline completed")
            logger.info(f"Original features: {len(df.columns)}")
            logger.info(f"Engineered features: {len(df_engineered.columns)}")

            return df_engineered

        except Exception as e:
            raise FeatureEngineeringError(f"Feature engineering pipeline failed: {str(e)}")
