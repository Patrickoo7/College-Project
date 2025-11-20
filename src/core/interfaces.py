"""Abstract base classes defining interfaces for core components.

These interfaces promote loose coupling and enable dependency injection.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


class IDataLoader(ABC):
    """Interface for data loading components."""

    @abstractmethod
    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Load a single dataset.

        Args:
            dataset_name: Name of the dataset to load

        Returns:
            Loaded dataset as DataFrame
        """
        pass

    @abstractmethod
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all configured datasets.

        Returns:
            Dictionary mapping dataset names to DataFrames
        """
        pass

    @abstractmethod
    def combine_datasets(self, dataset_names: List[str]) -> pd.DataFrame:
        """Combine multiple datasets into one.

        Args:
            dataset_names: List of dataset names to combine

        Returns:
            Combined DataFrame
        """
        pass


class IDataValidator(ABC):
    """Interface for data validation components."""

    @abstractmethod
    def validate_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate dataset schema.

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        pass

    @abstractmethod
    def validate_ranges(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate value ranges.

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        pass

    @abstractmethod
    def validate_all(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run all validation checks.

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary containing validation results
        """
        pass


class IDataPreprocessor(ABC):
    """Interface for data preprocessing components."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "IDataPreprocessor":
        """Fit the preprocessor on training data.

        Args:
            X: Features DataFrame
            y: Optional target Series

        Returns:
            self
        """
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted preprocessor.

        Args:
            X: Features DataFrame to transform

        Returns:
            Transformed DataFrame
        """
        pass

    @abstractmethod
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step.

        Args:
            X: Features DataFrame
            y: Optional target Series

        Returns:
            Transformed DataFrame
        """
        pass


class IFeatureEngineer(ABC):
    """Interface for feature engineering components."""

    @abstractmethod
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with engineered features
        """
        pass

    @abstractmethod
    def select_features(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, List[str]]:
        """Select best features.

        Args:
            X: Features DataFrame
            y: Target Series

        Returns:
            Tuple of (selected_features_df, selected_feature_names)
        """
        pass


class IModelTrainer(ABC):
    """Interface for model training components."""

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> Dict[str, Any]:
        """Train model(s).

        Args:
            X_train: Training features
            y_train: Training target
            **kwargs: Additional training parameters

        Returns:
            Dictionary containing trained models and metadata
        """
        pass

    @abstractmethod
    def evaluate(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate a model.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target

        Returns:
            Dictionary of evaluation metrics
        """
        pass

    @abstractmethod
    def save_model(self, model: Any, path: str, metadata: Optional[Dict] = None) -> None:
        """Save a trained model.

        Args:
            model: Model to save
            path: Save path
            metadata: Optional metadata to save with model
        """
        pass

    @abstractmethod
    def load_model(self, path: str) -> Tuple[Any, Optional[Dict]]:
        """Load a saved model.

        Args:
            path: Path to model file

        Returns:
            Tuple of (model, metadata)
        """
        pass


class IPredictor(ABC):
    """Interface for prediction components."""

    @abstractmethod
    def predict(self, X: pd.DataFrame, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Make predictions.

        Args:
            X: Features DataFrame
            model_name: Optional specific model to use

        Returns:
            Dictionary containing predictions and probabilities
        """
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame, model_name: Optional[str] = None) -> np.ndarray:
        """Get prediction probabilities.

        Args:
            X: Features DataFrame
            model_name: Optional specific model to use

        Returns:
            Array of prediction probabilities
        """
        pass

    @abstractmethod
    def explain_prediction(self, X: pd.DataFrame, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Explain predictions using SHAP/LIME.

        Args:
            X: Features DataFrame
            model_name: Optional specific model to use

        Returns:
            Dictionary containing explanation values
        """
        pass


class IDriftDetector(ABC):
    """Interface for drift detection components."""

    @abstractmethod
    def detect_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Detect drift between reference and current data.

        Args:
            reference_data: Reference (baseline) data
            current_data: Current data to check for drift
            feature_names: List of features to check

        Returns:
            Dictionary containing drift detection results
        """
        pass

    @abstractmethod
    def calculate_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        bins: int = 10
    ) -> float:
        """Calculate Population Stability Index.

        Args:
            reference: Reference data array
            current: Current data array
            bins: Number of bins for PSI calculation

        Returns:
            PSI value
        """
        pass


class IMonitor(ABC):
    """Interface for model monitoring components."""

    @abstractmethod
    def log_prediction(
        self,
        model_name: str,
        features: Dict[str, Any],
        prediction: Any,
        probability: Optional[float] = None,
        latency_ms: Optional[float] = None
    ) -> None:
        """Log a prediction.

        Args:
            model_name: Name of the model used
            features: Input features
            prediction: Prediction result
            probability: Optional prediction probability
            latency_ms: Optional prediction latency
        """
        pass

    @abstractmethod
    def check_model_performance(self, model_name: str) -> Dict[str, Any]:
        """Check current model performance.

        Args:
            model_name: Name of the model to check

        Returns:
            Dictionary containing performance metrics
        """
        pass

    @abstractmethod
    def create_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """Create a monitoring alert.

        Args:
            alert_type: Type of alert (performance, drift, error, etc.)
            severity: Alert severity (info, warning, critical)
            message: Alert message
            metadata: Optional additional metadata
        """
        pass


class IRetrainer(ABC):
    """Interface for automated retraining components."""

    @abstractmethod
    def should_retrain(self, current_performance: Dict[str, float]) -> Tuple[bool, str]:
        """Determine if model should be retrained.

        Args:
            current_performance: Current model performance metrics

        Returns:
            Tuple of (should_retrain, reason)
        """
        pass

    @abstractmethod
    def retrain(
        self,
        trigger_reason: str,
        models_to_retrain: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Execute model retraining.

        Args:
            trigger_reason: Reason for retraining
            models_to_retrain: Optional list of specific models to retrain

        Returns:
            Dictionary containing retraining results
        """
        pass

    @abstractmethod
    def promote_model(self, model_name: str, new_version: str) -> bool:
        """Promote a new model version to production.

        Args:
            model_name: Name of the model
            new_version: Version to promote

        Returns:
            True if promotion successful, False otherwise
        """
        pass
