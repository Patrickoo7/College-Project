"""Model training module with MLflow integration."""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from ..utils.config import get_config
from ..utils.exceptions import ModelTrainingError
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """Train and manage machine learning models with MLflow tracking."""

    def __init__(self):
        """Initialize ModelTrainer."""
        self.config = get_config()
        self.models = {}
        self.trained_models = {}
        self.best_model = None
        self.best_score = 0.0

        # Setup MLflow
        self._setup_mlflow()

    def _setup_mlflow(self):
        """Setup MLflow experiment tracking."""
        experiment_name = self.config.get("mlflow.experiment_name", "heart-disease-prediction")
        tracking_uri = self.config.get("mlflow.tracking_uri", "mlruns")

        # Create tracking directory if needed
        tracking_path = Path(tracking_uri)
        tracking_path.mkdir(parents=True, exist_ok=True)

        mlflow.set_tracking_uri(str(tracking_path))

        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(f"Created MLflow experiment: {experiment_name} (ID: {experiment_id})")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing MLflow experiment: {experiment_name} (ID: {experiment_id})")

            mlflow.set_experiment(experiment_name)

        except Exception as e:
            logger.warning(f"Failed to setup MLflow: {str(e)}")

    def _get_model_instance(self, model_name: str) -> Any:
        """
        Get model instance based on configuration.

        Args:
            model_name: Name of the model

        Returns:
            Initialized model instance

        Raises:
            ModelTrainingError: If model cannot be created
        """
        if not self.config.is_model_enabled(model_name):
            raise ModelTrainingError(f"Model '{model_name}' is not enabled in configuration")

        params = self.config.get_model_params(model_name)

        try:
            if model_name == "logistic_regression":
                return LogisticRegression(**params)

            elif model_name == "random_forest":
                return RandomForestClassifier(**params)

            elif model_name == "knn":
                return KNeighborsClassifier(**params)

            elif model_name == "svm":
                return SVC(**params)

            elif model_name == "decision_tree":
                return DecisionTreeClassifier(**params)

            elif model_name == "xgboost":
                if not XGBOOST_AVAILABLE:
                    raise ModelTrainingError("XGBoost not installed")
                return xgb.XGBClassifier(**params)

            elif model_name == "lightgbm":
                if not LIGHTGBM_AVAILABLE:
                    raise ModelTrainingError("LightGBM not installed")
                return lgb.LGBMClassifier(**params)

            elif model_name == "catboost":
                if not CATBOOST_AVAILABLE:
                    raise ModelTrainingError("CatBoost not installed")
                return cb.CatBoostClassifier(**params)

            else:
                raise ModelTrainingError(f"Unknown model type: {model_name}")

        except Exception as e:
            raise ModelTrainingError(f"Failed to create model '{model_name}': {str(e)}")

    def train_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        log_mlflow: bool = True
    ) -> Any:
        """
        Train a single model.

        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            log_mlflow: Whether to log to MLflow

        Returns:
            Trained model

        Raises:
            ModelTrainingError: If training fails
        """
        logger.info(f"Training {model_name}...")

        try:
            model = self._get_model_instance(model_name)
            params = self.config.get_model_params(model_name)

            if log_mlflow:
                with mlflow.start_run(run_name=model_name):
                    # Log parameters
                    mlflow.log_params(params)
                    mlflow.log_param("model_type", model_name)
                    mlflow.log_param("n_train_samples", len(X_train))

                    # Train model
                    model.fit(X_train, y_train)

                    # Log model
                    mlflow.sklearn.log_model(model, model_name)

                    logger.info(f"Successfully trained and logged {model_name} to MLflow")
            else:
                model.fit(X_train, y_train)
                logger.info(f"Successfully trained {model_name}")

            self.trained_models[model_name] = model
            return model

        except Exception as e:
            raise ModelTrainingError(f"Failed to train {model_name}: {str(e)}")

    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Train all enabled models.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)

        Returns:
            Dictionary of trained models
        """
        logger.info("Training all enabled models...")

        enabled_models = self.config.get_enabled_models()
        logger.info(f"Enabled models: {enabled_models}")

        for model_name in enabled_models:
            try:
                self.train_model(
                    model_name,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    log_mlflow=True
                )
            except ModelTrainingError as e:
                logger.error(f"Failed to train {model_name}: {str(e)}")
                continue

        logger.info(f"Successfully trained {len(self.trained_models)} models")
        return self.trained_models

    def save_model(
        self,
        model: Any,
        model_name: str,
        save_dir: Optional[Path] = None
    ) -> Path:
        """
        Save trained model to disk.

        Args:
            model: Trained model
            model_name: Name of the model
            save_dir: Directory to save model. If None, uses config default

        Returns:
            Path to saved model

        Raises:
            ModelTrainingError: If saving fails
        """
        if save_dir is None:
            save_dir = self.config.get_path("paths.models.artifacts")

        save_dir.mkdir(parents=True, exist_ok=True)

        model_path = save_dir / f"{model_name}.pkl"

        try:
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            logger.info(f"Saved {model_name} to {model_path}")
            return model_path

        except Exception as e:
            raise ModelTrainingError(f"Failed to save model: {str(e)}")

    def save_all_models(self, save_dir: Optional[Path] = None) -> Dict[str, Path]:
        """
        Save all trained models.

        Args:
            save_dir: Directory to save models

        Returns:
            Dictionary mapping model names to save paths
        """
        logger.info("Saving all trained models...")

        saved_paths = {}
        for model_name, model in self.trained_models.items():
            try:
                path = self.save_model(model, model_name, save_dir)
                saved_paths[model_name] = path
            except ModelTrainingError as e:
                logger.error(f"Failed to save {model_name}: {str(e)}")

        logger.info(f"Saved {len(saved_paths)} models")
        return saved_paths

    def load_model(self, model_path: Path) -> Any:
        """
        Load a trained model from disk.

        Args:
            model_path: Path to the saved model

        Returns:
            Loaded model

        Raises:
            ModelTrainingError: If loading fails
        """
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)

            logger.info(f"Loaded model from {model_path}")
            return model

        except Exception as e:
            raise ModelTrainingError(f"Failed to load model: {str(e)}")

    def create_ensemble(
        self,
        ensemble_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Any:
        """
        Create and train ensemble model.

        Args:
            ensemble_type: Type of ensemble ('voting' or 'stacking')
            X_train: Training features
            y_train: Training targets

        Returns:
            Trained ensemble model

        Raises:
            ModelTrainingError: If ensemble creation fails
        """
        logger.info(f"Creating {ensemble_type} ensemble...")

        try:
            if ensemble_type == "voting":
                ensemble_config = self.config.model_config["models"]["voting_classifier"]
                estimator_names = ensemble_config["estimators"]

                estimators = [
                    (name, self._get_model_instance(name))
                    for name in estimator_names
                    if self.config.is_model_enabled(name)
                ]

                ensemble = VotingClassifier(
                    estimators=estimators,
                    voting=ensemble_config["voting"],
                    weights=ensemble_config.get("weights")
                )

            elif ensemble_type == "stacking":
                ensemble_config = self.config.model_config["models"]["stacking_classifier"]
                estimator_names = ensemble_config["estimators"]

                estimators = [
                    (name, self._get_model_instance(name))
                    for name in estimator_names
                    if self.config.is_model_enabled(name)
                ]

                final_estimator_name = ensemble_config["final_estimator"]
                final_estimator = self._get_model_instance(final_estimator_name)

                ensemble = StackingClassifier(
                    estimators=estimators,
                    final_estimator=final_estimator,
                    cv=ensemble_config.get("cv", 5)
                )

            else:
                raise ModelTrainingError(f"Unknown ensemble type: {ensemble_type}")

            # Train ensemble
            with mlflow.start_run(run_name=f"{ensemble_type}_ensemble"):
                mlflow.log_param("model_type", f"{ensemble_type}_ensemble")
                mlflow.log_param("n_train_samples", len(X_train))

                ensemble.fit(X_train, y_train)

                mlflow.sklearn.log_model(ensemble, f"{ensemble_type}_ensemble")

            logger.info(f"Successfully trained {ensemble_type} ensemble")
            self.trained_models[f"{ensemble_type}_ensemble"] = ensemble

            return ensemble

        except Exception as e:
            raise ModelTrainingError(f"Failed to create ensemble: {str(e)}")

    def get_model_summary(self) -> pd.DataFrame:
        """
        Get summary of all trained models.

        Returns:
            DataFrame with model information
        """
        summary_data = []

        for model_name, model in self.trained_models.items():
            summary_data.append({
                "model_name": model_name,
                "model_type": type(model).__name__,
                "n_features": getattr(model, "n_features_in_", "N/A"),
            })

        return pd.DataFrame(summary_data)
