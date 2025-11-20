"""Automated hyperparameter tuning using Optuna."""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HyperparameterTuner:
    """
    Automated hyperparameter tuning using Optuna.

    Features:
    - Bayesian optimization
    - Automatic pruning of unpromising trials
    - Parallel optimization support
    - Study persistence and resume capability
    """

    def __init__(
        self,
        n_trials: int = 100,
        cv_folds: int = 5,
        random_state: int = 42
    ):
        """
        Initialize hyperparameter tuner.

        Args:
            n_trials: Number of optimization trials
            cv_folds: Number of cross-validation folds
            random_state: Random state for reproducibility
        """
        self.config = Config()
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state

        # Paths
        self.metadata_dir = Path(self.config.get("paths.models.metadata", "models/metadata"))
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        # Optuna settings
        self.sampler = TPESampler(seed=random_state)
        self.pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)

    def tune_model(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        metric: str = "accuracy"
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters for a specific model.

        Args:
            model_name: Name of the model to tune
            X: Feature matrix
            y: Target variable
            metric: Optimization metric (accuracy, f1, roc_auc)

        Returns:
            Dictionary with best parameters and study results
        """
        logger.info(f"Starting hyperparameter tuning for {model_name}...")

        # Create objective function
        objective = self._create_objective(model_name, X, y, metric)

        # Create study
        study_name = f"{model_name}_{metric}"
        storage = f"sqlite:///{self.metadata_dir}/optuna_studies.db"

        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=self.sampler,
            pruner=self.pruner,
            storage=storage,
            load_if_exists=True
        )

        # Optimize
        study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=True
        )

        # Get results
        results = {
            "model_name": model_name,
            "metric": metric,
            "best_params": study.best_params,
            "best_value": study.best_value,
            "n_trials": len(study.trials),
            "best_trial": study.best_trial.number
        }

        # Save results
        self._save_tuning_results(results)

        logger.info(
            f"Tuning completed. Best {metric}: {study.best_value:.4f} "
            f"(trial {study.best_trial.number})"
        )

        return results

    def _create_objective(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        metric: str
    ) -> Callable:
        """Create optimization objective function."""

        def objective(trial: optuna.Trial) -> float:
            """Objective function for Optuna."""

            # Get hyperparameters based on model type
            if model_name == "random_forest":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 30),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                    "max_features": trial.suggest_categorical(
                        "max_features", ["sqrt", "log2", None]
                    ),
                    "random_state": self.random_state
                }
                model = RandomForestClassifier(**params)

            elif model_name == "gradient_boosting":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "random_state": self.random_state
                }
                model = GradientBoostingClassifier(**params)

            elif model_name == "logistic_regression":
                params = {
                    "C": trial.suggest_float("C", 1e-3, 100, log=True),
                    "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
                    "solver": "liblinear",  # Supports both L1 and L2
                    "max_iter": 1000,
                    "random_state": self.random_state
                }
                model = LogisticRegression(**params)

            elif model_name == "svm":
                params = {
                    "C": trial.suggest_float("C", 1e-3, 100, log=True),
                    "kernel": trial.suggest_categorical("kernel", ["linear", "rbf", "poly"]),
                    "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
                    "random_state": self.random_state
                }
                model = SVC(**params)

            elif model_name == "knn":
                params = {
                    "n_neighbors": trial.suggest_int("n_neighbors", 3, 50),
                    "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                    "metric": trial.suggest_categorical(
                        "metric", ["euclidean", "manhattan", "minkowski"]
                    ),
                    "p": trial.suggest_int("p", 1, 5)
                }
                model = KNeighborsClassifier(**params)

            elif model_name == "decision_tree":
                params = {
                    "max_depth": trial.suggest_int("max_depth", 3, 30),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                    "max_features": trial.suggest_categorical(
                        "max_features", ["sqrt", "log2", None]
                    ),
                    "random_state": self.random_state
                }
                model = DecisionTreeClassifier(**params)

            else:
                raise ValueError(f"Unknown model: {model_name}")

            # Cross-validation
            try:
                scores = cross_val_score(
                    model, X, y,
                    cv=self.cv_folds,
                    scoring=metric,
                    n_jobs=-1
                )
                return scores.mean()
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return 0.0

        return objective

    def tune_all_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        models: Optional[list] = None,
        metric: str = "accuracy"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Tune hyperparameters for multiple models.

        Args:
            X: Feature matrix
            y: Target variable
            models: List of model names (None = all)
            metric: Optimization metric

        Returns:
            Dictionary with results for each model
        """
        if models is None:
            models = [
                "random_forest",
                "gradient_boosting",
                "logistic_regression",
                "svm",
                "knn",
                "decision_tree"
            ]

        results = {}

        for model_name in models:
            try:
                results[model_name] = self.tune_model(model_name, X, y, metric)
            except Exception as e:
                logger.error(f"Tuning failed for {model_name}: {e}")
                results[model_name] = {"error": str(e)}

        return results

    def _save_tuning_results(self, results: Dict[str, Any]):
        """Save tuning results."""
        results_path = self.metadata_dir / "tuning_results.json"

        # Load existing results
        all_results = {}
        if results_path.exists():
            try:
                with open(results_path, "r") as f:
                    all_results = json.load(f)
            except Exception as e:
                logger.error(f"Error loading tuning results: {e}")

        # Update with new results
        model_name = results["model_name"]
        if model_name not in all_results:
            all_results[model_name] = []

        all_results[model_name].append(results)

        # Save updated results
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

        logger.info(f"Saved tuning results to {results_path}")

    def get_best_params(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get best hyperparameters for a model."""
        results_path = self.metadata_dir / "tuning_results.json"

        if not results_path.exists():
            return None

        try:
            with open(results_path, "r") as f:
                all_results = json.load(f)

            if model_name not in all_results:
                return None

            # Get most recent results
            latest_result = all_results[model_name][-1]
            return latest_result.get("best_params")

        except Exception as e:
            logger.error(f"Error getting best params: {e}")
            return None

    def visualize_optimization(self, model_name: str, metric: str = "accuracy"):
        """
        Generate optimization visualization.

        Args:
            model_name: Name of the model
            metric: Metric used for optimization

        Returns:
            Path to saved visualization
        """
        try:
            import plotly.graph_objects as go
            from optuna.visualization import (
                plot_optimization_history,
                plot_param_importances,
                plot_slice
            )

            # Load study
            study_name = f"{model_name}_{metric}"
            storage = f"sqlite:///{self.metadata_dir}/optuna_studies.db"

            study = optuna.load_study(
                study_name=study_name,
                storage=storage
            )

            # Create visualizations
            vis_dir = self.metadata_dir / "visualizations"
            vis_dir.mkdir(exist_ok=True)

            # Optimization history
            fig = plot_optimization_history(study)
            fig.write_html(vis_dir / f"{model_name}_optimization_history.html")

            # Parameter importances
            fig = plot_param_importances(study)
            fig.write_html(vis_dir / f"{model_name}_param_importances.html")

            # Slice plot
            fig = plot_slice(study)
            fig.write_html(vis_dir / f"{model_name}_slice_plot.html")

            logger.info(f"Saved optimization visualizations to {vis_dir}")

            return str(vis_dir)

        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            return None


if __name__ == "__main__":
    # Example usage
    from src.data.data_loader import DataLoader
    from src.data.data_preprocessor import DataPreprocessor

    # Load and prepare data
    loader = DataLoader()
    datasets = loader.load_all_datasets()
    df = loader.combine_datasets(datasets)

    preprocessor = DataPreprocessor()
    df_clean = preprocessor.preprocess_pipeline(
        df,
        handle_missing=True,
        remove_outliers=True,
        encode_categorical=True
    )

    X_train, X_test, y_train, y_test = preprocessor.split_data(df_clean)

    # Tune hyperparameters
    tuner = HyperparameterTuner(n_trials=50)

    # Tune single model
    results = tuner.tune_model("random_forest", X_train, y_train)

    print("\nTuning Results:")
    print(f"Best accuracy: {results['best_value']:.4f}")
    print(f"Best parameters: {results['best_params']}")

    # Visualize optimization
    tuner.visualize_optimization("random_forest")
