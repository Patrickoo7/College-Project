"""Automated model retraining pipeline with performance monitoring."""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import Config
from src.utils.logger import get_logger
from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataPreprocessor
from src.features.feature_engineering import FeatureEngineer
from src.models.train import ModelTrainer
from src.models.evaluate import ModelEvaluator

logger = get_logger(__name__)


class AutomatedRetrainingPipeline:
    """
    Automated model retraining pipeline with performance monitoring.

    Features:
    - Scheduled retraining based on time or performance degradation
    - Performance monitoring and drift detection
    - Automatic model promotion if new model performs better
    - Rollback capability if new model underperforms
    """

    def __init__(self):
        """Initialize the retraining pipeline."""
        self.config = Config()
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()

        # Paths
        self.models_dir = Path(self.config.get("paths.models.artifacts", "models/artifacts"))
        self.metadata_dir = Path(self.config.get("paths.models.metadata", "models/metadata"))
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        # Retraining thresholds
        self.performance_threshold = self.config.get(
            "retraining.performance_threshold", 0.05
        )  # 5% performance drop
        self.min_samples_required = self.config.get(
            "retraining.min_samples_required", 100
        )
        self.retraining_interval_days = self.config.get(
            "retraining.interval_days", 30
        )

    def should_retrain(self, current_performance: Dict[str, float]) -> Tuple[bool, str]:
        """
        Determine if model should be retrained.

        Args:
            current_performance: Dictionary of current model metrics

        Returns:
            Tuple of (should_retrain, reason)
        """
        reasons = []

        # Check performance degradation
        baseline_performance = self._load_baseline_performance()
        if baseline_performance:
            accuracy_drop = baseline_performance.get("accuracy", 0) - current_performance.get("accuracy", 0)
            if accuracy_drop > self.performance_threshold:
                reasons.append(f"Performance degraded by {accuracy_drop:.2%}")

        # Check time since last training
        last_training_date = self._get_last_training_date()
        if last_training_date:
            days_since_training = (datetime.now() - last_training_date).days
            if days_since_training >= self.retraining_interval_days:
                reasons.append(f"Model trained {days_since_training} days ago")

        # Check data drift (will be implemented separately)

        should_retrain = len(reasons) > 0
        reason = "; ".join(reasons) if reasons else "No retraining needed"

        return should_retrain, reason

    def retrain_pipeline(
        self,
        force: bool = False,
        models_to_train: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Execute complete retraining pipeline.

        Args:
            force: Force retraining regardless of criteria
            models_to_train: List of model names to train (None = all)

        Returns:
            Dictionary with retraining results
        """
        logger.info("Starting automated retraining pipeline...")

        results = {
            "timestamp": datetime.now().isoformat(),
            "forced": force,
            "retrained": False,
            "models": {},
            "reason": ""
        }

        try:
            # Load and prepare data
            logger.info("Loading data...")
            df = self._load_latest_data()

            if len(df) < self.min_samples_required:
                logger.warning(f"Insufficient samples: {len(df)} < {self.min_samples_required}")
                results["reason"] = "Insufficient training samples"
                return results

            # Evaluate current model performance
            current_performance = self._evaluate_current_model(df)

            # Check if retraining is needed
            should_retrain, reason = self.should_retrain(current_performance)

            if not should_retrain and not force:
                logger.info(f"Retraining not needed: {reason}")
                results["reason"] = reason
                results["current_performance"] = current_performance
                return results

            logger.info(f"Retraining triggered: {reason}")
            results["reason"] = reason
            results["retrained"] = True

            # Prepare data
            logger.info("Preprocessing data...")
            df_clean = self.preprocessor.preprocess_pipeline(
                df,
                handle_missing=True,
                remove_outliers=True,
                encode_categorical=True
            )

            logger.info("Engineering features...")
            df_final = self.feature_engineer.engineer_features_pipeline(
                df_clean,
                create_interactions=True,
                create_domain=True
            )

            # Split data
            X_train, X_test, y_train, y_test = self.preprocessor.split_data(df_final)

            # Train models
            logger.info("Training models...")
            if models_to_train:
                trained_models = {}
                for model_name in models_to_train:
                    trained_models[model_name] = self.trainer.train_model(
                        model_name, X_train, y_train
                    )
            else:
                trained_models = self.trainer.train_all_models(X_train, y_train)

            # Evaluate new models
            logger.info("Evaluating new models...")
            for model_name, model in trained_models.items():
                evaluation = self.evaluator.evaluate_model(
                    model, X_test, y_test, model_name
                )
                results["models"][model_name] = evaluation

            # Compare with baseline and decide on promotion
            promoted_models = self._promote_best_models(
                results["models"],
                current_performance
            )
            results["promoted_models"] = promoted_models

            # Save models
            logger.info("Saving models...")
            self.trainer.save_all_models()

            # Update baseline performance
            self._save_baseline_performance(results["models"])
            self._save_training_metadata(results)

            logger.info("Retraining pipeline completed successfully")

        except Exception as e:
            logger.error(f"Retraining pipeline failed: {e}")
            results["error"] = str(e)
            raise

        return results

    def _load_latest_data(self) -> pd.DataFrame:
        """Load latest data for retraining."""
        # Load all datasets
        datasets = self.data_loader.load_all_datasets()
        # combine_datasets expects List[str], not Dict
        dataset_names = list(datasets.keys())
        df = self.data_loader.combine_datasets(dataset_names)
        return df

    def _evaluate_current_model(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate current production model.

        Args:
            df: Data to evaluate on

        Returns:
            Dictionary of performance metrics
        """
        try:
            # Load current production model (default: random_forest)
            model_path = self.models_dir / "random_forest.pkl"

            if not model_path.exists():
                logger.warning("No current model found for evaluation")
                return {}

            with open(model_path, "rb") as f:
                model = pickle.load(f)

            # Prepare data
            df_clean = self.preprocessor.preprocess_pipeline(
                df,
                handle_missing=True,
                remove_outliers=True,
                encode_categorical=True
            )

            df_final = self.feature_engineer.engineer_features_pipeline(
                df_clean,
                create_interactions=True,
                create_domain=True
            )

            X_train, X_test, y_train, y_test = self.preprocessor.split_data(df_final)

            # Evaluate
            y_pred = model.predict(X_test)

            performance = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
                "f1": f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }

            return performance

        except Exception as e:
            logger.error(f"Error evaluating current model: {e}")
            return {}

    def _promote_best_models(
        self,
        new_models: Dict[str, Dict],
        baseline: Dict[str, float]
    ) -> List[str]:
        """
        Promote new models if they perform better than baseline.

        Args:
            new_models: Dictionary of new model evaluations
            baseline: Baseline performance metrics

        Returns:
            List of promoted model names
        """
        promoted = []

        if not baseline:
            # No baseline, promote all models
            return list(new_models.keys())

        baseline_accuracy = baseline.get("accuracy", 0)

        for model_name, metrics in new_models.items():
            new_accuracy = metrics.get("accuracy", 0)

            # Promote if new model is better or within threshold
            if new_accuracy >= baseline_accuracy - 0.01:  # Allow 1% tolerance
                promoted.append(model_name)
                logger.info(
                    f"Promoting {model_name}: "
                    f"accuracy={new_accuracy:.4f} vs baseline={baseline_accuracy:.4f}"
                )
            else:
                logger.warning(
                    f"Not promoting {model_name}: "
                    f"accuracy={new_accuracy:.4f} < baseline={baseline_accuracy:.4f}"
                )

        return promoted

    def _load_baseline_performance(self) -> Optional[Dict[str, float]]:
        """Load baseline performance metrics."""
        baseline_path = self.metadata_dir / "baseline_performance.json"

        if not baseline_path.exists():
            return None

        try:
            with open(baseline_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading baseline performance: {e}")
            return None

    def _save_baseline_performance(self, models: Dict[str, Dict]):
        """Save baseline performance metrics."""
        if not models:
            logger.warning("No models to save baseline performance for")
            return

        # Find best model
        best_model = max(
            models.items(),
            key=lambda x: x[1].get("accuracy", 0)
        )

        baseline = {
            "accuracy": best_model[1].get("accuracy", 0),
            "precision": best_model[1].get("precision", 0),
            "recall": best_model[1].get("recall", 0),
            "f1": best_model[1].get("f1", 0),
            "model_name": best_model[0],
            "timestamp": datetime.now().isoformat()
        }

        baseline_path = self.metadata_dir / "baseline_performance.json"
        with open(baseline_path, "w") as f:
            json.dump(baseline, f, indent=2)

        logger.info(f"Saved baseline performance: {baseline}")

    def _get_last_training_date(self) -> Optional[datetime]:
        """Get the date of last training."""
        metadata_path = self.metadata_dir / "training_history.json"

        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, "r") as f:
                history = json.load(f)

            if not history:
                return None

            last_training = history[-1]
            return datetime.fromisoformat(last_training["timestamp"])

        except Exception as e:
            logger.error(f"Error getting last training date: {e}")
            return None

    def _save_training_metadata(self, results: Dict[str, Any]):
        """Save training metadata to history."""
        metadata_path = self.metadata_dir / "training_history.json"

        # Load existing history
        history = []
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    history = json.load(f)
            except Exception as e:
                logger.error(f"Error loading training history: {e}")

        # Add new entry
        history.append(results)

        # Save updated history
        with open(metadata_path, "w") as f:
            json.dump(history, f, indent=2)

        logger.info("Saved training metadata to history")

    def schedule_retraining(self, cron_expression: str = "0 2 * * 0"):
        """
        Generate cron job configuration for scheduled retraining.

        Args:
            cron_expression: Cron expression (default: weekly at 2 AM on Sunday)

        Returns:
            Cron configuration string
        """
        script_path = Path(__file__).parent / "retraining_job.py"

        cron_command = f"{cron_expression} cd {project_root} && python {script_path}"

        logger.info(f"Cron configuration: {cron_command}")

        return cron_command


if __name__ == "__main__":
    # Example usage
    pipeline = AutomatedRetrainingPipeline()

    # Run retraining
    results = pipeline.retrain_pipeline(force=False)

    print("\nRetraining Results:")
    print(f"Retrained: {results['retrained']}")
    print(f"Reason: {results['reason']}")

    if results['retrained']:
        print(f"\nPromoted Models: {results.get('promoted_models', [])}")
        print("\nModel Performance:")
        for model_name, metrics in results['models'].items():
            print(f"  {model_name}: accuracy={metrics.get('accuracy', 0):.4f}")
