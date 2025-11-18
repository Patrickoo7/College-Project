"""Model evaluation module with comprehensive metrics and visualization."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import cross_val_score, cross_validate

from ..utils.config import get_config
from ..utils.exceptions import EvaluationError
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """Evaluate machine learning models with comprehensive metrics."""

    def __init__(self):
        """Initialize ModelEvaluator."""
        self.config = get_config()
        self.evaluation_results = {}

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate all evaluation metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)

        Returns:
            Dictionary of metric names and values
        """
        logger.info("Calculating evaluation metrics...")

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="binary", zero_division=0),
        }

        # Add probability-based metrics if available
        if y_prob is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
                metrics["pr_auc"] = average_precision_score(y_true, y_prob)
            except Exception as e:
                logger.warning(f"Could not calculate probability-based metrics: {str(e)}")

        # Calculate specificity and sensitivity (clinical metrics)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0  # Same as recall

        # Positive and Negative Predictive Values
        metrics["ppv"] = tp / (tp + fp) if (tp + fp) > 0 else 0  # Same as precision
        metrics["npv"] = tn / (tn + fn) if (tn + fn) > 0 else 0

        logger.info("Metrics calculated successfully")
        return metrics

    def evaluate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "model"
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a single model.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model

        Returns:
            Dictionary containing all evaluation results
        """
        logger.info(f"Evaluating {model_name}...")

        try:
            # Get predictions
            y_pred = model.predict(X_test)

            # Get probabilities if available
            y_prob = None
            if hasattr(model, "predict_proba"):
                y_prob_full = model.predict_proba(X_test)
                y_prob = y_prob_full[:, 1]  # Probability of positive class
            elif hasattr(model, "decision_function"):
                y_prob = model.decision_function(X_test)

            # Calculate metrics
            metrics = self.calculate_metrics(y_test, y_pred, y_prob)

            # Get confusion matrix
            cm = confusion_matrix(y_test, y_pred)

            # Get classification report
            report = classification_report(y_test, y_pred, output_dict=True)

            # Store results
            results = {
                "model_name": model_name,
                "metrics": metrics,
                "confusion_matrix": cm.tolist(),
                "classification_report": report,
                "predictions": y_pred.tolist(),
            }

            if y_prob is not None:
                results["probabilities"] = y_prob.tolist()

            self.evaluation_results[model_name] = results

            # Log to MLflow if available
            self._log_to_mlflow(model_name, metrics, cm)

            logger.info(f"{model_name} evaluation complete")
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, ROC-AUC: {metrics.get('roc_auc', 'N/A')}")

            return results

        except Exception as e:
            raise EvaluationError(f"Failed to evaluate {model_name}: {str(e)}")

    def _log_to_mlflow(
        self,
        model_name: str,
        metrics: Dict[str, float],
        confusion_matrix: np.ndarray
    ):
        """
        Log evaluation results to MLflow.

        Args:
            model_name: Name of the model
            metrics: Dictionary of metrics
            confusion_matrix: Confusion matrix
        """
        try:
            # Find the latest run for this model
            experiment = mlflow.get_experiment_by_name(
                self.config.get("mlflow.experiment_name", "heart-disease-prediction")
            )

            if experiment:
                runs = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    filter_string=f"tags.mlflow.runName = '{model_name}'"
                )

                if not runs.empty:
                    run_id = runs.iloc[0]["run_id"]

                    with mlflow.start_run(run_id=run_id):
                        # Log all metrics
                        for metric_name, value in metrics.items():
                            mlflow.log_metric(f"test_{metric_name}", value)

                        # Log confusion matrix as artifact
                        cm_dict = {
                            "confusion_matrix": confusion_matrix.tolist()
                        }
                        mlflow.log_dict(cm_dict, "confusion_matrix.json")

        except Exception as e:
            logger.warning(f"Could not log to MLflow: {str(e)}")

    def cross_validate_model(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        cv: Optional[int] = None,
        scoring: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Perform cross-validation on a model.

        Args:
            model: Model to evaluate
            X: Features
            y: Labels
            cv: Number of folds. If None, uses config default
            scoring: List of metrics to compute. If None, uses config default

        Returns:
            Dictionary of cross-validation scores
        """
        if cv is None:
            cv = self.config.get("training.cv_folds", 5)

        if scoring is None:
            scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

        logger.info(f"Performing {cv}-fold cross-validation...")

        try:
            cv_results = cross_validate(
                model, X, y,
                cv=cv,
                scoring=scoring,
                return_train_score=True,
                n_jobs=self.config.get("training.n_jobs", -1)
            )

            # Log results
            for metric in scoring:
                test_scores = cv_results[f"test_{metric}"]
                logger.info(f"{metric}: {test_scores.mean():.4f} (+/- {test_scores.std() * 2:.4f})")

            return cv_results

        except Exception as e:
            raise EvaluationError(f"Cross-validation failed: {str(e)}")

    def compare_models(
        self,
        models: Dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> pd.DataFrame:
        """
        Compare multiple models and return summary DataFrame.

        Args:
            models: Dictionary of model name to model object
            X_test: Test features
            y_test: Test labels

        Returns:
            DataFrame comparing all models
        """
        logger.info(f"Comparing {len(models)} models...")

        comparison_data = []

        for model_name, model in models.items():
            try:
                results = self.evaluate_model(model, X_test, y_test, model_name)
                metrics = results["metrics"]

                comparison_data.append({
                    "Model": model_name,
                    "Accuracy": metrics["accuracy"],
                    "Precision": metrics["precision"],
                    "Recall": metrics["recall"],
                    "F1-Score": metrics["f1"],
                    "ROC-AUC": metrics.get("roc_auc", np.nan),
                    "Specificity": metrics["specificity"],
                    "Sensitivity": metrics["sensitivity"],
                })

            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {str(e)}")

        comparison_df = pd.DataFrame(comparison_data)

        # Sort by F1-Score
        comparison_df = comparison_df.sort_values("F1-Score", ascending=False)

        logger.info("Model comparison complete")
        return comparison_df

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"]
        )

        ax.set_title(f"Confusion Matrix - {model_name}")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved confusion matrix to {save_path}")

        return fig

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot ROC curve.

        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            model_name: Name of the model
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve - {model_name}")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved ROC curve to {save_path}")

        return fig

    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot Precision-Recall curve.

        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            model_name: Name of the model
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color="blue", lw=2, label=f"PR curve (AUC = {pr_auc:.3f})")

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall Curve - {model_name}")
        ax.legend(loc="lower left")
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved PR curve to {save_path}")

        return fig

    def plot_model_comparison(
        self,
        comparison_df: pd.DataFrame,
        metric: str = "F1-Score",
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot model comparison bar chart.

        Args:
            comparison_df: DataFrame with model comparison results
            metric: Metric to plot
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        comparison_df.plot(
            x="Model",
            y=metric,
            kind="bar",
            ax=ax,
            color="steelblue",
            legend=False
        )

        ax.set_title(f"Model Comparison - {metric}")
        ax.set_xlabel("Model")
        ax.set_ylabel(metric)
        ax.set_ylim([0, 1.0])
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f")

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved comparison plot to {save_path}")

        return fig

    def save_evaluation_report(
        self,
        save_dir: Optional[Path] = None
    ) -> Path:
        """
        Save comprehensive evaluation report as JSON.

        Args:
            save_dir: Directory to save report. If None, uses default

        Returns:
            Path to saved report
        """
        if save_dir is None:
            save_dir = self.config.get_path("paths.models.metadata")

        save_dir.mkdir(parents=True, exist_ok=True)

        report_path = save_dir / "evaluation_report.json"

        try:
            with open(report_path, "w") as f:
                json.dump(self.evaluation_results, f, indent=2, default=str)

            logger.info(f"Saved evaluation report to {report_path}")
            return report_path

        except Exception as e:
            raise EvaluationError(f"Failed to save evaluation report: {str(e)}")

    def get_best_model(
        self,
        metric: str = "f1"
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Get the best performing model based on a metric.

        Args:
            metric: Metric to use for comparison

        Returns:
            Tuple of (model_name, evaluation_results)
        """
        if not self.evaluation_results:
            raise EvaluationError("No evaluation results available")

        best_model_name = None
        best_score = -np.inf

        for model_name, results in self.evaluation_results.items():
            score = results["metrics"].get(metric, -np.inf)
            if score > best_score:
                best_score = score
                best_model_name = model_name

        logger.info(f"Best model: {best_model_name} ({metric} = {best_score:.4f})")

        return best_model_name, self.evaluation_results[best_model_name]
