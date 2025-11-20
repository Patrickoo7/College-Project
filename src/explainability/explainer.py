"""Model explainability using SHAP and LIME."""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelExplainer:
    """
    Explain model predictions using SHAP and LIME.

    Features:
    - SHAP values for global and local explanations
    - LIME for local interpretable explanations
    - Feature importance visualization
    - Individual prediction explanations
    """

    def __init__(self, model, feature_names: Optional[List[str]] = None):
        """
        Initialize model explainer.

        Args:
            model: Trained model to explain
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names or []
        self.config = Config()

        # Paths
        self.output_dir = Path(self.config.get("paths.data.processed", "data/processed")) / "explanations"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.shap_explainer = None
        self.lime_explainer = None

    def initialize_shap(self, X_background: pd.DataFrame, model_type: str = "auto"):
        """
        Initialize SHAP explainer.

        Args:
            X_background: Background dataset for SHAP
            model_type: Type of SHAP explainer (auto, tree, linear, kernel)
        """
        try:
            import shap

            logger.info(f"Initializing SHAP explainer (type: {model_type})...")

            if model_type == "auto":
                # Auto-detect explainer type
                if hasattr(self.model, "predict_proba"):
                    try:
                        # Try tree explainer first
                        self.shap_explainer = shap.TreeExplainer(self.model)
                        logger.info("Using SHAP TreeExplainer")
                    except:
                        # Fall back to kernel explainer
                        self.shap_explainer = shap.KernelExplainer(
                            self.model.predict_proba,
                            shap.sample(X_background, min(100, len(X_background)))
                        )
                        logger.info("Using SHAP KernelExplainer")
                else:
                    self.shap_explainer = shap.Explainer(self.model, X_background)

            elif model_type == "tree":
                self.shap_explainer = shap.TreeExplainer(self.model)

            elif model_type == "linear":
                self.shap_explainer = shap.LinearExplainer(self.model, X_background)

            elif model_type == "kernel":
                self.shap_explainer = shap.KernelExplainer(
                    self.model.predict_proba if hasattr(self.model, "predict_proba") else self.model.predict,
                    shap.sample(X_background, min(100, len(X_background)))
                )

            logger.info("SHAP explainer initialized successfully")

        except ImportError:
            logger.warning("SHAP not installed. Install with: pip install shap")
            self.shap_explainer = None
        except Exception as e:
            logger.error(f"Error initializing SHAP: {e}")
            self.shap_explainer = None

    def initialize_lime(self, X_train: pd.DataFrame, feature_names: Optional[List[str]] = None):
        """
        Initialize LIME explainer.

        Args:
            X_train: Training data
            feature_names: Feature names
        """
        try:
            from lime.lime_tabular import LimeTabularExplainer

            logger.info("Initializing LIME explainer...")

            if feature_names is None:
                feature_names = self.feature_names

            self.lime_explainer = LimeTabularExplainer(
                training_data=X_train.values,
                feature_names=feature_names,
                class_names=["No Disease", "Disease"],
                mode="classification",
                random_state=42
            )

            logger.info("LIME explainer initialized successfully")

        except ImportError:
            logger.warning("LIME not installed. Install with: pip install lime")
            self.lime_explainer = None
        except Exception as e:
            logger.error(f"Error initializing LIME: {e}")
            self.lime_explainer = None

    def explain_prediction_shap(
        self,
        X: pd.DataFrame,
        show_plot: bool = False,
        save_plot: bool = True
    ) -> Optional[Any]:
        """
        Explain prediction using SHAP.

        Args:
            X: Input features
            show_plot: Whether to display plot
            save_plot: Whether to save plot

        Returns:
            SHAP values or None
        """
        if self.shap_explainer is None:
            logger.warning("SHAP explainer not initialized")
            return None

        try:
            import shap

            logger.info("Computing SHAP values...")

            # Compute SHAP values
            shap_values = self.shap_explainer(X)

            # Create visualizations
            if show_plot or save_plot:
                # Waterfall plot for first instance
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.plots.waterfall(shap_values[0], show=False)

                if save_plot:
                    plot_path = self.output_dir / "shap_waterfall.png"
                    plt.savefig(plot_path, bbox_inches="tight", dpi=150)
                    logger.info(f"Saved SHAP waterfall plot to {plot_path}")

                if show_plot:
                    plt.show()

                plt.close()

            return shap_values

        except Exception as e:
            logger.error(f"Error computing SHAP values: {e}")
            return None

    def explain_prediction_lime(
        self,
        X: np.ndarray,
        instance_idx: int = 0,
        num_features: int = 10,
        show_plot: bool = False,
        save_plot: bool = True
    ) -> Optional[Any]:
        """
        Explain prediction using LIME.

        Args:
            X: Input features
            instance_idx: Index of instance to explain
            num_features: Number of features to show
            show_plot: Whether to display plot
            save_plot: Whether to save plot

        Returns:
            LIME explanation or None
        """
        if self.lime_explainer is None:
            logger.warning("LIME explainer not initialized")
            return None

        try:
            logger.info(f"Computing LIME explanation for instance {instance_idx}...")

            # Get prediction function
            if hasattr(self.model, "predict_proba"):
                predict_fn = self.model.predict_proba
            else:
                predict_fn = self.model.predict

            # Explain instance
            explanation = self.lime_explainer.explain_instance(
                data_row=X[instance_idx],
                predict_fn=predict_fn,
                num_features=num_features
            )

            # Save/show visualization
            if show_plot or save_plot:
                fig = explanation.as_pyplot_figure()

                if save_plot:
                    plot_path = self.output_dir / f"lime_explanation_{instance_idx}.png"
                    plt.savefig(plot_path, bbox_inches="tight", dpi=150)
                    logger.info(f"Saved LIME explanation to {plot_path}")

                if show_plot:
                    plt.show()

                plt.close()

            return explanation

        except Exception as e:
            logger.error(f"Error computing LIME explanation: {e}")
            return None

    def plot_feature_importance_shap(
        self,
        X: pd.DataFrame,
        max_display: int = 20,
        show_plot: bool = False,
        save_plot: bool = True
    ):
        """
        Plot SHAP feature importance.

        Args:
            X: Input features
            max_display: Maximum features to display
            show_plot: Whether to display plot
            save_plot: Whether to save plot
        """
        if self.shap_explainer is None:
            logger.warning("SHAP explainer not initialized")
            return

        try:
            import shap

            logger.info("Computing SHAP feature importance...")

            # Compute SHAP values
            shap_values = self.shap_explainer(X)

            # Summary plot
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.plots.beeswarm(shap_values, max_display=max_display, show=False)

            if save_plot:
                plot_path = self.output_dir / "shap_feature_importance.png"
                plt.savefig(plot_path, bbox_inches="tight", dpi=150)
                logger.info(f"Saved SHAP feature importance to {plot_path}")

            if show_plot:
                plt.show()

            plt.close()

            # Bar plot
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.plots.bar(shap_values, max_display=max_display, show=False)

            if save_plot:
                plot_path = self.output_dir / "shap_feature_importance_bar.png"
                plt.savefig(plot_path, bbox_inches="tight", dpi=150)
                logger.info(f"Saved SHAP bar plot to {plot_path}")

            if show_plot:
                plt.show()

            plt.close()

        except Exception as e:
            logger.error(f"Error plotting SHAP feature importance: {e}")

    def get_feature_contributions(
        self,
        X: pd.DataFrame,
        instance_idx: int = 0
    ) -> Optional[Dict[str, float]]:
        """
        Get feature contributions for a single prediction.

        Args:
            X: Input features
            instance_idx: Index of instance

        Returns:
            Dictionary of feature contributions
        """
        if self.shap_explainer is None:
            logger.warning("SHAP explainer not initialized")
            return None

        try:
            import shap

            # Compute SHAP values
            shap_values = self.shap_explainer(X.iloc[[instance_idx]])

            # Extract feature contributions
            if hasattr(shap_values, "values"):
                values = shap_values.values[0]
            else:
                values = shap_values[0]

            # Create dictionary
            feature_names = self.feature_names or X.columns.tolist()
            contributions = {
                feature: float(value)
                for feature, value in zip(feature_names, values)
            }

            # Sort by absolute contribution
            contributions = dict(
                sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
            )

            return contributions

        except Exception as e:
            logger.error(f"Error getting feature contributions: {e}")
            return None

    def generate_explanation_report(
        self,
        X: pd.DataFrame,
        instance_idx: int = 0,
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate comprehensive explanation report.

        Args:
            X: Input features
            instance_idx: Index of instance to explain
            output_path: Output path for report

        Returns:
            Path to generated report
        """
        if output_path is None:
            output_path = str(self.output_dir / f"explanation_report_{instance_idx}.txt")

        try:
            # Get prediction
            if hasattr(self.model, "predict_proba"):
                prediction_proba = self.model.predict_proba(X.iloc[[instance_idx]])[0]
                prediction = self.model.predict(X.iloc[[instance_idx]])[0]
            else:
                prediction = self.model.predict(X.iloc[[instance_idx]])[0]
                prediction_proba = None

            # Get feature contributions
            contributions = self.get_feature_contributions(X, instance_idx)

            # Generate report
            report = f"""
Model Explanation Report
========================

Instance Index: {instance_idx}

Prediction
----------
Predicted Class: {"Disease" if prediction == 1 else "No Disease"}
"""

            if prediction_proba is not None:
                report += f"""Probability No Disease: {prediction_proba[0]:.4f}
Probability Disease: {prediction_proba[1]:.4f}
"""

            report += """
Input Features
--------------
"""
            for col in X.columns:
                report += f"{col}: {X.iloc[instance_idx][col]}\n"

            if contributions:
                report += """
Feature Contributions (SHAP)
----------------------------
"""
                for feature, value in contributions.items():
                    direction = "→ Disease" if value > 0 else "→ No Disease"
                    report += f"{feature}: {value:+.4f} {direction}\n"

            # Save report
            with open(output_path, "w") as f:
                f.write(report)

            logger.info(f"Saved explanation report to {output_path}")

            return output_path

        except Exception as e:
            logger.error(f"Error generating explanation report: {e}")
            return None


if __name__ == "__main__":
    # Example usage
    from src.data.data_loader import DataLoader
    from src.data.data_preprocessor import DataPreprocessor

    # Load data
    loader = DataLoader()
    datasets = loader.load_all_datasets()
    df = loader.combine_datasets(datasets)

    # Preprocess
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.preprocess_pipeline(
        df,
        handle_missing=True,
        remove_outliers=True,
        encode_categorical=True
    )

    X_train, X_test, y_train, y_test = preprocessor.split_data(df_clean)

    # Load model
    model_path = Path("models/artifacts/random_forest.pkl")
    if model_path.exists():
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Initialize explainer
        explainer = ModelExplainer(model, feature_names=X_train.columns.tolist())

        # Initialize SHAP
        explainer.initialize_shap(X_train[:100])

        # Explain predictions
        shap_values = explainer.explain_prediction_shap(X_test[:10], save_plot=True)

        # Plot feature importance
        explainer.plot_feature_importance_shap(X_test[:100], save_plot=True)

        # Get feature contributions
        contributions = explainer.get_feature_contributions(X_test, instance_idx=0)
        print("\nFeature Contributions:")
        for feature, value in list(contributions.items())[:10]:
            print(f"  {feature}: {value:+.4f}")

        # Generate report
        explainer.generate_explanation_report(X_test, instance_idx=0)
    else:
        print("Model not found. Train a model first.")
