"""Prediction module for making heart disease predictions."""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..utils.config import get_config
from ..utils.exceptions import ModelPredictionError, ModelLoadError
from ..utils.logger import get_logger

logger = get_logger(__name__)


class HeartDiseasePredictor:
    """Make predictions using trained heart disease models."""

    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize predictor.

        Args:
            model_path: Path to saved model. If None, must call load_model manually
        """
        self.config = get_config()
        self.model = None
        self.scaler = None
        self.feature_names = None

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: Path) -> None:
        """
        Load a trained model from disk.

        Args:
            model_path: Path to the saved model pickle file

        Raises:
            ModelLoadError: If model loading fails
        """
        if not model_path.exists():
            raise ModelLoadError(f"Model file not found: {model_path}")

        try:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)

            logger.info(f"Loaded model from {model_path}")

            # Try to get feature names if available
            if hasattr(self.model, "feature_names_in_"):
                self.feature_names = self.model.feature_names_in_

        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {str(e)}")

    def load_scaler(self, scaler_path: Path) -> None:
        """
        Load a fitted scaler from disk.

        Args:
            scaler_path: Path to the saved scaler pickle file

        Raises:
            ModelLoadError: If scaler loading fails
        """
        if not scaler_path.exists():
            raise ModelLoadError(f"Scaler file not found: {scaler_path}")

        try:
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)

            logger.info(f"Loaded scaler from {scaler_path}")

        except Exception as e:
            raise ModelLoadError(f"Failed to load scaler: {str(e)}")

    def preprocess_input(
        self,
        input_data: Union[Dict, pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Preprocess input data for prediction.

        Args:
            input_data: Input data as dictionary, DataFrame, or numpy array

        Returns:
            Preprocessed numpy array

        Raises:
            ModelPredictionError: If preprocessing fails
        """
        try:
            # Convert to DataFrame if needed
            if isinstance(input_data, dict):
                input_df = pd.DataFrame([input_data])
            elif isinstance(input_data, np.ndarray):
                if self.feature_names is not None:
                    input_df = pd.DataFrame(input_data, columns=self.feature_names)
                else:
                    input_df = pd.DataFrame(input_data)
            elif isinstance(input_data, pd.DataFrame):
                input_df = input_data.copy()
            else:
                raise ModelPredictionError(f"Unsupported input type: {type(input_data)}")

            # Scale if scaler is available
            if self.scaler is not None:
                input_array = self.scaler.transform(input_df)
            else:
                input_array = input_df.values

            return input_array

        except Exception as e:
            raise ModelPredictionError(f"Failed to preprocess input: {str(e)}")

    def predict(
        self,
        input_data: Union[Dict, pd.DataFrame, np.ndarray],
        preprocess: bool = True
    ) -> np.ndarray:
        """
        Make predictions on input data.

        Args:
            input_data: Input data
            preprocess: Whether to preprocess the input

        Returns:
            Array of predictions (0 or 1)

        Raises:
            ModelPredictionError: If prediction fails
        """
        if self.model is None:
            raise ModelPredictionError("No model loaded. Call load_model() first.")

        try:
            if preprocess:
                processed_data = self.preprocess_input(input_data)
            else:
                if isinstance(input_data, (pd.DataFrame, dict)):
                    processed_data = pd.DataFrame(input_data).values
                else:
                    processed_data = input_data

            predictions = self.model.predict(processed_data)

            logger.info(f"Made predictions for {len(predictions)} samples")

            return predictions

        except Exception as e:
            raise ModelPredictionError(f"Prediction failed: {str(e)}")

    def predict_proba(
        self,
        input_data: Union[Dict, pd.DataFrame, np.ndarray],
        preprocess: bool = True
    ) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            input_data: Input data
            preprocess: Whether to preprocess the input

        Returns:
            Array of prediction probabilities

        Raises:
            ModelPredictionError: If prediction fails or model doesn't support probabilities
        """
        if self.model is None:
            raise ModelPredictionError("No model loaded. Call load_model() first.")

        if not hasattr(self.model, "predict_proba"):
            raise ModelPredictionError("Model does not support probability predictions")

        try:
            if preprocess:
                processed_data = self.preprocess_input(input_data)
            else:
                if isinstance(input_data, (pd.DataFrame, dict)):
                    processed_data = pd.DataFrame(input_data).values
                else:
                    processed_data = input_data

            probabilities = self.model.predict_proba(processed_data)

            logger.info(f"Got prediction probabilities for {len(probabilities)} samples")

            return probabilities

        except Exception as e:
            raise ModelPredictionError(f"Probability prediction failed: {str(e)}")

    def predict_single(
        self,
        age: int,
        sex: int,
        cp: int,
        trestbps: int,
        chol: int,
        restecg: int,
        thalach: int,
        exang: int,
        oldpeak: float,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make prediction for a single patient.

        Args:
            age: Age in years
            sex: Sex (1 = male, 0 = female)
            cp: Chest pain type (0-3)
            trestbps: Resting blood pressure (mm Hg)
            chol: Serum cholesterol (mg/dl)
            restecg: Resting ECG results (0-2)
            thalach: Maximum heart rate achieved
            exang: Exercise induced angina (1 = yes, 0 = no)
            oldpeak: ST depression induced by exercise
            **kwargs: Additional features if needed

        Returns:
            Dictionary with prediction and probability
        """
        # Create input dictionary
        input_dict = {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
        }

        # Add any additional features
        input_dict.update(kwargs)

        # Make prediction
        prediction = self.predict(input_dict)[0]

        # Get probability if available
        probability = None
        if hasattr(self.model, "predict_proba"):
            proba = self.predict_proba(input_dict)[0]
            probability = {
                "no_disease": float(proba[0]),
                "disease": float(proba[1])
            }

        result = {
            "prediction": int(prediction),
            "prediction_label": "Heart Disease" if prediction == 1 else "No Heart Disease",
            "probability": probability,
            "input_data": input_dict
        }

        logger.info(f"Single prediction: {result['prediction_label']}")

        return result

    def predict_batch(
        self,
        data: pd.DataFrame,
        include_probabilities: bool = True
    ) -> pd.DataFrame:
        """
        Make predictions on a batch of data.

        Args:
            data: DataFrame with patient data
            include_probabilities: Whether to include probabilities

        Returns:
            DataFrame with predictions and probabilities
        """
        predictions = self.predict(data)

        results_df = data.copy()
        results_df["prediction"] = predictions
        results_df["prediction_label"] = results_df["prediction"].map({
            0: "No Heart Disease",
            1: "Heart Disease"
        })

        if include_probabilities and hasattr(self.model, "predict_proba"):
            probabilities = self.predict_proba(data)
            results_df["probability_no_disease"] = probabilities[:, 0]
            results_df["probability_disease"] = probabilities[:, 1]

        logger.info(f"Batch prediction complete for {len(data)} samples")

        return results_df

    def explain_prediction(
        self,
        input_data: Union[Dict, pd.DataFrame],
        method: str = "feature_importance"
    ) -> Dict[str, Any]:
        """
        Explain a prediction (basic feature importance).

        Args:
            input_data: Input data for prediction
            method: Explanation method

        Returns:
            Dictionary with explanation
        """
        if method == "feature_importance":
            if hasattr(self.model, "feature_importances_"):
                importance = self.model.feature_importances_

                if isinstance(input_data, dict):
                    features = list(input_data.keys())
                elif isinstance(input_data, pd.DataFrame):
                    features = input_data.columns.tolist()
                else:
                    features = self.feature_names if self.feature_names is not None else []

                if len(features) == len(importance):
                    feature_importance = dict(zip(features, importance))
                    sorted_importance = dict(
                        sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                    )

                    return {
                        "method": "feature_importance",
                        "importance": sorted_importance
                    }

        logger.warning(f"Explanation method '{method}' not available for this model")
        return {"method": method, "available": False}

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"loaded": False}

        info = {
            "loaded": True,
            "model_type": type(self.model).__name__,
            "has_scaler": self.scaler is not None,
            "supports_probabilities": hasattr(self.model, "predict_proba"),
            "n_features": getattr(self.model, "n_features_in_", "Unknown"),
        }

        if self.feature_names is not None:
            info["feature_names"] = self.feature_names.tolist()

        return info
