"""Pydantic schemas for API request/response validation."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, validator


class PatientInput(BaseModel):
    """Input schema for patient data."""

    age: int = Field(..., ge=0, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (1=male, 0=female)")
    cp: int = Field(..., ge=0, le=4, description="Chest pain type (0-4)")
    trestbps: int = Field(..., ge=50, le=250, description="Resting blood pressure (mm Hg)")
    chol: int = Field(..., ge=100, le=600, description="Serum cholesterol (mg/dl)")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: int = Field(..., ge=50, le=250, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina (1=yes, 0=no)")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")

    class Config:
        schema_extra = {
            "example": {
                "age": 63,
                "sex": 1,
                "cp": 3,
                "trestbps": 145,
                "chol": 233,
                "restecg": 0,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3
            }
        }

    @validator("age")
    def validate_age(cls, v):
        """Validate age range."""
        if v < 0 or v > 120:
            raise ValueError("Age must be between 0 and 120")
        return v

    @validator("sex")
    def validate_sex(cls, v):
        """Validate sex value."""
        if v not in [0, 1]:
            raise ValueError("Sex must be 0 (female) or 1 (male)")
        return v


class PredictionResponse(BaseModel):
    """Response schema for prediction."""

    prediction: int = Field(..., description="Prediction (0=no disease, 1=disease)")
    prediction_label: str = Field(..., description="Human-readable prediction")
    probability: Optional[Dict[str, float]] = Field(None, description="Prediction probabilities")
    confidence: Optional[float] = Field(None, description="Confidence score")
    model_used: str = Field(..., description="Model used for prediction")
    input_data: Dict = Field(..., description="Input patient data")

    class Config:
        schema_extra = {
            "example": {
                "prediction": 1,
                "prediction_label": "Heart Disease",
                "probability": {
                    "no_disease": 0.25,
                    "disease": 0.75
                },
                "confidence": 0.75,
                "model_used": "random_forest",
                "input_data": {
                    "age": 63,
                    "sex": 1,
                    "cp": 3,
                    "trestbps": 145,
                    "chol": 233,
                    "restecg": 0,
                    "thalach": 150,
                    "exang": 0,
                    "oldpeak": 2.3
                }
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""

    patients: List[PatientInput] = Field(..., description="List of patient data")

    class Config:
        schema_extra = {
            "example": {
                "patients": [
                    {
                        "age": 63,
                        "sex": 1,
                        "cp": 3,
                        "trestbps": 145,
                        "chol": 233,
                        "restecg": 0,
                        "thalach": 150,
                        "exang": 0,
                        "oldpeak": 2.3
                    },
                    {
                        "age": 55,
                        "sex": 0,
                        "cp": 2,
                        "trestbps": 130,
                        "chol": 210,
                        "restecg": 1,
                        "thalach": 165,
                        "exang": 0,
                        "oldpeak": 1.5
                    }
                ]
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""

    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_count: int = Field(..., description="Total number of predictions")
    disease_count: int = Field(..., description="Number of patients with disease")
    no_disease_count: int = Field(..., description="Number of patients without disease")

    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "prediction": 1,
                        "prediction_label": "Heart Disease",
                        "probability": {"no_disease": 0.25, "disease": 0.75},
                        "confidence": 0.75,
                        "model_used": "random_forest",
                        "input_data": {}
                    }
                ],
                "total_count": 2,
                "disease_count": 1,
                "no_disease_count": 1
            }
        }


class ModelInfo(BaseModel):
    """Schema for model information."""

    name: str = Field(..., description="Model name")
    type: str = Field(..., description="Model type")
    available: bool = Field(..., description="Whether model is available")
    path: Optional[str] = Field(None, description="Path to model file")
    metrics: Optional[Dict[str, float]] = Field(None, description="Model performance metrics")

    class Config:
        schema_extra = {
            "example": {
                "name": "random_forest",
                "type": "RandomForestClassifier",
                "available": True,
                "path": "models/artifacts/random_forest.pkl",
                "metrics": {
                    "accuracy": 0.85,
                    "f1_score": 0.82,
                    "roc_auc": 0.88
                }
            }
        }


class ModelsListResponse(BaseModel):
    """Response schema for list of available models."""

    models: List[ModelInfo] = Field(..., description="List of available models")
    default_model: str = Field(..., description="Default model used for predictions")
    total_count: int = Field(..., description="Total number of models")


class HealthResponse(BaseModel):
    """Response schema for health check."""

    status: str = Field(..., description="API status")
    api_version: str = Field(..., description="API version")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    gpu_count: int = Field(0, description="Number of GPUs available")

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "api_version": "1.0.0",
                "gpu_available": True,
                "gpu_count": 1
            }
        }


class ErrorResponse(BaseModel):
    """Response schema for errors."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    status_code: int = Field(..., description="HTTP status code")

    class Config:
        schema_extra = {
            "example": {
                "error": "Validation Error",
                "detail": "Age must be between 0 and 120",
                "status_code": 422
            }
        }
