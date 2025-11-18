"""API routes for heart disease prediction."""

import io
import re
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, File, UploadFile, Query
from fastapi.responses import StreamingResponse
import pandas as pd

from .schemas import (
    PatientInput,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfo,
    ModelsListResponse,
    HealthResponse,
)
from ..models.predict import HeartDiseasePredictor
from ..utils.config import get_config
from ..utils.exceptions import ModelPredictionError, ModelLoadError
from ..utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Load default model
config = get_config()
default_model_name = "random_forest"  # Can be made configurable
predictor = None

# Try to load default model
try:
    default_model_path = config.get_path("paths.models.artifacts") / f"{default_model_name}.pkl"
    if default_model_path.exists():
        predictor = HeartDiseasePredictor(model_path=default_model_path)
        logger.info(f"Loaded default model: {default_model_name}")
    else:
        logger.warning(f"Default model not found: {default_model_path}")
except Exception as e:
    logger.error(f"Failed to load default model: {str(e)}")


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health status."""
    from ..utils.gpu_utils import get_gpu_manager

    gpu_manager = get_gpu_manager()
    gpu_available, gpu_info = gpu_manager.detect_gpu()

    return HealthResponse(
        status="healthy",
        api_version="1.0.0",
        gpu_available=gpu_available,
        gpu_count=gpu_info.get("device_count", 0)
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    patient: PatientInput,
    model_name: Optional[str] = Query(None, description="Model to use for prediction")
):
    """
    Make a prediction for a single patient.

    Args:
        patient: Patient data
        model_name: Optional model name to use (defaults to random_forest)

    Returns:
        Prediction result with probability and confidence
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="No model loaded. Please train a model first."
        )

    try:
        # Convert patient data to dictionary
        patient_data = patient.dict()

        # Make prediction
        result = predictor.predict_single(**patient_data)

        # Safely extract probability and confidence
        probability = result.get("probability") or {}
        confidence = probability.get("disease") if isinstance(probability, dict) else None

        # Format response
        response = PredictionResponse(
            prediction=result["prediction"],
            prediction_label=result["prediction_label"],
            probability=result.get("probability"),
            confidence=confidence,
            model_used=model_name or default_model_name,
            input_data=patient_data
        )

        logger.info(f"Prediction made: {result['prediction_label']}")
        return response

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Make predictions for multiple patients.

    Args:
        request: Batch prediction request with list of patients

    Returns:
        Batch prediction results with summary statistics
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="No model loaded. Please train a model first."
        )

    try:
        predictions = []
        disease_count = 0
        no_disease_count = 0

        # Process each patient
        for patient in request.patients:
            patient_data = patient.dict()

            # Make prediction
            result = predictor.predict_single(**patient_data)

            # Safely extract probability and confidence
            probability = result.get("probability") or {}
            confidence = probability.get("disease") if isinstance(probability, dict) else None

            # Create response
            pred_response = PredictionResponse(
                prediction=result["prediction"],
                prediction_label=result["prediction_label"],
                probability=result.get("probability"),
                confidence=confidence,
                model_used=default_model_name,
                input_data=patient_data
            )

            predictions.append(pred_response)

            # Update counts
            if result["prediction"] == 1:
                disease_count += 1
            else:
                no_disease_count += 1

        # Create batch response
        response = BatchPredictionResponse(
            predictions=predictions,
            total_count=len(predictions),
            disease_count=disease_count,
            no_disease_count=no_disease_count
        )

        logger.info(f"Batch prediction complete: {len(predictions)} patients")
        return response

    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@router.post("/predict/upload")
async def predict_from_file(file: UploadFile = File(...)):
    """
    Make predictions from uploaded CSV file.

    Args:
        file: CSV file with patient data

    Returns:
        CSV file with predictions
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="No model loaded. Please train a model first."
        )

    try:
        # Validate file type
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        if not file.filename.lower().endswith('.csv'):
            raise HTTPException(
                status_code=400,
                detail="Only CSV files are allowed"
            )

        # Read file contents with size limit
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
        contents = await file.read()

        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE / 1024 / 1024} MB"
            )

        # Read CSV with row limit
        MAX_ROWS = 10000
        df = pd.read_csv(io.BytesIO(contents), nrows=MAX_ROWS)

        if len(df) == 0:
            raise HTTPException(status_code=400, detail="CSV file is empty")

        logger.info(f"Processing uploaded file with {len(df)} records")

        # Make predictions
        results = predictor.predict_batch(df, include_probabilities=True)

        # Convert to CSV
        output = io.StringIO()
        results.to_csv(output, index=False)
        output.seek(0)

        # Return as downloadable file
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"}
        )

    except Exception as e:
        logger.error(f"File prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File prediction failed: {str(e)}")


@router.get("/models", response_model=ModelsListResponse)
async def list_models():
    """
    List all available models.

    Returns:
        List of available models with information
    """
    try:
        models_path = config.get_path("paths.models.artifacts")

        if not models_path.exists():
            return ModelsListResponse(
                models=[],
                default_model=default_model_name,
                total_count=0
            )

        # Find all model files
        model_files = list(models_path.glob("*.pkl"))

        models = []
        for model_file in model_files:
            model_name = model_file.stem

            model_info = ModelInfo(
                name=model_name,
                type="Unknown",  # Could load and inspect model
                available=True,
                path=str(model_file)
            )

            models.append(model_info)

        response = ModelsListResponse(
            models=models,
            default_model=default_model_name,
            total_count=len(models)
        )

        logger.info(f"Listed {len(models)} available models")
        return response

    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@router.get("/models/{model_name}", response_model=ModelInfo)
async def get_model_info(model_name: str):
    """
    Get information about a specific model.

    Args:
        model_name: Name of the model

    Returns:
        Model information
    """
    # Validate model name to prevent path traversal
    if not re.match(r'^[a-zA-Z0-9_-]+$', model_name):
        raise HTTPException(
            status_code=400,
            detail="Invalid model name. Only alphanumeric characters, underscores, and hyphens are allowed."
        )

    try:
        models_path = config.get_path("paths.models.artifacts")
        model_file = models_path / f"{model_name}.pkl"

        if not model_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found"
            )

        # Load model to get info
        temp_predictor = HeartDiseasePredictor(model_path=model_file)
        model_info_dict = temp_predictor.get_model_info()

        model_info = ModelInfo(
            name=model_name,
            type=model_info_dict.get("model_type", "Unknown"),
            available=model_info_dict.get("loaded", False),
            path=str(model_file)
        )

        logger.info(f"Retrieved info for model: {model_name}")
        return model_info

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


@router.post("/models/{model_name}/use")
async def use_model(model_name: str):
    """
    Switch to using a different model.

    Args:
        model_name: Name of the model to use

    Returns:
        Success message
    """
    # Validate model name to prevent path traversal
    if not re.match(r'^[a-zA-Z0-9_-]+$', model_name):
        raise HTTPException(
            status_code=400,
            detail="Invalid model name. Only alphanumeric characters, underscores, and hyphens are allowed."
        )

    global predictor, default_model_name

    try:
        models_path = config.get_path("paths.models.artifacts")
        model_file = models_path / f"{model_name}.pkl"

        if not model_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found"
            )

        # Load new model
        predictor = HeartDiseasePredictor(model_path=model_file)
        default_model_name = model_name

        logger.info(f"Switched to model: {model_name}")

        return {
            "message": f"Successfully switched to model: {model_name}",
            "model": model_name
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to switch model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to switch model: {str(e)}")


@router.get("/metrics")
async def get_metrics():
    """
    Get API metrics and statistics.

    Returns:
        API usage metrics
    """
    # This could be enhanced with actual metrics tracking
    return {
        "total_predictions": "N/A",
        "uptime": "N/A",
        "current_model": default_model_name,
        "gpu_enabled": config.get("gpu.enabled", True)
    }
