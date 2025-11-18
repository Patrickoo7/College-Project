"""FastAPI application for heart disease prediction API."""

from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import uvicorn

from .routes import router
from .schemas import PredictionResponse, HealthResponse
from ..utils.config import get_config
from ..utils.logger import get_logger
from ..utils.gpu_utils import get_gpu_manager

logger = get_logger(__name__)
config = get_config()

# Create FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="REST API for predicting heart disease using machine learning models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix="/api/v1")


@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    logger.info("Starting Heart Disease Prediction API...")

    # Print GPU info
    gpu_manager = get_gpu_manager()
    gpu_available, gpu_info = gpu_manager.detect_gpu()

    if gpu_available:
        logger.info(f"GPU acceleration available: {gpu_info['device_count']} device(s)")
    else:
        logger.info("Running on CPU")

    logger.info("API startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    logger.info("Shutting down Heart Disease Prediction API...")


@app.get("/", response_model=Dict)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Heart Disease Prediction API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/health",
        "endpoints": {
            "predict": "/api/v1/predict",
            "predict_batch": "/api/v1/predict/batch",
            "models": "/api/v1/models",
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    gpu_manager = get_gpu_manager()
    gpu_available, gpu_info = gpu_manager.detect_gpu()

    return HealthResponse(
        status="healthy",
        api_version="1.0.0",
        gpu_available=gpu_available,
        gpu_count=gpu_info.get("device_count", 0)
    )


if __name__ == "__main__":
    # Run with: python -m src.api.app
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
