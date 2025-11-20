"""FastAPI application for heart disease prediction API."""

from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import uvicorn

from .routes import router
from .auth_routes import router as auth_router
from .schemas import PredictionResponse, HealthResponse
from ..config import get_config
from ..utils.logger import get_logger
from ..utils.gpu_utils import get_gpu_manager

logger = get_logger(__name__)
config = get_config()

# Create FastAPI app with config-driven settings
app = FastAPI(
    title=config.api.title,
    description=config.api.description,
    version=config.api.version,
    docs_url="/docs",
    redoc_url="/redoc",
    debug=config.debug,
)

# Configure CORS from config
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors.allowed_origins,
    allow_credentials=config.api.cors.allow_credentials,
    allow_methods=config.api.cors.allow_methods,
    allow_headers=config.api.cors.allow_headers,
)

# Optional: Add custom middleware (uncomment to enable)
# from .middleware import RequestLoggingMiddleware, AuthenticationMiddleware, RateLimitMiddleware
# app.add_middleware(RequestLoggingMiddleware)
# app.add_middleware(AuthenticationMiddleware)
# app.add_middleware(RateLimitMiddleware, max_requests=100, window_seconds=60)

# Include routers with config-driven prefix
app.include_router(router, prefix=config.api.prefix)
app.include_router(auth_router, prefix=config.api.prefix)


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
    prefix = config.api.prefix
    return {
        "name": config.api.title,
        "version": config.api.version,
        "environment": config.environment,
        "status": "running",
        "authentication_enabled": config.api.enable_auth,
        "docs": "/docs",
        "health": f"{prefix}/health",
        "endpoints": {
            "predict": f"{prefix}/predict",
            "predict_batch": f"{prefix}/predict/batch",
            "models": f"{prefix}/models",
            "auth_login": f"{prefix}/auth/login",
            "auth_status": f"{prefix}/auth/status",
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    gpu_manager = get_gpu_manager()
    gpu_available, gpu_info = gpu_manager.detect_gpu()

    return HealthResponse(
        status="healthy",
        api_version=config.api.version,
        gpu_available=gpu_available,
        gpu_count=gpu_info.get("device_count", 0)
    )


if __name__ == "__main__":
    # Run with: python -m src.api.app
    # Configuration is loaded from configs/app_config.yaml and environment-specific overrides
    uvicorn.run(
        "src.api.app:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.debug,
        log_level=config.logging.level.lower()
    )
