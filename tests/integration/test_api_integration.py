"""Integration tests for the FastAPI application.

This module demonstrates end-to-end API testing with authentication and prediction flows.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
import pandas as pd

from src.api.app import app
from src.config import get_config


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_patient_data():
    """Sample patient data for predictions."""
    return {
        "age": 55,
        "sex": 1,
        "cp": 0,
        "trestbps": 140,
        "chol": 250,
        "fbs": 0,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 1,
        "ca": 0,
        "thal": 2
    }


@pytest.fixture
def auth_token(client):
    """Get authentication token for testing (if auth is enabled)."""
    config = get_config()

    if not config.api.enable_auth:
        return None

    # Login to get token
    response = client.post(
        f"{config.api.prefix}/auth/login",
        data={"username": "admin", "password": "secret"}
    )

    if response.status_code == 200:
        return response.json()["access_token"]

    return None


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns API information."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert "name" in data
        assert "version" in data
        assert "status" in data
        assert data["status"] == "running"

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert data["status"] == "healthy"
        assert "api_version" in data

    def test_prefixed_health_endpoint(self, client):
        """Test health endpoint with API prefix."""
        config = get_config()
        response = client.get(f"{config.api.prefix}/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestAuthenticationEndpoints:
    """Test authentication endpoints."""

    def test_auth_status(self, client):
        """Test auth status endpoint."""
        config = get_config()
        response = client.get(f"{config.api.prefix}/auth/status")

        assert response.status_code == 200
        data = response.json()

        assert "enabled" in data
        assert data["enabled"] == config.api.enable_auth

    @pytest.mark.skipif(
        not get_config().api.enable_auth,
        reason="Authentication is disabled"
    )
    def test_login_success(self, client):
        """Test successful login."""
        config = get_config()
        response = client.post(
            f"{config.api.prefix}/auth/login",
            data={"username": "admin", "password": "secret"}
        )

        assert response.status_code == 200
        data = response.json()

        assert "access_token" in data
        assert "token_type" in data
        assert data["token_type"] == "bearer"
        assert "user" in data

    @pytest.mark.skipif(
        not get_config().api.enable_auth,
        reason="Authentication is disabled"
    )
    def test_login_invalid_credentials(self, client):
        """Test login with invalid credentials."""
        config = get_config()
        response = client.post(
            f"{config.api.prefix}/auth/login",
            data={"username": "admin", "password": "wrong_password"}
        )

        assert response.status_code == 401

    @pytest.mark.skipif(
        not get_config().api.enable_auth,
        reason="Authentication is disabled"
    )
    def test_get_current_user(self, client, auth_token):
        """Test getting current user info."""
        config = get_config()

        if auth_token is None:
            pytest.skip("Could not obtain auth token")

        response = client.get(
            f"{config.api.prefix}/auth/me",
            headers={"Authorization": f"Bearer {auth_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        assert "username" in data
        assert data["username"] == "admin"

    @pytest.mark.skipif(
        not get_config().api.enable_auth,
        reason="Authentication is disabled"
    )
    def test_validate_token(self, client, auth_token):
        """Test token validation endpoint."""
        config = get_config()

        if auth_token is None:
            pytest.skip("Could not obtain auth token")

        response = client.post(
            f"{config.api.prefix}/auth/validate",
            headers={"Authorization": f"Bearer {auth_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        assert "valid" in data
        assert data["valid"] is True


class TestPredictionEndpoints:
    """Test prediction endpoints."""

    @patch('src.api.routes.predictor')
    def test_predict_single(self, mock_predictor, client, sample_patient_data, auth_token):
        """Test single prediction endpoint."""
        # Mock predictor
        mock_predictor.predict_single.return_value = {
            "prediction": 1,
            "prediction_label": "Disease",
            "probability": {"disease": 0.75, "no_disease": 0.25}
        }

        config = get_config()
        headers = {}

        if config.api.enable_auth and auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        response = client.post(
            f"{config.api.prefix}/predict",
            json=sample_patient_data,
            headers=headers
        )

        # Skip if predictor is None (no model loaded)
        if response.status_code == 503:
            pytest.skip("No model loaded")

        assert response.status_code == 200
        data = response.json()

        assert "prediction" in data
        assert "prediction_label" in data
        assert "confidence" in data

    @patch('src.api.routes.predictor')
    def test_predict_batch(self, mock_predictor, client, sample_patient_data, auth_token):
        """Test batch prediction endpoint."""
        # Mock predictor
        mock_predictor.predict_single.return_value = {
            "prediction": 1,
            "prediction_label": "Disease",
            "probability": {"disease": 0.75, "no_disease": 0.25}
        }

        config = get_config()
        headers = {}

        if config.api.enable_auth and auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        batch_request = {
            "patients": [sample_patient_data, sample_patient_data]
        }

        response = client.post(
            f"{config.api.prefix}/predict/batch",
            json=batch_request,
            headers=headers
        )

        # Skip if predictor is None
        if response.status_code == 503:
            pytest.skip("No model loaded")

        assert response.status_code == 200
        data = response.json()

        assert "predictions" in data
        assert "total_count" in data
        assert data["total_count"] == 2


class TestModelEndpoints:
    """Test model management endpoints."""

    def test_list_models(self, client, auth_token):
        """Test listing available models."""
        config = get_config()
        headers = {}

        if config.api.enable_auth and auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        response = client.get(
            f"{config.api.prefix}/models",
            headers=headers
        )

        assert response.status_code == 200
        data = response.json()

        assert "models" in data
        assert "default_model" in data
        assert "total_count" in data

    def test_get_metrics(self, client, auth_token):
        """Test getting API metrics."""
        config = get_config()
        headers = {}

        if config.api.enable_auth and auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        response = client.get(
            f"{config.api.prefix}/metrics",
            headers=headers
        )

        assert response.status_code == 200
        data = response.json()

        assert "current_model" in data
        assert "gpu_enabled" in data


class TestEndToEndFlow:
    """Test complete end-to-end workflows."""

    @patch('src.api.routes.predictor')
    def test_complete_prediction_flow(self, mock_predictor, client, sample_patient_data):
        """Test complete flow: health check -> auth -> predict."""
        config = get_config()

        # 1. Check health
        health_response = client.get("/health")
        assert health_response.status_code == 200

        # 2. Get auth token if needed
        headers = {}
        if config.api.enable_auth:
            login_response = client.post(
                f"{config.api.prefix}/auth/login",
                data={"username": "admin", "password": "secret"}
            )
            if login_response.status_code == 200:
                token = login_response.json()["access_token"]
                headers["Authorization"] = f"Bearer {token}"

        # 3. Mock predictor
        mock_predictor.predict_single.return_value = {
            "prediction": 1,
            "prediction_label": "Disease",
            "probability": {"disease": 0.75, "no_disease": 0.25}
        }

        # 4. Make prediction
        predict_response = client.post(
            f"{config.api.prefix}/predict",
            json=sample_patient_data,
            headers=headers
        )

        if predict_response.status_code == 503:
            pytest.skip("No model loaded")

        assert predict_response.status_code == 200
        prediction_data = predict_response.json()

        assert "prediction" in prediction_data
        assert "model_used" in prediction_data

    def test_config_driven_behavior(self, client):
        """Test that API behavior matches configuration."""
        config = get_config()

        # Test that authentication status matches config
        auth_status = client.get(f"{config.api.prefix}/auth/status")
        assert auth_status.json()["enabled"] == config.api.enable_auth

        # Test that API version matches config
        root = client.get("/")
        assert root.json()["version"] == config.api.version

        # Test that API prefix is used correctly
        health_with_prefix = client.get(f"{config.api.prefix}/health")
        assert health_with_prefix.status_code == 200


# Run tests with: pytest tests/integration/test_api_integration.py -v
# Run with authentication: APP_API__ENABLE_AUTH=true pytest tests/integration/test_api_integration.py -v
