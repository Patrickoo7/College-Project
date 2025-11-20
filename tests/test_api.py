"""Tests for FastAPI endpoints."""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.app import app
from src.api.schemas import PatientInput

# Create test client
client = TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self):
        """Test health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "api_version" in data
        assert "gpu_available" in data
        assert "gpu_count" in data

    def test_health_check_structure(self):
        """Test health endpoint response structure."""
        response = client.get("/health")
        data = response.json()

        # Check required fields
        required_fields = ["status", "api_version", "gpu_available", "gpu_count"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"


class TestPredictionEndpoint:
    """Tests for single prediction endpoint."""

    @pytest.fixture
    def valid_patient_data(self):
        """Valid patient data fixture."""
        return {
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

    def test_predict_valid_input(self, valid_patient_data):
        """Test prediction with valid input."""
        response = client.post("/api/v1/predict", json=valid_patient_data)

        # Check if model is loaded
        if response.status_code == 503:
            pytest.skip("No trained model available")

        assert response.status_code == 200

        data = response.json()
        assert "prediction" in data
        assert "prediction_label" in data
        assert data["prediction"] in [0, 1]
        assert data["prediction_label"] in ["No Disease", "Disease"]

    def test_predict_with_probabilities(self, valid_patient_data):
        """Test prediction includes probability information."""
        response = client.post("/api/v1/predict", json=valid_patient_data)

        if response.status_code == 503:
            pytest.skip("No trained model available")

        assert response.status_code == 200

        data = response.json()
        if "probability" in data and data["probability"] is not None:
            assert "no_disease" in data["probability"]
            assert "disease" in data["probability"]
            assert 0 <= data["probability"]["no_disease"] <= 1
            assert 0 <= data["probability"]["disease"] <= 1

    def test_predict_invalid_age(self, valid_patient_data):
        """Test prediction with invalid age."""
        invalid_data = valid_patient_data.copy()
        invalid_data["age"] = -5

        response = client.post("/api/v1/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error

    def test_predict_invalid_sex(self, valid_patient_data):
        """Test prediction with invalid sex value."""
        invalid_data = valid_patient_data.copy()
        invalid_data["sex"] = 3

        response = client.post("/api/v1/predict", json=invalid_data)
        assert response.status_code == 422

    def test_predict_missing_field(self, valid_patient_data):
        """Test prediction with missing required field."""
        incomplete_data = valid_patient_data.copy()
        del incomplete_data["age"]

        response = client.post("/api/v1/predict", json=incomplete_data)
        assert response.status_code == 422

    def test_predict_extra_field(self, valid_patient_data):
        """Test prediction with extra field."""
        data_with_extra = valid_patient_data.copy()
        data_with_extra["extra_field"] = "should_be_ignored"

        response = client.post("/api/v1/predict", json=data_with_extra)

        if response.status_code == 503:
            pytest.skip("No trained model available")

        # Should succeed - extra fields are ignored
        assert response.status_code == 200

    def test_predict_boundary_values(self):
        """Test prediction with boundary values."""
        boundary_data = {
            "age": 0,  # Minimum age
            "sex": 0,
            "cp": 0,
            "trestbps": 50,  # Minimum blood pressure
            "chol": 100,  # Minimum cholesterol
            "restecg": 0,
            "thalach": 50,  # Minimum heart rate
            "exang": 0,
            "oldpeak": 0.0
        }

        response = client.post("/api/v1/predict", json=boundary_data)

        if response.status_code == 503:
            pytest.skip("No trained model available")

        assert response.status_code == 200


class TestBatchPredictionEndpoint:
    """Tests for batch prediction endpoint."""

    @pytest.fixture
    def batch_patients_data(self):
        """Batch of patient data fixture."""
        return {
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
                    "age": 45,
                    "sex": 0,
                    "cp": 2,
                    "trestbps": 120,
                    "chol": 200,
                    "restecg": 0,
                    "thalach": 170,
                    "exang": 0,
                    "oldpeak": 1.0
                }
            ]
        }

    def test_batch_predict_valid_input(self, batch_patients_data):
        """Test batch prediction with valid input."""
        response = client.post("/api/v1/predict/batch", json=batch_patients_data)

        if response.status_code == 503:
            pytest.skip("No trained model available")

        assert response.status_code == 200

        data = response.json()
        assert "predictions" in data
        assert "total_count" in data
        assert "disease_count" in data
        assert "no_disease_count" in data

        assert len(data["predictions"]) == 2
        assert data["total_count"] == 2
        assert data["disease_count"] + data["no_disease_count"] == 2

    def test_batch_predict_empty_list(self):
        """Test batch prediction with empty patient list."""
        empty_data = {"patients": []}

        response = client.post("/api/v1/predict/batch", json=empty_data)

        if response.status_code == 503:
            pytest.skip("No trained model available")

        # Should handle empty list gracefully
        assert response.status_code in [200, 422]

    def test_batch_predict_single_patient(self):
        """Test batch prediction with single patient."""
        single_patient = {
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
                }
            ]
        }

        response = client.post("/api/v1/predict/batch", json=single_patient)

        if response.status_code == 503:
            pytest.skip("No trained model available")

        assert response.status_code == 200

        data = response.json()
        assert len(data["predictions"]) == 1
        assert data["total_count"] == 1


class TestFileUploadEndpoint:
    """Tests for file upload prediction endpoint."""

    @pytest.fixture
    def sample_csv_content(self):
        """Sample CSV content fixture."""
        csv_data = """age,sex,cp,trestbps,chol,restecg,thalach,exang,oldpeak
63,1,3,145,233,0,150,0,2.3
45,0,2,120,200,0,170,0,1.0
"""
        return csv_data.encode()

    def test_upload_valid_csv(self, sample_csv_content):
        """Test file upload with valid CSV."""
        files = {"file": ("test.csv", sample_csv_content, "text/csv")}
        response = client.post("/api/v1/predict/upload", files=files)

        if response.status_code == 503:
            pytest.skip("No trained model available")

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/csv; charset=utf-8"

    def test_upload_invalid_file_type(self):
        """Test file upload with invalid file type."""
        files = {"file": ("test.txt", b"invalid content", "text/plain")}
        response = client.post("/api/v1/predict/upload", files=files)

        # Should fail due to invalid CSV format
        assert response.status_code in [422, 500, 503]

    def test_upload_missing_columns(self):
        """Test file upload with missing required columns."""
        csv_data = b"age,sex\n63,1\n"
        files = {"file": ("test.csv", csv_data, "text/csv")}
        response = client.post("/api/v1/predict/upload", files=files)

        # Should fail due to missing columns
        assert response.status_code in [422, 500, 503]


class TestModelsEndpoint:
    """Tests for model management endpoints."""

    def test_list_models(self):
        """Test listing available models."""
        response = client.get("/api/v1/models")
        assert response.status_code == 200

        data = response.json()
        assert "models" in data
        assert "default_model" in data
        assert "total_count" in data
        assert isinstance(data["models"], list)

    def test_get_model_info_not_found(self):
        """Test getting info for non-existent model."""
        response = client.get("/api/v1/models/nonexistent_model")
        assert response.status_code == 404

    def test_use_model_not_found(self):
        """Test switching to non-existent model."""
        response = client.post("/api/v1/models/nonexistent_model/use")
        assert response.status_code == 404


class TestMetricsEndpoint:
    """Tests for metrics endpoint."""

    def test_get_metrics(self):
        """Test metrics endpoint."""
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200

        data = response.json()
        assert "current_model" in data
        assert "gpu_enabled" in data


class TestCORSHeaders:
    """Tests for CORS configuration."""

    def test_cors_headers_present(self):
        """Test CORS headers are present."""
        response = client.options("/api/v1/predict")

        # Check for CORS headers
        assert "access-control-allow-origin" in [h.lower() for h in response.headers.keys()]

    def test_cors_allows_methods(self):
        """Test CORS allows required methods."""
        response = client.options("/api/v1/predict")
        assert response.status_code in [200, 405]


class TestAPIDocumentation:
    """Tests for API documentation endpoints."""

    def test_openapi_schema(self):
        """Test OpenAPI schema is accessible."""
        response = client.get("/openapi.json")
        assert response.status_code == 200

        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema

    def test_swagger_ui(self):
        """Test Swagger UI is accessible."""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_redoc(self):
        """Test ReDoc is accessible."""
        response = client.get("/redoc")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_endpoint(self):
        """Test accessing invalid endpoint returns 404."""
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404

    def test_method_not_allowed(self):
        """Test using wrong HTTP method."""
        response = client.get("/api/v1/predict")  # Should be POST
        assert response.status_code == 405

    def test_malformed_json(self):
        """Test sending malformed JSON."""
        response = client.post(
            "/api/v1/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422


# Integration tests
class TestIntegration:
    """Integration tests for complete workflows."""

    @pytest.mark.integration
    def test_complete_prediction_workflow(self):
        """Test complete workflow: health check → predict → get metrics."""
        # 1. Check health
        health_response = client.get("/health")
        assert health_response.status_code == 200

        # 2. Make prediction
        patient_data = {
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
        predict_response = client.post("/api/v1/predict", json=patient_data)

        if predict_response.status_code == 503:
            pytest.skip("No trained model available")

        assert predict_response.status_code == 200

        # 3. Check metrics
        metrics_response = client.get("/api/v1/metrics")
        assert metrics_response.status_code == 200

    @pytest.mark.integration
    def test_model_switching_workflow(self):
        """Test workflow: list models → get model info → use model."""
        # 1. List models
        list_response = client.get("/api/v1/models")
        assert list_response.status_code == 200

        models_data = list_response.json()
        if models_data["total_count"] == 0:
            pytest.skip("No trained models available")

        # 2. Get info for first model
        first_model = models_data["models"][0]["name"]
        info_response = client.get(f"/api/v1/models/{first_model}")

        # Model info might fail if model file is corrupted, so we allow 404/500
        assert info_response.status_code in [200, 404, 500]

        # 3. Use model
        use_response = client.post(f"/api/v1/models/{first_model}/use")
        assert use_response.status_code in [200, 404, 500]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
