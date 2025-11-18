# API Quick Start Guide

Get up and running with the Heart Disease Prediction API in minutes!

## Table of Contents

- [Prerequisites](#prerequisites)
- [Local Development Setup](#local-development-setup)
- [Running the API](#running-the-api)
- [Testing the API](#testing-the-api)
- [Docker Setup](#docker-setup)
- [Next Steps](#next-steps)

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Trained model files in `models/artifacts/`

## Local Development Setup

### 1. Install Dependencies

```bash
# Navigate to project root
cd College-Project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Models (if not already trained)

```bash
# Run training pipeline
make train

# Or use Python directly
python -m src.models.train
```

This will create model files in `models/artifacts/`:
- `random_forest.pkl`
- `xgboost.pkl`
- `lightgbm.pkl`
- etc.

### 3. Verify Configuration

Check that `configs/config.yaml` has correct paths:

```yaml
paths:
  models:
    artifacts: "models/artifacts"
```

## Running the API

### Option 1: Using Uvicorn (Recommended for Development)

```bash
# Start the API server
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

**Options:**
- `--reload`: Auto-reload on code changes
- `--host 0.0.0.0`: Accept connections from any IP
- `--port 8000`: Run on port 8000

### Option 2: Using Python

```bash
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

### Option 3: Using Makefile

```bash
make run-api
```

### Verify API is Running

Open your browser to: http://localhost:8000/docs

You should see the interactive Swagger UI documentation.

## Testing the API

### 1. Health Check

```bash
curl http://localhost:8000/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "api_version": "1.0.0",
  "gpu_available": false,
  "gpu_count": 0
}
```

### 2. Make a Prediction

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3
  }'
```

**Expected Response:**
```json
{
  "prediction": 1,
  "prediction_label": "Disease",
  "probability": {
    "no_disease": 0.23,
    "disease": 0.77
  },
  "confidence": 0.77,
  "model_used": "random_forest",
  "input_data": {...}
}
```

### 3. Test with Python

Create a test script `test_api.py`:

```python
import requests

url = "http://localhost:8000/api/v1/predict"
data = {
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

response = requests.post(url, json=data)
result = response.json()

print(f"Prediction: {result['prediction_label']}")
print(f"Confidence: {result['confidence']:.2%}")
```

Run it:
```bash
python test_api.py
```

### 4. Test Batch Prediction

```bash
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "patients": [
      {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
        "chol": 233, "restecg": 0, "thalach": 150,
        "exang": 0, "oldpeak": 2.3
      },
      {
        "age": 45, "sex": 0, "cp": 2, "trestbps": 120,
        "chol": 200, "restecg": 0, "thalach": 170,
        "exang": 0, "oldpeak": 1.0
      }
    ]
  }'
```

### 5. Test File Upload

Create a sample CSV file `test_patients.csv`:

```csv
age,sex,cp,trestbps,chol,restecg,thalach,exang,oldpeak
63,1,3,145,233,0,150,0,2.3
45,0,2,120,200,0,170,0,1.0
```

Upload it:
```bash
curl -X POST http://localhost:8000/api/v1/predict/upload \
  -F "file=@test_patients.csv" \
  -o predictions.csv
```

View results:
```bash
cat predictions.csv
```

### 6. List Available Models

```bash
curl http://localhost:8000/api/v1/models
```

### 7. Switch Models

```bash
curl -X POST http://localhost:8000/api/v1/models/xgboost/use
```

## Docker Setup

### 1. Build Docker Image

```bash
# Build API image
docker build -t heart-disease-api -f docker/Dockerfile.api .
```

### 2. Run Container

```bash
# Run API container
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/configs:/app/configs \
  heart-disease-api
```

### 3. Using Docker Compose

```bash
# Start all services (API + Web)
docker-compose up

# Start in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

Access:
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Web UI: http://localhost:8501

## Running Tests

### 1. Install Test Dependencies

```bash
pip install pytest pytest-cov
```

### 2. Run API Tests

```bash
# Run all tests
pytest tests/test_api.py -v

# Run with coverage
pytest tests/test_api.py --cov=src.api --cov-report=html

# Run specific test
pytest tests/test_api.py::TestHealthEndpoint::test_health_check -v
```

### 3. Run Integration Tests

```bash
# Run integration tests only
pytest tests/test_api.py -m integration -v
```

## Troubleshooting

### Issue: "No module named 'src'"

**Solution:**
```bash
# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or install in editable mode
pip install -e .
```

### Issue: "No model loaded"

**Solution:**
Train models first:
```bash
make train
# Or
python -m src.models.train
```

### Issue: "Port 8000 already in use"

**Solution:**
```bash
# Use a different port
uvicorn src.api.app:app --port 8001

# Or kill the process using port 8000
lsof -ti:8000 | xargs kill -9
```

### Issue: "ModuleNotFoundError"

**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Check Python version
python --version  # Should be 3.8+
```

## Interactive Documentation

Once the API is running, explore the interactive documentation:

### Swagger UI
- URL: http://localhost:8000/docs
- Features: Test endpoints, view schemas, try authentication

### ReDoc
- URL: http://localhost:8000/redoc
- Features: Clean documentation, code samples, search

### OpenAPI Schema
- URL: http://localhost:8000/openapi.json
- Use with tools like Postman, Insomnia, or code generators

## Development Tips

### 1. Auto-reload on Changes

Use `--reload` flag for development:
```bash
uvicorn src.api.app:app --reload
```

### 2. View Logs

Check `logs/app.log` for detailed logs:
```bash
tail -f logs/app.log
```

### 3. Debug Mode

Set environment variable:
```bash
export LOG_LEVEL=DEBUG
uvicorn src.api.app:app --reload
```

### 4. Test Different Models

```bash
# Switch to XGBoost
curl -X POST http://localhost:8000/api/v1/models/xgboost/use

# Test prediction
curl -X POST http://localhost:8000/api/v1/predict -H "Content-Type: application/json" -d '{...}'
```

## Next Steps

1. **Explore Full API Documentation**: See [API_DOCUMENTATION.md](API_DOCUMENTATION.md)

2. **Deploy to Production**:
   - [Azure Deployment Guide](../azure/README.md)
   - [CI/CD Setup Guide](../.github/CICD_SETUP.md)

3. **Try the Web Interface**:
   ```bash
   streamlit run src/web/streamlit_app.py
   ```

4. **Customize Configuration**:
   - Edit `configs/config.yaml`
   - Edit `configs/model_config.yaml`

5. **Train Custom Models**:
   - See [Model Training Guide](../README.md#model-training)
   - Experiment with hyperparameters

6. **Monitor Performance**:
   - Set up MLflow: `make mlflow-ui`
   - View metrics at http://localhost:5000

## Useful Commands Cheat Sheet

```bash
# Start API
uvicorn src.api.app:app --reload

# Start API + Web with Docker
docker-compose up

# Run tests
pytest tests/test_api.py -v

# Check health
curl http://localhost:8000/health

# Make prediction
curl -X POST http://localhost:8000/api/v1/predict -H "Content-Type: application/json" -d '{...}'

# List models
curl http://localhost:8000/api/v1/models

# View docs
open http://localhost:8000/docs

# View logs
tail -f logs/app.log
```

## Resources

- [Full API Documentation](API_DOCUMENTATION.md)
- [Project README](../README.md)
- [Azure Deployment Guide](../azure/README.md)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Uvicorn Documentation](https://www.uvicorn.org/)

---

**Need help?** Open an issue on GitHub or check the troubleshooting section above.
