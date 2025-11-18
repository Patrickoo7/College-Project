# Heart Disease Prediction API Documentation

## Table of Contents

- [Overview](#overview)
- [Base URL](#base-url)
- [Authentication](#authentication)
- [Endpoints](#endpoints)
  - [Health Check](#health-check)
  - [Single Prediction](#single-prediction)
  - [Batch Prediction](#batch-prediction)
  - [File Upload Prediction](#file-upload-prediction)
  - [List Models](#list-models)
  - [Get Model Info](#get-model-info)
  - [Switch Model](#switch-model)
  - [Get Metrics](#get-metrics)
- [Data Models](#data-models)
- [Error Handling](#error-handling)
- [Code Examples](#code-examples)
- [Rate Limiting](#rate-limiting)
- [Versioning](#versioning)

## Overview

The Heart Disease Prediction API is a RESTful API that provides machine learning-powered predictions for heart disease risk assessment. The API is built with FastAPI and provides comprehensive endpoints for single and batch predictions.

**Features:**
- ✅ Real-time heart disease predictions
- ✅ Batch processing support
- ✅ CSV file upload for bulk predictions
- ✅ Multiple trained models
- ✅ Model switching capability
- ✅ GPU acceleration support
- ✅ Interactive API documentation (Swagger UI)
- ✅ Comprehensive error handling

## Base URL

### Local Development
```
http://localhost:8000
```

### Azure Deployment
```
https://heart-disease-api.azurewebsites.net
```

### API Version
All endpoints are prefixed with `/api/v1` for versioning.

```
http://localhost:8000/api/v1/predict
```

## Authentication

Currently, the API does not require authentication. For production deployments, consider adding:
- API Key authentication
- OAuth 2.0
- JWT tokens

## Endpoints

### Health Check

Check the API health status and GPU availability.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "api_version": "1.0.0",
  "gpu_available": true,
  "gpu_count": 1
}
```

**Example:**
```bash
curl http://localhost:8000/health
```

---

### Single Prediction

Make a prediction for a single patient.

**Endpoint:** `POST /api/v1/predict`

**Request Body:**
```json
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
```

**Parameters:**

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `age` | integer | 0-120 | Age in years |
| `sex` | integer | 0-1 | Sex (0: Female, 1: Male) |
| `cp` | integer | 0-4 | Chest pain type (0-3: type, 4: asymptomatic) |
| `trestbps` | integer | 50-250 | Resting blood pressure (mm Hg) |
| `chol` | integer | 100-600 | Serum cholesterol (mg/dl) |
| `restecg` | integer | 0-2 | Resting ECG results |
| `thalach` | integer | 50-250 | Maximum heart rate achieved |
| `exang` | integer | 0-1 | Exercise induced angina (0: No, 1: Yes) |
| `oldpeak` | float | 0-10 | ST depression induced by exercise |

**Query Parameters:**
- `model_name` (optional): Name of the model to use (default: `random_forest`)

**Response:**
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
```

**Example:**
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

---

### Batch Prediction

Make predictions for multiple patients in a single request.

**Endpoint:** `POST /api/v1/predict/batch`

**Request Body:**
```json
{
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
```

**Response:**
```json
{
  "predictions": [
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
    },
    {
      "prediction": 0,
      "prediction_label": "No Disease",
      "probability": {
        "no_disease": 0.85,
        "disease": 0.15
      },
      "confidence": 0.15,
      "model_used": "random_forest",
      "input_data": {...}
    }
  ],
  "total_count": 2,
  "disease_count": 1,
  "no_disease_count": 1
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

---

### File Upload Prediction

Upload a CSV file for bulk predictions.

**Endpoint:** `POST /api/v1/predict/upload`

**Request:**
- Content-Type: `multipart/form-data`
- File parameter: `file`
- File format: CSV

**CSV Format:**
```csv
age,sex,cp,trestbps,chol,restecg,thalach,exang,oldpeak
63,1,3,145,233,0,150,0,2.3
45,0,2,120,200,0,170,0,1.0
```

**Response:**
- Content-Type: `text/csv`
- Returns CSV file with predictions added

**Response CSV:**
```csv
age,sex,cp,trestbps,chol,restecg,thalach,exang,oldpeak,prediction,prediction_label,probability_no_disease,probability_disease
63,1,3,145,233,0,150,0,2.3,1,Disease,0.23,0.77
45,0,2,120,200,0,170,0,1.0,0,No Disease,0.85,0.15
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/predict/upload \
  -F "file=@patients.csv" \
  -o predictions.csv
```

---

### List Models

Get a list of all available trained models.

**Endpoint:** `GET /api/v1/models`

**Response:**
```json
{
  "models": [
    {
      "name": "random_forest",
      "type": "RandomForestClassifier",
      "available": true,
      "path": "/app/models/artifacts/random_forest.pkl"
    },
    {
      "name": "xgboost",
      "type": "XGBClassifier",
      "available": true,
      "path": "/app/models/artifacts/xgboost.pkl"
    }
  ],
  "default_model": "random_forest",
  "total_count": 2
}
```

**Example:**
```bash
curl http://localhost:8000/api/v1/models
```

---

### Get Model Info

Get detailed information about a specific model.

**Endpoint:** `GET /api/v1/models/{model_name}`

**Path Parameters:**
- `model_name`: Name of the model (e.g., `random_forest`)

**Response:**
```json
{
  "name": "random_forest",
  "type": "RandomForestClassifier",
  "available": true,
  "path": "/app/models/artifacts/random_forest.pkl"
}
```

**Example:**
```bash
curl http://localhost:8000/api/v1/models/random_forest
```

**Error Response (404):**
```json
{
  "detail": "Model 'nonexistent_model' not found"
}
```

---

### Switch Model

Switch to using a different model for predictions.

**Endpoint:** `POST /api/v1/models/{model_name}/use`

**Path Parameters:**
- `model_name`: Name of the model to switch to

**Response:**
```json
{
  "message": "Successfully switched to model: xgboost",
  "model": "xgboost"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/models/xgboost/use
```

---

### Get Metrics

Get API metrics and statistics.

**Endpoint:** `GET /api/v1/metrics`

**Response:**
```json
{
  "total_predictions": "N/A",
  "uptime": "N/A",
  "current_model": "random_forest",
  "gpu_enabled": true
}
```

**Example:**
```bash
curl http://localhost:8000/api/v1/metrics
```

---

## Data Models

### PatientInput

```python
{
  "age": int,          # 0-120
  "sex": int,          # 0 or 1
  "cp": int,           # 0-4
  "trestbps": int,     # 50-250
  "chol": int,         # 100-600
  "restecg": int,      # 0-2
  "thalach": int,      # 50-250
  "exang": int,        # 0 or 1
  "oldpeak": float     # 0.0-10.0
}
```

### PredictionResponse

```python
{
  "prediction": int,                  # 0 or 1
  "prediction_label": str,            # "Disease" or "No Disease"
  "probability": {
    "no_disease": float,              # 0.0-1.0
    "disease": float                  # 0.0-1.0
  },
  "confidence": float,                # 0.0-1.0
  "model_used": str,
  "input_data": PatientInput
}
```

### BatchPredictionRequest

```python
{
  "patients": List[PatientInput]
}
```

### BatchPredictionResponse

```python
{
  "predictions": List[PredictionResponse],
  "total_count": int,
  "disease_count": int,
  "no_disease_count": int
}
```

## Error Handling

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request |
| 404 | Not Found |
| 422 | Validation Error |
| 500 | Internal Server Error |
| 503 | Service Unavailable (No model loaded) |

### Error Response Format

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Validation Errors (422)

```json
{
  "detail": [
    {
      "loc": ["body", "age"],
      "msg": "ensure this value is greater than or equal to 0",
      "type": "value_error.number.not_ge"
    }
  ]
}
```

## Code Examples

### Python (requests)

```python
import requests

# Single prediction
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

### Python (file upload)

```python
import requests

url = "http://localhost:8000/api/v1/predict/upload"
files = {"file": open("patients.csv", "rb")}

response = requests.post(url, files=files)

with open("predictions.csv", "wb") as f:
    f.write(response.content)
```

### JavaScript (fetch)

```javascript
const url = "http://localhost:8000/api/v1/predict";
const data = {
  age: 63,
  sex: 1,
  cp: 3,
  trestbps: 145,
  chol: 233,
  restecg: 0,
  thalach: 150,
  exang: 0,
  oldpeak: 2.3
};

fetch(url, {
  method: "POST",
  headers: {
    "Content-Type": "application/json"
  },
  body: JSON.stringify(data)
})
  .then(response => response.json())
  .then(result => {
    console.log("Prediction:", result.prediction_label);
    console.log("Confidence:", result.confidence);
  });
```

### cURL

```bash
# Single prediction
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3}'

# Batch prediction
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"patients": [{"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3}]}'

# File upload
curl -X POST http://localhost:8000/api/v1/predict/upload \
  -F "file=@patients.csv" \
  -o predictions.csv

# List models
curl http://localhost:8000/api/v1/models

# Switch model
curl -X POST http://localhost:8000/api/v1/models/xgboost/use
```

## Rate Limiting

Currently, there are no rate limits implemented. For production deployments, consider adding:

- Rate limiting per IP address
- API key-based quotas
- Request throttling

## Versioning

The API uses URL-based versioning with the prefix `/api/v1`.

**Current version:** v1.0.0

Future versions will be available at:
- `/api/v2/predict`
- `/api/v3/predict`

## Interactive Documentation

The API provides interactive documentation powered by Swagger UI and ReDoc:

### Swagger UI
```
http://localhost:8000/docs
```

Features:
- Interactive API explorer
- Try out endpoints directly
- View request/response schemas
- Authentication testing

### ReDoc
```
http://localhost:8000/redoc
```

Features:
- Clean, three-panel design
- Detailed endpoint documentation
- Code samples
- Searchable interface

## Best Practices

1. **Always validate input data** before sending to API
2. **Use batch endpoints** for multiple predictions to reduce overhead
3. **Handle errors gracefully** and check status codes
4. **Cache model lists** to reduce unnecessary API calls
5. **Use file upload** for large datasets (>100 patients)
6. **Monitor API health** with the `/health` endpoint
7. **Switch models** based on your accuracy/speed requirements

## Support

For API issues or questions:
- Check the [GitHub repository](https://github.com/yourusername/heart-disease-prediction)
- Review the [troubleshooting guide](../azure/README.md#troubleshooting)
- Open an issue on GitHub

---

**Last Updated:** 2024-01-XX
**API Version:** 1.0.0
