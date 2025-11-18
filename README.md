# Heart Disease Prediction - End-to-End ML System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-ready, end-to-end machine learning system for predicting heart disease using multiple UCI heart disease datasets. This project implements MLOps best practices including experiment tracking, model versioning, automated testing, and comprehensive evaluation.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [API & Web Interface](#api--web-interface)
- [Docker Deployment](#docker-deployment)
- [Azure Deployment](#azure-deployment)
- [Configuration](#configuration)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [MLflow Tracking](#mlflow-tracking)
- [API Documentation](#api-documentation)
- [Development](#development)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Features

### Core ML Capabilities
- **Multiple Datasets**: Supports 4 UCI heart disease datasets (Cleveland, Hungarian, Switzerland, VA)
- **8+ ML Algorithms**: Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, SVM, KNN, Decision Trees
- **Ensemble Methods**: Voting and Stacking classifiers for improved performance
- **GPU Acceleration**: Automatic GPU detection and usage for XGBoost, LightGBM, CatBoost (5-15x speedup)
- **Feature Engineering**: Automated feature creation, interaction features, domain-specific features
- **Hyperparameter Optimization**: Integrated Optuna for automated tuning

### MLOps & Production Features
- **Experiment Tracking**: MLflow integration for tracking all experiments
- **Model Registry**: Versioned model storage with metadata
- **Automated Retraining**: Intelligent retraining based on performance degradation or schedule
- **Data Drift Detection**: Multiple statistical methods (KS test, Chi-square, PSI)
- **Hyperparameter Tuning**: Bayesian optimization with Optuna for automated tuning
- **Prediction Store**: SQLite database for tracking all predictions and performance
- **Feature Store**: Centralized feature management with versioning
- **Data Versioning**: DVC integration for dataset version control
- **Performance Monitoring**: Real-time monitoring with automated alerts
- **Data Validation**: Comprehensive data quality checks
- **Automated Pipeline**: End-to-end training and evaluation pipeline
- **Configuration Management**: YAML-based configuration for easy customization
- **Logging**: Structured logging with rotation
- **Testing**: Comprehensive test suite with pytest

### Evaluation & Monitoring
- **15+ Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Specificity, Sensitivity, PPV, NPV
- **Visualizations**: Confusion matrices, ROC curves, PR curves, feature importance plots
- **Cross-Validation**: K-fold cross-validation with stratification
- **Model Comparison**: Automated comparison of all models

### API & Web Interface
- **FastAPI REST API**: Production-ready API with auto-documentation (Swagger UI, ReDoc)
- **Streamlit Web App**: Interactive web interface for predictions and visualizations
- **Multiple Endpoints**: Single prediction, batch prediction, file upload, model switching
- **Input Validation**: Pydantic models for request/response validation
- **Real-time Predictions**: Instant predictions with probability scores
- **Model Management**: List, switch, and get info on available models

### Model Explainability & Reporting
- **SHAP Integration**: Global and local feature importance with SHapley values
- **LIME Support**: Local interpretable model-agnostic explanations
- **Feature Contributions**: Individual prediction explanations
- **Automated Reports**: HTML reports for model evaluation and drift detection
- **Model Cards**: Comprehensive model documentation following best practices
- **Visualization**: Waterfall plots, beeswarm plots, feature importance charts

### Deployment & DevOps
- **Docker Containers**: Multi-stage builds for API and web applications
- **Docker Compose**: Orchestrated services with networking
- **Azure Deployment**: Automated deployment to Azure App Service
- **CI/CD Pipeline**: GitHub Actions for automated testing and deployment
- **Infrastructure as Code**: Bash scripts for Azure resource provisioning
- **Health Monitoring**: Health check endpoints and application insights ready
- **Alert System**: Multi-channel alerts (email, Slack) for model performance

## Project Structure

```
heart-disease-prediction/
├── src/
│   ├── api/
│   │   ├── app.py                   # FastAPI application
│   │   ├── routes.py                # API endpoints
│   │   └── schemas.py               # Pydantic models
│   ├── web/
│   │   └── streamlit_app.py        # Streamlit web interface
│   ├── data/
│   │   ├── data_loader.py          # Load data from multiple sources
│   │   ├── data_validator.py       # Validate data quality
│   │   └── data_preprocessor.py    # Clean & preprocess data
│   ├── features/
│   │   ├── feature_engineering.py  # Create engineered features
│   │   └── transformers.py         # Custom sklearn transformers
│   ├── models/
│   │   ├── train.py               # Training pipeline with MLflow
│   │   ├── evaluate.py            # Comprehensive evaluation
│   │   └── predict.py             # Prediction interface
│   ├── mlops/
│   │   ├── retraining.py          # Automated retraining pipeline
│   │   ├── drift_detection.py     # Data drift detection
│   │   ├── hyperparameter_tuning.py # Bayesian optimization
│   │   ├── prediction_store.py    # Prediction database
│   │   ├── feature_store.py       # Feature management
│   │   └── monitoring.py          # Performance monitoring & alerts
│   ├── reporting/
│   │   ├── report_generator.py    # Automated HTML reports
│   │   └── model_card.py          # Model documentation
│   ├── explainability/
│   │   └── explainer.py           # SHAP & LIME explanations
│   └── utils/
│       ├── config.py              # Configuration management
│       ├── logger.py              # Logging setup
│       ├── gpu_utils.py           # GPU detection and management
│       └── exceptions.py          # Custom exceptions
├── configs/
│   ├── config.yaml               # Main configuration
│   └── model_config.yaml         # Model hyperparameters
├── data/
│   ├── raw/                      # Original datasets
│   ├── processed/                # Cleaned datasets
│   └── external/                 # External data sources
├── models/
│   ├── artifacts/                # Saved models
│   ├── scaler/                   # Saved scalers
│   └── metadata/                 # Model metadata
├── docker/
│   ├── Dockerfile.api           # API container
│   └── Dockerfile.web           # Web container
├── azure/
│   ├── deploy-to-azure.sh       # Azure deployment script
│   └── README.md                # Azure deployment guide
├── .github/
│   ├── workflows/
│   │   └── azure-deploy.yml    # CI/CD pipeline
│   └── CICD_SETUP.md           # CI/CD setup guide
├── notebooks/                    # Jupyter notebooks
├── tests/                        # Test suite
│   ├── test_api.py              # API endpoint tests
│   └── ...                      # Other tests
├── docs/                         # Documentation
│   ├── API_DOCUMENTATION.md     # Complete API reference
│   ├── API_QUICKSTART.md        # API quick start guide
│   ├── GPU_SETUP.md             # GPU setup guide
│   ├── PHASES_6_9_DOCUMENTATION.md # MLOps & Advanced Features guide
│   ├── model_cards/             # Generated model cards
│   ├── old_notebooks/           # Original project notebooks
│   ├── reports/                 # Project reports
│   └── presentations/           # Presentations
├── docker-compose.yml           # Docker orchestration
├── pytest.ini                   # Pytest configuration
├── requirements.txt             # Python dependencies
├── requirements-gpu.txt         # GPU dependencies
├── setup.py                     # Package setup
├── Makefile                     # Automation commands
└── README.md                    # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Option 1: Standard Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Option 2: Development Installation

```bash
# Install with development dependencies
make install-dev

# Or manually
pip install -e ".[dev]"
```

### Option 3: Using Makefile

```bash
# Complete development setup
make dev-setup
```

### Option 4: GPU Acceleration (Optional but Recommended)

For 5-15x faster training with XGBoost, LightGBM, and CatBoost:

```bash
# Install base requirements first
pip install -r requirements.txt

# Install GPU-accelerated packages (requires CUDA Toolkit)
pip install xgboost lightgbm catboost

# For PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU detection
python -c "from src.utils.gpu_utils import get_gpu_manager; get_gpu_manager().print_gpu_summary()"
```

**Note:** GPU support requires NVIDIA GPU with CUDA Toolkit installed. See [GPU Setup Guide](docs/GPU_SETUP.md) for detailed instructions.

The system automatically detects and uses GPU when available, with seamless fallback to CPU.

## Quick Start

### 1. Validate Your Data

```bash
# Validate the raw data
make data-validate
```

### 2. Train Models

```bash
# Train all enabled models
make train

# Or train a specific model
make train-single MODEL=random_forest
```

### 3. View Results in MLflow

```bash
# Start MLflow UI
make mlflow-ui

# Open browser to http://localhost:5000
```

### 4. Evaluate Models

```bash
# Evaluate all trained models
make evaluate
```

### 5. Make Predictions

```python
from src.models.predict import HeartDiseasePredictor

# Load a trained model
predictor = HeartDiseasePredictor(model_path="models/artifacts/random_forest.pkl")

# Make a prediction
result = predictor.predict_single(
    age=63,
    sex=1,
    cp=3,
    trestbps=145,
    chol=233,
    restecg=0,
    thalach=150,
    exang=0,
    oldpeak=2.3
)

print(f"Prediction: {result['prediction_label']}")
print(f"Probability: {result['probability']}")
```

## Usage

### Data Loading

```python
from src.data.data_loader import DataLoader

# Initialize loader
loader = DataLoader()

# Load single dataset
cleveland_data = loader.load_cleveland()

# Load all UCI datasets
all_data = loader.load_and_combine_uci()

# Get dataset info
info = loader.get_dataset_info("cleveland")
print(info)
```

### Data Preprocessing

```python
from src.data.data_preprocessor import DataPreprocessor

# Initialize preprocessor
preprocessor = DataPreprocessor()

# Run complete preprocessing pipeline
df_clean = preprocessor.preprocess_pipeline(
    df,
    handle_missing=True,
    remove_outliers=True,
    encode_categorical=True
)

# Split data
X_train, X_test, y_train, y_test = preprocessor.split_data(df_clean)
```

### Feature Engineering

```python
from src.features.feature_engineering import FeatureEngineer

# Initialize feature engineer
engineer = FeatureEngineer()

# Create domain-specific features
df_engineered = engineer.create_domain_features(df)

# Create interaction features
df_with_interactions = engineer.create_interaction_features(df)

# Run complete feature engineering pipeline
df_final = engineer.engineer_features_pipeline(
    df,
    create_interactions=True,
    create_domain=True
)
```

### Model Training

```python
from src.models.train import ModelTrainer

# Initialize trainer
trainer = ModelTrainer()

# Train all enabled models
trained_models = trainer.train_all_models(X_train, y_train)

# Train specific model
model = trainer.train_model("random_forest", X_train, y_train)

# Create ensemble
ensemble = trainer.create_ensemble("voting", X_train, y_train)

# Save models
trainer.save_all_models()
```

### Model Evaluation

```python
from src.models.evaluate import ModelEvaluator

# Initialize evaluator
evaluator = ModelEvaluator()

# Evaluate single model
results = evaluator.evaluate_model(model, X_test, y_test, "RandomForest")

# Compare multiple models
comparison = evaluator.compare_models(trained_models, X_test, y_test)
print(comparison)

# Generate visualizations
evaluator.plot_confusion_matrix(y_test, y_pred, "RandomForest")
evaluator.plot_roc_curve(y_test, y_prob, "RandomForest")

# Save evaluation report
evaluator.save_evaluation_report()
```

## API & Web Interface

### FastAPI REST API

The system includes a production-ready REST API built with FastAPI.

#### Starting the API

```bash
# Using Uvicorn
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# Using Docker
docker-compose up api
```

#### API Endpoints

- `GET /health` - Health check with GPU info
- `POST /api/v1/predict` - Single patient prediction
- `POST /api/v1/predict/batch` - Batch predictions
- `POST /api/v1/predict/upload` - CSV file upload
- `GET /api/v1/models` - List available models
- `POST /api/v1/models/{name}/use` - Switch active model

#### Interactive Documentation

Once running, access:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

#### API Usage Example

```python
import requests

# Make a prediction
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

**For complete API documentation, see [API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md)**

### Streamlit Web Application

Interactive web interface for predictions and visualizations.

#### Starting the Web App

```bash
# Using Streamlit
streamlit run src/web/streamlit_app.py

# Using Docker
docker-compose up web
```

#### Features

- **Single Prediction**: Interactive form with real-time predictions
- **Batch Prediction**: CSV file upload for bulk predictions
- **Model Information**: View model details and feature importance
- **Visualizations**: Gauge charts, bar charts, confusion matrices
- **Model Switching**: Change active model on-the-fly

#### Access

Open browser to: http://localhost:8501

## Docker Deployment

### Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+

### Quick Start

```bash
# Build and start all services
docker-compose up --build

# Start in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| API | 8000 | FastAPI REST API |
| Web | 8501 | Streamlit web interface |

### Custom Build

```bash
# Build API image
docker build -t heart-disease-api -f docker/Dockerfile.api .

# Build Web image
docker build -t heart-disease-web -f docker/Dockerfile.web .

# Run API container
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/configs:/app/configs \
  heart-disease-api

# Run Web container
docker run -p 8501:8501 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/configs:/app/configs \
  heart-disease-web
```

### Environment Variables

Configure via `docker-compose.yml` or command line:

```yaml
environment:
  - PYTHONUNBUFFERED=1
  - LOG_LEVEL=INFO
  - GPU_ENABLED=false
```

## Azure Deployment

### Prerequisites

- Azure account with active subscription
- Azure CLI installed
- Docker installed

### Quick Deployment

```bash
# Run automated deployment script
chmod +x azure/deploy-to-azure.sh
./azure/deploy-to-azure.sh
```

The script will:
1. Create Azure Resource Group
2. Set up Azure Container Registry (ACR)
3. Build and push Docker images
4. Create App Service Plan
5. Deploy API and Web applications
6. Configure environment variables

**Deployment time:** ~10-15 minutes

### Manual Deployment

See the comprehensive guide: [azure/README.md](azure/README.md)

### Access Deployed Applications

After deployment:

```bash
# Get URLs
az webapp show --name heart-disease-api --resource-group heart-disease-rg --query "defaultHostName" -o tsv
az webapp show --name heart-disease-web --resource-group heart-disease-rg --query "defaultHostName" -o tsv
```

**URLs:**
- API: `https://heart-disease-api.azurewebsites.net`
- API Docs: `https://heart-disease-api.azurewebsites.net/docs`
- Web App: `https://heart-disease-web.azurewebsites.net`

### CI/CD Pipeline

Automated deployment with GitHub Actions:

1. **Setup Azure credentials** (one-time):
   ```bash
   az ad sp create-for-rbac --name "github-actions-heart-disease" \
     --role contributor \
     --scopes /subscriptions/$SUBSCRIPTION_ID/resourceGroups/heart-disease-rg \
     --sdk-auth
   ```

2. **Add GitHub secret**: `AZURE_CREDENTIALS` with the output from step 1

3. **Push to main branch** - automatic deployment starts

**For detailed CI/CD setup, see [.github/CICD_SETUP.md](.github/CICD_SETUP.md)**

## Configuration

### Main Configuration (`configs/config.yaml`)

```yaml
project:
  name: "heart-disease-prediction"
  version: "1.0.0"

paths:
  data:
    raw: "data/raw"
    processed: "data/processed"
  models:
    artifacts: "models/artifacts"

preprocessing:
  test_size: 0.25
  random_state: 42
  scaling_method: "standard"

training:
  cv_folds: 5
  n_jobs: -1
```

### Model Configuration (`configs/model_config.yaml`)

```yaml
models:
  random_forest:
    enabled: true
    params:
      n_estimators: 400
      max_depth: null
      random_state: 42
    hyperparameter_search:
      n_estimators: [100, 200, 400]
      max_depth: [5, 10, 15, null]
```

## Model Training

### Available Models

| Model | Description | Key Parameters |
|-------|-------------|----------------|
| Logistic Regression | Linear baseline model | C, penalty, solver |
| Random Forest | Ensemble of decision trees | n_estimators, max_depth |
| XGBoost | Gradient boosting | learning_rate, max_depth |
| LightGBM | Fast gradient boosting | num_leaves, learning_rate |
| CatBoost | Categorical boosting | depth, iterations |
| SVM | Support vector machine | C, gamma, kernel |
| KNN | K-nearest neighbors | n_neighbors, weights |
| Decision Tree | Single decision tree | max_depth, min_samples_split |

### Training Pipeline

```bash
# Run complete pipeline
make pipeline

# This executes:
# 1. Data validation
# 2. Data processing
# 3. Model training
# 4. Model evaluation
```

## Evaluation

### Metrics

- **Accuracy**: Overall correctness
- **Precision**: Positive predictive value
- **Recall (Sensitivity)**: True positive rate
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve
- **PR-AUC**: Area under precision-recall curve
- **Specificity**: True negative rate
- **PPV/NPV**: Positive/negative predictive values

### Cross-Validation

```python
from src.models.evaluate import ModelEvaluator

evaluator = ModelEvaluator()
cv_results = evaluator.cross_validate_model(
    model, X, y,
    cv=5,
    scoring=['accuracy', 'f1', 'roc_auc']
)
```

## MLflow Tracking

### View Experiments

```bash
# Start MLflow UI
make mlflow-ui

# Access at http://localhost:5000
```

### Logged Information

- **Parameters**: All model hyperparameters
- **Metrics**: All evaluation metrics
- **Artifacts**: Trained models, confusion matrices
- **Tags**: Model type, dataset, version

## Development

### Code Formatting

```bash
# Format code
make format

# Check formatting
make format-check
```

### Linting

```bash
# Run linters
make lint
```

### Testing

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Fast tests (exit on first failure)
make test-fast
```

### Makefile Commands

```bash
# View all available commands
make help

# Common commands:
make install          # Install dependencies
make setup           # Setup project directories
make train           # Train models
make evaluate        # Evaluate models
make mlflow-ui       # Start MLflow UI
make clean           # Clean temporary files
make pipeline        # Run complete pipeline
```

## API Documentation

### DataLoader

```python
loader = DataLoader()
loader.load_cleveland()              # Load Cleveland dataset
loader.load_all_uci_datasets()      # Load all 4 UCI datasets
loader.combine_datasets([...])       # Combine multiple datasets
loader.save_processed_data(df, ...)  # Save processed data
```

### DataValidator

```python
validator = DataValidator()
validator.validate_schema(df)        # Validate column schema
validator.validate_data_types(df)    # Validate data types
validator.validate_all(df)           # Run all validations
validator.get_data_quality_report(df) # Get quality report
```

### ModelTrainer

```python
trainer = ModelTrainer()
trainer.train_model(name, X, y)      # Train single model
trainer.train_all_models(X, y)       # Train all models
trainer.create_ensemble(type, X, y)  # Create ensemble
trainer.save_model(model, name)      # Save model
```

### ModelEvaluator

```python
evaluator = ModelEvaluator()
evaluator.evaluate_model(model, X, y) # Evaluate model
evaluator.compare_models(models, X, y) # Compare models
evaluator.plot_confusion_matrix(...)  # Plot confusion matrix
evaluator.plot_roc_curve(...)        # Plot ROC curve
```

## Testing

### Run Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific test file
pytest tests/test_data_loader.py -v
```

### Test Structure

```
tests/
├── test_data_loader.py
├── test_data_validator.py
├── test_preprocessor.py
├── test_models.py
└── test_api.py
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- UCI Machine Learning Repository for the heart disease datasets
- Cleveland Clinic Foundation, Hungarian Institute of Cardiology, University Hospital Zurich, and VA Medical Center Long Beach for data collection

## Citation

If you use this code in your research, please cite:

```bibtex
@software{heart_disease_prediction_2024,
  author = {College Project - Group 1},
  title = {Heart Disease Prediction: End-to-End ML System},
  year = {2024},
  url = {https://github.com/yourusername/heart-disease-prediction}
}
```

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Made with ❤️ for better healthcare through AI**
