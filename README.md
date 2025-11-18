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
- **Feature Engineering**: Automated feature creation, interaction features, domain-specific features
- **Hyperparameter Optimization**: Integrated Optuna for automated tuning

### MLOps & Production Features
- **Experiment Tracking**: MLflow integration for tracking all experiments
- **Model Registry**: Versioned model storage with metadata
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

## Project Structure

```
heart-disease-prediction/
├── src/
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
│   └── utils/
│       ├── config.py              # Configuration management
│       ├── logger.py              # Logging setup
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
├── notebooks/                    # Jupyter notebooks
├── tests/                        # Test suite
├── docs/                         # Documentation
│   ├── old_notebooks/           # Original project notebooks
│   ├── reports/                 # Project reports
│   └── presentations/           # Presentations
├── requirements.txt             # Python dependencies
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
